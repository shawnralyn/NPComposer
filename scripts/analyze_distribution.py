"""Compare property distributions between raw and processed datasets.

Generates histogram plots and summary statistics for MW, atom count,
ring count, SA, QED, NPL, and superclass.

Input:
    --raw: path to raw dataset CSV.
    --processed: path to processed subset CSV.
    -o/--output: output prefix (e.g. data/processed/dist_coconut).
    --sample: max rows to sample from raw for RDKit computation (default 10000).
Output:
    {output}_histograms.png: overlay histograms per property.
    {output}_summary.csv: descriptive statistics for both datasets.
    {output}_superclass.csv: superclass distribution comparison.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from rdkit.Contrib.SA_Score import sascorer
    HAS_SA = True
except ImportError:
    HAS_SA = False

try:
    from rdkit.Chem import QED as QED_module
    HAS_QED = True
except ImportError:
    HAS_QED = False

try:
    from rdkit.Contrib.NP_Score import npscorer
    NP_MODEL = npscorer.readNPModel()
except (ImportError, FileNotFoundError, OSError):
    NP_MODEL = None


# Column name candidates for each property
RAW_CANDIDATES = {
    'mw': ['molecular_weight'],
    'atom_count': ['total_atom_count', 'heavy_atom_count'],
    'ring_count': ['number_of_minimal_rings'],
    'sa': [],
    'qed': ['qed_drug_likeliness'],
    'npl': ['np_likeness'],
    'superclass': ['chemical_super_class'],
}

PROC_CANDIDATES = {
    'mw': ['molecular_weight'],
    'atom_count': ['atom_count', 'total_atom_count'],
    'ring_count': ['ring_count', 'number_of_minimal_rings'],
    'sa': ['sa_score(RDKit)'],
    'qed': ['qed(RDKit)', 'qed_drug_likeliness'],
    'npl': ['npl_score(RDKit)', 'np_likeness'],
    'superclass': ['superclass', 'chemical_super_class'],
}

SMILES_CANDIDATES = ['canonical_smiles', 'SMILES', 'smiles']


def detect_col(df, candidates):
    """Find the first matching column name from candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_smiles(df):
    """Find SMILES column name."""
    return detect_col(df, SMILES_CANDIDATES)


def compute_from_smiles(df, smiles_col, prop, sample_n=10000):
    """Compute a property from SMILES, sampling if dataset is large.

    Input:
        df: DataFrame with SMILES column.
        smiles_col: SMILES column name.
        prop: property name ('mw', 'atom_count', 'ring_count', 'sa', 'qed', 'npl').
        sample_n: max rows to compute (default 10000).
    Output:
        pd.Series of computed values (may contain NaN).
    """
    if len(df) > sample_n:
        sample_df = df.sample(n=sample_n, random_state=42)
    else:
        sample_df = df

    smiles_list = sample_df[smiles_col].tolist()
    values = []

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
        if mol is None:
            values.append(np.nan)
            continue
        try:
            if prop == 'mw':
                values.append(Descriptors.MolWt(mol))
            elif prop == 'atom_count':
                values.append(mol.GetNumAtoms())
            elif prop == 'ring_count':
                values.append(Descriptors.RingCount(mol))
            elif prop == 'sa' and HAS_SA:
                values.append(sascorer.calculateScore(mol))
            elif prop == 'qed' and HAS_QED:
                values.append(QED_module.qed(mol))
            elif prop == 'npl' and NP_MODEL is not None:
                values.append(npscorer.scoreMol(mol, NP_MODEL))
            else:
                values.append(np.nan)
        except Exception:
            values.append(np.nan)

    return pd.Series(values, index=sample_df.index, name=prop)


def get_property(df, prop, candidates, smiles_col, sample_n):
    """Get property values: use existing column or compute from SMILES.

    Input:
        df: DataFrame.
        prop: property key.
        candidates: list of column name candidates.
        smiles_col: SMILES column name.
        sample_n: max sample size for computation.
    Output:
        pd.Series of property values (NaN-dropped).
    """
    col = detect_col(df, candidates)
    if col is not None:
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        print(f"    {prop}: using column '{col}' ({len(vals):,} values)")
        return vals

    if smiles_col is None:
        return pd.Series(dtype=float)

    print(f"    {prop}: computing from SMILES (sample={sample_n})...")
    vals = compute_from_smiles(df, smiles_col, prop, sample_n).dropna()
    print(f"    {prop}: {len(vals):,} values computed")
    return vals


def summary_stats(series):
    """Compute summary statistics for a numeric series.

    Input:
        series: pd.Series.
    Output:
        dict with count, mean, std, min, q25, median, q75, max.
    """
    if len(series) == 0:
        return {}
    return {
        'count': len(series),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'q25': series.quantile(0.25),
        'median': series.median(),
        'q75': series.quantile(0.75),
        'max': series.max(),
    }


def plot_histograms(raw_props, proc_props, output_path):
    """Generate overlay histograms comparing raw vs processed distributions.

    Input:
        raw_props: dict of {prop_name: pd.Series} for raw data.
        proc_props: dict of {prop_name: pd.Series} for processed data.
        output_path: path to save PNG.
    """
    if not HAS_MPL:
        print("  matplotlib not available, skipping plot")
        return

    numeric_props = [p for p in raw_props if p != 'superclass'
                     and len(raw_props[p]) > 0 and len(proc_props.get(p, [])) > 0]

    if not numeric_props:
        print("  No numeric properties to plot")
        return

    n = len(numeric_props)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    labels = {
        'mw': 'Molecular Weight',
        'atom_count': 'Atom Count',
        'ring_count': 'Ring Count',
        'sa': 'SA Score',
        'qed': 'QED',
        'npl': 'NPL Score',
    }

    for i, prop in enumerate(numeric_props):
        ax = axes[i]
        raw_vals = raw_props[prop].values
        proc_vals = proc_props[prop].values

        lo = min(np.percentile(raw_vals, 1), np.percentile(proc_vals, 1))
        hi = max(np.percentile(raw_vals, 99), np.percentile(proc_vals, 99))
        bins = np.linspace(lo, hi, 50)

        ax.hist(raw_vals, bins=bins, alpha=0.5, density=True, label='Raw')
        ax.hist(proc_vals, bins=bins, alpha=0.5, density=True, label='Processed')
        ax.set_title(labels.get(prop, prop))
        ax.set_ylabel('Density')
        ax.legend()

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_superclass(raw_series, proc_series, output_path):
    """Generate bar chart comparing superclass distributions.

    Input:
        raw_series: pd.Series of raw superclass labels.
        proc_series: pd.Series of processed superclass labels.
        output_path: path to save PNG.
    """
    if not HAS_MPL:
        return
    if len(raw_series) == 0 and len(proc_series) == 0:
        return

    raw_counts = Counter(raw_series.dropna())
    proc_counts = Counter(proc_series.dropna())

    all_classes = set(raw_counts.keys()) | set(proc_counts.keys())
    if not all_classes:
        return

    # Top 15 by combined count
    combined = {c: raw_counts.get(c, 0) + proc_counts.get(c, 0) for c in all_classes}
    top = sorted(combined, key=combined.get, reverse=True)[:15]

    raw_n = sum(raw_counts.values()) or 1
    proc_n = sum(proc_counts.values()) or 1

    raw_pct = [100 * raw_counts.get(c, 0) / raw_n for c in top]
    proc_pct = [100 * proc_counts.get(c, 0) / proc_n for c in top]

    x = np.arange(len(top))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(x - w / 2, raw_pct, w, label='Raw')
    ax.barh(x + w / 2, proc_pct, w, label='Processed')
    ax.set_yticks(x)
    ax.set_yticklabels([c[:30] for c in top], fontsize=8)
    ax.set_xlabel('Percentage (%)')
    ax.set_title('Superclass Distribution')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare raw vs processed dataset distributions")
    parser.add_argument("--raw", required=True, help="Raw dataset CSV")
    parser.add_argument("--processed", required=True, help="Processed subset CSV")
    parser.add_argument("-o", "--output", required=True, help="Output prefix")
    parser.add_argument("--sample", type=int, default=10000,
                        help="Max rows to sample from raw for RDKit computation")
    args = parser.parse_args()

    out_prefix = args.output
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw: {args.raw}")
    raw_df = pd.read_csv(args.raw, low_memory=False)
    print(f"  {len(raw_df):,} rows")

    print(f"Loading processed: {args.processed}")
    proc_df = pd.read_csv(args.processed, low_memory=False)
    print(f"  {len(proc_df):,} rows")

    raw_smiles = detect_smiles(raw_df)
    proc_smiles = detect_smiles(proc_df)

    numeric_props = ['mw', 'atom_count', 'ring_count', 'sa', 'qed', 'npl']

    print("Extracting raw properties...")
    raw_props = {}
    for prop in numeric_props:
        raw_props[prop] = get_property(
            raw_df, prop, RAW_CANDIDATES[prop], raw_smiles, args.sample)

    print("Extracting processed properties...")
    proc_props = {}
    for prop in numeric_props:
        proc_props[prop] = get_property(
            proc_df, prop, PROC_CANDIDATES[prop], proc_smiles, args.sample)

    # Superclass
    raw_sc_col = detect_col(raw_df, RAW_CANDIDATES['superclass'])
    proc_sc_col = detect_col(proc_df, PROC_CANDIDATES['superclass'])
    raw_sc = raw_df[raw_sc_col].dropna() if raw_sc_col else pd.Series(dtype=str)
    proc_sc = proc_df[proc_sc_col].dropna() if proc_sc_col else pd.Series(dtype=str)

    # Summary statistics
    rows = []
    for prop in numeric_props:
        raw_stats = summary_stats(raw_props[prop])
        proc_stats = summary_stats(proc_props[prop])
        if raw_stats:
            rows.append({'property': prop, 'dataset': 'raw', **raw_stats})
        if proc_stats:
            rows.append({'property': prop, 'dataset': 'processed', **proc_stats})

    summary_path = f"{out_prefix}_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    # Superclass comparison
    if len(raw_sc) > 0 or len(proc_sc) > 0:
        sc_rows = []
        raw_counts = Counter(raw_sc)
        proc_counts = Counter(proc_sc)
        all_classes = set(raw_counts.keys()) | set(proc_counts.keys())
        for cls in sorted(all_classes):
            sc_rows.append({
                'superclass': cls,
                'raw_count': raw_counts.get(cls, 0),
                'processed_count': proc_counts.get(cls, 0),
            })
        sc_path = f"{out_prefix}_superclass.csv"
        pd.DataFrame(sc_rows).to_csv(sc_path, index=False)
        print(f"  Saved: {sc_path}")

    # Plots
    hist_path = f"{out_prefix}_histograms.png"
    plot_histograms(raw_props, proc_props, hist_path)

    sc_plot_path = f"{out_prefix}_superclass.png"
    plot_superclass(raw_sc, proc_sc, sc_plot_path)

    print("Done")


if __name__ == "__main__":
    main()
