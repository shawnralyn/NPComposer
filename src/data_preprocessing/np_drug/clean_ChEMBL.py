"""Concatenate, validate, canonicalize, deduplicate, filter, and slim ChEMBL CSVs.

Performs six steps in order:
  1. Concatenate all ChEMBL CSV files in the input directory (handles the
     single-header-file quirk of ChEMBL bulk exports)
  2. Canonicalize SMILES and drop invalid molecules
  3. Drop duplicate canonical SMILES
  4. Compute Lipinski HBD/HBA from SMILES via RDKit (overwriting export values)
  5. Drop rows with missing values in required property columns
  6. Drop rows whose canonical SMILES appear in an exclusion CSV (optional)
  7. Keep only required property columns and rename to standard names
  8. Compute RDKit SA score and append as 'sa_score' column

Steps 2, 4, and 8 are parallelized across all CPU cores using multiprocessing.

Column names are standardized on output so they match the COCONUT clean output
and bin_cont_variables_npdrug.py defaults:
  Smiles              → canonical_smiles
  QED Weighted        → qed
  Molecular Weight    → molecular_weight
  AlogP               → alogp
  Polar Surface Area  → tpsa
  HBD (Lipinski)      → hbd
  HBA (Lipinski)      → hba
  Aromatic Rings      → aromatic_rings_count

ChEMBL compound exports use semicolons as separators. These are the defaults
but can be overridden.

Usage:
    python src/data_preprocessing/np_drug/clean_ChEMBL.py \\
        --input-dir data/raw/ChEMBL \\
        --output    data/processed/ChEMBL_clean.csv

    # Exclude COCONUT molecules from the output
    python src/data_preprocessing/np_drug/clean_ChEMBL.py \\
        --input-dir data/raw/ChEMBL \\
        --output    data/processed/ChEMBL_clean.csv \\
        --exclude-csv  data/processed/coconut_clean.csv \\
        --exclude-smiles-col canonical_smiles
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski, RDConfig

RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402 — must come after sys.path update


# ---------------------------------------------------------------------------
# Columns to keep from ChEMBL export (pre-rename names)
# ---------------------------------------------------------------------------

CHEMBL_KEEP_RENAME = {
    "Smiles":             "canonical_smiles",
    "QED Weighted":       "qed",
    "Molecular Weight":   "molecular_weight",
    "AlogP":              "alogp",
    "Polar Surface Area": "tpsa",
    "HBD (Lipinski)":     "hbd",
    "HBA (Lipinski)":     "hba",
    "Aromatic Rings":     "aromatic_rings_count",
}

DEFAULT_REQUIRED_COLS = [
    "QED Weighted",
    "Molecular Weight",
    "AlogP",
    "Polar Surface Area",
    "HBD (Lipinski)",
    "HBA (Lipinski)",
    "Aromatic Rings",
]

_CHUNKSIZE = 2000  # rows per task sent to each worker


# ---------------------------------------------------------------------------
# Module-level worker functions (must be top-level for pickling)
# ---------------------------------------------------------------------------

def _canonicalize_worker(smi: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None


def _hbd_hba_worker(smi: str) -> tuple[Optional[int], Optional[int]]:
    """Return (HBD, HBA) in a single mol parse."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    return Lipinski.NumHDonors(mol), Lipinski.NumHAcceptors(mol)


def _sa_worker(smi: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smi)
    return sascorer.calculateScore(mol) if mol else None


# ---------------------------------------------------------------------------
# Parallel map helper
# ---------------------------------------------------------------------------

def parallel_map(func, series: pd.Series, n_workers: int) -> list:
    """Apply func to every element of series using a process pool."""
    with Pool(processes=n_workers) as pool:
        return pool.map(func, series.tolist(), chunksize=_CHUNKSIZE)


# ---------------------------------------------------------------------------
# Step 1: Concatenation
# ---------------------------------------------------------------------------

def _detect_header_index(csv_files: list[Path], sep: str, smiles_col: str) -> int:
    for i, path in enumerate(csv_files):
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
        fields = [field.strip().strip('"') for field in first_line.split(sep)]
        if smiles_col in fields:
            print(f"  Header detected in {path.name}")
            return i
    raise ValueError(
        f"Could not find a header row containing '{smiles_col}' in any file. "
        f"Check --smiles-col and --sep."
    )


def concatenate_csvs(input_dir: str, sep: str, smiles_col: str) -> pd.DataFrame:
    csv_files = sorted(Path(input_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    header_idx = _detect_header_index(csv_files, sep, smiles_col)
    header_file = csv_files[header_idx]

    print(f"  Reading {header_file.name} (with header)...")
    header_df = _read_chembl_csv(header_file, sep=sep, col_names=None)
    col_names = list(header_df.columns)
    print(f"    {len(header_df):,} rows, {len(col_names)} columns")

    dfs = [header_df]
    for i, path in enumerate(csv_files):
        if i == header_idx:
            continue
        print(f"  Reading {path.name} (no header)...")
        df = _read_chembl_csv(path, sep=sep, col_names=col_names)
        print(f"    {len(df):,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined: {len(combined):,} rows")
    return combined


def _read_chembl_csv(path: Path, sep: str, col_names: list[str] | None) -> pd.DataFrame:
    kwargs = dict(
        sep=sep,
        quoting=csv.QUOTE_NONE,
        engine='python',
        encoding='utf-8',
        on_bad_lines='skip',
    )
    if col_names is not None:
        kwargs['header'] = None
        kwargs['names'] = col_names

    df = pd.read_csv(path, **kwargs)
    df.columns = df.columns.str.strip('"')
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip('"')
    return df


# ---------------------------------------------------------------------------
# Steps 2–3: Canonicalize + deduplicate  (parallelized)
# ---------------------------------------------------------------------------

def prepare(df: pd.DataFrame, smiles_col: str, n_workers: int) -> pd.DataFrame:
    """Canonicalize SMILES in parallel, then drop invalids and duplicates."""
    n_start = len(df)
    df = df.copy()

    t0 = time.time()
    print(f"  Canonicalizing {n_start:,} SMILES across {n_workers} workers...")
    df[smiles_col] = parallel_map(_canonicalize_worker, df[smiles_col], n_workers)
    print(f"  Done in {time.time() - t0:.0f}s")

    n_invalid = df[smiles_col].isna().sum()
    df = df.dropna(subset=[smiles_col])

    n_dupes = df.duplicated(subset=[smiles_col]).sum()
    df = df.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)

    print(f"  Start:       {n_start:,}")
    print(f"  Invalid:    -{n_invalid:,}")
    print(f"  Duplicates: -{n_dupes:,}")
    print(f"  After:       {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Step 4: Compute Lipinski HBD / HBA  (parallelized, single mol parse)
# ---------------------------------------------------------------------------

def add_lipinski_hbd_hba(df: pd.DataFrame, smiles_col: str, n_workers: int) -> pd.DataFrame:
    """Compute HBD and HBA in parallel with a single mol parse per molecule."""
    df = df.copy()
    t0 = time.time()
    print(f"  Computing HBD/HBA for {len(df):,} molecules across {n_workers} workers...")
    results = parallel_map(_hbd_hba_worker, df[smiles_col], n_workers)
    print(f"  Done in {time.time() - t0:.0f}s")

    hbd, hba = zip(*results)
    df["HBD (Lipinski)"] = list(hbd)
    df["HBA (Lipinski)"] = list(hba)

    print(f"  HBD — min: {df['HBD (Lipinski)'].min():.0f}  "
          f"max: {df['HBD (Lipinski)'].max():.0f}  "
          f"mean: {df['HBD (Lipinski)'].mean():.2f}")
    print(f"  HBA — min: {df['HBA (Lipinski)'].min():.0f}  "
          f"max: {df['HBA (Lipinski)'].max():.0f}  "
          f"mean: {df['HBA (Lipinski)'].mean():.2f}")
    return df


# ---------------------------------------------------------------------------
# Step 5: Drop missing properties
# ---------------------------------------------------------------------------

def drop_missing_properties(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    present = [c for c in required_cols if c in df.columns]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  Warning: required columns not found, skipping: {missing_cols}")
    n_before = len(df)
    df = df.dropna(subset=present).reset_index(drop=True)
    print(f"  Missing properties: -{n_before - len(df):,}  ({len(df):,} remaining)")
    return df


# ---------------------------------------------------------------------------
# Step 6: Exclude by SMILES
# ---------------------------------------------------------------------------

def exclude_by_smiles(
    df: pd.DataFrame,
    smiles_col: str,
    exclude_csv: str,
    exclude_smiles_col: str,
    exclude_sep: str = ",",
) -> pd.DataFrame:
    print(f"  Loading exclusion list from {exclude_csv}...")
    excl_df = pd.read_csv(exclude_csv, sep=exclude_sep, low_memory=False)

    if exclude_smiles_col not in excl_df.columns:
        raise KeyError(
            f"Column '{exclude_smiles_col}' not found in {exclude_csv}. "
            f"Available columns: {list(excl_df.columns)}"
        )

    # Exclusion SMILES are already canonical (output of clean_coconut_npdrug.py)
    excl_set = set(excl_df[exclude_smiles_col].dropna().unique())
    print(f"  Exclusion list: {len(excl_set):,} unique SMILES")

    n_before = len(df)
    df = df[~df[smiles_col].isin(excl_set)].reset_index(drop=True)
    print(f"  Excluded matches: -{n_before - len(df):,}  ({len(df):,} remaining)")
    return df


# ---------------------------------------------------------------------------
# Step 7: Keep required columns and rename
# ---------------------------------------------------------------------------

def slim_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in CHEMBL_KEEP_RENAME if c in df.columns]
    missing = [c for c in CHEMBL_KEEP_RENAME if c not in df.columns]
    if missing:
        print(f"  Warning: expected columns not found, skipping: {missing}")
    df = df[present].rename(columns=CHEMBL_KEEP_RENAME)
    print(f"  Kept {len(present)} columns, renamed to: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Step 8: SA score  (parallelized)
# ---------------------------------------------------------------------------

def compute_sa_scores(df: pd.DataFrame, smiles_col: str, n_workers: int) -> pd.DataFrame:
    """Compute RDKit SA score in parallel and append as 'sa_score' column."""
    df = df.copy()
    t0 = time.time()
    print(f"  Computing SA scores for {len(df):,} molecules across {n_workers} workers...")
    df["sa_score"] = parallel_map(_sa_worker, df[smiles_col], n_workers)
    print(f"  Done in {time.time() - t0:.0f}s")

    n_null = df["sa_score"].isna().sum()
    print(
        f"  SA score — min: {df['sa_score'].min():.2f}  "
        f"max: {df['sa_score'].max():.2f}  "
        f"mean: {df['sa_score'].mean():.2f}  "
        f"nulls: {n_null:,}"
    )
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing ChEMBL CSV files (e.g. data/raw/ChEMBL)",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Path for cleaned output CSV",
    )
    parser.add_argument(
        "--sep", default=";",
        help="Column separator in ChEMBL CSV files (default: ;)",
    )
    parser.add_argument(
        "--smiles-col", default="Smiles",
        help="SMILES column name in ChEMBL files (default: Smiles)",
    )
    parser.add_argument(
        "--required-cols", nargs="+", default=DEFAULT_REQUIRED_COLS,
        help="Columns (pre-rename names) that must be non-null.",
    )
    parser.add_argument(
        "--exclude-csv", default=None,
        help="Path to CSV containing SMILES to remove from the output (optional)",
    )
    parser.add_argument(
        "--exclude-smiles-col", default="canonical_smiles",
        help="SMILES column in --exclude-csv (default: canonical_smiles)",
    )
    parser.add_argument(
        "--exclude-sep", default=",",
        help="Separator used in --exclude-csv (default: ,)",
    )
    parser.add_argument(
        "--n-workers", type=int, default=os.cpu_count(),
        help=f"Parallel worker processes (default: {os.cpu_count()} = all cores)",
    )
    args = parser.parse_args()

    print(f"Using {args.n_workers} worker processes\n")

    print(f"Step 1: Concatenating CSVs from {args.input_dir}...")
    df = concatenate_csvs(args.input_dir, sep=args.sep, smiles_col=args.smiles_col)

    print("\nStep 2: Canonicalize, validate, deduplicate...")
    df = prepare(df, args.smiles_col, args.n_workers)

    print("\nStep 3: Compute Lipinski HBD / HBA from SMILES...")
    df = add_lipinski_hbd_hba(df, args.smiles_col, args.n_workers)

    print("\nStep 4: Drop rows with missing required properties...")
    print(f"  Required: {args.required_cols}")
    df = drop_missing_properties(df, args.required_cols)

    if args.exclude_csv:
        print("\nStep 5: Exclude molecules in exclusion list...")
        df = exclude_by_smiles(
            df, args.smiles_col,
            args.exclude_csv, args.exclude_smiles_col,
            args.exclude_sep,
        )
    else:
        print("\nStep 5: Skipped (no --exclude-csv provided)")

    print("\nStep 6: Keep required columns and rename to standard names...")
    df = slim_and_rename(df)

    print("\nStep 7: Compute SA scores...")
    df = compute_sa_scores(df, "canonical_smiles", args.n_workers)

    df.to_csv(args.output, index=False)
    print(f"\nFinal: {len(df):,} rows, {len(df.columns)} columns saved to {args.output}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
