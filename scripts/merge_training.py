"""Merge COCONUT and NPASS subsets into a single training dataset.

Input:
    --coconut: path to COCONUT subset CSV.
    --npass: path to NPASS subset CSV.
    -o/--output: output CSV path.
Output:
    CSV with unified columns, source tag, superclass labels,
    and deduplicated by SMILES.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd



SMILES_CANDIDATES = [
    "canonical_smiles", "SMILES", "smiles", "Canonical_SMILES", "smi",
]


def detect_smiles_col(df):
    """Detect SMILES column name from DataFrame.

    Input:
        df: pandas DataFrame.
    Output:
        str, detected column name.
    """
    for c in SMILES_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")


def load_subset(path, source_name):
    """Load subset CSV and add source column.

    Input:
        path: CSV file path.
        source_name: dataset name tag (e.g. 'coconut', 'npass').
    Output:
        pandas DataFrame with 'source' column added.
    """
    df = pd.read_csv(path)
    df["source"] = source_name
    print(f"Loaded {source_name}: {len(df):,} rows, {len(df.columns)} cols")
    return df


def unify_columns(df, smiles_col):
    """Ensure standard column naming.

    Input:
        df: pandas DataFrame.
        smiles_col: original SMILES column name.
    Output:
        pandas DataFrame with 'smiles' column guaranteed.
    """
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})
    return df


def merge_and_deduplicate(dfs):
    """Concatenate DataFrames and drop duplicate SMILES.

    Input:
        dfs: list of pandas DataFrames, each with 'smiles' column.
    Output:
        pandas DataFrame, deduplicated.
    """
    merged = pd.concat(dfs, ignore_index=True)
    n_before = len(merged)
    merged = merged.drop_duplicates(subset="smiles", keep="first")
    n_after = len(merged)
    if n_before != n_after:
        print(f"Removed {n_before - n_after:,} duplicate SMILES")
    return merged


def fill_superclass(df, cache_dir="."):
    """Ensure every row has a superclass label.

    Priority: superclass > np_classifier_superclass > chemical_super_class.
    Remaining NaN values are filled via NPClassifier (local server).

    Input:
        df: merged DataFrame with 'smiles' column.
        cache_dir: directory for NPClassifier cache file.
    Output:
        pandas DataFrame with 'superclass' column filled.
    """
    # Unify from existing columns
    if "superclass" not in df.columns:
        df["superclass"] = pd.NA

    fallback_cols = ["np_classifier_superclass", "chemical_super_class"]
    for col in fallback_cols:
        if col in df.columns:
            mask = df["superclass"].isna()
            df.loc[mask, "superclass"] = df.loc[mask, col]

    missing = df["superclass"].isna()
    n_missing = missing.sum()
    print(f"Superclass: {len(df) - n_missing:,} filled, {n_missing:,} missing")

    if n_missing > 0:
        try:
            import requests, time
            print(f"Classifying {n_missing:,} molecules via NPClassifier API...")
            missing_smiles = df.loc[missing, "smiles"].tolist()
            labels = []
            for smi in missing_smiles:
                try:
                    r = requests.get("https://npclassifier.ucsd.edu/classify",
                                     params={"smiles": smi}, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    sc = data.get("superclass_results") or data.get("superclass")
                    labels.append(sc[0] if isinstance(sc, list) and sc else "Unknown")
                    time.sleep(0.5)
                except Exception:
                    labels.append("Unknown")
            df.loc[missing, "superclass"] = labels
        except ImportError:
            print("Warning: 'requests' not installed, filling with 'Unknown'")
            df.loc[missing, "superclass"] = "Unknown"

    return df


def main():
    parser = argparse.ArgumentParser(description="Merge subsets into training data")
    parser.add_argument("--coconut", required=True, help="COCONUT subset CSV")
    parser.add_argument("--npass", required=True, help="NPASS subset CSV")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()

    frames = []
    for path, name in [(args.coconut, "coconut"), (args.npass, "npass")]:
        df = load_subset(path, name)
        scol = detect_smiles_col(df)
        df = unify_columns(df, scol)
        frames.append(df)

    merged = merge_and_deduplicate(frames)

    # Fill superclass for all molecules
    cache_dir = str(Path(args.output).parent)
    merged = fill_superclass(merged, cache_dir=cache_dir)

    # Reorder: smiles and source first, rest alphabetical
    priority = ["smiles", "source"]
    rest = sorted([c for c in merged.columns if c not in priority])
    merged = merged[priority + rest]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    print(f"\nTraining data saved: {args.output}")
    print(f"  Total: {len(merged):,} molecules")
    for src in ["coconut", "npass"]:
        n = (merged["source"] == src).sum()
        print(f"  {src}: {n:,}")

    kept = [c for c in merged.columns if c not in ("smiles", "source")]
    print(f"  Columns ({len(merged.columns)}): smiles, source, {', '.join(kept[:10])}")
    if len(kept) > 10:
        print(f"    ... and {len(kept) - 10} more")


if __name__ == "__main__":
    main()
