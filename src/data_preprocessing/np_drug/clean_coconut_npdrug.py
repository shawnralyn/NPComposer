"""Validate, canonicalize, deduplicate, filter, and slim a COCONUT CSV for training.

Performs five steps in order:
  1. Canonicalize SMILES and drop invalid molecules
  2. Drop duplicate canonical SMILES
  3. Keep only required property columns (drops all metadata)
  4. Drop rows with missing values in required property columns
  5. Drop rows whose canonical SMILES appear in an exclusion CSV (optional)
  6. Compute RDKit SA score and append as 'sa_score' column

Column names are standardized on output so they match the ChEMBL clean output
and bin_cont_variables_npdrug.py defaults:
  qed_drug_likeliness        → qed
  topological_polar_surface_area → tpsa
  hydrogen_bond_donors_lipinski  → hbd
  hydrogen_bond_acceptors_lipinski → hba
  (all others kept as-is)

Usage:
    python src/data_preprocessing/np_drug/clean_coconut_npdrug.py \\
        --input  data/raw/coconut_csv_full.csv \\
        --output data/processed/coconut_clean.csv

    # Exclude molecules that appear in the np_drug set
    python src/data_preprocessing/np_drug/clean_coconut_npdrug.py \\
        --input  data/raw/coconut_csv_full.csv \\
        --output data/processed/coconut_clean.csv \\
        --exclude-csv  data/processed/np_drug.csv \\
        --exclude-smiles-col smiles
"""
import argparse
import os
import sys
from typing import Optional

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import RDConfig

RDLogger.DisableLog("rdApp.*")

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402 — must come after sys.path update


# ---------------------------------------------------------------------------
# Columns to keep from the raw COCONUT CSV (pre-rename names)
# All other columns are dropped before saving.
# ---------------------------------------------------------------------------

COCONUT_KEEP = [
    "canonical_smiles",
    "qed_drug_likeliness",           # → qed
    "molecular_weight",
    "alogp",
    "topological_polar_surface_area", # → tpsa
    "hydrogen_bond_donors_lipinski",  # → hbd
    "hydrogen_bond_acceptors_lipinski", # → hba
    "aromatic_rings_count",
]

COCONUT_RENAME = {
    "qed_drug_likeliness":              "qed",
    "topological_polar_surface_area":   "tpsa",
    "hydrogen_bond_donors_lipinski":    "hbd",
    "hydrogen_bond_acceptors_lipinski": "hba",
}

# Required to be non-null (uses original pre-rename names for the filter step)
DEFAULT_REQUIRED_COLS = [
    "qed_drug_likeliness",
    "molecular_weight",
    "alogp",
    "topological_polar_surface_area",
    "hydrogen_bond_donors_lipinski",
    "hydrogen_bond_acceptors_lipinski",
    "aromatic_rings_count",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def canonicalize(smiles: str) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def prepare(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Canonicalize SMILES and drop invalids and duplicates."""
    n_start = len(df)
    df = df.copy()

    df[smiles_col] = df[smiles_col].map(canonicalize)

    n_invalid = df[smiles_col].isna().sum()
    df = df.dropna(subset=[smiles_col])

    n_dupes = df.duplicated(subset=[smiles_col]).sum()
    df = df.drop_duplicates(subset=[smiles_col]).reset_index(drop=True)

    print(f"  Start:       {n_start:,}")
    print(f"  Invalid:    -{n_invalid:,}")
    print(f"  Duplicates: -{n_dupes:,}")
    print(f"  After:       {len(df):,}")
    return df


def slim_and_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed downstream and rename to standard names.

    Silently drops COCONUT_KEEP columns that are absent (e.g. if the raw CSV
    format changes), so the caller can inspect nulls from drop_missing_properties.
    """
    present = [c for c in COCONUT_KEEP if c in df.columns]
    missing = [c for c in COCONUT_KEEP if c not in df.columns]
    if missing:
        print(f"  Warning: expected columns not found, skipping: {missing}")
    df = df[present].rename(columns=COCONUT_RENAME)
    print(f"  Kept {len(present)} columns")
    return df


def drop_missing_properties(df: pd.DataFrame, required_cols: list[str]) -> pd.DataFrame:
    """Drop rows with missing values in any required property column."""
    present = [c for c in required_cols if c in df.columns]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  Warning: required columns not found, skipping: {missing_cols}")

    n_before = len(df)
    df = df.dropna(subset=present).reset_index(drop=True)
    print(f"  Missing properties: -{n_before - len(df):,}  ({len(df):,} remaining)")
    return df


def exclude_by_smiles(
    df: pd.DataFrame,
    smiles_col: str,
    exclude_csv: str,
    exclude_smiles_col: str,
) -> pd.DataFrame:
    """Drop rows whose canonical SMILES appear in an exclusion CSV."""
    print(f"  Loading exclusion list from {exclude_csv}...")
    excl_df = pd.read_csv(exclude_csv, low_memory=False)

    if exclude_smiles_col not in excl_df.columns:
        raise KeyError(
            f"Column '{exclude_smiles_col}' not found in {exclude_csv}. "
            f"Available columns: {list(excl_df.columns)}"
        )

    excl_smiles = (
        excl_df[exclude_smiles_col]
        .dropna()
        .map(canonicalize)
        .dropna()
        .unique()
    )
    excl_set = set(excl_smiles)
    print(f"  Exclusion list: {len(excl_set):,} unique canonical SMILES")

    n_before = len(df)
    df = df[~df[smiles_col].isin(excl_set)].reset_index(drop=True)
    print(f"  Excluded matches: -{n_before - len(df):,}  ({len(df):,} remaining)")
    return df


def compute_sa_scores(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Compute RDKit SA score for each molecule and append as 'sa_score' column.

    SA score ranges from 1 (easy to synthesize) to 10 (hard). Lower is better.
    """
    def _sa(smi):
        mol = Chem.MolFromSmiles(smi)
        return sascorer.calculateScore(mol) if mol else None

    df = df.copy()
    print(f"  Computing SA scores for {len(df):,} molecules...")
    df["sa_score"] = df[smiles_col].map(_sa)
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
    parser.add_argument("-i", "--input",   required=True, help="Path to raw COCONUT CSV")
    parser.add_argument("-o", "--output",  required=True, help="Path for cleaned output CSV")
    parser.add_argument(
        "--smiles-col", default="canonical_smiles",
        help="SMILES column name in raw CSV (default: canonical_smiles)",
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
        "--exclude-smiles-col", default="smiles",
        help="SMILES column in --exclude-csv (default: smiles)",
    )
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("\nStep 1: Canonicalize, validate, deduplicate...")
    df = prepare(df, args.smiles_col)

    print("\nStep 2: Keep required columns and rename to standard names...")
    df = slim_and_rename(df)
    print(f"  Output columns: {list(df.columns)}")

    print("\nStep 3: Drop rows with missing required properties...")
    # Required cols may have been renamed — map to post-rename names
    renamed_required = [COCONUT_RENAME.get(c, c) for c in args.required_cols]
    df = drop_missing_properties(df, renamed_required)

    if args.exclude_csv:
        print("\nStep 4: Exclude molecules in exclusion list...")
        df = exclude_by_smiles(df, args.smiles_col, args.exclude_csv, args.exclude_smiles_col)
    else:
        print("\nStep 4: Skipped (no --exclude-csv provided)")

    print("\nStep 5: Compute SA scores...")
    df = compute_sa_scores(df, args.smiles_col)

    df.to_csv(args.output, index=False)
    print(f"\nFinal: {len(df):,} rows, {len(df.columns)} columns saved to {args.output}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
