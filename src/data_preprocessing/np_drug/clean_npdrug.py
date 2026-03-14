"""Clean and process NP-Drug Excel file into a structured CSV.
"""

import argparse
import pandas as pd
import pubchempy as pcp
from pathlib import Path
from rdkit import Chem


def load_npdrug(path):
    """Load the Raw sheet from the NP-Drug Excel file.

    Input:
        path: path to np_drug.xlsx.
    Output:
        pd.DataFrame with all raw columns.
    """
    return pd.read_excel(
        path,
        sheet_name="Raw",
        header=1,    # row 2 in Excel = index 1
        nrows=593,   # rows 3-595 = 593 data rows
    )


def filter_and_rename(df):
    """Filter rows and retain only relevant columns.

    Keeps rows where 'Parent Natural Product' is not blank and
    'Structure' is not 'duplicate'. Renames columns to snake_case.

    Input:
        df: raw NP-Drug DataFrame.
    Output:
        Filtered and renamed pd.DataFrame.
    """
    df = df[
        df["Parent Natural Product"].notna() &
        (df["Structure"].str.strip().str.lower() != "duplicate")
    ]

    df = df[["Generic Name", "SMILES", "Parent Natural Product"]].rename(columns={
        "Generic Name": "drug_name",
        "SMILES": "drug_smiles",
        "Parent Natural Product": "parent_np_name",
    }).reset_index(drop=True)

    return df


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string using RDKit.

    Input:
        smiles: SMILES string.
    Output:
        Canonical SMILES string, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def name_to_smiles(name):
    """Look up a SMILES string from a compound name via PubChem.

    Input:
        name: compound name string.
    Output:
        SMILES string, or None if not found.
    """
    try:
        compounds = pcp.get_compounds(name, "name")
        if compounds:
            return compounds[0].connectivity_smiles
        return None
    except Exception as e:
        print(f"Warning: error looking up '{name}': {e}")
        return None


def clean_npdrug(input_path, output_path):
    """Full cleaning pipeline for NP-Drug Excel file.

    Input:
        input_path: path to np_drug.xlsx.
        output_path: path for the cleaned output CSV.
    Output:
        Writes cleaned CSV to output_path.
    """
    print("Loading NP-Drug Excel file...")
    df = load_npdrug(input_path)
    print(f"  {len(df):,} rows loaded")

    print("Filtering and renaming columns...")
    df = filter_and_rename(df)
    print(f"  {len(df):,} rows after filtering")

    print("Canonicalizing SMILES...")
    df["drug_smiles"] = df["drug_smiles"].apply(canonicalize_smiles)
    n_invalid = df["drug_smiles"].isna().sum()
    if n_invalid:
        print(f"  Warning: {n_invalid} invalid SMILES dropped")
    df = df.dropna(subset=["drug_smiles"]).reset_index(drop=True)
    print(f"  {len(df):,} rows with valid SMILES")

    print("Looking up parent NP SMILES from PubChem...")
    df["parent_np_smiles"] = df["parent_np_name"].apply(name_to_smiles)
    n_missing = df["parent_np_smiles"].isna().sum()
    print(f"  {n_missing:,} parent NP SMILES not found")

    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    print("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")


def main():
    parser = argparse.ArgumentParser(description="Clean NP-Drug Excel file into CSV")
    parser.add_argument("-i", "--input", required=True, help="Path to np_drug.xlsx")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    args = parser.parse_args()
    clean_npdrug(args.input, args.output)


if __name__ == "__main__":
    main()
