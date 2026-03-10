"""Merge NPASS 3.0 TSV files into a single CSV."""

import argparse
import pandas as pd
from pathlib import Path


def load_tsv(path):
    """Load a TSV/TXT file.

    Input:
        path: file path.
    Output:
        pd.DataFrame.
    """
    return pd.read_csv(path, sep='\t', low_memory=False)


def merge_npass(input_dir, output_path):
    """Merge NPASS 3.0 data files into a single CSV.

    Input:
        input_dir: directory containing NPASS TSV/TXT files.
        output_path: path for the merged output CSV.
    Output:
        Writes merged CSV to output_path.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    required_files = [
        "NPASS3.0_naturalproducts_generalinfo.txt",
        "NPASS3.0_naturalproducts_structure.txt",
    ]
    for fname in required_files:
        if not (input_dir / fname).exists():
            raise FileNotFoundError(f"Required file missing: {input_dir / fname}")

    print("Loading generalinfo...")
    general = load_tsv(input_dir / "NPASS3.0_naturalproducts_generalinfo.txt")
    print(f"  {len(general):,} molecules")

    print("Loading structure...")
    structure = load_tsv(input_dir / "NPASS3.0_naturalproducts_structure.txt")
    print(f"  {len(structure):,} structures")

    print("Loading activities...")
    activities = load_tsv(input_dir / "NPASS3.0_activities.txt")
    print(f"  {len(activities):,} activity records")

    print("Loading species_pair...")
    species_pair = load_tsv(input_dir / "NPASS3.0_naturalproducts_species_pair.txt")
    print(f"  {len(species_pair):,} species pairs")

    print("Loading target...")
    target = load_tsv(input_dir / "NPASS3.0_target.txt")
    print(f"  {len(target):,} targets")

    print("Loading species_info...")
    species_info = load_tsv(input_dir / "NPASS3.0_species_info.txt")
    print(f"  {len(species_info):,} species")

    print("Loading toxicity...")
    toxicity = load_tsv(input_dir / "NPASS3.0_toxicity.txt")
    print(f"  {len(toxicity):,} toxicity records")

    # Structure file has different np_ids from generalinfo in NPASS 3.0
    # Use structure as the base (it contains SMILES), left-join generalinfo
    print("\nMerging structure (base) with generalinfo...")
    # Check overlap
    common = set(structure['np_id']) & set(general['np_id'])
    print(f"  np_id overlap: {len(common):,} / structure={len(structure):,}, general={len(general):,}")

    if len(common) > len(structure) * 0.5:
        # Good overlap: merge normally
        df = general.merge(structure, on='np_id', how='left')
    else:
        # Low overlap: use structure as base, left-join generalinfo
        print("  Low overlap — using structure as base")
        df = structure.merge(general, on='np_id', how='left')
    print(f"  {len(df):,} molecules")

    print("Aggregating activities...")
    if len(activities) > 0:
        activities['activity_value'] = pd.to_numeric(
            activities['activity_value'], errors='coerce')
        act_agg = activities.groupby('np_id').agg({
            'target_id': 'count',
            'activity_type': lambda x: '|'.join(x.dropna().astype(str).unique()[:5]),
            'activity_value': 'mean'
        }).rename(columns={
            'target_id': 'activity_count',
            'activity_type': 'activity_types',
            'activity_value': 'activity_value_mean'
        }).reset_index()
        df = df.merge(act_agg, on='np_id', how='left')

    print("Aggregating species...")
    if len(species_pair) > 0:
        sp_cols = species_pair.columns.tolist()
        name_col = next((c for c in ['org_name', 'species_name', 'name']
                         if c in sp_cols), None)
        count_col = next((c for c in ['org_id', 'species_id']
                          if c in sp_cols), sp_cols[1] if len(sp_cols) > 1 else sp_cols[0])
        agg_dict = {count_col: 'count'}
        rename_dict = {count_col: 'organism_count'}
        if name_col:
            agg_dict[name_col] = lambda x: '|'.join(x.dropna().astype(str).unique()[:5])
            rename_dict[name_col] = 'organisms'
        sp_agg = species_pair.groupby('np_id').agg(agg_dict).rename(
            columns=rename_dict).reset_index()
        df = df.merge(sp_agg, on='np_id', how='left')

    print("Aggregating toxicity...")
    if len(toxicity) > 0:
        toxicity['activity_value'] = pd.to_numeric(
            toxicity['activity_value'], errors='coerce')
        tox_agg = toxicity.groupby('np_id').agg({
            'activity_type': lambda x: '|'.join(x.dropna().astype(str).unique()[:3]),
            'activity_value': 'mean'
        }).rename(columns={
            'activity_type': 'toxicity_types',
            'activity_value': 'toxicity_value_mean'
        }).reset_index()
        df = df.merge(tox_agg, on='np_id', how='left')

    print("\nFiltering molecules with SMILES...")
    n_before = len(df)
    df = df[df['SMILES'].notna() & (df['SMILES'] != 'n.a.')].copy()
    print(f"  {n_before:,} -> {len(df):,}")

    df = df.rename(columns={
        'SMILES': 'canonical_smiles',
        'InChI': 'standard_inchi',
        'InChIKey': 'standard_inchi_key',
        'pref_name': 'name'
    })

    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(df):,} molecules, {len(df.columns)} columns")

    print("\nColumns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")


def main():
    parser = argparse.ArgumentParser(description="Merge NPASS files into CSV")
    parser.add_argument("-i", "--input", required=True, help="NPASS directory")
    parser.add_argument("-o", "--output", required=True, help="Output CSV")
    args = parser.parse_args()
    merge_npass(args.input, args.output)


if __name__ == "__main__":
    main()
