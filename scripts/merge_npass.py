"""
Merge NPASS 3.0 files into single CSV

Usage:
    python merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv
"""

import argparse
import pandas as pd
from pathlib import Path


def load_tsv(path):
    """Load TSV/TXT file"""
    return pd.read_csv(path, sep='\t', low_memory=False)


def merge_npass(input_dir: str, output_path: str):
    
    input_dir = Path(input_dir)
    
    # Load main files
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
    
    # Merge structure into general (1:1 on np_id)
    print("\nMerging structure...")
    df = general.merge(structure, on='np_id', how='left')
    print(f"  {len(df):,} molecules")
    
    # Aggregate activities per np_id
    print("Aggregating activities...")
    if len(activities) > 0:
        act_agg = activities.groupby('np_id').agg({
            'target_id': 'count',
            'activity_type': lambda x: '|'.join(x.dropna().unique()[:5]),
            'activity_value': 'mean'
        }).rename(columns={
            'target_id': 'activity_count',
            'activity_type': 'activity_types',
            'activity_value': 'activity_value_mean'
        }).reset_index()
        df = df.merge(act_agg, on='np_id', how='left')
    
    # Aggregate species per np_id
    print("Aggregating species...")
    if len(species_pair) > 0:
        sp_agg = species_pair.groupby('np_id').agg({
            'org_id': 'count',
            'org_name': lambda x: '|'.join(x.dropna().unique()[:5])
        }).rename(columns={
            'org_id': 'organism_count',
            'org_name': 'organisms'
        }).reset_index()
        df = df.merge(sp_agg, on='np_id', how='left')
    
    # Aggregate toxicity per np_id
    print("Aggregating toxicity...")
    if len(toxicity) > 0:
        tox_agg = toxicity.groupby('np_id').agg({
            'activity_type': lambda x: '|'.join(x.dropna().unique()[:3]),
            'activity_value': 'mean'
        }).rename(columns={
            'activity_type': 'toxicity_types',
            'activity_value': 'toxicity_value_mean'
        }).reset_index()
        df = df.merge(tox_agg, on='np_id', how='left')
    
    # Filter to molecules with SMILES
    print("\nFiltering molecules with SMILES...")
    n_before = len(df)
    df = df[df['SMILES'].notna() & (df['SMILES'] != 'n.a.')].copy()
    print(f"  {n_before:,} -> {len(df):,}")
    
    # Rename for consistency with COCONUT
    df = df.rename(columns={
        'SMILES': 'canonical_smiles',
        'InChI': 'standard_inchi',
        'InChIKey': 'standard_inchi_key',
        'pref_name': 'name'
    })
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(df):,} molecules, {len(df.columns)} columns")
    
    # Show columns
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
