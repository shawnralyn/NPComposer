"""
COCONUT Subset Creator
SA filtering + K-medoids clustering (CSV + SDF)

Usage:
    python create_subset.py -i data/raw/coconut_csv_full.csv -o data/processed/subset_5k -s 5000
    python create_subset.py -i data/raw/coconut_csv_full.csv --sdf data/raw/coconut_sdf_3d_full.sdf -o data/processed/subset_5k -s 5000
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import QED
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from sklearn_extra.cluster import KMedoids

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def calc_sa(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return sascorer.calculateScore(mol)
    except:
        return None


def calc_qed(smiles: str) -> Optional[float]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return QED.qed(mol)
    except:
        return None


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except:
        return None


def apply_with_progress(series, func, desc):
    """Apply function with progress bar"""
    results = []
    iterator = tqdm(series, desc=desc) if HAS_TQDM else series
    for val in iterator:
        results.append(func(val))
    return results


def load_and_filter(
    input_path: str,
    smiles_col: str = "canonical_smiles",
    sa_max: float = 6.0,
    mw_min: float = 150,
    mw_max: float = 800
) -> tuple:
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    n_init = len(df)
    print(f"  {n_init:,} molecules")
    
    if smiles_col not in df.columns:
        for col in ['smiles', 'SMILES', 'canonical_smiles']:
            if col in df.columns:
                smiles_col = col
                break
    
    print(f"Filtering valid SMILES...")
    valid_mask = df[smiles_col].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
    df = df[valid_mask].copy()
    print(f"  {n_init:,} -> {len(df):,}")
    
    print(f"Filtering MW ({mw_min}-{mw_max})...")
    if 'molecular_weight' not in df.columns:
        df['molecular_weight'] = df[smiles_col].apply(
            lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))
        )
    n_before = len(df)
    df = df[(df['molecular_weight'] >= mw_min) & (df['molecular_weight'] <= mw_max)].copy()
    print(f"  {n_before:,} -> {len(df):,}")
    
    print(f"Calculating SA score (filtering <= {sa_max})...")
    df['sa_score'] = apply_with_progress(df[smiles_col].tolist(), calc_sa, "SA")
    n_before = len(df)
    df = df[df['sa_score'].notna() & (df['sa_score'] <= sa_max)].copy()
    print(f"  {n_before:,} -> {len(df):,}")
    
    print("Calculating QED...")
    df['qed'] = apply_with_progress(df[smiles_col].tolist(), calc_qed, "QED")
    
    print(f"Filtering done: {n_init:,} -> {len(df):,} ({100*len(df)/n_init:.1f}%)")
    
    return df, smiles_col


def kmedoids_subset(
    df: pd.DataFrame,
    smiles_col: str,
    target_size: int = 5000,
    seed: int = 42
) -> pd.DataFrame:
    
    print(f"K-medoids clustering (n={target_size})...")
    
    if len(df) <= target_size:
        print(f"  Data ({len(df)}) <= target, returning all")
        return df
    
    print("  Computing fingerprints...")
    fps, valid_idx = [], []
    
    smiles_list = df[smiles_col].tolist()
    indices = df.index.tolist()
    iterator = tqdm(zip(indices, smiles_list), total=len(smiles_list), desc="FP") if HAS_TQDM else zip(indices, smiles_list)
    
    for idx, smi in iterator:
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)
    
    X = np.array(fps)
    print(f"  {len(X):,} fingerprints")
    
    print(f"  Clustering (k={target_size})...")
    kmedoids = KMedoids(
        n_clusters=target_size,
        random_state=seed,
        method='alternate',
        max_iter=100
    )
    kmedoids.fit(X)
    
    # Medoids are actual data points
    medoid_indices = kmedoids.medoid_indices_
    selected = [valid_idx[i] for i in medoid_indices]
    
    subset = df.loc[selected].copy()
    print(f"  Selected {len(subset):,} molecules")
    
    return subset


def extract_sdf_subset(sdf_path: str, output_path: str, identifiers: Set[str]):
    
    print(f"Extracting SDF subset...")
    print(f"  Target: {len(identifiers)} molecules")
    
    suppl = Chem.SDMolSupplier(sdf_path)
    writer = Chem.SDWriter(output_path)
    
    found_ids = set()
    
    iterator = tqdm(suppl, desc="SDF") if HAS_TQDM else suppl
    
    for mol in iterator:
        if mol is None:
            continue
        
        mol_id = None
        for prop in ['IDENTIFIER', 'identifier', 'ID', 'id', '_Name']:
            if mol.HasProp(prop):
                mol_id = mol.GetProp(prop)
                break
        
        if mol_id is None:
            continue
        
        # Get base ID (remove version suffix like .0, .1)
        base_id = mol_id.split('.')[0] if '.' in mol_id else mol_id
        
        # Only take first match per identifier
        if base_id in identifiers and base_id not in found_ids:
            writer.write(mol)
            found_ids.add(base_id)
        
        if len(found_ids) >= len(identifiers):
            break
    
    writer.close()
    print(f"  Found {len(found_ids)}/{len(identifiers)} molecules")
    print(f"  Saved: {output_path}")
    
    return len(found_ids)


def save_subset(df: pd.DataFrame, output_path: str, smiles_col: str, id_col: str = None):
    
    cols = [smiles_col, 'sa_score', 'qed']
    optional = ['molecular_weight', 'npl_score', 'NPClassifier_pathway', 
                'NPClassifier_superclass', 'NPClassifier_class']
    cols += [c for c in optional if c in df.columns]
    
    if id_col is None:
        for c in ['identifier', 'coconut_id', 'ID']:
            if c in df.columns:
                id_col = c
                break
    
    if id_col and id_col not in cols:
        cols = [id_col] + cols
    
    out_df = df[cols].copy()
    out_df.to_csv(output_path, index=False)
    
    print(f"Saved CSV: {output_path}")
    print(f"  {len(out_df):,} molecules")
    
    for col in ['sa_score', 'qed', 'npl_score']:
        if col in out_df.columns:
            v = out_df[col].dropna()
            print(f"  {col}: mean={v.mean():.3f}, std={v.std():.3f}")
    
    return id_col


def main():
    parser = argparse.ArgumentParser(description="Create COCONUT subset (CSV + SDF)")
    parser.add_argument("-i", "--input", required=True, help="Input CSV")
    parser.add_argument("--sdf", help="Input SDF (optional)")
    parser.add_argument("-o", "--output", required=True, help="Output prefix")
    parser.add_argument("-s", "--size", type=int, default=5000, help="Target size")
    parser.add_argument("--sa_max", type=float, default=6.0, help="Max SA score")
    parser.add_argument("--mw_min", type=float, default=150, help="Min MW")
    parser.add_argument("--mw_max", type=float, default=800, help="Max MW")
    parser.add_argument("--smiles_col", default="canonical_smiles", help="SMILES column")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    df, smiles_col = load_and_filter(
        args.input, args.smiles_col, args.sa_max, args.mw_min, args.mw_max
    )
    
    subset = kmedoids_subset(df, smiles_col, args.size, args.seed)
    
    csv_path = f"{args.output}.csv"
    id_col = save_subset(subset, csv_path, smiles_col)
    
    if args.sdf:
        if id_col and id_col in subset.columns:
            # Use base IDs only (without version suffix)
            identifiers = set()
            for i in subset[id_col].astype(str).tolist():
                base_id = i.split('.')[0] if '.' in i else i
                identifiers.add(base_id)
            
            sdf_path = f"{args.output}.sdf"
            extract_sdf_subset(args.sdf, sdf_path, identifiers)
        else:
            print("SDF skipped: no identifier column")
    
    print("Done")


if __name__ == "__main__":
    main()
