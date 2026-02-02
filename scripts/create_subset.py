"""
NP Subset Creator (COCONUT / NPASS)
SA filtering + K-medoids clustering (CSV + SDF)

Usage:
    # COCONUT
    python create_subset.py -i data/raw/coconut_csv_full.csv -o data/processed/coconut_5k -s 5000
    python create_subset.py -i data/raw/coconut_csv_full.csv --sdf data/raw/coconut_sdf_3d_full.sdf -o data/processed/coconut_5k -s 5000
    
    # NPASS
    python create_subset.py -i data/raw/npass_full.csv -o data/processed/npass_5k -s 5000
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

try:
    from rdkit.Contrib.NP_Score import npscorer
    NP_MODEL = npscorer.readNPModel()
except:
    NP_MODEL = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def tanimoto_distance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    intersection = np.dot(X, X.T)
    bits = np.sum(X, axis=1)
    union = bits[:, None] + bits[None, :] - intersection
    union = np.where(union == 0, 1, union)
    return 1 - intersection / union


def euclidean_distance_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    sq_sum = np.sum(X ** 2, axis=1)
    dist = sq_sum[:, None] + sq_sum[None, :] - 2 * np.dot(X, X.T)
    dist = np.sqrt(np.maximum(dist, 0))
    max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist
    return dist


def kmedoids_combined(X_fp, X_props, n_clusters, max_iter=100, seed=42, fp_weight=0.7):
    np.random.seed(seed)
    n_samples = len(X_fp)
    
    dist_fp = tanimoto_distance_matrix(X_fp)
    dist_props = euclidean_distance_matrix(X_props)
    dist_matrix = fp_weight * dist_fp + (1 - fp_weight) * dist_props
    
    medoid_indices = np.random.choice(n_samples, n_clusters, replace=False)
    
    for _ in range(max_iter):
        distances_to_medoids = dist_matrix[:, medoid_indices]
        labels = np.argmin(distances_to_medoids, axis=1)
        
        new_medoids = []
        for k in range(n_clusters):
            cluster_mask = labels == k
            if not cluster_mask.any():
                new_medoids.append(medoid_indices[k])
                continue
            cluster_indices = np.where(cluster_mask)[0]
            cluster_dists = dist_matrix[np.ix_(cluster_indices, cluster_indices)]
            total_dists = cluster_dists.sum(axis=1)
            best_idx = cluster_indices[np.argmin(total_dists)]
            new_medoids.append(best_idx)
        
        new_medoids = np.array(new_medoids)
        if np.array_equal(new_medoids, medoid_indices):
            break
        medoid_indices = new_medoids
    
    return medoid_indices


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


def calc_npl(smiles: str) -> Optional[float]:
    if NP_MODEL is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return npscorer.scoreMol(mol, NP_MODEL)
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
    results = []
    iterator = tqdm(series, desc=desc) if HAS_TQDM else series
    for val in iterator:
        results.append(func(val))
    return results


def detect_smiles_column(df):
    """Detect SMILES column name"""
    for col in ['canonical_smiles', 'SMILES', 'smiles']:
        if col in df.columns:
            return col
    return None


def detect_id_column(df):
    """Detect ID column name"""
    for col in ['identifier', 'np_id', 'coconut_id', 'ID']:
        if col in df.columns:
            return col
    return None


def load_and_filter(
    input_path: str,
    smiles_col: str = None,
    sa_max: float = 6.0,
    mw_min: float = 150,
    mw_max: float = 800
) -> tuple:
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    n_init = len(df)
    print(f"  {n_init:,} molecules")
    
    # Auto-detect SMILES column
    if smiles_col is None or smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)
        if smiles_col is None:
            raise ValueError("SMILES column not found")
    print(f"  SMILES column: {smiles_col}")
    
    # Filter valid SMILES
    print(f"Filtering valid SMILES...")
    valid_mask = df[smiles_col].apply(lambda x: pd.notna(x) and x != 'n.a.' and Chem.MolFromSmiles(str(x)) is not None)
    df = df[valid_mask].copy()
    print(f"  {n_init:,} -> {len(df):,}")
    
    # Calculate MW if not present
    print(f"Filtering MW ({mw_min}-{mw_max})...")
    if 'molecular_weight' not in df.columns:
        df['molecular_weight'] = df[smiles_col].apply(
            lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))
        )
    n_before = len(df)
    df = df[(df['molecular_weight'] >= mw_min) & (df['molecular_weight'] <= mw_max)].copy()
    print(f"  {n_before:,} -> {len(df):,}")
    
    # Calculate SA score
    print(f"Calculating SA score (filtering <= {sa_max})...")
    df['sa_score(RDKit)'] = apply_with_progress(df[smiles_col].tolist(), calc_sa, "SA")
    n_before = len(df)
    df = df[df['sa_score(RDKit)'].notna() & (df['sa_score(RDKit)'] <= sa_max)].copy()
    print(f"  {n_before:,} -> {len(df):,}")
    
    # Calculate QED
    print("Calculating QED...")
    df['qed(RDKit)'] = apply_with_progress(df[smiles_col].tolist(), calc_qed, "QED")
    
    # Calculate NPL score
    print("Calculating NPL score...")
    if NP_MODEL:
        df['npl_score(RDKit)'] = apply_with_progress(df[smiles_col].tolist(), calc_npl, "NPL")
    else:
        print("  NPL model not available, skipping")
    
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
    
    X_fp = np.array(fps)
    print(f"  {len(X_fp):,} fingerprints")
    
    # Get properties for clustering
    valid_df = df.loc[valid_idx]
    
    sa_vals = valid_df['sa_score(RDKit)'].fillna(valid_df['sa_score(RDKit)'].mean()).values
    qed_vals = valid_df['qed(RDKit)'].fillna(valid_df['qed(RDKit)'].mean()).values
    
    if 'npl_score(RDKit)' in valid_df.columns:
        npl_vals = valid_df['npl_score(RDKit)'].fillna(valid_df['npl_score(RDKit)'].mean()).values
        use_npl = True
    else:
        use_npl = False
    
    def normalize(arr):
        min_val, max_val = arr.min(), arr.max()
        if max_val - min_val == 0:
            return np.zeros_like(arr)
        return (arr - min_val) / (max_val - min_val)
    
    sa_norm = normalize(sa_vals)
    qed_norm = normalize(qed_vals)
    
    if use_npl:
        npl_norm = normalize(npl_vals)
        X_props = np.column_stack([sa_norm, qed_norm, npl_norm])
        print(f"  Properties: SA, QED, NPL (normalized)")
    else:
        X_props = np.column_stack([sa_norm, qed_norm])
        print(f"  Properties: SA, QED (normalized)")
    
    print(f"  Clustering (k={target_size}, Tanimoto + Euclidean)...")
    medoid_indices = kmedoids_combined(X_fp, X_props, target_size, max_iter=100, seed=seed)
    
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
        for prop in ['IDENTIFIER', 'identifier', 'ID', 'id', 'np_id', '_Name']:
            if mol.HasProp(prop):
                mol_id = mol.GetProp(prop)
                break
        
        if mol_id is None:
            continue
        
        base_id = mol_id.split('.')[0] if '.' in mol_id else mol_id
        
        if base_id in identifiers and base_id not in found_ids:
            writer.write(mol)
            found_ids.add(base_id)
        
        if len(found_ids) >= len(identifiers):
            break
    
    writer.close()
    print(f"  Found {len(found_ids)}/{len(identifiers)} molecules")
    print(f"  Saved: {output_path}")
    
    return len(found_ids)


def save_subset(df: pd.DataFrame, output_path: str, smiles_col: str):
    
    out_df = df.copy()
    out_df.to_csv(output_path, index=False)
    
    print(f"Saved CSV: {output_path}")
    print(f"  {len(out_df):,} molecules, {len(out_df.columns)} columns")
    
    for col in ['sa_score(RDKit)', 'qed(RDKit)', 'npl_score(RDKit)']:
        if col in out_df.columns:
            v = out_df[col].dropna()
            print(f"  {col}: mean={v.mean():.3f}, std={v.std():.3f}")
    
    id_col = detect_id_column(df)
    return id_col


def main():
    parser = argparse.ArgumentParser(description="Create NP subset (COCONUT/NPASS)")
    parser.add_argument("-i", "--input", required=True, help="Input CSV (COCONUT or NPASS)")
    parser.add_argument("--sdf", help="Input SDF (optional)")
    parser.add_argument("-o", "--output", required=True, help="Output prefix")
    parser.add_argument("-s", "--size", type=int, default=5000, help="Target size")
    parser.add_argument("--sa_max", type=float, default=6.0, help="Max SA score")
    parser.add_argument("--mw_min", type=float, default=150, help="Min MW")
    parser.add_argument("--mw_max", type=float, default=800, help="Max MW")
    parser.add_argument("--smiles_col", help="SMILES column (auto-detect if not set)")
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
