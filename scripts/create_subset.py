"""NP subset creator with SA filtering and K-medoids clustering."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set
from collections import Counter
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
except (ImportError, FileNotFoundError, OSError):
    NP_MODEL = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ClassyFire classification (optional)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "classification"))
try:
    from classyfire import classify_batch
    HAS_CLASSYFIRE = True
except ImportError:
    HAS_CLASSYFIRE = False


def tanimoto_distance_matrix(X):
    """Compute pairwise Tanimoto distance matrix (full n*n).

    Input:
        X: np.ndarray of shape (n, d), binary fingerprint vectors.
    Output:
        np.ndarray of shape (n, n), distance values in [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    intersection = np.dot(X, X.T)
    bits = np.sum(X, axis=1)
    union = bits[:, None] + bits[None, :] - intersection
    union = np.where(union == 0, 1, union)
    return 1 - intersection / union


def euclidean_distance_matrix(X):
    """Compute normalized pairwise Euclidean distance matrix (full n*n).

    Input:
        X: np.ndarray of shape (n, d).
    Output:
        np.ndarray of shape (n, n), normalized to [0, 1].
    """
    X = np.asarray(X, dtype=np.float64)
    sq_sum = np.sum(X ** 2, axis=1)
    dist = sq_sum[:, None] + sq_sum[None, :] - 2 * np.dot(X, X.T)
    dist = np.sqrt(np.maximum(dist, 0))
    max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist
    return dist


def _tanimoto_cross(A, B):
    """Tanimoto distance between rows of A and rows of B.

    Input:
        A: np.ndarray of shape (m, d), float32 fingerprints.
        B: np.ndarray of shape (p, d), float32 fingerprints.
    Output:
        np.ndarray of shape (m, p), Tanimoto distances in [0, 1].
    """
    inter = A @ B.T
    bits_a = A.sum(axis=1)
    bits_b = B.sum(axis=1)
    union = bits_a[:, None] + bits_b[None, :] - inter
    union = np.where(union == 0, 1, union)
    return 1.0 - inter / union


def _euclidean_cross_norm(A, B, norm_factor):
    """Normalized Euclidean distance between rows of A and rows of B.

    Input:
        A: np.ndarray of shape (m, d).
        B: np.ndarray of shape (p, d).
        norm_factor: float, normalization divisor (e.g. sqrt(ndim)).
    Output:
        np.ndarray of shape (m, p), distances in [0, ~1].
    """
    sq_a = (A ** 2).sum(axis=1)
    sq_b = (B ** 2).sum(axis=1)
    d2 = sq_a[:, None] + sq_b[None, :] - 2.0 * A @ B.T
    return np.sqrt(np.maximum(d2, 0)) / max(norm_factor, 1e-10)


def kmedoids_combined(X_fp, X_props, n_clusters, max_iter=100, seed=42,
                      fp_weight=0.7, chunk_size=2000):
    """Memory-efficient K-medoids on combined Tanimoto + Euclidean distance.

    Avoids building the full n*n distance matrix. Instead, computes
    distances in chunks during assignment (n*k per chunk) and uses
    small within-cluster matrices for medoid updates.

    Memory: O(chunk_size * k + max_cluster_size^2) instead of O(n^2).

    Input:
        X_fp: np.ndarray, fingerprint matrix (n, d).
        X_props: np.ndarray, property matrix (n, p), values in [0, 1].
        n_clusters: int, number of medoids to select.
        max_iter: int, maximum iterations.
        seed: int, random seed (default 42).
        fp_weight: float, weight for Tanimoto vs Euclidean distance.
        chunk_size: int, batch size for distance computation.
    Output:
        np.ndarray of selected medoid indices.
    """
    np.random.seed(seed)
    n = len(X_fp)
    p = X_props.shape[1]
    norm_factor = np.sqrt(p)

    X_fp_f32 = np.asarray(X_fp, dtype=np.float32)
    X_props_f32 = np.asarray(X_props, dtype=np.float32)

    medoid_indices = np.random.choice(n, n_clusters, replace=False)

    for it in range(max_iter):
        # Assignment: find nearest medoid for each point (chunked)
        med_fp = X_fp_f32[medoid_indices]
        med_props = X_props_f32[medoid_indices]
        labels = np.empty(n, dtype=np.int32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            d_tan = _tanimoto_cross(X_fp_f32[start:end], med_fp)
            d_euc = _euclidean_cross_norm(X_props_f32[start:end], med_props,
                                          norm_factor)
            d = fp_weight * d_tan + (1 - fp_weight) * d_euc
            labels[start:end] = np.argmin(d, axis=1)
            del d_tan, d_euc, d

        # Update: find best medoid in each cluster
        new_medoids = np.empty(n_clusters, dtype=np.intp)
        for k in range(n_clusters):
            cluster_idx = np.where(labels == k)[0]
            if len(cluster_idx) == 0:
                new_medoids[k] = medoid_indices[k]
                continue
            if len(cluster_idx) == 1:
                new_medoids[k] = cluster_idx[0]
                continue

            cl_fp = X_fp_f32[cluster_idx]
            cl_props = X_props_f32[cluster_idx]
            d_tan = _tanimoto_cross(cl_fp, cl_fp)
            d_euc = _euclidean_cross_norm(cl_props, cl_props, norm_factor)
            d = fp_weight * d_tan + (1 - fp_weight) * d_euc
            best = np.argmin(d.sum(axis=1))
            new_medoids[k] = cluster_idx[best]

        if np.array_equal(new_medoids, medoid_indices):
            print(f"    Converged at iteration {it + 1}")
            break
        medoid_indices = new_medoids

    return medoid_indices


def calc_sa(smiles: str) -> Optional[float]:
    """Compute synthetic accessibility score.

    Input:
        smiles: SMILES string.
    Output:
        float in [1, 10] (lower = easier) or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return sascorer.calculateScore(mol)
    except (ValueError, RuntimeError) as e:
        print(f"  Warning: SA failed for {smiles[:30]}: {e}")
        return None


def calc_qed(smiles: str) -> Optional[float]:
    """Compute quantitative estimate of drug-likeness.

    Input:
        smiles: SMILES string.
    Output:
        float in [0, 1] (higher = better) or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return QED.qed(mol)
    except (ValueError, RuntimeError) as e:
        print(f"  Warning: QED failed for {smiles[:30]}: {e}")
        return None


def calc_npl(smiles: str) -> Optional[float]:
    """Compute NP-likeness score.

    Input:
        smiles: SMILES string.
    Output:
        float in [-3, +3] (higher = more NP-like) or None on failure.
    """
    if NP_MODEL is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return npscorer.scoreMol(mol, NP_MODEL)
    except (ValueError, RuntimeError) as e:
        print(f"  Warning: NPL failed for {smiles[:30]}: {e}")
        return None


def smiles_to_fp(smiles: str, radius: int = 2, n_bits: int = 1024) -> Optional[np.ndarray]:
    """Convert SMILES to Morgan fingerprint bit vector.

    Input:
        smiles: SMILES string.
        radius: Morgan radius (default 2).
        n_bits: fingerprint length (default 1024).
    Output:
        np.ndarray of shape (n_bits,) with dtype int8, or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except (ValueError, RuntimeError) as e:
        print(f"  Warning: FP failed for {smiles[:30]}: {e}")
        return None


def apply_with_progress(series, func, desc):
    """Apply func to each element with optional tqdm progress bar."""
    results = []
    iterator = tqdm(series, desc=desc) if HAS_TQDM else series
    for val in iterator:
        results.append(func(val))
    return results


def detect_smiles_column(df):
    """Auto-detect SMILES column name.

    Input:
        df: pd.DataFrame.
    Output:
        str column name or None if not found.
    """
    for col in ['canonical_smiles', 'SMILES', 'smiles']:
        if col in df.columns:
            return col
    return None


def detect_id_column(df):
    """Auto-detect ID column name.

    Input:
        df: pd.DataFrame.
    Output:
        str column name or None if not found.
    """
    for col in ['identifier', 'np_id', 'coconut_id', 'ID']:
        if col in df.columns:
            return col
    return None


def load_and_filter(input_path, smiles_col=None, sa_max=6.0,
                    max_atoms=150, max_rings=10):
    """Load CSV and apply SMILES/atom count/ring count/SA filters.

    Input:
        input_path: path to input CSV.
        smiles_col: SMILES column name (auto-detected if None).
        sa_max: max SA score threshold.
        max_atoms: max total atom count (default 150).
        max_rings: max ring count (default 10).
    Output:
        tuple of (filtered DataFrame, SMILES column name).
    """
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if max_atoms <= 0:
        raise ValueError(f"Invalid max_atoms: {max_atoms}")
    if max_rings < 0:
        raise ValueError(f"Invalid max_rings: {max_rings}")
    if sa_max <= 0:
        raise ValueError(f"Invalid SA max: {sa_max}")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    n_init = len(df)
    print(f"  {n_init:,} molecules")

    if smiles_col is None or smiles_col not in df.columns:
        smiles_col = detect_smiles_column(df)
        if smiles_col is None:
            raise ValueError("SMILES column not found")
    print(f"  SMILES column: {smiles_col}")

    print("Filtering valid SMILES...")
    valid_mask = df[smiles_col].apply(
        lambda x: pd.notna(x) and x != 'n.a.' and Chem.MolFromSmiles(str(x)) is not None
    )
    df = df[valid_mask].copy()
    print(f"  {n_init:,} -> {len(df):,}")

    print(f"Filtering atom count (<= {max_atoms})...")
    df['atom_count'] = df[smiles_col].apply(
        lambda x: Chem.MolFromSmiles(x).GetNumAtoms()
    )
    n_before = len(df)
    df = df[df['atom_count'] <= max_atoms].copy()
    print(f"  {n_before:,} -> {len(df):,}")

    print(f"Filtering ring count (<= {max_rings})...")
    df['ring_count'] = df[smiles_col].apply(
        lambda x: Descriptors.RingCount(Chem.MolFromSmiles(x))
    )
    n_before = len(df)
    df = df[df['ring_count'] <= max_rings].copy()
    print(f"  {n_before:,} -> {len(df):,}")

    print(f"Calculating SA score (filtering <= {sa_max})...")
    df['sa_score(RDKit)'] = apply_with_progress(df[smiles_col].tolist(), calc_sa, "SA")
    n_before = len(df)
    df = df[df['sa_score(RDKit)'].notna() & (df['sa_score(RDKit)'] <= sa_max)].copy()
    print(f"  {n_before:,} -> {len(df):,}")

    print("Calculating QED...")
    df['qed(RDKit)'] = apply_with_progress(df[smiles_col].tolist(), calc_qed, "QED")

    print("Calculating NPL score...")
    if NP_MODEL:
        df['npl_score(RDKit)'] = apply_with_progress(df[smiles_col].tolist(), calc_npl, "NPL")
    else:
        print("  NPL model not available, skipping")

    print(f"Filtering done: {n_init:,} -> {len(df):,} ({100*len(df)/n_init:.1f}%)")
    return df, smiles_col


def kmedoids_subset(df, smiles_col, target_size=100000, seed=42):
    """Select diverse subset via K-medoids clustering.

    Input:
        df: filtered DataFrame with SMILES and computed properties.
        smiles_col: name of SMILES column.
        target_size: number of molecules to select.
        seed: random seed (default 42).
    Output:
        pd.DataFrame subset of selected molecules.
    """
    print(f"K-medoids clustering (n={target_size})...")

    if len(df) <= target_size:
        print(f"  Data ({len(df)}) <= target, returning all")
        return df

    print("  Computing fingerprints...")
    fps, valid_idx = [], []
    smiles_list = df[smiles_col].tolist()
    indices = df.index.tolist()
    iterator = (
        tqdm(zip(indices, smiles_list), total=len(smiles_list), desc="FP")
        if HAS_TQDM else zip(indices, smiles_list)
    )

    for idx, smi in iterator:
        fp = smiles_to_fp(smi)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)

    X_fp = np.array(fps)
    print(f"  {len(X_fp):,} fingerprints")

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
        print("  Properties: SA, QED, NPL (normalized)")
    else:
        X_props = np.column_stack([sa_norm, qed_norm])
        print("  Properties: SA, QED (normalized)")

    # Estimate memory usage
    chunk_sz = min(2000, len(X_fp))
    mem_mb = chunk_sz * target_size * 4 / 1024 / 1024
    print(f"  Chunked K-medoids (chunk={chunk_sz}, ~{mem_mb:.0f} MB/chunk)")
    print(f"  Clustering (k={target_size}, Tanimoto + Euclidean)...")
    medoid_indices = kmedoids_combined(X_fp, X_props, target_size,
                                       max_iter=100, seed=seed,
                                       chunk_size=chunk_sz)

    selected = [valid_idx[i] for i in medoid_indices]
    subset = df.loc[selected].copy()
    print(f"  Selected {len(subset):,} molecules")
    return subset


def extract_sdf_subset(sdf_path, output_path, identifiers):
    """Extract matching molecules from SDF by identifier.

    Input:
        sdf_path: path to source SDF file.
        output_path: path for output SDF file.
        identifiers: set of molecule IDs to extract.
    Output:
        int, number of molecules found and written.
    """
    print("Extracting SDF subset...")
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


def save_subset(df, output_path, smiles_col):
    """Save subset DataFrame to CSV and print summary stats.

    Input:
        df: subset DataFrame.
        output_path: CSV output path.
        smiles_col: SMILES column name.
    Output:
        str or None, detected ID column name.
    """
    out_df = df.copy()
    out_df.to_csv(output_path, index=False)
    print(f"Saved CSV: {output_path}")
    print(f"  {len(out_df):,} molecules, {len(out_df.columns)} columns")
    for col in ['sa_score(RDKit)', 'qed(RDKit)', 'npl_score(RDKit)']:
        if col in out_df.columns:
            v = out_df[col].dropna()
            print(f"  {col}: mean={v.mean():.3f}, std={v.std():.3f}")
    if 'superclass' in out_df.columns:
        dist = Counter(out_df['superclass'])
        print(f"  Superclass distribution ({len(dist)} classes):")
        for cls, count in dist.most_common(10):
            print(f"    {cls}: {count}")
    return detect_id_column(df)


def main():
    parser = argparse.ArgumentParser(description="Create NP subset (COCONUT/NPASS)")
    parser.add_argument("-i", "--input", required=True, help="Input CSV")
    parser.add_argument("--sdf", help="Input SDF (optional)")
    parser.add_argument("-o", "--output", required=True, help="Output prefix")
    parser.add_argument("-s", "--size", type=int, default=100000, help="Target size")
    parser.add_argument("--sa_max", type=float, default=6.0, help="Max SA score")
    parser.add_argument("--max_atoms", type=int, default=150, help="Max atom count")
    parser.add_argument("--max_rings", type=int, default=10, help="Max ring count")
    parser.add_argument("--smiles_col", help="SMILES column (auto-detect if not set)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--classify", action="store_true",
                        help="Add ClassyFire superclass labels")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, smiles_col = load_and_filter(
        args.input, args.smiles_col, args.sa_max, args.max_atoms, args.max_rings
    )
    subset = kmedoids_subset(df, smiles_col, args.size, args.seed)

    # ClassyFire superclass classification
    if args.classify:
        if HAS_CLASSYFIRE:
            print("Classifying superclass (ClassyFire API)...")
            cache_dir = str(out_path.parent)
            superclasses = classify_batch(
                subset[smiles_col].tolist(), cache_dir=cache_dir
            )
            subset['superclass'] = superclasses
        else:
            print("Warning: ClassyFire not available (install 'requests')")

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
