"""NP subset creator with SA filtering and K-means clustering in Tanimoto space."""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set
from collections import Counter
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

N_JOBS = 1

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

# NPClassifier classification (local server preferred)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "classification"))
try:
    from npclassifier import classify_batch
    HAS_NPCLASSIFIER = True
except ImportError:
    HAS_NPCLASSIFIER = False


def _torch_nearest(X, centers, data_chunk=4096, center_chunk=20000):
    """Assign each row of X to its nearest center (chunked for memory).

    Input:
        X: torch.Tensor (n, d).
        centers: torch.Tensor (k, d).
        data_chunk: rows per chunk.
        center_chunk: centers per chunk.
    Output:
        torch.LongTensor (n,) of cluster assignments.
    """
    n, k = X.shape[0], centers.shape[0]
    labels = torch.empty(n, dtype=torch.long, device=X.device)
    for i in range(0, n, data_chunk):
        xi = X[i:min(i + data_chunk, n)]
        m = len(xi)
        if k <= center_chunk:
            labels[i:i + m] = torch.cdist(xi, centers).argmin(1)
        else:
            best_d = torch.full((m,), float('inf'), device=X.device)
            best_k = torch.zeros(m, dtype=torch.long, device=X.device)
            for j in range(0, k, center_chunk):
                cj = centers[j:min(j + center_chunk, k)]
                d = torch.cdist(xi, cj)
                md, mi = d.min(1)
                upd = md < best_d
                best_d[upd] = md[upd]
                best_k[upd] = mi[upd] + j
            labels[i:i + m] = best_k
    return labels


def _kmeans_torch(X_np, n_clusters, batch_size, max_iter, seed):
    """Mini-batch K-means via torch (GPU/CPU).

    Input:
        X_np: np.ndarray (n, d).
        n_clusters: int.
        batch_size: int.
        max_iter: int.
        seed: int.
    Output:
        tuple of (np.ndarray labels, np.ndarray centers).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n, d = X_np.shape
    print(f"    K-means (k={n_clusters}, batch={batch_size}, "
          f"iter={max_iter}, device={device})...")

    X = torch.from_numpy(np.ascontiguousarray(X_np, dtype=np.float32)).to(device)
    gen = torch.Generator()
    gen.manual_seed(seed)
    init_idx = torch.randperm(n, generator=gen)[:n_clusters]
    centers = X[init_idx].clone()
    counts = torch.ones(n_clusters, device=device)

    for it in range(max_iter):
        bidx = torch.randint(0, n, (batch_size,), generator=gen)
        batch = X[bidx]
        assignments = _torch_nearest(batch, centers)

        sums = torch.zeros(n_clusters, d, device=device)
        cnts = torch.zeros(n_clusters, device=device)
        sums.index_add_(0, assignments, batch)
        cnts.index_add_(0, assignments,
                        torch.ones(batch_size, device=device))

        active = cnts > 0
        counts[active] += cnts[active]
        lr = (cnts[active] / counts[active]).unsqueeze(1)
        batch_means = sums[active] / cnts[active].unsqueeze(1)
        centers[active] += lr * (batch_means - centers[active])

        if (it + 1) % 10 == 0:
            print(f"      iter {it+1}/{max_iter}")

    print("    Assigning all points...")
    labels = _torch_nearest(X, centers).cpu().numpy()
    return labels, centers.cpu().numpy()


def _kmeans_sklearn(X_np, n_clusters, batch_size, max_iter, seed):
    """Mini-batch K-means via sklearn (fallback when torch unavailable).

    Input:
        X_np: np.ndarray (n, d).
        n_clusters: int.
        batch_size: int.
        max_iter: int.
        seed: int.
    Output:
        tuple of (np.ndarray labels, np.ndarray centers).
    """
    from sklearn.cluster import MiniBatchKMeans
    print(f"    MiniBatchKMeans (k={n_clusters}, batch={batch_size})...")
    X_f32 = np.asarray(X_np, dtype=np.float32)
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, batch_size=batch_size,
        max_iter=max_iter, random_state=seed, n_init=1
    )
    labels = kmeans.fit_predict(X_f32)
    return labels, kmeans.cluster_centers_


def kmeans_select(X_combined, target_size, seed=42, oversample=1.2,
                  batch_size=4096, max_iter=1000):
    """K-means subset selection: oversample clusters, prune, centroid select.

    1. Cluster into target_size * oversample groups.
    2. Drop smallest clusters until target_size clusters remain.
    3. Select closest-to-centroid molecule from each surviving cluster.

    Input:
        X_combined: np.ndarray (n, d), feature matrix (FP + properties).
        target_size: int, desired number of molecules.
        seed: int, random seed (default 42).
        oversample: float, cluster multiplier (default 1.2).
        batch_size: int, mini-batch size (default 4096).
        max_iter: int, maximum iterations (default 1000).
    Output:
        np.ndarray of selected indices.
    """
    n = len(X_combined)
    n_clusters = min(int(target_size * oversample), n)

    if HAS_TORCH:
        labels, centers = _kmeans_torch(X_combined, n_clusters, batch_size,
                                        max_iter, seed)
    else:
        print("    torch not available, falling back to sklearn")
        labels, centers = _kmeans_sklearn(X_combined, n_clusters, batch_size,
                                          max_iter, seed)

    # Drop smallest clusters until target_size clusters remain
    cluster_ids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)
    n_drop = max(0, len(cluster_ids) - target_size)
    drop_set = set(cluster_ids[order[:n_drop]])
    keep_ids = [cid for cid in cluster_ids if cid not in drop_set]

    print(f"    Kept {len(keep_ids)} clusters, dropped {n_drop} smallest")

    # Select closest-to-centroid from each surviving cluster
    print("    Selecting closest-to-centroid representatives...")
    X_f32 = np.asarray(X_combined, dtype=np.float32)
    selected = []
    for cid in keep_ids:
        members = np.where(labels == cid)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(X_f32[members] - centers[cid], axis=1)
        selected.append(members[np.argmin(dists)])

    selected = np.array(selected, dtype=np.int64)
    selected.sort()
    print(f"    Selected {len(selected):,} representatives")
    return selected


def compute_properties(smiles):
    """Compute all molecular properties from SMILES in a single pass.

    Parses MolFromSmiles once and computes atom count, ring count,
    SA, QED, and NPL.

    Input:
        smiles: SMILES string.
    Output:
        dict with keys: valid, atom_count, ring_count, sa, qed, npl.
    """
    result = {'valid': False, 'atom_count': None, 'ring_count': None,
              'sa': None, 'qed': None, 'npl': None}
    if not isinstance(smiles, str) or smiles == 'n.a.' or not smiles:
        return result
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return result
    result['valid'] = True
    result['atom_count'] = mol.GetNumAtoms()
    result['ring_count'] = Descriptors.RingCount(mol)
    try:
        result['sa'] = sascorer.calculateScore(mol)
    except (ValueError, RuntimeError):
        pass
    try:
        result['qed'] = QED.qed(mol)
    except (ValueError, RuntimeError):
        pass
    if NP_MODEL is not None:
        try:
            result['npl'] = npscorer.scoreMol(mol, NP_MODEL)
        except (ValueError, RuntimeError):
            pass
    return result


def smiles_to_fp(smiles, radius=2, n_bits=1024):
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
    except (ValueError, RuntimeError):
        return None


def parallel_map(func, data, desc=""):
    """Apply func in parallel with optional progress bar.

    Input:
        func: callable applied to each element.
        data: list of values.
        desc: tqdm description.
    Output:
        list of results.
    """
    n = N_JOBS if N_JOBS > 0 else cpu_count()
    if n > 1:
        with Pool(n) as pool:
            if HAS_TQDM:
                return list(tqdm(pool.imap(func, data), total=len(data), desc=desc))
            return pool.map(func, data)
    if HAS_TQDM:
        return [func(x) for x in tqdm(data, desc=desc)]
    return [func(x) for x in data]


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

    # Single-pass parallel: compute all properties per molecule
    print("Computing properties (single pass, parallel)...")
    smiles_list = df[smiles_col].tolist()
    props = parallel_map(compute_properties, smiles_list, desc="Properties")

    df['_valid'] = [p['valid'] for p in props]
    df['atom_count'] = [p['atom_count'] for p in props]
    df['ring_count'] = [p['ring_count'] for p in props]
    df['sa_score(RDKit)'] = [p['sa'] for p in props]
    df['qed(RDKit)'] = [p['qed'] for p in props]
    if NP_MODEL:
        df['npl_score(RDKit)'] = [p['npl'] for p in props]

    # Filter
    print("Filtering...")
    df = df[df['_valid']].copy()
    print(f"  Valid SMILES: {n_init:,} -> {len(df):,}")

    if len(df) == 0:
        raise ValueError(f"No valid molecules found in {input_path}. "
                         "Check that the file has data (not just headers).")

    n_before = len(df)
    df = df[df['atom_count'] <= max_atoms].copy()
    print(f"  Atom count (<= {max_atoms}): {n_before:,} -> {len(df):,}")

    n_before = len(df)
    df = df[df['ring_count'] <= max_rings].copy()
    print(f"  Ring count (<= {max_rings}): {n_before:,} -> {len(df):,}")

    n_before = len(df)
    df = df[df['sa_score(RDKit)'].notna() & (df['sa_score(RDKit)'] <= sa_max)].copy()
    print(f"  SA (<= {sa_max}): {n_before:,} -> {len(df):,}")

    df.drop(columns=['_valid'], inplace=True)

    if not NP_MODEL:
        print("  NPL model not available, skipping")

    print(f"Filtering done: {n_init:,} -> {len(df):,} ({100*len(df)/n_init:.1f}%)")
    return df, smiles_col


def kmeans_subset(df, smiles_col, target_size=10000, seed=42, fp_weight=0.7,
                  fp_dim=3):
    """Select diverse subset via K-means in Tanimoto fingerprint space.

    Binary Morgan fingerprints are reduced to fp_dim dimensions via PCA,
    combined with molecular properties, then clustered with
    mini-batch K-means (closest-to-centroid selection).

    Input:
        df: filtered DataFrame with SMILES and computed properties.
        smiles_col: name of SMILES column.
        target_size: number of molecules to select.
        seed: random seed (default 42).
        fp_weight: float, weight for fingerprint features vs properties.
        fp_dim: int, PCA output dimensions for fingerprint space (default 3).
    Output:
        pd.DataFrame subset of selected molecules.
    """
    from sklearn.decomposition import PCA

    print(f"K-means subset selection (n={target_size})...")

    if len(df) <= target_size:
        print(f"  Data ({len(df)}) <= target, returning all")
        return df

    print("  Computing fingerprints...")
    smiles_list = df[smiles_col].tolist()
    indices = df.index.tolist()

    fp_results = parallel_map(smiles_to_fp, smiles_list, desc="FP")

    fps, valid_idx = [], []
    for idx, fp in zip(indices, fp_results):
        if fp is not None:
            fps.append(fp)
            valid_idx.append(idx)

    X_fp = np.array(fps, dtype=np.float32)
    print(f"  {len(X_fp):,} fingerprints ({X_fp.shape[1]} bits)")

    # Reduce FP to Tanimoto space via PCA
    n_components = min(fp_dim, X_fp.shape[1], len(X_fp))
    pca = PCA(n_components=n_components, random_state=seed)
    X_fp_reduced = pca.fit_transform(X_fp)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  Tanimoto space: {X_fp.shape[1]} -> {n_components} dims "
          f"({var_explained:.1%} variance)")

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

    # Normalize PCA dimensions to [0,1]
    X_fp_norm = np.column_stack([normalize(X_fp_reduced[:, i])
                                 for i in range(X_fp_reduced.shape[1])])
    print(f"  FP dimensions normalized to [0,1]")

    sa_norm = normalize(sa_vals)
    qed_norm = normalize(qed_vals)

    if use_npl:
        npl_norm = normalize(npl_vals)
        X_props = np.column_stack([sa_norm, qed_norm, npl_norm])
        print("  Properties: SA, QED, NPL (normalized)")
    else:
        X_props = np.column_stack([sa_norm, qed_norm])
        print("  Properties: SA, QED (normalized)")

    # Combine normalized FP and properties with weighting
    prop_weight = 1.0 - fp_weight
    X_combined = np.hstack([X_fp_norm * fp_weight, X_props * prop_weight])
    print(f"  Feature dim: {X_combined.shape[1]} "
          f"(FP={X_fp_reduced.shape[1]}, props={X_props.shape[1]})")

    selected_idx = kmeans_select(X_combined, target_size, seed=seed)

    selected = [valid_idx[i] for i in selected_idx]
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
    parser.add_argument("-s", "--size", type=int, default=5000, help="Target size (default: 5000)")
    parser.add_argument("--sa_max", type=float, default=6.0, help="Max SA score")
    parser.add_argument("--max_atoms", type=int, default=150, help="Max atom count")
    parser.add_argument("--max_rings", type=int, default=10, help="Max ring count")
    parser.add_argument("--smiles_col", help="SMILES column (auto-detect if not set)")
    parser.add_argument("--fp_dim", type=int, default=3, help="PCA dims for Tanimoto space")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Parallel workers (-1 = all cores, 1 = single)")
    parser.add_argument("--classify", action="store_true",
                        help="Run NPClassifier superclass classification (off by default)")
    parser.add_argument("--np_root",
                        default=os.environ.get("NP_CLASSIFIER_ROOT"),
                        help="Path to NP-Classifier repo clone (or set NP_CLASSIFIER_ROOT env)")
    args = parser.parse_args()

    global N_JOBS
    N_JOBS = args.n_jobs if args.n_jobs > 0 else cpu_count()
    print(f"Workers: {N_JOBS}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df, smiles_col = load_and_filter(
        args.input, args.smiles_col, args.sa_max, args.max_atoms, args.max_rings
    )

    subset = kmeans_subset(df, smiles_col, args.size, args.seed, fp_dim=args.fp_dim)

    # NPClassifier superclass classification (only with --classify flag)
    if args.classify:
        if not HAS_NPCLASSIFIER:
            print("Warning: NPClassifier module not available, skipping classification")
        elif not args.np_root:
            print("Warning: NP_CLASSIFIER_ROOT not set, skipping classification. "
                  "Set via --np_root or export NP_CLASSIFIER_ROOT=<path>")
        else:
            print("Classifying superclass (NPClassifier local model)...")
            cache_dir = str(out_path.parent)
            superclasses = classify_batch(
                subset[smiles_col].tolist(),
                cache_dir=cache_dir,
                repo_root=args.np_root,
                level="superclass",
            )
            subset['superclass'] = superclasses

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
