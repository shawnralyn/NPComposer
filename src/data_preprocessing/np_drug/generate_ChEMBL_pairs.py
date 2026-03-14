"""Find COCONUT → ChEMBL pairs for NP-to-drug generative model training.

For each COCONUT molecule, queries a FAISS binary index of ChEMBL Morgan
fingerprints to retrieve K nearest neighbors by Hamming distance, then
re-scores with exact Tanimoto similarity and filters by:
  - Tanimoto similarity >= threshold (shared structural core)
  - ChEMBL QED > COCONUT QED (drug-likeness improvement)
  - ChEMBL SA  < COCONUT SA  (synthesizability improvement, if SA available)
  - ChEMBL passes Lipinski Ro5 (MW<=500, HBD<=5, HBA<=10, LogP<=5)

Outputs a CSV with exact property values (not binned) for both members of
each pair, ready for downstream binning and training sequence construction.

Usage:
    python scripts/generate_chembl_pairs.py \\
        --coconut-fp    data/coconut_fp.npy \\
        --coconut-meta  data/coconut_fp_meta.csv \\
        --chembl-fp     data/chembl_fp.npy \\
        --chembl-meta   data/chembl_fp_meta.csv \\
        --faiss-index   data/chembl.faiss \\
        --output        data/chembl_pairs.csv

    # Stricter similarity, more drug-like improvement required
    python scripts/generate_chembl_pairs.py \\
        --coconut-fp    data/coconut_fp.npy \\
        --coconut-meta  data/coconut_fp_meta.csv \\
        --chembl-fp     data/chembl_fp.npy \\
        --chembl-meta   data/chembl_fp_meta.csv \\
        --faiss-index   data/chembl.faiss \\
        --output        data/chembl_pairs.csv \\
        --tanimoto-threshold 0.5 \\
        --min-qed-improvement 0.15 \\
        --k 100
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss is required. Install with:\n"
        "  conda install -c pytorch faiss-cpu"
    ) from e


# ---------------------------------------------------------------------------
# Tanimoto from packed uint8 fingerprints
# ---------------------------------------------------------------------------

def tanimoto_batch(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Compute Tanimoto similarity between one query and multiple candidates.

    Operates on packed uint8 fingerprints without unpacking, using bitwise
    operations on bytes. Faster than RDKit BulkTanimotoSimilarity for large
    candidate sets.

    Args:
        query: Packed fingerprint of shape (n_bytes,).
        candidates: Packed fingerprints of shape (k, n_bytes).

    Returns:
        Array of Tanimoto similarities of shape (k,).
    """
    # Bitwise AND/OR on packed bytes, then popcount via unpackbits
    and_bits = np.unpackbits(query[None] & candidates, axis=1).sum(axis=1)
    or_bits  = np.unpackbits(query[None] | candidates, axis=1).sum(axis=1)
    return np.where(or_bits > 0, and_bits / or_bits, 0.0)


# ---------------------------------------------------------------------------
# Lipinski filter
# ---------------------------------------------------------------------------

def passes_lipinski(
    row: pd.Series,
    mw_col: str,
    hbd_col: str,
    hba_col: str,
    logp_col: str,
) -> bool:
    """Return True if a molecule passes Lipinski's Rule of Five."""
    try:
        return (
            float(row[mw_col])  <= 500 and
            float(row[hbd_col]) <= 5   and
            float(row[hba_col]) <= 10  and
            float(row[logp_col]) <= 5
        )
    except (ValueError, KeyError):
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Inputs
    parser.add_argument("--coconut-fp",   required=True, help="COCONUT packed fingerprint array (.npy)")
    parser.add_argument("--coconut-meta", required=True, help="COCONUT metadata CSV aligned with fingerprint rows")
    parser.add_argument("--chembl-fp",    required=True, help="ChEMBL packed fingerprint array (.npy)")
    parser.add_argument("--chembl-meta",  required=True, help="ChEMBL metadata CSV aligned with fingerprint rows")
    parser.add_argument("--faiss-index",  required=True, help="FAISS binary index file (.faiss)")
    parser.add_argument("-o", "--output", required=True, help="Output pairs CSV")

    # Filtering thresholds
    parser.add_argument("--tanimoto-threshold",   type=float, default=0.4,
                        help="Minimum Tanimoto similarity (default: 0.4)")
    parser.add_argument("--min-qed-improvement",  type=float, default=0.1,
                        help="Minimum QED(ChEMBL) - QED(COCONUT) (default: 0.1)")
    parser.add_argument("--min-sa-improvement",   type=float, default=0.5,
                        help="Minimum SA(COCONUT) - SA(ChEMBL) (default: 0.5). "
                             "Skipped if SA columns are absent.")
    parser.add_argument("--k",            type=int,   default=50,
                        help="FAISS nearest neighbors to retrieve per molecule (default: 50)")
    parser.add_argument("--batch-size",   type=int,   default=4096,
                        help="COCONUT molecules to query per FAISS batch (default: 4096)")
    parser.add_argument("--max-pairs-per-molecule", type=int, default=3,
                        help="Max ChEMBL matches to keep per COCONUT molecule (default: 3)")

    # Column names — COCONUT (matches clean_coconut_npdrug.py output)
    parser.add_argument("--coconut-smiles-col", default="canonical_smiles")
    parser.add_argument("--coconut-qed-col",    default="qed")
    parser.add_argument("--coconut-sa-col",     default="sa_score")

    # Column names — ChEMBL (matches clean_ChEMBL.py output — same standard names)
    parser.add_argument("--chembl-smiles-col",  default="canonical_smiles")
    parser.add_argument("--chembl-qed-col",     default="qed")
    parser.add_argument("--chembl-sa-col",      default="sa_score")
    parser.add_argument("--chembl-mw-col",      default="molecular_weight")
    parser.add_argument("--chembl-hbd-col",     default="hbd")
    parser.add_argument("--chembl-hba-col",     default="hba")
    parser.add_argument("--chembl-logp-col",    default="alogp")

    args = parser.parse_args()

    # ---- load data ----
    print("Loading fingerprints and metadata...")
    coconut_fp   = np.load(args.coconut_fp)
    chembl_fp    = np.load(args.chembl_fp)
    coconut_meta = pd.read_csv(args.coconut_meta, low_memory=False)
    chembl_meta  = pd.read_csv(args.chembl_meta,  low_memory=False)

    assert len(coconut_fp)   == len(coconut_meta), "COCONUT fp/meta row count mismatch"
    assert len(chembl_fp)    == len(chembl_meta),  "ChEMBL fp/meta row count mismatch"
    assert coconut_fp.shape[1] == chembl_fp.shape[1], "Fingerprint size mismatch"

    n_bits = coconut_fp.shape[1] * 8
    print(f"  COCONUT:  {len(coconut_fp):,} molecules")
    print(f"  ChEMBL:   {len(chembl_fp):,} molecules")
    print(f"  FP size:  {n_bits} bits")

    # ---- SA column availability ----
    use_sa = (
        args.coconut_sa_col in coconut_meta.columns and
        args.chembl_sa_col  in chembl_meta.columns
    )
    if not use_sa:
        print("  SA score columns not found in both datasets — skipping SA filter")

    # ---- load FAISS index ----
    print(f"\nLoading FAISS index from {args.faiss_index}...")
    index = faiss.read_index_binary(args.faiss_index)
    print(f"  {index.ntotal:,} vectors indexed")

    # ---- pair generation ----
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_coconut   = len(coconut_fp)
    total_pairs = 0
    first_write = True
    t0 = time.time()

    print(f"\nGenerating pairs (k={args.k}, tanimoto>={args.tanimoto_threshold}, "
          f"min_qed_improvement={args.min_qed_improvement})...")

    for batch_start in tqdm(range(0, n_coconut, args.batch_size), desc="Batches"):
        batch_end  = min(batch_start + args.batch_size, n_coconut)
        batch_fp   = coconut_fp[batch_start:batch_end]
        batch_meta = coconut_meta.iloc[batch_start:batch_end].reset_index(drop=True)

        # FAISS batch search — returns Hamming distances and ChEMBL indices
        _, chembl_indices = index.search(batch_fp, args.k)
        # chembl_indices: (batch_size, k)

        batch_pairs = []

        for i in range(len(batch_fp)):
            coconut_row = batch_meta.iloc[i]
            coconut_qed = float(coconut_row[args.coconut_qed_col])
            coconut_sa  = float(coconut_row[args.coconut_sa_col]) if use_sa else None

            # Candidate ChEMBL indices for this COCONUT molecule
            cand_idx = chembl_indices[i]
            cand_idx = cand_idx[cand_idx >= 0]  # FAISS returns -1 for empty slots
            if len(cand_idx) == 0:
                continue

            # Exact Tanimoto re-scoring
            cand_fp   = chembl_fp[cand_idx]
            tanimotos = tanimoto_batch(batch_fp[i], cand_fp)

            passing = []
            for j, (cidx, tanimoto) in enumerate(zip(cand_idx, tanimotos)):
                if tanimoto < args.tanimoto_threshold:
                    continue

                chembl_row = chembl_meta.iloc[cidx]

                # QED must improve
                try:
                    chembl_qed = float(chembl_row[args.chembl_qed_col])
                except (ValueError, KeyError):
                    continue
                if chembl_qed <= coconut_qed + args.min_qed_improvement:
                    continue

                # SA must improve (if available)
                if use_sa:
                    try:
                        chembl_sa = float(chembl_row[args.chembl_sa_col])
                    except (ValueError, KeyError):
                        continue
                    if chembl_sa >= coconut_sa - args.min_sa_improvement:
                        continue

                # Lipinski Ro5
                if not passes_lipinski(
                    chembl_row,
                    args.chembl_mw_col,
                    args.chembl_hbd_col,
                    args.chembl_hba_col,
                    args.chembl_logp_col,
                ):
                    continue

                passing.append((tanimoto, cidx, chembl_row))

            # Keep top-N by Tanimoto
            passing.sort(key=lambda x: x[0], reverse=True)
            for tanimoto, cidx, chembl_row in passing[:args.max_pairs_per_molecule]:
                pair = {
                    "coconut_smiles": coconut_row[args.coconut_smiles_col],
                    "chembl_smiles":  chembl_row[args.chembl_smiles_col],
                    "tanimoto":       round(tanimoto, 4),
                    "coconut_qed":    coconut_qed,
                    "chembl_qed":     float(chembl_row[args.chembl_qed_col]),
                }
                # Attach remaining COCONUT properties
                for col in coconut_meta.columns:
                    if col != args.coconut_smiles_col:
                        pair[f"coconut_{col}"] = coconut_row[col]
                # Attach remaining ChEMBL properties
                for col in chembl_meta.columns:
                    if col != args.chembl_smiles_col:
                        pair[f"chembl_{col}"] = chembl_row[col]

                batch_pairs.append(pair)

        if batch_pairs:
            df_batch = pd.DataFrame(batch_pairs)
            df_batch.to_csv(
                out_path,
                mode='w' if first_write else 'a',
                header=first_write,
                index=False,
            )
            first_write = False
            total_pairs += len(batch_pairs)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Total pairs: {total_pairs:,}")
    print(f"Pairs per COCONUT molecule: {total_pairs / n_coconut:.2f} avg")
    print(f"Output saved to {out_path}")


if __name__ == "__main__":
    main()
