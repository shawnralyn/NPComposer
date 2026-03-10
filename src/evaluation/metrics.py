"""Evaluation metrics for generated molecules."""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from rdkit.Contrib.NP_Score import npscorer
    NP_MODEL = npscorer.readNPModel()
except (ImportError, FileNotFoundError, OSError):
    NP_MODEL = None

# NPClassifier — pure local inference (no HTTP)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))
try:
    from npclassifier import classify_batch_full
    HAS_NPCLASSIFIER = True
except ImportError:
    HAS_NPCLASSIFIER = False


def _smiles_to_fp(smiles, radius=2, n_bits=2048):
    """Convert SMILES to Morgan fingerprint.

    Input:
        smiles (str): SMILES string.
        radius (int): Morgan radius.
        n_bits (int): fingerprint length.
    Output:
        RDKit ExplicitBitVect or None.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_internal_diversity(smiles_list: List[str], sample_size: int = 1000) -> Dict:
    """Compute pairwise Tanimoto diversity among generated molecules.

    Randomly samples pairs if the list is large to keep runtime manageable.

    Input:
        smiles_list (List[str]): valid SMILES strings.
        sample_size (int): max molecules to use for pairwise comparison.
    Output:
        dict with mean, std, min, max of pairwise Tanimoto distances (1 - similarity).
    """
    fps = [_smiles_to_fp(s) for s in smiles_list]
    fps = [fp for fp in fps if fp is not None]

    if len(fps) < 2:
        return {"mean": None, "std": None, "min": None, "max": None, "n_pairs": 0}

    # subsample if too many
    if len(fps) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(fps), size=sample_size, replace=False)
        fps = [fps[i] for i in idx]

    distances = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            distances.append(1.0 - sim)

    return {
        "mean": round(np.mean(distances), 4),
        "std": round(np.std(distances), 4),
        "min": round(np.min(distances), 4),
        "max": round(np.max(distances), 4),
        "n_pairs": len(distances),
    }


def compute_uniqueness(generated_smiles: List[str], training_smiles: List[str]) -> Dict:
    """Compute uniqueness: fraction of generated molecules not in training set.

    Compares canonical SMILES strings. A molecule is "unique" if it does not
    appear in the training data (i.e., it is a genuinely new structure).

    Input:
        generated_smiles (List[str]): valid generated SMILES.
        training_smiles (List[str]): training set SMILES.
    Output:
        dict with n_unique, n_total, uniqueness ratio.
    """
    def canon(s):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    train_canonical = set()
    for s in training_smiles:
        c = canon(s)
        if c:
            train_canonical.add(c)

    n_unique = 0
    n_total = 0
    for s in generated_smiles:
        c = canon(s)
        if c is None:
            continue
        n_total += 1
        if c not in train_canonical:
            n_unique += 1

    return {
        "n_unique": n_unique,
        "n_total": n_total,
        "uniqueness": round(n_unique / n_total, 4) if n_total > 0 else 0,
    }


def _fps_to_numpy(fps, nbits=2048):
    """Convert list of RDKit fingerprints to numpy bit array (n_mols x nbits)."""
    arr = np.zeros((len(fps), nbits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        on_bits = fp.GetOnBits()
        for b in on_bits:
            arr[i, b] = 1
    return arr


def compute_novelty(generated_smiles: List[str],
                    training_smiles: List[str],
                    threshold: float = 0.4,
                    batch_size: int = 100) -> Dict:
    """Compute novelty following f-RAG (NeurIPS 2024) definition.

    A generated molecule is "novel" if its nearest-neighbor Tanimoto
    similarity to the training set is below the threshold (default 0.4).
    Uses Morgan fingerprint radius=2, 1024 bits (same as f-RAG).

    Novelty = fraction of generated molecules where NN_sim < threshold.

    Uses numpy vectorized Tanimoto for efficiency:
        Tanimoto(A, B) = dot(A,B) / (|A| + |B| - dot(A,B))

    Reference:
        f-RAG (NeurIPS 2024): NN Tanimoto < 0.4 → novel.

    Input:
        generated_smiles (List[str]): valid generated SMILES.
        training_smiles (List[str]): training set SMILES (K-means subset).
        threshold (float): similarity threshold (default 0.4).
        batch_size (int): process generated molecules in batches to save memory.
    Output:
        dict with n_novel, n_total, novelty (ratio), nn_sim_mean/std,
        and threshold used.
    """
    # f-RAG uses Morgan FP radius=2, 1024 bits
    FP_BITS = 1024
    gen_fps = [_smiles_to_fp(s, radius=2, n_bits=FP_BITS) for s in generated_smiles]
    gen_fps = [fp for fp in gen_fps if fp is not None]

    train_fps = [_smiles_to_fp(s, radius=2, n_bits=FP_BITS) for s in training_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]

    if not gen_fps or not train_fps:
        return {"n_novel": 0, "n_total": 0, "novelty": 0,
                "nn_sim_mean": None, "nn_sim_std": None, "threshold": threshold}

    print(f"  Computing novelty (f-RAG): {len(gen_fps)} generated vs "
          f"{len(train_fps)} training, threshold={threshold}")

    # Convert training set to numpy (done once, reused across batches)
    train_np = _fps_to_numpy(train_fps, nbits=FP_BITS).astype(np.float32)
    train_bits = train_np.sum(axis=1)

    nn_sims = []
    for i in range(0, len(gen_fps), batch_size):
        batch_fps = gen_fps[i:i + batch_size]
        gen_np = _fps_to_numpy(batch_fps, nbits=FP_BITS).astype(np.float32)
        gen_bits = gen_np.sum(axis=1)

        intersection = gen_np @ train_np.T
        union = gen_bits[:, None] + train_bits[None, :] - intersection
        sim_matrix = np.divide(intersection, union,
                               out=np.zeros_like(intersection), where=union > 0)
        max_sims = sim_matrix.max(axis=1)
        nn_sims.extend(max_sims.tolist())

    nn_sims = np.array(nn_sims)
    n_novel = int((nn_sims < threshold).sum())
    n_total = len(nn_sims)

    return {
        "n_novel": n_novel,
        "n_total": n_total,
        "novelty": round(n_novel / n_total, 4) if n_total > 0 else 0,
        "nn_sim_mean": round(float(nn_sims.mean()), 4),
        "nn_sim_std": round(float(nn_sims.std()), 4),
        "threshold": threshold,
    }


def evaluate(smiles_list: List[str],
             np_repo_root: str = None,
             keep_np_per_mol: bool = False,
             training_file: str = None,
             novelty_file: str = None) -> Dict:
    """Compute validity, SA, QED, NP-likeness, diversity, uniqueness, novelty.

    Input:
        smiles_list: list of SMILES strings.
        np_repo_root: path to NP-Classifier repo clone (or set NP_CLASSIFIER_ROOT env).
        keep_np_per_mol: if True, store per-molecule NPClassifier assignments.
        training_file: path to full training CSV for uniqueness (exact match).
        novelty_file: path to K-means subset CSV for novelty (Tanimoto NN < 0.4).
                      If not given, falls back to training_file.
    Output:
        dict with keys: n_total, n_valid, validity, sa_score, qed, np_score,
        internal_diversity, uniqueness, novelty,
        and optionally npclassifier distributions.
    """
    sa, qed_scores, np_scores = [], [], []
    valid = 0
    valid_smiles = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        valid += 1
        valid_smiles.append(s)
        sa.append(sascorer.calculateScore(mol))
        qed_scores.append(QED.qed(mol))
        if NP_MODEL:
            try:
                np_scores.append(npscorer.scoreMol(mol, NP_MODEL))
            except (ValueError, ZeroDivisionError):
                pass

    n = len(smiles_list)

    results = {
        "n_total": n,
        "n_valid": valid,
        "validity": round(valid / n, 4) if n else 0,
        "sa_score": {
            "mean": round(np.mean(sa), 3) if sa else None,
            "std": round(np.std(sa), 3) if sa else None,
            "min": round(min(sa), 3) if sa else None,
            "max": round(max(sa), 3) if sa else None,
        },
        "qed": {
            "mean": round(np.mean(qed_scores), 3) if qed_scores else None,
            "std": round(np.std(qed_scores), 3) if qed_scores else None,
        },
        "np_score": {
            "mean": round(np.mean(np_scores), 3) if np_scores else None,
            "std": round(np.std(np_scores), 3) if np_scores else None,
        } if NP_MODEL else None,
    }

    # Internal diversity (pairwise Tanimoto distance among generated molecules)
    if valid_smiles:
        print("Computing internal diversity...")
        results["internal_diversity"] = compute_internal_diversity(valid_smiles)
        results["non_duplicate"] = round(len(set(valid_smiles)) / len(valid_smiles), 4) if valid_smiles else 0

    def _load_smiles(filepath, label="data"):
        """Load SMILES from a CSV file."""
        import pandas as pd
        print(f"Loading {label} from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        for col in ['canonical_smiles', 'SMILES', 'smiles']:
            if col in df.columns:
                smiles = df[col].dropna().tolist()
                print(f"  {label}: {len(smiles):,} molecules")
                return smiles
        print(f"Warning: no SMILES column found in {filepath}")
        return None

    # Uniqueness: exact match against full training set
    if training_file and valid_smiles:
        train_smiles = _load_smiles(training_file, "training data (full)")
        if train_smiles:
            print("Computing uniqueness (Geo2Seq: not in training set)...")
            results["uniqueness"] = compute_uniqueness(valid_smiles, train_smiles)

    # Novelty: NN Tanimoto < 0.4 against K-means subset
    nov_file = novelty_file or training_file
    if nov_file and valid_smiles:
        nov_smiles = _load_smiles(nov_file, "novelty reference (K-means subset)") if nov_file != training_file else train_smiles
        if nov_smiles:
            print("Computing novelty (f-RAG: NN Tanimoto < 0.4)...")
            results["novelty"] = compute_novelty(valid_smiles, nov_smiles)

    # NPClassifier (pure local inference) classification
    if valid_smiles:
        if HAS_NPCLASSIFIER:
            print("Classifying with NPClassifier (local model)...")
            np_results = classify_batch_full(
                valid_smiles, cache_dir=".", repo_root=np_repo_root
            )

            if np_results:
                results["npclassifier"] = {
                    "pathway_distribution": np_results.get("pathway_distribution", {}),
                    "superclass_distribution": np_results.get("superclass_distribution", {}),
                    "class_distribution": np_results.get("class_distribution", {}),
                }
                # Backward-compatible superclass_distribution at top level
                sc_dist = np_results.get("superclass_distribution", {})
                results["superclass_distribution"] = sc_dist
                results["n_superclasses"] = len([k for k in sc_dist if k != "Unknown"])

                if keep_np_per_mol:
                    results["npclassifier"]["per_molecule"] = np_results.get("per_molecule", [])
        else:
            print("Warning: NPClassifier module not available (check src/classification/npclassifier.py)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("-i", "--input", required=True, help="SMILES file (one per line)")
    parser.add_argument("-o", "--output", help="Output JSON")
    parser.add_argument("--np_root",
                        default=os.environ.get("NP_CLASSIFIER_ROOT"),
                        help="Path to NP-Classifier repo clone (or set NP_CLASSIFIER_ROOT env)")
    parser.add_argument("--keep_np_per_mol", action="store_true",
                        help="Store per-molecule NPClassifier assignments in output JSON")
    parser.add_argument("--training", default=None,
                        help="Full training CSV for uniqueness (exact match)")
    parser.add_argument("--novelty_ref", default=None,
                        help="K-means subset CSV for novelty (Tanimoto NN). Falls back to --training if not given.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(args.input) as f:
        smiles = [line.strip() for line in f if line.strip()]

    if not smiles:
        print("Error: No SMILES found in input file", file=sys.stderr)
        sys.exit(1)

    print(f"Evaluating {len(smiles)} molecules...")
    results = evaluate(
        smiles,
        np_repo_root=args.np_root,
        keep_np_per_mol=args.keep_np_per_mol,
        training_file=args.training,
        novelty_file=args.novelty_ref,
    )

    print(f"\nResults:")
    print(f"  Valid: {results['n_valid']}/{results['n_total']} ({results['validity']*100:.1f}%)")
    print(f"  SA: {results['sa_score']['mean']} +/- {results['sa_score']['std']}")
    print(f"  QED: {results['qed']['mean']} +/- {results['qed']['std']}")
    if results['np_score']:
        print(f"  NP: {results['np_score']['mean']} +/- {results['np_score']['std']}")
    if 'non_duplicate' in results:
        print(f"  Non-duplicate: {results['non_duplicate']*100:.1f}%")
    if 'internal_diversity' in results and results['internal_diversity']['mean'] is not None:
        d = results['internal_diversity']
        print(f"  Internal diversity: {d['mean']} +/- {d['std']} (min={d['min']}, max={d['max']})")
    if 'uniqueness' in results:
        u = results['uniqueness']
        print(f"  Uniqueness (not in training): {u['n_unique']}/{u['n_total']} ({u['uniqueness']*100:.1f}%)")
    if 'novelty' in results and results['novelty']['n_total'] > 0:
        nov = results['novelty']
        print(f"  Novelty (NN sim < {nov['threshold']}): {nov['n_novel']}/{nov['n_total']} ({nov['novelty']*100:.1f}%)")
        print(f"    NN similarity: {nov['nn_sim_mean']} +/- {nov['nn_sim_std']}")
    if 'superclass_distribution' in results:
        print(f"  Superclasses ({results['n_superclasses']}):")
        for cls, count in sorted(results['superclass_distribution'].items(),
                                  key=lambda x: -x[1])[:10]:
            print(f"    {cls}: {count}")
    if "npclassifier" in results and results["npclassifier"]:
        npd = results["npclassifier"]
        print("\nNPClassifier (top 10):")
        for name, dist in [("Pathways", npd.get("pathway_distribution", {})),
                           ("Superclasses", npd.get("superclass_distribution", {})),
                           ("Classes", npd.get("class_distribution", {}))]:
            if dist:
                print(f"  {name}:")
                for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]:
                    print(f"    {k}: {v}")

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
