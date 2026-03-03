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


def compute_novelty(generated_smiles: List[str], training_smiles: List[str]) -> Dict:
    """Compute novelty: fraction of generated molecules not in training set.

    Compares canonical SMILES strings.

    Input:
        generated_smiles (List[str]): valid generated SMILES.
        training_smiles (List[str]): training set SMILES.
    Output:
        dict with n_novel, n_total, novelty ratio.
    """
    # canonicalize both sets for fair comparison
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

    n_novel = 0
    n_total = 0
    for s in generated_smiles:
        c = canon(s)
        if c is None:
            continue
        n_total += 1
        if c not in train_canonical:
            n_novel += 1

    return {
        "n_novel": n_novel,
        "n_total": n_total,
        "novelty": round(n_novel / n_total, 4) if n_total > 0 else 0,
    }


def compute_training_similarity(generated_smiles: List[str],
                                 training_smiles: List[str],
                                 sample_size: int = 1000) -> Dict:
    """Compute nearest-neighbor Tanimoto similarity to training set.

    For each generated molecule, find the most similar molecule in the training
    set and report statistics over these nearest-neighbor similarities.

    Input:
        generated_smiles (List[str]): valid generated SMILES.
        training_smiles (List[str]): training set SMILES.
        sample_size (int): max training molecules to compare against.
    Output:
        dict with mean, std, min, max of nearest-neighbor similarities.
    """
    gen_fps = [(s, _smiles_to_fp(s)) for s in generated_smiles]
    gen_fps = [(s, fp) for s, fp in gen_fps if fp is not None]

    train_fps = [_smiles_to_fp(s) for s in training_smiles]
    train_fps = [fp for fp in train_fps if fp is not None]

    if not gen_fps or not train_fps:
        return {"mean": None, "std": None, "min": None, "max": None}

    # subsample training fps if too large
    if len(train_fps) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_fps), size=sample_size, replace=False)
        train_fps = [train_fps[i] for i in idx]

    nn_sims = []
    for _, gen_fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(gen_fp, train_fps)
        nn_sims.append(max(sims))

    return {
        "mean": round(np.mean(nn_sims), 4),
        "std": round(np.std(nn_sims), 4),
        "min": round(np.min(nn_sims), 4),
        "max": round(np.max(nn_sims), 4),
    }


def evaluate(smiles_list: List[str],
             np_repo_root: str = None,
             keep_np_per_mol: bool = False,
             training_file: str = None) -> Dict:
    """Compute validity, SA, QED, NP-likeness, diversity, novelty, and training similarity.

    Input:
        smiles_list: list of SMILES strings.
        np_repo_root: path to NP-Classifier repo clone (or set NP_CLASSIFIER_ROOT env).
        keep_np_per_mol: if True, store per-molecule NPClassifier assignments.
        training_file: path to training CSV for novelty/similarity checks.
    Output:
        dict with keys: n_total, n_valid, validity, sa_score, qed, np_score,
        internal_diversity, novelty, training_similarity,
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
        results["uniqueness"] = round(len(set(valid_smiles)) / len(valid_smiles), 4) if valid_smiles else 0

    # Novelty and training similarity (requires training data)
    if training_file and valid_smiles:
        import pandas as pd
        print(f"Loading training data from {training_file}...")
        train_df = pd.read_csv(training_file, low_memory=False)

        # auto-detect SMILES column
        smiles_col = None
        for col in ['canonical_smiles', 'SMILES', 'smiles']:
            if col in train_df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            print("Warning: could not detect SMILES column in training file, skipping novelty/similarity")
        else:
            train_smiles = train_df[smiles_col].dropna().tolist()
            print(f"  Training set: {len(train_smiles):,} molecules")

            print("Computing novelty...")
            results["novelty"] = compute_novelty(valid_smiles, train_smiles)

            print("Computing training similarity (nearest-neighbor Tanimoto)...")
            results["training_similarity"] = compute_training_similarity(valid_smiles, train_smiles)

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
                        help="Training CSV file for novelty and similarity evaluation")
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
    )

    print(f"\nResults:")
    print(f"  Valid: {results['n_valid']}/{results['n_total']} ({results['validity']*100:.1f}%)")
    print(f"  SA: {results['sa_score']['mean']} +/- {results['sa_score']['std']}")
    print(f"  QED: {results['qed']['mean']} +/- {results['qed']['std']}")
    if results['np_score']:
        print(f"  NP: {results['np_score']['mean']} +/- {results['np_score']['std']}")
    if 'uniqueness' in results:
        print(f"  Uniqueness: {results['uniqueness']*100:.1f}%")
    if 'internal_diversity' in results and results['internal_diversity']['mean'] is not None:
        d = results['internal_diversity']
        print(f"  Internal diversity: {d['mean']} +/- {d['std']} (min={d['min']}, max={d['max']})")
    if 'novelty' in results:
        nov = results['novelty']
        print(f"  Novelty: {nov['n_novel']}/{nov['n_total']} ({nov['novelty']*100:.1f}%)")
    if 'training_similarity' in results and results['training_similarity']['mean'] is not None:
        ts = results['training_similarity']
        print(f"  NN similarity to training: {ts['mean']} +/- {ts['std']} (min={ts['min']}, max={ts['max']})")
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
