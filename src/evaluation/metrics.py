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
from rdkit.Chem import QED
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


def evaluate(smiles_list: List[str],
             classify: bool = False,
             np_repo_root: str = None,
             keep_np_per_mol: bool = False) -> Dict:
    """Compute validity, SA, QED, NP-likeness, and optionally NPClassifier classification.

    Input:
        smiles_list: list of SMILES strings.
        classify: if True, run NPClassifier (local) for pathway/superclass/class.
        np_repo_root: path to NP-Classifier repo clone (or set NP_CLASSIFIER_ROOT env).
        keep_np_per_mol: if True, store per-molecule NPClassifier assignments.
    Output:
        dict with keys: n_total, n_valid, validity, sa_score, qed, np_score,
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

    # NPClassifier (pure local inference) classification
    if classify and valid_smiles:
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
    parser.add_argument("--classify", action="store_true",
                        help="Run NPClassifier (local) for pathway/superclass/class classification")
    parser.add_argument("--np_root",
                        default=os.environ.get("NP_CLASSIFIER_ROOT"),
                        help="Path to NP-Classifier repo clone (or set NP_CLASSIFIER_ROOT env)")
    parser.add_argument("--keep_np_per_mol", action="store_true",
                        help="Store per-molecule NPClassifier assignments in output JSON")
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
        classify=args.classify,
        np_repo_root=args.np_root,
        keep_np_per_mol=args.keep_np_per_mol,
    )

    print(f"\nResults:")
    print(f"  Valid: {results['n_valid']}/{results['n_total']} ({results['validity']*100:.1f}%)")
    print(f"  SA: {results['sa_score']['mean']} +/- {results['sa_score']['std']}")
    print(f"  QED: {results['qed']['mean']} +/- {results['qed']['std']}")
    if results['np_score']:
        print(f"  NP: {results['np_score']['mean']} +/- {results['np_score']['std']}")
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
