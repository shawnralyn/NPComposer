"""Evaluation metrics for generated molecules."""

import argparse
import json
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

# ClassyFire classification (optional)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))
try:
    from classyfire import classify_batch
    HAS_CLASSYFIRE = True
except ImportError:
    HAS_CLASSYFIRE = False


def evaluate(smiles_list: List[str], classify: bool = False) -> Dict:
    """Compute validity, SA, QED, NP-likeness, and optionally superclass stats.

    Input:
        smiles_list: list of SMILES strings.
        classify: if True, query ClassyFire API for superclass distribution.
    Output:
        dict with keys: n_total, n_valid, validity, sa_score, qed, np_score,
        and optionally superclass_distribution.
    """
    sa, qed_scores, np_scores = [], [], []
    valid = 0
    valid_smiles = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        valid += 1
        valid_smiles.append(s)
        sa.append(sascorer.calculateScore(mol))
        qed_scores.append(QED.qed(mol))
        if NP_MODEL:
            np_scores.append(npscorer.scoreMol(mol, NP_MODEL))

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

    # ClassyFire superclass distribution
    if classify and valid_smiles:
        if HAS_CLASSYFIRE:
            print("Classifying superclass (ClassyFire API)...")
            superclasses = classify_batch(valid_smiles)
            dist = dict(Counter(superclasses))
            results["superclass_distribution"] = dist
            results["n_superclasses"] = len([k for k in dist if k != "Unknown"])
        else:
            print("Warning: ClassyFire not available (install 'requests')")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("-i", "--input", required=True, help="SMILES file (one per line)")
    parser.add_argument("-o", "--output", help="Output JSON")
    parser.add_argument("--classify", action="store_true",
                        help="Compute ClassyFire superclass distribution")
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
    results = evaluate(smiles, classify=args.classify)

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

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
