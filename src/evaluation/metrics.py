"""Evaluation metrics for generated molecules."""

import argparse
import json
import sys
import numpy as np
from pathlib import Path
from typing import List, Dict
from collections import Counter
import time

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


try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def _get_first_str(x):
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    return None


def npclassifier_classify(smiles: str,
                          base_url: str = "https://npclassifier.ucsd.edu/classify",
                          cached: bool = True,
                          timeout: int = 30,
                          max_retries: int = 3,
                          sleep_s: float = 0.05) -> dict:
    """Call NPClassifier API for one SMILES. Returns JSON dict or {'error': ...}."""
    if not HAS_REQUESTS:
        return {"error": "requests not installed"}

    params = {"smiles": smiles}
    if cached:
        params["cached"] = "true"

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(base_url, params=params, timeout=timeout)
            r.raise_for_status()
            time.sleep(sleep_s)  
            return r.json()
        except Exception as e:
            last_err = str(e)
            time.sleep(sleep_s * (2 ** attempt))
    return {"error": last_err}


def evaluate(smiles_list: List[str],
             classify: bool = False,
             npclassify: bool = False,
             np_url: str = "https://npclassifier.ucsd.edu/classify",
             keep_np_per_mol: bool = False) -> Dict:
    """Compute validity, SA, QED, NP-likeness, and optionally superclass stats and NP pathway.

    Input:
        smiles_list: list of SMILES strings.
        classify: if True, query ClassyFire API for superclass distribution.
        npclassify: if True, query NPClassifier API for pathway classification
        np_url: URL for NPClassifier API
    Output:
        dict with keys: n_total, n_valid, validity, sa_score, qed, np_score,
        and optionally superclass_distribution.
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

    # NPClassifier
    if npclassify and valid_smiles:
        if not HAS_REQUESTS:
            print("Warning: NPClassifier requires 'requests' (pip install requests)")
        else:
            print("Classifying with NPClassifier API...")
            pathway_dist = Counter()
            superclass_dist = Counter()
            class_dist = Counter()
            per_mol = [] if keep_np_per_mol else None

            for s in valid_smiles:
                res = npclassifier_classify(s, base_url=np_url, cached=True)

                if "error" in res:
                    if keep_np_per_mol:
                        per_mol.append({"smiles": s, "error": res["error"]})
                    continue

                pathway = _get_first_str(res.get("pathway_results")) or _get_first_str(res.get("pathway"))
                superclass = _get_first_str(res.get("superclass_results")) or _get_first_str(res.get("superclass"))
                cls = _get_first_str(res.get("class_results")) or _get_first_str(res.get("class"))

                if pathway: pathway_dist[pathway] += 1
                if superclass: superclass_dist[superclass] += 1
                if cls: class_dist[cls] += 1

                if keep_np_per_mol:
                    per_mol.append({
                        "smiles": s,
                        "pathway": pathway,
                        "superclass": superclass,
                        "class": cls,
                    })

            results["npclassifier"] = {
                "pathway_distribution": dict(pathway_dist),
                "superclass_distribution": dict(superclass_dist),
                "class_distribution": dict(class_dist),
            }
            if keep_np_per_mol:
                results["npclassifier"]["per_molecule"] = per_mol

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("-i", "--input", required=True, help="SMILES file (one per line)")
    parser.add_argument("-o", "--output", help="Output JSON")
    parser.add_argument("--classify", action="store_true",
                        help="Compute ClassyFire superclass distribution")
    parser.add_argument("--npclassify", action="store_true",
                        help="Run NPClassifier on valid SMILES via API")
    parser.add_argument("--np_url", default="https://npclassifier.ucsd.edu/classify",
                        help="NPClassifier classify endpoint URL")
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
        npclassify=args.npclassify,
        np_url=args.np_url,
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
