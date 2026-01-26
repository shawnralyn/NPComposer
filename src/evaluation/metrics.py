"""
Evaluation metrics for generated molecules

Usage:
    python metrics.py -i generated.txt
    python metrics.py -i generated.txt -o results.json
"""

import argparse
import json
import numpy as np
from typing import List, Dict

from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

try:
    from rdkit.Contrib.NP_Score import npscorer
    NP_MODEL = npscorer.readNPModel()
except:
    NP_MODEL = None


def evaluate(smiles_list: List[str]) -> Dict:
    """
    Evaluate a list of SMILES
    
    Returns dict with validity, sa_score, qed, np_score stats
    """
    sa, qed_scores, np_scores = [], [], []
    valid = 0
    
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        valid += 1
        
        sa.append(sascorer.calculateScore(mol))
        qed_scores.append(QED.qed(mol))
        if NP_MODEL:
            np_scores.append(npscorer.scoreMol(mol, NP_MODEL))
    
    n = len(smiles_list)
    
    return {
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


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")
    parser.add_argument("-i", "--input", required=True, help="SMILES file (one per line)")
    parser.add_argument("-o", "--output", help="Output JSON")
    
    args = parser.parse_args()
    
    with open(args.input) as f:
        smiles = [line.strip() for line in f if line.strip()]
    
    print(f"Evaluating {len(smiles)} molecules...")
    results = evaluate(smiles)
    
    print(f"\nResults:")
    print(f"  Valid: {results['n_valid']}/{results['n_total']} ({results['validity']*100:.1f}%)")
    print(f"  SA: {results['sa_score']['mean']} +/- {results['sa_score']['std']}")
    print(f"  QED: {results['qed']['mean']} +/- {results['qed']['std']}")
    if results['np_score']:
        print(f"  NP: {results['np_score']['mean']} +/- {results['np_score']['std']}")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
