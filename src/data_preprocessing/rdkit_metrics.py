"""Compute RDKit SA scores for COCONUT CSV."""

import argparse
from typing import Dict, Optional, Any
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Contrib.SA_Score import sascorer


def compute_sa_score(smiles: str) -> Dict[str, Any]:
    """Compute synthetic accessibility score from SMILES string.

    Input:
        smiles: SMILES string.
    Output:
        dict: {'valid': bool, 'sa': float or None}
    """
    result: Dict[str, Any] = {'valid': False, 'sa': None}

    if not isinstance(smiles, str) or smiles == 'n.a.' or not smiles:
        return result

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return result

    result['valid'] = True

    try:
        result['sa'] = sascorer.calculateScore(mol)
    except (ValueError, RuntimeError):
        pass

    return result


def main() -> None:
    """Compute SA scores for COCONUT CSV.

    Input:
        --coconut: Path to input CSV.
        --out_file: Path to output CSV.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut", required=True, help="Path to COCONUT CSV")
    ap.add_argument("--out_file", required=True,
                    help="Path to updated COCONUT CSV")
    args = ap.parse_args()

    df: pd.DataFrame = pd.read_csv(args.coconut).dropna(
        subset=['canonical_smiles'])

    sa_scores: Dict[int, Optional[float]] = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing SA scores:"):
        smiles: str = row['canonical_smiles']
        result = compute_sa_score(smiles)
        sa_scores[idx] = result['sa']

    df['sa_score'] = pd.Series(sa_scores)

    df.to_csv(args.out_file)


if __name__ == "__main__":
    main()
