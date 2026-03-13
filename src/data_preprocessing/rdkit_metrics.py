"""Append rdkit metrics to COCONUT database CSV.

Synthetic accessibility (SA) score only in this version, but can be easily updated for 
additional rdkit metrics.
"""

import argparse
from typing import Dict, Optional, Any
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Contrib.SA_Score import sascorer


def compute_sa_score(smiles: str) -> Dict[str, Any]:
    """Compute synthetic accessibility score from a SMILES string.

    Args:
        smiles (str): SMILES string representing a molecule.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'valid' (bool): Whether the SMILES was successfully parsed.
            - 'sa' (Optional[float]): The SA score, or None if computation failed.
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
    """Compute synthetic accessibility (SA) scores for molecules in a COCONUT database CSV.

    Reads the input CSV, computes SA scores for each valid SMILES in the 'canonical_smiles'
    column, appends the scores as a new 'sa_score' column, and writes the result to an output CSV.

    Command-line Arguments:
        --coconut (str): Path to input COCONUT CSV file.
        --out_file (str): Path to output CSV file with SA scores appended.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut", required=True, help="Path to COCONUT CSV")
    ap.add_argument("--out_file", required=True,
                    help="Path to updated COCONUT CSV")
    args = ap.parse_args()

    # read in COCONUT csv and drop missing values in canonical smiles column
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
