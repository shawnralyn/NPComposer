"""Append rdkit metrics to COCONUT database

Synthetic accessibility (SA) score only in this version, but can be easily updated for 
additional rdkit metrics
"""

import argparse
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Contrib.SA_Score import sascorer


def compute_sa_score(smiles):
    """
    Compute all synthetic accessibility score from SMILES string

    Input:
        smiles: SMILES string
    Output:
        dict with keys: valid, sa
    """
    result = {'valid': False, 'sa': None}
    
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coconut", required=True, help="Path to COCONUT CSV")
    ap.add_argument("--out_file", required=True, help="Path to updated COCONUT CSV")
    args = ap.parse_args()
    
    # read in COCONUT csv and drop missing values in canonical smiles column
    df = pd.read_csv(args.coconut).dropna(subset=['canonical_smiles'])

    sa_scores = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing SA scores:"):
        smiles = row['canonical_smiles']
        result = compute_sa_score(smiles)
        sa_scores[idx] = result['sa']

    df['sa_score'] = pd.Series(sa_scores)

    df.to_csv(args.out_file)


if __name__ == "__main__":
    main()


        