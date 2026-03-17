"""Compute Morgan (ECFP) fingerprints for molecules in a CSV file.

Outputs a packed uint8 numpy array (.npy) ready for FAISS binary index
construction, and a companion CSV aligned row-for-row with the array.

Fingerprint format:
  - ECFP4 by default (radius=2, n_bits=2048)
  - Stored as packed bits: shape (n_valid, n_bits // 8), dtype uint8
  - Each row is 256 bytes for the default 2048-bit fingerprint
  - Load with np.load() and pass directly to faiss.IndexBinaryFlat

Usage:
    # Basic
    python scripts/morgan_fingerprints.py \\
        --input  data/coconut_clean.csv \\
        --output data/coconut_fp.npy \\
        --smiles-col canonical_smiles

    # ChEMBL (different SMILES column name)
    python scripts/morgan_fingerprints.py \\
        --input  data/ChEMBL_clean.csv \\
        --output data/chembl_fp.npy \\
        --smiles-col Smiles

    # ECFP6 with 1024 bits
    python scripts/morgan_fingerprints.py \\
        --input  data/coconut_clean.csv \\
        --output data/coconut_fp.npy \\
        --smiles-col canonical_smiles \\
        --radius 3 \\
        --n-bits 1024
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")


def compute_fingerprints(
    smiles_list: list[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> tuple[list[int], np.ndarray]:
    """Compute packed Morgan fingerprints for a list of SMILES.

    Invalid SMILES are silently skipped. The returned valid_indices allow
    the caller to align the fingerprint rows with the original DataFrame.

    Args:
        smiles_list: List of SMILES strings.
        radius: Morgan radius (2 = ECFP4, 3 = ECFP6).
        n_bits: Fingerprint length in bits.

    Returns:
        Tuple of:
          - valid_indices: positions in smiles_list that were successfully
            fingerprinted (same length as the fingerprint array rows)
          - fp_array: uint8 packed fingerprint array of shape
            (n_valid, n_bits // 8), ready for faiss.IndexBinaryFlat
    """
    valid_indices = []
    fps = []

    for i, smi in enumerate(tqdm(smiles_list, desc="Computing fingerprints")):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        valid_indices.append(i)

    if not fps:
        return [], np.zeros((0, n_bits // 8), dtype=np.uint8)

    fp_matrix = np.stack(fps, axis=0)                    # (n_valid, n_bits)
    fp_packed = np.packbits(fp_matrix, axis=1)            # (n_valid, n_bits // 8)
    return valid_indices, fp_packed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input CSV file containing SMILES",
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="Output path for packed fingerprint array (.npy)",
    )
    parser.add_argument(
        "--smiles-col", default="canonical_smiles",
        help="SMILES column name (default: canonical_smiles)",
    )
    parser.add_argument(
        "--radius", type=int, default=2,
        help="Morgan radius — 2=ECFP4, 3=ECFP6 (default: 2)",
    )
    parser.add_argument(
        "--n-bits", type=int, default=2048,
        help="Fingerprint length in bits (default: 2048)",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = output_path.with_suffix("").with_suffix("") \
        .parent / (output_path.stem + "_meta.csv")

    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"  {len(df):,} rows")

    if args.smiles_col not in df.columns:
        raise KeyError(
            f"SMILES column '{args.smiles_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    smiles_list = df[args.smiles_col].fillna("").tolist()

    print(f"\nComputing ECFP{args.radius * 2} fingerprints ({args.n_bits} bits)...")
    valid_indices, fp_packed = compute_fingerprints(
        smiles_list, radius=args.radius, n_bits=args.n_bits
    )

    n_invalid = len(smiles_list) - len(valid_indices)
    if n_invalid:
        print(f"  Skipped {n_invalid:,} invalid SMILES")
    print(f"  Fingerprinted: {len(valid_indices):,} molecules")
    print(f"  Array shape: {fp_packed.shape}  dtype: {fp_packed.dtype}")
    print(f"  Memory: {fp_packed.nbytes / 1e6:.1f} MB")

    np.save(output_path, fp_packed)
    print(f"\nFingerprints saved to {output_path}")

    meta = df.iloc[valid_indices].reset_index(drop=True)
    meta.to_csv(meta_path, index=False)
    print(f"Companion metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
