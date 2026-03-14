"""Build a FAISS binary index from a packed Morgan fingerprint array.

Loads a uint8 packed fingerprint array produced by morgan_fingerprints.py,
builds a faiss.IndexBinaryFlat (exact Hamming distance search), and saves
it to disk. The index can then be queried repeatedly without rebuilding.

Usage:
    python scripts/build_faiss_index.py \\
        --fingerprints data/chembl_fp.npy \\
        --output       data/chembl.faiss
"""
import argparse
import time
from pathlib import Path

import numpy as np

try:
    import faiss
except ImportError as e:
    raise ImportError(
        "faiss is required. Install with:\n"
        "  conda install -c pytorch faiss-cpu\n"
        "  or: pip install faiss-cpu"
    ) from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--fingerprints", required=True,
        help="Packed uint8 fingerprint array (.npy) from morgan_fingerprints.py",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output path for FAISS binary index (.faiss)",
    )
    args = parser.parse_args()

    print(f"Loading fingerprints from {args.fingerprints}...")
    fp = np.load(args.fingerprints)
    assert fp.dtype == np.uint8, f"Expected uint8 array, got {fp.dtype}"
    n, packed_bits = fp.shape
    n_bits = packed_bits * 8
    print(f"  {n:,} molecules  {n_bits}-bit fingerprints  ({fp.nbytes / 1e6:.1f} MB)")

    print(f"Building faiss.IndexBinaryFlat ({n_bits} bits)...")
    t0 = time.time()
    index = faiss.IndexBinaryFlat(n_bits)
    index.add(fp)
    print(f"  Built in {time.time() - t0:.1f}s  ({index.ntotal:,} vectors indexed)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index_binary(index, str(out_path))
    print(f"Index saved to {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
