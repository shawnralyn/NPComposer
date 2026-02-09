"""Split subset into train/val/test sets."""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def split_data(input_path, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split CSV data into train/val/test sets.

    Input:
        input_path: path to input CSV.
        output_dir: directory to save split files.
        train_ratio: fraction for training set.
        val_ratio: fraction for validation set.
        test_ratio: fraction for test set.
        seed: random seed (default 42).
    Output:
        Writes train.csv, val.csv, test.csv to output_dir.
    """
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if any(r < 0 for r in [train_ratio, val_ratio, test_ratio]):
        raise ValueError("Ratios must be non-negative")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio:.4f}")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    n = len(df)
    print(f"Total: {n:,} molecules")

    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)

    print(f"\nSaved to {output_dir}/")
    print(f"  train.csv: {len(train_df):,} ({100*len(train_df)/n:.0f}%)")
    print(f"  val.csv:   {len(val_df):,} ({100*len(val_df)/n:.0f}%)")
    print(f"  test.csv:  {len(test_df):,} ({100*len(test_df)/n:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test")
    parser.add_argument("-i", "--input", required=True, help="Input CSV")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("--train", type=float, default=0.8, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.1, help="Val ratio")
    parser.add_argument("--test", type=float, default=0.1, help="Test ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    split_data(args.input, args.output, args.train, args.val, args.test, args.seed)


if __name__ == "__main__":
    main()
