"""
Data Splitter
Split subset into train/val/test sets

Usage:
    python split_data.py -i subset_5k.csv -o splits/
    python split_data.py -i subset_5k.csv -o splits/ --train 0.8 --val 0.1 --test 0.1
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def split_data(
    input_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Split data into train/val/test"""
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    n = len(df)
    print(f"Total: {n:,} molecules")
    
    # Shuffle
    np.random.seed(seed)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Split indices
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    
    # Save
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
