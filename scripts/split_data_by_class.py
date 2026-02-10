"""
Stratified train/test split by natural product class.

- Exact (or very close) 80/20 molecule split
- Preserves class proportions (stratified)
- Writes train.csv, test.csv, and val.csv
"""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split_3way(df, class_col, train_frac=0.8, val_frac=0.1, seed=42):
    """
    Stratified split into train/val/test.
    test_frac = 1 - train_frac - val_frac
    """
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    # 1) Split off train
    train, temp = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        stratify=df[class_col],
        random_state=seed,
        shuffle=True,
    )

    # 2) Split remaining temp into val/test with correct proportions
    # temp size = val_frac + test_frac
    val_share_of_temp = val_frac / (val_frac + test_frac)

    val, test = train_test_split(
        temp,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp[class_col],
        random_state=seed,
        shuffle=True,
    )

    return train, val, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--class_col", default="np_classifier_pathway", help="Class label column")
    ap.add_argument("--smiles_col", default=None, help="Optional SMILES column to require non-null")
    ap.add_argument("--train_frac", type=float, default=0.8, help="Train fraction (default 0.8)")
    ap.add_argument("--val_frac", type=float, default=0.1, help="Val fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--min_class_count",
        type=int,
        default=3,
        help="Drop classes with fewer than this many rows (default 3; needed for 3-way stratify)",
    )
    ap.add_argument("--out_train", default="train.csv")
    ap.add_argument("--out_val", default="val.csv")
    ap.add_argument("--out_test", default="test.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)

    # Basic cleaning: require class labels; optionally require SMILES too
    keep_cols = [args.class_col]
    if args.smiles_col:
        keep_cols.append(args.smiles_col)

    df = df.dropna(subset=keep_cols).copy()
    if args.smiles_col:
        df = df[df[args.smiles_col].astype(str).str.strip().ne("")].copy()

    # Drop very small classes (3-way stratified split needs >=3 examples/class in practice)
    vc = df[args.class_col].value_counts()
    keep_classes = vc[vc >= args.min_class_count].index
    df = df[df[args.class_col].isin(keep_classes)].copy()

    train, val, test = stratified_split_3way(
        df, args.class_col, args.train_frac, args.val_frac, args.seed
    )

    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)
    test.to_csv(args.out_test, index=False)

    total = len(train) + len(val) + len(test)
    print(f"Saved {args.out_train}: {len(train):,} ({len(train)/total:.3f})")
    print(f"Saved {args.out_val}:   {len(val):,} ({len(val)/total:.3f})")
    print(f"Saved {args.out_test}:  {len(test):,} ({len(test)/total:.3f})")

    # count per class splits
    counts = pd.DataFrame({
        "total": df[args.class_col].value_counts(),
        "train": train[args.class_col].value_counts(),
        "val": val[args.class_col].value_counts(),
        "test": test[args.class_col].value_counts(),
    }).fillna(0).astype(int)

    counts["train_frac"] = (counts["train"] / counts["total"]).round(3)
    counts["val_frac"]   = (counts["val"] / counts["total"]).round(3)
    counts["test_frac"]  = (counts["test"] / counts["total"]).round(3)

    counts = counts.sort_values("total", ascending=False)

    print("\n Per-class counts:")
    print(counts)


if __name__ == "__main__":
    main()