"""Stratified train/val/test split by superclass and glycoside."""

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 0)


def stratified_split_3way(
    df: pd.DataFrame,
    strata_col: str,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform stratified 3-way split into train, val, test.

    Input:
        df: DataFrame to split.
        strata_col: Column name for stratification.
        train_frac: Training fraction (default 0.8).
        val_frac: Validation fraction (default 0.1).
        seed: Random seed (default 42).
    Output:
        tuple: (train_df, val_df, test_df)
    """
    test_frac = 1.0 - train_frac - val_frac
    if test_frac <= 0:
        raise ValueError("train_frac + val_frac must be < 1.0")

    train, temp = train_test_split(
        df,
        test_size=(1.0 - train_frac),
        stratify=df[strata_col],
        random_state=seed,
        shuffle=True,
    )

    val_share_of_temp = val_frac / (val_frac + test_frac)

    val, test = train_test_split(
        temp,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp[strata_col],
        random_state=seed,
        shuffle=True,
    )

    return train, val, test


def main() -> None:
    """Split COCONUT CSV into stratified train/val/test sets.

    Input:
        --input: Input CSV path.
        --smiles_col: SMILES column name (default: canonical_smiles).
        --superclass_col: Superclass column name (default: np_classifier_superclass).
        --glycoside_col: Glycoside column name (default: np_classifier_is_glycoside).
        --train_frac: Training fraction (default: 0.8).
        --val_frac: Validation fraction (default: 0.1).
        --seed: Random seed (default: 42).
        --min_class_count: Min samples per stratum (default: 10).
        --out_train: Output training CSV (default: data/splits/train_v2.csv).
        --out_val: Output validation CSV (default: data/splits/val_v2.csv).
        --out_test: Output test CSV (default: data/splits/test_v2.csv).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--smiles_col", default="canonical_smiles",
                    help="SMILES column")
    ap.add_argument(
        "--superclass_col",
        default="np_classifier_superclass",
        help="name of NPClassifier superclass column in COCONUT",
    )
    ap.add_argument(
        "--glycoside_col",
        default="np_classifier_is_glycoside",
        help="name of column indicating presence of glycoside in molecule",
    )
    ap.add_argument("--train_frac", type=float, default=0.8,
                    help="Train fraction (default 0.8)")
    ap.add_argument("--val_frac", type=float, default=0.1,
                    help="Val fraction (default 0.1)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument(
        "--min_class_count",
        type=int,
        default=10,
        help="Drop strata with fewer than this many rows (default 10; needed for 3-way stratify)",
    )
    ap.add_argument("--out_train", default="data/splits/train_v2.csv")
    ap.add_argument("--out_val", default="data/splits/val_v2.csv")
    ap.add_argument("--out_test", default="data/splits/test_v2.csv")
    args = ap.parse_args()

    df: pd.DataFrame = pd.read_csv(args.input, low_memory=False)
    print("Length of original df:", len(df))

    keep_cols = [args.smiles_col, args.superclass_col, args.glycoside_col]

    missing = [col for col in keep_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df.dropna(subset=keep_cols).copy()
    for col in keep_cols:
        df = df[df[col].astype(str).str.strip() != ""].copy()
    print("Length of df after removing missing/unknown values:", len(df))

    df["_strata"] = (
        df[args.superclass_col].astype(str) + "|" + df[args.glycoside_col].astype(str)
    )

    vc = df["_strata"].value_counts()
    df = df[df["_strata"].map(vc).ge(args.min_class_count)].copy()

    train, val, test = stratified_split_3way(
        df, "_strata", args.train_frac, args.val_frac, args.seed
    )

    train.to_csv(args.out_train, index=False)
    val.to_csv(args.out_val, index=False)
    test.to_csv(args.out_test, index=False)

    total = len(train) + len(val) + len(test)
    print(f"Saved {args.out_train}: {len(train):,} ({len(train)/total:.3f})")
    print(f"Saved {args.out_val}:   {len(val):,} ({len(val)/total:.3f})")
    print(f"Saved {args.out_test}:  {len(test):,} ({len(test)/total:.3f})")

    counts = pd.DataFrame({
        "total": df["_strata"].value_counts(),
        "train": train["_strata"].value_counts(),
        "val": val["_strata"].value_counts(),
        "test": test["_strata"].value_counts(),
    }).fillna(0).astype(int)

    counts["train_frac"] = (counts["train"] / counts["total"]).round(3)
    counts["val_frac"] = (counts["val"] / counts["total"]).round(3)
    counts["test_frac"] = (counts["test"] / counts["total"]).round(3)

    counts = counts.sort_values("total", ascending=False)

    print("\nPer-stratum counts (superclass|glycoside):")
    print(counts)
    print(f"Number of strata: {counts.shape[0]}")

    retained_strata = df["_strata"].nunique()
    print(
        f"Number of strata retained (>= {args.min_class_count} samples): {retained_strata}")


if __name__ == "__main__":
    main()
