"""Row-wise binning of QED and SA scores."""

import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def define_bins() -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Define bins and labels for QED and SA scores.

    Output:
        tuple: (qed_bins, qed_labels, sa_bins, sa_labels)
    """

    # create QED bins
    qed_bins = np.arange(0.0, 1.0 + 1e-12, 0.1)
    qed_labels = []
    for i in range(len(qed_bins) - 1):
        label = f"{qed_bins[i]:g}<=qed<{qed_bins[i+1]:g}"
        qed_labels.append(label)

    # create SA bins
    sa_bins = np.arange(1.0, 10.0 + 1e-12, 1.0)
    sa_labels = []
    for i in range(len(sa_bins) - 1):
        label = f"{sa_bins[i]:g}<=sa<{sa_bins[i+1]:g}"
        sa_labels.append(label)

    return qed_bins, qed_labels, sa_bins, sa_labels


def bin_row(
    qed_val: Optional[float],
    sa_val: Optional[float],
    qed_bins: np.ndarray,
    qed_labels: List[str],
    sa_bins: np.ndarray,
    sa_labels: List[str]
) -> Dict[str, Optional[str]]:
    """Assign bin labels to QED and SA values.

    Input:
        qed_val: QED score or None.
        sa_val: SA score or None.
        qed_bins: Bin edges for QED.
        qed_labels: Labels for QED bins.
        sa_bins: Bin edges for SA.
        sa_labels: Labels for SA bins.
    Output:
        dict: {'qed_bin': label or None, 'sa_bin': label or None}
    """

    if qed_val is None or (isinstance(qed_val, float) and np.isnan(qed_val)):
        qed_bin = None
    else:
        qed = float(qed_val)
        if qed >= qed_bins[-1]:
            qed = qed_bins[-1] - 1e-8
        bin_idx = np.digitize(qed, qed_bins) - 1
        qed_bin = qed_labels[bin_idx] if 0 <= bin_idx < len(
            qed_labels) else None

    if sa_val is None or (isinstance(sa_val, float) and np.isnan(sa_val)):
        sa_bin = None
    else:
        sa = float(sa_val)
        if sa >= sa_bins[-1]:
            sa = sa_bins[-1] - 1e-8
        bin_idx = np.digitize(sa, sa_bins) - 1
        sa_bin = sa_labels[bin_idx] if 0 <= bin_idx < len(sa_labels) else None

    return {"qed_bin": qed_bin, "sa_bin": sa_bin}


def main() -> None:
    """Bin QED and SA scores in COCONUT CSV.

    Input:
        --input_csv: Path to COCONUT CSV with 'qed_drug_likeliness' and 'sa_score'.
        --output: Path to output CSV.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True,
                    help="Path to COCONUT CSV with added SA scores")
    ap.add_argument("--output", required=True,
                    help="Path to updated COCONUT CSV with binned QED and SA values")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    required_cols = ['qed_drug_likeliness', 'sa_score']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    df = df.dropna(subset=required_cols)

    qed_bins, qed_labels, sa_bins, sa_labels = define_bins()

    binned = {}

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Binning QED + SA"):
        res = bin_row(
            row["qed_drug_likeliness"],
            row["sa_score"],
            qed_bins, qed_labels,
            sa_bins, sa_labels,
        )
        binned[idx] = res

    binned_df = pd.DataFrame.from_dict(binned, orient="index")
    df = df.join(binned_df)

    df.to_csv(args.output)


if __name__ == "__main__":
    main()
