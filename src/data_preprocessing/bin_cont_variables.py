"""
Row-wise binning of QED (0-1 by 0.1) and SA (1-10 by 1.0) for downstream tokenization. 
Adds 'qed_bin' and 'sa_bin' columns to COCONUT input csv using a per-row loop keyed by index.
"""

import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm


def define_bins() -> Tuple[np.ndarray, List[str], np.ndarray, List[str]]:
    """Define bins and labels for QED and SA scores.

    Creates bin edges and human-readable labels for QED (0.0 to 1.0, step 0.1)
    and SA (1.0 to 10.0, step 1.0) scores.

    Returns:
        Tuple[np.ndarray, List[str], np.ndarray, List[str]]: A tuple containing:
            - qed_bins (np.ndarray): Bin edges for QED scores.
            - qed_labels (List[str]): Labels for each QED bin, e.g. '0<=qed<0.1'.
            - sa_bins (np.ndarray): Bin edges for SA scores.
            - sa_labels (List[str]): Labels for each SA bin, e.g. '1<=sa<2'.
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
    """Assign bin labels to a single row's QED and SA values.

    Args:
        qed_val (Optional[float]): The QED score for the row, or None if missing.
        sa_val (Optional[float]): The SA score for the row, or None if missing.
        qed_bins (np.ndarray): Array of bin edges for QED.
        qed_labels (List[str]): Labels for each QED bin.
        sa_bins (np.ndarray): Array of bin edges for SA.
        sa_labels (List[str]): Labels for each SA bin.

    Returns:
        Dict[str, Optional[str]]: Dictionary with keys 'qed_bin' and 'sa_bin',
            each mapping to a bin label string or None if the value was missing.
    """

    # qed drug likeliness
    if qed_val is None or (isinstance(qed_val, float) and np.isnan(qed_val)):
        qed_bin = None  # If missing, set bin to None
    else:
        qed = float(qed_val)

        # if value is at or above the last bin edge, set it just below the upper bound
        if qed >= qed_bins[-1]:
            qed = qed_bins[-1] - 1e-8

        # find which bin q belongs to (returns index)
        bin_idx = np.digitize(qed, qed_bins) - 1

        # get the label for this bin, or None if out of range
        qed_bin = qed_labels[bin_idx] if 0 <= bin_idx < len(
            qed_labels) else None

    # synthetic accessibility (SA)
    if sa_val is None or (isinstance(sa_val, float) and np.isnan(sa_val)):
        sa_bin = None
    else:
        sa = float(sa_val)

        if sa >= sa_bins[-1]:
            sa = sa_bins[-1] - 1e-8

        bin_idx = np.digitize(sa, sa_bins) - 1

        sa_bin = sa_labels[bin_idx] if 0 <= bin_idx < len(sa_labels) else None

    # return dict with bin labels for row
    return {"qed_bin": qed_bin, "sa_bin": sa_bin}


def main() -> None:
    """Bin QED and SA scores in a COCONUT database CSV.

    Reads the input CSV, bins 'qed_drug_likeliness' and 'sa_score' columns into
    discrete ranges, adds 'qed_bin' and 'sa_bin' columns, and writes the result
    to an output CSV.

    Command-line Arguments:
        --input_csv (str): Path to input COCONUT CSV with 'qed_drug_likeliness'
            and 'sa_score' columns.
        --output (str): Path to output CSV with added 'qed_bin' and 'sa_bin' columns.

    Raises:
        ValueError: If required columns are missing from the input CSV.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True,
                    help="Path to COCONUT CSV with added SA scores")
    ap.add_argument("--output", required=True,
                    help="Path to updated COCONUT CSV with binned QED and SA values")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    # check df for required columns
    required_cols = ['qed_drug_likeliness', 'sa_score']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    # drop rows missing QED, SA score values
    df = df.dropna(subset=required_cols)

    # create bins
    qed_bins, qed_labels, sa_bins, sa_labels = define_bins()

    binned = {}

    # loop over each row of dataframe to bin QED, SA scores
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Binning QED + SA"):
        res = bin_row(
            row["qed_drug_likeliness"],
            row["sa_score"],
            qed_bins, qed_labels,
            sa_bins, sa_labels,
        )
        binned[idx] = res

    binned_df = pd.DataFrame.from_dict(binned, orient="index")

    # join binned df with original df to add qed_bin and sa_bin columns
    df = df.join(binned_df)

    df.to_csv(args.output)


if __name__ == "__main__":
    main()
