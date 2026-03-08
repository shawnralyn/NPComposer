"""
Row-wise binning of QED (0-1 by 0.1) and SA (1-10 by 1.0) for downstream tokenization. 
Adds 'qed_bin' and 'sa_bin' columns, using a per-row loop keyed by index.
"""

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm


def define_bins():
    """
    Define bins and labels for QED and SA scores

    Returns:
        tuple: (qed_bins, qed_labels, sa_bins, sa_labels)
            - qed_bins (np.ndarray): Bins for QED scores (from 0.0 to 1.0, step 0.1).
            - qed_labels (list of str): Labels for each QED bin, e.g. '0<=qed<0.1'.
            - sa_bins (np.ndarray): Bins for SA scores (from 1.0 to 10.0, step 1.0).
            - sa_labels (list of str): Labels for each SA bin, e.g. '1<=sa<2'.
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


def bin_row(qed_val, sa_val, qed_bins, qed_labels, sa_bins, sa_labels):
    """
    Assigns bin labels to a single row's QED and SA values.

    inputs:
        qed_val (float or None): The QED score for the row 
        sa_val (float or None): The SA score for the row 
        qed_bins (np.ndarray): Array of bins for QED
        qed_labels (list of str): Labels for each QED bin
        sa_bins (np.ndarray): Array of bins for SA
        sa_labels (list of str): Human-readable labels for each SA bin

    outputs:
        dict: Dictionary containing 'qed_bin' and 'sa_bin' (e.g., {"qed_bin": <label|None>, "sa_bin": <label|None>})
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
        qed_bin = qed_labels[bin_idx] if 0 <= bin_idx < len(qed_labels) else None

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="Path to COCONUT CSV with added SA scores")
    ap.add_argument("--output", required=True, help="Path to updated COCONUT CSV with binned QED and SA values")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)

    # check df for required columns
    required_cols = ['qed_drug_likeliness','sa_score']
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
    df = df.join(binned_df) # join binned df with original df to add qed_bin and sa_bin columns

    df.to_csv(args.output)


if __name__ == "__main__":
    main()