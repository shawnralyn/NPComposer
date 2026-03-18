import pandas as pd
import pytest

from src.data_preprocessing.stratified_train_split import stratified_split_3way


def test_stratified_split_3way_sizes_and_strata_distribution():
    # Build a small dataset with 2 strata, enough samples per stratum.
    n_per = 50
    df = pd.DataFrame(
        {
            "smiles": ["CCO"] * (2 * n_per),
            "_strata": (["A|0"] * n_per) + (["B|1"] * n_per),
        }
    )

    train, val, test = stratified_split_3way(df, strata_col="_strata", train_frac=0.8, val_frac=0.1, seed=123)

    assert len(train) + len(val) + len(test) == len(df)
    assert len(train) == 80
    assert len(val) == 10
    assert len(test) == 10

    # Each split should preserve the 50/50 composition.
    for split in (train, val, test):
        counts = split["_strata"].value_counts(normalize=True)
        assert counts.loc["A|0"] == pytest.approx(0.5, abs=0.05)
        assert counts.loc["B|1"] == pytest.approx(0.5, abs=0.05)


def test_stratified_split_3way_invalid_fractions():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "strata": ["a", "a", "b", "b"]})
    with pytest.raises(ValueError):
        stratified_split_3way(df, strata_col="strata", train_frac=0.95, val_frac=0.1)
