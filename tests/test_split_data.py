"""Tests for scripts/split_data.py"""

import pytest
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from split_data import split_data


class TestSplitData:
    def test_default_ratios(self, tmp_csv, tmp_path):
        out_dir = str(tmp_path / "splits")
        split_data(tmp_csv, out_dir)

        train = pd.read_csv(Path(out_dir) / "train.csv")
        val = pd.read_csv(Path(out_dir) / "val.csv")
        test = pd.read_csv(Path(out_dir) / "test.csv")

        total = len(train) + len(val) + len(test)
        assert total == 20  # sample_df has 20 rows
        assert len(train) == 16  # 80%
        assert len(val) == 2    # 10%
        assert len(test) == 2   # 10%

    def test_custom_ratios(self, tmp_csv, tmp_path):
        out_dir = str(tmp_path / "splits")
        split_data(tmp_csv, out_dir, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

        train = pd.read_csv(Path(out_dir) / "train.csv")
        val = pd.read_csv(Path(out_dir) / "val.csv")
        test = pd.read_csv(Path(out_dir) / "test.csv")

        assert len(train) == 12
        assert len(val) == 4

    def test_reproducibility(self, tmp_csv, tmp_path):
        out1 = str(tmp_path / "split1")
        out2 = str(tmp_path / "split2")
        split_data(tmp_csv, out1, seed=42)
        split_data(tmp_csv, out2, seed=42)

        t1 = pd.read_csv(Path(out1) / "train.csv")
        t2 = pd.read_csv(Path(out2) / "train.csv")
        pd.testing.assert_frame_equal(t1, t2)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            split_data("/nonexistent/file.csv", str(tmp_path / "out"))

    def test_invalid_ratios_sum(self, tmp_csv, tmp_path):
        with pytest.raises(ValueError, match="sum to 1"):
            split_data(tmp_csv, str(tmp_path / "out"),
                       train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_negative_ratio(self, tmp_csv, tmp_path):
        with pytest.raises(ValueError, match="non-negative"):
            split_data(tmp_csv, str(tmp_path / "out"),
                       train_ratio=-0.1, val_ratio=0.5, test_ratio=0.6)

    def test_no_data_overlap(self, tmp_csv, tmp_path):
        out_dir = str(tmp_path / "splits")
        split_data(tmp_csv, out_dir)

        train = pd.read_csv(Path(out_dir) / "train.csv")
        val = pd.read_csv(Path(out_dir) / "val.csv")
        test = pd.read_csv(Path(out_dir) / "test.csv")

        train_ids = set(train["identifier"])
        val_ids = set(val["identifier"])
        test_ids = set(test["identifier"])

        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
