"""Tests for scripts/create_subset.py"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from create_subset import (
    kmeans_select,
    compute_properties,
    smiles_to_fp,
    detect_smiles_column,
    detect_id_column,
    load_and_filter,
)


class TestKmeansSelect:
    def test_returns_correct_count(self):
        rng = np.random.default_rng(42)
        X = rng.random((100, 20)).astype(np.float32)
        selected = kmeans_select(X, target_size=30, seed=42)
        assert len(selected) == 30

    def test_selected_are_unique(self):
        rng = np.random.default_rng(42)
        X = rng.random((100, 20)).astype(np.float32)
        selected = kmeans_select(X, target_size=50, seed=42)
        assert len(set(selected)) == len(selected)

    def test_selected_in_range(self):
        n = 80
        rng = np.random.default_rng(42)
        X = rng.random((n, 10)).astype(np.float32)
        selected = kmeans_select(X, target_size=20, seed=42)
        assert all(0 <= s < n for s in selected)

    def test_returns_all_if_small(self):
        rng = np.random.default_rng(42)
        X = rng.random((10, 5)).astype(np.float32)
        selected = kmeans_select(X, target_size=10, seed=42)
        assert len(selected) == 10


class TestComputeProperties:
    def test_valid_smiles(self):
        result = compute_properties("CCO")
        assert result['valid'] is True
        assert result['atom_count'] > 0
        assert result['ring_count'] == 0
        assert result['sa'] is not None
        assert 1.0 <= result['sa'] <= 10.0
        assert result['qed'] is not None
        assert 0.0 <= result['qed'] <= 1.0

    def test_invalid_smiles(self):
        result = compute_properties("not_a_smiles")
        assert result['valid'] is False

    def test_empty_string(self):
        result = compute_properties("")
        assert result['valid'] is False

    def test_aspirin(self):
        result = compute_properties("CC(=O)Oc1ccccc1C(=O)O")
        assert result['valid'] is True
        assert result['ring_count'] == 1
        assert result['qed'] is not None

    def test_benzene_ring(self):
        result = compute_properties("c1ccccc1")
        assert result['valid'] is True
        assert result['ring_count'] == 1


class TestSmilesToFP:
    def test_valid_smiles(self):
        fp = smiles_to_fp("CCO")
        assert fp is not None
        assert fp.shape == (1024,)
        assert fp.dtype == np.int8
        assert set(fp).issubset({0, 1})

    def test_invalid_smiles(self):
        assert smiles_to_fp("not_a_smiles") is None

    def test_custom_nbits(self):
        fp = smiles_to_fp("CCO", n_bits=512)
        assert fp is not None
        assert fp.shape == (512,)


class TestDetectSmilesColumn:
    def test_canonical_smiles(self):
        df = pd.DataFrame({"canonical_smiles": ["CCO"], "other": [1]})
        assert detect_smiles_column(df) == "canonical_smiles"

    def test_smiles_upper(self):
        df = pd.DataFrame({"SMILES": ["CCO"], "id": [1]})
        assert detect_smiles_column(df) == "SMILES"

    def test_not_found(self):
        df = pd.DataFrame({"mol_str": ["CCO"], "id": [1]})
        assert detect_smiles_column(df) is None


class TestDetectIdColumn:
    def test_identifier(self):
        df = pd.DataFrame({"identifier": ["A"], "smiles": ["CCO"]})
        assert detect_id_column(df) == "identifier"

    def test_np_id(self):
        df = pd.DataFrame({"np_id": ["A"], "smiles": ["CCO"]})
        assert detect_id_column(df) == "np_id"

    def test_not_found(self):
        df = pd.DataFrame({"name": ["A"], "smiles": ["CCO"]})
        assert detect_id_column(df) is None


class TestLoadAndFilter:
    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_and_filter("/nonexistent/path.csv")

    def test_invalid_max_atoms(self, tmp_csv):
        with pytest.raises(ValueError, match="Invalid max_atoms"):
            load_and_filter(tmp_csv, max_atoms=0)

    def test_invalid_sa_max(self, tmp_csv):
        with pytest.raises(ValueError, match="Invalid SA max"):
            load_and_filter(tmp_csv, sa_max=-1)

    def test_basic_load(self, tmp_csv):
        df, smiles_col = load_and_filter(tmp_csv, max_atoms=500, max_rings=50, sa_max=10.0)
        assert smiles_col == "canonical_smiles"
        assert len(df) > 0
