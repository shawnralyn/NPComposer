"""Tests for scripts/create_subset.py"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from create_subset import (
    tanimoto_distance_matrix,
    euclidean_distance_matrix,
    kmedoids_combined,
    calc_sa,
    calc_qed,
    calc_npl,
    smiles_to_fp,
    detect_smiles_column,
    detect_id_column,
    load_and_filter,
)


class TestTanimotoDistanceMatrix:
    def test_identity(self):
        X = np.eye(3, dtype=np.float64)
        D = tanimoto_distance_matrix(X)
        # diagonal should be 0 (distance to self)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(5, 10)).astype(np.float64)
        D = tanimoto_distance_matrix(X)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_range(self):
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(10, 20)).astype(np.float64)
        D = tanimoto_distance_matrix(X)
        assert D.min() >= 0.0
        assert D.max() <= 1.0

    def test_zero_vectors(self):
        X = np.zeros((3, 5), dtype=np.float64)
        D = tanimoto_distance_matrix(X)
        # union=0 → treated as 1 → distance = 1 - 0/1 = 1, but diag 0/0 → 0
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)


class TestEuclideanDistanceMatrix:
    def test_identity(self):
        X = np.eye(3, dtype=np.float64)
        D = euclidean_distance_matrix(X)
        np.testing.assert_array_almost_equal(np.diag(D), 0.0)

    def test_symmetry(self):
        rng = np.random.default_rng(42)
        X = rng.random((5, 3))
        D = euclidean_distance_matrix(X)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_normalized(self):
        rng = np.random.default_rng(42)
        X = rng.random((10, 5))
        D = euclidean_distance_matrix(X)
        assert D.max() <= 1.0 + 1e-10


class TestKmedoids:
    def test_returns_correct_count(self):
        rng = np.random.default_rng(42)
        X_fp = rng.integers(0, 2, size=(20, 32)).astype(np.float64)
        X_props = rng.random((20, 2))
        medoids = kmedoids_combined(X_fp, X_props, n_clusters=5, max_iter=10)
        assert len(medoids) == 5

    def test_medoids_are_unique(self):
        rng = np.random.default_rng(42)
        X_fp = rng.integers(0, 2, size=(30, 32)).astype(np.float64)
        X_props = rng.random((30, 2))
        medoids = kmedoids_combined(X_fp, X_props, n_clusters=10, max_iter=10)
        assert len(set(medoids)) == 10

    def test_medoids_in_range(self):
        rng = np.random.default_rng(42)
        n = 25
        X_fp = rng.integers(0, 2, size=(n, 16)).astype(np.float64)
        X_props = rng.random((n, 3))
        medoids = kmedoids_combined(X_fp, X_props, n_clusters=5, max_iter=10)
        assert all(0 <= m < n for m in medoids)


class TestCalcSA:
    def test_valid_smiles(self):
        score = calc_sa("CCO")
        assert score is not None
        assert 1.0 <= score <= 10.0

    def test_invalid_smiles(self):
        assert calc_sa("not_a_smiles") is None

    def test_empty_string(self):
        assert calc_sa("") is None


class TestCalcQED:
    def test_valid_smiles(self):
        score = calc_qed("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
        assert score is not None
        assert 0.0 <= score <= 1.0

    def test_invalid_smiles(self):
        assert calc_qed("XXXXX") is None


class TestCalcNPL:
    def test_valid_smiles(self):
        score = calc_npl("c1ccccc1")
        # NP_MODEL may not be available
        if score is not None:
            assert -5.0 <= score <= 5.0

    def test_invalid_smiles(self):
        assert calc_npl("not_a_smiles") is None


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
