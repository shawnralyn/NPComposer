"""Tests for src/evaluation/metrics.py"""

import pytest
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "evaluation"))

from metrics import evaluate


class TestEvaluate:
    def test_all_valid(self, valid_smiles):
        results = evaluate(valid_smiles)
        assert results["n_total"] == len(valid_smiles)
        assert results["n_valid"] == len(valid_smiles)
        assert results["validity"] == 1.0

    def test_all_invalid(self, invalid_smiles):
        results = evaluate(invalid_smiles)
        assert results["n_total"] == len(invalid_smiles)
        assert results["n_valid"] == 0
        assert results["validity"] == 0.0

    def test_mixed(self, valid_smiles, invalid_smiles):
        mixed = valid_smiles + invalid_smiles
        results = evaluate(mixed)
        assert results["n_total"] == len(mixed)
        assert results["n_valid"] == len(valid_smiles)

    def test_empty_list(self):
        results = evaluate([])
        assert results["n_total"] == 0
        assert results["validity"] == 0

    def test_sa_score_range(self, valid_smiles):
        results = evaluate(valid_smiles)
        assert results["sa_score"]["mean"] is not None
        assert 1.0 <= results["sa_score"]["mean"] <= 10.0

    def test_qed_range(self, valid_smiles):
        results = evaluate(valid_smiles)
        assert results["qed"]["mean"] is not None
        assert 0.0 <= results["qed"]["mean"] <= 1.0

    def test_result_keys(self, valid_smiles):
        results = evaluate(valid_smiles)
        assert "n_total" in results
        assert "n_valid" in results
        assert "validity" in results
        assert "sa_score" in results
        assert "qed" in results
        assert "np_score" in results

    def test_json_serializable(self, valid_smiles):
        results = evaluate(valid_smiles)
        # Should not raise
        json_str = json.dumps(results)
        parsed = json.loads(json_str)
        assert parsed["n_total"] == results["n_total"]
