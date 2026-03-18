"""Tests for NP→drug data preprocessing utilities."""

import sys
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data_preprocessing" / "np_drug"))

from clean_coconut_npdrug import canonicalize, prepare, slim_and_rename, COCONUT_RENAME

OUTPUT_SCHEMA = {"canonical_smiles", "qed", "molecular_weight", "alogp", "tpsa", "hbd", "hba", "aromatic_rings_count"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def raw_coconut_df():
    """Minimal DataFrame mimicking raw COCONUT CSV columns."""
    return pd.DataFrame({
        "canonical_smiles":                ["CCO", "c1ccccc1", "CC(=O)O", "INVALID_SMILES", "CCO"],
        "qed_drug_likeliness":             [0.4,   0.6,        0.5,        0.3,               0.4],
        "molecular_weight":                [46.07, 78.11,      60.05,      100.0,             46.07],
        "alogp":                           [-0.3,  1.6,        -0.1,       1.0,               -0.3],
        "topological_polar_surface_area":  [20.2,  0.0,        37.3,       20.0,              20.2],
        "hydrogen_bond_donors_lipinski":   [1,     0,          1,          0,                 1],
        "hydrogen_bond_acceptors_lipinski":[1,     0,          2,          1,                 1],
        "aromatic_rings_count":            [0,     1,          0,          0,                 0],
    })


@pytest.fixture
def clean_coconut_df(raw_coconut_df):
    """COCONUT df after prepare + slim_and_rename."""
    df = prepare(raw_coconut_df, "canonical_smiles")
    return slim_and_rename(df)


@pytest.fixture
def minimal_pairs_df():
    """Minimal pairs CSV with required columns."""
    return pd.DataFrame({
        "coconut_smiles": ["CCO", "c1ccccc1", "CC(=O)O"],
        "chembl_smiles":  ["CCCO", "c1ccc(O)cc1", "CC(=O)OCC"],
        "tanimoto":       [0.42, 0.51, 0.38],
    })


# ---------------------------------------------------------------------------
# canonicalize()
# ---------------------------------------------------------------------------

class TestCanonicalize:
    def test_valid_smiles_returns_string(self):
        result = canonicalize("CCO")
        assert isinstance(result, str)
        assert Chem.MolFromSmiles(result) is not None

    def test_invalid_smiles_returns_none(self):
        assert canonicalize("INVALID") is None
        assert canonicalize("C(C)(C)(C)(C)(C)") is None

    def test_canonical_form_is_stable(self):
        assert canonicalize("OCC") == canonicalize("CCO")


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------

class TestPrepare:
    def test_drops_invalid_smiles(self, raw_coconut_df):
        result = prepare(raw_coconut_df, "canonical_smiles")
        assert "INVALID_SMILES" not in result["canonical_smiles"].values

    def test_drops_duplicates(self, raw_coconut_df):
        result = prepare(raw_coconut_df, "canonical_smiles")
        assert result["canonical_smiles"].duplicated().sum() == 0

    def test_all_output_smiles_valid(self, raw_coconut_df):
        result = prepare(raw_coconut_df, "canonical_smiles")
        for smi in result["canonical_smiles"]:
            assert Chem.MolFromSmiles(smi) is not None


# ---------------------------------------------------------------------------
# slim_and_rename()
# ---------------------------------------------------------------------------

class TestSlimAndRename:
    def test_output_schema(self, clean_coconut_df):
        assert OUTPUT_SCHEMA.issubset(set(clean_coconut_df.columns))

    def test_columns_renamed(self, clean_coconut_df):
        for old_name in COCONUT_RENAME:
            assert old_name not in clean_coconut_df.columns
        for new_name in COCONUT_RENAME.values():
            assert new_name in clean_coconut_df.columns

    def test_no_unexpected_columns(self, clean_coconut_df):
        extra = set(clean_coconut_df.columns) - OUTPUT_SCHEMA - {"sa_score"}
        assert extra == set()


# ---------------------------------------------------------------------------
# Pairs CSV schema
# ---------------------------------------------------------------------------

class TestPairsCsvSchema:
    def test_required_columns_present(self, minimal_pairs_df):
        for col in ["coconut_smiles", "chembl_smiles", "tanimoto"]:
            assert col in minimal_pairs_df.columns

    def test_tanimoto_range(self, minimal_pairs_df):
        assert (minimal_pairs_df["tanimoto"] >= 0).all()
        assert (minimal_pairs_df["tanimoto"] <= 1).all()

    def test_smiles_parseable(self, minimal_pairs_df):
        for col in ["coconut_smiles", "chembl_smiles"]:
            for smi in minimal_pairs_df[col]:
                assert Chem.MolFromSmiles(smi) is not None
