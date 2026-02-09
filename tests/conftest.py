"""Shared fixtures for NPComposer tests."""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path


VALID_SMILES = [
    "CCO",                          # ethanol
    "c1ccccc1",                     # benzene
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "O=C(O)CC(O)(CC(=O)O)C(=O)O",  # citric acid
]

INVALID_SMILES = [
    "not_a_smiles",
    "",
    "XXXXX",
    "C(C)(C)(C)(C)(C)",  # too many bonds on C
]


@pytest.fixture
def valid_smiles():
    return VALID_SMILES.copy()


@pytest.fixture
def invalid_smiles():
    return INVALID_SMILES.copy()


@pytest.fixture
def sample_df():
    """Create a small DataFrame mimicking COCONUT/NPASS data."""
    return pd.DataFrame({
        "identifier": [f"NPC{i}" for i in range(20)],
        "canonical_smiles": [
            "CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O",
            "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "O=C(O)CC(O)(CC(=O)O)C(=O)O",
            "CC(=O)O", "CCCCCCCC", "c1ccc2ccccc2c1",
            "OC(=O)c1ccccc1", "CCCCCO",
            "c1ccc(cc1)O", "CC(O)CC", "CCC(=O)O",
            "c1ccc(cc1)N", "CCCCCCCCCCC",
            "OC(=O)CC(=O)O", "c1ccncc1", "CC(=O)N",
            "CCOC(=O)C", "c1ccc(cc1)Cl",
        ],
        "molecular_weight": [
            46.07, 78.11, 180.16, 206.28, 192.12,
            60.05, 114.23, 128.17, 122.12, 88.15,
            94.11, 74.12, 74.08, 93.13, 156.31,
            134.09, 79.10, 59.07, 88.11, 112.56,
        ],
    })


@pytest.fixture
def tmp_csv(sample_df, tmp_path):
    """Write sample_df to a temporary CSV and return the path."""
    path = tmp_path / "test_data.csv"
    sample_df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def tmp_smiles_file(valid_smiles, tmp_path):
    """Write valid SMILES to a temporary file (one per line)."""
    path = tmp_path / "smiles.txt"
    path.write_text("\n".join(valid_smiles) + "\n")
    return str(path)
