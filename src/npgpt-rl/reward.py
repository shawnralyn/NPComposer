"""Multi-objective reward functions for RL-based molecular optimization of natural products."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, QED

# Suppress RDKit warnings during batch scoring
RDLogger.logger().setLevel(RDLogger.ERROR)

_sa_score_module = None


def _lazy_load_sa_score():
    """Lazily load SA score calculator."""
    global _sa_score_module
    if _sa_score_module is not None:
        return _sa_score_module

    try:
        from rdkit.Contrib.SA_Score import sascorer

        _sa_score_module = sascorer
    except ImportError:
        try:
            import sascorer  # type: ignore

            _sa_score_module = sascorer
        except ImportError:
            warnings.warn(
                "SA Score module not found. Install rdkit-contrib or place sascorer.py "
                "on your PYTHONPATH. SA reward will return 0.0 for all molecules."
            )
            _sa_score_module = None
    return _sa_score_module


def sa_score(mol: Chem.Mol) -> float:
    """Return SA score [1, 10] (1=easy to synthesize)."""
    scorer = _lazy_load_sa_score()
    if scorer is None:
        return 5.0  # neutral fallback
    return scorer.calculateScore(mol)


def sa_score_normalized(mol: Chem.Mol) -> float:
    """Normalize SA score to [0, 1] (1=easiest to synthesize)."""
    raw = sa_score(mol)
    return (10.0 - raw) / 9.0


_np_score_module = None


def _lazy_load_np_score():
    """Lazily load NP-likeness score calculator."""
    global _np_score_module
    if _np_score_module is not None:
        return _np_score_module

    try:
        from rdkit.Contrib.NP_Score import npscorer

        _np_score_module = npscorer
        # Load the NP model (fingerprint-based)
        _np_score_module._fscore = npscorer.readNPModel()
    except (ImportError, Exception):
        try:
            import npscorer  # type: ignore

            _np_score_module = npscorer
            _np_score_module._fscore = npscorer.readNPModel()
        except (ImportError, Exception):
            warnings.warn(
                "NP Score module not found. NP-likeness reward will return 0.0."
            )
            _np_score_module = None
    return _np_score_module


def np_likeness_score(mol: Chem.Mol) -> float:
    """Return NP-likeness score [-5, 5] (higher=more NP-like)."""
    scorer = _lazy_load_np_score()
    if scorer is None:
        return 0.0
    return scorer.scoreMol(mol, scorer._fscore)


def np_likeness_normalized(mol: Chem.Mol) -> float:
    """Normalize NP-likeness score to [0, 1] using sigmoid mapping."""
    raw = np_likeness_score(mol)
    return 1.0 / (1.0 + np.exp(-raw))


def validity_score(smiles_list: list[str]) -> list[float]:
    """Return 1.0 if valid SMILES, 0.0 otherwise."""
    return [1.0 if Chem.MolFromSmiles(smi) is not None else 0.0 for smi in smiles_list]


def qed_score(smiles_list: list[str]) -> list[float]:
    """Return QED score in [0, 1]."""
    rewards = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            rewards.append(QED.qed(mol))
        else:
            rewards.append(0.0)
    return rewards


def sa_reward(smiles_list: list[str]) -> list[float]:
    """Return normalized SA score in [0, 1] (higher = easier to synthesize)."""
    rewards = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            rewards.append(sa_score_normalized(mol))
        else:
            rewards.append(0.0)
    return rewards


def np_likeness_reward(smiles_list: list[str]) -> list[float]:
    """Return normalized NP-likeness score in [0, 1]."""
    rewards = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            rewards.append(np_likeness_normalized(mol))
        else:
            rewards.append(0.0)
    return rewards


# ---------------------------------------------------------------------------
# Multi-objective reward
# ---------------------------------------------------------------------------


@dataclass
class RewardConfig:
    """Configuration for multi-objective reward weighting."""

    w_validity: float = 1.0
    w_qed: float = 0.3
    w_sa: float = 0.3
    w_np_likeness: float = 0.4
    invalid_penalty: float = -0.5


class MultiObjectiveReward:
    """Multi-objective reward function for molecular generation.

    Computes weighted sum of validity, QED, SA, and NP-likeness rewards.
    """

    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def __call__(self, smiles_list: list[str]) -> list[float]:
        return self.score(smiles_list)

    def score(self, smiles_list: list[str]) -> list[float]:
        """Score batch of SMILES strings."""
        cfg = self.config
        rewards = []

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                rewards.append(cfg.invalid_penalty)
                continue

            r = cfg.w_validity * 1.0  # valid molecule
            r += cfg.w_qed * QED.qed(mol)
            r += cfg.w_sa * sa_score_normalized(mol)
            r += cfg.w_np_likeness * np_likeness_normalized(mol)

            # Normalize by total weight so reward stays roughly in [0, 1]
            total_weight = cfg.w_validity + cfg.w_qed + cfg.w_sa + cfg.w_np_likeness
            r /= total_weight

            rewards.append(r)

        return rewards

    def score_detailed(
        self, smiles_list: list[str]
    ) -> dict[str, list[float]]:
        """Return per-component scores."""
        results = {
            "smiles": smiles_list,
            "validity": [],
            "qed": [],
            "sa": [],
            "np_likeness": [],
            "total": [],
        }

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results["validity"].append(0.0)
                results["qed"].append(0.0)
                results["sa"].append(0.0)
                results["np_likeness"].append(0.0)
                results["total"].append(self.config.invalid_penalty)
            else:
                v = 1.0
                q = QED.qed(mol)
                s = sa_score_normalized(mol)
                n = np_likeness_normalized(mol)
                results["validity"].append(v)
                results["qed"].append(q)
                results["sa"].append(s)
                results["np_likeness"].append(n)

                cfg = self.config
                total = cfg.w_validity * v + cfg.w_qed * q + cfg.w_sa * s + cfg.w_np_likeness * n
                total /= cfg.w_validity + cfg.w_qed + cfg.w_sa + cfg.w_np_likeness
                results["total"].append(total)

        return results

# ---------------------------------------------------------------------------
# Individual reward components (operate on SMILES strings)
# ---------------------------------------------------------------------------
