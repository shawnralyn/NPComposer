"""Evaluate molecular generation models (NPGPT, GP-MoLFormer) on validity, QED, SA, NP-likeness, diversity, and optionally uniqueness/novelty."""

import sys
import os
import math
import warnings
import gc
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import QED as QED_module, AllChem, DataStructs
from rdkit.Contrib.SA_Score import sascorer

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

try:
    from rdkit.Contrib.NP_Score import npscorer
    NP_MODEL = npscorer.readNPModel()
except (ImportError, FileNotFoundError, OSError):
    NP_MODEL = None


def _load_npgpt(ckpt_path, tokenizer_path, is_rl_ckpt=False):
    """Load NPGPT checkpoint (pretrained or RL).

    Input:
        ckpt_path: path to checkpoint.
        tokenizer_path: path to tokenizer.json.
        is_rl_ckpt: if True, load as RL checkpoint.
    Output:
        model, tokenizer."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "externals" / "smiles-gpt"))
    from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer

    cfg = SmilesGptTrainingConfig()
    tok = get_tokenizer(cfg, tokenizer_path)
    if is_rl_ckpt:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = SmilesGptModel(config=cfg, tokenizer=tok)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model = SmilesGptModel.load_from_checkpoint(
            ckpt_path, config=cfg, tokenizer=tok, strict=False
        )
    model.eval()
    return model, tok


def _generate_npgpt(model, tokenizer, n=50, batch=20, temperature=1.5, top_p=1.0):
    """Generate SMILES using NPGPT.

    Input:
        model, tokenizer: NPGPT model and tokenizer.
        n: total molecules to generate.
        batch, temperature, top_p: generation parameters.
    Output:
        list of SMILES strings."""
    device = torch.device("cpu")
    model.to(device)
    smiles = []
    with torch.no_grad():
        for _ in range(math.ceil(n / batch)):
            bs = min(batch, n - len(smiles))
            ids = torch.tensor([[tokenizer.bos_token_id]] * bs).to(device)
            out = model.model.generate(
                ids, max_length=512, do_sample=True,
                temperature=temperature, top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            smiles.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return smiles[:n]


def _load_gpmolformer(model_name_or_path):
    """Load GP-MoLFormer from HuggingFace.

    Input:
        model_name_or_path: HuggingFace model identifier or local path.
    Output:
        model, tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def _generate_gpmolformer(model, tokenizer, n=50, batch=20, temperature=1.0, **kwargs):
    """Generate SMILES using GP-MoLFormer.

    Input:
        model, tokenizer: GP-MoLFormer model and tokenizer.
        n: total molecules to generate.
        batch, temperature: generation parameters.
    Output:
        list of SMILES strings."""
    device = next(model.parameters()).device
    smiles = []
    with torch.no_grad():
        for _ in range(math.ceil(n / batch)):
            bs = min(batch, n - len(smiles))
            out = model.generate(
                do_sample=True,
                temperature=temperature,
                top_k=None,
                max_length=model.config.max_position_embeddings,
                num_return_sequences=bs,
            )
            smiles.extend(tokenizer.batch_decode(out.cpu(), skip_special_tokens=True))
    return smiles[:n]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sanitize_smiles(smi):
    """Sanitize SMILES and categorize errors.

    Input:
        smi: SMILES string.
    Output:
        (mol_or_None, error_category: 'valid'|'parse'|'valence'|'kekulize'|'other')."""
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None, "parse"
    try:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)
        return mol, "valid"
    except Exception as e:
        err = str(e).lower()
        if "valence" in err:
            return None, "valence"
        elif "kekul" in err:
            return None, "kekulize"
        else:
            return None, "other"


def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_internal_diversity(fps):
    """Compute pairwise Tanimoto similarity.

    Input:
        fps: list of fingerprints.
    Output:
        dict with mean and std of pairwise similarities."""
    if len(fps) < 2:
        return {"mean": None, "std": None}
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return {
        "mean": round(float(np.mean(sims)), 4),
        "std": round(float(np.std(sims)), 4),
    }


def compute_uniqueness(valid_smiles, train_set):
    valid_set = set(valid_smiles)
    unique = valid_set - train_set
    n = len(valid_smiles)
    return {
        "n_unique": len(unique),
        "n_total": n,
        "ratio": round(len(unique) / n, 4) if n else 0.0,
    }


def compute_novelty(fps_valid, fps_ref, n_valid, threshold=0.6):
    if not fps_valid or not fps_ref:
        return {"n_novel": 0, "n_total": n_valid, "ratio": 0.0, "nn_sim_mean": None}
    nn_sims = []
    for fp in fps_valid:
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
        nn_sims.append(max(sims))
    n_novel = sum(1 for s in nn_sims if s < threshold)
    return {
        "n_novel": n_novel,
        "n_total": n_valid,
        "ratio": round(n_novel / n_valid, 4) if n_valid else 0.0,
        "threshold": threshold,
        "nn_sim_mean": round(float(np.mean(nn_sims)), 4),
        "nn_sim_std": round(float(np.std(nn_sims)), 4),
    }


def evaluate_smiles(smiles_list, train_set=None, ref_fps=None):
    """Compute validity, QED, SA, NP-likeness, diversity, and optionally uniqueness/novelty.

    Input:
        smiles_list: list of SMILES strings.
        train_set: set of training SMILES (for uniqueness).
        ref_fps: reference fingerprints (for novelty).
    Output:
        dict with metrics and raw arrays."""
    n_total = len(smiles_list)
    valid_smiles = []
    sa_scores, qed_scores, np_scores = [], [], []
    validity_arr = []
    error_counts = Counter()

    for smi in smiles_list:
        mol, err = sanitize_smiles(smi)
        error_counts[err] += 1
        if mol is not None:
            validity_arr.append(1)
            valid_smiles.append(smi)
            qed_scores.append(QED_module.qed(mol))
            sa_scores.append(sascorer.calculateScore(mol))
            if NP_MODEL:
                try:
                    np_scores.append(npscorer.scoreMol(mol, NP_MODEL))
                except (ValueError, ZeroDivisionError):
                    pass
        else:
            validity_arr.append(0)

    n_valid = len(valid_smiles)
    validity = round(n_valid / n_total, 4) if n_total else 0.0

    results = {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity": validity,
        "qed": {
            "mean": round(float(np.mean(qed_scores)), 4) if qed_scores else 0.0,
            "std": round(float(np.std(qed_scores)), 4) if qed_scores else 0.0,
        },
        "sa": {
            "mean": round(float(np.mean(sa_scores)), 4) if sa_scores else 0.0,
            "std": round(float(np.std(sa_scores)), 4) if sa_scores else 0.0,
        },
        "np_likeness": {
            "mean": round(float(np.mean(np_scores)), 4) if np_scores else 0.0,
            "std": round(float(np.std(np_scores)), 4) if np_scores else 0.0,
        } if NP_MODEL else None,
        "sanitize_breakdown": dict(error_counts),
        "_qed_scores": qed_scores,
        "_sa_scores": sa_scores,
        "_np_scores": np_scores,
        "_validity_arr": validity_arr,
    }

    fps_valid = [smiles_to_fp(s) for s in valid_smiles]
    fps_valid = [fp for fp in fps_valid if fp is not None]

    if len(fps_valid) > 1:
        results["internal_diversity"] = compute_internal_diversity(fps_valid)
    if train_set is not None and valid_smiles:
        results["uniqueness"] = compute_uniqueness(valid_smiles, train_set)
    if ref_fps is not None and fps_valid:
        results["novelty"] = compute_novelty(fps_valid, ref_fps, n_valid)

    return results


def load_training_smiles(csv_path):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "canonical_smiles" in df.columns:
        smiles = df["canonical_smiles"].dropna().astype(str).tolist()
    elif "smiles" in df.columns:
        smiles = df["smiles"].dropna().astype(str).tolist()
    else:
        smiles = df.iloc[:, 0].dropna().astype(str).tolist()
    print(f"  Loaded {len(smiles)} training SMILES from {csv_path}")
    return set(smiles)


def load_ref_fingerprints(csv_path, max_n=10000):
    """Load reference fingerprints for novelty computation.

    Input:
        csv_path: CSV file with SMILES.
        max_n: max fingerprints to load.
    Output:
        list of Morgan fingerprints."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "canonical_smiles" in df.columns:
        smiles = df["canonical_smiles"].dropna().astype(str).tolist()
    elif "smiles" in df.columns:
        smiles = df["smiles"].dropna().astype(str).tolist()
    else:
        smiles = df.iloc[:, 0].dropna().astype(str).tolist()
    if len(smiles) > max_n:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(smiles), max_n, replace=False)
        smiles = [smiles[i] for i in idx]
        print(f"  Subsampled {max_n} for novelty reference")
    fps = []
    for s in tqdm(smiles, desc="  Reference FPs", leave=False):
        fp = smiles_to_fp(s)
        if fp is not None:
            fps.append(fp)
    print(f"  Computed {len(fps)} reference fingerprints")
    return fps


def _strip_raw(m):
    """Remove raw arrays before JSON export.

    Input:
        m: results dict.
    Output:
        dict without raw score arrays."""
    out = dict(m)
    for k in ["_qed_scores", "_sa_scores", "_np_scores", "_validity_arr", "smiles"]:
        out.pop(k, None)
    return out


def _merge_scores_single(all_results, score_key, seeds):
    merged = []
    for seed in seeds:
        merged.extend(all_results[seed].get(score_key, []))
    return np.array(merged)


def _merge_sanitize_single(all_results, seeds):
    merged = Counter()
    for seed in seeds:
        sb = all_results[seed].get("sanitize_breakdown", {})
        for k, v in sb.items():
            merged[k] += v
    return merged


def _merge_scores_multi(all_results, model_name, score_key, seeds):
    merged = []
    for seed in seeds:
        merged.extend(all_results[model_name][seed].get(score_key, []))
    return np.array(merged)


def _merge_sanitize_multi(all_results, model_name, seeds):
    merged = Counter()
    for seed in seeds:
        sb = all_results[model_name][seed].get("sanitize_breakdown", {})
        for k, v in sb.items():
            merged[k] += v
    return merged


def _get_metric_val(m, key):
    """Extract scalar metric value from per-seed results.

    Input:
        m: per-seed metrics dict.
        key: metric name.
    Output:
        scalar value or None."""
    if key == "validity":
        return m["validity"]
    elif key in ("qed", "sa"):
        return m[key]["mean"]
    elif key == "np_likeness" and m.get("np_likeness"):
        return m["np_likeness"]["mean"]
    elif key == "uniqueness" and "uniqueness" in m:
        return m["uniqueness"]["ratio"]
    elif key == "novelty" and "novelty" in m:
        return m["novelty"]["ratio"]
    elif key == "internal_diversity" and "internal_diversity" in m:
        v = m["internal_diversity"]["mean"]
        return v if v is not None else None
    return None


def _aggregate_metric(all_results_dict, seeds, key):
    """Aggregate metric across seeds.

    Input:
        all_results_dict: {seed: metrics} dict.
        seeds: list of seed values.
        key: metric name.
    Output:
        dict with mean, std, values or None."""
    vals = []
    for seed in seeds:
        v = _get_metric_val(all_results_dict[seed], key)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return {
        "mean": round(float(np.mean(vals)), 4),
        "std": round(float(np.std(vals)), 4),
        "values": vals,
    }


def _print_seed_summary(metrics):
    parts = [f"Validity: {metrics['validity']*100:.1f}%"]
    parts.append(f"QED: {metrics['qed']['mean']:.4f}")
    parts.append(f"SA: {metrics['sa']['mean']:.4f}")
    if metrics.get("np_likeness"):
        parts.append(f"NP: {metrics['np_likeness']['mean']:.4f}")
    print(f"    {', '.join(parts)}")
    extra = []
    if "uniqueness" in metrics:
        extra.append(f"Uniqueness: {metrics['uniqueness']['ratio']*100:.1f}%")
    if "novelty" in metrics:
        extra.append(f"Novelty: {metrics['novelty']['ratio']*100:.1f}%")
    if "internal_diversity" in metrics and metrics["internal_diversity"]["mean"] is not None:
        extra.append(f"IntDiv: {metrics['internal_diversity']['mean']:.4f}")
    if extra:
        print(f"    {', '.join(extra)}")


def _draw_pie(ax, sb, valid_color):
    color_map = {
        "valid": valid_color, "parse": "#FFB74D", "valence": "#E57373",
        "kekulize": "#BA68C8", "other": "#90A4AE",
    }
    labels_pie, sizes, colors_pie = [], [], []
    for cat in ["valid", "parse", "valence", "kekulize", "other"]:
        if sb.get(cat, 0) > 0:
            labels_pie.append(cat)
            sizes.append(sb[cat])
            colors_pie.append(color_map.get(cat, "#90A4AE"))
    if sizes:
        _, _, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors_pie, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10},
        )
        for at in autotexts:
            at.set_fontweight("bold")


def plot_single_histogram(all_results, seeds, title, color, out_path):
    """Plot 2x3 histogram of metrics for single model.

    Input:
        all_results: {seed: metrics} dict.
        seeds: list of seed values.
        title, color: plot title and bar color.
        out_path: output file path.
    Output:
        none (saves PNG)."""
    qed_all = _merge_scores_single(all_results, "_qed_scores", seeds)
    sa_all = _merge_scores_single(all_results, "_sa_scores", seeds)
    np_all = _merge_scores_single(all_results, "_np_scores", seeds)
    val_all = _merge_scores_single(all_results, "_validity_arr", seeds)
    sb = _merge_sanitize_single(all_results, seeds)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

    # (0,0) Validity bar
    ax = axes[0, 0]
    v_pct = float(val_all.mean()) * 100 if len(val_all) else 0
    bar = ax.bar(["Validity"], [v_pct], color=color, width=0.4, edgecolor="white")
    ax.text(bar[0].get_x() + bar[0].get_width()/2, bar[0].get_height() + 1,
            f"{v_pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 110); ax.set_ylabel("Valid (%)")
    ax.set_title("Validity (sanitized)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # (0,1) QED
    ax = axes[0, 1]
    if len(qed_all):
        ax.hist(qed_all, bins=np.linspace(0, 1, 21), alpha=0.75, color=color, edgecolor="white")
        ax.axvline(qed_all.mean(), color="#333", ls="--", lw=2, label=f"mean={qed_all.mean():.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("QED", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # (0,2) SA
    ax = axes[0, 2]
    if len(sa_all):
        lo, hi = max(1, float(sa_all.min()) - 0.5), min(10, float(sa_all.max()) + 0.5)
        ax.hist(sa_all, bins=np.linspace(lo, hi, 21), alpha=0.75, color=color, edgecolor="white")
        ax.axvline(sa_all.mean(), color="#333", ls="--", lw=2, label=f"mean={sa_all.mean():.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("SA Score (raw, lower=better)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # (1,0) NP-likeness
    ax = axes[1, 0]
    if len(np_all):
        lo, hi = float(np_all.min()) - 0.5, float(np_all.max()) + 0.5
        ax.hist(np_all, bins=np.linspace(lo, hi, 21), alpha=0.75, color=color, edgecolor="white")
        ax.axvline(np_all.mean(), color="#333", ls="--", lw=2, label=f"mean={np_all.mean():.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("NP-likeness (raw)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    # (1,1) Sanitization pie
    ax = axes[1, 1]
    _draw_pie(ax, sb, color)
    ax.set_title("Error Breakdown", fontsize=12, fontweight="bold")

    # (1,2) Summary
    ax = axes[1, 2]
    ax.axis("off")
    n_total = sum(all_results[s]["n_total"] for s in seeds)
    all_val = [all_results[s]["validity"] for s in seeds]
    all_qed = [all_results[s]["qed"]["mean"] for s in seeds]
    all_sa = [all_results[s]["sa"]["mean"] for s in seeds]
    lines = [
        f"Total molecules: {n_total}",
        f"Validity: {np.mean(all_val)*100:.1f}% +/- {np.std(all_val)*100:.1f}%",
        f"QED: {np.mean(all_qed):.4f} +/- {np.std(all_qed):.4f}",
        f"SA: {np.mean(all_sa):.4f} +/- {np.std(all_sa):.4f}",
    ]
    all_np = [all_results[s]["np_likeness"]["mean"] for s in seeds if all_results[s].get("np_likeness")]
    if all_np:
        lines.append(f"NP: {np.mean(all_np):.4f} +/- {np.std(all_np):.4f}")
    all_div = [all_results[s]["internal_diversity"]["mean"] for s in seeds
               if "internal_diversity" in all_results[s] and all_results[s]["internal_diversity"]["mean"] is not None]
    if all_div:
        lines.append(f"Int. Diversity: {np.mean(all_div):.4f} +/- {np.std(all_div):.4f}")
    ax.text(0.1, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=12, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Summary", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved -> {out_path}")



def plot_comparison_histogram(all_results, seeds, title, out_path):
    """Plot 2x3 histogram comparing two models.

    Input:
        all_results: {model_name: {seed: metrics}} dict.
        seeds: list of seed values.
        title: plot title.
        out_path: output file path.
    Output:
        none (saves PNG)."""
    c_orig, c_rl = "#4C8BF5", "#F5534C"

    qed_o = _merge_scores_multi(all_results, "pretrained", "_qed_scores", seeds)
    qed_r = _merge_scores_multi(all_results, "rl_finetuned", "_qed_scores", seeds)
    sa_o  = _merge_scores_multi(all_results, "pretrained", "_sa_scores", seeds)
    sa_r  = _merge_scores_multi(all_results, "rl_finetuned", "_sa_scores", seeds)
    np_o  = _merge_scores_multi(all_results, "pretrained", "_np_scores", seeds)
    np_r  = _merge_scores_multi(all_results, "rl_finetuned", "_np_scores", seeds)
    val_o = _merge_scores_multi(all_results, "pretrained", "_validity_arr", seeds)
    val_r = _merge_scores_multi(all_results, "rl_finetuned", "_validity_arr", seeds)
    sb_o  = _merge_sanitize_multi(all_results, "pretrained", seeds)
    sb_r  = _merge_sanitize_multi(all_results, "rl_finetuned", seeds)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

    sa_all = np.concatenate([sa_o, sa_r]) if (len(sa_o) or len(sa_r)) else np.array([])
    np_all_merged = np.concatenate([np_o, np_r]) if (len(np_o) or len(np_r)) else np.array([])
    sa_range = (max(1, float(sa_all.min()) - 0.5), min(10, float(sa_all.max()) + 0.5)) if len(sa_all) else (1, 10)
    np_range = (float(np_all_merged.min()) - 0.5, float(np_all_merged.max()) + 0.5) if len(np_all_merged) else (-2, 3)

    titles = ["Validity (sanitized)", "QED", "SA Score (lower=better)",
              "NP-likeness", "Error: Pretrained", "Error: RL-finetuned"]
    hist_data = [
        (val_o, val_r, "bar", None),
        (qed_o, qed_r, "hist", (0, 1)),
        (sa_o,  sa_r,  "hist", sa_range),
        (np_o,  np_r,  "hist", np_range),
        (sb_o,  None,  "pie_orig", None),
        (sb_r,  None,  "pie_rl", None),
    ]

    for idx, (ax, ttl) in enumerate(zip(axes.flat, titles)):
        d1, d2, mode, rng = hist_data[idx]

        if mode == "bar":
            vo = float(d1.mean()) * 100 if len(d1) else 0
            vr = float(d2.mean()) * 100 if len(d2) else 0
            bars = ax.bar(["Pretrained", "RL-finetuned"], [vo, vr],
                          color=[c_orig, c_rl], width=0.5, edgecolor="white")
            for bar, val in zip(bars, [vo, vr]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
            ax.set_ylim(0, 110); ax.set_ylabel("Valid (%)")
            ax.spines[["top", "right"]].set_visible(False)

        elif mode == "hist":
            bins = np.linspace(rng[0], rng[1], 21)
            ax.hist(d1, bins=bins, alpha=0.6, color=c_orig, edgecolor="white", label="Pretrained")
            ax.hist(d2, bins=bins, alpha=0.6, color=c_rl, edgecolor="white", label="RL-finetuned")
            if len(d1): ax.axvline(d1.mean(), color=c_orig, ls="--", lw=1.5)
            if len(d2): ax.axvline(d2.mean(), color=c_rl, ls="--", lw=1.5)
            ax.set_ylabel("Count"); ax.legend(fontsize=9)
            ax.spines[["top", "right"]].set_visible(False)

        else:
            sb = d1
            vc = c_orig if "orig" in mode else c_rl
            _draw_pie(ax, sb, vc)

        ax.set_title(ttl, fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Histogram saved -> {out_path}")


NPGPT_RL_DIR = PROJECT_ROOT / "src" / "npgpt-rl"
DEFAULT_TOKENIZER = str(PROJECT_ROOT / "external/npgpt/externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json")


def run_single_model(args, model_type):
    """Evaluate single model and generate results.

    Input:
        args: parsed arguments.
        model_type: 'npgpt' or 'gpmolformer'.
    Output:
        none (saves JSON and histogram)."""
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ref data if provided
    train_set, ref_fps = None, None
    if args.training and Path(args.training).exists():
        print("Loading training data for uniqueness/novelty...")
        train_set = load_training_smiles(args.training)
        ref_fps = load_ref_fingerprints(args.training, max_n=args.novelty_max_ref)

    # Load model
    if model_type == "npgpt":
        print(f"\nLoading NPGPT pretrained: {args.orig_ckpt}")
        model, tok = _load_npgpt(args.orig_ckpt, args.tokenizer)
        gen_fn = lambda n, seed: (set_seed(seed), _generate_npgpt(model, tok, n, temperature=args.temperature, top_p=args.top_p))[1]
        color = "#4C8BF5"
        label = "NPGPT Pretrained"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nLoading GP-MoLFormer: {args.gpmolformer_model}")
        model, tok = _load_gpmolformer(args.gpmolformer_model)
        model = model.to(device)
        gen_fn = lambda n, seed: (set_seed(seed), _generate_gpmolformer(model, tok, n, temperature=args.temperature))[1]
        color = "#2ECC71"
        label = "GP-MoLFormer"

    all_results = {}
    for seed in args.seeds:
        print(f"\n  Seed {seed}: generating {args.n_samples} molecules...")
        smiles = gen_fn(args.n_samples, seed)
        metrics = evaluate_smiles(smiles, train_set=train_set, ref_fps=ref_fps)
        metrics["seed"] = seed
        metrics["smiles"] = smiles
        all_results[seed] = metrics
        _print_seed_summary(metrics)

    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Aggregate
    metric_keys = ["validity", "qed", "sa", "np_likeness", "internal_diversity"]
    if train_set: metric_keys += ["uniqueness", "novelty"]

    aggregate = {}
    print(f"\n{'='*50}")
    print(f"AGGREGATE: {label} (seeds={args.seeds}, n={args.n_samples}/seed, temp={args.temperature})")
    print(f"{'='*50}")
    print(f"{'Metric':<22} {'Value':>20}")
    print("-" * 42)
    for key in metric_keys:
        agg = _aggregate_metric(all_results, args.seeds, key)
        if agg:
            aggregate[key] = agg
            if key == "validity":
                print(f"{key:<22} {agg['mean']*100:>8.1f}% +/-{agg['std']*100:>5.1f}%")
            else:
                print(f"{key:<22} {agg['mean']:>8.4f} +/-{agg['std']:>6.4f}")
    print("=" * 42)

    # JSON
    prefix = model_type.replace("-", "")
    json_results = {
        "model": label,
        "config": {"seeds": args.seeds, "n_samples": args.n_samples, "temperature": args.temperature},
        "aggregate": aggregate,
        "per_seed": {s: _strip_raw(all_results[s]) for s in args.seeds},
    }
    json_path = OUT_DIR / f"{prefix}_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON -> {json_path}")

    # Histogram
    n_total = args.n_samples * len(args.seeds)
    plot_single_histogram(
        all_results, args.seeds,
        f"{label}  ({n_total} molecules, seeds={args.seeds}, temp={args.temperature})",
        color, OUT_DIR / f"{prefix}_histogram.png"
    )


def run_npgpt_rl(args):
    """Evaluate NPGPT pretrained vs RL-finetuned comparison.

    Input:
        args: parsed arguments.
    Output:
        none (saves JSON and histogram)."""
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_set, ref_fps = None, None
    if args.training and Path(args.training).exists():
        print("Loading training data for uniqueness/novelty...")
        train_set = load_training_smiles(args.training)
        ref_fps = load_ref_fingerprints(args.training, max_n=args.novelty_max_ref)

    all_results = {"pretrained": {}, "rl_finetuned": {}}

    for model_name, ckpt_path, is_rl in [
        ("pretrained", args.orig_ckpt, False),
        ("rl_finetuned", args.rl_ckpt, True),
    ]:
        print(f"\n{'='*60}")
        print(f"Loading {model_name} model...")
        model, tok = _load_npgpt(ckpt_path, args.tokenizer, is_rl_ckpt=is_rl)

        for seed in args.seeds:
            print(f"\n  Seed {seed}: generating {args.n_samples} molecules...")
            set_seed(seed)
            smiles = _generate_npgpt(model, tok, args.n_samples,
                                     temperature=args.temperature, top_p=args.top_p)
            metrics = evaluate_smiles(smiles, train_set=train_set, ref_fps=ref_fps)
            metrics["seed"] = seed
            metrics["smiles"] = smiles
            all_results[model_name][seed] = metrics
            _print_seed_summary(metrics)

        del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Aggregate
    metric_keys = ["validity", "qed", "sa", "np_likeness", "internal_diversity"]
    if train_set: metric_keys += ["uniqueness", "novelty"]

    aggregate = {"pretrained": {}, "rl_finetuned": {}}
    print(f"\n{'='*62}")
    print(f"AGGREGATE (seeds={args.seeds}, n={args.n_samples}/seed, temp={args.temperature}, top_p={args.top_p})")
    print(f"{'='*62}")
    print(f"{'Metric':<22} {'Pretrained':>20} {'RL-finetuned':>20}")
    print("-" * 62)

    for key in metric_keys:
        for mn in ["pretrained", "rl_finetuned"]:
            agg = _aggregate_metric(all_results[mn], args.seeds, key)
            if agg:
                aggregate[mn][key] = agg
        p = aggregate["pretrained"].get(key)
        r = aggregate["rl_finetuned"].get(key)
        if p and r:
            if key == "validity":
                print(f"{key:<22} {p['mean']*100:>8.1f}% +/-{p['std']*100:>5.1f}%"
                      f"  {r['mean']*100:>8.1f}% +/-{r['std']*100:>5.1f}%")
            else:
                print(f"{key:<22} {p['mean']:>8.4f} +/-{p['std']:>6.4f}"
                      f"  {r['mean']:>8.4f} +/-{r['std']:>6.4f}")
    print("=" * 62)

    # JSON
    json_results = {
        "model": "NPGPT pretrained vs RL-finetuned",
        "config": {
            "seeds": args.seeds, "n_samples": args.n_samples,
            "temperature": args.temperature, "top_p": args.top_p,
            "orig_ckpt": args.orig_ckpt, "rl_ckpt": args.rl_ckpt,
        },
        "aggregate": aggregate,
        "per_seed": {
            mn: {s: _strip_raw(all_results[mn][s]) for s in args.seeds}
            for mn in ["pretrained", "rl_finetuned"]
        },
    }
    json_path = OUT_DIR / "npgpt_rl_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"\nJSON -> {json_path}")

    n_total = args.n_samples * len(args.seeds)
    plot_comparison_histogram(
        all_results, args.seeds,
        f"NPGPT: Pretrained vs RL-finetuned  "
        f"({n_total} molecules, seeds={args.seeds}, temp={args.temperature}, top_p={args.top_p})",
        OUT_DIR / "npgpt_rl_histogram.png"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Unified molecular generation evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py npgpt
  python evaluate.py npgpt-rl --seeds 1 2 3 --n_samples 50
  python evaluate.py gpmolformer --temperature 1.0
  python evaluate.py npgpt-rl --training ../../data/processed/coconut_100000.csv
        """,
    )
    parser.add_argument("model", choices=["npgpt", "npgpt-rl", "gpmolformer"],
                        help="Model to evaluate")

    # Common args
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_samples", type=int, default=50, help="Molecules per seed")
    parser.add_argument("--temperature", type=float, default=1.5, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (NPGPT only)")
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "results" / "evaluation"))

    # NPGPT args
    parser.add_argument("--orig_ckpt", type=str, default=str(NPGPT_RL_DIR / "npgpt.ckpt"))
    parser.add_argument("--rl_ckpt", type=str, default=str(NPGPT_RL_DIR / "npgpt_rl_step_600.ckpt"))
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)

    # GP-MoLFormer args
    parser.add_argument("--gpmolformer_model", type=str, default="ibm-research/GP-MoLFormer-Uniq",
                        help="HuggingFace model name or local path")

    # Uniqueness / novelty (npgpt, npgpt-rl only; ignored for gpmolformer)
    parser.add_argument("--training", type=str,
                        default=str(PROJECT_ROOT / "data/processed/coconut_100000.csv"),
                        help="Training CSV for uniqueness/novelty (npgpt/npgpt-rl only)")
    parser.add_argument("--novelty_max_ref", type=int, default=10000)

    args = parser.parse_args()

    # gpmolformer: force disable uniqueness/novelty
    if args.model == "gpmolformer":
        args.training = None

    if args.model == "npgpt-rl":
        run_npgpt_rl(args)
    elif args.model == "npgpt":
        run_single_model(args, "npgpt")
    elif args.model == "gpmolformer":
        run_single_model(args, "gpmolformer")


if __name__ == "__main__":
    main()
