"""Evaluate NPComposer (Shawn) across 4 conditioning configurations with metrics and optional pathway classification."""

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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def npclassifier_classify(smiles, base_url="https://npclassifier.ucsd.edu/classify",
                          timeout=30, max_retries=3, sleep_s=0.5):
    """Call NPClassifier API.

    Input:
        smiles: SMILES string.
        base_url, timeout, max_retries, sleep_s: API parameters.
    Output:
        dict with pathway/superclass/class or error."""
    import time
    if not HAS_REQUESTS:
        return {"error": "requests not installed"}
    params = {"smiles": smiles, "cached": "true"}
    for attempt in range(max_retries):
        try:
            r = requests.get(base_url, params=params, timeout=timeout)
            r.raise_for_status()
            time.sleep(sleep_s)
            return r.json()
        except Exception as e:
            time.sleep(sleep_s * (2 ** attempt))
    return {"error": "max retries exceeded"}


def classify_pathway(valid_smiles, expected_pathway):
    """Classify SMILES and compute pathway accuracy.

    Input:
        valid_smiles: list of valid SMILES.
        expected_pathway: expected pathway label.
    Output:
        dict with accuracy, per-molecule results, pathway distribution."""
    if not HAS_REQUESTS:
        print("    Warning: 'requests' not installed, skipping classification")
        return None

    correct = 0
    total = 0
    pathway_dist = Counter()
    per_mol = []

    for smi in tqdm(valid_smiles, desc="    NPClassifier", leave=False):
        res = npclassifier_classify(smi)
        if "error" in res:
            per_mol.append({"smiles": smi, "predicted": None, "correct": False})
            continue

        # Extract pathway (first result)
        predicted = None
        pw = res.get("pathway_results") or res.get("pathway")
        if isinstance(pw, list) and pw:
            predicted = pw[0]
        elif isinstance(pw, str):
            predicted = pw

        total += 1
        is_correct = (predicted == expected_pathway)
        if is_correct:
            correct += 1
        if predicted:
            pathway_dist[predicted] += 1

        per_mol.append({"smiles": smi, "predicted": predicted, "correct": is_correct})

    accuracy = round(correct / total, 4) if total > 0 else 0.0

    return {
        "expected_pathway": expected_pathway,
        "n_classified": total,
        "n_correct": correct,
        "accuracy": accuracy,
        "pathway_distribution": dict(pathway_dist),
    }


_BASE_TOKENS = "<qed_bin:0.9<=qed<1><sa_bin:1<=sa<2>"

ALL_PATHWAYS = [
    "Alkaloids",
    "Amino acids and Peptides",
    "Carbohydrates",
    "Fatty acids",
    "Polyketides",
    "Shikimates and Phenylpropanoids",
    "Terpenoids",
]

_PATHWAY_COLORS = [
    "#E74C3C", "#E67E22", "#F1C40F", "#27AE60",
    "#2980B9", "#8E44AD", "#3498DB",
]

_PATHWAY_SHORT = {
    "Alkaloids": "Alkaloids",
    "Amino acids and Peptides": "Amino acids",
    "Carbohydrates": "Carbohydrates",
    "Fatty acids": "Fatty acids",
    "Polyketides": "Polyketides",
    "Shikimates and Phenylpropanoids": "Shikimates",
    "Terpenoids": "Terpenoids",
}

EVAL_CONFIGS = {"optimal_params": {
    "prompt": _BASE_TOKENS,
    "label": "Optimal QED+SA",
    "color": "#9B59B6",
}}

for _pw, _color in zip(ALL_PATHWAYS, _PATHWAY_COLORS):
    _key = "pathway_" + _pw.split()[0].lower().rstrip("s")
    EVAL_CONFIGS[_key] = {
        "prompt": f"<np_classifier_pathway:{_pw}>",
        "pathway": _pw,
        "label": _PATHWAY_SHORT[_pw],
        "color": _color,
    }


def load_model(ckpt_path):
    """Load NPComposer model.

    Input:
        ckpt_path: HuggingFace model path.
    Output:
        model, tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    return model, tokenizer


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_smiles(model, tokenizer, prompt, n=50, top_p=0.95, temperature=1.0, max_new_tokens=200):
    """Generate SMILES for a conditioning prompt.

    Input:
        model, tokenizer: NPComposer model and tokenizer.
        prompt: conditioning prompt.
        n, top_p, temperature, max_new_tokens: generation parameters.
    Output:
        list of SMILES strings."""
    device = next(model.parameters()).device
    x = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = x["input_ids"].shape[1]
    x = {k: v.to(device) for k, v in x.items()}

    smiles_list = []
    with torch.no_grad():
        for _ in range(n):
            y = model.generate(
                **x,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
            raw = tokenizer.decode(y[0][prompt_len:], skip_special_tokens=True).strip()
            smi = raw.split(".")[0].strip()
            if smi:
                smiles_list.append(smi)
    return smiles_list


def sanitize_smiles(smi):
    """Sanitize SMILES and categorize errors.

    Input:
        smi: SMILES string.
    Output:
        (mol_or_None, error_category)."""
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
        return None, "other"


def smiles_to_fp(smi):
    """Convert SMILES to Morgan fingerprint.

    Input:
        smi: SMILES string.
    Output:
        fingerprint or None."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def compute_internal_diversity(fps):
    """Compute pairwise Tanimoto similarity.

    Input:
        fps: list of fingerprints.
    Output:
        dict with mean and std."""
    if len(fps) < 2:
        return {"mean": None, "std": None}
    sims = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    return {"mean": round(float(np.mean(sims)), 4), "std": round(float(np.std(sims)), 4)}


def evaluate_smiles(smiles_list):
    """Compute validity, QED, SA, NP-likeness, diversity.

    Input:
        smiles_list: list of SMILES strings.
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
        "qed": {"mean": round(float(np.mean(qed_scores)), 4) if qed_scores else 0.0,
                "std": round(float(np.std(qed_scores)), 4) if qed_scores else 0.0},
        "sa": {"mean": round(float(np.mean(sa_scores)), 4) if sa_scores else 0.0,
               "std": round(float(np.std(sa_scores)), 4) if sa_scores else 0.0},
        "np_likeness": {"mean": round(float(np.mean(np_scores)), 4) if np_scores else 0.0,
                        "std": round(float(np.std(np_scores)), 4) if np_scores else 0.0} if NP_MODEL else None,
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

    return results


def _strip_raw(m):
    """Remove raw arrays before JSON export.

    Input:
        m: results dict.
    Output:
        dict without raw arrays."""
    out = dict(m)
    for k in ["_qed_scores", "_sa_scores", "_np_scores", "_validity_arr", "smiles"]:
        out.pop(k, None)
    return out


def _draw_pie(ax, sb, valid_color):
    color_map = {"valid": valid_color, "parse": "#FFB74D", "valence": "#E57373",
                 "kekulize": "#BA68C8", "other": "#90A4AE"}
    labels, sizes, colors = [], [], []
    for cat in ["valid", "parse", "valence", "kekulize", "other"]:
        if sb.get(cat, 0) > 0:
            labels.append(cat); sizes.append(sb[cat]); colors.append(color_map.get(cat, "#90A4AE"))
    if sizes:
        _, _, ats = ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                           startangle=90, textprops={"fontsize": 10})
        for at in ats:
            at.set_fontweight("bold")


def plot_single_config(all_results, seeds, config_label, color, out_path):
    """Plot 2x3 histogram for single config.

    Input:
        all_results: {seed: metrics} dict.
        seeds: list of seed values.
        config_label, color: config name and bar color.
        out_path: output file path.
    Output:
        none (saves PNG)."""
    qed_all = np.concatenate([np.array(all_results[s].get("_qed_scores", [])) for s in seeds])
    sa_all = np.concatenate([np.array(all_results[s].get("_sa_scores", [])) for s in seeds])
    np_all = np.concatenate([np.array(all_results[s].get("_np_scores", [])) for s in seeds])
    val_all = np.concatenate([np.array(all_results[s].get("_validity_arr", [])) for s in seeds])
    sb = Counter()
    for s in seeds:
        for k, v in all_results[s].get("sanitize_breakdown", {}).items():
            sb[k] += v
    n_total = sum(all_results[s]["n_total"] for s in seeds)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    fig.suptitle(f"NPComposer: {config_label}  ({n_total} molecules, seeds={seeds})",
                 fontsize=14, fontweight="bold", y=0.97)

    # (0,0) Validity
    ax = axes[0, 0]
    v_pct = float(val_all.mean()) * 100 if len(val_all) else 0
    bar = ax.bar(["Validity"], [v_pct], color=color, width=0.4, edgecolor="white")
    ax.text(bar[0].get_x() + bar[0].get_width()/2, bar[0].get_height() + 1,
            f"{v_pct:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 110); ax.set_ylabel("Valid (%)"); ax.spines[["top","right"]].set_visible(False)
    ax.set_title("Validity (sanitized)", fontsize=12, fontweight="bold")

    # (0,1) QED
    ax = axes[0, 1]
    if len(qed_all):
        ax.hist(qed_all, bins=np.linspace(0, 1, 21), alpha=0.75, color=color, edgecolor="white")
        ax.axvline(qed_all.mean(), color="#333", ls="--", lw=2, label=f"mean={qed_all.mean():.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("QED", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # (0,2) SA
    ax = axes[0, 2]
    if len(sa_all):
        lo, hi = max(1, float(sa_all.min())-0.5), min(10, float(sa_all.max())+0.5)
        ax.hist(sa_all, bins=np.linspace(lo, hi, 21), alpha=0.75, color=color, edgecolor="white")
        ax.axvline(sa_all.mean(), color="#333", ls="--", lw=2, label=f"mean={sa_all.mean():.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("SA Score (raw, lower=better)", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # (1,0) NP
    ax = axes[1, 0]
    if len(np_all):
        lo, hi = float(np_all.min())-0.5, float(np_all.max())+0.5
        ax.hist(np_all, bins=np.linspace(lo, hi, 21), alpha=0.75, color=color, edgecolor="white")
        ax.axvline(np_all.mean(), color="#333", ls="--", lw=2, label=f"mean={np_all.mean():.3f}")
        ax.legend(fontsize=9)
    ax.set_ylabel("Count"); ax.set_title("NP-likeness (raw)", fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

    # (1,1) Pie
    ax = axes[1, 1]
    _draw_pie(ax, sb, color)
    ax.set_title("Error Breakdown", fontsize=12, fontweight="bold")

    # (1,2) Summary
    ax = axes[1, 2]; ax.axis("off")
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
        lines.append(f"Int. Div: {np.mean(all_div):.4f} +/- {np.std(all_div):.4f}")
    ax.text(0.1, 0.95, "\n".join(lines), transform=ax.transAxes, fontsize=12,
            va="top", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("Summary", fontsize=12, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  Histogram -> {out_path}")


def plot_pathway_accuracy(pathway_classification, out_path):
    """Plot pathway classification accuracy for all pathways.

    Input:
        pathway_classification: {pathway: accuracy_dict} dict.
        out_path: output file path.
    Output:
        none (saves PNG)."""
    pathways = list(pathway_classification.keys())
    accuracies = [pathway_classification[p]["accuracy"] * 100 for p in pathways]
    n_correct = [pathway_classification[p]["n_correct"] for p in pathways]
    n_total = [pathway_classification[p]["n_classified"] for p in pathways]
    short_labels = [_PATHWAY_SHORT.get(p, p) for p in pathways]

    # Match colors from EVAL_CONFIGS
    color_map = {}
    for cfg in EVAL_CONFIGS.values():
        pw = cfg.get("pathway")
        if pw:
            color_map[pw] = cfg["color"]
    colors = [color_map.get(p, "#999") for p in pathways]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: accuracy bar chart
    ax = axes[0]
    x = np.arange(len(pathways))
    bars = ax.bar(x, accuracies, color=colors, edgecolor="white", alpha=0.85, width=0.6)

    pad = max(accuracies) * 0.02 if accuracies else 1
    for bar, acc, nc, nt in zip(bars, accuracies, n_correct, n_total):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + pad,
                f"{acc:.1f}%\n({nc}/{nt})", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, rotation=25, ha="right")
    ax.set_ylim(0, max(accuracies) * 1.25 if accuracies else 110)
    ax.set_ylabel("Classification Accuracy (%)", fontsize=11)
    ax.set_title("Pathway Classification Accuracy", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    # Right panel: predicted distribution stacked bar
    ax2 = axes[1]
    all_predicted = set()
    for p in pathways:
        all_predicted.update(pathway_classification[p].get("pathway_distribution", {}).keys())
    all_predicted = sorted(all_predicted)

    dist_cmap = plt.colormaps.get_cmap("tab20").resampled(len(all_predicted))
    pred_colors = {name: dist_cmap(i) for i, name in enumerate(all_predicted)}

    bottoms = np.zeros(len(pathways))
    legend_handles = {}
    for pred_pw in all_predicted:
        vals = []
        for p in pathways:
            dist = pathway_classification[p].get("pathway_distribution", {})
            total = sum(dist.values()) if dist else 1
            vals.append(dist.get(pred_pw, 0) / total * 100 if total else 0)
        b = ax2.bar(x, vals, bottom=bottoms, color=pred_colors[pred_pw],
                    edgecolor="white", linewidth=0.3, width=0.6)
        if any(v > 0 for v in vals):
            short = _PATHWAY_SHORT.get(pred_pw, pred_pw)
            legend_handles[short] = b[0]
        bottoms += vals

    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, fontsize=9, rotation=25, ha="right")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Predicted Distribution (%)", fontsize=11)
    ax2.set_title("Predicted Pathway Distribution", fontsize=13, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(legend_handles.values(), legend_handles.keys(),
               fontsize=7, loc="upper right", ncol=1)

    fig.suptitle("NPComposer: Pathway Adherence Analysis (NPClassifier API)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pathway accuracy plot -> {out_path}")


def plot_overview(all_configs_agg, out_path):
    """Plot comparison of all 4 configurations.

    Input:
        all_configs_agg: {config_name: aggregate_dict} dict.
        out_path: output file path.
    Output:
        none (saves PNG)."""
    metric_keys = ["validity", "qed", "sa"]
    display = {"validity": "Validity", "qed": "QED", "sa": "SA (raw)"}

    config_names = list(all_configs_agg.keys())
    n_configs = len(config_names)
    n_metrics = len(metric_keys)

    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = [EVAL_CONFIGS.get(cn, {}).get("color", "#999") for cn in config_names]
    labels = [EVAL_CONFIGS.get(cn, {}).get("label", cn) for cn in config_names]

    for ax, key in zip(axes, metric_keys):
        vals = []
        errs = []
        for cn in config_names:
            agg = all_configs_agg[cn].get(key, {})
            vals.append(agg.get("mean", 0))
            errs.append(agg.get("std", 0))

        x = np.arange(n_configs)
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=colors, edgecolor="white", alpha=0.85, width=0.6)
        for bar, val in zip(bars, vals):
            if key == "validity":
                txt = f"{val*100:.1f}%"
            else:
                txt = f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    txt, ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(display[key], fontsize=13, fontweight="bold")
        if key == "validity":
            ax.set_ylim(0, 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("NPComposer: Comparison of 4 Conditioning Configs", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"\nOverview plot -> {out_path}")


def run_config(config_name, model, tokenizer, args, out_dir):
    """Run evaluation for single config.

    Input:
        config_name: config key.
        model, tokenizer: NPComposer model and tokenizer.
        args: parsed arguments.
        out_dir: output directory.
    Output:
        aggregate metrics dict."""
    cfg = EVAL_CONFIGS[config_name]
    label = cfg["label"]
    color = cfg["color"]
    prompt = cfg["prompt"]
    expected_pathway = cfg.get("pathway")  # None for non-pathway configs

    print(f"\n{'='*60}")
    print(f"Config: {label}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}")

    all_results = {}
    all_valid_smiles = []

    for seed in args.seeds:
        print(f"\n  Seed {seed}: generating {args.n_samples} molecules...")
        set_seed(seed)
        smiles = generate_smiles(model, tokenizer, prompt, args.n_samples,
                                 top_p=args.top_p, temperature=args.temperature)
        metrics = evaluate_smiles(smiles)
        metrics["seed"] = seed
        metrics["smiles"] = smiles
        all_results[seed] = metrics

        # Collect valid SMILES for pathway classification
        for smi in smiles:
            mol, err = sanitize_smiles(smi)
            if mol is not None:
                all_valid_smiles.append(smi)

        print(f"    Validity={metrics['validity']*100:.1f}% "
              f"QED={metrics['qed']['mean']:.4f} SA={metrics['sa']['mean']:.4f}")

    # Histogram
    plot_single_config(all_results, args.seeds, label, color,
                       out_dir / f"shawn_{config_name}_histogram.png")

    # Aggregate
    aggregate = {}
    for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
        vals = []
        for seed in args.seeds:
            m = all_results[seed]
            if mk == "validity": vals.append(m["validity"])
            elif mk in ("qed", "sa"): vals.append(m[mk]["mean"])
            elif mk == "np_likeness" and m.get("np_likeness"): vals.append(m["np_likeness"]["mean"])
            elif mk == "internal_diversity" and "internal_diversity" in m:
                v = m["internal_diversity"]["mean"]
                if v is not None: vals.append(v)
        if vals:
            aggregate[mk] = {"mean": round(float(np.mean(vals)), 4),
                             "std": round(float(np.std(vals)), 4),
                             "values": vals}

    # Pathway classification accuracy (only for pathway configs + --classify)
    pathway_classification = None
    if expected_pathway and args.classify and all_valid_smiles:
        print(f"\n  Classifying {len(all_valid_smiles)} valid molecules for pathway accuracy...")
        cls_result = classify_pathway(all_valid_smiles, expected_pathway)
        if cls_result:
            pathway_classification = cls_result
            aggregate["pathway_accuracy"] = {
                "mean": cls_result["accuracy"],
                "n_correct": cls_result["n_correct"],
                "n_classified": cls_result["n_classified"],
            }
            print(f"  Pathway accuracy: {cls_result['accuracy']*100:.1f}% "
                  f"({cls_result['n_correct']}/{cls_result['n_classified']})")
            print(f"  Predicted distribution: {cls_result['pathway_distribution']}")

    # JSON
    json_out = {
        "config_name": config_name, "label": label, "prompt": prompt,
        "aggregate": aggregate,
        "per_seed": {s: _strip_raw(all_results[s]) for s in args.seeds},
    }
    if pathway_classification:
        json_out["pathway_classification"] = pathway_classification

    json_path = out_dir / f"shawn_{config_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  JSON -> {json_path}")

    # Print aggregate
    print(f"\n  Aggregate ({label}):")
    for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
        a = aggregate.get(mk)
        if a:
            if mk == "validity":
                print(f"    {mk:<20} {a['mean']*100:.1f}% +/- {a['std']*100:.1f}%")
            else:
                print(f"    {mk:<20} {a['mean']:.4f} +/- {a['std']:.4f}")
    if aggregate.get("pathway_accuracy"):
        pa = aggregate["pathway_accuracy"]
        print(f"    {'pathway_accuracy':<20} {pa['mean']*100:.1f}% ({pa['n_correct']}/{pa['n_classified']})")

    return aggregate, pathway_classification


def main():
    parser = argparse.ArgumentParser(description="Evaluate NPComposer (Shawn model)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_samples", type=int, default=50, help="Molecules per seed per config")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--ckpt_path", type=str, default="ralyn/NPComposer-v2",
                        help="HuggingFace model name or local checkpoint path")
    parser.add_argument("--configs", type=str, nargs="+",
                        default=list(EVAL_CONFIGS.keys()),
                        choices=list(EVAL_CONFIGS.keys()),
                        help="Which configs to evaluate")
    parser.add_argument("--classify", action="store_true", default=False,
                        help="Enable NPClassifier API calls for pathway classification accuracy")
    parser.add_argument("--out_dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "evaluation_shawn"))

    args = parser.parse_args()

    # --classify without explicit --configs: run pathway configs only
    if args.classify and "--configs" not in sys.argv:
        args.configs = [k for k, v in EVAL_CONFIGS.items() if v.get("pathway")]

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model once
    print(f"Loading NPComposer: {args.ckpt_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(args.ckpt_path)
    model = model.to(device)

    # Run all configs
    all_agg = {}
    all_pathway_cls = {}
    for config_name in args.configs:
        agg, pw_cls = run_config(config_name, model, tokenizer, args, OUT_DIR)
        all_agg[config_name] = agg
        expected_pw = EVAL_CONFIGS[config_name].get("pathway")
        if pw_cls and expected_pw:
            all_pathway_cls[expected_pw] = pw_cls

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Overview comparison plot
    if len(all_agg) > 1:
        plot_overview(all_agg, OUT_DIR / "shawn_overview_comparison.png")

    # Combined pathway accuracy plot (all pathways in one chart)
    if all_pathway_cls:
        plot_pathway_accuracy(all_pathway_cls, OUT_DIR / "shawn_all_pathway_accuracy.png")

    # Summary JSON
    summary = {
        "model": args.ckpt_path,
        "config": {"seeds": args.seeds, "n_samples": args.n_samples,
                    "temperature": args.temperature, "top_p": args.top_p},
        "aggregate_by_config": all_agg,
    }
    summary_path = OUT_DIR / "shawn_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON -> {summary_path}")

    # Final table
    print(f"\n{'='*70}")
    print(f"FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'Validity':>10} {'QED':>10} {'SA':>10}")
    print("-" * 55)
    for cn in args.configs:
        a = all_agg[cn]
        v = a.get("validity", {})
        q = a.get("qed", {})
        s = a.get("sa", {})
        print(f"{EVAL_CONFIGS[cn]['label']:<25} "
              f"{v.get('mean',0)*100:>7.1f}%  "
              f"{q.get('mean',0):>8.4f}  "
              f"{s.get('mean',0):>8.4f}")
    print("=" * 55)


if __name__ == "__main__":
    main()
