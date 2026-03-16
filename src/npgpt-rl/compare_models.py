"""Compare pretrained NPGPT vs RL-finetuned NPGPT with metrics and visualization."""

import sys, os, math, warnings, argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "externals" / "smiles-gpt"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "src"))

from rdkit import Chem, RDLogger
from rdkit.Chem import QED as QED_module

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer
from npgpt.config import SmilesGptGenerationConfig
from npgpt.reward import sa_score_normalized, np_likeness_normalized


def load_model(ckpt_path, tokenizer_path, is_rl_ckpt=False):
    """Load NPGPT model from checkpoint.

    Input:
        ckpt_path: path to checkpoint.
        tokenizer_path: path to tokenizer.
        is_rl_ckpt: whether checkpoint is RL-trained.
    Output:
        (model, tokenizer)
    """
    cfg = SmilesGptTrainingConfig()
    tok = get_tokenizer(cfg, tokenizer_path)

    if is_rl_ckpt:
        # RL checkpoint saved by save_npgpt_checkpoint: dict with 'state_dict'
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = SmilesGptModel(config=cfg, tokenizer=tok)
        # state_dict keys are "model.transformer.*", "model.lm_head.*"
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model = SmilesGptModel.load_from_checkpoint(
            ckpt_path, config=cfg, tokenizer=tok, strict=False
        )

    model.eval()
    return model, tok


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate(model, tokenizer, n=100, batch=20, temperature=1.5, top_p=1.0):
    """Generate SMILES from model.

    Input:
        model: NPGPT model.
        tokenizer: tokenizer.
        n: number of molecules.
        batch: batch size.
        temperature: sampling temperature.
        top_p: top-p nucleus sampling.
    Output:
        list of SMILES strings.
    """
    device = torch.device("cpu")   # safe default for comparison
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


def sanitize_smiles(smi):
    """Sanitize SMILES and categorize errors.

    Input:
        smi: SMILES string.
    Output:
        (mol or None, error_category)
    """
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    if mol is None:
        return None, "parse"
    try:
        san_flags = Chem.SanitizeFlags.SANITIZE_ALL
        Chem.SanitizeMol(mol, san_flags)
        return mol, "valid"
    except Exception as e:
        err = str(e).lower()
        if "valence" in err or "explicit valence" in err:
            return None, "valence"
        elif "kekulize" in err or "kekul" in err:
            return None, "kekulize"
        else:
            return None, "other_sanitize"


def compute_metrics(smiles_list):
    """Compute metrics and sanitization breakdown.

    Input:
        smiles_list: list of SMILES strings.
    Output:
        dict with validity, qed, sa, np_likeness, and sanitize_breakdown.
    """
    validity, qed, sa, npl = [], [], [], []
    error_counts = {"valid": 0, "parse": 0, "valence": 0, "kekulize": 0, "other_sanitize": 0}
    for smi in smiles_list:
        mol, err_cat = sanitize_smiles(smi)
        error_counts[err_cat] += 1
        if mol is None:
            validity.append(0); qed.append(0); sa.append(0); npl.append(0)
        else:
            validity.append(1)
            qed.append(QED_module.qed(mol))
            sa.append(sa_score_normalized(mol))
            npl.append(np_likeness_normalized(mol))
    return {
        "validity": np.array(validity),
        "qed": np.array(qed),
        "sa": np.array(sa),
        "np_likeness": np.array(npl),
        "sanitize_breakdown": error_counts,
    }


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--n_samples", type=int, default=100, help="Number of molecules to generate")
parser.add_argument("--temperature", type=float, default=1.5, help="Sampling temperature")
parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")
parser.add_argument("--rl_ckpt", type=str,
                    default=str(SCRIPT_DIR / "npgpt_rl_step_600.ckpt"))
parser.add_argument("--orig_ckpt", type=str,
                    default=str(SCRIPT_DIR / "npgpt.ckpt"))
parser.add_argument("--tokenizer", type=str,
                    default=str(PROJECT_ROOT / "externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"))
cmd_args = parser.parse_args()

TOKENIZER  = cmd_args.tokenizer
ORIG_CKPT  = cmd_args.orig_ckpt
RL_CKPT    = cmd_args.rl_ckpt
N_SAMPLES  = cmd_args.n_samples
TEMPERATURE = cmd_args.temperature
TOP_P      = cmd_args.top_p
OUT_PATH   = str(PROJECT_ROOT / "results" / "comparison_histogram.png")

import gc

set_seed(cmd_args.seed)
print(f"Seed: {cmd_args.seed}, N_SAMPLES: {N_SAMPLES}, temp: {TEMPERATURE}, top_p: {TOP_P}")
print("Loading pretrained NPGPT …")
model_orig, tok = load_model(ORIG_CKPT, TOKENIZER)
print(f"Generating {N_SAMPLES} molecules (pretrained) …")
set_seed(cmd_args.seed)
smiles_orig = generate(model_orig, tok, N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P)
metrics_orig = compute_metrics(smiles_orig)
del model_orig; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

print("Loading RL-finetuned NPGPT …")
model_rl, tok_rl = load_model(RL_CKPT, TOKENIZER, is_rl_ckpt=True)
print(f"Generating {N_SAMPLES} molecules (RL) …")
set_seed(cmd_args.seed)
smiles_rl = generate(model_rl, tok_rl, N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P)
metrics_rl = compute_metrics(smiles_rl)
del model_rl; gc.collect()

# ── print summary ────────────────────────────────────────────────────────

print("\n" + "="*60)
print(f"{'Metric':<16} {'Pretrained':>12} {'RL-finetuned':>14}")
print("="*60)
for k in ["validity", "qed", "sa", "np_likeness"]:
    vo = metrics_orig[k]
    vr = metrics_rl[k]
    if k == "validity":
        print(f"{k:<16} {vo.mean()*100:>11.1f}% {vr.mean()*100:>13.1f}%")
    else:
        valid_o = vo[metrics_orig["validity"] == 1]
        valid_r = vr[metrics_rl["validity"] == 1]
        mo = valid_o.mean() if len(valid_o) else 0
        mr = valid_r.mean() if len(valid_r) else 0
        print(f"{k:<16} {mo:>12.4f} {mr:>14.4f}")
print("="*60)

print("\nSANITIZATION BREAKDOWN")
print("="*60)
sb_orig = metrics_orig["sanitize_breakdown"]
sb_rl = metrics_rl["sanitize_breakdown"]
print(f"{'Error Type':<20} {'Pretrained':>12} {'RL-finetuned':>14}")
print("="*60)
for cat in ["valid", "parse", "valence", "kekulize", "other_sanitize"]:
    o_cnt = sb_orig[cat]
    r_cnt = sb_rl[cat]
    o_pct = o_cnt / N_SAMPLES * 100
    r_pct = r_cnt / N_SAMPLES * 100
    print(f"{cat:<20} {o_cnt:>6} ({o_pct:4.1f}%) {r_cnt:>6} ({r_pct:4.1f}%)")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle(f"NPGPT: Pretrained vs RL-finetuned  ({N_SAMPLES} molecules, temp={TEMPERATURE}, top_p={TOP_P}, seed={cmd_args.seed})",
             fontsize=14, fontweight="bold", y=0.97)

c_orig, c_rl = "#4C8BF5", "#F5534C"

# Row 1: Validity bar + QED + SA
# Row 2: NP-likeness + Sanitization pie (Pretrained) + Sanitization pie (RL)

titles_top  = ["Validity (sanitized)", "QED", "SA Score (norm)"]
keys_top    = ["validity", "qed", "sa"]
titles_bot  = ["NP-likeness (norm)", "Error Breakdown: Pretrained", "Error Breakdown: RL-finetuned"]
keys_bot    = ["np_likeness", None, None]

all_titles = titles_top + titles_bot
all_keys   = keys_top + keys_bot

for idx, (ax, title, key) in enumerate(zip(axes.flat, all_titles, all_keys)):
    if key is not None:
        d_orig = metrics_orig[key]
        d_rl   = metrics_rl[key]
        if key == "validity":
            v_orig = d_orig.mean() * 100
            v_rl   = d_rl.mean() * 100
            bars = ax.bar(["Pretrained", "RL-finetuned"], [v_orig, v_rl],
                           color=[c_orig, c_rl], width=0.5, edgecolor="white")
            for bar, val in zip(bars, [v_orig, v_rl]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=12)
            ax.set_ylim(0, 110)
            ax.set_ylabel("Valid (%)")
        else:
            vo = d_orig[metrics_orig["validity"] == 1]
            vr = d_rl[metrics_rl["validity"] == 1]
            rng = (0, 1)
            bins = np.linspace(rng[0], rng[1], 21)
            ax.hist(vo, bins=bins, alpha=0.6, color=c_orig, edgecolor="white", label="Pretrained")
            ax.hist(vr, bins=bins, alpha=0.6, color=c_rl, edgecolor="white", label="RL-finetuned")
            if len(vo): ax.axvline(vo.mean(), color=c_orig, ls="--", lw=1.5)
            if len(vr): ax.axvline(vr.mean(), color=c_rl,   ls="--", lw=1.5)
            ax.set_ylabel("Count")
            ax.legend(fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
    else:
        # Pie chart for sanitization breakdown
        if "Pretrained" in title:
            sb = sb_orig
            pie_color_valid = c_orig
        else:
            sb = sb_rl
            pie_color_valid = c_rl
        # filter out zero categories
        labels, sizes, colors_pie = [], [], []
        color_map = {"valid": pie_color_valid, "parse": "#FFB74D", "valence": "#E57373",
                     "kekulize": "#BA68C8", "other_sanitize": "#90A4AE"}
        for cat in ["valid", "parse", "valence", "kekulize", "other_sanitize"]:
            if sb[cat] > 0:
                labels.append(cat)
                sizes.append(sb[cat])
                colors_pie.append(color_map[cat])
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie, autopct="%1.1f%%",
            startangle=90, textprops={"fontsize": 10}
        )
        for at in autotexts:
            at.set_fontweight("bold")

    ax.set_title(title, fontsize=12, fontweight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.94])
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nHistogram saved → {OUT_PATH}")
