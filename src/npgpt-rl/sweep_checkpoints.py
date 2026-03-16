"""
Sweep all RL checkpoints and plot validity/QED/SA/NP-likeness vs training step.
Generates N_SAMPLES molecules per checkpoint using temp=1.5, top_p=1.0.
"""

import sys, os, math, warnings, gc, re
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from npgpt.reward import sa_score_normalized, np_likeness_normalized

# ── config ──
TOKENIZER = str(PROJECT_ROOT / "externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json")
ORIG_CKPT = str(SCRIPT_DIR / "npgpt.ckpt")
RL_DIR    = str(PROJECT_ROOT / "results/rl")
N_SAMPLES = 100
TEMPERATURE = 1.5
TOP_P = 1.0
OUT_PATH  = str(PROJECT_ROOT / "results" / "sweep_metrics.png")


def load_model(ckpt_path, tokenizer_path, is_rl_ckpt=False):
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


def generate(model, tokenizer, n=100, batch=20, temperature=1.5, top_p=1.0):
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


def sanitize_smiles(smi):
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


def compute_metrics(smiles_list):
    validity, qed, sa, npl = [], [], [], []
    for smi in smiles_list:
        mol, err = sanitize_smiles(smi)
        if mol is None:
            validity.append(0)
        else:
            validity.append(1)
            qed.append(QED_module.qed(mol))
            sa.append(sa_score_normalized(mol))
            npl.append(np_likeness_normalized(mol))
    n = len(smiles_list)
    return {
        "validity": sum(validity) / n * 100,
        "qed": np.mean(qed) if qed else 0,
        "sa": np.mean(sa) if sa else 0,
        "np_likeness": np.mean(npl) if npl else 0,
    }


# ── Evaluate pretrained ──
print("Evaluating pretrained model ...")
model, tok = load_model(ORIG_CKPT, TOKENIZER)
smiles = generate(model, tok, N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P)
pretrained_metrics = compute_metrics(smiles)
print(f"  Pretrained: validity={pretrained_metrics['validity']:.1f}%, "
      f"QED={pretrained_metrics['qed']:.3f}, SA={pretrained_metrics['sa']:.3f}, "
      f"NP={pretrained_metrics['np_likeness']:.3f}")
del model; gc.collect()

# ── Find all checkpoints ──
ckpt_files = sorted(Path(RL_DIR).glob("actor_step_*.pt.npgpt.ckpt"),
                    key=lambda p: int(re.search(r"step_(\d+)", p.name).group(1)))

steps = []
results = {"validity": [], "qed": [], "sa": [], "np_likeness": []}

for ckpt_path in ckpt_files:
    step = int(re.search(r"step_(\d+)", ckpt_path.name).group(1))
    steps.append(step)
    print(f"Evaluating step {step} ...", end=" ", flush=True)

    model, tok = load_model(str(ckpt_path), TOKENIZER, is_rl_ckpt=True)
    smiles = generate(model, tok, N_SAMPLES, temperature=TEMPERATURE, top_p=TOP_P)
    m = compute_metrics(smiles)

    for k in results:
        results[k].append(m[k])

    print(f"validity={m['validity']:.1f}%, QED={m['qed']:.3f}, SA={m['sa']:.3f}, NP={m['np_likeness']:.3f}")
    del model; gc.collect()

# ── Plot ──
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(f"NPGPT RL Training Sweep ({N_SAMPLES} molecules/step, temp={TEMPERATURE}, top_p={TOP_P})",
             fontsize=14, fontweight="bold")

metrics_info = [
    ("validity", "Validity (%)", "%"),
    ("qed", "QED", ""),
    ("sa", "SA Score (norm)", ""),
    ("np_likeness", "NP-likeness (norm)", ""),
]

for ax, (key, title, fmt) in zip(axes.flat, metrics_info):
    vals = results[key]
    baseline = pretrained_metrics[key]

    ax.plot(steps, vals, "o-", color="#F5534C", linewidth=2, markersize=5, label="RL-finetuned", zorder=3)
    ax.axhline(baseline, color="#4C8BF5", linestyle="--", linewidth=2, label=f"Pretrained ({baseline:.1f}{'%' if key=='validity' else ''})")

    # Mark best point
    best_idx = np.argmax(vals)
    ax.plot(steps[best_idx], vals[best_idx], "*", color="gold", markersize=18,
            markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax.annotate(f"Best: step {steps[best_idx]}",
                xy=(steps[best_idx], vals[best_idx]),
                xytext=(10, 10), textcoords="offset points", fontsize=9,
                fontweight="bold", color="darkred",
                arrowprops=dict(arrowstyle="->", color="darkred"))

    ax.set_xlabel("Training Step")
    ax.set_ylabel(title)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"\nSweep plot saved → {OUT_PATH}")
