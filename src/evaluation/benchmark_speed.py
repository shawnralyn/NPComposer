"""Benchmark generation speed: measure wall-clock time to generate molecules per model.

Runs multiple seeds (default: 1, 2, 3) with N molecules each (default: 50),
then reports mean ± std for time, throughput, and validity with error bars.

Usage:
    python src/evaluation/benchmark_speed.py
    python src/evaluation/benchmark_speed.py --n_molecules 50 --seeds "1 2 3"
    python src/evaluation/benchmark_speed.py --models npcomposer gpmolformer
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

# Enable MPS fallback for ops not yet implemented on Apple Silicon
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
RDLogger.logger().setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
# Device selection: CUDA > MPS (Apple Silicon) > CPU
# ---------------------------------------------------------------------------

def get_device():
    """Detect the best available device.

    Output:
        torch.device for cuda, mps, or cpu.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(device):
    """Human-readable device name.

    Input:
        device: torch.device.
    Output:
        string like 'cuda (NVIDIA ...)', 'mps (Apple Silicon)', or 'cpu'.
    """
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        return f"cuda ({name})"
    if device.type == "mps":
        return "mps (Apple Silicon)"
    return "cpu"


def set_seed(seed, device):
    """Set random seeds for reproducibility.

    Input:
        seed: integer seed.
        device: torch.device.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_npgpt(ckpt_path, tokenizer_path, is_rl=False):
    """Load NPGPT model (pretrained or RL).

    Input:
        ckpt_path: path to checkpoint.
        tokenizer_path: path to tokenizer.json.
        is_rl: load as RL checkpoint.
    Output:
        (model, tokenizer)
    """
    sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "externals" / "smiles-gpt"))
    from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer

    cfg = SmilesGptTrainingConfig()
    tok = get_tokenizer(cfg, tokenizer_path)
    if is_rl:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = SmilesGptModel(config=cfg, tokenizer=tok)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model = SmilesGptModel.load_from_checkpoint(
            ckpt_path, config=cfg, tokenizer=tok, strict=False
        )
    model.eval()
    return model, tok


def load_hf_model(ckpt_path, tokenizer_name=None):
    """Load HuggingFace causal LM.

    Input:
        ckpt_path: model name or path.
        tokenizer_name: tokenizer name/path (defaults to ckpt_path).
    Output:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_path = tokenizer_name or ckpt_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_npgpt_batch(model, tokenizer, n, batch_size=20, temperature=1.5, top_p=1.0):
    """Generate n SMILES with NPGPT using batched decoding.

    Input:
        model, tokenizer: NPGPT model and tokenizer.
        n: number of molecules to generate.
        batch_size, temperature, top_p: generation parameters.
    Output:
        list of SMILES strings.
    """
    import math
    device = next(model.parameters()).device
    smiles = []
    with torch.no_grad():
        for _ in range(math.ceil(n / batch_size)):
            bs = min(batch_size, n - len(smiles))
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


def generate_hf_batch(model, tokenizer, n, prompt="", temperature=1.0, top_p=0.95, max_new_tokens=200):
    """Generate n SMILES with HuggingFace model (one-by-one).

    Input:
        model, tokenizer: HF model and tokenizer.
        n: number of molecules to generate.
        prompt, temperature, top_p, max_new_tokens: generation parameters.
    Output:
        list of SMILES strings.
    """
    device = next(model.parameters()).device
    if prompt:
        x = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    else:
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.cls_token_id or 0
        x = {"input_ids": torch.tensor([[bos_id]])}
    prompt_len = x["input_ids"].shape[1]
    x = {k: v.to(device) for k, v in x.items()}

    smiles = []
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
                smiles.append(smi)
    return smiles


def generate_gpmolformer_batch(model, tokenizer, n, batch_size=20, temperature=1.0):
    """Generate n SMILES with GP-MoLFormer using batched decoding.

    Input:
        model, tokenizer: GP-MoLFormer model and tokenizer.
        n: number of molecules to generate.
        batch_size, temperature: generation parameters.
    Output:
        list of SMILES strings.
    """
    import math
    device = next(model.parameters()).device
    smiles = []
    with torch.no_grad():
        for _ in range(math.ceil(n / batch_size)):
            bs = min(batch_size, n - len(smiles))
            out = model.generate(
                do_sample=True,
                temperature=temperature,
                top_k=None,
                max_length=model.config.max_position_embeddings,
                num_return_sequences=bs,
            )
            smiles.extend(tokenizer.batch_decode(out.cpu(), skip_special_tokens=True))
    return smiles[:n]


# ---------------------------------------------------------------------------
# Single-seed benchmark runner
# ---------------------------------------------------------------------------

def benchmark_one_seed(name, generate_fn, n_molecules, seed, device):
    """Run benchmark for a single seed.

    Input:
        name: model display name.
        generate_fn: callable(n) -> list of SMILES.
        n_molecules: number of molecules to generate.
        seed: random seed.
        device: torch.device.
    Output:
        dict with per-seed timing and validity.
    """
    set_seed(seed, device)

    t0 = time.perf_counter()
    smiles = generate_fn(n_molecules)
    total_time = time.perf_counter() - t0

    n_generated = len(smiles)
    n_valid = sum(1 for s in smiles if Chem.MolFromSmiles(s) is not None)
    per_mol = total_time / n_generated if n_generated else 0.0

    return {
        "seed": seed,
        "n_generated": n_generated,
        "n_valid": n_valid,
        "validity": n_valid / n_generated if n_generated else 0.0,
        "total_time_s": total_time,
        "per_molecule_s": per_mol,
        "molecules_per_min": 60 / per_mol if per_mol > 0 else 0.0,
    }


def benchmark_model(name, generate_fn, n_molecules, seeds, device):
    """Benchmark a model across multiple seeds.

    Input:
        name: model display name.
        generate_fn: callable(n) -> list of SMILES.
        n_molecules: number of molecules per seed.
        seeds: list of integer seeds.
        device: torch.device.
    Output:
        dict with per-seed results and aggregate mean/std.
    """
    print(f"\n{'='*65}")
    print(f"  {name}  ({n_molecules} mol x {len(seeds)} seeds)")
    print(f"  Device: {device_label(device)}")
    print(f"{'='*65}")

    # Warmup (not counted)
    print("  Warmup...", end=" ", flush=True)
    set_seed(0, device)
    t0 = time.perf_counter()
    _ = generate_fn(2)
    warmup_time = time.perf_counter() - t0
    print(f"{warmup_time:.2f}s")

    per_seed = []
    for seed in seeds:
        print(f"  Seed {seed}...", end=" ", flush=True)
        r = benchmark_one_seed(name, generate_fn, n_molecules, seed, device)
        per_seed.append(r)
        print(f"{r['total_time_s']:.2f}s  "
              f"({r['per_molecule_s']:.4f}s/mol, "
              f"validity {r['validity']*100:.1f}%)")

    # Aggregate
    total_times = [r["total_time_s"] for r in per_seed]
    per_mol_times = [r["per_molecule_s"] for r in per_seed]
    throughputs = [r["molecules_per_min"] for r in per_seed]
    validities = [r["validity"] for r in per_seed]

    agg = {
        "model": name,
        "device": device_label(device),
        "n_molecules": n_molecules,
        "seeds": seeds,
        "warmup_time_s": round(warmup_time, 3),
        "total_time_s": {"mean": round(np.mean(total_times), 3),
                         "std": round(np.std(total_times), 3),
                         "values": [round(v, 3) for v in total_times]},
        "per_molecule_s": {"mean": round(np.mean(per_mol_times), 4),
                           "std": round(np.std(per_mol_times), 4),
                           "values": [round(v, 4) for v in per_mol_times]},
        "molecules_per_min": {"mean": round(np.mean(throughputs), 1),
                              "std": round(np.std(throughputs), 1),
                              "values": [round(v, 1) for v in throughputs]},
        "validity": {"mean": round(np.mean(validities), 4),
                     "std": round(np.std(validities), 4),
                     "values": [round(v, 4) for v in validities]},
        "per_seed": per_seed,
    }

    print(f"  ── Mean ± Std ──")
    print(f"  Total time:   {agg['total_time_s']['mean']:.2f} ± {agg['total_time_s']['std']:.2f}s")
    print(f"  Per molecule: {agg['per_molecule_s']['mean']:.4f} ± {agg['per_molecule_s']['std']:.4f}s")
    print(f"  Throughput:   {agg['molecules_per_min']['mean']:.1f} ± {agg['molecules_per_min']['std']:.1f} mol/min")
    print(f"  Validity:     {agg['validity']['mean']*100:.1f} ± {agg['validity']['std']*100:.1f}%")

    return agg


# ---------------------------------------------------------------------------
# Visualization with error bars
# ---------------------------------------------------------------------------

def plot_results(results, out_path, n_molecules, seeds):
    """Create bar chart with error bars comparing models.

    Input:
        results: list of aggregate result dicts.
        out_path: output PNG path.
        n_molecules: molecules per seed.
        seeds: list of seeds used.
    """
    names = [r["model"] for r in results]
    colors = ["#4C8BF5", "#F5534C", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    # --- Panel 1: Per-molecule time ---
    means = [r["per_molecule_s"]["mean"] for r in results]
    stds = [r["per_molecule_s"]["std"] for r in results]
    bars = axes[0].bar(names, means, yerr=stds, capsize=5,
                       color=colors[:len(names)], edgecolor="white",
                       linewidth=0.5, alpha=0.88,
                       error_kw={"lw": 1.5, "capthick": 1.5})
    axes[0].set_ylabel("Time per molecule (s)")
    axes[0].set_title("Generation Speed", fontweight="bold")
    for bar, m, s in zip(bars, means, stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.002,
                     f"{m:.3f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # --- Panel 2: Throughput ---
    means = [r["molecules_per_min"]["mean"] for r in results]
    stds = [r["molecules_per_min"]["std"] for r in results]
    bars = axes[1].bar(names, means, yerr=stds, capsize=5,
                       color=colors[:len(names)], edgecolor="white",
                       linewidth=0.5, alpha=0.88,
                       error_kw={"lw": 1.5, "capthick": 1.5})
    axes[1].set_ylabel("Molecules / min")
    axes[1].set_title("Throughput", fontweight="bold")
    for bar, m, s in zip(bars, means, stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                     f"{m:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # --- Panel 3: Validity ---
    means = [r["validity"]["mean"] * 100 for r in results]
    stds = [r["validity"]["std"] * 100 for r in results]
    bars = axes[2].bar(names, means, yerr=stds, capsize=5,
                       color=colors[:len(names)], edgecolor="white",
                       linewidth=0.5, alpha=0.88,
                       error_kw={"lw": 1.5, "capthick": 1.5})
    axes[2].set_ylabel("Validity (%)")
    axes[2].set_title("Validity", fontweight="bold")
    axes[2].set_ylim(0, 110)
    for bar, m, s in zip(bars, means, stds):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 1,
                     f"{m:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", rotation=15, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    device_str = results[0]["device"] if results else "unknown"
    seed_str = ", ".join(str(s) for s in seeds)
    fig.suptitle(
        f"Generation Benchmark  ({n_molecules} mol × {len(seeds)} seeds [{seed_str}],  {device_str})",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

NPGPT_RL_DIR = PROJECT_ROOT / "src" / "npgpt-rl"
DEFAULT_TOKENIZER = str(
    PROJECT_ROOT / "external/npgpt/externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"
)
GPMOLFORMER_TOKENIZER = "ibm-research/MoLFormer-XL-both-10pct"


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark generation speed with multiple seeds and error bars."
    )
    parser.add_argument("--n_molecules", type=int, default=50,
                        help="Number of molecules per seed (default: 50)")
    parser.add_argument("--seeds", type=str, default="1 2 3",
                        help="Space-separated seeds (default: '1 2 3')")
    parser.add_argument("--models", nargs="+",
                        default=["npgpt", "npgpt-rl", "gpmolformer", "npcomposer"],
                        help="Models to benchmark")
    parser.add_argument("--orig_ckpt", type=str, default=str(NPGPT_RL_DIR / "npgpt.ckpt"))
    parser.add_argument("--rl_ckpt", type=str, default=str(NPGPT_RL_DIR / "npgpt_rl_step_600.ckpt"))
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--npcomposer_ckpt", type=str, default="ralyn/NPComposer-v2")
    parser.add_argument("--gpmolformer_ckpt", type=str, default="ibm-research/GP-MoLFormer-Uniq")
    parser.add_argument("--npcomposer_prompt", type=str,
                        default="<qed_bin:0.9<=qed<1><sa_bin:1<=sa<2>",
                        help="NPComposer conditioning prompt")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split()]
    out_dir = PROJECT_ROOT / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    device_str = device_label(device)
    print(f"Device: {device_str}")
    print(f"Molecules per seed: {args.n_molecules}")
    print(f"Seeds: {seeds}")
    print(f"Models: {', '.join(args.models)}")

    results = []

    # --- NPGPT Pretrained ---
    if "npgpt" in args.models:
        try:
            model, tok = load_npgpt(args.orig_ckpt, args.tokenizer, is_rl=False)
            model.to(device)
            gen_fn = lambda n: generate_npgpt_batch(model, tok, n)
            r = benchmark_model("NPGPT (Pretrained)", gen_fn, args.n_molecules, seeds, device)
            results.append(r)
            del model, tok
        except Exception as e:
            print(f"  [SKIP] NPGPT Pretrained: {e}")

    # --- NPGPT-RL ---
    if "npgpt-rl" in args.models:
        try:
            model, tok = load_npgpt(args.rl_ckpt, args.tokenizer, is_rl=True)
            model.to(device)
            gen_fn = lambda n: generate_npgpt_batch(model, tok, n)
            r = benchmark_model("NPGPT-RL", gen_fn, args.n_molecules, seeds, device)
            results.append(r)
            del model, tok
        except Exception as e:
            print(f"  [SKIP] NPGPT-RL: {e}")

    # --- GP-MoLFormer ---
    if "gpmolformer" in args.models:
        try:
            model, tok = load_hf_model(args.gpmolformer_ckpt, tokenizer_name=GPMOLFORMER_TOKENIZER)
            model = model.to(device)
            gen_fn = lambda n: generate_gpmolformer_batch(model, tok, n)
            r = benchmark_model("GP-MoLFormer", gen_fn, args.n_molecules, seeds, device)
            results.append(r)
            del model, tok
        except Exception as e:
            print(f"  [SKIP] GP-MoLFormer: {e}")

    # --- NPComposer ---
    if "npcomposer" in args.models:
        try:
            model, tok = load_hf_model(args.npcomposer_ckpt)
            model = model.to(device)
            prompt = args.npcomposer_prompt
            gen_fn = lambda n: generate_hf_batch(model, tok, n, prompt=prompt)
            r = benchmark_model("NPComposer", gen_fn, args.n_molecules, seeds, device)
            results.append(r)
            del model, tok
        except Exception as e:
            print(f"  [SKIP] NPComposer: {e}")

    if not results:
        print("\nNo models were benchmarked successfully.")
        return

    # Save JSON
    output = {
        "config": {
            "n_molecules": args.n_molecules,
            "seeds": seeds,
            "device": device_str,
            "device_type": device.type,
        },
        "results": results,
    }
    json_path = out_dir / "speed_benchmark.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved -> {json_path}")

    # Plot with error bars
    plot_path = out_dir / "speed_benchmark.png"
    plot_results(results, plot_path, args.n_molecules, seeds)

    # Summary table
    print(f"\n{'='*78}")
    print(f"  GENERATION SPEED BENCHMARK  ({args.n_molecules} mol × {len(seeds)} seeds, {device_str})")
    print(f"{'='*78}")
    print(f"  {'Model':<22} {'Total (s)':>14} {'Per mol (s)':>16} {'mol/min':>14} {'Valid%':>12}")
    print(f"  {'-'*74}")
    for r in results:
        t = r["total_time_s"]
        p = r["per_molecule_s"]
        tp = r["molecules_per_min"]
        v = r["validity"]
        print(f"  {r['model']:<22} "
              f"{t['mean']:>6.2f}±{t['std']:<5.2f} "
              f"{p['mean']:>7.4f}±{p['std']:<6.4f} "
              f"{tp['mean']:>6.1f}±{tp['std']:<5.1f} "
              f"{v['mean']*100:>5.1f}±{v['std']*100:<4.1f}%")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
