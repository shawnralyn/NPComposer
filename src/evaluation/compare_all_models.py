"""Compare all molecular generation models with metrics and statistical significance testing."""

import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


def _safe_load(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _extract_per_seed_values(per_seed_dict, metric_key):
    """Extract per-seed values for metric.

    Input:
        per_seed_dict: {seed: metric_dict}.
        metric_key: metric name.
    Output:
        list of scalar values."""
    vals = []
    for seed_key in sorted(per_seed_dict.keys(), key=str):
        m = per_seed_dict[seed_key]
        if metric_key == "validity":
            vals.append(m.get("validity", 0.0))
        elif metric_key in ("qed", "sa", "np_likeness", "internal_diversity"):
            sub = m.get(metric_key)
            if sub and isinstance(sub, dict) and sub.get("mean") is not None:
                vals.append(sub["mean"])
        elif metric_key in ("uniqueness", "novelty"):
            sub = m.get(metric_key)
            if sub and isinstance(sub, dict):
                vals.append(sub.get("ratio", sub.get("mean", 0.0)))
    return vals


class ModelResult:
    """Container for model evaluation results.

    Attributes:
        name: display name.
        color: bar color.
        aggregate: metric dict with mean/std.
        per_seed_values: per-seed metric values for testing.
    """

    def __init__(self, name, color, aggregate, per_seed_values=None):
        self.name = name
        self.color = color
        self.aggregate = aggregate
        self.per_seed_values = per_seed_values or {}

    def get_mean(self, key):
        a = self.aggregate.get(key)
        if a and a.get("mean") is not None:
            return a["mean"]
        return None

    def get_std(self, key):
        a = self.aggregate.get(key)
        if a and a.get("std") is not None:
            return a["std"]
        return None

    def get_values(self, key):
        """Get per-seed values for statistical testing.

        Input:
            key: metric name.
        Output:
            list of per-seed values or None."""
        if key in self.per_seed_values and self.per_seed_values[key]:
            return self.per_seed_values[key]
        a = self.aggregate.get(key)
        if a and "values" in a and a["values"]:
            return a["values"]
        return None


def load_all_models(npgpt_dir, shawn_dir):
    """Load results from all evaluation JSONs.

    Input:
        npgpt_dir, shawn_dir: result directories.
    Output:
        list of ModelResult objects."""
    models = []
    npgpt_dir = Path(npgpt_dir)
    shawn_dir = Path(shawn_dir)

    # Color palette
    C = {
        "npgpt_pre": "#5DADE2",
        "npgpt_rl":  "#1F618D",
        "gpmol":     "#F39C12",
        "shawn_qed": "#E74C3C",
        "shawn_sa":  "#3498DB",
        "shawn_opt": "#9B59B6",
        "shawn_pw":  "#2ECC71",
    }

    # ── 1 & 2. NPGPT pretrained & RL ─────────────────────────────────────
    rl_data = _safe_load(npgpt_dir / "npgpt_rl_results.json")
    if rl_data:
        for mn, display, color in [
            ("pretrained",   "NPGPT (Pretrained)", C["npgpt_pre"]),
            ("rl_finetuned", "NPGPT (RL)",         C["npgpt_rl"]),
        ]:
            agg = rl_data["aggregate"].get(mn, {})
            per_seed = rl_data.get("per_seed", {}).get(mn, {})
            psv = {}
            for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity",
                        "uniqueness", "novelty"]:
                vals = agg.get(mk, {}).get("values")
                if not vals:
                    vals = _extract_per_seed_values(per_seed, mk)
                if vals:
                    psv[mk] = vals
            models.append(ModelResult(display, color, agg, psv))
    else:
        # Try standalone npgpt
        npgpt_data = _safe_load(npgpt_dir / "npgpt_results.json")
        if npgpt_data:
            agg = npgpt_data.get("aggregate", {})
            per_seed = npgpt_data.get("per_seed", {})
            psv = {}
            for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
                vals = agg.get(mk, {}).get("values")
                if not vals:
                    vals = _extract_per_seed_values(per_seed, mk)
                if vals:
                    psv[mk] = vals
            models.append(ModelResult("NPGPT (Pretrained)", C["npgpt_pre"], agg, psv))

    # ── 3. GP-MoLFormer ──────────────────────────────────────────────────
    gp_data = _safe_load(npgpt_dir / "gpmolformer_results.json")
    if gp_data:
        agg = gp_data.get("aggregate", {})
        per_seed = gp_data.get("per_seed", {})
        psv = {}
        for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
            vals = agg.get(mk, {}).get("values")
            if not vals:
                vals = _extract_per_seed_values(per_seed, mk)
            if vals:
                psv[mk] = vals
        models.append(ModelResult("GP-MoLFormer", C["gpmol"], agg, psv))

    # ── 4-7. NPComposer (Shawn) configs ──────────────────────────────────
    shawn_summary = _safe_load(shawn_dir / "shawn_summary.json")
    shawn_config_map = {
        "optimal_params":      ("NPComposer (QED+SA)",        C["shawn_opt"]),
        "pathway_alkaloids":   ("NPComposer (Alkaloids)",     C["shawn_qed"]),
        "pathway_terpenoids":  ("NPComposer (Terpenoids)",    C["shawn_sa"]),
        "pathway_shikimates":  ("NPComposer (Shikimates)",    C["shawn_pw"]),
    }

    if shawn_summary:
        agg_by_config = shawn_summary.get("aggregate_by_config", {})
        for cfg_name, (display, color) in shawn_config_map.items():
            agg = agg_by_config.get(cfg_name)
            if not agg:
                continue
            # Try to load per-run details from individual config JSON
            cfg_json = _safe_load(shawn_dir / f"shawn_{cfg_name}_results.json")
            psv = {}
            if cfg_json:
                per_run = cfg_json.get("per_seed") or cfg_json.get("per_run", {})
                for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
                    vals = agg.get(mk, {}).get("values")
                    if not vals:
                        vals = _extract_per_seed_values(per_run, mk)
                    if vals:
                        psv[mk] = vals
            models.append(ModelResult(display, color, agg, psv))
    else:
        # Try loading individual config JSONs
        for cfg_name, (display, color) in shawn_config_map.items():
            cfg_json = _safe_load(shawn_dir / f"shawn_{cfg_name}_results.json")
            if cfg_json:
                agg = cfg_json.get("aggregate", {})
                per_run = cfg_json.get("per_seed") or cfg_json.get("per_run", {})
                psv = {}
                for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
                    vals = agg.get(mk, {}).get("values")
                    if not vals:
                        vals = _extract_per_seed_values(per_run, mk)
                    if vals:
                        psv[mk] = vals
                models.append(ModelResult(display, color, agg, psv))

    return models


def welch_ttest(vals_a, vals_b):
    """Welch's t-test (two-tailed).

    Input:
        vals_a, vals_b: value lists.
    Output:
        p-value or None."""
    if vals_a is None or vals_b is None:
        return None
    if len(vals_a) < 2 or len(vals_b) < 2:
        return None
    try:
        _, p = stats.ttest_ind(vals_a, vals_b, equal_var=False)
        return p
    except Exception:
        return None


def significance_stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _parse_pairs(pair_str, n_models):
    """Parse pair specification string.

    Input:
        pair_str: 'all', 'adjacent', or '0-1,0-2'.
        n_models: number of models.
    Output:
        list of (i, j) index pairs."""
    if pair_str == "all":
        return list(itertools.combinations(range(n_models), 2))
    if pair_str and pair_str != "adjacent":
        pairs = []
        for p in pair_str.split(","):
            a, b = p.strip().split("-")
            pairs.append((int(a), int(b)))
        return pairs
    # Default: adjacent pairs
    return [(i, i + 1) for i in range(n_models - 1)]


def _find_ref_model(models, ref_name):
    """Find reference model index by name.

    Input:
        models: list of ModelResult.
        ref_name: model name to search.
    Output:
        index or None."""
    ref_lower = ref_name.lower()
    for i, m in enumerate(models):
        if ref_lower in m.name.lower():
            return i
    return None


METRICS = [
    ("validity",           "Validity (%)",     True),   # (key, display, is_percentage)
    ("qed",                "QED",              False),
    ("sa",                 "SA Score (raw)",   False),
    ("np_likeness",        "NP-likeness",      False),
    ("internal_diversity", "Internal Diversity", False),
]


# ---------------------------------------------------------------------------
# Speed benchmark loading & plotting
# ---------------------------------------------------------------------------

def load_speed_benchmark(benchmark_dir):
    """Load speed benchmark results.

    Input:
        benchmark_dir: path to results/benchmark directory.
    Output:
        dict with config and results, or None.
    """
    path = Path(benchmark_dir) / "speed_benchmark.json"
    return _safe_load(path)


def _extract_mean_std(value):
    """Extract mean and std from a benchmark value (supports both old and new formats).

    Input:
        value: scalar (old format) or dict with mean/std (new format).
    Output:
        (mean, std) tuple.
    """
    if isinstance(value, dict):
        return value.get("mean", 0), value.get("std", 0)
    return value, 0


def plot_speed_panel(speed_data, models, out_path):
    """Plot generation speed with error bars.

    Input:
        speed_data: loaded speed benchmark JSON.
        models: list of ModelResult (for consistent color mapping).
        out_path: output PNG path.
    """
    if not speed_data or "results" not in speed_data:
        return

    bench_results = speed_data["results"]
    cfg = speed_data.get("config", {})
    device_str = cfg.get("device", "unknown")
    n_mol = cfg.get("n_molecules", "?")
    seeds = cfg.get("seeds", [])

    # Build name->color map from ModelResult for consistent colors
    color_map = {}
    for m in models:
        name_lower = m.name.lower()
        color_map[name_lower] = m.color
    fallback_colors = ["#5DADE2", "#1F618D", "#F39C12", "#9B59B6", "#2ECC71"]

    names, pm_means, pm_stds = [], [], []
    tp_means, tp_stds = [], []
    tt_means, tt_stds = [], []
    colors = []

    for i, r in enumerate(bench_results):
        bname = r["model"]
        names.append(bname)

        m, s = _extract_mean_std(r["per_molecule_s"])
        pm_means.append(m); pm_stds.append(s)
        m, s = _extract_mean_std(r["molecules_per_min"])
        tp_means.append(m); tp_stds.append(s)
        m, s = _extract_mean_std(r["total_time_s"])
        tt_means.append(m); tt_stds.append(s)

        # Match color
        matched = False
        for key, col in color_map.items():
            bn = bname.lower().replace(" ", "").replace("(", "").replace(")", "")
            kn = key.replace(" ", "").replace("(", "").replace(")", "")
            if bn in kn or kn in bn:
                colors.append(col)
                matched = True
                break
        if not matched:
            colors.append(fallback_colors[i % len(fallback_colors)])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))

    # Panel 1: Per-molecule time
    bars = axes[0].bar(names, pm_means, yerr=pm_stds, capsize=5,
                       color=colors, edgecolor="white", alpha=0.88,
                       error_kw={"lw": 1.5, "capthick": 1.5})
    axes[0].set_ylabel("Time per molecule (s)")
    axes[0].set_title("Generation Speed", fontsize=12, fontweight="bold")
    for bar, m, s in zip(bars, pm_means, pm_stds):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.002,
                     f"{m:.3f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 2: Throughput
    bars = axes[1].bar(names, tp_means, yerr=tp_stds, capsize=5,
                       color=colors, edgecolor="white", alpha=0.88,
                       error_kw={"lw": 1.5, "capthick": 1.5})
    axes[1].set_ylabel("Molecules / min")
    axes[1].set_title("Throughput", fontsize=12, fontweight="bold")
    for bar, m, s in zip(bars, tp_means, tp_stds):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.5,
                     f"{m:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Panel 3: Total time
    bars = axes[2].bar(names, tt_means, yerr=tt_stds, capsize=5,
                       color=colors, edgecolor="white", alpha=0.88,
                       error_kw={"lw": 1.5, "capthick": 1.5})
    axes[2].set_ylabel("Total time (s)")
    axes[2].set_title(f"Total Time ({n_mol} mol/seed)", fontsize=12, fontweight="bold")
    for bar, m, s in zip(bars, tt_means, tt_stds):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + s + 0.2,
                     f"{m:.1f}s", ha="center", va="bottom", fontsize=8, fontweight="bold")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    seed_info = f"seeds {seeds}" if seeds else ""
    fig.suptitle(f"Generation Speed Benchmark  —  {device_str}  ({seed_info})",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Speed benchmark plot saved -> {out_path}")


def print_speed_table(speed_data):
    """Print speed benchmark as formatted table with mean ± std.

    Input:
        speed_data: loaded speed benchmark JSON.
    """
    if not speed_data or "results" not in speed_data:
        return

    cfg = speed_data.get("config", {})
    device_str = cfg.get("device", "unknown")
    n_mol = cfg.get("n_molecules", "?")
    seeds = cfg.get("seeds", [])

    seed_str = f"{n_mol} mol × {len(seeds)} seeds" if seeds else f"{n_mol} molecules"
    print(f"\n{'='*78}")
    print(f"  GENERATION SPEED BENCHMARK  ({seed_str}, {device_str})")
    print(f"{'='*78}")
    print(f"  {'Model':<25} {'Total (s)':>14} {'Per mol (s)':>16} {'mol/min':>14} {'Valid%':>12}")
    print(f"  {'-'*74}")
    for r in speed_data["results"]:
        t_m, t_s = _extract_mean_std(r["total_time_s"])
        p_m, p_s = _extract_mean_std(r["per_molecule_s"])
        tp_m, tp_s = _extract_mean_std(r["molecules_per_min"])
        v_m, v_s = _extract_mean_std(r.get("validity", 0))
        if t_s > 0:
            print(f"  {r['model']:<25} "
                  f"{t_m:>6.2f}±{t_s:<5.2f} "
                  f"{p_m:>7.4f}±{p_s:<6.4f} "
                  f"{tp_m:>6.1f}±{tp_s:<5.1f} "
                  f"{v_m*100:>5.1f}±{v_s*100:<4.1f}%")
        else:
            print(f"  {r['model']:<25} {t_m:>14.2f} {p_m:>16.4f} "
                  f"{tp_m:>14.1f} {v_m*100:>11.1f}%")
    print(f"{'='*78}")


def plot_comparison(models, metric_pairs, out_path, title=None):
    """Plot multi-panel comparison of all models.

    Input:
        models: list of ModelResult.
        metric_pairs: significance test pairs.
        out_path: output file path.
        title: optional figure title.
    Output:
        none (saves PNG)."""
    # Filter metrics to only those that have at least one model with data
    active_metrics = []
    for key, display, is_pct in METRICS:
        if any(m.get_mean(key) is not None for m in models):
            active_metrics.append((key, display, is_pct))

    n_metrics = len(active_metrics)
    if n_metrics == 0:
        print("No metrics found to plot.")
        return

    n_models = len(models)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 7))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(n_models)
    bar_width = 0.6

    for ax, (key, display, is_pct) in zip(axes, active_metrics):
        means = []
        stds = []
        colors = []
        for m in models:
            val = m.get_mean(key)
            err = m.get_std(key)
            if val is not None:
                means.append(val * 100 if is_pct else val)
                stds.append((err * 100 if is_pct else err) if err else 0)
            else:
                means.append(0)
                stds.append(0)
            colors.append(m.color)

        bars = ax.bar(x, means, yerr=stds, capsize=3, color=colors,
                      edgecolor="white", alpha=0.88, width=bar_width,
                      error_kw={"lw": 1.2, "capthick": 1.2})

        # Value labels directly above bars
        y_max = max(abs(m) + e for m, e in zip(means, stds)) if means else 1
        pad = y_max * 0.02
        for bar, val, err in zip(bars, means, stds):
            if val == 0:
                continue
            if is_pct:
                txt = f"{val:.1f}%"
            else:
                txt = f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + pad,
                    txt, ha="center", va="bottom", fontsize=7, fontweight="bold")

        # Axis formatting
        ax.set_xticks(x)
        ax.set_xticklabels([m.name for m in models], fontsize=7.5,
                           rotation=35, ha="right")
        ax.set_title(display, fontsize=12, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

        if is_pct:
            ax.set_ylim(0, max(110, y_max * 1.15))
            ax.set_ylabel("%")
        else:
            ax.set_ylim(bottom=None, top=y_max * 1.15)

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    else:
        fig.suptitle("Molecular Generation Model Comparison", fontsize=14, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved -> {out_path}")


def print_summary_table(models, metric_pairs):
    """Print formatted comparison table.

    Input:
        models: list of ModelResult.
        metric_pairs: significance test pairs.
    Output:
        none (prints to stdout)."""
    metric_keys = ["validity", "qed", "sa", "np_likeness", "internal_diversity"]
    header = f"{'Model':<25}"
    for mk in metric_keys:
        header += f" {mk:>15}"
    print(f"\n{'='*100}")
    print("COMPARISON TABLE")
    print(f"{'='*100}")
    print(header)
    print("-" * 100)

    for m in models:
        row = f"{m.name:<25}"
        for mk in metric_keys:
            val = m.get_mean(mk)
            err = m.get_std(mk)
            if val is not None:
                if mk == "validity":
                    row += f" {val*100:>6.1f}±{err*100:>4.1f}%"
                else:
                    row += f" {val:>6.4f}±{err:>5.4f}"
            else:
                row += f" {'N/A':>15}"
        print(row)
    print("=" * 100)

    # Significance pairs
    if metric_pairs:
        print(f"\nSignificance tests (Welch's t-test, two-tailed):")
        print(f"{'Pair':<50} {'Metric':<18} {'p-value':>10} {'Sig':>5}")
        print("-" * 85)
        for mk in metric_keys:
            for (i, j) in metric_pairs:
                if i >= len(models) or j >= len(models):
                    continue
                vals_a = models[i].get_values(mk)
                vals_b = models[j].get_values(mk)
                p = welch_ttest(vals_a, vals_b)
                stars = significance_stars(p)
                if p is not None:
                    pair_name = f"{models[i].name} vs {models[j].name}"
                    print(f"{pair_name:<50} {mk:<18} {p:>10.6f} {stars:>5}")
        print("=" * 85)


def main():
    parser = argparse.ArgumentParser(
        description="Compare all molecular generation models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_all_models.py
  python compare_all_models.py --ref_model "NPGPT (RL)"
  python compare_all_models.py --pairs "0-1,0-2,1-3,1-4"
  python compare_all_models.py --pairs all
        """,
    )
    parser.add_argument("--npgpt_dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "evaluation"),
                        help="Directory with NPGPT/GP-MoLFormer results")
    parser.add_argument("--shawn_dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "evaluation_shawn"),
                        help="Directory with NPComposer (Shawn) results")
    parser.add_argument("--out_dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "comparison"),
                        help="Output directory for comparison plots")
    parser.add_argument("--ref_model", type=str, default=None,
                        help="Reference model name for pairwise comparisons "
                             "(e.g., 'NPGPT (RL)' — compare all others against this)")
    parser.add_argument("--pairs", type=str, default=None,
                        help="Pairs for significance testing: "
                             "'all', 'adjacent', or '0-1,0-2,...' (0-indexed)")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom figure title")
    parser.add_argument("--benchmark_dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "benchmark"),
                        help="Directory with speed benchmark results")
    parser.add_argument("--save_json", action="store_true", default=True,
                        help="Save comparison summary JSON")

    args = parser.parse_args()

    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all models
    print("Loading evaluation results...")
    models = load_all_models(args.npgpt_dir, args.shawn_dir)

    if not models:
        print("ERROR: No evaluation results found!")
        print(f"  Checked: {args.npgpt_dir}")
        print(f"  Checked: {args.shawn_dir}")
        print("\nRun the evaluation scripts first:")
        print("  python evaluate.py npgpt-rl")
        print("  python evaluate.py gpmolformer")
        print("  python evaluate_shawn.py")
        return

    print(f"\nLoaded {len(models)} models:")
    for i, m in enumerate(models):
        print(f"  [{i}] {m.name}")

    # Determine significance pairs
    if args.ref_model:
        ref_idx = _find_ref_model(models, args.ref_model)
        if ref_idx is None:
            print(f"\nWARNING: Reference model '{args.ref_model}' not found. Using adjacent pairs.")
            metric_pairs = _parse_pairs(None, len(models))
        else:
            print(f"\nReference model: [{ref_idx}] {models[ref_idx].name}")
            metric_pairs = [(ref_idx, j) for j in range(len(models)) if j != ref_idx]
    elif args.pairs:
        metric_pairs = _parse_pairs(args.pairs, len(models))
    else:
        # Default: compare adjacent models
        metric_pairs = _parse_pairs(None, len(models))

    print(f"Significance pairs: {metric_pairs}")

    # Plot
    plot_comparison(
        models, metric_pairs,
        OUT_DIR / "all_models_comparison.png",
        title=args.title,
    )

    # Table
    print_summary_table(models, metric_pairs)

    # JSON summary
    if args.save_json:
        summary = {
            "models": [
                {"index": i, "name": m.name, "aggregate": m.aggregate}
                for i, m in enumerate(models)
            ],
            "significance_pairs": [
                {"pair": f"{models[i].name} vs {models[j].name}", "i": i, "j": j}
                for i, j in metric_pairs
                if i < len(models) and j < len(models)
            ],
        }

        # Add p-values
        for mk in ["validity", "qed", "sa", "np_likeness", "internal_diversity"]:
            for entry in summary["significance_pairs"]:
                i, j = entry["i"], entry["j"]
                vals_a = models[i].get_values(mk)
                vals_b = models[j].get_values(mk)
                p = welch_ttest(vals_a, vals_b)
                entry[f"{mk}_pvalue"] = p
                entry[f"{mk}_sig"] = significance_stars(p)

        # Include speed data in JSON if available
        speed_data = load_speed_benchmark(args.benchmark_dir)
        if speed_data:
            summary["speed_benchmark"] = speed_data

        json_path = OUT_DIR / "comparison_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nJSON summary -> {json_path}")

    # ── Speed benchmark ──────────────────────────────────────────────────
    speed_data = load_speed_benchmark(args.benchmark_dir)
    if speed_data:
        plot_speed_panel(speed_data, models, OUT_DIR / "speed_benchmark_comparison.png")
        print_speed_table(speed_data)
    else:
        print(f"\nNo speed benchmark found at {args.benchmark_dir}/speed_benchmark.json")
        print("  Run: python src/evaluation/benchmark_speed.py")

    print("\nDone!")


if __name__ == "__main__":
    main()
