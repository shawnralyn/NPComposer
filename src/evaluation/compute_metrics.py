import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm.auto import tqdm
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Contrib.NP_Score import npscorer

    NP_MODEL = npscorer.readNPModel()
except (ImportError, FileNotFoundError, OSError):
    NP_MODEL = None

# ClassyFire classification (optional)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "classification"))
try:
    from classyfire import classify_batch

    HAS_CLASSYFIRE = True
except ImportError:
    HAS_CLASSYFIRE = False

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ── Module-level caches ──────────────────────────────────────────────────────
_REF_FP_CACHE: Dict[str, List] = {}
_REF_SMILES_CACHE: Dict[str, set] = {}


def _get_first_str(x):
    if isinstance(x, list) and x:
        return x[0]
    if isinstance(x, str):
        return x
    return None


def npclassifier_classify(
    smiles: str,
    base_url: str = "https://npclassifier.ucsd.edu/classify",
    cached: bool = True,
    timeout: int = 30,
    max_retries: int = 3,
    sleep_s: float = 0.05,
) -> dict:
    if not HAS_REQUESTS:
        return {"error": "requests not installed"}

    params = {"smiles": smiles}
    if cached:
        params["cached"] = "true"

    last_err = None
    for attempt in range(max_retries):
        try:
            r = requests.get(base_url, params=params, timeout=timeout)
            r.raise_for_status()
            time.sleep(sleep_s)
            return r.json()
        except Exception as e:
            last_err = str(e)
            time.sleep(sleep_s * (2 ** attempt))

    return {"error": last_err}


def _smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def _batch_smiles_to_fps(smiles_list: List[str], desc: str = "Fingerprints") -> List:
    """Compute fingerprints for a list of SMILES, computing each only once."""
    fps = []
    for s in tqdm(smiles_list, desc=f"  {desc}", leave=False):
        fp = _smiles_to_fp(s)
        if fp is not None:
            fps.append(fp)
    return fps


def compute_internal_diversity(fps: List) -> Dict:
    """Compute pairwise Tanimoto similarity. Expects pre-computed fingerprints."""
    if len(fps) < 2:
        return {"mean": None, "std": None, "min": None, "max": None}
    sims = []
    for i in tqdm(range(len(fps)), desc="  Internal diversity", leave=False):
        for j in range(i + 1, len(fps)):
            sims.append(DataStructs.TanimotoSimilarity(fps[i], fps[j]))
    if not sims:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": round(float(np.mean(sims)), 3),
        "std": round(float(np.std(sims)), 3),
        "min": round(float(np.min(sims)), 3),
        "max": round(float(np.max(sims)), 3),
    }


def _load_smiles(csv_path: str, desc: str = "") -> List[str]:
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if "canonical_smiles" in df.columns:
            smiles = df["canonical_smiles"].dropna().astype(str).tolist()
        elif "smiles" in df.columns:
            smiles = df["smiles"].dropna().astype(str).tolist()
        else:
            smiles = df.iloc[:, 0].dropna().astype(str).tolist()
        print(f"  Loaded {len(smiles)} SMILES from {csv_path} ({desc})")
        return smiles
    except Exception as e:
        print(f"Warning: Could not load {desc} from {csv_path}: {e}")
        return []


def _load_smiles_set_cached(csv_path: str, desc: str = "") -> set:
    """Load SMILES as a set, cached across calls."""
    if csv_path in _REF_SMILES_CACHE:
        print(f"  Using cached SMILES set for {desc}")
        return _REF_SMILES_CACHE[csv_path]
    smiles = _load_smiles(csv_path, desc)
    s = set(smiles)
    _REF_SMILES_CACHE[csv_path] = s
    return s


def _load_ref_fps_cached(csv_path: str, desc: str = "") -> List:
    """Load and cache reference fingerprints (expensive, only done once)."""
    if csv_path in _REF_FP_CACHE:
        print(f"  Using cached fingerprints for {desc} ({len(_REF_FP_CACHE[csv_path])} fps)")
        return _REF_FP_CACHE[csv_path]
    smiles = _load_smiles(csv_path, desc)
    fps = _batch_smiles_to_fps(smiles, desc=f"Ref FPs ({desc})")
    _REF_FP_CACHE[csv_path] = fps
    return fps


def compute_uniqueness(valid_smiles: List[str], train_set: set) -> Dict:
    """Compute uniqueness against a pre-loaded training set."""
    valid_set = set(valid_smiles)
    unique = valid_set - train_set
    return {
        "n_unique": len(unique),
        "n_total": len(valid_smiles),
        "uniqueness": round(len(unique) / len(valid_smiles), 4) if valid_smiles else 0.0,
    }


def compute_novelty(fps_valid: List, fps_ref: List, n_valid: int, threshold: float = 0.6) -> Dict:
    """Compute novelty using pre-computed fingerprints."""
    if not fps_valid or not fps_ref:
        return {
            "n_novel": 0,
            "n_total": n_valid,
            "novelty": 0.0,
            "threshold": threshold,
            "nn_sim_mean": None,
            "nn_sim_std": None,
        }
    nn_sims = []
    for fp in tqdm(fps_valid, desc="  Novelty NN search", leave=False):
        sims = DataStructs.BulkTanimotoSimilarity(fp, fps_ref)
        nn_sims.append(max(sims))
    n_novel = sum(1 for sim in nn_sims if sim < threshold)
    return {
        "n_novel": n_novel,
        "n_total": n_valid,
        "novelty": round(n_novel / n_valid, 4) if n_valid else 0.0,
        "threshold": threshold,
        "novelty_at_0.4": round(sum(1 for s in nn_sims if s < 0.4) / n_valid, 4) if n_valid else 0.0,
        "novelty_at_0.5": round(sum(1 for s in nn_sims if s < 0.5) / n_valid, 4) if n_valid else 0.0,
        "novelty_at_0.6": round(sum(1 for s in nn_sims if s < 0.6) / n_valid, 4) if n_valid else 0.0,
        "novelty_at_0.7": round(sum(1 for s in nn_sims if s < 0.7) / n_valid, 4) if n_valid else 0.0,
        "nn_sim_mean": round(float(np.mean(nn_sims)), 3),
        "nn_sim_std": round(float(np.std(nn_sims)), 3),
    }


def evaluate(
    smiles_list: List[str],
    classify: bool = False,
    npclassify: bool = False,
    np_url: str = "https://npclassifier.ucsd.edu/classify",
    keep_np_per_mol: bool = False,
    training_file: str = None,
    novelty_file: str = None,
) -> Dict:
    """
    Compute validity, SA, QED, NP-likeness, internal diversity, uniqueness,
    novelty, and optionally superclass stats and NPClassifier distributions.
    """
    sa_scores = []
    qed_scores = []
    np_scores = []
    valid = 0
    valid_smiles = []

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None or mol.GetNumAtoms() == 0:
            continue

        valid += 1
        valid_smiles.append(s)
        sa_scores.append(sascorer.calculateScore(mol))
        qed_scores.append(QED.qed(mol))

        if NP_MODEL:
            try:
                np_scores.append(npscorer.scoreMol(mol, NP_MODEL))
            except (ValueError, ZeroDivisionError):
                pass

    n_total = len(smiles_list)

    results = {
        "n_total": n_total,
        "n_valid": valid,
        "validity": round(valid / n_total, 4) if n_total else 0,
        "sa_score": {
            "mean": round(float(np.mean(sa_scores)), 3) if sa_scores else None,
            "std": round(float(np.std(sa_scores)), 3) if sa_scores else None,
            "min": round(float(min(sa_scores)), 3) if sa_scores else None,
            "max": round(float(max(sa_scores)), 3) if sa_scores else None,
        },
        "qed": {
            "mean": round(float(np.mean(qed_scores)), 3) if qed_scores else None,
            "std": round(float(np.std(qed_scores)), 3) if qed_scores else None,
            "min": round(float(min(qed_scores)), 3) if qed_scores else None,
            "max": round(float(max(qed_scores)), 3) if qed_scores else None,
        },
        "np_score": {
            "mean": round(float(np.mean(np_scores)), 3) if np_scores else None,
            "std": round(float(np.std(np_scores)), 3) if np_scores else None,
            "min": round(float(min(np_scores)), 3) if np_scores else None,
            "max": round(float(max(np_scores)), 3) if np_scores else None,
        }
        if NP_MODEL
        else None,
    }

    if valid_smiles:
        # Compute fingerprints once for all downstream metrics
        print("Computing fingerprints for generated molecules...")
        fps_valid = _batch_smiles_to_fps(valid_smiles, desc="Generated FPs")

        # Internal diversity
        if len(fps_valid) > 1:
            print("Computing internal diversity...")
            results["internal_diversity"] = compute_internal_diversity(fps_valid)

        # Uniqueness (uses cached SMILES set)
        if training_file:
            print("Computing uniqueness...")
            train_set = _load_smiles_set_cached(training_file, "training data (full)")
            if train_set:
                results["uniqueness"] = compute_uniqueness(valid_smiles, train_set)

        # Novelty (uses cached reference fingerprints)
        nov_file = novelty_file or training_file
        if nov_file:
            print("Computing novelty...")
            fps_ref = _load_ref_fps_cached(nov_file, "novelty reference")
            if fps_ref:
                results["novelty"] = compute_novelty(fps_valid, fps_ref, len(valid_smiles))

    # ClassyFire superclass distribution
    if classify and valid_smiles:
        if HAS_CLASSYFIRE:
            print("Classifying superclass (ClassyFire API)...")
            superclasses = classify_batch(valid_smiles)
            dist = dict(Counter(superclasses))
            results["superclass_distribution"] = dist
            results["n_superclasses"] = len([k for k in dist if k != "Unknown"])
        else:
            print("Warning: ClassyFire not available (install 'requests')")

    # NPClassifier
    if npclassify and valid_smiles:
        if not HAS_REQUESTS:
            print("Warning: NPClassifier requires 'requests' (pip install requests)")
        else:
            print("Classifying with NPClassifier API...")
            pathway_dist = Counter()
            superclass_dist = Counter()
            class_dist = Counter()
            per_mol = [] if keep_np_per_mol else None

            for s in tqdm(valid_smiles, desc="  NPClassifier API", leave=False):
                res = npclassifier_classify(s, base_url=np_url, cached=True)

                if "error" in res:
                    if keep_np_per_mol:
                        per_mol.append({"smiles": s, "error": res["error"]})
                    continue

                pathway = _get_first_str(res.get("pathway_results")) or _get_first_str(
                    res.get("pathway")
                )
                superclass = _get_first_str(
                    res.get("superclass_results")
                ) or _get_first_str(res.get("superclass"))
                cls = _get_first_str(res.get("class_results")) or _get_first_str(
                    res.get("class")
                )

                if pathway:
                    pathway_dist[pathway] += 1
                if superclass:
                    superclass_dist[superclass] += 1
                if cls:
                    class_dist[cls] += 1

                if keep_np_per_mol:
                    per_mol.append(
                        {
                            "smiles": s,
                            "pathway": pathway,
                            "superclass": superclass,
                            "class": cls,
                        }
                    )

            results["npclassifier"] = {
                "pathway_distribution": dict(pathway_dist),
                "superclass_distribution": dict(superclass_dist),
                "class_distribution": dict(class_dist),
            }

            if keep_np_per_mol:
                results["npclassifier"]["per_molecule"] = per_mol

    return results


def read_smiles_file(input_path: Path) -> List[str]:
    with input_path.open("r") as f:
        smiles = [line.strip() for line in f if line.strip()]
    return smiles


def print_results_summary(results: Dict):
    print("\nResults:")
    print(f"  Valid: {results['n_valid']}/{results['n_total']} ({results['validity'] * 100:.1f}%)")
    print(f"  SA: {results['sa_score']['mean']} +/- {results['sa_score']['std']}")
    print(f"  QED: {results['qed']['mean']} +/- {results['qed']['std']}")

    if results.get("np_score"):
        print(f"  NP: {results['np_score']['mean']} +/- {results['np_score']['std']}")

    if "internal_diversity" in results and results["internal_diversity"]["mean"] is not None:
        d = results["internal_diversity"]
        print(f"  Internal diversity: {d['mean']} +/- {d['std']} (min={d['min']}, max={d['max']})")

    if "uniqueness" in results:
        u = results["uniqueness"]
        print(f"  Uniqueness (not in training): {u['n_unique']}/{u['n_total']} ({u['uniqueness'] * 100:.1f}%)")

    if "novelty" in results and results["novelty"]["n_total"] > 0:
        nov = results["novelty"]
        print(f"  Novelty (NN sim < {nov['threshold']}): {nov['n_novel']}/{nov['n_total']} ({nov['novelty'] * 100:.1f}%)")
        print(f"    NN similarity: {nov['nn_sim_mean']} +/- {nov['nn_sim_std']}")

    if "superclass_distribution" in results:
        print(f"  Superclasses ({results['n_superclasses']}):")
        for cls, count in sorted(
            results["superclass_distribution"].items(), key=lambda x: -x[1]
        )[:10]:
            print(f"    {cls}: {count}")

    if "npclassifier" in results and results["npclassifier"]:
        npd = results["npclassifier"]
        print("\nNPClassifier (top 10):")
        for name, dist in [
            ("Pathways", npd.get("pathway_distribution", {})),
            ("Superclasses", npd.get("superclass_distribution", {})),
            ("Classes", npd.get("class_distribution", {})),
        ]:
            if dist:
                print(f"  {name}:")
                for k, v in sorted(dist.items(), key=lambda x: -x[1])[:10]:
                    print(f"    {k}: {v}")


def evaluate_file(
    input_path: Path,
    output_path: Path = None,
    classify: bool = False,
    npclassify: bool = False,
    np_url: str = "https://npclassifier.ucsd.edu/classify",
    keep_np_per_mol: bool = False,
    training_file: str = None,
    novelty_file: str = None,
) -> Dict:
    smiles = read_smiles_file(input_path)

    if not smiles:
        raise ValueError(f"No SMILES found in input file: {input_path}")

    print(f"\nEvaluating file: {input_path}")
    print(f"Found {len(smiles)} molecules...")

    results = evaluate(
        smiles,
        classify=classify,
        npclassify=npclassify,
        np_url=np_url,
        keep_np_per_mol=keep_np_per_mol,
        training_file=training_file,
        novelty_file=novelty_file,
    )

    print_results_summary(results)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {output_path}")

    return results


def evaluate_directory(
    input_dir: Path,
    output_dir: Path,
    classify: bool = False,
    npclassify: bool = False,
    np_url: str = "https://npclassifier.ucsd.edu/classify",
    keep_np_per_mol: bool = False,
    training_file: str = None,
    novelty_file: str = None,
):
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        raise ValueError(f"No .txt files found in input directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(txt_files)} .txt files in {input_dir}")

    # Pre-warm caches before processing files
    if training_file:
        print("Pre-loading training SMILES set...")
        _load_smiles_set_cached(training_file, "training data (full)")

    nov_file = novelty_file or training_file
    if nov_file:
        print("Pre-loading novelty reference fingerprints (this may take a while)...")
        _load_ref_fps_cached(nov_file, "novelty reference")

    summary = []

    for i, input_path in enumerate(txt_files, start=1):
        output_path = output_dir / f"{input_path.stem}.json"
        if output_path.exists():
            print(f"[{i}/{len(txt_files)}] Skipping {input_path.name} (output exists)")
            summary.append(
                {
                    "input_file": input_path.name,
                    "output_json": output_path.name,
                    "skipped": True,
                }
            )
            continue

        print(f"\n[{i}/{len(txt_files)}] Processing {input_path.name}")

        try:
            results = evaluate_file(
                input_path=input_path,
                output_path=output_path,
                classify=classify,
                npclassify=npclassify,
                np_url=np_url,
                keep_np_per_mol=keep_np_per_mol,
                training_file=training_file,
                novelty_file=novelty_file,
            )
            summary.append(
                {
                    "input_file": input_path.name,
                    "output_json": output_path.name,
                    "n_total": results["n_total"],
                    "n_valid": results["n_valid"],
                    "validity": results["validity"],
                }
            )
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")
            summary.append(
                {
                    "input_file": input_path.name,
                    "output_json": output_path.name,
                    "error": str(e),
                }
            )

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved batch summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate generated molecules")

    parser.add_argument("-i", "--input", help="Single SMILES file (one per line)")
    parser.add_argument("--input_dir", help="Directory containing .txt SMILES files")
    parser.add_argument("-o", "--output", help="Output JSON for single-file mode")
    parser.add_argument("--output_dir", help="Directory for JSON outputs in batch mode")
    parser.add_argument("--classify", action="store_true", help="Compute ClassyFire superclass distribution")
    parser.add_argument("--npclassify", action="store_true", help="Run NPClassifier on valid SMILES via API")
    parser.add_argument("--np_url", default="https://npclassifier.ucsd.edu/classify", help="NPClassifier classify endpoint URL")
    parser.add_argument("--keep_np_per_mol", action="store_true", help="Store per-molecule NPClassifier assignments in output JSON")
    parser.add_argument("--training", default=None, help="Full training CSV for uniqueness (exact match)")
    parser.add_argument("--novelty_ref", default=None, help="K-means subset CSV for novelty (Tanimoto NN). Falls back to --training if not given.")

    args = parser.parse_args()

    if bool(args.input) == bool(args.input_dir):
        print("Error: provide exactly one of --input or --input_dir", file=sys.stderr)
        sys.exit(1)

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        output_path = Path(args.output) if args.output else None

        try:
            evaluate_file(
                input_path=input_path,
                output_path=output_path,
                classify=args.classify,
                npclassify=args.npclassify,
                np_url=args.np_url,
                keep_np_per_mol=args.keep_np_per_mol,
                training_file=args.training,
                novelty_file=args.novelty_ref,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    else:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
            sys.exit(1)

        if not args.output_dir:
            print("Error: --output_dir is required when using --input_dir", file=sys.stderr)
            sys.exit(1)

        output_dir = Path(args.output_dir)

        try:
            evaluate_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                classify=args.classify,
                npclassify=args.npclassify,
                np_url=args.np_url,
                keep_np_per_mol=args.keep_np_per_mol,
                training_file=args.training,
                novelty_file=args.novelty_ref,
            )
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()