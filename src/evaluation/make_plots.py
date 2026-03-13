#!/usr/bin/env python3
"""
Plot validity, conditioning adherence, uniqueness, and novelty from JSON metric files.

Expected filename pattern:
    np_classifier_<level>_<label>.json

Examples:
    np_classifier_pathway_Alkaloids.json
    np_classifier_superclass_Small_peptides.json
    np_classifier_class_Isoquinoline_alkaloids.json

This script:
1. Reads only .json files from a directory
2. Extracts:
    - n_total
    - n_valid
    - conditioning level from filename (pathway/superclass/class)
    - conditioning label from filename
    - uniqueness and novelty metrics
3. Plots:
    - validity = n_valid / n_total
    - conditioning adherence = matched_count / n_valid
    - uniqueness = n_unique / n_total
    - novelty = n_novel / n_total
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


LEVEL_TO_DIST_KEY = {
    "pathway": "pathway_distribution",
    "superclass": "superclass_distribution",
    "class": "class_distribution",
}


def parse_filename(json_path: Path):
    """
    Parse filename like:
        np_classifier_pathway_Alkaloids.json

    Returns:
        level: pathway / superclass / class
        label: Alkaloids
        label_for_json: Alkaloids (underscores replaced with spaces)
    """
    stem = json_path.stem
    parts = stem.split("_")

    if len(parts) < 4:
        raise ValueError(
            f"Filename does not match expected pattern 'np_classifier_<level>_<label>.json': {json_path.name}"
        )

    level = parts[2]
    if level not in LEVEL_TO_DIST_KEY:
        raise ValueError(
            f"Unrecognized conditioning level '{level}' in filename: {json_path.name}"
        )

    raw_label = "_".join(parts[3:])
    plot_label = raw_label.replace("_", " ")
    json_label = plot_label

    return level, plot_label, json_label


def normalize_label(label):
    label = label.replace("_", " ")
    label = label.lower()
    label = re.sub(r"\s+", " ", label).strip()
    return label


def normalize_json_key(key):
    key = key.replace("_", " ").lower()
    m = re.match(r"(.*)\s*\(([^)]+)\)", key)
    if m:
        base = m.group(1).strip()
        paren = m.group(2).strip()
        key = f"{base} {paren}"
    else:
        key = key.strip()
    key = re.sub(r"\s+", " ", key)
    return key


def load_json_metrics(json_path: Path):
    """
    Load one JSON file and return summary info needed for plotting.
    Handles label mismatches (e.g., underscores, parentheses).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    n_total = data.get("n_total", 0)
    n_valid = data.get("n_valid", 0)

    level, plot_label, json_label = parse_filename(json_path)
    dist_key = LEVEL_TO_DIST_KEY[level]
    dist = data.get("npclassifier", {}).get(dist_key, {})

    norm_label = normalize_label(json_label)
    matched_count = 0

    # Try direct match first
    for k, v in dist.items():
        norm_key = normalize_json_key(k)
        if norm_key == norm_label:
            matched_count = v
            break

    # If not found, try partial match
    if matched_count == 0:
        for k, v in dist.items():
            norm_key = normalize_json_key(k)
            if norm_label in norm_key or norm_key in norm_label:
                matched_count = v
                break

    validity = n_valid / n_total if n_total > 0 else 0.0
    adherence = matched_count / n_valid if n_valid > 0 else 0.0

    # Uniqueness
    uniq_data = data.get("uniqueness", {})
    n_unique = uniq_data.get("n_unique", 0)
    uniqueness = n_unique / n_valid if n_valid > 0 else 0.0

    # Novelty
    nov_data = data.get("novelty", {})
    n_novel = nov_data.get("n_novel", 0)
    novelty = n_novel / n_valid if n_valid > 0 else 0.0
    novelty_threshold = nov_data.get("threshold", None)

    return {
        "file": json_path.name,
        "label": plot_label,
        "level": level,
        "n_total": n_total,
        "n_valid": n_valid,
        "matched_count": matched_count,
        "validity": validity,
        "adherence": adherence,
        "n_unique": n_unique,
        "uniqueness": uniqueness,
        "n_novel": n_novel,
        "novelty": novelty,
        "novelty_threshold": novelty_threshold,
    }


def plot_validity(results, out_path: Path):
    """
    Plot n_valid / n_total for each file (horizontal bar).
    """
    labels = [r["label"] for r in results]
    values = [r["validity"] for r in results]
    annotations = [f'{r["n_valid"]}/{r["n_total"]}' for r in results]

    plt.figure(figsize=(6, max(8, len(labels) * 0.5)))
    bars = plt.barh(labels, values)
    plt.xlabel("Valid fraction")
    plt.ylabel("Condition")
    plt.title("Validity: n_valid / n_total")
    plt.xlim(0, 1.05)

    for bar, text in zip(bars, annotations):
        plt.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_adherence(results, out_path: Path):
    """
    Plot matched_count / n_valid for each file (horizontal bar).
    """
    labels = [r["label"] for r in results]
    values = [r["adherence"] for r in results]
    annotations = [f'{r["matched_count"]}/{r["n_valid"]}' for r in results]

    plt.figure(figsize=(6, max(8, len(labels) * 0.5)))
    bars = plt.barh(labels, values)
    plt.xlabel("Conditioning adherence")
    plt.ylabel("Condition")
    plt.title("Conditioning adherence: matched_count / n_valid")
    plt.xlim(0, 1.05)

    for bar, text in zip(bars, annotations):
        plt.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uniqueness(results, out_path: Path):
    """
    Plot n_unique / n_valid for each file (horizontal bar).
    """
    labels = [r["label"] for r in results]
    values = [r["uniqueness"] for r in results]
    annotations = [f'{r["n_unique"]}/{r["n_valid"]}' for r in results]

    plt.figure(figsize=(6, max(8, len(labels) * 0.5)))
    bars = plt.barh(labels, values, color="teal")
    plt.xlabel("Uniqueness (not in training)")
    plt.ylabel("Condition")
    plt.title("Uniqueness: n_unique / n_valid")
    plt.xlim(0, 1.05)

    for bar, text in zip(bars, annotations):
        plt.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_novelty(results, out_path: Path):
    """
    Plot n_novel / n_valid for each file (horizontal bar).
    """
    labels = [r["label"] for r in results]
    values = [r["novelty"] for r in results]
    thresholds = [r.get("novelty_threshold") for r in results]
    annotations = [f'{r["n_novel"]}/{r["n_valid"]}' for r in results]

    # Use the first available threshold for the title
    threshold_str = ""
    for t in thresholds:
        if t is not None:
            threshold_str = f" (NN sim < {t})"
            break

    plt.figure(figsize=(6, max(8, len(labels) * 0.5)))
    bars = plt.barh(labels, values, color="coral")
    plt.xlabel("Novelty fraction")
    plt.ylabel("Condition")
    plt.title(f"Novelty: n_novel / n_valid{threshold_str}")
    plt.xlim(0, 1.05)

    for bar, text in zip(bars, annotations):
        plt.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            text,
            va="center",
            ha="left",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot validity, adherence, uniqueness, and novelty from JSON files."
    )
    parser.add_argument(
        "--json_dir",
        type=Path,
        required=True,
        help="Directory containing JSON files",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=True,
        help="Directory to save plots",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted([p for p in args.json_dir.iterdir() if p.suffix == ".json"])

    if not json_files:
        raise ValueError(f"No .json files found in {args.json_dir}")

    results = []
    for json_file in json_files:
        try:
            results.append(load_json_metrics(json_file))
        except Exception as e:
            print(f"Skipping {json_file.name}: {e}")

    if not results:
        raise ValueError("No valid JSON files could be processed.")

    # Filter out results that have uniqueness/novelty data
    has_uniqueness = [r for r in results if r["n_unique"] > 0 or r["uniqueness"] > 0]
    has_novelty = [r for r in results if r["n_novel"] > 0 or r["novelty"] > 0 or r["novelty_threshold"] is not None]

    plot_validity(results, args.out_dir / "validity_by_condition.png")
    plot_adherence(results, args.out_dir / "conditioning_adherence_by_condition.png")

    if has_uniqueness:
        plot_uniqueness(results, args.out_dir / "uniqueness_by_condition.png")
        print(f"Saved: {args.out_dir / 'uniqueness_by_condition.png'}")

    if has_novelty:
        plot_novelty(results, args.out_dir / "novelty_by_condition.png")
        print(f"Saved: {args.out_dir / 'novelty_by_condition.png'}")

    print(f"Processed {len(results)} JSON files")
    print(f"Saved: {args.out_dir / 'validity_by_condition.png'}")
    print(f"Saved: {args.out_dir / 'conditioning_adherence_by_condition.png'}")


if __name__ == "__main__":
    main()