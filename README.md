# NPComposer

Natural product generation using COCONUT and NPASS datasets.

## Structure

```
NPComposer/
├── conf/
│   └── config.yaml             # Hydra configuration
├── data/
│   ├── raw/                    # Original data
│   │   ├── npass/              # NPASS files
│   │   ├── coconut_csv_full.csv
│   │   └── coconut_sdf_3d_full.sdf
│   ├── processed/              # Filtered subsets + training_data.csv
│   └── splits/                 # Train/Val/Test
├── scripts/
│   ├── download_data.sh        # Download COCONUT
│   ├── download_npass.sh       # Download NPASS
│   ├── merge_npass.py          # Merge NPASS files
│   ├── create_subset.py        # Create subset (K-means)
│   ├── merge_training.py       # Merge subsets into training data
│   ├── split_data.py           # Train/Val/Test split
│   ├── analyze_distribution.py # Raw vs processed distribution analysis
│   └── analyze_sdf.py          # Analyze SDF structure
├── src/
│   ├── classification/
│   │   ├── npclassifier.py     # NPClassifier local inference
│   │   └── classyfire.py       # ClassyFire API (deprecated)
│   └── evaluation/
│       └── metrics.py          # Evaluation metrics
├── tests/                      # pytest test suite
├── Makefile                    # Build automation
├── npcomposer.def              # Apptainer container definition
└── requirements.txt
```

## Quick Start

```bash
make all
```

Runs: setup -> download -> subset -> split -> merge -> test.

For individual datasets:

```bash
make all-coconut
make all-npass
```

To override defaults:

```bash
make subset-coconut SIZE=100000 SA_MAX=5.0 SEED=42
```

## NPClassifier Setup

NPClassifier runs locally (no server, no network). One-time setup:

```bash
git clone https://github.com/mwang87/NP-Classifier
cd NP-Classifier/Classifier/models_folder/models
wget -O models.zip "https://zenodo.org/record/5068687/files/model.zip?download=1"
unzip models.zip
```

Set the environment variable:

```bash
export NP_CLASSIFIER_ROOT=/path/to/NP-Classifier
```

Or pass `--np_root` to scripts directly.

### NPClassifier in the Pipeline

Classification is enabled by default (`CLASSIFY=true`). Set `NP_CLASSIFIER_ROOT` before running:

```bash
export NP_CLASSIFIER_ROOT=~/NP-Classifier
make all
```

To skip classification:

```bash
make all CLASSIFY=false
```

### Standalone Evaluation with Classification

```bash
python src/evaluation/metrics.py -i generated.txt -o results.json \
    --classify --np_root ~/NP-Classifier
```

## Setup

```bash
make setup
# Or
pip install -r requirements.txt
```

Requires `tensorflow` for NPClassifier local inference (already in requirements.txt).

## Test

```bash
make test
```

Skip slow tests:

```bash
make test-quick
```

## Manual Usage

```bash
# COCONUT
bash scripts/download_data.sh
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv \
    --sdf data/raw/coconut_sdf_3d_full.sdf \
    -o data/processed/coconut_100000 -s 100000 --classify --seed 42

# NPASS
bash scripts/download_npass.sh
python scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv
python scripts/create_subset.py -i data/raw/npass_full.csv \
    -o data/processed/npass_100000 -s 100000 --classify --seed 42

# Merge training data
python scripts/merge_training.py \
    --coconut data/processed/coconut_100000.csv \
    --npass data/processed/npass_100000.csv \
    -o data/processed/training_data.csv

# Split & Evaluate
python scripts/split_data.py -i data/processed/coconut_100000.csv -o data/splits/ --seed 42
python src/evaluation/metrics.py -i generated.txt -o results.json --classify
```

## Apptainer

```bash
make apptainer
make shell
apptainer exec --bind ./data:/app/data npcomposer.sif make pipeline-coconut
```

## Pipeline

```
Raw Data (COCONUT 715K / NPASS 203K)
    ↓ Valid SMILES filter
    ↓ Atom count filter (<= 150)
    ↓ Ring count filter (<= 10)
    ↓ SA filter (<= 6.0)
    ↓ Tanimoto space embedding (FP → PCA 3D)
    ↓ K-means clustering (Tanimoto FP + SA/QED/NPL)
    ↓ NPClassifier superclass labeling (local)
Subset (100K each) → Merge → training_data.csv → Train/Val/Test (seed=42)
    ↓ Distribution analysis (raw vs processed histograms + stats)
```

## Computed Columns (RDKit)

| Column | Description | Range |
|--------|-------------|-------|
| sa_score | Synthetic accessibility | 1-10 (lower=easier) |
| qed | Drug-likeness | 0-1 (higher=better) |
| npl_score | NP-likeness | -3~+3 (higher=more NP-like) |

## Configuration

All defaults are managed via `conf/config.yaml` (Hydra). Override from CLI:

```bash
python scripts/create_subset.py filtering.sa_max=5.0 subset.size=100000
```

## Citation

This project uses [NP-Classifier](https://github.com/mwang87/NP-Classifier) for natural product classification. If you use NPComposer, please cite:

```bibtex
@article{kim2021npclassifier,
  title={NPClassifier: A Deep Neural Network-Based Structural Classification Tool for Natural Products},
  author={Kim, Hyun Woo and Wang, Mingxun and Leber, Christopher A and Nothias, Louis-F{\'e}lix and Reher, Raphael and Kang, Kyo Bin and van der Hooft, Justin JJ and Dorrestein, Pieter C and Gerwick, William H and Cottrell, Garrison W},
  journal={Journal of Natural Products},
  volume={84},
  number={11},
  pages={2795--2807},
  year={2021},
  publisher={ACS Publications},
  doi={10.1021/acs.jnatprod.1c00399}
}
```

## Third-Party Licenses

This project uses code adapted from
[NP-Classifier](https://github.com/mwang87/NP-Classifier) (MIT License).
See [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES) for full details.
