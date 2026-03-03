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
│   │   └── npclassifier.py     # NPClassifier local inference
│   └── evaluation/
│       └── metrics.py          # Evaluation metrics
├── tests/                      # pytest test suite
├── Makefile                    # Build automation
├── npcomposer.def              # Apptainer container definition
└── requirements.txt
```

## Quick Start

```bash
# Option 1: Using Make
make all

# Option 2: Using the full pipeline script
bash scripts/run_pipeline.sh

# Option 3: Skip download if data already exists
bash scripts/run_pipeline.sh --skip-download --size 100000
```

`make all` runs: setup -> download -> subset -> split -> merge -> test.
`run_pipeline.sh` runs the same steps as a single script with configurable flags.

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
    --np_root ~/NP-Classifier
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
    -o data/processed/coconut_100000 -s 100000 --seed 42

# NPASS
bash scripts/download_npass.sh
python scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv
python scripts/create_subset.py -i data/raw/npass_full.csv \
    -o data/processed/npass_100000 -s 100000 --seed 42

# Merge training data
python scripts/merge_training.py \
    --coconut data/processed/coconut_100000.csv \
    --npass data/processed/npass_100000.csv \
    -o data/processed/training_data.csv

# Split & Evaluate
python scripts/split_data.py -i data/processed/coconut_100000.csv -o data/splits/ --seed 42
python src/evaluation/metrics.py -i generated.txt -o results.json
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

### Filter Justifications

- SA <= 6.0: Molecules with SA scores above 6 are considered very difficult to synthesize in practice (Ertl & Schuffenhauer, 2009). Since the goal of this tool is to propose synthetically accessible natural product candidates, excluding hard-to-synthesize molecules prevents the model from learning to generate impractical outputs. The SA scale ranges from 1 (easy) to 10 (hard), and a threshold of 6.0 retains the majority of drug-like natural products while filtering out highly complex structures.
- Ring count <= 10: Natural products with more than 10 rings are rare outliers in COCONUT (< 1% of the dataset) and tend to be large macrocyclic or polymeric structures that are difficult to synthesize and characterize. Removing them reduces noise and keeps the training distribution focused on typical drug-like NP scaffolds.
- Atom count <= 150: Molecules exceeding 150 heavy atoms are typically large polymeric or peptidic natural products that produce very long SMILES strings, leading to tokenization issues and disproportionate memory usage during training. This cutoff retains > 99% of the dataset.


## Dataset Sources

- COCONUT (COlleCtion of Open Natural prodUcTs): An open-access database aggregating natural product structures from over 50 individual sources including DNP, UNPD, NUBBEDB, and others. Contains ~715K molecules with SMILES, InChI, and molecular descriptors. Source: https://coconut.naturalproducts.net. License: open access. Reference: Sorokina et al., "COCONUT online: Collection of Open Natural Products database," J. Cheminform., 2021.
- NPASS (Natural Product Activity and Species Source Database): A curated database of ~203K natural products with recorded biological activity data, species source information, and target annotations. Maintained by CSBIO, NUS. Source: https://bidd.group/NPASS/. Reference: Zeng et al., "NPASS: Natural product activity and species source database," Nucleic Acids Res., 2018.
- NP-Drug: A dataset of ~3K natural products with known drug activity, used for downstream evaluation and drug discovery benchmarking. Source: downloaded via `scripts/download_np_drug.sh`.

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
