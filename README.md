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
│   ├── create_subset.py        # Create subset (K-medoids)
│   ├── merge_training.py       # Merge subsets into training data
│   ├── split_data.py           # Train/Val/Test split
│   └── analyze_sdf.py          # Analyze SDF structure
├── src/
│   ├── classification/
│   │   └── classyfire.py       # ClassyFire superclass API
│   └── evaluation/
│       └── metrics.py          # Evaluation metrics
├── tests/                      # pytest test suite
├── Makefile                    # Build automation
├── npcomposer.def              # Apptainer container definition
└── requirements.txt
```

## Setup

```bash
make setup
# Or manually
pip install -r requirements.txt
```

## Quick Start (Make)

```bash
# Full run (setup -> download -> subset -> split -> merge -> test)
make all

# COCONUT only
make all-coconut

# NPASS only
make all-npass

# Override defaults
make subset-coconut SIZE=100000 SA_MAX=5.0 SEED=42

# Run tests
make test

# Show all targets
make help
```

## Manual Usage

```bash
# COCONUT
bash scripts/download_data.sh
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv \
    --sdf data/raw/coconut_sdf_3d_full.sdf \
    -o data/processed/coconut_100k -s 100000 --classify --seed 42

# NPASS
bash scripts/download_npass.sh
python scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv
python scripts/create_subset.py -i data/raw/npass_full.csv \
    -o data/processed/npass_100k -s 100000 --classify --seed 42

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
    ↓ K-medoids (Tanimoto + SA/QED/NPL)
    ↓ ClassyFire superclass labeling
Subset (100K each) → Merge → training_data.csv → Train/Val/Test (seed=42)
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
