# NPComposer

Small-scale NP generation using COCONUT dataset.

## Structure

```
NPComposer/
├── data/
│   ├── raw/                    # COCONUT data (download manually)
│   ├── processed/              # Filtered subset
│   └── splits/                 # Train/Val/Test
│
├── scripts/
│   ├── analyze_sdf.py          # Analyze SDF structure
│   ├── create_subset.py        # Create subset (SA + K-medoids)
│   └── split_data.py           # Train/Val/Test split
│
└── src/
    └── evaluation/
        └── metrics.py          # Evaluation metrics
```

## Setup

```bash
pip install rdkit pandas numpy scikit-learn scikit-learn-extra tqdm
```

## Usage

```bash
# 1. Download COCONUT data
bash scripts/download_data.sh

# 2. Create subset (CSV + SDF)
python scripts/create_subset.py \
    -i data/raw/coconut_csv_full.csv \
    --sdf data/raw/coconut_sdf_3d_full.sdf \
    -o data/processed/subset_5k \
    -s 5000

# 3. Split into train/val/test
python scripts/split_data.py -i data/processed/subset_5k.csv -o data/splits/

# 4. Evaluate generated molecules
python src/evaluation/metrics.py -i generated.txt -o results.json
```

## Pipeline

```
COCONUT (715K)
    ↓ Valid SMILES filter
    ↓ MW filter (150-800)
    ↓ SA filter (<= 6.0)
    ↓ K-medoids clustering
Subset (5K) → Train/Val/Test
```

## Subset Columns

| Column | Description |
|--------|-------------|
| canonical_smiles | Normalized SMILES |
| npl_score | NP-likeness (COCONUT) |
| sa_score | Synthetic accessibility (RDKit) |
| qed | Drug-likeness (RDKit) |
