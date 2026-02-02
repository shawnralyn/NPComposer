# NPComposer

Natural product generation using COCONUT and NPASS datasets.

## Structure

```
NPComposer/
├── data/
│   ├── raw/                    # Original data
│   │   ├── npass/              # NPASS files
│   │   ├── coconut_csv_full.csv
│   │   └── coconut_sdf_3d_full.sdf
│   ├── processed/              # Filtered subsets
│   └── splits/                 # Train/Val/Test
│
├── scripts/
│   ├── download_data.sh        # Download COCONUT
│   ├── download_npass.sh       # Download NPASS
│   ├── merge_npass.py          # Merge NPASS files
│   ├── create_subset.py        # Create subset (K-medoids)
│   ├── split_data.py           # Train/Val/Test split
│   └── analyze_sdf.py          # Analyze SDF structure
│
└── src/
    └── evaluation/
        └── metrics.py          # Evaluation metrics
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# COCONUT
bash scripts/download_data.sh
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv --sdf data/raw/coconut_sdf_3d_full.sdf -o data/processed/coconut_5k -s 5000

# NPASS
bash scripts/download_npass.sh
python scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv
python scripts/create_subset.py -i data/raw/npass_full.csv -o data/processed/npass_5k -s 5000

# Split & Evaluate
python scripts/split_data.py -i data/processed/coconut_5k.csv -o data/splits/
python src/evaluation/metrics.py -i generated.txt -o results.json
```

## Pipeline

```
Raw Data (COCONUT 715K / NPASS 100K+)
    ↓ Valid SMILES filter
    ↓ MW filter (150-800)
    ↓ SA filter (<= 6.0)
    ↓ K-medoids (Tanimoto + SA/QED/NPL)
Subset (5K) → Train/Val/Test
```

## Computed Columns (RDKit)

| Column | Description | Range |
|--------|-------------|-------|
| sa_score(RDKit) | Synthetic accessibility | 1-10 (lower=easier) |
| qed(RDKit) | Drug-likeness | 0-1 (higher=better) |
| npl_score(RDKit) | NP-likeness | -3~+3 (higher=more NP-like) |
