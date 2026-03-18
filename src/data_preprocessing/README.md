# Data preprocessing for model training

This repository’s training pipeline expects **CSV splits** (train/val/test) containing a SMILES column plus any conditioning label columns used by NPComposer (e.g., pathway/superclass/QED bin/SA bin).

This README documents a **minimal preprocessing path** based on the three scripts in the top-level `scripts/` folder:

- `scripts/create_subset.py`
- `scripts/merge_training.py`
- `scripts/split_data.py`

If you want a different preprocessing flow (e.g., stratified splits, NP-drug pairing), see `src/data_preprocessing/`.

---

## 0) Inputs / assumptions

- You have one or more source datasets exported as CSVs (e.g., COCONUT, NPASS).
- Each CSV must contain a SMILES-like column (common names that are auto-detected in `merge_training.py` include: `canonical_smiles`, `SMILES`, `smiles`, `Canonical_SMILES`, `smi`).

---

## 1) Create a representative subset (optional)

Use `scripts/create_subset.py` to downsample a large natural products dataset into a smaller, diverse subset.

`create_subset.py` can:
- Filter/score molecules using RDKit-derived properties (e.g., SA score, QED; and NP-likeness if available).
- Build Morgan fingerprints.
- Select a diverse subset via **mini-batch K-means** over a combined feature space (fingerprints + properties).

Typical usage (example):

```bash
python scripts/create_subset.py --help
```

Notes:
- If PyTorch is installed, K-means can use GPU/CPU via torch; otherwise it falls back to scikit-learn.
- Output is a CSV subset you can then merge/split.

---

## 2) Merge datasets into one training CSV

Use `scripts/merge_training.py` to combine two subset CSVs (COCONUT + NPASS) into a single deduplicated training dataset.

What it does:
- Loads both CSVs and adds a `source` column (`coconut` or `npass`).
- Detects and renames the SMILES column to a standard `smiles`.
- Concatenates and **deduplicates by `smiles`**.
- Ensures a `superclass` column exists:
  - Fills from existing columns in priority order: `superclass` → `np_classifier_superclass` → `chemical_super_class`.
  - For remaining missing values, it queries the **NPClassifier API** and fills unknowns as `"Unknown"`.

Example:

```bash
python scripts/merge_training.py \
  --coconut data/processed/coconut_subset.csv \
  --npass data/processed/npass_subset.csv \
  -o data/processed/training_merged.csv
```

Notes:
- NPClassifier lookup requires the `requests` package.
- The script currently queries `https://npclassifier.ucsd.edu/classify` with a short delay; this may be slow for large numbers of missing labels.

---

## 3) Split into train/val/test

Use `scripts/split_data.py` to create random splits.

Example:

```bash
python scripts/split_data.py \
  -i data/processed/training_merged.csv \
  -o data/splits/ \
  --train 0.8 --val 0.1 --test 0.1 \
  --seed 42
```

This writes:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

---

## 4) Train

Once you have CSV splits, run training via:

```bash
python src/training/train.py \
  --yaml conf/train.yaml \
  --train_csv data/splits/train.csv \
  --val_csv data/splits/val.csv \
  --test_csv data/splits/test.csv
```

Ensure that the columns referenced in `conf/train.yaml` exist in the split CSVs.

---

## Troubleshooting

### “No SMILES column found”
`merge_training.py` only auto-detects a short list of SMILES column names. Rename your SMILES column to one of the supported names or edit `SMILES_CANDIDATES` in `merge_training.py`.

### Very slow `merge_training.py`
If many rows are missing `superclass`, the NPClassifier API calls can dominate runtime.

### Split ratios error
`split_data.py` requires `train + val + test == 1.0`.
