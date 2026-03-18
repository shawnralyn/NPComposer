# Data preprocessing for model training

This repository’s training pipeline expects **CSV splits** (train/val/test) containing a SMILES column plus any conditioning label columns used by NPComposer.

This README documents the **current minimal preprocessing path** implemented in `src/data_preprocessing/`. Run the scripts in this order:

1. `src/data_preprocessing/rdkit_metrics.py`
2. `src/data_preprocessing/bin_cont_variables.py`
3. `src/data_preprocessing/stratified_train_split.py`

---

## 0) Inputs / assumptions

- You start from a **COCONUT-like CSV** that contains at least:
  - `canonical_smiles` (SMILES string)
  - `qed_drug_likeliness` (continuous QED score)
  - `np_classifier_superclass` (NPClassifier superclass label)
  - `np_classifier_is_glycoside` (boolean/flag-like glycoside label)

If your file uses different column names, either rename columns up-front or pass the appropriate CLI options where supported.

---

## 1) Compute RDKit metrics (`rdkit_metrics.py`)

`rdkit_metrics.py` computes **synthetic accessibility (SA) score** from `canonical_smiles` and writes an updated CSV with a new `sa_score` column.

### What it does

- Reads the input CSV and drops rows missing `canonical_smiles`.
- Parses SMILES with RDKit.
- Computes `sa_score` using `rdkit.Contrib.SA_Score.sascorer`.
- Writes the updated CSV.

### Usage

```bash
python src/data_preprocessing/rdkit_metrics.py \
  --coconut path/to/coconut.csv \
  --out_file path/to/coconut_with_sa.csv
```

### Output

- Adds:
  - `sa_score`: float (may be empty/NaN if RDKit parsing fails)

---

## 2) Bin continuous variables (`bin_cont_variables.py`)

`bin_cont_variables.py` converts continuous `qed_drug_likeliness` and `sa_score` values into **string bin labels** (`qed_bin`, `sa_bin`). These binned columns can be used as conditioning labels.

### What it does

- Requires columns `qed_drug_likeliness` and `sa_score`.
- Drops rows with missing values in those columns.
- Creates bins:
  - QED: `[0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0)`
  - SA: `[1, 2), [2, 3), ..., [9, 10)`
- Writes the updated CSV.

### Usage

```bash
python src/data_preprocessing/bin_cont_variables.py \
  --input_csv path/to/coconut_with_sa.csv \
  --output path/to/coconut_with_sa_and_bins.csv
```

### Output

- Adds:
  - `qed_bin`: string label like `0.3<=qed<0.4`
  - `sa_bin`: string label like `4<=sa<5`

---

## 3) Stratified train/val/test split (`stratified_train_split.py`)

`stratified_train_split.py` creates train/val/test splits using **stratification over a combined stratum** of:

- superclass (`np_classifier_superclass` by default), and
- glycoside flag (`np_classifier_is_glycoside` by default)

This helps keep label proportions similar across splits.

### What it does

- Loads the input CSV.
- Drops rows with missing/blank values in the required columns.
- Builds a stratum key: `superclass|glycoside`.
- Drops strata with fewer than `--min_class_count` rows (default: 10) to ensure 3-way stratification is feasible.
- Writes split CSVs.

### Usage

```bash
python src/data_preprocessing/stratified_train_split.py \
  --input path/to/coconut_with_sa_and_bins.csv \
  --smiles_col canonical_smiles \
  --superclass_col np_classifier_superclass \
  --glycoside_col np_classifier_is_glycoside \
  --train_frac 0.8 \
  --val_frac 0.1 \
  --seed 42 \
  --min_class_count 10 \
  --out_train data/splits/train_v2.csv \
  --out_val data/splits/val_v2.csv \
  --out_test data/splits/test_v2.csv
```

### Notes

- Test fraction is computed as `1 - train_frac - val_frac`.
- If you need different stratification behavior, edit how `_strata` is constructed in the script.

---

## 4) Train

Once you have CSV splits, run training via:

```bash
python src/training/train.py \
  --yaml conf/train.yaml \
  --train_csv data/splits/train_v2.csv \
  --val_csv data/splits/val_v2.csv \
  --test_csv data/splits/test_v2.csv
```

Ensure that the columns referenced in `conf/train.yaml` exist in the split CSVs (e.g., `canonical_smiles`, `np_classifier_superclass`, `np_classifier_is_glycoside`, and optionally `qed_bin` / `sa_bin`).

---

## Troubleshooting

### Missing required columns
- `rdkit_metrics.py` expects `canonical_smiles`.
- `bin_cont_variables.py` expects `qed_drug_likeliness` and `sa_score`.
- `stratified_train_split.py` expects the columns passed via `--smiles_col`, `--superclass_col`, `--glycoside_col`.

### Stratified split error / too few samples per class
If you get errors from stratified splitting, increase dataset size, decrease `--min_class_count`, or reduce the number of strata (e.g., stratify only by superclass).
