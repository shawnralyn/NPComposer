# scripts/

## download_data.sh

Download COCONUT dataset.

```bash
bash scripts/download_data.sh
```

## download_npass.sh

Download NPASS 3.0 dataset.

```bash
bash scripts/download_npass.sh
```

## merge_npass.py

Merge NPASS files into single CSV.

```bash
python scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv
```

## create_subset.py

Create subset using SA filtering + K-means clustering in Tanimoto fingerprint space. Default seed: 42.

```bash
# COCONUT
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv \
    -o data/processed/coconut_100000 -s 100000

# NPASS
python scripts/create_subset.py -i data/raw/npass_full.csv \
    -o data/processed/npass_100000 -s 100000

# With SDF (NPClassifier runs by default if NP_CLASSIFIER_ROOT is set)
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv \
    --sdf data/raw/coconut_sdf_3d_full.sdf \
    -o data/processed/coconut_100000 -s 100000
```

| Option | Default | Description |
|--------|---------|-------------|
| -s, --size | 100000 | Target subset size |
| --sdf | None | Input SDF (optional) |
| --sa_max | 6.0 | Max SA score |
| --max_atoms | 150 | Max atom count |
| --max_rings | 10 | Max ring count |
| --fp_dim | 3 | PCA dimensions for Tanimoto space |
| --n_jobs | -1 | Parallel workers (-1 = all cores) |
| --seed | 42 | Random seed |
| --np_root | NP_CLASSIFIER_ROOT | Path to NP-Classifier repo |

## merge_training.py

Merge COCONUT and NPASS subsets into a single training dataset. Deduplicates by SMILES.

```bash
python scripts/merge_training.py \
    --coconut data/processed/coconut_100000.csv \
    --npass data/processed/npass_100000.csv \
    -o data/processed/training_data.csv
```

| Option | Description |
|--------|-------------|
| --coconut | COCONUT subset CSV |
| --npass | NPASS subset CSV |
| -o, --output | Output CSV path |

## split_data.py

Split into train/val/test. Default seed: 42.

```bash
python scripts/split_data.py -i data/processed/coconut_100000.csv -o data/splits/
python scripts/split_data.py -i data/processed/coconut_100000.csv -o data/splits/ \
    --train 0.7 --val 0.15 --test 0.15 --seed 42
```

## analyze_distribution.py

Compare property distributions between raw and processed datasets. Generates histograms and summary statistics for MW, atom count, ring count, SA, QED, NPL, and superclass.

```bash
python scripts/analyze_distribution.py \
    --raw data/raw/coconut_csv_full.csv \
    --processed data/processed/coconut_100000.csv \
    -o data/processed/dist_coconut
```

| Option | Default | Description |
|--------|---------|-------------|
| --raw | (required) | Raw dataset CSV |
| --processed | (required) | Processed subset CSV |
| -o, --output | (required) | Output prefix |
| --sample | 10000 | Max rows to sample from raw for RDKit computation |

## analyze_sdf.py

Analyze SDF file structure and molecular properties.

```bash
python scripts/analyze_sdf.py -i data/raw/coconut_sdf_3d_full.sdf -n 5000
```

## clean_npdrug.py

Clean and process NP-Drug Excel file into a structured CSV.

```bash
python scripts/clean_npdrug.py -i data/raw/np_drug.xlsx -o data/processed/np_drug.csv
```