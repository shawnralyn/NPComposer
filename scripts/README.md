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

Create subset using SA filtering + K-means clustering. Default seed: 42.

```bash
# COCONUT
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv \
    -o data/processed/coconut_100000 -s 100000

# NPASS
python scripts/create_subset.py -i data/raw/npass_full.csv \
    -o data/processed/npass_100000 -s 100000

# With SDF + ClassyFire
python scripts/create_subset.py -i data/raw/coconut_csv_full.csv \
    --sdf data/raw/coconut_sdf_3d_full.sdf \
    -o data/processed/coconut_100000 -s 100000 --classify
```

| Option | Default | Description |
|--------|---------|-------------|
| -s, --size | 100000 | Target subset size |
| --sdf | None | Input SDF (optional) |
| --sa_max | 6.0 | Max SA score |
| --max_atoms | 150 | Max atom count |
| --max_rings | 10 | Max ring count |
| --seed | 42 | Random seed |
| --classify | false | Add ClassyFire superclass labels |

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

## analyze_sdf.py

Analyze SDF file structure and molecular properties.

```bash
python scripts/analyze_sdf.py -i data/raw/coconut_sdf_3d_full.sdf -n 5000
```
