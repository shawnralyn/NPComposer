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

Create subset using SA filtering + K-medoids clustering.

```bash
# COCONUT
python create_subset.py -i data/raw/coconut_csv_full.csv -o data/processed/coconut_5k -s 5000

# NPASS
python create_subset.py -i data/raw/npass_full.csv -o data/processed/npass_5k -s 5000

# With SDF
python create_subset.py -i data/raw/coconut_csv_full.csv --sdf data/raw/coconut_sdf_3d_full.sdf -o data/processed/coconut_5k -s 5000
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| -s, --size | 5000 | Target subset size |
| --sdf | None | Input SDF (optional) |
| --sa_max | 6.0 | Max SA score |
| --mw_min | 150 | Min molecular weight |
| --mw_max | 800 | Max molecular weight |

## split_data.py

Split into train/val/test.

```bash
python split_data.py -i data/processed/coconut_5k.csv -o data/splits/
python split_data.py -i data/processed/coconut_5k.csv -o data/splits/ --train 0.7 --val 0.15 --test 0.15
```

## analyze_sdf.py

Analyze SDF file structure.

```bash
python analyze_sdf.py -i data/raw/coconut_sdf_3d_full.sdf -n 5000
```
