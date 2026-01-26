# scripts/

Data processing scripts for COCONUT subset creation.

## download_data.sh

Download and extract COCONUT dataset.

```bash
bash scripts/download_data.sh
```

Downloads:
- `coconut_csv_full.csv` (~208 MB) → CSV with SMILES and properties
- `coconut_sdf_3d_full.sdf` (~305 MB) → 3D structures

---

## analyze_sdf.py

Analyze SDF file structure and molecular properties.

```bash
# Basic usage
python analyze_sdf.py -i data/raw/coconut_sdf_3d_full.sdf

# More samples
python analyze_sdf.py -i data/raw/coconut_sdf_3d_full.sdf -n 5000

# Save to file
python analyze_sdf.py -i data/raw/coconut_sdf_3d_full.sdf -o report.txt
```

**Output:** molecule count, properties, 3D coords info, MW/atom stats, atom types

---

## create_subset.py

Create diverse subset from COCONUT (CSV + SDF) using SA filtering + K-means.

```bash
# CSV only
python create_subset.py -i data/raw/coconut_csv_full.csv -o data/processed/subset_5k -s 5000

# CSV + SDF
python create_subset.py -i data/raw/coconut_csv_full.csv --sdf data/raw/coconut_sdf_3d_full.sdf -o data/processed/subset_5k -s 5000

# Stricter SA filter
python create_subset.py -i data/raw/coconut_csv_full.csv --sdf data/raw/coconut_sdf_3d_full.sdf -o data/processed/subset_5k -s 5000 --sa_max 4.0
```

**Options:**
- `-s, --size`: Target subset size (default: 5000)
- `--sdf`: Input SDF file (optional, extracts matching molecules)
- `--sa_max`: Max SA score (default: 6.0)
- `--mw_min/--mw_max`: MW range (default: 150-800)

**Output:**
- `{output}.csv` - Subset CSV with sa_score, qed, npl_score
- `{output}.sdf` - Subset SDF with 3D structures (if --sdf provided)

---

## split_data.py

Split subset into train/val/test sets.

```bash
# Default 80/10/10 split
python split_data.py -i subset_5k.csv -o splits/

# Custom split ratio
python split_data.py -i subset_5k.csv -o splits/ --train 0.7 --val 0.15 --test 0.15
```

**Output:** train.csv, val.csv, test.csv in output directory
