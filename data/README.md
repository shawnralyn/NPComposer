# data/

## raw/

Original COCONUT data. Run `bash scripts/download_data.sh` to download.

Files after download:
- `coconut_csv_full.csv` - SMILES and properties (715K molecules)
- `coconut_sdf_3d_full.sdf` - 3D structures

## processed/

Output from `scripts/create_subset.py`

- `subset_5k.csv` - Filtered CSV subset
- `subset_5k.sdf` - Matching 3D structures

## splits/

Output from `scripts/split_data.py`

- `train.csv` (80%)
- `val.csv` (10%)
- `test.csv` (10%)
