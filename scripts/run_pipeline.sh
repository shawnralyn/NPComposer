#!/bin/bash
# run_pipeline.sh — Run the full NPComposer pipeline end-to-end.
#
# Usage:
#   bash scripts/run_pipeline.sh
#   bash scripts/run_pipeline.sh --skip-download   # skip data download
#   bash scripts/run_pipeline.sh --size 50000      # custom subset size
#
# Prerequisites:
#   - pip install -r requirements.txt
#   - export NP_CLASSIFIER_ROOT=/path/to/NP-Classifier (optional, for classification)

set -e

# Defaults
SIZE=100000
SA_MAX=6.0
SEED=42
SKIP_DOWNLOAD=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        --size) SIZE=$2; shift 2 ;;
        --sa-max) SA_MAX=$2; shift 2 ;;
        --seed) SEED=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "=== NPComposer Full Pipeline ==="
echo "  Subset size: $SIZE"
echo "  SA max: $SA_MAX"
echo "  Seed: $SEED"
echo ""

# Step 1: Download data
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo "--- Step 1/6: Downloading datasets ---"
    bash scripts/download_data.sh
    bash scripts/download_npass.sh
else
    echo "--- Step 1/6: Skipping download ---"
fi

# Step 2: Merge NPASS
echo ""
echo "--- Step 2/6: Merging NPASS files ---"
python3 scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv

# Step 3: Create subsets
echo ""
echo "--- Step 3/6: Creating subsets ---"
mkdir -p data/processed

NP_FLAG=""
if [ -n "$NP_CLASSIFIER_ROOT" ]; then
    NP_FLAG="--np_root $NP_CLASSIFIER_ROOT"
fi

python3 scripts/create_subset.py \
    -i data/raw/coconut_csv_full.csv \
    -o data/processed/coconut_${SIZE} \
    -s $SIZE --sa_max $SA_MAX --seed $SEED $NP_FLAG

python3 scripts/create_subset.py \
    -i data/raw/npass_full.csv \
    -o data/processed/npass_${SIZE} \
    -s $SIZE --sa_max $SA_MAX --seed $SEED $NP_FLAG

# Step 4: Merge training data
echo ""
echo "--- Step 4/6: Merging training data ---"
python3 scripts/merge_training.py \
    --coconut data/processed/coconut_${SIZE}.csv \
    --npass data/processed/npass_${SIZE}.csv \
    -o data/processed/training_data.csv

# Step 5: Split data
echo ""
echo "--- Step 5/6: Splitting data ---"
mkdir -p data/splits
python3 scripts/split_data.py \
    -i data/processed/training_data.csv \
    -o data/splits/ --seed $SEED

# Step 6: Run tests
echo ""
echo "--- Step 6/6: Running tests ---"
python3 -m pytest tests/ -v

echo ""
echo "=== Pipeline complete ==="
echo "Training data: data/processed/training_data.csv"
echo "Splits: data/splits/"
