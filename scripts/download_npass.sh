#!/bin/bash
# Download NPASS 3.0 dataset

set -e

DATA_DIR="data/raw/npass"
mkdir -p "$DATA_DIR"

BASE_URL="https://bidd.group/NPASS/downloadFiles"

FILES=(
    "NPASS3.0_naturalproducts_generalinfo.txt"
    "NPASS3.0_naturalproducts_structure.txt"
    "NPASS3.0_activities.txt"
    "NPASS3.0_naturalproducts_species_pair.txt"
    "NPASS3.0_target.txt"
    "NPASS3.0_species_info.txt"
    "NPASS3.0_toxicity.txt"
    "NPASS3.0_Symbiont.tsv"
    "NPASS3.0_Elicitation.tsv"
    "NPASS3.0_Coculture.tsv"
    "NPASS3.0_Engineer.tsv"
)

echo "Downloading NPASS 3.0 files..."

for file in "${FILES[@]}"; do
    echo "  $file"
    curl -L -o "$DATA_DIR/$file" "$BASE_URL/$file" --progress-bar
done

echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
