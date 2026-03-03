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
    if [ -f "$DATA_DIR/$file" ] && [ -s "$DATA_DIR/$file" ]; then
        echo "  $file (already exists, skipping)"
        continue
    fi
    echo "  $file"
    curl -L --connect-timeout 120 --max-time 7200 --retry 10 --retry-delay 30 \
        -o "$DATA_DIR/$file" "$BASE_URL/$file" --progress-bar
    sleep 5  # Be polite to the server
done

echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
