#!/bin/bash
# Download and extract NP drug data

set -e

DATA_DIR="data/raw/ChEMBL"
mkdir -p "$DATA_DIR"

ZIP_URL="https://www.ebi.ac.uk/chembl/interface_api/delayed_jobs/outputs/DOWNLOAD-heRwUfDRJj-nAMSjmA31y8RyPmnAG6YgOpVpkx4anHU_eq_/DOWNLOAD-heRwUfDRJj-nAMSjmA31y8RyPmnAG6YgOpVpkx4anHU_eq_.zip"
ZIP_FILE="$DATA_DIR/ChEMBL.zip"

echo "Downloading ZIP (~490 MB)..."
curl -L -o "$ZIP_FILE" "$ZIP_URL" --progress-bar

echo "Extracting files..."
# -o: overwrite files without prompting
# -d: specify the destination directory
unzip -o "$ZIP_FILE" -d "$DATA_DIR"

# Optional: Remove the zip file after extraction to save space
rm "$ZIP_FILE"

echo "Renaming CSV files..."
i=1
while IFS= read -r -d '' csv_file; do
    mv "$csv_file" "$DATA_DIR/ChEMBL_${i}.csv"
    echo "  $csv_file -> ChEMBL_${i}.csv"
    i=$((i + 1))
done < <(find "$DATA_DIR" -maxdepth 1 -name "*.csv" -print0 | sort -z)

echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
