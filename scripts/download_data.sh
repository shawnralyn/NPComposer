#!/bin/bash
# Download and extract COCONUT dataset

set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

CSV_URL="https://coconut.s3.uni-jena.de/prod/downloads/2026-01/coconut_csv-01-2026.zip"
#SDF_URL="https://coconut.s3.uni-jena.de/prod/downloads/2026-01/coconut_sdf_3d-01-2026.zip"

echo "Downloading CSV (~208 MB)..."
curl -L -o "$DATA_DIR/coconut_csv_full.zip" "$CSV_URL" --progress-bar

sleep 5  # Be polite to the server

#echo "Downloading SDF 3D (~305 MB)..."
#curl -L -o "$DATA_DIR/coconut_sdf_3d_full.zip" "$SDF_URL" --progress-bar

echo "Extracting CSV..."
unzip -o "$DATA_DIR/coconut_csv_full.zip" -d "$DATA_DIR"
mv "$DATA_DIR"/*.csv "$DATA_DIR/coconut_csv_full.csv" 2>/dev/null || true

#echo "Extracting SDF..."
#unzip -o "$DATA_DIR/coconut_sdf_3d_full.zip" -d "$DATA_DIR"
#mv "$DATA_DIR"/*.sdf "$DATA_DIR/coconut_sdf_3d_full.sdf" 2>/dev/null || true

read -p "Delete zip files? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm "$DATA_DIR/coconut_csv_full.zip" #"$DATA_DIR/coconut_sdf_3d_full.zip"
    echo "Zip files deleted"
fi

echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
