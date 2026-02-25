#!/bin/bash
# Download and extract NP drug data

set -e

DATA_DIR="data/raw"
mkdir -p "$DATA_DIR"

EXCEL_URL="https://www.rsc.org/suppdata/d1/np/d1np00039j/d1np00039j1.xlsx"

echo "Downloading EXCEL (~62 MB)..."
curl -L -o "$DATA_DIR/np_drug.xlsx" "$EXCEL_URL" --progress-bar

echo "Done. Files in $DATA_DIR:"
ls -lh "$DATA_DIR"
