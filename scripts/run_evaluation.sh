#!/bin/bash
# run_evaluation.sh — Generate molecules per class (seeds 1-5) and evaluate.
#
# Usage:
#   bash scripts/run_evaluation.sh
#   bash scripts/run_evaluation.sh --top 10           # top 10 classes only
#   bash scripts/run_evaluation.sh --seeds "1 2 3"    # custom seeds
#   bash scripts/run_evaluation.sh --num 200          # 200 molecules per run
#   bash scripts/run_evaluation.sh --classes "Flavones,Chalcones"  # specific classes
#
# Prerequisites:
#   - pip install transformers torch
#   - Training data at data/processed/training_data.csv (for novelty check)

set -e

# Defaults
TOP=10
SEEDS="1 2 3 4 5"
NUM_MOLECULES=100
YAML="conf/inference.yaml"
TRAINING_DATA="data/processed/training_data.csv"
OUTPUT_DIR="outputs/evaluation"
CLASSES=""
DATA_CSV="data/processed/coconut_100000.csv"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --top) TOP=$2; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --num) NUM_MOLECULES=$2; shift 2 ;;
        --yaml) YAML=$2; shift 2 ;;
        --training) TRAINING_DATA=$2; shift 2 ;;
        --output) OUTPUT_DIR=$2; shift 2 ;;
        --classes) CLASSES="$2"; shift 2 ;;
        --data) DATA_CSV=$2; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# Get class list
if [ -n "$CLASSES" ]; then
    # User-specified classes (comma-separated)
    IFS=',' read -ra CLASS_LIST <<< "$CLASSES"
else
    # Use the 7 pathway-level classes the model was trained on
    CLASS_LIST=(
        "Alkaloids"
        "Amino acids and Peptides"
        "Carbohydrates"
        "Fatty acids"
        "Polyketides"
        "Shikimates and Phenylpropanoids"
        "Terpenoids"
    )
fi

echo "=== NPComposer Evaluation ==="
echo "  Classes: ${#CLASS_LIST[@]}"
echo "  Seeds: $SEEDS"
echo "  Molecules per run: $NUM_MOLECULES"
echo "  Output: $OUTPUT_DIR"
echo ""

# Training data flag
TRAIN_FLAG=""
if [ -f "$TRAINING_DATA" ]; then
    TRAIN_FLAG="--training $TRAINING_DATA"
fi

# NP root flag
NP_FLAG=""
if [ -n "$NP_CLASSIFIER_ROOT" ]; then
    NP_FLAG="--np_root $NP_CLASSIFIER_ROOT"
fi

# Generate and evaluate per class per seed
for CLASS_NAME in "${CLASS_LIST[@]}"; do
    SAFE_NAME=$(echo "$CLASS_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    echo "=== Class: $CLASS_NAME ==="

    for SEED in $SEEDS; do
        OUT_FILE="$OUTPUT_DIR/${SAFE_NAME}_seed${SEED}.txt"
        RESULT_FILE="$OUTPUT_DIR/${SAFE_NAME}_seed${SEED}_results.json"

        echo "  Seed $SEED: generating $NUM_MOLECULES molecules ..."
        python3 src/inference/inference.py \
            --yaml "$YAML" \
            --seed "$SEED" \
            --np_class "<NP:${CLASS_NAME}>" \
            --output "$OUT_FILE" \
            --num_molecules "$NUM_MOLECULES"

        echo "  Seed $SEED: evaluating ..."
        python3 src/evaluation/metrics.py \
            -i "$OUT_FILE" \
            -o "$RESULT_FILE" \
            $NP_FLAG $TRAIN_FLAG

        echo "  Seed $SEED: done -> $RESULT_FILE"
    done
    echo ""
done

# Summary
echo "=== Aggregating results ==="
python3 -c "
import json, glob, os
from collections import defaultdict

results_dir = '$OUTPUT_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*_results.json')))

if not files:
    print('No result files found.')
    exit()

# Group by class
by_class = defaultdict(list)
for f in files:
    name = os.path.basename(f).replace('_results.json','')
    parts = name.rsplit('_seed', 1)
    cls = parts[0].replace('_', ' ')
    with open(f) as fh:
        data = json.load(fh)
    by_class[cls].append(data)

print(f'{'Class':<40} {'Valid%':>7} {'SA':>8} {'QED':>8} {'Div':>8} {'Novel%':>8}')
print('-' * 80)
for cls, runs in sorted(by_class.items()):
    vals = [r['validity']*100 for r in runs]
    sas = [r['sa_score']['mean'] for r in runs]
    qeds = [r['qed']['mean'] for r in runs]
    divs = [r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0) for r in runs]
    novs = [(r['novelty']*100 if isinstance(r.get('novelty'), (int, float)) else 0) for r in runs if 'novelty' in r]

    avg = lambda xs: sum(xs)/len(xs) if xs else 0
    print(f'{cls:<40} {avg(vals):>6.1f}% {avg(sas):>7.2f} {avg(qeds):>7.3f} {avg(divs):>7.3f} {avg(novs):>6.1f}%')

# Save summary
summary = {}
for cls, runs in by_class.items():
    summary[cls] = {
        'n_seeds': len(runs),
        'validity_mean': sum(r['validity'] for r in runs)/len(runs),
        'sa_mean': sum(r['sa_score']['mean'] for r in runs)/len(runs),
        'qed_mean': sum(r['qed']['mean'] for r in runs)/len(runs),
    }
with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to {results_dir}/summary.json')
"

echo ""
echo "=== Evaluation complete ==="
