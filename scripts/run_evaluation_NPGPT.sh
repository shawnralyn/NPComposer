#!/bin/bash
# run_evaluation_NPGPT.sh — Evaluate NPGPT (unconditional NP generation baseline).
#
# NPGPT generates SMILES without class conditioning, so we evaluate by:
#   1. Generating N molecules per seed (unconditional)
#   2. Running our metrics.py (validity, SA, QED, diversity, uniqueness, novelty)
#   3. Optionally running NPClassifier to see superclass distribution
#
# Usage:
#   bash scripts/run_evaluation_NPGPT.sh                                  # default: 760 molecules × 3 seeds
#   bash scripts/run_evaluation_NPGPT.sh --num 1000                       # 1000 molecules per seed
#   bash scripts/run_evaluation_NPGPT.sh --seeds "1 2 3 4 5"             # 5 seeds
#   bash scripts/run_evaluation_NPGPT.sh --checkpoint path/to/model.ckpt  # custom checkpoint
#   bash scripts/run_evaluation_NPGPT.sh --classify                       # run NPClassifier on generated molecules
#
# Prerequisites:
#   - cd external/npgpt && git submodule update --init --recursive && uv sync
#   - Download checkpoint to external/npgpt/checkpoints/smiles-gpt/model.ckpt
#   - Training data at data/raw/coconut_csv_full.csv (for uniqueness)
#   - K-means subset at data/processed/coconut_5000.csv (for novelty)

set -e

# Defaults
SEEDS="1 2 3"
NUM_MOLECULES=760              # 76 superclasses × 10 = 760 (fair comparison with Shawn_model1)
NPGPT_DIR="external/npgpt"
CHECKPOINT="external/npgpt/checkpoints/smiles-gpt/model.ckpt"
TOKENIZER="external/npgpt/externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"
TRAINING_DATA="data/raw/coconut_csv_full.csv"       # full COCONUT for uniqueness
NOVELTY_REF="data/processed/coconut_5000.csv"         # K-means subset for novelty
OUTPUT_DIR="outputs/NPGPT/evaluation"
BATCH_SIZE=1000
TEMPERATURE=""
TOP_P=""
CLASSIFY=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift 2 ;;
        --num) NUM_MOLECULES=$2; shift 2 ;;
        --checkpoint) CHECKPOINT=$2; shift 2 ;;
        --tokenizer) TOKENIZER=$2; shift 2 ;;
        --training) TRAINING_DATA=$2; shift 2 ;;
        --novelty_ref) NOVELTY_REF=$2; shift 2 ;;
        --output) OUTPUT_DIR=$2; shift 2 ;;
        --batch_size) BATCH_SIZE=$2; shift 2 ;;
        --temperature) TEMPERATURE=$2; shift 2 ;;
        --top_p) TOP_P=$2; shift 2 ;;
        --classify) CLASSIFY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "=== NPGPT Evaluation ==="
echo "  Seeds: $SEEDS"
echo "  Molecules per seed: $NUM_MOLECULES"
echo "  Checkpoint: $CHECKPOINT"
echo "  Tokenizer: $TOKENIZER"
echo "  Output: $OUTPUT_DIR"
echo "  Classify: $CLASSIFY"
echo ""

# Check checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    echo "  Download from: https://drive.google.com/drive/folders/1olCPouDkaJ2OBdNaM-G7IU8T6fBpvPMy"
    echo "  Place as: $CHECKPOINT"
    exit 1
fi

# Check tokenizer exists
if [ ! -f "$TOKENIZER" ]; then
    echo "ERROR: Tokenizer not found at $TOKENIZER"
    echo "  Run: cd $NPGPT_DIR && git submodule update --init --recursive"
    exit 1
fi

# Training data flag for metrics.py
TRAIN_FLAG=""
NOV_FLAG=""
if [ -f "$TRAINING_DATA" ]; then
    TRAIN_FLAG="--training $TRAINING_DATA"
fi
if [ -f "$NOVELTY_REF" ]; then
    NOV_FLAG="--novelty_ref $NOVELTY_REF"
fi


# Generate and evaluate per seed
for SEED in $SEEDS; do
    SMI_FILE="$OUTPUT_DIR/npgpt_seed${SEED}.smi"
    TXT_FILE="$OUTPUT_DIR/npgpt_seed${SEED}.txt"
    RESULT_FILE="$OUTPUT_DIR/npgpt_seed${SEED}_results.json"

    echo "=== Seed $SEED: generating $NUM_MOLECULES molecules ==="

    # --- Step 1: Generate SMILES via NPGPT wrapper (handles seed control) ---
    GEN_CMD="python3 scripts/npgpt_generate.py \
        --npgpt_dir $NPGPT_DIR \
        --tokenizer $TOKENIZER \
        --checkpoint $CHECKPOINT \
        --num_samples $NUM_MOLECULES \
        --batch_size $BATCH_SIZE \
        --seed $SEED \
        --output $SMI_FILE"

    # Add optional overrides
    [ -n "$TEMPERATURE" ] && GEN_CMD="$GEN_CMD --temperature $TEMPERATURE"
    [ -n "$TOP_P" ] && GEN_CMD="$GEN_CMD --top_p $TOP_P"

    eval $GEN_CMD

    echo "  Generated -> $SMI_FILE"

    # --- Step 2: Convert .smi to .txt (one SMILES per line, same format) ---
    # npgpt outputs .smi which is already one SMILES per line
    cp "$SMI_FILE" "$TXT_FILE"

    # --- Step 3: Evaluate with our metrics ---
    echo "  Evaluating ..."
    python3 src/evaluation/metrics.py \
        -i "$TXT_FILE" \
        -o "$RESULT_FILE" \
        $TRAIN_FLAG $NOV_FLAG

    # --- Step 4 (optional): NPClassifier distribution ---
    if [ "$CLASSIFY" = true ]; then
        CLASS_FILE="$OUTPUT_DIR/npgpt_seed${SEED}_classification.json"
        echo "  Classifying with NPClassifier API ..."
        python3 -c "
import json, requests, time
from collections import Counter

with open('$TXT_FILE') as f:
    smiles_list = [line.strip() for line in f if line.strip()]

superclass_counts = Counter()
classified = 0
for smi in smiles_list:
    try:
        r = requests.get('https://npclassifier.ucsd.edu/classify',
                         params={'smiles': smi}, timeout=30)
        r.raise_for_status()
        data = r.json()
        sc = data.get('superclass_results') or data.get('superclass')
        if isinstance(sc, list) and sc:
            for s in sc: superclass_counts[s] += 1
            classified += 1
        time.sleep(0.5)
    except: pass

output = {
    'total_molecules': len(smiles_list),
    'classified': classified,
    'superclass_distribution': dict(superclass_counts.most_common()),
    'n_unique_superclasses': len(superclass_counts),
}
with open('$CLASS_FILE', 'w') as f:
    json.dump(output, f, indent=2)
print(f'  Classification saved -> $CLASS_FILE')
print(f'  Unique superclasses: {len(superclass_counts)}')
print(f'  Top 10: {dict(superclass_counts.most_common(10))}')
"
    fi

    echo "  Seed $SEED: done -> $RESULT_FILE"
    echo ""
done

# Summary
echo "=== Aggregating results ==="
python3 -c "
import json, glob, os

results_dir = '$OUTPUT_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*_results.json')))

if not files:
    print('No result files found.')
    exit()

results = []
for f in files:
    seed = os.path.basename(f).replace('npgpt_seed','').replace('_results.json','')
    with open(f) as fh:
        r = json.load(fh)
        r['seed'] = seed
        results.append(r)

avg = lambda xs: sum(xs)/len(xs) if xs else 0

def get_uniq(r):
    u = r.get('uniqueness', {})
    if isinstance(u, dict):
        return u.get('uniqueness', 0) * 100
    return u * 100

def get_nov(r):
    n = r.get('novelty', {})
    if isinstance(n, dict):
        return n.get('novelty', 0) * 100
    return n * 100

print(f\"{'Seed':<8} {'Valid%':>7} {'SA':>7} {'QED':>7} {'Div':>7} {'Uniq%':>7} {'Nov%':>7}\")
print('-' * 55)
for r in results:
    val = r['validity'] * 100
    sa = r['sa_score']['mean']
    qed = r['qed']['mean']
    div = r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0)
    uniq = get_uniq(r)
    nov = get_nov(r)
    print(f\"Seed {r['seed']:<4} {val:>6.1f}% {sa:>6.2f} {qed:>6.3f} {div:>6.3f} {uniq:>6.1f}% {nov:>6.1f}%\")

# Average
vals = [r['validity']*100 for r in results]
sas = [r['sa_score']['mean'] for r in results]
qeds = [r['qed']['mean'] for r in results]
divs = [r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0) for r in results]
uniqs = [get_uniq(r) for r in results]
novs = [get_nov(r) for r in results]
print('-' * 55)
print(f\"{'Average':<8} {avg(vals):>6.1f}% {avg(sas):>6.2f} {avg(qeds):>6.3f} {avg(divs):>6.3f} {avg(uniqs):>6.1f}% {avg(novs):>6.1f}%\")

# Save summary
summary = {
    'model': 'NPGPT',
    'num_molecules_per_seed': $NUM_MOLECULES,
    'n_seeds': len(results),
    'validity_mean': avg([r['validity'] for r in results]),
    'sa_mean': avg(sas),
    'qed_mean': avg(qeds),
    'diversity_mean': avg(divs),
    'uniqueness_mean': avg([get_uniq(r)/100 for r in results]),
    'novelty_mean': avg(novs),
    'per_seed': {r['seed']: r for r in results},
}
with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to {results_dir}/summary.json')
"

# Classification summary (if --classify was used)
if [ "$CLASSIFY" = true ]; then
    echo ""
    echo "=== Superclass Distribution (aggregated) ==="
    python3 -c "
import json, glob, os
from collections import Counter

results_dir = '$OUTPUT_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*_classification.json')))
if not files:
    print('No classification files found.')
    exit()

total_counts = Counter()
for f in files:
    with open(f) as fh:
        data = json.load(fh)
        for sc, cnt in data.get('superclass_distribution', {}).items():
            total_counts[sc] += cnt

print(f\"{'Superclass':<45} {'Count':>6}\")
print('-' * 55)
for sc, cnt in total_counts.most_common():
    print(f'{sc:<45} {cnt:>6}')
print(f\"{'TOTAL':<45} {sum(total_counts.values()):>6}\")
print(f'Unique superclasses: {len(total_counts)}')
"
fi

echo ""
echo "=== NPGPT Evaluation complete ==="
