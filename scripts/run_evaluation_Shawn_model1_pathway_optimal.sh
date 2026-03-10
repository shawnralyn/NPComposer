#!/bin/bash
# run_evaluation_Shawn_model1_pathway_optimal.sh — Shawn_model1: Generate per pathway with optimal SA + QED.
#
# Conditions on pathway + best SA bin (1<=sa<2) + best QED bin (0.9<=qed<1).
#
# Usage:
#   bash scripts/run_evaluation_Shawn_model1_pathway_optimal.sh
#   bash scripts/run_evaluation_Shawn_model1_pathway_optimal.sh --seeds "1 2 3"
#   bash scripts/run_evaluation_Shawn_model1_pathway_optimal.sh --num 200
#   bash scripts/run_evaluation_Shawn_model1_pathway_optimal.sh --pathways "Alkaloids,Terpenoids"
#   bash scripts/run_evaluation_Shawn_model1_pathway_optimal.sh --qed_bin "0.7<=qed<0.8" --sa_bin "2<=sa<3"

set -e

SEEDS="1 2 3"
NUM_MOLECULES=10
YAML="conf/inference.yaml"
TRAINING_DATA="data/raw/coconut_csv_full.csv"    # full COCONUT for uniqueness
NOVELTY_REF="data/processed/coconut_5000.csv"      # K-means subset for novelty
OUTPUT_DIR="outputs/Shawn_model1/evaluation_pathway_optimal"
PATHWAYS=""
SA_BIN="1<=sa<2"
QED_BIN="0.9<=qed<1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift 2 ;;
        --num) NUM_MOLECULES=$2; shift 2 ;;
        --yaml) YAML=$2; shift 2 ;;
        --training) TRAINING_DATA=$2; shift 2 ;;
        --novelty_ref) NOVELTY_REF=$2; shift 2 ;;
        --output) OUTPUT_DIR=$2; shift 2 ;;
        --pathways) PATHWAYS="$2"; shift 2 ;;
        --sa_bin) SA_BIN="$2"; shift 2 ;;
        --qed_bin) QED_BIN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

if [ -n "$PATHWAYS" ]; then
    IFS=',' read -ra PW_LIST <<< "$PATHWAYS"
else
    PW_LIST=(
        "Alkaloids"
        "Amino acids and Peptides"
        "Carbohydrates"
        "Fatty acids"
        "Polyketides"
        "Shikimates and Phenylpropanoids"
        "Terpenoids"
    )
fi

echo "=== Shawn_model1 Pathway + Optimal SA/QED ==="
echo "  Pathways: ${#PW_LIST[@]}"
echo "  Seeds: $SEEDS"
echo "  Molecules per run: $NUM_MOLECULES"
echo "  SA bin: $SA_BIN"
echo "  QED bin: $QED_BIN"
echo "  Output: $OUTPUT_DIR"
echo ""

TRAIN_FLAG=""
NOV_FLAG=""
    TRAIN_FLAG="--training $TRAINING_DATA"

NP_FLAG=""
[ -n "$NP_CLASSIFIER_ROOT" ] && NP_FLAG="--np_root $NP_CLASSIFIER_ROOT"

for PW_NAME in "${PW_LIST[@]}"; do
    SAFE_NAME=$(echo "$PW_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    echo "=== Pathway: $PW_NAME ==="

    for SEED in $SEEDS; do
        OUT_FILE="$OUTPUT_DIR/${SAFE_NAME}_seed${SEED}.txt"
        RESULT_FILE="$OUTPUT_DIR/${SAFE_NAME}_seed${SEED}_results.json"

        echo "  Seed $SEED: generating $NUM_MOLECULES molecules ..."
        python3 src/inference/inference.py \
            --yaml "$YAML" \
            --seed "$SEED" \
            --pathway "$PW_NAME" \
            --sa_bin "$SA_BIN" \
            --qed_bin "$QED_BIN" \
            --output "$OUT_FILE" \
            --num_molecules "$NUM_MOLECULES"

        echo "  Seed $SEED: evaluating ..."
        python3 src/evaluation/metrics.py \
            -i "$OUT_FILE" \
            -o "$RESULT_FILE" \
            $NP_FLAG $TRAIN_FLAG $NOV_FLAG

        echo "  Seed $SEED: done -> $RESULT_FILE"
    done
    echo ""
done

# Summary with baseline comparison
echo "=== Aggregating results ==="
python3 -c "
import json, glob, os
from collections import defaultdict

def load_results(d):
    files = sorted(glob.glob(os.path.join(d, '*_results.json')))
    by_class = defaultdict(list)
    for f in files:
        name = os.path.basename(f).replace('_results.json','')
        parts = name.rsplit('_seed', 1)
        cls = parts[0].replace('_', ' ')
        with open(f) as fh:
            by_class[cls].append(json.load(fh))
    return by_class

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

opt = load_results('$OUTPUT_DIR')
base_dir = 'outputs/Shawn_model1/evaluation'
has_base = os.path.isdir(base_dir) and glob.glob(os.path.join(base_dir, '*_results.json'))
base = load_results(base_dir) if has_base else {}

if not opt:
    print('No result files found.')
    exit()

h = f\"{'Pathway':<40} {'Valid%':>7} {'SA':>7} {'QED':>7} {'Div':>7} {'Uniq%':>7} {'Nov%':>7}\"
if has_base:
    h += f\"  || base: {'V%':>6} {'SA':>7} {'QED':>7}\"
print(h)
print('-' * len(h))

for cls, runs in sorted(opt.items()):
    vals = [r['validity']*100 for r in runs]
    sas = [r['sa_score']['mean'] for r in runs]
    qeds = [r['qed']['mean'] for r in runs]
    divs = [r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0) for r in runs]
    uniqs = [get_uniq(r) for r in runs]
    novs = [get_nov(r) for r in runs]

    line = f'{cls:<40} {avg(vals):>6.1f}% {avg(sas):>6.2f} {avg(qeds):>6.3f} {avg(divs):>6.3f} {avg(uniqs):>6.1f}% {avg(novs):>6.1f}%'

    if has_base and cls in base:
        b = base[cls]
        bv = avg([r['validity']*100 for r in b])
        bs = avg([r['sa_score']['mean'] for r in b])
        bq = avg([r['qed']['mean'] for r in b])
        line += f'  || {bv:>6.1f}% {bs:>7.2f} {bq:>7.3f}'

    print(line)

summary = {}
for cls, runs in opt.items():
    n = len(runs)
    summary[cls] = {
        'n_seeds': n,
        'validity_mean': sum(r['validity'] for r in runs)/n,
        'sa_mean': sum(r['sa_score']['mean'] for r in runs)/n,
        'qed_mean': sum(r['qed']['mean'] for r in runs)/n,
        'diversity_mean': avg([r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0) for r in runs]),
        'uniqueness_mean': avg([get_uniq(r)/100 for r in runs]),
        'novelty_mean': avg([get_nov(r) for r in runs]),
        'conditioning': {'sa_bin': '$SA_BIN', 'qed_bin': '$QED_BIN'},
    }
with open(os.path.join('$OUTPUT_DIR', 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to $OUTPUT_DIR/summary.json')
"

echo ""
echo "=== Evaluation complete ==="
