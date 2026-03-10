#!/bin/bash
# run_evaluation_Shawn_model1.sh — Shawn_model1: Generate molecules per superclass and evaluate.
#
# Usage:
#   bash scripts/run_evaluation_Shawn_model1.sh                              # all 76 superclasses
#   bash scripts/run_evaluation_Shawn_model1.sh --top 10                     # top 10 superclasses
#   bash scripts/run_evaluation_Shawn_model1.sh --seeds "1 2 3"              # custom seeds
#   bash scripts/run_evaluation_Shawn_model1.sh --num 200                    # 200 molecules per run
#   bash scripts/run_evaluation_Shawn_model1.sh --superclasses "Flavonoids,Steroids"
#   bash scripts/run_evaluation_Shawn_model1.sh --pathway Alkaloids          # also condition on pathway
#
# Prerequisites:
#   - pip install transformers torch
#   - Training data at data/processed/coconut_100000.csv (K-means 5k subset for novelty check)
#   - Model checkpoint accessible (HuggingFace or local path in conf/inference.yaml)

set -e

# Defaults
TOP=0          # 0 = all superclasses
SEEDS="1 2 3"
NUM_MOLECULES=10
YAML="conf/inference.yaml"
TRAINING_DATA="data/raw/coconut_csv_full.csv"    # full COCONUT for uniqueness
NOVELTY_REF="data/processed/coconut_5000.csv"      # K-means subset for novelty
OUTPUT_DIR="outputs/Shawn_model1/evaluation"
SUPERCLASSES=""
PATHWAY=""
IS_GLYCOSIDE=""
AROMATIC_RINGS=""
QED_BIN=""
SA_BIN=""

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --top) TOP=$2; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --num) NUM_MOLECULES=$2; shift 2 ;;
        --yaml) YAML=$2; shift 2 ;;
        --training) TRAINING_DATA=$2; shift 2 ;;
        --novelty_ref) NOVELTY_REF=$2; shift 2 ;;
        --output) OUTPUT_DIR=$2; shift 2 ;;
        --superclasses) SUPERCLASSES="$2"; shift 2 ;;
        --pathway) PATHWAY="$2"; shift 2 ;;
        --is_glycoside) IS_GLYCOSIDE="$2"; shift 2 ;;
        --aromatic_rings) AROMATIC_RINGS="$2"; shift 2 ;;
        --qed_bin) QED_BIN="$2"; shift 2 ;;
        --sa_bin) SA_BIN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# All 76 superclasses from the v2 model's special_tokens_map
ALL_SUPERCLASSES=(
    "Alkylresorcinols"
    "Amino acid glycosides"
    "Aminosugars and aminoglycosides"
    "Anthranilic acid alkaloids"
    "Apocarotenoids"
    "Aromatic polyketides"
    "Carotenoids (C40)"
    "Carotenoids (C45)"
    "Carotenoids (C50)"
    "Chromanes"
    "Coumarins"
    "Cyclic polyketides"
    "Diarylheptanoids"
    "Diazotetronic acids and derivatives"
    "Diphenyl ethers (DPEs)"
    "Diterpenoids"
    "Docosanoids"
    "Eicosanoids"
    "Fatty Acids and Conjugates"
    "Fatty acyl glycosides"
    "Fatty acyls"
    "Fatty amides"
    "Fatty esters"
    "Flavonoids"
    "Fluorenes"
    "Glycerolipids"
    "Glycerophospholipids"
    "Guanidine alkaloids"
    "Histidine alkaloids"
    "Isoflavonoids"
    "Lignans"
    "Linear polyketides"
    "Lysine alkaloids"
    "Macrolides"
    "Meroterpenoids"
    "Miscellaneous alkaloids"
    "Miscellaneous polyketides"
    "Mitomycin derivatives"
    "Monoterpenoids"
    "Mycosporine derivatives"
    "Naphthalenes"
    "Nicotinic acid alkaloids"
    "Nucleosides"
    "Octadecanoids"
    "Oligopeptides"
    "Ornithine alkaloids"
    "Peptide alkaloids"
    "Phenanthrenoids"
    "Phenolic acids (C6-C1)"
    "Phenylethanoids (C6-C2)"
    "Phenylpropanoids (C6-C3)"
    "Phloroglucinols"
    "Polycyclic aromatic polyketides"
    "Polyethers"
    "Polyols"
    "Polyprenols"
    "Proline alkaloids"
    "Pseudoalkaloids"
    "Saccharides"
    "Serine alkaloids"
    "Sesquiterpenoids"
    "Sesterterpenoids"
    "Small peptides"
    "Sphingolipids"
    "Steroids"
    "Stilbenoids"
    "Styrylpyrones"
    "Terphenyls"
    "Tetramate alkaloids"
    "Triterpenoids"
    "Tropolones"
    "Tryptophan alkaloids"
    "Tyrosine alkaloids"
    "Xanthones"
    "β-lactams"
    "γ-lactam-β-lactones"
)

# Build class list
if [ -n "$SUPERCLASSES" ]; then
    IFS=',' read -ra CLASS_LIST <<< "$SUPERCLASSES"
elif [ "$TOP" -gt 0 ]; then
    # Take first N from the list
    CLASS_LIST=("${ALL_SUPERCLASSES[@]:0:$TOP}")
else
    CLASS_LIST=("${ALL_SUPERCLASSES[@]}")
fi

echo "=== Shawn_model1 Evaluation ==="
echo "  Superclasses: ${#CLASS_LIST[@]}"
echo "  Seeds: $SEEDS"
echo "  Molecules per run: $NUM_MOLECULES"
echo "  Output: $OUTPUT_DIR"
[ -n "$PATHWAY" ] && echo "  Pathway filter: $PATHWAY"
echo ""

# Training data flag
TRAIN_FLAG=""
NOV_FLAG=""
if [ -f "$TRAINING_DATA" ]; then
    TRAIN_FLAG="--training $TRAINING_DATA"
fi
if [ -f "$NOVELTY_REF" ]; then
    NOV_FLAG="--novelty_ref $NOVELTY_REF"
fi

# NP root flag
NP_FLAG=""
if [ -n "$NP_CLASSIFIER_ROOT" ]; then
    NP_FLAG="--np_root $NP_CLASSIFIER_ROOT"
fi

# Build extra conditioning flags
EXTRA_FLAGS=""
[ -n "$PATHWAY" ] && EXTRA_FLAGS="$EXTRA_FLAGS --pathway $PATHWAY"
[ -n "$IS_GLYCOSIDE" ] && EXTRA_FLAGS="$EXTRA_FLAGS --is_glycoside $IS_GLYCOSIDE"
[ -n "$AROMATIC_RINGS" ] && EXTRA_FLAGS="$EXTRA_FLAGS --aromatic_rings $AROMATIC_RINGS"
[ -n "$QED_BIN" ] && EXTRA_FLAGS="$EXTRA_FLAGS --qed_bin $QED_BIN"
[ -n "$SA_BIN" ] && EXTRA_FLAGS="$EXTRA_FLAGS --sa_bin $SA_BIN"

# Generate and evaluate per superclass per seed
for SC_NAME in "${CLASS_LIST[@]}"; do
    SAFE_NAME=$(echo "$SC_NAME" | tr ' ' '_' | tr -cd '[:alnum:]_-')
    echo "=== Superclass: $SC_NAME ==="

    for SEED in $SEEDS; do
        OUT_FILE="$OUTPUT_DIR/${SAFE_NAME}_seed${SEED}.txt"
        RESULT_FILE="$OUTPUT_DIR/${SAFE_NAME}_seed${SEED}_results.json"

        echo "  Seed $SEED: generating $NUM_MOLECULES molecules ..."
        python3 src/inference/inference.py \
            --yaml "$YAML" \
            --seed "$SEED" \
            --superclass "$SC_NAME" \
            $EXTRA_FLAGS \
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

by_class = defaultdict(list)
for f in files:
    name = os.path.basename(f).replace('_results.json','')
    parts = name.rsplit('_seed', 1)
    cls = parts[0].replace('_', ' ')
    with open(f) as fh:
        by_class[cls].append(json.load(fh))

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

print(f\"{'Superclass':<40} {'Valid%':>7} {'SA':>7} {'QED':>7} {'Div':>7} {'Uniq%':>7} {'Nov%':>7}\")
print('-' * 90)
for cls, runs in sorted(by_class.items()):
    vals = [r['validity']*100 for r in runs]
    sas = [r['sa_score']['mean'] for r in runs]
    qeds = [r['qed']['mean'] for r in runs]
    divs = [r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0) for r in runs]
    uniqs = [get_uniq(r) for r in runs]
    novs = [get_nov(r) for r in runs]
    print(f'{cls:<40} {avg(vals):>6.1f}% {avg(sas):>6.2f} {avg(qeds):>6.3f} {avg(divs):>6.3f} {avg(uniqs):>6.1f}% {avg(novs):>6.1f}%')

summary = {}
for cls, runs in by_class.items():
    n = len(runs)
    summary[cls] = {
        'n_seeds': n,
        'validity_mean': sum(r['validity'] for r in runs)/n,
        'sa_mean': sum(r['sa_score']['mean'] for r in runs)/n,
        'qed_mean': sum(r['qed']['mean'] for r in runs)/n,
        'diversity_mean': avg([r['internal_diversity']['mean'] if isinstance(r.get('internal_diversity'), dict) else r.get('internal_diversity', 0) for r in runs]),
        'uniqueness_mean': avg([get_uniq(r)/100 for r in runs]),
        'novelty_mean': avg([get_nov(r) for r in runs]),
    }
with open(os.path.join(results_dir, 'summary.json'), 'w') as f:
    json.dump(summary, f, indent=2)
print(f'\nSummary saved to {results_dir}/summary.json')
"

echo ""
echo "=== Evaluation complete ==="
