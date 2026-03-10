#!/bin/bash
# run_evaluation_Shawn_model1_optimal.sh — Shawn_model1: Generate molecules with optimal SA + QED.
#
# Conditions on superclass + best SA bin (1<=sa<2) + best QED bin (0.9<=qed<1)
# to test whether the model can generate drug-like, easily synthesizable molecules.
#
# Usage:
#   bash scripts/run_evaluation_Shawn_model1_optimal.sh                              # all 76 superclasses
#   bash scripts/run_evaluation_Shawn_model1_optimal.sh --top 10                     # top 10 superclasses
#   bash scripts/run_evaluation_Shawn_model1_optimal.sh --superclasses "Flavonoids,Steroids"
#   bash scripts/run_evaluation_Shawn_model1_optimal.sh --qed_bin "0.7<=qed<0.8"    # custom QED bin
#   bash scripts/run_evaluation_Shawn_model1_optimal.sh --sa_bin "2<=sa<3"           # custom SA bin

set -e

# Defaults
TOP=0
SEEDS="1 2 3"
NUM_MOLECULES=10
YAML="conf/inference.yaml"
TRAINING_DATA="data/raw/coconut_csv_full.csv"    # full COCONUT for uniqueness
NOVELTY_REF="data/processed/coconut_5000.csv"      # K-means subset for novelty
OUTPUT_DIR="outputs/Shawn_model1/evaluation_optimal"
SUPERCLASSES=""
PATHWAY=""
SA_BIN="1<=sa<2"        # best SA (easiest to synthesize)
QED_BIN="0.9<=qed<1"    # best QED (most drug-like)

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
        --sa_bin) SA_BIN="$2"; shift 2 ;;
        --qed_bin) QED_BIN="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# All 76 superclasses
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
    CLASS_LIST=("${ALL_SUPERCLASSES[@]:0:$TOP}")
else
    CLASS_LIST=("${ALL_SUPERCLASSES[@]}")
fi

echo "=== Shawn_model1 Optimal Evaluation ==="
echo "  Superclasses: ${#CLASS_LIST[@]}"
echo "  Seeds: $SEEDS"
echo "  Molecules per run: $NUM_MOLECULES"
echo "  SA bin: $SA_BIN"
echo "  QED bin: $QED_BIN"
echo "  Output: $OUTPUT_DIR"
[ -n "$PATHWAY" ] && echo "  Pathway filter: $PATHWAY"
echo ""

# Flags
TRAIN_FLAG=""
NOV_FLAG=""
if [ -f "$TRAINING_DATA" ]; then
    TRAIN_FLAG="--training $TRAINING_DATA"
fi
if [ -f "$NOVELTY_REF" ]; then
    NOV_FLAG="--novelty_ref $NOVELTY_REF"
fi

NP_FLAG=""
[ -n "$NP_CLASSIFIER_ROOT" ] && NP_FLAG="--np_root $NP_CLASSIFIER_ROOT"

EXTRA_FLAGS="--sa_bin $SA_BIN --qed_bin $QED_BIN"
[ -n "$PATHWAY" ] && EXTRA_FLAGS="$EXTRA_FLAGS --pathway $PATHWAY"

# Generate and evaluate
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

# Summary: compare with baseline (outputs/evaluation if exists)
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

header = f\"{'Superclass':<40} {'Valid%':>7} {'SA':>7} {'QED':>7} {'Div':>7} {'Uniq%':>7} {'Nov%':>7}\"
if has_base:
    header += f\"  || base: {'V%':>6} {'SA':>7} {'QED':>7}\"
print(header)
print('-' * len(header))

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
