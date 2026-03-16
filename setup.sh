#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# NPComposer — Full Automated Setup
# ═══════════════════════════════════════════════════════════════════════════
#
# This script automates the complete setup of NPComposer:
#   1. Python dependencies
#   2. Git submodules (npgpt, acegen-open, gp-molformer)
#   3. NPGPT checkpoint download
#   4. Directory structure
#   5. Verification
#
# Usage:
#   bash setup.sh              # Full setup
#   bash setup.sh --skip-npclassifier   # Skip NPClassifier
#   bash setup.sh --skip-checkpoints    # Skip checkpoint downloads
#   bash setup.sh --gpu                 # Install GPU-enabled PyTorch
#
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
fail()  { echo -e "${RED}[FAIL]${NC}  $*"; exit 1; }
step()  { echo -e "\n${GREEN}━━━ Step $1: $2 ━━━${NC}"; }

# ── Parse args ────────────────────────────────────────────────────────────
SKIP_CHECKPOINTS=false
GPU=false

for arg in "$@"; do
    case $arg in
        --skip-checkpoints)  SKIP_CHECKPOINTS=true ;;
        --gpu)               GPU=true ;;
        --help|-h)
            echo "Usage: bash setup.sh [OPTIONS]"
            echo "  --skip-checkpoints   Skip model checkpoint downloads"
            echo "  --gpu                Install GPU-enabled PyTorch"
            exit 0 ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           NPComposer — Automated Setup                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ══════════════════════════════════════════════════════════════════════════
# Step 1: Python environment check
# ══════════════════════════════════════════════════════════════════════════
step 1 "Python Environment"

PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip}"

if ! command -v "$PYTHON" &>/dev/null; then
    fail "Python not found. Install Python 3.9+ first."
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python version: $PY_VERSION"

PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
    fail "Python 3.9+ required. Found: $PY_VERSION"
fi
ok "Python $PY_VERSION"

# ══════════════════════════════════════════════════════════════════════════
# Step 2: Install Python dependencies
# ══════════════════════════════════════════════════════════════════════════
step 2 "Python Dependencies"

info "Installing core dependencies..."
$PIP install -r requirements.txt -q

# matplotlib & scipy for evaluation plots
info "Installing evaluation dependencies..."
$PIP install matplotlib scipy -q

if [ "$GPU" = true ]; then
    info "Installing GPU-enabled PyTorch..."
    $PIP install torch --index-url https://download.pytorch.org/whl/cu121 -q
else
    info "Using existing PyTorch (CPU or pre-installed GPU)"
fi

# Editable install for project imports
$PIP install --no-deps -e . -q 2>/dev/null || true
ok "Python dependencies installed"

# ══════════════════════════════════════════════════════════════════════════
# Step 3: Git submodules
# ══════════════════════════════════════════════════════════════════════════
step 3 "Git Submodules"

if [ -f .gitmodules ]; then
    info "Initializing submodules..."
    git submodule update --init --recursive 2>/dev/null || {
        warn "git submodule update failed — trying individual submodules"

        # NPGPT
        if [ ! -d "external/npgpt/.git" ] && [ ! -f "external/npgpt/.git" ]; then
            info "Cloning external/npgpt..."
            git submodule update --init external/npgpt 2>/dev/null || \
                warn "Failed to init npgpt submodule (may need manual setup)"
        fi

        # acegen-open
        if [ ! -d "external/acegen-open/.git" ] && [ ! -f "external/acegen-open/.git" ]; then
            info "Cloning external/acegen-open..."
            git submodule update --init external/acegen-open 2>/dev/null || \
                warn "Failed to init acegen-open submodule"
        fi

        # gp-molformer
        if [ ! -d "external/gp-molformer/.git" ] && [ ! -f "external/gp-molformer/.git" ]; then
            info "Cloning external/gp-molformer..."
            git submodule update --init external/gp-molformer 2>/dev/null || \
                warn "Failed to init gp-molformer submodule"
        fi
    }
    ok "Git submodules initialized"
else
    warn "No .gitmodules found — skipping submodule setup"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 4: Model checkpoints
# ══════════════════════════════════════════════════════════════════════════
step 4 "Model Checkpoints"

if [ "$SKIP_CHECKPOINTS" = true ]; then
    info "Skipping checkpoint downloads (--skip-checkpoints)"
else
    # NPGPT checkpoint
    NPGPT_CKPT="src/npgpt-rl/npgpt.ckpt"
    if [ -f "$NPGPT_CKPT" ]; then
        ok "NPGPT checkpoint already exists: $NPGPT_CKPT"
    else
        warn "NPGPT checkpoint not found at: $NPGPT_CKPT"
        info "Download manually from Google Drive:"
        info "  https://drive.google.com/drive/folders/1olCPouDkaJ2OBdNaM-G7IU8T6fBpvPMy"
        info "  Place as: $NPGPT_CKPT"
    fi

    # NPGPT RL checkpoint
    RL_CKPT="src/npgpt-rl/npgpt_rl_step_600.ckpt"
    if [ -f "$RL_CKPT" ]; then
        ok "NPGPT RL checkpoint exists: $RL_CKPT"
    else
        warn "NPGPT RL checkpoint not found: $RL_CKPT"
        info "Train with: python src/npgpt-rl/train_rl.py"
    fi

    # NPComposer (Shawn model) — auto-downloads from HuggingFace
    info "NPComposer (ralyn/NPComposer-v2) will auto-download from HuggingFace on first run"
    ok "Checkpoint check complete"
fi

# ══════════════════════════════════════════════════════════════════════════
# Step 5: Directory structure
# ══════════════════════════════════════════════════════════════════════════
step 5 "Directory Structure"

mkdir -p data/raw data/processed data/splits
mkdir -p results/evaluation results/evaluation_shawn results/comparison
mkdir -p outputs

ok "Directory structure ready"

# ══════════════════════════════════════════════════════════════════════════
# Step 7: Verification
# ══════════════════════════════════════════════════════════════════════════
step 6 "Verification"

CHECKS_PASSED=0
CHECKS_TOTAL=0

check() {
    CHECKS_TOTAL=$((CHECKS_TOTAL + 1))
    if eval "$2" 2>/dev/null; then
        ok "$1"
        CHECKS_PASSED=$((CHECKS_PASSED + 1))
    else
        warn "$1 — FAILED"
    fi
}

check "Python imports (rdkit)" "$PYTHON -c 'from rdkit import Chem; print(\"RDKit OK\")'"
check "Python imports (torch)" "$PYTHON -c 'import torch; print(f\"PyTorch {torch.__version__}\")'"
check "Python imports (transformers)" "$PYTHON -c 'import transformers; print(f\"Transformers {transformers.__version__}\")'"
check "Python imports (matplotlib)" "$PYTHON -c 'import matplotlib; print(f\"Matplotlib {matplotlib.__version__}\")'"
check "Python imports (scipy)" "$PYTHON -c 'from scipy import stats; print(\"SciPy OK\")'"
check "Evaluation scripts" "$PYTHON -c 'import py_compile; py_compile.compile(\"src/evaluation/evaluate.py\", doraise=True); py_compile.compile(\"src/evaluation/evaluate_shawn.py\", doraise=True); py_compile.compile(\"src/evaluation/compare_all_models.py\", doraise=True)'"
check "RL training script" "$PYTHON -c 'import py_compile; py_compile.compile(\"src/npgpt-rl/train_rl.py\", doraise=True)'"
check "Submodule: npgpt" "test -d external/npgpt"
check "Submodule: gp-molformer" "test -d external/gp-molformer"

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "  Checks passed: ${GREEN}${CHECKS_PASSED}${NC} / ${CHECKS_TOTAL}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

# ══════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  Setup Complete!                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next steps:"
echo "  1. Download data:     make download-all"
echo "  2. Process data:      make pipeline-all"
echo "  3. Evaluate models:"
echo "     NPGPT:             python src/evaluation/evaluate.py npgpt-rl"
echo "     GP-MoLFormer:      python src/evaluation/evaluate.py gpmolformer"
echo "     NPComposer:        python src/evaluation/evaluate_shawn.py"
echo "     Compare all:       python src/evaluation/compare_all_models.py"
echo "  4. RL fine-tuning:    python src/npgpt-rl/train_rl.py"
echo ""
