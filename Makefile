# NPComposer Makefile
#
# Usage:
#   make all                Full run (setup -> download -> pipeline -> test)
#   make all-coconut        COCONUT only (setup -> download -> pipeline)
#   make all-npass          NPASS only (setup -> download -> pipeline)
#   make pipeline-all       Both pipelines + merge training data
#   make pipeline-coconut   COCONUT pipeline (subset -> split)
#   make pipeline-npass     NPASS pipeline (merge -> subset -> split)
#   make merge-training     Merge COCONUT + NPASS subsets into training_data.csv
#   make analyze-dist       Compare raw vs processed distributions
#   make eval-shawn         Evaluate Shawn_model1 (superclass-conditioned)
#   make eval-npgpt         Evaluate NPGPT baseline (unconditional)
#   make molecule MODEL=npgpt   Generate & visualize a single molecule
#   make eval-all           Run all model evaluations
#   make test               Run pytest
#   make apptainer          Build Apptainer SIF image
#   make clean              Remove generated artifacts
#   make help               Show this help
#
#   make pipeline-np-drug       NP→drug data pipeline (clean COCONUT + ChEMBL + generate pairs)
#   make train-np-drug-lora     Train NP→drug LoRA model
#   make eval-np-drug-lora      Evaluate LoRA on synthetic test pairs
#   make eval-np-drug-real      Evaluate LoRA on real NP→drug pairs
#   make infer-np-drug SMILES=  Generate drug candidates for a single NP

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Configurable variables (override via CLI: make subset-coconut SIZE=100000)
PYTHON       ?= python3
PIP          ?= pip
SIZE         ?= 10000
SA_MAX       ?= 6.0
MAX_ATOMS    ?= 150
MAX_RINGS    ?= 10
FP_DIM       ?= 3
SEED         ?= 42
N_JOBS       ?= -1
TRAIN_RATIO  ?= 0.8
VAL_RATIO    ?= 0.1
TEST_RATIO   ?= 0.1
N_SAMPLES    ?= 1000

# NP→Drug pipeline
CHEMBL_CSV      ?= data/raw/chembl_drugs.csv
COCONUT_CLEAN   ?= data/processed/coconut_npdrug.csv
CHEMBL_CLEAN    ?= data/processed/chembl_npdrug.csv
PAIRS_CSV       ?= data/processed/ChEMBL_pairs.csv
LORA_DIR        ?= models/np_drug_lora_r16
LORA_RANK       ?= 16
TRAIN_EPOCHS    ?= 50
TRAIN_BATCH     ?= 256
TRAIN_LR        ?= 1e-4
K               ?= 10
WANDB_PROJECT   ?= npcomposer

# Paths (aligned with conf/config.yaml)
DATA_RAW       := data/raw
DATA_PROCESSED := data/processed
DATA_SPLITS    := data/splits
NPASS_DIR      := $(DATA_RAW)/npass

COCONUT_CSV := $(DATA_RAW)/coconut_csv_full.csv
COCONUT_SDF := $(DATA_RAW)/coconut_sdf_3d_full.sdf
NPASS_MERGED := $(DATA_RAW)/npass_full.csv
TRAINING_DATA := $(DATA_PROCESSED)/training_data.csv
NP_DRUG_DATA := $(DATA_PROCESSED)/np_drug.csv

# Apptainer
SIF_NAME     ?= npcomposer.sif
DEF_FILE     := npcomposer.def

# Setup

.PHONY: setup
setup: ## Install Python dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install matplotlib scipy -q

.PHONY: setup-full
setup-full: ## Full automated setup (deps + submodules + checkpoints)
	bash setup.sh

.PHONY: setup-gpu
setup-gpu: ## Full setup with GPU-enabled PyTorch
	bash setup.sh --gpu

.PHONY: setup-dev
setup-dev: ## Install all dependencies including dev/test
	$(PIP) install -r requirements.txt
	$(PIP) install matplotlib scipy -q

# Data Download

.PHONY: download-coconut
download-coconut: ## Download COCONUT dataset (~500 MB)
	bash scripts/download_data.sh

.PHONY: download-npass
download-npass: ## Download NPASS 3.0 dataset
	bash scripts/download_npass.sh

.PHONY: download-np_drug
download-np_drug: ## Download np drug dataset (~60 MB)
	bash scripts/download_np_drug.sh

.PHONY: download-all
download-all: download-coconut download-npass download-np_drug ## Download all datasets

# NPASS Merge

.PHONY: merge-npass
merge-npass: ## Merge NPASS TSV files into single CSV
	@test -d $(NPASS_DIR) || { echo "Error: $(NPASS_DIR) not found. Run 'make download-npass' first."; exit 1; }
	$(PYTHON) scripts/merge_npass.py -i $(NPASS_DIR) -o $(NPASS_MERGED)

# Subset Creation (K-means in Tanimoto space)

.PHONY: subset-coconut
subset-coconut: ## Create COCONUT subset
	@test -f $(COCONUT_CSV) || { echo "Error: $(COCONUT_CSV) not found. Run 'make download-coconut' first."; exit 1; }
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) scripts/create_subset.py \
		-i $(COCONUT_CSV) \
		$(if $(wildcard $(COCONUT_SDF)),--sdf $(COCONUT_SDF),) \
		-o $(DATA_PROCESSED)/coconut_$(SIZE) \
		-s $(SIZE) \
		--sa_max $(SA_MAX) \
		--max_atoms $(MAX_ATOMS) \
		--max_rings $(MAX_RINGS) \
		--fp_dim $(FP_DIM) \
		--n_jobs $(N_JOBS) \
		--seed $(SEED) \


.PHONY: subset-npass
subset-npass: ## Create NPASS subset
	@test -f $(NPASS_MERGED) || { echo "Error: $(NPASS_MERGED) not found. Run 'make merge-npass' first."; exit 1; }
	@test $$(wc -l < $(NPASS_MERGED)) -gt 1 || { echo "Error: $(NPASS_MERGED) is empty. Run 'make merge-npass' first."; exit 1; }
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) scripts/create_subset.py \
		-i $(NPASS_MERGED) \
		-o $(DATA_PROCESSED)/npass_$(SIZE) \
		-s $(SIZE) \
		--sa_max $(SA_MAX) \
		--max_atoms $(MAX_ATOMS) \
		--max_rings $(MAX_RINGS) \
		--fp_dim $(FP_DIM) \
		--n_jobs $(N_JOBS) \
		--seed $(SEED) \


# Merge Training Data

.PHONY: merge-training
merge-training: ## Merge COCONUT + NPASS subsets into training_data.csv
	@test -f $(DATA_PROCESSED)/coconut_$(SIZE).csv || { echo "Error: coconut subset not found. Run 'make subset-coconut' first."; exit 1; }
	@test -f $(DATA_PROCESSED)/npass_$(SIZE).csv || { echo "Error: npass subset not found. Run 'make subset-npass' first."; exit 1; }
	$(PYTHON) scripts/merge_training.py \
		--coconut $(DATA_PROCESSED)/coconut_$(SIZE).csv \
		--npass $(DATA_PROCESSED)/npass_$(SIZE).csv \
		-o $(TRAINING_DATA)

# ── NP→Drug Pipeline ──────────────────────────────────────────────────

.PHONY: clean-coconut-npdrug
clean-coconut-npdrug: ## Clean COCONUT for NP→drug pipeline
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) src/data_preprocessing/np_drug/clean_coconut_npdrug.py \
		--input  $(COCONUT_CSV) \
		--output $(COCONUT_CLEAN)

.PHONY: clean-chembl-npdrug
clean-chembl-npdrug: ## Clean ChEMBL for NP→drug pipeline
	@test -f $(CHEMBL_CSV) || { echo "Error: $(CHEMBL_CSV) not found."; exit 1; }
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) src/data_preprocessing/np_drug/clean_ChEMBL.py \
		--input  $(CHEMBL_CSV) \
		--output $(CHEMBL_CLEAN)

.PHONY: generate-pairs
generate-pairs: ## Generate COCONUT×ChEMBL FAISS pairs
	@test -f $(COCONUT_CLEAN) || { echo "Error: run make clean-coconut-npdrug first."; exit 1; }
	@test -f $(CHEMBL_CLEAN)  || { echo "Error: run make clean-chembl-npdrug first."; exit 1; }
	$(PYTHON) src/data_preprocessing/np_drug/generate_ChEMBL_pairs.py \
		--coconut $(COCONUT_CLEAN) \
		--chembl  $(CHEMBL_CLEAN) \
		--output  $(PAIRS_CSV)

.PHONY: train-np-drug-lora
train-np-drug-lora: ## Train NP→drug LoRA model
	@test -f $(PAIRS_CSV) || { echo "Error: $(PAIRS_CSV) not found. Run make generate-pairs first."; exit 1; }
	$(PYTHON) src/training/train_np_drug_pairs.py \
		--pairs-csv  $(PAIRS_CSV) \
		--output-dir $(LORA_DIR) \
		--num-epochs $(TRAIN_EPOCHS) \
		--batch-size $(TRAIN_BATCH) \
		--lr         $(TRAIN_LR) \
		--lora \
		--lora-rank  $(LORA_RANK) \
		--no-eval \
		--save-epochs 5 \
		--wandb-project $(WANDB_PROJECT)

.PHONY: eval-np-drug-lora
eval-np-drug-lora: ## Evaluate LoRA checkpoint on synthetic test pairs
	$(PYTHON) src/evaluation/eval_np_drug_pairs.py \
		--checkpoint $(LORA_DIR)/final \
		--test-csv   $(LORA_DIR)/test_pairs.csv \
		--lora --lora-rank $(LORA_RANK) \
		--k $(K) \
		--output-csv $(LORA_DIR)/eval_results.csv

.PHONY: eval-np-drug-real
eval-np-drug-real: ## Evaluate LoRA checkpoint on real NP→drug pairs
	$(PYTHON) src/evaluation/eval_np_drug_pairs.py \
		--checkpoint      $(LORA_DIR)/final \
		--test-csv        $(NP_DRUG_DATA) \
		--np-smiles-col   parent_np_smiles \
		--drug-smiles-col drug_smiles \
		--lora --lora-rank $(LORA_RANK) \
		--k $(K) \
		--output-csv $(LORA_DIR)/eval_real_pairs.csv

.PHONY: infer-np-drug
infer-np-drug: ## Generate drug candidates for a single NP (SMILES=<smiles>)
ifndef SMILES
	$(error SMILES is required. Usage: make infer-np-drug SMILES="CCO...")
endif
	$(PYTHON) src/evaluation/infer_np_drug.py \
		--checkpoint $(LORA_DIR)/final \
		--smiles "$(SMILES)" \
		--lora --lora-rank $(LORA_RANK) \
		--k $(K)

.PHONY: pipeline-np-drug
pipeline-np-drug: clean-coconut-npdrug clean-chembl-npdrug generate-pairs ## Full NP→drug data pipeline

# Create Drug Discovery Dataset

.PHONY: create-drug-dataset
create-drug-dataset: ## refine np drug dataset into np_drug.csv
	@test -f $(DATA_RAW)/np_drug.xlsx || { echo "Error: np drug dataset not found. Run 'make download-np_drug' first."; exit 1; }
	$(PYTHON) scripts/clean_npdrug.py \
		-i $(DATA_RAW)/np_drug.xlsx \
		-o $(NP_DRUG_DATA)

# Train / Val / Test Split

.PHONY: split-coconut
split-coconut: ## Split COCONUT subset into train/val/test
	@mkdir -p $(DATA_SPLITS)
	$(PYTHON) scripts/split_data.py \
		-i $(DATA_PROCESSED)/coconut_$(SIZE).csv \
		-o $(DATA_SPLITS) \
		--train $(TRAIN_RATIO) \
		--val $(VAL_RATIO) \
		--test $(TEST_RATIO) \
		--seed $(SEED)

.PHONY: split-npass
split-npass: ## Split NPASS subset into train/val/test
	@mkdir -p $(DATA_SPLITS)
	$(PYTHON) scripts/split_data.py \
		-i $(DATA_PROCESSED)/npass_$(SIZE).csv \
		-o $(DATA_SPLITS) \
		--train $(TRAIN_RATIO) \
		--val $(VAL_RATIO) \
		--test $(TEST_RATIO) \
		--seed $(SEED)

# SDF Analysis

.PHONY: analyze-coconut
analyze-coconut: ## Analyze COCONUT SDF file
	$(PYTHON) scripts/analyze_sdf.py \
		-i $(COCONUT_SDF) \
		-n $(N_SAMPLES) \
		-o $(DATA_PROCESSED)/coconut_analysis.txt

# Evaluation

.PHONY: evaluate
evaluate: ## Evaluate generated molecules (set INPUT=<file>)
ifndef INPUT
	$(error INPUT is required. Usage: make evaluate INPUT=generated.txt)
endif
	$(PYTHON) src/evaluation/metrics.py \
		-i $(INPUT) \
		-o results.json \


# ── New Unified Evaluation System ──────────────────────────────────────
EVAL_SEEDS   ?= "1 2 3"
EVAL_NUM     ?= 50

.PHONY: eval-npgpt
eval-npgpt: ## Evaluate NPGPT pretrained (unconditional)
	$(PYTHON) src/evaluation/evaluate.py npgpt \
		--seeds $(EVAL_SEEDS) --n_samples $(EVAL_NUM)

.PHONY: eval-npgpt-rl
eval-npgpt-rl: ## Evaluate NPGPT pretrained vs RL-finetuned
	$(PYTHON) src/evaluation/evaluate.py npgpt-rl \
		--seeds $(EVAL_SEEDS) --n_samples $(EVAL_NUM)

.PHONY: eval-gpmolformer
eval-gpmolformer: ## Evaluate GP-MoLFormer baseline
	$(PYTHON) src/evaluation/evaluate.py gpmolformer \
		--seeds $(EVAL_SEEDS) --n_samples $(EVAL_NUM)

.PHONY: eval-npcomposer
eval-npcomposer: ## Evaluate NPComposer (4 configs: QED+SA, 3 pathways)
	$(PYTHON) src/evaluation/evaluate_shawn.py \
		--seeds $(EVAL_SEEDS) --n_samples $(EVAL_NUM)

.PHONY: eval-npcomposer-classify
eval-npcomposer-classify: ## Evaluate NPComposer + pathway classification accuracy
	$(PYTHON) src/evaluation/evaluate_shawn.py \
		--seeds $(EVAL_SEEDS) --n_samples $(EVAL_NUM) --classify

.PHONY: eval-compare
eval-compare: ## Compare all models (bar chart + significance)
	$(PYTHON) src/evaluation/compare_all_models.py

.PHONY: eval-benchmark
eval-benchmark: ## Benchmark generation speed (50 mol x 3 seeds per model)
	$(PYTHON) src/evaluation/benchmark_speed.py \
		--n_molecules 50 --seeds "1 2 3"

.PHONY: eval-all
eval-all: eval-npgpt-rl eval-gpmolformer eval-npcomposer eval-compare eval-benchmark eval-np-drug-lora eval-np-drug-real ## Run all evaluations + comparison + benchmark

# ── Molecule Generation ───────────────────────────────────────────────
MODEL    ?= npgpt
PROMPT   ?=

.PHONY: molecule
molecule: ## Generate & display a single molecule (MODEL=npgpt|npgpt-rl|gpmolformer|npcomposer PROMPT=...)
	$(PYTHON) scripts/make_molecule.py $(MODEL) \
		$(if $(PROMPT),--prompt "$(PROMPT)",)

# ── Legacy Evaluation (shell scripts) ─────────────────────────────────
EVAL_TOP     ?= 0

.PHONY: eval-shawn-legacy
eval-shawn-legacy: ## [Legacy] Evaluate Shawn_model1 via shell script
	bash scripts/run_evaluation_Shawn_model1.sh \
		--seeds $(EVAL_SEEDS) \
		--num $(EVAL_NUM) \
		$(if $(filter-out 0,$(EVAL_TOP)),--top $(EVAL_TOP),)

.PHONY: eval-npgpt-legacy
eval-npgpt-legacy: ## [Legacy] Evaluate NPGPT via shell script
	bash scripts/run_evaluation_NPGPT.sh \
		--seeds $(EVAL_SEEDS) \
		--num 760

# Distribution Analysis

.PHONY: analyze-dist
analyze-dist: ## Compare raw vs processed distributions (histograms + stats)
	@test -f $(COCONUT_CSV) || { echo "Error: $(COCONUT_CSV) not found."; exit 1; }
	@test -f $(DATA_PROCESSED)/coconut_$(SIZE).csv || { echo "Error: coconut subset not found. Run 'make subset-coconut' first."; exit 1; }
	$(PYTHON) scripts/analyze_distribution.py \
		--raw $(COCONUT_CSV) \
		--processed $(DATA_PROCESSED)/coconut_$(SIZE).csv \
		-o $(DATA_PROCESSED)/dist_coconut
	@if [ -f $(NPASS_MERGED) ] && [ -f $(DATA_PROCESSED)/npass_$(SIZE).csv ]; then \
		$(PYTHON) scripts/analyze_distribution.py \
			--raw $(NPASS_MERGED) \
			--processed $(DATA_PROCESSED)/npass_$(SIZE).csv \
			-o $(DATA_PROCESSED)/dist_npass; \
	fi

# Full Pipelines

.PHONY: pipeline-coconut
pipeline-coconut: subset-coconut split-coconut ## Full COCONUT pipeline: subset -> split

.PHONY: pipeline-npass
pipeline-npass: merge-npass subset-npass split-npass ## Full NPASS pipeline: merge -> subset -> split

.PHONY: pipeline-np_drug
pipeline-np_drug: create-drug-dataset ## Full COCONUT pipeline: subset -> split

.PHONY: pipeline-all
pipeline-all: pipeline-coconut pipeline-npass pipeline-np_drug merge-training analyze-dist ## Run both pipelines + merge + analyze

# One-shot targets (setup + download + pipeline)

.PHONY: all
all: setup download-all pipeline-all test ## Full run: setup -> download -> pipeline -> test

.PHONY: all-coconut
all-coconut: setup download-coconut pipeline-coconut ## Full COCONUT: setup -> download -> pipeline

.PHONY: all-npass
all-npass: setup download-npass pipeline-npass ## Full NPASS: setup -> download -> pipeline

# Testing

.PHONY: test
test: ## Run all tests (editable install + pytest)
	$(PIP) install --no-deps -e .
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-quick
test-quick: ## Run tests (no slow markers)
	$(PIP) install --no-deps -e .
	$(PYTHON) -m pytest tests/ -v -m "not slow"

# Apptainer / Singularity

.PHONY: apptainer
apptainer: $(DEF_FILE) ## Build Apptainer SIF image
	apptainer build $(SIF_NAME) $(DEF_FILE)

.PHONY: apptainer-sandbox
apptainer-sandbox: $(DEF_FILE) ## Build writable sandbox (for debugging)
	apptainer build --sandbox npcomposer_sandbox/ $(DEF_FILE)

.PHONY: shell
shell: $(SIF_NAME) ## Open interactive shell in container
	apptainer shell $(SIF_NAME)

.PHONY: run
run: $(SIF_NAME) ## Run default pipeline inside container
	apptainer run $(SIF_NAME)

# Cleanup

.PHONY: clean
clean: ## Remove processed data, splits, and caches
	rm -rf $(DATA_PROCESSED)/*.csv $(DATA_PROCESSED)/*.sdf $(DATA_PROCESSED)/*.png
	rm -rf $(DATA_SPLITS)/*.csv
	rm -rf results.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf outputs/ multirun/

.PHONY: clean-all
clean-all: clean ## Also remove raw data and container image
	rm -rf $(DATA_RAW)/*.csv $(DATA_RAW)/*.zip $(DATA_RAW)/*.sdf
	rm -rf $(NPASS_DIR)
	rm -f $(SIF_NAME)
	rm -rf npcomposer_sandbox/

# Help

.PHONY: help
help: ## Show available targets
	@echo ""
	@echo "NPComposer — Available targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Override variables:  make subset-coconut SIZE=100000 SA_MAX=5.0 SEED=123"
	@echo "Evaluation:         make eval-shawn EVAL_NUM=10 EVAL_SEEDS='1 2 3' EVAL_TOP=3"
	@echo "NPClassifier:      python src/evaluation/evaluate_shawn.py --classify"
	@echo ""
