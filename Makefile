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
#   make test               Run pytest
#   make apptainer          Build Apptainer SIF image
#   make clean              Remove generated artifacts
#   make help               Show this help

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Configurable variables (override via CLI: make subset-coconut SIZE=100000)
PYTHON       ?= python3
PIP          ?= pip
SIZE         ?= 100000
SA_MAX       ?= 6.0
MAX_ATOMS    ?= 150
MAX_RINGS    ?= 10
SEED         ?= 42
CLASSIFY     ?= true
TRAIN_RATIO  ?= 0.8
VAL_RATIO    ?= 0.1
TEST_RATIO   ?= 0.1
N_SAMPLES    ?= 1000

# Paths (aligned with conf/config.yaml)
DATA_RAW       := data/raw
DATA_PROCESSED := data/processed
DATA_SPLITS    := data/splits
NPASS_DIR      := $(DATA_RAW)/npass

COCONUT_CSV := $(DATA_RAW)/coconut_csv_full.csv
COCONUT_SDF := $(DATA_RAW)/coconut_sdf_3d_full.sdf
NPASS_MERGED := $(DATA_RAW)/npass_full.csv
TRAINING_DATA := $(DATA_PROCESSED)/training_data.csv

# Apptainer
SIF_NAME     ?= npcomposer.sif
DEF_FILE     := npcomposer.def

# Setup

.PHONY: setup
setup: ## Install Python dependencies
	$(PIP) install -r requirements.txt

.PHONY: setup-dev
setup-dev: ## Install all dependencies including dev/test
	$(PIP) install -r requirements.txt

# Data Download

.PHONY: download-coconut
download-coconut: ## Download COCONUT dataset (~500 MB)
	bash scripts/download_data.sh

.PHONY: download-npass
download-npass: ## Download NPASS 3.0 dataset
	bash scripts/download_npass.sh

.PHONY: download-all
download-all: download-coconut download-npass ## Download all datasets

# NPASS Merge

.PHONY: merge-npass
merge-npass: ## Merge NPASS TSV files into single CSV
	@test -d $(NPASS_DIR) || { echo "Error: $(NPASS_DIR) not found. Run 'make download-npass' first."; exit 1; }
	$(PYTHON) scripts/merge_npass.py -i $(NPASS_DIR) -o $(NPASS_MERGED)

# Subset Creation (K-medoids)

.PHONY: subset-coconut
subset-coconut: ## Create COCONUT subset
	@test -f $(COCONUT_CSV) || { echo "Error: $(COCONUT_CSV) not found. Run 'make download-coconut' first."; exit 1; }
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) scripts/create_subset.py \
		-i $(COCONUT_CSV) \
		--sdf $(COCONUT_SDF) \
		-o $(DATA_PROCESSED)/coconut_$(SIZE) \
		-s $(SIZE) \
		--sa_max $(SA_MAX) \
		--max_atoms $(MAX_ATOMS) \
		--max_rings $(MAX_RINGS) \
		--seed $(SEED) \
		$(if $(filter true,$(CLASSIFY)),--classify,)

.PHONY: subset-npass
subset-npass: ## Create NPASS subset
	@test -f $(NPASS_MERGED) || { echo "Error: $(NPASS_MERGED) not found. Run 'make merge-npass' first."; exit 1; }
	@mkdir -p $(DATA_PROCESSED)
	$(PYTHON) scripts/create_subset.py \
		-i $(NPASS_MERGED) \
		-o $(DATA_PROCESSED)/npass_$(SIZE) \
		-s $(SIZE) \
		--sa_max $(SA_MAX) \
		--max_atoms $(MAX_ATOMS) \
		--max_rings $(MAX_RINGS) \
		--seed $(SEED) \
		$(if $(filter true,$(CLASSIFY)),--classify,)

# Merge Training Data

.PHONY: merge-training
merge-training: ## Merge COCONUT + NPASS subsets into training_data.csv
	@test -f $(DATA_PROCESSED)/coconut_$(SIZE).csv || { echo "Error: coconut subset not found. Run 'make subset-coconut' first."; exit 1; }
	@test -f $(DATA_PROCESSED)/npass_$(SIZE).csv || { echo "Error: npass subset not found. Run 'make subset-npass' first."; exit 1; }
	$(PYTHON) scripts/merge_training.py \
		--coconut $(DATA_PROCESSED)/coconut_$(SIZE).csv \
		--npass $(DATA_PROCESSED)/npass_$(SIZE).csv \
		-o $(TRAINING_DATA)

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
		$(if $(filter true,$(CLASSIFY)),--classify,)

# Full Pipelines

.PHONY: pipeline-coconut
pipeline-coconut: subset-coconut split-coconut ## Full COCONUT pipeline: subset -> split

.PHONY: pipeline-npass
pipeline-npass: merge-npass subset-npass split-npass ## Full NPASS pipeline: merge -> subset -> split

.PHONY: pipeline-all
pipeline-all: pipeline-coconut pipeline-npass merge-training ## Run both pipelines + merge

# One-shot targets (setup + download + pipeline)

.PHONY: all
all: setup download-all pipeline-all test ## Full run: setup -> download -> pipeline -> test

.PHONY: all-coconut
all-coconut: setup download-coconut pipeline-coconut ## Full COCONUT: setup -> download -> pipeline

.PHONY: all-npass
all-npass: setup download-npass pipeline-npass ## Full NPASS: setup -> download -> pipeline

# Testing

.PHONY: test
test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v

.PHONY: test-quick
test-quick: ## Run tests (no slow markers)
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
	rm -rf $(DATA_PROCESSED)/*.csv $(DATA_PROCESSED)/*.sdf
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
	@echo ""
