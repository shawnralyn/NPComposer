# Training (NPComposer)

This folder contains scripts and utilities for fine-tuning **GP-MolFormer**, a causal language model for SMILES generation, using Hugging Face `transformers`.

## What `train.py` does

`train.py` fine-tunes a base `AutoModelForCausalLM` on SMILES strings, optionally **conditioned on NPComposer v2 special tokens** (e.g., pathway, superclass, glycoside flag, aromatic ring count, QED bin, SA bin).

High-level flow:

1. **Load config** from a YAML file (`--yaml`).
2. **Load base model/tokenizer** from `configs["base"]`.
3. **Create & add conditioning tokens**
   - Reads the training CSV and collects unique values from the configured conditioning columns.
   - Builds tokens of the form `<{column}:{value}>`.
   - Adds them to the tokenizer as `additional_special_tokens` and resizes model embeddings.
4. **Build training text** per example:

   ```text
   <pathway:...> <superclass:...> <is_glycoside:...> <aromatic_rings_count:...> <qed_bin:...> <sa_bin:...> SMILES<eos>
   ```

5. **Tokenize + filter**
   - Tokenizes with truncation (`max_token_length` from YAML).
   - Filters out examples whose tokenized length exceeds `filter_len` (default: 200 in `dataframe_to_tokenized_dataset`).
6. **Train with `Trainer`**
   - Runs training/eval according to `TrainingArguments` from the YAML.
   - Uses a custom data collator that supports **conditioning dropout** and masks special tokens from contributing to the loss.
7. **Save / push**
   - Saves the trained model to `training.output_dir`.
   - Optionally pushes to the Hugging Face Hub (`push_to_hub`).

## Inputs

### Required CLI arguments

- `--yaml`: training configuration YAML.
- `--train_csv`: training split CSV.
- `--val_csv`: validation split CSV.
- `--test_csv`: test split CSV.

### Expected CSV columns

Your YAML must define the SMILES column and all conditioning columns under `base`, for example:

- `smiles_col`
- `pathway_col`
- `superclass_col`
- `is_glycoside`
- `aromatic_rings_count`
- `qed_bin`
- `sa_bin`

`train.py` will raise an error if any required column is missing.

## Outputs

- A trained checkpoint in `configs["training"]["output_dir"]`.
- Standard Hugging Face Trainer logs.
- If enabled, a Hub push to `configs["training"]["HuggingFace_repo"]`.

## Usage

From the repository root:

- Basic run:

  ```bash
  python src/training/train.py \
    --yaml conf/train.yaml \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --test_csv data/splits/test.csv
  ```

Adjust paths to match your project.

## Notes / Tips

- **Conditioning dropout**: controlled by
  - `training.special_token_dropout_prob`
  - `training.drop_all_special_tokens_prob`
  This helps the model remain usable when some conditioning signals are missing.

- **Padding**: if the tokenizer has no `pad_token`, the script assigns `pad_token = eos_token`.

- **Device**: `Trainer` will select CPU/GPU automatically (CUDA if available). The script prints CUDA availability and the pre-training device.

- **Reproducibility**: seed handling is not currently centralized in `train.py`; if you need strict reproducibility, set seeds in the YAML/CLI and add explicit `torch`/`numpy`/`random` seeding.
