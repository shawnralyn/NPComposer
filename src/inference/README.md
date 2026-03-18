# Inference

This folder contains simple scripts for running inference with trained models.

## Files

- `inference.py`
  - Main inference entry point.
  - Loads a Hugging Face causal LM checkpoint (default: NPComposer), builds an optional conditioning prompt using NPComposer v2 special tokens, generates SMILES via sampling, and writes outputs to disk.
  - Supports:
    - **Single-prompt mode**: generate `num_molecules` for one prompt.
    - **Batch special-token mode**: generate molecules for every pathway/superclass special token in a provided `special_tokens_map.json`.

- `test_inference.py`
  - Minimal smoke test that generates a single SMILES and scores it with RDKit (QED/SA).

- `test_inference.ipynb`
  - Notebook version of the inference workflow (interactive exploration).
  - Useful for quick generation + **visualization** (e.g., rendering SMILES as 2D molecule depictions).

- `infer_np_drug.py`
  - Separate inference script for NP→drug candidate generation using GP-MoLFormer adapters (prompt-tuning/LoRA). This is not required for NPComposer single-token conditional generation.

## Basic usage

### 1) Run inference from a YAML config

```bash
python src/inference/inference.py --yaml conf/inference.yaml
```

The YAML is expected to contain an `inference` section with keys such as:
- `ckpt_path` (Hugging Face model name or local path)
- `output_file`
- `num_molecules`
- `top_p`, `temperature`
- optional `seed`

### 2) Override conditions from the command line

You can build the conditioning prompt from individual fields (pathway, superclass, glycoside flag, aromatic ring count, QED bin, SA bin). See `inference.py --help` for the full set of arguments.

### 3) Quick smoke test

```bash
python src/inference/test_inference.py
```

### 4) Notebook (interactive generation + visualization)

Open `src/inference/test_inference.ipynb` to quickly generate molecules and visualize them in-notebook.

## Output format

Generated sequences are decoded and post-processed by taking the first fragment before a `.` (period). The resulting SMILES strings are written one-per-line to output text files.

## Notes

- Inference uses `trust_remote_code=True` for NPComposer checkpoints.
- If CUDA is available, tensors are moved to GPU automatically.
