# NPGPT-RL: Reinforcement Learning Fine-tuning for NPGPT

NPGPT-RL applies the REINFORCE policy gradient algorithm to fine-tune the pretrained [NPGPT](https://github.com/ohuelab/npgpt) model (~2.6M parameters), optimizing for multi-objective molecular properties. The RL approach is based on [Thomas et al. (2025)](https://doi.org/10.1021/acs.jcim.5c02053) and uses the [AceGen](https://github.com/acellera/acegen-open) framework.

![NPGPT-RL Pipeline](../../docs/npgpt_rl_pipeline.png)

## Overview

The RL fine-tuning pipeline consists of the following components:

| File | Description |
|------|-------------|
| `train_rl.py` | Main RL training script (REINFORCE + experience replay + KL regularization) |
| `reward.py` | Multi-objective reward function (validity, QED, SA, NP-likeness) |
| `acegen_adapter.py` | Adapter to bridge NPGPT (HuggingFace GPT-2) with AceGen's RL environment |
| `sweep_checkpoints.py` | Evaluate all checkpoints and plot metrics vs training step |
| `compare_models.py` | Compare pretrained vs RL-finetuned model with histograms |
| `npgpt.ckpt` | Pretrained NPGPT checkpoint |
| `npgpt_rl_step_600.ckpt` | Best RL-finetuned checkpoint (step 600) |

## Algorithm

NPGPT-RL uses REINFORCE with the following components:

**Policy Gradient (REINFORCE with baseline)**:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (R - b)\right]$$

where $b = \mathbb{E}[R]$ is the baseline for variance reduction.

**KL Regularization**:

A frozen copy of the pretrained model ($\pi_{\text{ref}}$) serves as a reference to prevent mode collapse:

$$\mathcal{L} = -\log \pi_\theta(a|s) \cdot (R - b) + \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{ref}})$$

**Multi-Objective Reward**:

$$R = \frac{w_v \cdot r_{\text{valid}} + w_q \cdot r_{\text{QED}} + w_s \cdot r_{\text{SA}} + w_n \cdot r_{\text{NP}}}{w_v + w_q + w_s + w_n}$$

For invalid SMILES, $R = -0.5$ (penalty).

| Component | Weight | Range | Description |
|-----------|--------|-------|-------------|
| Validity | 1.0 | {0, 1} | RDKit parseable |
| QED | 0.3 | [0, 1] | Drug-likeness (Bickerton et al.) |
| SA | 0.3 | [0, 1] | Synthetic accessibility (inverted, normalized) |
| NP-likeness | 0.4 | [0, 1] | Natural product likeness (sigmoid normalized) |
| Invalid penalty | — | -0.5 | Applied when SMILES is invalid |

## Usage

### 1. RL Training

```bash
# Default settings (recommended)
python src/NPGPT-rl/train_rl.py

# Custom settings
python src/NPGPT-rl/train_rl.py \
    --checkpoint src/NPGPT-rl/npgpt.ckpt \
    --tokenizer externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json \
    --total_smiles 10000 \
    --num_envs 128 \
    --kl_coefficient 0.01 \
    --lr 1e-4 \
    --save_every 50

# GPU training
python src/NPGPT-rl/train_rl.py --device cuda

# MPS (Apple Silicon)
python src/NPGPT-rl/train_rl.py --device mps
```

**Key Arguments**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | `src/NPGPT-rl/npgpt.ckpt` | Pretrained NPGPT checkpoint |
| `--tokenizer` | `externals/smiles-gpt/.../tokenizer.json` | HuggingFace tokenizer |
| `--total_smiles` | 10000 | Total SMILES generation budget |
| `--num_envs` | 128 | Parallel generation batch size |
| `--lr` | 1e-4 | Learning rate |
| `--kl_coefficient` | 0.01 | KL regularization strength ($\beta$) |
| `--temperature` | 1.0 | Sampling temperature |
| `--save_every` | 50 | Checkpoint save interval (steps) |
| `--experience_replay` | True | Enable experience replay buffer |
| `--max_grad_norm` | 1.0 | Gradient clipping |
| `--device` | auto | Device (auto/cpu/cuda/mps) |

**Reward Weight Arguments**:

| Argument | Default |
|----------|---------|
| `--w_validity` | 1.0 |
| `--w_qed` | 0.3 |
| `--w_sa` | 0.3 |
| `--w_np_likeness` | 0.4 |
| `--invalid_penalty` | -0.5 |

**Output**: Checkpoints are saved to `results/rl/npgpt_rl_<timestamp>/` in two formats:
- `actor_step_N.pt` — AceGen actor state (for resuming RL training)
- `actor_step_N.pt.npgpt.ckpt` — NPGPT-compatible checkpoint (for inference)

### 2. Checkpoint Sweep

Evaluate all saved RL checkpoints and find the best one:

```bash
python src/NPGPT-rl/sweep_checkpoints.py
```

This generates 100 molecules per checkpoint (temp=1.5, top_p=1.0), computes validity/QED/SA/NP-likeness, and saves a plot to `results/sweep_metrics.png`.

### 3. Model Comparison

Compare pretrained vs RL-finetuned model:

```bash
# Default (uses step 600 checkpoint)
python src/NPGPT-rl/compare_models.py

# Custom settings
python src/NPGPT-rl/compare_models.py \
    --n_samples 500 \
    --temperature 1.5 \
    --seed 42 \
    --rl_ckpt src/NPGPT-rl/npgpt_rl_step_600.ckpt
```

Outputs a comparison histogram to `results/comparison_histogram.png` with:
- Validity bar chart
- QED / SA / NP-likeness distribution histograms
- Sanitization error breakdown (parse / valence / kekulize)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        NPGPT-RL Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────────┐  │
│  │  NPGPT   │───▶│  AceGen       │───▶│  Generate SMILES     │  │
│  │Pretrained│    │  Adapter      │    │  Batch (128 envs)    │  │
│  └──────────┘    │(acegen_adapter│    └──────────┬───────────┘  │
│       │          │    .py)       │               │              │
│       │          └───────────────┘               ▼              │
│       │                              ┌──────────────────────┐   │
│       │                              │  Multi-Objective     │   │
│       │                              │  Reward (reward.py)  │   │
│       │                              │  ┌────┬────┬───┬───┐ │   │
│       │                              │  │Val │QED │SA │NP │ │   │
│       │                              │  │1.0 │0.3 │0.3│0.4│ │   │
│       │                              │  └────┴────┴───┴───┘ │   │
│       │                              └──────────┬───────────┘   │
│       │                                         │               │
│       ▼                                         ▼               │
│  ┌──────────┐                        ┌──────────────────────┐   │
│  │  πref    │─── KL Divergence ────▶ │  REINFORCE Loss      │   │
│  │ (Frozen) │                        │  + KL Penalty        │   │
│  └──────────┘                        │  + Exp. Replay       │   │
│                                      └──────────┬───────────┘   │
│                                                 │               │
│                                                 ▼               │
│                                      ┌──────────────────────┐   │
│                                      │  Update πθ (Adam)    │   │
│                                      │  grad clip = 1.0     │   │
│                                      └──────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  Checkpoint saved every 50 steps → sweep → best: step 600      │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies

NPGPT-RL requires the following additional packages beyond the base NPComposer dependencies:

```
torchrl
tensordict
acegen (submodule: external/acegen-open)
rdkit
numpy
matplotlib
```

These are installed automatically via `bash setup.sh` or `make setup`.

## References

- Thomas, M. et al. "REINFORCE-ING Chemical Language Models for Drug Discovery." *J. Chem. Inf. Model.* 65, 12752–12763 (2025). [DOI](https://doi.org/10.1021/acs.jcim.5c02053)
- Bou, A. et al. "ACEGEN: Reinforcement Learning of Generative Chemical Agents for Drug Discovery." *J. Chem. Inf. Model.* 64(15), 5900–5911 (2024). [DOI](https://doi.org/10.1021/acs.jcim.4c00895)
- Sakano, K. et al. "NPGPT: Natural Product-Like Compound Generation with GPT-based Chemical Language Models." *J. Supercomput.* 81, 1–20 (2025).
