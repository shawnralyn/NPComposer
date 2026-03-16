"""
RL fine-tuning script for NPGPT using ACEGEN's REINFORCE pipeline.

This script implements:
    1. REINFORCE with experience replay (from ACEGEN)
    2. KL divergence regularization against pretrained prior
    3. Multi-objective reward (validity + QED + SA + NP-likeness)

Based on:
    - Thomas et al., "REINFORCE-ING Chemical Language Models for Drug Discovery", JCIM 2025.
    - Bou et al., "ACEGEN: Reinforcement Learning of Generative Chemical Agents", JCIM 2024.

Usage:
    python src/scripts/train_rl.py \\
        --checkpoint checkpoints/smiles-gpt/model.ckpt \\
        --tokenizer externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json \\
        --total_smiles 10000 \\
        --num_envs 128
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import tqdm

# Ensure the project root is on the path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "externals" / "smiles-gpt"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "src"))

# ACEGEN imports (submodule: external/acegen-open)
sys.path.insert(0, str(PROJECT_ROOT / "external" / "acegen-open"))

from acegen.rl_env import generate_complete_smiles, TokenEnv
from tensordict.utils import isin
from torchrl.data import (
    LazyTensorStorage,
    PrioritizedSampler,
    TensorDictMaxValueWriter,
    TensorDictReplayBuffer,
)
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules.utils import get_primers_from_module

# NPGPT imports
from npgpt.config import SmilesGptTrainingConfig
from npgpt.tokenizer import get_tokenizer
from npgpt.acegen_adapter import load_npgpt_for_acegen, NPGPTVocabulary, save_npgpt_checkpoint
from npgpt.reward import MultiObjectiveReward, RewardConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task wrapper (compatible with ACEGEN's scoring loop)
# ---------------------------------------------------------------------------


class NPGPTTask:
    """Simple task wrapper that tracks budget and calls the scoring function.

    Compatible with ACEGEN's `task.finished` / `task(smiles_list)` interface.
    """

    def __init__(self, scoring_function, budget: int, output_dir: str | None = None):
        self.scoring_function = scoring_function
        self.budget = budget
        self.total_calls = 0
        self.output_dir = output_dir
        self.all_smiles = []
        self.all_scores = []

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @property
    def finished(self) -> bool:
        return self.total_calls >= self.budget

    def __call__(self, smiles_list: list[str]) -> list[float]:
        scores = self.scoring_function(smiles_list)
        self.total_calls += len(smiles_list)
        self.all_smiles.extend(smiles_list)
        self.all_scores.extend(scores)
        return scores

    def save_results(self, path: str | None = None):
        path = path or (self.output_dir and os.path.join(self.output_dir, "results.json"))
        if path:
            with open(path, "w") as f:
                json.dump(
                    {"smiles": self.all_smiles, "scores": self.all_scores},
                    f,
                    indent=2,
                )


# ---------------------------------------------------------------------------
# Loss computation with KL regularization
# ---------------------------------------------------------------------------


def get_log_prob(data, model):
    """Compute log-probabilities of actions under the given model."""
    actions = data.get("action")
    model_in = data.select(*model.in_keys, strict=False)
    log_prob = model.get_dist(model_in).log_prob(actions)
    return log_prob


def get_prior_log_prob(data, prior_gpt2):
    """Compute per-step log-probabilities under the frozen prior GPT2 model.

    Uses the raw GPT2LMHeadModel directly (no ProbabilisticActor/TensorDict).

    Data shapes from generate_complete_smiles:
        - sequence:      [N, T]       (per-step observation tokens)
        - sequence_mask: [N, T, T]    (per-step causal attention mask)
        - action:        [N, T]       (chosen token at each step)

    GPT2 accepts 3D attention_mask as [batch, seq, seq] attention matrix.
    logits[:, t, :] predicts action[:, t] (next token after position t).
    """
    sequence = data.get("sequence")        # [N, T]
    seq_mask = data.get("sequence_mask")   # [N, T, T]
    actions = data.get("action")           # [N, T]

    with torch.no_grad():
        logits = prior_gpt2(
            input_ids=sequence,
            attention_mask=seq_mask.long(),
        ).logits  # [N, T, vocab_size]

        log_probs = torch.log_softmax(logits, dim=-1)  # [N, T, vocab_size]
        action_log_probs = log_probs.gather(
            -1, actions.unsqueeze(-1).long()
        ).squeeze(-1)  # [N, T]

    return action_log_probs


def compute_loss(data, actor_training, prior_gpt2=None, kl_coefficient=0.0):
    """Compute REINFORCE loss with optional KL regularization.

    Loss = -log_prob(actions) * reward + kl_coeff * KL(policy || prior)

    The KL regularization prevents mode collapse by penalizing deviation
    from the pretrained prior distribution (Thomas et al., JCIM 2025).
    """
    mask = data.get("mask").squeeze(-1)
    agent_log_prob = get_log_prob(data, actor_training)
    agent_likelihood = (agent_log_prob * mask).sum(-1)
    reward = data.get(("next", "reward")).squeeze(-1).sum(-1)

    # REINFORCE loss with baseline (variance reduction)
    # Subtracting mean reward reduces gradient variance → more stable training
    baseline = reward.mean()
    advantage = reward - baseline
    loss = -agent_likelihood * advantage

    # KL regularization against prior
    if prior_gpt2 is not None and kl_coefficient > 0.0:
        prior_log_prob = get_prior_log_prob(data, prior_gpt2)
        kl_div = (agent_log_prob - prior_log_prob) * mask
        kl_penalty = kl_div.sum(-1)
        loss = loss + kl_coefficient * kl_penalty

    return data, loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train_rl(args):
    """Main RL training loop for NPGPT."""

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------
    # 1. Load pretrained NPGPT
    # -------------------------------------------------------------------
    logger.info("Loading pretrained NPGPT model...")
    npgpt_config = SmilesGptTrainingConfig()
    actor_training, actor_inference, vocabulary, _ = load_npgpt_for_acegen(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        npgpt_config=npgpt_config,
    )

    actor_inference = actor_inference.to(device)
    actor_training = actor_training.to(device)

    # Create frozen prior for KL regularization (raw GPT2LMHeadModel)
    prior_gpt2 = None
    if args.kl_coefficient > 0.0:
        logger.info("Creating frozen prior for KL regularization...")
        from npgpt.model import SmilesGptModel
        hf_tokenizer = get_tokenizer(npgpt_config, args.tokenizer)
        ckpt_model = SmilesGptModel.load_from_checkpoint(
            args.checkpoint, config=npgpt_config, tokenizer=hf_tokenizer, strict=False,
        )
        prior_gpt2 = ckpt_model.model  # raw GPT2LMHeadModel
        prior_gpt2 = prior_gpt2.to(device)
        prior_gpt2.eval()
        for param in prior_gpt2.parameters():
            param.requires_grad = False

    logger.info(
        f"Model loaded. Vocabulary size: {len(vocabulary)}, "
        f"Architecture: {npgpt_config.n_layer}L-{npgpt_config.n_head}H-{npgpt_config.n_embd}E"
    )

    # -------------------------------------------------------------------
    # 2. Create reward function
    # -------------------------------------------------------------------
    reward_config = RewardConfig(
        w_validity=args.w_validity,
        w_qed=args.w_qed,
        w_sa=args.w_sa,
        w_np_likeness=args.w_np_likeness,
        invalid_penalty=args.invalid_penalty,
    )
    reward_fn = MultiObjectiveReward(reward_config)
    logger.info(f"Reward config: {reward_config}")

    # -------------------------------------------------------------------
    # 3. Create RL environment
    # -------------------------------------------------------------------
    env_kwargs = {
        "start_token": vocabulary.start_token_index,
        "end_token": vocabulary.end_token_index,
        "length_vocabulary": len(vocabulary),
        "batch_size": args.num_envs,
        "device": device,
        "max_length": args.max_length,
    }

    def create_env_fn():
        env = TokenEnv(**env_kwargs)
        env = TransformedEnv(env)
        env.append_transform(InitTracker())
        if primers := get_primers_from_module(actor_inference):
            env.append_transform(primers)
        return env

    env = create_env_fn()

    # -------------------------------------------------------------------
    # 4. Create task (budget-tracked scoring function)
    # -------------------------------------------------------------------
    output_dir = os.path.join(
        args.output_dir,
        f"npgpt_rl_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    task = NPGPTTask(
        scoring_function=reward_fn,
        budget=args.total_smiles,
        output_dir=output_dir,
    )

    # -------------------------------------------------------------------
    # 5. Create replay buffer
    # -------------------------------------------------------------------
    storage = LazyTensorStorage(args.replay_buffer_size, device=device)
    experience_replay_buffer = TensorDictReplayBuffer(
        storage=storage,
        sampler=PrioritizedSampler(storage.max_size, alpha=1.0, beta=1.0),
        batch_size=args.replay_batch_size,
        writer=TensorDictMaxValueWriter(rank_key="priority"),
        priority_key="priority",
    )

    # -------------------------------------------------------------------
    # 6. Create optimizer
    # -------------------------------------------------------------------
    optim = torch.optim.Adam(
        actor_training.parameters(),
        lr=args.lr,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )

    # -------------------------------------------------------------------
    # 7. Training loop
    # -------------------------------------------------------------------
    logger.info(f"Starting RL training. Budget: {args.total_smiles} SMILES")
    total_done = 0
    step = 0
    pbar = tqdm.tqdm(total=args.total_smiles, desc="RL Training")

    while not task.finished:
        step += 1

        # Generate data
        data = generate_complete_smiles(
            policy_sample=actor_inference,
            policy_evaluate=actor_training,
            vocabulary=vocabulary,
            scoring_function=task,
            environment=env,
            remove_duplicates=True,
            temperature=args.temperature,
        )

        log_info = {}
        data_next = data.get("next")
        done = data_next.get("done").squeeze(-1)
        total_done += done.sum().item()
        pbar.update(done.sum().item())

        # Log episode metrics
        episode_rewards = data_next["reward"][done]
        episode_length = (data_next["observation"] != 0.0).float().sum(-1).mean()
        if len(episode_rewards) > 0:
            log_info.update(
                {
                    "step": step,
                    "total_smiles": total_done,
                    "reward_mean": episode_rewards.mean().item(),
                    "reward_min": episode_rewards.min().item(),
                    "reward_max": episode_rewards.max().item(),
                    "episode_length": episode_length.item(),
                }
            )

        # Compute REINFORCE loss with KL regularization
        data, loss = compute_loss(
            data,
            actor_training,
            prior_gpt2=prior_gpt2,
            kl_coefficient=args.kl_coefficient,
        )

        # Experience replay loss
        if (
            args.experience_replay
            and len(experience_replay_buffer) > args.replay_batch_size
        ):
            replay_batch = experience_replay_buffer.sample()
            _, replay_loss = compute_loss(
                replay_batch,
                actor_training,
                prior_gpt2=prior_gpt2,
                kl_coefficient=args.kl_coefficient,
            )
            loss = torch.cat((loss, replay_loss), 0)

        # Backprop
        loss = loss.mean()
        optim.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(actor_training.parameters(), args.max_grad_norm)

        optim.step()

        # Update replay buffer
        if args.experience_replay:
            replay_data = data.clone()
            replay_data.batch_size = [replay_data.batch_size[0]]

            if len(experience_replay_buffer) > 0:
                is_duplicated = isin(
                    input=replay_data,
                    key="action",
                    reference=experience_replay_buffer[:],
                )
                replay_data = replay_data[~is_duplicated]

            if len(replay_data) > 0:
                reward = replay_data.get(("next", "reward"))
                replay_data.set("priority", reward)
                experience_replay_buffer.extend(replay_data)

        # Log
        if log_info and step % args.log_every == 0:
            log_info["loss"] = loss.item()
            logger.info(
                f"Step {step} | "
                f"SMILES: {total_done} | "
                f"Reward: {log_info.get('reward_mean', 0):.4f} "
                f"(min={log_info.get('reward_min', 0):.3f}, max={log_info.get('reward_max', 0):.3f}) | "
                f"Loss: {loss.item():.4f}"
            )

        # Save checkpoint periodically
        if step % args.save_every == 0:
            save_path = os.path.join(output_dir, f"actor_step_{step}.pt")
            acegen_path, npgpt_path = save_npgpt_checkpoint(
                actor=actor_inference,
                save_path=save_path,
                tokenizer_path=args.tokenizer,
                npgpt_config=npgpt_config,
            )
            logger.info(f"Checkpoint saved: {acegen_path} (ACEGEN) / {npgpt_path} (NPGPT)")

    pbar.close()

    # -------------------------------------------------------------------
    # 8. Save final model and results
    # -------------------------------------------------------------------
    final_path = os.path.join(output_dir, "actor_final.pt")
    acegen_path, npgpt_path = save_npgpt_checkpoint(
        actor=actor_inference,
        save_path=final_path,
        tokenizer_path=args.tokenizer,
        npgpt_config=npgpt_config,
    )
    task.save_results()

    logger.info(f"Training complete!")
    logger.info(f"  ACEGEN actor:  {acegen_path}  (for resuming RL training)")
    logger.info(f"  NPGPT model:   {npgpt_path}  (for inference with inference.py)")
    logger.info(f"  Results:       {output_dir}/results.json")

    # Print summary statistics
    scores = np.array(task.all_scores)
    valid_scores = scores[scores > 0]  # exclude invalid (penalty) molecules
    logger.info(
        f"\nFinal stats over {len(scores)} generated molecules:\n"
        f"  Valid ratio:  {(scores > 0).mean():.4f}\n"
        f"  Mean reward:  {valid_scores.mean():.4f}\n"
        f"  Max reward:   {valid_scores.max():.4f}\n"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="RL fine-tuning of NPGPT using REINFORCE + experience replay"
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(SCRIPT_DIR / "npgpt.ckpt"),
        help="Path to pretrained NPGPT checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(PROJECT_ROOT / "externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"),
        help="Path to HuggingFace tokenizer file",
    )

    # Device
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")

    # RL environment
    parser.add_argument("--num_envs", type=int, default=128, help="Parallel SMILES generation batch size")
    parser.add_argument("--total_smiles", type=int, default=10_000, help="Total SMILES generation budget")
    parser.add_argument("--max_length", type=int, default=200, help="Max SMILES token sequence length")

    # Reward weights
    parser.add_argument("--w_validity", type=float, default=1.0, help="Weight for validity reward")
    parser.add_argument("--w_qed", type=float, default=0.3, help="Weight for QED reward")
    parser.add_argument("--w_sa", type=float, default=0.3, help="Weight for SA reward")
    parser.add_argument("--w_np_likeness", type=float, default=0.4, help="Weight for NP-likeness reward")
    parser.add_argument("--invalid_penalty", type=float, default=-0.5, help="Penalty for invalid SMILES")

    # RL algorithm
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping (0 = disabled)")
    parser.add_argument("--kl_coefficient", type=float, default=0.01, help="KL divergence regularization coefficient (0 = disabled)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for molecule generation")

    # Experience replay
    parser.add_argument("--experience_replay", action="store_true", default=True)
    parser.add_argument("--no_experience_replay", action="store_false", dest="experience_replay")
    parser.add_argument("--replay_buffer_size", type=int, default=100, help="Replay buffer capacity")
    parser.add_argument("--replay_batch_size", type=int, default=10, help="Replay sample batch size")

    # Logging & saving
    parser.add_argument("--output_dir", type=str, default="results/rl", help="Output directory")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_rl(args)
