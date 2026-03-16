"""ACEGEN adapter for NPGPT language model integration."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import ExplorationType
from torchrl.modules import MaskedCategorical, ProbabilisticActor

from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from npgpt.config import SmilesGptTrainingConfig
from npgpt.tokenizer import get_tokenizer


class NPGPTVocabulary:
    """Vocabulary adapter for NPGPT HF tokenizer to ACEGEN interface."""

    def __init__(self, hf_tokenizer: PreTrainedTokenizerFast):
        self.hf_tokenizer = hf_tokenizer
        self.start_token_index = hf_tokenizer.bos_token_id
        self.end_token_index = hf_tokenizer.eos_token_id
        self.vocab = hf_tokenizer.get_vocab()
        self.reversed_vocab = {v: k for k, v in self.vocab.items()}
        self.vocab_size = hf_tokenizer.vocab_size
        self.start_token = hf_tokenizer.bos_token
        self.end_token = hf_tokenizer.eos_token
        self.chars = list(self.vocab.keys())

    def __len__(self) -> int:
        return self.vocab_size

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.reversed_vocab.get(key, "")
        if isinstance(key, str):
            return self.vocab.get(key, 0)

    def encode(
        self, smiles: str, with_start: bool = True, with_end: bool = True
    ):
        """Encode SMILES string to token index array.

        Input:
            smiles: SMILES string.
            with_start: prepend start token.
            with_end: append end token.
        Output:
            token indices as numpy array.
        """
        import numpy as np

        token_ids = self.hf_tokenizer.encode(smiles, add_special_tokens=False)

        if with_start:
            token_ids = [self.start_token_index] + token_ids
        if with_end:
            token_ids = token_ids + [self.end_token_index]

        return np.array(token_ids, dtype=np.float32)

    def decode(self, token_indices, ignore_indices=()) -> str:
        """Decode token index array to SMILES string.

        Input:
            token_indices: array of token indices.
            ignore_indices: indices to skip.
        Output:
            decoded SMILES string.
        """
        indices = []
        for idx in token_indices:
            idx = int(idx)
            if idx in ignore_indices:
                continue
            if idx == self.start_token_index:
                continue
            if idx == self.end_token_index:
                break
            indices.append(idx)
        return self.hf_tokenizer.decode(indices, skip_special_tokens=False)

    def add_characters(self, chars):
        """No-op for BPE tokenizer (vocabulary is fixed)."""
        pass


class NPGPTFeatureExtractor(nn.Module):
    """Feature extractor wrapping NPGPT GPT2Model."""

    def __init__(self, gpt2_model: GPT2LMHeadModel):
        super().__init__()
        self.feature_extractor = gpt2_model.transformer
        self._train_mode = False

    @property
    def train_mode(self) -> bool:
        return self._train_mode

    def set_train_mode(self, train_mode: bool = True) -> "NPGPTFeatureExtractor":
        if train_mode is self._train_mode:
            return self
        out = NPGPTFeatureExtractor.__new__(NPGPTFeatureExtractor)
        nn.Module.__init__(out)
        out.feature_extractor = self.feature_extractor
        out._train_mode = train_mode
        return out

    def forward(self, sequence: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(
            input_ids=sequence,
            attention_mask=sequence_mask.long(),
        ).last_hidden_state

        if not self._train_mode:
            obs_length = sequence_mask.sum(-1)
            out = out[torch.arange(len(out), device=out.device), obs_length.to(torch.int64) - 1]

        return out


def create_npgpt_actor(
    vocabulary_size: int,
    checkpoint_path: str | None = None,
    tokenizer_path: str | None = None,
    npgpt_config: SmilesGptTrainingConfig | None = None,
    action_mask_key: str = "action_mask",
    return_log_prob: bool = True,
):
    """Create NPGPT actor for ACEGEN RL pipeline.

    Input:
        vocabulary_size: token vocabulary size.
        checkpoint_path: path to pretrained checkpoint (.ckpt).
        tokenizer_path: path to HF tokenizer (.json).
        npgpt_config: NPGPT training configuration.
        action_mask_key: key for action masking.
        return_log_prob: whether to return log probabilities.
    Output:
        tuple of (training_actor, inference_actor) with shared weights.
    """
    if npgpt_config is None:
        npgpt_config = SmilesGptTrainingConfig()

    gpt2_config = GPT2Config(
        vocab_size=vocabulary_size,
        bos_token_id=1,
        eos_token_id=2,
        n_layer=npgpt_config.n_layer,
        n_head=npgpt_config.n_head,
        n_embd=npgpt_config.n_embd,
        n_positions=npgpt_config.max_length,
        n_ctx=npgpt_config.max_length,
    )

    gpt2_model = GPT2LMHeadModel(gpt2_config)

    if checkpoint_path is not None:
        from npgpt.model import SmilesGptModel

        if tokenizer_path is None:
            raise ValueError("tokenizer_path is required when loading a checkpoint.")

        tokenizer = get_tokenizer(npgpt_config, tokenizer_path)
        ckpt_model = SmilesGptModel.load_from_checkpoint(
            checkpoint_path,
            config=npgpt_config,
            tokenizer=tokenizer,
            strict=False,
        )
        gpt2_model.load_state_dict(ckpt_model.model.state_dict())

    lm = NPGPTFeatureExtractor(gpt2_model)

    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    lm_head_linear = nn.Linear(gpt2_config.n_embd, vocabulary_size, bias=False)
    lm_head_linear.weight = gpt2_model.lm_head.weight

    lm_head = TensorDictModule(
        lm_head_linear,
        in_keys=["features"],
        out_keys=["logits"],
    )

    policy_training = TensorDictSequential(lm_training, lm_head)
    policy_inference = TensorDictSequential(lm_inference, lm_head)

    if action_mask_key:
        inf_keys = {"logits": "logits", "mask": action_mask_key}
        inf_dist = MaskedCategorical
    else:
        inf_keys = ["logits"]
        inf_dist = torch.distributions.Categorical

    probabilistic_policy_training = ProbabilisticActor(
        module=policy_training,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=torch.distributions.Categorical,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    probabilistic_policy_inference = ProbabilisticActor(
        module=policy_inference,
        in_keys=inf_keys,
        out_keys=["action"],
        distribution_class=inf_dist,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )

    return probabilistic_policy_training, probabilistic_policy_inference


def create_npgpt_critic(
    vocabulary_size: int,
    npgpt_config: SmilesGptTrainingConfig | None = None,
    critic_value_per_action: bool = False,
):
    """Create NPGPT critic for actor-critic RL algorithms.

    Input:
        vocabulary_size: token vocabulary size.
        npgpt_config: NPGPT training configuration.
        critic_value_per_action: value per action vs state value.
    Output:
        tuple of (critic_training, critic_inference).
    """
    if npgpt_config is None:
        npgpt_config = SmilesGptTrainingConfig()

    gpt2_config = GPT2Config(
        vocab_size=vocabulary_size,
        n_layer=npgpt_config.n_layer,
        n_head=npgpt_config.n_head,
        n_embd=npgpt_config.n_embd,
        n_positions=npgpt_config.max_length,
        n_ctx=npgpt_config.max_length,
    )

    gpt2_model = GPT2LMHeadModel(gpt2_config)
    lm = NPGPTFeatureExtractor(gpt2_model)

    lm_training = TensorDictModule(
        lm.set_train_mode(True),
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )
    lm_inference = TensorDictModule(
        lm,
        in_keys=["sequence", "sequence_mask"],
        out_keys=["features"],
    )

    lm_head = TensorDictModule(
        nn.Linear(
            gpt2_config.n_embd,
            vocabulary_size if critic_value_per_action else 1,
            bias=False,
        ),
        in_keys=["features"],
        out_keys=["action_value"] if critic_value_per_action else ["state_value"],
    )

    critic_training = TensorDictSequential(lm_training, lm_head)
    critic_inference = TensorDictSequential(lm_inference, lm_head)
    return critic_training, critic_inference


def load_npgpt_for_acegen(
    checkpoint_path: str,
    tokenizer_path: str,
    npgpt_config: SmilesGptTrainingConfig | None = None,
) -> tuple:
    """Load pretrained NPGPT model and vocabulary for ACEGEN RL training.

    Input:
        checkpoint_path: path to NPGPT checkpoint.
        tokenizer_path: path to HF tokenizer.
        npgpt_config: NPGPT training configuration.
    Output:
        (actor_training, actor_inference, vocabulary, npgpt_config)
    """
    if npgpt_config is None:
        npgpt_config = SmilesGptTrainingConfig()

    hf_tokenizer = get_tokenizer(npgpt_config, tokenizer_path)
    vocabulary = NPGPTVocabulary(hf_tokenizer)

    actor_training, actor_inference = create_npgpt_actor(
        vocabulary_size=len(vocabulary),
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        npgpt_config=npgpt_config,
    )

    return actor_training, actor_inference, vocabulary, npgpt_config


def _extract_gpt2_state_dict(actor) -> dict:
    """Extract GPT2LMHeadModel-compatible state_dict from ACEGEN actor.

    Input:
        actor: ACEGEN actor wrapping GPT2 model.
    Output:
        GPT2LMHeadModel state_dict.
    """
    actor_sd = actor.state_dict()
    gpt2_sd = {}

    for key, value in actor_sd.items():
        if "feature_extractor" in key:
            idx = key.index("feature_extractor")
            hf_key = "transformer." + key[idx + len("feature_extractor."):]
            gpt2_sd[hf_key] = value
        elif key.endswith(".weight") and "feature_extractor" not in key:
            gpt2_sd["lm_head.weight"] = value

    return gpt2_sd


def save_npgpt_checkpoint(
    actor,
    save_path: str,
    tokenizer_path: str | None = None,
    npgpt_config: SmilesGptTrainingConfig | None = None,
):
    """Save RL-trained actor as NPGPT-compatible checkpoint.

    Saves two files: ACEGEN actor state and GPT2LMHeadModel state.

    Input:
        actor: ACEGEN ProbabilisticActor.
        save_path: base path to save checkpoint.
        tokenizer_path: path to tokenizer.
        npgpt_config: model config.
    Output:
        (save_path, npgpt_path)
    """
    import os

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    torch.save(actor.state_dict(), save_path)

    gpt2_sd = _extract_gpt2_state_dict(actor)
    npgpt_path = save_path + ".npgpt.ckpt"
    save_data = {
        "state_dict": {f"model.{k}": v for k, v in gpt2_sd.items()},
    }
    if npgpt_config is not None:
        save_data["hyper_parameters"] = {"config": npgpt_config.model_dump()}
    if tokenizer_path is not None:
        save_data["tokenizer_path"] = tokenizer_path

    torch.save(save_data, npgpt_path)

    return save_path, npgpt_path


def load_rl_checkpoint(
    checkpoint_path: str,
    tokenizer_path: str,
    npgpt_config: SmilesGptTrainingConfig | None = None,
) -> tuple:
    """Load RL checkpoint for resuming training or inference.

    Input:
        checkpoint_path: path to ACEGEN actor state_dict.
        tokenizer_path: path to HF tokenizer.
        npgpt_config: model config.
    Output:
        (actor_training, actor_inference, vocabulary, npgpt_config)
    """
    if npgpt_config is None:
        npgpt_config = SmilesGptTrainingConfig()

    hf_tokenizer = get_tokenizer(npgpt_config, tokenizer_path)
    vocabulary = NPGPTVocabulary(hf_tokenizer)

    actor_training, actor_inference = create_npgpt_actor(
        vocabulary_size=len(vocabulary),
        npgpt_config=npgpt_config,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    actor_inference.load_state_dict(ckpt)

    return actor_training, actor_inference, vocabulary, npgpt_config
