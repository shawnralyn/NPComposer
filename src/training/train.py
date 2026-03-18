"""Fine-tune NPComposer (a causal language model for SMILES) from CSV splits using a YAML config.

This script trains a Hugging Face `AutoModelForCausalLM` on SMILES strings optionally
conditioned on NPComposer v2 special tokens (e.g., pathway, superclass, glycoside flag,
aromatic ring count, QED bin, SA bin). It:

- Loads a base model + tokenizer from `configs["base"]`.
- Builds a set of conditioning special tokens from the training CSV columns and adds
  them to the tokenizer vocabulary, then resizes the model embeddings accordingly.
- Constructs a training text field of the form:
    "<pathway:...> <superclass:...> <is_glycoside:...> <aromatic_rings_count:...> <qed_bin:...> <sa_bin:...> SMILES<EOS>"
- Tokenizes the dataset with truncation and filters out overlong sequences.
- Uses a custom data collator that can stochastically drop conditioning tokens during
  training (conditioning dropout) and masks special tokens from loss computation.
- Runs training/evaluation via `transformers.Trainer` and `TrainingArguments`,
  then saves the model and optionally pushes it to the Hugging Face Hub.

Inputs:
- `--yaml`: path to a YAML configuration file defining model, columns, and training args.
- `--train_csv`, `--val_csv`, `--test_csv`: dataset splits as CSV files.

Outputs:
- A trained checkpoint written to `training.output_dir` (from the YAML), plus Trainer logs.
- If enabled in the YAML, a push to the Hugging Face Hub.

Example:
    python src/training/train.py --yaml configs/train.yml \
        --train_csv data/train.csv --val_csv data/val.csv --test_csv data/test.csv
"""

import os
import argparse
import yaml
import pandas as pd
import random
from typing import Any, Dict, IO, List, Optional, Tuple
import torch
from datasets import Dataset
import sys
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# filter 'fast_transformers' warning in stdout
class LineFilterStream:
    def __init__(self, stream: IO[str], banned_substrings: List[str]) -> None:
        """Stream wrapper to filter lines containing banned substrings.

        Input:
            stream: stream to wrap.
            banned_substrings: substrings to filter.
        """
        self.stream = stream
        self.banned = banned_substrings

    def write(self, s: str) -> Optional[int]:
        """Write string unless it contains a banned substring.

        Input:
            s: string to write.
        Output:
            result of stream write, or None if filtered.
        """
        if any(b in s for b in self.banned):
            return
        return self.stream.write(s)

    def flush(self) -> None:
        """Flush the wrapped stream."""
        return self.stream.flush()

sys.stdout = LineFilterStream(sys.stdout, ["No module named 'fast_transformers'"])
sys.stderr = LineFilterStream(sys.stderr, ["No module named 'fast_transformers'"])


def parse_yaml(yml: str) -> Optional[Dict[str, Any]]:
    """Read YAML configuration file.

    Input:
        yml: path to YAML file.
    Output:
        parsed YAML config dict or None if error.
    """
    try:
        with open(yml, "r") as file:
            configs = yaml.safe_load(file)
            return configs
    except FileNotFoundError:
        print(f"Error: The file {yml} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")


def build_special_class_tokens(
    ds_train: str,
    pathway_col: str,
    superclass_col: str,
    is_glycoside_col: str,
    num_aromatic_rings_col: str,
    qed_bin_col: str,
    sa_bin_col: str
) -> Dict[str, List[str]]:
    """Build special class tokens for tokenizer vocabulary.

    Input:
        ds_train: CSV path of COCONUT database (train split).
        pathway_col: column name for pathway.
        superclass_col: column name for superclass.
        is_glycoside_col: column name for glycoside.
        num_aromatic_rings_col: column name for aromatic rings count.
        qed_bin_col: column name for QED bin.
        sa_bin_col: column name for SA bin.
    Output:
        dict containing special tokens to add to tokenizer.
    """
    train_df = pd.read_csv(ds_train)
    special_tokens = set()
    for class_col in [
        pathway_col,
        superclass_col,
        is_glycoside_col,
        num_aromatic_rings_col,
        qed_bin_col,
        sa_bin_col,
    ]:
        classes = sorted(set(train_df[class_col].astype(str)))
        for c in classes:
            special_tokens.add(f"<{class_col}:{c}>")

    special_tokens_dict = {"additional_special_tokens": sorted(special_tokens)}

    return special_tokens_dict


def update_tokenizer_with_special_tokens(
    model: Any,
    tokenizer: Any,
    special_tokens_dict: Dict[str, List[str]]
) -> Tuple[Any, Any]:
    """Update tokenizer vocabulary and model embedding matrix.

    Input:
        model: transformer model to fine-tune.
        tokenizer: tokenizer associated with model.
        special_tokens_dict: dict of special class tokens.
    Output:
        updated model and tokenizer.
    """
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens.")

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def dataframe_to_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer: Any,
    smiles_col: str,
    pathway_col: str,
    superclass_col: str,
    is_glycoside_col: str,
    num_aromatic_rings_col: str,
    qed_bin_col: str,
    sa_bin_col: str,
    max_len: int,
    filter_len: int = 200,
) -> Dataset:
    """Convert pandas DataFrame to tokenized Hugging Face Dataset.

    Input:
        df: DataFrame with SMILES and class columns.
        tokenizer: tokenizer for text.
        smiles_col: column name for SMILES.
        pathway_col: column name for pathway.
        superclass_col: column name for superclass.
        is_glycoside_col: column name for glycoside.
        num_aromatic_rings_col: column name for aromatic rings count.
        qed_bin_col: column name for QED bin.
        sa_bin_col: column name for SA bin.
        max_len: maximum token length.
        filter_len: filter rows exceeding this token length (default 200).
    Output:
        tokenized Hugging Face Dataset.
    """
    for col in [smiles_col, pathway_col, superclass_col, is_glycoside_col, num_aromatic_rings_col, qed_bin_col, sa_bin_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df columns: {list(df.columns)}")

    df = df.dropna(subset=[smiles_col, pathway_col, superclass_col, is_glycoside_col, num_aromatic_rings_col, qed_bin_col, sa_bin_col]).copy()

    for col in [pathway_col, superclass_col, is_glycoside_col, num_aromatic_rings_col, qed_bin_col, sa_bin_col]:
        df[col] = df[col].astype(str)

    df["train_text"] = (
        "<" + pathway_col + ":" + df[pathway_col] + "> " +
        "<" + superclass_col + ":" + df[superclass_col] + "> " +
        "<" + is_glycoside_col + ":" + df[is_glycoside_col] + "> " +
        "<" + num_aromatic_rings_col + ":" + df[num_aromatic_rings_col] + "> " +
        "<" + qed_bin_col + ":" + df[qed_bin_col] + "> " +
        "<" + sa_bin_col + ":" + df[sa_bin_col] + "> " +
        df[smiles_col].astype(str) +
        (tokenizer.eos_token or "")
    )

    df["tokenized_len"] = df["train_text"].apply(
        lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"])
    )
    df = df[df["tokenized_len"] < filter_len].copy()

    ds = Dataset.from_pandas(df[["train_text"]], preserve_index=False)

    def _tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["train_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            padding=False,
            return_attention_mask=True,
        )

    ds = ds.map(_tokenize_batch, batched=True, remove_columns=["train_text"])
    return ds


def conditioning_dropout_collator(
    tokenizer: Any,
    special_token_dropout_prob: float,
    drop_all_special_tokens_prob: float
) -> Any:
    """Data collator with stochastic special token dropout.

    Randomly drops conditioning tokens during training, pads batches, and excludes
    special tokens from loss computation.

    Input:
        tokenizer: tokenizer for token IDs.
        special_token_dropout_prob: probability to drop each special token.
        drop_all_special_tokens_prob: probability to drop all special tokens.
    Output:
        data collator function.
    """
    pad_to_batch = DataCollatorWithPadding(tokenizer)
    special_token_ids = set(
        tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
    )

    def data_collator(batch_smiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        enable_dropout = torch.is_grad_enabled() and (special_token_dropout_prob > 0 or drop_all_special_tokens_prob > 0)

        if enable_dropout:
            for smiles_seq in batch_smiles:
                token_ids = smiles_seq["input_ids"]

                if drop_all_special_tokens_prob and random.random() < drop_all_special_tokens_prob:
                    new_token_ids = []
                    for t in token_ids:
                        if t not in special_token_ids:
                            new_token_ids.append(t)
                    token_ids = new_token_ids
                else:
                    new_token_ids = []
                    for t in token_ids:
                        if t in special_token_ids:
                            if random.random() < special_token_dropout_prob:
                                continue
                        new_token_ids.append(t)
                    token_ids = new_token_ids

                smiles_seq["input_ids"] = token_ids
                smiles_seq["attention_mask"] = [1] * len(token_ids)

        batch = pad_to_batch(batch_smiles)

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        for tok_id in special_token_ids:
            labels[batch["input_ids"] == tok_id] = -100

        batch["labels"] = labels

        return batch

    return data_collator


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Training yaml configuration file")
    ap.add_argument("--train_csv", required=True, help="Path to training split CSV")
    ap.add_argument("--test_csv", required=True, help="Path to test split CSV")
    ap.add_argument("--val_csv", required=True, help="Path to validation split CSV")
    args = ap.parse_args()

    configs = parse_yaml(args.yaml)

    # wandb environment variables
    os.environ["WANDB_PROJECT"] = configs["wandb"]["wandb_project"]
    os.environ["WANDB_LOG_MODEL"] = configs["wandb"]["wandb_log_model"]

    # load transformer model for fine-tuning
    model = AutoModelForCausalLM.from_pretrained(
        configs["base"]["model"], trust_remote_code=True
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # load tokenizer associated with transformer model
    tokenizer = AutoTokenizer.from_pretrained(
        configs["base"]["tokenizer"], trust_remote_code=True
    )

    special_tokens_dict = build_special_class_tokens(
        args.train_csv,
        configs["base"]["pathway_col"],
        configs["base"]["superclass_col"],
        configs["base"]["is_glycoside"],
        configs["base"]["aromatic_rings_count"],
        configs["base"]["qed_bin"],
        configs["base"]["sa_bin"],
    )

    model, tokenizer = update_tokenizer_with_special_tokens(
        model, tokenizer, special_tokens_dict
    )

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    val_df = pd.read_csv(args.val_csv)

    ds_train = dataframe_to_tokenized_dataset(
        train_df,
        tokenizer=tokenizer,
        smiles_col=configs["base"]["smiles_col"],
        pathway_col=configs["base"]["pathway_col"],
        superclass_col=configs["base"]["superclass_col"],
        is_glycoside_col=configs["base"]["is_glycoside"],
        num_aromatic_rings_col=configs["base"]["aromatic_rings_count"],
        qed_bin_col=configs["base"]["qed_bin"],
        sa_bin_col=configs["base"]["sa_bin"],
        max_len=configs["base"]["max_token_length"],
    )

    ds_test = dataframe_to_tokenized_dataset(
        test_df,
        tokenizer=tokenizer,
        smiles_col=configs["base"]["smiles_col"],
        pathway_col=configs["base"]["pathway_col"],
        superclass_col=configs["base"]["superclass_col"],
        is_glycoside_col=configs["base"]["is_glycoside"],
        num_aromatic_rings_col=configs["base"]["aromatic_rings_count"],
        qed_bin_col=configs["base"]["qed_bin"],
        sa_bin_col=configs["base"]["sa_bin"],
        max_len=configs["base"]["max_token_length"],
    )

    ds_val = dataframe_to_tokenized_dataset(
        val_df,
        tokenizer=tokenizer,
        smiles_col=configs["base"]["smiles_col"],
        pathway_col=configs["base"]["pathway_col"],
        superclass_col=configs["base"]["superclass_col"],
        is_glycoside_col=configs["base"]["is_glycoside"],
        num_aromatic_rings_col=configs["base"]["aromatic_rings_count"],
        qed_bin_col=configs["base"]["qed_bin"],
        sa_bin_col=configs["base"]["sa_bin"],
        max_len=configs["base"]["max_token_length"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_collator = conditioning_dropout_collator(
        tokenizer,
        special_token_dropout_prob=configs["training"]["special_token_dropout_prob"],
        drop_all_special_tokens_prob=configs["training"]["drop_all_special_tokens_prob"]
    )

    training_args = TrainingArguments(
        learning_rate=float(configs["training"]["learning_rate"]),
        num_train_epochs=float(configs["training"]["num_train_epochs"]),
        weight_decay=float(configs["training"]["weight_decay"]),
        warmup_ratio=float(configs["training"]["warmup_ratio"]),
        per_device_train_batch_size=int(configs["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(configs["training"]["per_device_eval_batch_size"]),
        output_dir=configs["training"]["output_dir"],
        fp16=False,
        report_to=configs["training"]["report_to"],
        run_name=configs["training"]["run_name"],
        eval_strategy=configs["training"]["evaluation_strategy"],
        save_strategy=configs["training"]["save_strategy"],
        save_safetensors=False,
        metric_for_best_model=configs["training"]["metric_for_best_model"],
        greater_is_better=bool(configs["training"]["greater_is_better"]),
        logging_steps=int(configs["training"]["logging_steps"]),
        save_total_limit=int(configs["training"]["save_total_limit"]),
        push_to_hub=bool(configs["training"]["push_to_hub"]),
        load_best_model_at_end=bool(configs["training"]["load_best_model_at_end"]),
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
    )

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Model device (pre-train):", next(model.parameters()).device)

    trainer.train()
    trainer.save_model()
    trainer.push_to_hub(repo_name=configs["training"]["HuggingFace_repo"])


if __name__ == "__main__":
    main()
