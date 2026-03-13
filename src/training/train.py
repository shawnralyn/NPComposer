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
        """
        Stream wrapper to filter lines containing banned substrings.

        Args:
            stream (IO[str]): The original stream to wrap.
            banned_substrings (List[str]): Substrings to filter out from output.
        """
        self.stream = stream
        self.banned = banned_substrings

    def write(self, s: str) -> Optional[int]:
        """
        Write string to stream unless it contains a banned substring.

        Args:
            s (str): String to write.

        Returns:
            Optional[int]: Result of stream write, or None if filtered.
        """
        if any(b in s for b in self.banned):
            return
        return self.stream.write(s)

    def flush(self) -> None:
        """
        Flush the wrapped stream.
        """
        return self.stream.flush()

sys.stdout = LineFilterStream(sys.stdout, ["No module named 'fast_transformers'"])
sys.stderr = LineFilterStream(sys.stderr, ["No module named 'fast_transformers'"])


def parse_yaml(yml: str) -> Optional[Dict[str, Any]]:
    """
    Read in yaml configuration file

    Args:
        yml (str): Path to YAML file.

    Returns:
        Optional[Dict[str, Any]]: Parsed YAML configs or None if error.
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
    """
    Creates special class tokens for each value in each class column and returns dictionary to update tokenizer vocabulary

    inputs:
        - ds_train: csv path of COCONUT database (train split)
        - pathway_col, superclass_col, ...: column names specifying natural product classes

    outputs:
        - special_tokens_dict (dict): Dict containing special tokens specifying natural product class to add to tokenizer

    additional:
        - special tokens are added so that they won't be split during tokenization

    Args:
        ds_train (str): CSV path of COCONUT database (train split).
        pathway_col (str): Column name for pathway.
        superclass_col (str): Column name for superclass.
        is_glycoside_col (str): Column name for glycoside.
        num_aromatic_rings_col (str): Column name for aromatic rings count.
        qed_bin_col (str): Column name for QED bin.
        sa_bin_col (str): Column name for SA bin.

    Returns:
        Dict[str, List[str]]: Dict containing special tokens to add to tokenizer.
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

    # make dict for passing to tokenizer
    special_tokens_dict = {"additional_special_tokens": sorted(special_tokens)}

    return special_tokens_dict


def update_tokenizer_with_special_tokens(
    model: Any,
    tokenizer: Any,
    special_tokens_dict: Dict[str, List[str]]
) -> Tuple[Any, Any]:
    """
    Updates tokenizer vocabulary with special tokens and resizes model embedding matrix to match tokenizer vocabulary
    size

    inputs:
        -model: Loaded transformer model to be fine-tuned
        -tokenizer: Tokenizer associated with transformer model
        -special_tokens_dict (dict): Dictionary formatted to add special class tokens to tokenizer vocabulary

    outputs:
        -model: Transformer model with resized embedding matrix to match tokenizer vocabulary size after addition of
        special tokens
        -tokenizer: Tokenizer with updated vocabulary including special tokens

    Args:
        model: Loaded transformer model to be fine-tuned.
        tokenizer: Tokenizer associated with transformer model.
        special_tokens_dict (Dict[str, List[str]]): Dict of special class tokens.

    Returns:
        Tuple[Any, Any]: Updated model and tokenizer.
    """

    # add special tokens dict to tokenizer vocabulary
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added} special tokens.")

    # make model embedding matrix match tokenizer vocabulary size
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
    """
    Convert a pandas DataFrame with SMILES + class columns into a Hugging Face Dataset
    that contains tokenized fields ready for training.

    Args:
        df (pd.DataFrame): DataFrame containing data.
        tokenizer: Tokenizer for tokenizing text.
        smiles_col (str): Column name for SMILES.
        pathway_col (str): Column name for pathway.
        superclass_col (str): Column name for superclass.
        is_glycoside_col (str): Column name for glycoside.
        num_aromatic_rings_col (str): Column name for aromatic rings count.
        qed_bin_col (str): Column name for QED bin.
        sa_bin_col (str): Column name for SA bin.
        max_len (int): Maximum token length.
        filter_len (int, optional): Filter out rows exceeding this token length. Defaults to 200.

    Returns:
        Dataset: Tokenized Hugging Face Dataset.
    """
    # Ensure all columns exist
    for col in [smiles_col, pathway_col, superclass_col, is_glycoside_col, num_aromatic_rings_col, qed_bin_col, sa_bin_col]:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in df columns: {list(df.columns)}")

    # Drop rows with missing values in any required column
    df = df.dropna(subset=[smiles_col, pathway_col, superclass_col, is_glycoside_col, num_aromatic_rings_col, qed_bin_col, sa_bin_col]).copy()

    # Ensure all class columns are strings
    for col in [pathway_col, superclass_col, is_glycoside_col, num_aromatic_rings_col, qed_bin_col, sa_bin_col]:
        df[col] = df[col].astype(str)

    # Create string for training per row in df
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

    # print statements for dry run
    # print("\n--- train_text examples ---")
    # for i, s in enumerate(df["train_text"].head(5).tolist()):
    #     print(f"[{i}] {s}")

    # print("\n--- tokenization check (special tokens should remain intact) ---")
    # for s in df["train_text"].sample(n=3, random_state=0).tolist():
    #     ids = tokenizer(s, add_special_tokens=False)["input_ids"]
    #     toks = tokenizer.convert_ids_to_tokens(ids)
    #     print("TEXT:", s[:200], "..." if len(s) > 200 else "")
    #     print("TOKS:", toks[:60])
    #     print()

    # Filter out rows where tokenized length exceeds filter_len
    df["tokenized_len"] = df["train_text"].apply(
        lambda x: len(tokenizer(x, add_special_tokens=False)["input_ids"])
    )
    df = df[df["tokenized_len"] < filter_len].copy()

    # create Dataset from df using only the text column containing formatted examples
    ds = Dataset.from_pandas(df[["train_text"]], preserve_index=False)

    def _tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        return tokenizer(
            batch["train_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            padding=False,  # pad using data collator
            return_attention_mask=True,
        )

    ds = ds.map(_tokenize_batch, batched=True, remove_columns=["train_text"])
    return ds


def conditioning_dropout_collator(
    tokenizer: Any,
    special_token_dropout_prob: float,
    drop_all_special_tokens_prob: float
) -> Any:
    """
    The data collator takes a batch of tokenized inputs and processes them to produce tensors 
    ready for input to the model during training and evaluation. 

    This custom data collator will:
    1. randomly drop conditioning special tokens during training
    2. pad variable length examples in a batch
    3. exclude conditioning tokens and padding from next-token loss in training and evaluation

    Assumptions:
    - conditioning tokens are in tokenizer.additional_special_tokens

    Args:
        tokenizer: Tokenizer for token IDs.
        special_token_dropout_prob (float): Probability to drop each special token.
        drop_all_special_tokens_prob (float): Probability to drop all special tokens.

    Returns:
        Any: Data collator function.
    """

    pad_to_batch = DataCollatorWithPadding(tokenizer) # load base data collator

    # define set of special token ids for dropout
    special_token_ids = set( 
        tokenizer.convert_tokens_to_ids(tokenizer.additional_special_tokens)
    )

    def data_collator(batch_smiles: List[Dict[str, Any]]) -> Dict[str, Any]:
        # enable stochastic conditioning label dropout only during training when .is_grad_enabled() == True
        enable_dropout = torch.is_grad_enabled() and (special_token_dropout_prob > 0 or drop_all_special_tokens_prob > 0) 

        if enable_dropout:
            for smiles_seq in batch_smiles:                
                token_ids = smiles_seq["input_ids"] # extract list of token IDs for single input 

                # some percentage of the time remove all special conditioning tokens
                if drop_all_special_tokens_prob and random.random() < drop_all_special_tokens_prob:
                    new_token_ids = []
                    for t in token_ids:
                        if t not in special_token_ids:
                            new_token_ids.append(t)
                    token_ids = new_token_ids
                else:
                    # drop each special token independently with probability given by special_token_dropout_prob
                    new_token_ids = []
                    for t in token_ids:
                        if t in special_token_ids:
                            if random.random() < special_token_dropout_prob:
                                continue # drop special token
                        new_token_ids.append(t)
                    token_ids = new_token_ids
                        
                smiles_seq["input_ids"] = token_ids # write back the input sequence with stochastic dropout applied
                smiles_seq["attention_mask"] = [1] * len(token_ids)  # update attention mask with new sequence length

        batch = pad_to_batch(batch_smiles) # pad each sequence in batch to same length (to produce rectangular tensor)                

        labels = batch["input_ids"].clone()                
        labels[batch["attention_mask"] == 0] = -100 # mask padded positions from loss       

        # ignore special conditioning tokens in the loss
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

    # load csvs into pandas df
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    val_df = pd.read_csv(args.val_csv)

    # create tokenized training dataset
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

    # create tokenized test dataset
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

    # create tokenized validation dataset
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

    # ensure tokenizer has padding token for data collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # specify custom data collator that enables stochastic special token dropout during training. 
    data_collator = conditioning_dropout_collator(tokenizer, special_token_dropout_prob=configs["training"]["special_token_dropout_prob"], 
                                                  drop_all_special_tokens_prob=configs["training"]["drop_all_special_tokens_prob"])

    # define arguments for training
    training_args = TrainingArguments(
        learning_rate=float(configs["training"]["learning_rate"]),
        num_train_epochs=float(configs["training"]["num_train_epochs"]),
        weight_decay=float(configs["training"]["weight_decay"]),
        warmup_ratio=float(configs["training"]["warmup_ratio"]),
        per_device_train_batch_size=int(configs["training"]["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(configs["training"]["per_device_eval_batch_size"]),
        output_dir=configs["training"]["output_dir"],
        fp16=False,  # turn off mixed precision 

        # W&B
        report_to=configs["training"]["report_to"],
        run_name=configs["training"]["run_name"],

        # Eval/save
        eval_strategy=configs["training"]["evaluation_strategy"],
        save_strategy=configs["training"]["save_strategy"],
        save_safetensors=False,
        metric_for_best_model=configs["training"]["metric_for_best_model"],
        greater_is_better=bool(configs["training"]["greater_is_better"]),
        logging_steps=int(configs["training"]["logging_steps"]),
        save_total_limit=int(configs["training"]["save_total_limit"]),
        push_to_hub=bool(configs["training"]["push_to_hub"]),
        load_best_model_at_end=bool(configs["training"]["load_best_model_at_end"]),

        # dry run
        # eval_strategy="no",
        # max_steps=100,
        # logging_steps=1,
        # push_to_hub=False,
        # eval_steps=5,
        # save_strategy="no",
    )

    # Trainer contains all necessary components of a training loop
    # 1. Calculates loss from a training step
    # 2. Calculates the gradients 
    # 3. Update the model weights based on gradients
    # 4. Repeat for predetermined number of epochs
    trainer = Trainer(  # Trainer will automatically use GPU if available (device does not have to be manually specified)
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

    trainer.train()  # starts training
    trainer.save_model()  # save model so you can reload it using "from_pretrained()"
    
    trainer.push_to_hub(repo_name=configs["training"]["HuggingFace_repo"]) # push best model to HuggingFace Hub


if __name__ == "__main__":
    main()
