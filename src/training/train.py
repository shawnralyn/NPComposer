import pandas as pd
import argparse
import yaml
#from datasets import Dataset
#import tqdm
from transformers import (
AutoModelForCausalLM, AutoTokenizer,
DataCollatorForLanguageModeling,
Trainer, TrainingArguments,
#     wandb, Evaluator
)
from typing import Any, Dict
from datasets import Dataset

# drop molecules longer than 200 tokens 
# track which files used for training

def parse_yaml(yml):
    """
    Read in yaml configuration file
    """
    try:
        with open(yml, 'r') as file:
            configs = yaml.safe_load(file)
            return configs
    except FileNotFoundError:
        print(f"Error: The file {yml} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")


def build_special_class_tokens(ds_train, class_col):
    """
    Creates special class tokens for each value in class_col and returns dictionary to update tokenizer vocabulary

    inputs:
        -ds_train: csv path of COCONUT database (train split)
        -class_col: column name specifying natural product class (i.e. "np_classifier_pathway")

    outputs:
        -special_tokens (dict): Dict containing special tokens specifying natural product class to add to tokenizer

    additional:
        -special tokens are added (rather than tokens) so that they won't be split during tokenization
    """
    train_df = pd.read_csv(ds_train)
    classes = sorted(set(train_df[class_col]))
    special_tokens = [f"<NP:{c}>" for c in classes]
    
    # make dict for passing to tokenizer
    special_tokens_dict = {"additional_special_tokens": special_tokens}

    return special_tokens_dict


def update_tokenizer_with_special_tokens(model, tokenizer, special_tokens_dict):
    """
    Updates tokenizer vocabulary with special tokens and resizes model embedding matrix to match tokenizer
    vocabulary size

    inputs: 
        -model: Loaded transformer model to be fine-tuned
        -tokenizer: Tokenizer associated with transformer model
        -special_tokens_dict (dict): Dictionary formatted to add special class tokens to tokenizer vocabulary

    outputs:
        -model: Transformer model with resized embedding matrix to match tokenizer vocabulary size after addition of special tokens
        -tokenizer: Tokenizer with updated vocabulary including special tokens
    """

    # add special tokens dict to tokenizer vocabulary
    num_added = tokenizer.add_special_tokens(special_tokens_dict)

    # make model embedding matrix match tokenizer vocabulary size
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def dataframe_to_tokenized_dataset(
    df: pd.DataFrame,
    tokenizer,
    smiles_col: str,
    class_col: str,
    max_len: int,
) -> Dataset:
    """
    Convert a pandas DataFrame with SMILES + class into a Hugging Face Dataset
    that contains tokenized fields ready for training.
    """
    if smiles_col not in df.columns:
        raise KeyError(f"smiles_col '{smiles_col}' not found in df columns: {list(df.columns)}")
    if class_col not in df.columns:
        raise KeyError(f"class_col '{class_col}' not found in df columns: {list(df.columns)}")

    df = df.dropna(subset=[smiles_col, class_col]).copy()
    df[class_col] = df[class_col].astype(str) # ensure entries are strings

    # Create string for training per row in df
    df["train_text"] = ("<NP:" + df[class_col] + ">") + df[smiles_col].astype(str)

    # create Dataset from df using only the text column containing formatted examples (i.e. "<np class>+SMILES") 
    ds = Dataset.from_pandas(df[["train_text"]], preserve_index=False)

    def _tokenize_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        tokenize dataset in batches
        """
        return tokenizer(
            batch["train_text"],
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
            padding=False, # pad using data collator
            return_attention_mask=True,
        )

    # tokenize dataset in batches and remove original text column for memory efficiency
    ds = ds.map(_tokenize_batch, batched=True, remove_columns=["train_text"])

    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Training yaml configuration file")
    ap.add_argument("--train_csv", required=True, help="Path to training split CSV")
    ap.add_argument("--test_csv", required=True, help="Path to test split CSV")
    ap.add_argument("--val_csv", required=True, help="Path to validation split CSV")
    args = ap.parse_args()

    configs = parse_yaml(args.yaml)

    # load transformer model for fine-tuning
    model = AutoModelForCausalLM.from_pretrained(configs['base']['model'], trust_remote_code=True)

    # load tokenizer associated with transformer model
    tokenizer = AutoTokenizer.from_pretrained(configs['base']['tokenizer'], trust_remote_code=True)

    special_tokens_dict = build_special_class_tokens(args.train_csv, configs['base']['class_col'])
    model, tokenizer = update_tokenizer_with_special_tokens(model, tokenizer, special_tokens_dict)

    # load csvs into pandas df
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    val_df = pd.read_csv(args.val_csv)

    # create tokenized training dataset
    ds_train = dataframe_to_tokenized_dataset(
        train_df,
        tokenizer=tokenizer,
        smiles_col=configs["base"]["smiles_col"],
        class_col=configs["base"]["class_col"],
        max_len=configs["base"]["max_token_length"],
    )

    # create tokenized test dataset
    ds_test = dataframe_to_tokenized_dataset(
        test_df,
        tokenizer=tokenizer,
        smiles_col=configs["base"]["smiles_col"],
        class_col=configs["base"]["class_col"],
        max_len=configs["base"]["max_token_length"],
    )

    # create tokenized validation dataset
    ds_val = dataframe_to_tokenized_dataset(
        val_df,
        tokenizer=tokenizer,
        smiles_col=configs["base"]["smiles_col"],
        class_col=configs["base"]["class_col"],
        max_len=configs["base"]["max_token_length"],
    )

    # ensure tokenizer has padding token for data collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # data collator takes variable-length tokenized examples and pads them to the same length to produce rectangular tensor 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # set mlm (masked language model) = False for causal LM

    # define arguments for training
    training_args = TrainingArguments(
        learning_rate=configs["training"]["learning_rate"],
        num_train_epochs=configs["training"]["num_train_epochs"],
        weight_decay=configs["training"]["weight_decay"],
        per_device_train_batch_size=configs["training"]["per_device_train_batch_size"],
        save_strategy=configs["training"]["save_strategy"],
        load_best_model_at_end=configs["training"]["load_best_model_at_end"],
        output_dir=configs["training"]["output_dir"]
        fp16=True, # enable mixed precision for faster training
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        data_collator=data_collator,
    )

    trainer.train() # starts training

    metrics = trainer.evaluate()

    trainer.save_metrics("eval", metrics) # save metrics in json file
    trainer.save_model() # save model so you can reload it using "from_pretrained()"

if __name__ == "__main__":
    main()