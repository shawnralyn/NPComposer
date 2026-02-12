import pandas as pd
from datasets import Dataset
import tqdm
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments,
    wandb, Evaluator
)

# drop molecules longer than 200 tokens 

def build_special_class_tokens(ds_train, class_col: str) -> List[str]:
    """
    """
    classes = sorted(set(ds_train[class_col]))
    tokens = [f"<NP:{c}>" for c in classes]
    return tokens

def tokenize_

def main():
    model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)