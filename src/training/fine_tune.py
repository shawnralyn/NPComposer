import torch
import tqdm
import evaluator 
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    wandb
)

# add to requirements.txt

# think about beam search, DPO

# load COCONUT dataset train/test/val

# load GP-MolFormer-Uniq model and its tokenizer

# create special class tokens based on NP pathway class

# run tokenizer on all examples, pad, and apply attention mask

# data collator?

# create trainer arguments, instantiate trainer, and run

# use evaluator/weights and biases library to plot learning curves (loss/accuracy)



# 1) load GP-MoLFormer-Uniq model and its associated tokenizer
# wrap in try/except 
model = AutoModelForCausalLM.from_pretrained("ibm-research/GP-MoLFormer-Uniq", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("ibm-research/MoLFormer-XL-both-10pct", trust_remote_code=True)

# 2) load COCONUT dataset
train_data = 
test_data = 
val_data = 

# create condtional "class tokens" vocabulary 
# register those class toekns with the tokenizer

tokens = {}

tokenizer.add_special_tokens({"additional_special_tokens:": tokens})
model.resize_token_embeddings(len(tokenizer))

# construct training text for each example (prepend class token to SMILES)

# 3) tokenize dataset and add special class tokens

class_tokens = [
    "<CLS_TERPENOID>",
    "<CLS_ALKALOID>",
    "<CLS_POLYKETIDE>",
]

tok.add_special_tokens({"additional_special_tokens": class_tokens})

model.resize_token_embeddings(len(tok))

MAX_LEN = 256

SMILES_COL = "smiles"
CLASS_COL  = "np_class"   # change to your column name

def make_class_token(label: str) -> str:
    # Normalize to avoid spaces/special chars in token text
    safe = label.strip().lower().replace(" ", "_")
    return f"<CLS_{safe}>"

# Collect label vocabulary from the dataset (train split)
labels = sorted(set(train_ds[CLASS_COL]))

class_tokens = [make_class_token(l) for l in labels]

# Register as special tokens so they stay intact (not split)
specials = {"additional_special_tokens": class_tokens}
num_added = tokenizer.add_special_tokens(specials)

# Make sure model embedding matrix matches tokenizer size
if num_added > 0:
    model.resize_token_embeddings(len(tokenizer))

def tokenize_batch(batch):
    smiles_list = batch[SMILES_COL]
    class_list  = batch[CLASS_COL]

    texts = []
    for smi, cls in zip(smiles_list, class_list):
        cls_tok = make_class_token(cls)
        txt = f"{cls_tok} {smi}"
        if tokenizer.eos_token is not None:
            txt = txt + tokenizer.eos_token
        texts.append(txt)

    return tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LEN,
        # don't pad here if you’re using a collator for dynamic padding
    )


# 4) Pad and apply attention mask


# 5) Create trainer

# define training arguments (hyperparameters)
training_args = TrainingArguments(
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    num_train_epochs=10,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp=16 # used mixed precision for faster training
)

# instantiate trainer
trainer = Trainer(
    model=model,
    args=trainings_args,
    train_dataset=train_set,
    eval_dataset=eval_set,
    data_collator=collate_for_clm,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model()


# use evaluator and checkpointing to track training performance