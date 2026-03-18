"""Fine-tune GP-MolFormer for NP→drug generation using IBM's pair-tuning approach.

Follows the PEFT prompt-tuning framework from:
  https://github.com/IBM/gp-molformer/blob/main/scripts/pairtune_training.py

Only ~16k parameters are trained:
  - 20 virtual prompt token embeddings (PEFT PromptTuning)
  - The <unk> token embedding (acts as source→target separator)
  - The full 47M-parameter backbone is frozen

Training sequence format:
  [CLS] <NP_SMILES> [UNK] <drug_SMILES> [SEP]
  Loss computed only on drug SMILES tokens.

Evaluation (run every --eval-epochs epochs):
  Generate k candidate molecules per validation source SMILES and report
  validity, uniqueness, mean QED, and mean Tanimoto similarity to source NP.

Usage:
    python src/training/train_np_drug_pairs.py \\
        --pairs-csv  data/processed/ChEMBL_pairs.csv \\
        --output-dir models/np_drug_pairtune

    # Faster local experiment:
    python src/training/train_np_drug_pairs.py \\
        --pairs-csv  data/processed/ChEMBL_pairs.csv \\
        --output-dir models/np_drug_pairtune \\
        --k 5 --batch-size 32 --num-epochs 10 --max-pairs 20000
"""
import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Descriptors import qed as rdkit_qed
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

RDLogger.DisableLog("rdApp.*")

MODEL_ID     = "ibm-research/GP-MoLFormer-Uniq"
TOKENIZER_ID = "ibm-research/MoLFormer-XL-both-10pct"
MAX_LENGTH   = 202  # GP-MolFormer max position embeddings


# ---------------------------------------------------------------------------
# Frozen embedding — keeps only <unk> trainable (from IBM)
# ---------------------------------------------------------------------------

class FrozenEmbeddingMinusUnk(torch.nn.Module):
    """Freeze all token embeddings except <unk>, which acts as the separator."""
    def __init__(self, word_embeddings, unk_token_id):
        super().__init__()
        w = word_embeddings.weight
        self.frozen1 = torch.nn.Parameter(w[:unk_token_id].clone(),          requires_grad=False)
        self.unk     = torch.nn.Parameter(w[[unk_token_id]].clone(),          requires_grad=True)
        self.frozen2 = torch.nn.Parameter(w[unk_token_id + 1:].clone(),       requires_grad=False)

    def forward(self, x):
        return F.embedding(x, torch.cat([self.frozen1, self.unk, self.frozen2]))


# ---------------------------------------------------------------------------
# Data collator — replaces first SEP with UNK, masks source from loss
# ---------------------------------------------------------------------------

class DataCollatorForPairTuning(DataCollatorForLanguageModeling):
    """Adapted from IBM's pairtune_training.py.

    For each sequence [CLS][source][SEP][target][SEP]:
      - Replaces the first SEP (source/target boundary) with UNK
      - Sets labels = input_ids only for target tokens (token_type_ids == 1)
    """
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.sep_id = tokenizer.sep_token_id
        self.unk_id = tokenizer.unk_token_id

    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)

        # Find the first SEP in each sequence (boundary between source and target)
        first_sep = [
            torch.nonzero(ids == self.sep_id)[0]
            for ids in batch["input_ids"]
        ]
        row_idx = torch.arange(len(first_sep))
        col_idx = torch.cat(first_sep)

        # Replace first SEP with UNK (the learned separator)
        batch["input_ids"][row_idx, col_idx] = self.unk_id
        # Mark UNK position as part of the target (token_type_id = 1)
        batch["token_type_ids"][row_idx, col_idx] = 1

        # Loss only on target tokens (where token_type_ids == 1)
        batch["labels"] = batch["input_ids"].where(
            batch["token_type_ids"].bool(), torch.tensor(-100)
        )
        del batch["token_type_ids"]
        return batch


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def load_and_split(
    csv_path: str,
    np_col: str,
    drug_col: str,
    val_frac: float,
    test_frac: float,
    max_pairs: int | None,
    no_chirality: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pairs CSV, optionally subsample, validate SMILES, and split."""
    print(f"Loading pairs from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=[np_col, drug_col], low_memory=False)
    df = df.dropna(subset=[np_col, drug_col])
    print(f"  {len(df):,} pairs loaded")

    # Drop rows with unparseable SMILES
    def is_valid(s):
        return Chem.MolFromSmiles(str(s)) is not None

    mask = df[np_col].apply(is_valid) & df[drug_col].apply(is_valid)
    df = df[mask].reset_index(drop=True)
    print(f"  {len(df):,} pairs with valid SMILES")

    if no_chirality:
        def strip_chiral(s):
            mol = Chem.MolFromSmiles(s)
            return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else s
        df[np_col]   = df[np_col].apply(strip_chiral)
        df[drug_col] = df[drug_col].apply(strip_chiral)

    if max_pairs and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} pairs (--max-pairs)")

    train_df, temp_df = train_test_split(df, test_size=val_frac + test_frac, random_state=42)
    val_df, test_df   = train_test_split(
        temp_df, test_size=test_frac / (val_frac + test_frac), random_state=42
    )
    print(f"  Train: {len(train_df):,}  Val: {len(val_df):,}  Test: {len(test_df):,}")

    # Save test split so it can be used for post-training evaluation
    return train_df, val_df, test_df


def make_dataset(
    df: pd.DataFrame,
    tokenizer,
    np_col: str,
    drug_col: str,
    as_pairs: bool,
) -> Dataset:
    """Tokenize source+target pairs (training) or source-only (eval/generation)."""
    def _tokenize(batch):
        if as_pairs:
            return tokenizer(
                batch[np_col],
                batch[drug_col],
                return_token_type_ids=True,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )
        else:
            return tokenizer(
                batch[np_col],
                return_token_type_ids=True,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=False,
            )

    cols = [np_col, drug_col] if as_pairs else [np_col]
    ds = Dataset.from_pandas(df[cols].reset_index(drop=True))
    ds = ds.map(_tokenize, batched=True, remove_columns=ds.column_names)
    return ds


# ---------------------------------------------------------------------------
# Custom Trainer — generates k molecules per val input
# ---------------------------------------------------------------------------

class PairTuneTrainer(Trainer):
    def __init__(self, *args, k: int = 10, **kwargs):
        self.k = k
        super().__init__(*args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        # After collator, sequences look like: [CLS][source][UNK]
        # (val dataset has source-only inputs; collator replaced SEP with UNK)
        max_new_tokens = MAX_LENGTH - inputs["input_ids"].shape[1]
        max_new_tokens = max(max_new_tokens, 50)

        with torch.no_grad():
            try:
                generated = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    top_k=None,
                    num_return_sequences=self.k,
                    max_new_tokens=max_new_tokens,
                )
            except RuntimeError as e:
                print(f"  Generation error: {e}")
                generated = inputs["input_ids"].repeat_interleave(self.k, dim=0)

        # Reshape (batch*k, seq) → (batch, seq, k) so Trainer pads correctly
        batch_size = inputs["input_ids"].shape[0]
        generated = generated.reshape(batch_size, self.k, -1).transpose(1, 2)
        return (None, generated, generated)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_metrics(p, tokenizer, k: int):
    """Compute validity, uniqueness, mean QED, and mean Tanimoto to source."""
    # Undo Trainer's (batch, seq, k) → (batch*k, seq)
    pred_tok = p.predictions.transpose((0, 2, 1)).reshape(-1, p.predictions.shape[1])
    pred_tok = np.where(pred_tok == -100, tokenizer.pad_token_id, pred_tok)

    strings = tokenizer.batch_decode(pred_tok)

    src_smiles, gen_smiles = [], []
    for s in strings:
        try:
            src = s.split(tokenizer.unk_token)[0].split(tokenizer.cls_token)[1]
        except IndexError:
            src = ""
        try:
            gen = s.split(tokenizer.unk_token)[1].split(tokenizer.sep_token)[0]
        except IndexError:
            gen = ""
        src_smiles.append(src.strip())
        gen_smiles.append(gen.strip())

    src_mols = [Chem.MolFromSmiles(s) for s in src_smiles[::k]]  # one per original input
    gen_mols = [Chem.MolFromSmiles(s) for s in gen_smiles]

    valid_mols  = [m for m in gen_mols if m is not None]
    validity    = len(valid_mols) / len(gen_mols) if gen_mols else 0.0
    unique_smi  = {Chem.MolToSmiles(m, isomericSmiles=False) for m in valid_mols}
    uniqueness  = len(unique_smi) / len(valid_mols) if valid_mols else 0.0
    mean_qed    = float(np.mean([rdkit_qed(m) for m in valid_mols])) if valid_mols else 0.0

    # Tanimoto: generated vs source NP (one source per k generated molecules)
    tanimotos = []
    for i, src_mol in enumerate(src_mols):
        if src_mol is None:
            continue
        src_fp = AllChem.GetMorganFingerprintAsBitVect(src_mol, 2, 2048)
        batch_gen = gen_mols[i * k : (i + 1) * k]
        for gm in batch_gen:
            if gm is not None:
                gfp = AllChem.GetMorganFingerprintAsBitVect(gm, 2, 2048)
                tanimotos.append(DataStructs.TanimotoSimilarity(src_fp, gfp))

    mean_tanimoto_to_src = float(np.mean(tanimotos)) if tanimotos else 0.0

    return {
        "validity":            validity,
        "uniqueness":          uniqueness,
        "mean_qed":            mean_qed,
        "tanimoto_to_source":  mean_tanimoto_to_src,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Data
    ap.add_argument("--pairs-csv",      required=True, help="Pairs CSV from generate_ChEMBL_pairs.py")
    ap.add_argument("--output-dir",     required=True, help="Directory for checkpoints and outputs")
    ap.add_argument("--np-smiles-col",  default="coconut_smiles")
    ap.add_argument("--drug-smiles-col",default="chembl_smiles")
    ap.add_argument("--val-frac",       type=float, default=0.1)
    ap.add_argument("--test-frac",      type=float, default=0.1)
    ap.add_argument("--max-pairs",      type=int,   default=None,
                    help="Subsample to N pairs for fast local experiments")
    ap.add_argument("--no-chirality",   action="store_true",
                    help="Strip chirality from SMILES (IBM default; off by default here)")
    # Training
    ap.add_argument("--num-epochs",     type=float, default=100)
    ap.add_argument("--batch-size",     type=int,   default=512,
                    help="Total batch size across all GPUs (default: 512)")
    ap.add_argument("--lr",             type=float, default=3e-2)
    ap.add_argument("--eval-epochs",    type=int,   default=1,
                    help="Epochs between evaluations (default: 1)")
    ap.add_argument("--no-eval",        action="store_true",
                    help="Disable evaluation during training (faster)")
    ap.add_argument("--save-epochs",    type=int,   default=10,
                    help="Epochs between checkpoint saves (default: 10)")
    ap.add_argument("--num-virtual-tokens", type=int, default=20,
                    help="Number of PEFT prompt virtual tokens (default: 20)")
    ap.add_argument("--lora",           action="store_true",
                    help="Use LoRA instead of prompt tuning (~500k trainable params vs 16k)")
    ap.add_argument("--lora-rank",      type=int, default=16,
                    help="LoRA rank (default: 16; try 8 or 32)")
    ap.add_argument("--bf16",           action="store_true",
                    help="Use bfloat16 (recommended for Ampere+ GPUs on RunPod)")
    ap.add_argument("--seed",           type=int,   default=None)
    ap.add_argument("--resume-from-checkpoint", type=str, default=None,
                    help="Path to checkpoint to resume from, or 'latest' to auto-detect")
    # Evaluation
    ap.add_argument("--k",              type=int,   default=10,
                    help="Molecules generated per val input during training (default: 10)")
    ap.add_argument("--wandb-project",  default=None,
                    help="W&B project name (omit to disable W&B logging)")
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- tokenizer ----
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, padding_side="left", trust_remote_code=True
    )

    # ---- data ----
    train_df, val_df, test_df = load_and_split(
        args.pairs_csv,
        np_col=args.np_smiles_col,
        drug_col=args.drug_smiles_col,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        max_pairs=args.max_pairs,
        no_chirality=args.no_chirality,
    )

    # Save test split for post-training evaluation
    test_path = os.path.join(args.output_dir, "test_pairs.csv")
    test_df.to_csv(test_path, index=False)
    print(f"  Test split saved to {test_path}")

    print("Tokenizing datasets...")
    ds_train = make_dataset(train_df, tokenizer, args.np_smiles_col, args.drug_smiles_col, as_pairs=True)
    ds_val   = make_dataset(val_df,   tokenizer, args.np_smiles_col, args.drug_smiles_col, as_pairs=False)
    print(f"  Train: {len(ds_train):,} examples  Val: {len(ds_val):,} examples")

    # ---- model ----
    print(f"Loading model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        hidden_dropout_prob=0.0,
        embedding_dropout_prob=0.0,
        trust_remote_code=True,
    )

    if args.lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = get_peft_config({
            "peft_type":         "PROMPT_TUNING",
            "task_type":         "CAUSAL_LM",
            "num_virtual_tokens": args.num_virtual_tokens,
        })
        model = get_peft_model(model, peft_config)
        # Make <unk> the only trainable token embedding (acts as source→target separator)
        model.word_embeddings = FrozenEmbeddingMinusUnk(
            model.word_embeddings, tokenizer.unk_token_id
        )
    model.print_trainable_parameters()

    # ---- training args ----
    n_gpus = max(torch.cuda.device_count(), 1)
    steps_per_epoch = int(np.ceil(len(ds_train) / args.batch_size))

    report_to = ["wandb"] if args.wandb_project else ["none"]
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size // n_gpus,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        lr_scheduler_type="constant",
        evaluation_strategy="no" if args.no_eval else "steps",
        eval_steps=args.eval_epochs * steps_per_epoch,
        save_strategy="steps",
        save_steps=args.save_epochs * steps_per_epoch,
        save_total_limit=3,
        remove_unused_columns=False,
        bf16=args.bf16,
        report_to=report_to,
        dataloader_num_workers=4,
    )

    collator = DataCollatorForPairTuning(tokenizer)

    trainer = PairTuneTrainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=None if args.no_eval else ds_val,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=partial(compute_metrics, tokenizer=tokenizer, k=args.k),
        k=args.k,
    )

    # Resolve checkpoint to resume from
    resume = args.resume_from_checkpoint
    if resume == "latest":
        import glob as _glob
        ckpts = sorted(_glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
                       key=lambda p: int(p.split("-")[-1]))
        resume = ckpts[-1] if ckpts else None
        if resume:
            print(f"  Resuming from {resume}")

    print("\nStarting training...")
    print(f"  {len(ds_train):,} train pairs  |  {steps_per_epoch} steps/epoch  |  {args.num_epochs} epochs")
    trainer.train(resume_from_checkpoint=resume)

    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))
    print(f"\nDone. Model saved to {args.output_dir}/final")
    print(f"Test pairs saved to {test_path} — run eval script to compute Tanimoto-to-target.")


if __name__ == "__main__":
    main()
