"""Evaluate a fine-tuned GP-MolFormer NP→drug pair-tuning checkpoint.

Loads a trained checkpoint from train_np_drug_pairs.py, runs generation on
the held-out test pairs, and reports:
  - Validity rate (% generated SMILES that parse)
  - Uniqueness (unique valid / total valid)
  - Mean QED of generated molecules
  - Mean Tanimoto similarity to target ChEMBL molecule
  - Recovery rate at Tanimoto thresholds 0.3, 0.4, 0.5, 0.6
    (% of test NPs where best candidate >= threshold)

Usage:
    python src/training/eval_np_drug_pairs.py \\
        --checkpoint  models/np_drug_pairtune/final \\
        --test-csv    models/np_drug_pairtune/test_pairs.csv

    # More candidates per molecule for better recall:
    python src/training/eval_np_drug_pairs.py \\
        --checkpoint  models/np_drug_pairtune/final \\
        --test-csv    models/np_drug_pairtune/test_pairs.csv \\
        --k 25
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Descriptors import qed as rdkit_qed
from transformers import AutoModelForCausalLM, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

MODEL_ID     = "ibm-research/GP-MoLFormer-Uniq"
TOKENIZER_ID = "ibm-research/MoLFormer-XL-both-10pct"
MAX_LENGTH   = 202


# ---------------------------------------------------------------------------
# Frozen embedding — must match training setup exactly
# ---------------------------------------------------------------------------

class FrozenEmbeddingMinusUnk(torch.nn.Module):
    def __init__(self, word_embeddings, unk_token_id):
        super().__init__()
        w = word_embeddings.weight
        self.frozen1 = torch.nn.Parameter(w[:unk_token_id].clone(),      requires_grad=False)
        self.unk     = torch.nn.Parameter(w[[unk_token_id]].clone(),      requires_grad=True)
        self.frozen2 = torch.nn.Parameter(w[unk_token_id + 1:].clone(),   requires_grad=False)

    def forward(self, x):
        return F.embedding(x, torch.cat([self.frozen1, self.unk, self.frozen2]))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_dir: str,
    num_virtual_tokens: int,
    device: torch.device,
    lora: bool = False,
    lora_rank: int = 16,
):
    """Reconstruct the model architecture and load trained weights."""
    print(f"Loading base model {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        hidden_dropout_prob=0.0,
        embedding_dropout_prob=0.0,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, padding_side="left", trust_remote_code=True
    )

    if lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_rank * 2,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, peft_config)
    else:
        peft_config = get_peft_config({
            "peft_type":          "PROMPT_TUNING",
            "task_type":          "CAUSAL_LM",
            "num_virtual_tokens": num_virtual_tokens,
        })
        model = get_peft_model(model, peft_config)
        # Reconstruct FrozenEmbeddingMinusUnk before loading weights
        model.word_embeddings = FrozenEmbeddingMinusUnk(
            model.word_embeddings, tokenizer.unk_token_id
        )

    # Load trained adapter weights
    print(f"Loading checkpoint from {checkpoint_dir}...")
    safetensors_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    bin_path         = os.path.join(checkpoint_dir, "adapter_model.bin")
    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_path)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location="cpu")
    else:
        raise FileNotFoundError(
            f"No adapter weights found in {checkpoint_dir}. "
            "Expected adapter_model.safetensors or adapter_model.bin."
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"  Warning: unexpected keys in checkpoint: {unexpected[:5]}")

    model.eval()
    model.to(device)
    print(f"  Model on {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_candidates(
    model,
    tokenizer,
    np_smiles_list: list[str],
    k: int,
    batch_size: int,
    device: torch.device,
) -> list[list[str]]:
    """Generate k candidate drug SMILES for each NP SMILES.

    Returns a list of lists: one inner list of k SMILES per input NP.
    """
    all_candidates = []
    n_batches = (len(np_smiles_list) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(np_smiles_list), batch_size), total=n_batches, desc="Generating"):
        batch = np_smiles_list[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            return_token_type_ids=True,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Replace last SEP with UNK to form the generation prefix
        # (mirrors what DataCollatorForPairTuning does for single sequences)
        for j in range(len(input_ids)):
            sep_positions = (input_ids[j] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                input_ids[j, sep_positions[-1]] = tokenizer.unk_token_id

        max_new_tokens = max(MAX_LENGTH - input_ids.shape[1], 50)

        with torch.no_grad():
            try:
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=True,
                    top_k=None,
                    num_return_sequences=k,
                    max_new_tokens=max_new_tokens,
                )
            except RuntimeError as e:
                print(f"  Generation error on batch {i}: {e}")
                generated = input_ids.repeat_interleave(k, dim=0)

        # Parse generated SMILES: between UNK and SEP
        decoded = tokenizer.batch_decode(generated)
        for d in decoded:
            try:
                smi = d.split(tokenizer.unk_token)[1].split(tokenizer.sep_token)[0].strip()
            except IndexError:
                smi = ""
            all_candidates.append(smi)

    # Group into lists of k per input NP
    return [all_candidates[i * k : (i + 1) * k] for i in range(len(np_smiles_list))]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def morgan_fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)


def tanimoto(fp1, fp2) -> float:
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def evaluate(
    np_smiles_list: list[str],
    target_smiles_list: list[str],
    candidates_per_np: list[list[str]],
    thresholds: list[float],
) -> dict:
    """Compute evaluation metrics across all test pairs."""
    all_valid, all_qed = [], []
    best_tanimotos_to_target = []
    tanimotos_to_source = []

    for np_smi, tgt_smi, candidates in zip(np_smiles_list, target_smiles_list, candidates_per_np):
        np_mol  = Chem.MolFromSmiles(np_smi)
        tgt_mol = Chem.MolFromSmiles(tgt_smi)
        np_fp   = morgan_fp(np_mol)  if np_mol  else None
        tgt_fp  = morgan_fp(tgt_mol) if tgt_mol else None

        valid_mols = []
        for smi in candidates:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid_mols.append(mol)
                all_valid.append(smi)
                all_qed.append(rdkit_qed(mol))
                if np_fp:
                    tanimotos_to_source.append(tanimoto(np_fp, morgan_fp(mol)))

        # Best Tanimoto to target across k candidates
        if tgt_fp and valid_mols:
            sims = [tanimoto(tgt_fp, morgan_fp(m)) for m in valid_mols]
            best_tanimotos_to_target.append(max(sims))
        else:
            best_tanimotos_to_target.append(0.0)

    n_total   = len(np_smiles_list) * len(candidates_per_np[0])
    validity  = len(all_valid) / n_total if n_total else 0.0
    unique    = len({Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
                     for s in all_valid}) / len(all_valid) if all_valid else 0.0
    mean_qed  = float(np.mean(all_qed)) if all_qed else 0.0
    mean_tan_src = float(np.mean(tanimotos_to_source)) if tanimotos_to_source else 0.0
    mean_tan_tgt = float(np.mean(best_tanimotos_to_target))

    recovery = {
        t: float(np.mean([s >= t for s in best_tanimotos_to_target]))
        for t in thresholds
    }

    return {
        "n_test_pairs":          len(np_smiles_list),
        "validity":              validity,
        "uniqueness":            unique,
        "mean_qed":              mean_qed,
        "mean_tanimoto_to_src":  mean_tan_src,
        "mean_tanimoto_to_tgt":  mean_tan_tgt,
        "recovery":              recovery,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--checkpoint",       required=True,
                    help="Path to trained checkpoint directory (e.g. models/.../final)")
    ap.add_argument("--test-csv",         required=True,
                    help="Test pairs CSV (saved automatically by training script)")
    ap.add_argument("--np-smiles-col",    default="coconut_smiles",
                    help="Column name for NP SMILES in test CSV (default: coconut_smiles)")
    ap.add_argument("--drug-smiles-col",  default="chembl_smiles",
                    help="Column name for drug SMILES in test CSV (default: chembl_smiles)")
    ap.add_argument("--k",                type=int, default=10,
                    help="Candidates to generate per NP (default: 10)")
    ap.add_argument("--batch-size",       type=int, default=32,
                    help="Generation batch size (default: 32)")
    ap.add_argument("--num-virtual-tokens", type=int, default=20)
    ap.add_argument("--lora",             action="store_true",
                    help="Load a LoRA checkpoint instead of prompt tuning")
    ap.add_argument("--lora-rank",        type=int, default=16,
                    help="LoRA rank used during training (default: 16)")
    ap.add_argument("--thresholds",       type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6],
                    help="Tanimoto recovery thresholds (default: 0.3 0.4 0.5 0.6)")
    ap.add_argument("--output-csv",       default=None,
                    help="Save per-pair results to CSV (optional)")
    ap.add_argument("--max-test-pairs",   type=int, default=None,
                    help="Randomly subsample test set for faster eval (optional)")
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Load model
    model, tokenizer = load_model(
        args.checkpoint, args.num_virtual_tokens, device,
        lora=args.lora, lora_rank=args.lora_rank,
    )

    # Load test pairs
    print(f"\nLoading test pairs from {args.test_csv}...")
    df = pd.read_csv(args.test_csv)
    df = df.dropna(subset=[args.np_smiles_col, args.drug_smiles_col])
    if args.max_test_pairs and args.max_test_pairs < len(df):
        df = df.sample(args.max_test_pairs, random_state=42).reset_index(drop=True)
        print(f"  Subsampled to {len(df):,} test pairs")
    np_smiles  = df[args.np_smiles_col].tolist()
    tgt_smiles = df[args.drug_smiles_col].tolist()
    print(f"  {len(df):,} test pairs")

    # Generate
    print(f"\nGenerating {args.k} candidates per NP across {len(np_smiles):,} inputs...")
    candidates = generate_candidates(
        model, tokenizer, np_smiles, args.k, args.batch_size, device
    )

    # Evaluate
    print("\nEvaluating...")
    results = evaluate(np_smiles, tgt_smiles, candidates, args.thresholds)

    # Report
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Test pairs:              {results['n_test_pairs']:,}")
    print(f"Validity:                {results['validity']:.3f}")
    print(f"Uniqueness:              {results['uniqueness']:.3f}")
    print(f"Mean QED:                {results['mean_qed']:.3f}")
    print(f"Mean Tanimoto → source:  {results['mean_tanimoto_to_src']:.3f}")
    print(f"Mean Tanimoto → target:  {results['mean_tanimoto_to_tgt']:.3f}")
    print("\nRecovery rate (best of k >= threshold):")
    for t, rate in results["recovery"].items():
        print(f"  Tanimoto >= {t:.1f}:  {rate:.3f}  ({rate*results['n_test_pairs']:.0f}/{results['n_test_pairs']})")

    # Optionally save per-pair results
    if args.output_csv:
        rows = []
        for np_smi, tgt_smi, cands in zip(np_smiles, tgt_smiles, candidates):
            tgt_mol = Chem.MolFromSmiles(tgt_smi)
            tgt_fp  = morgan_fp(tgt_mol) if tgt_mol else None
            valid   = [c for c in cands if Chem.MolFromSmiles(c)]
            best_tan = 0.0
            best_smi = ""
            if tgt_fp and valid:
                sims = [(tanimoto(tgt_fp, morgan_fp(Chem.MolFromSmiles(c))), c) for c in valid]
                best_tan, best_smi = max(sims)
            rows.append({
                "np_smiles":    np_smi,
                "target_smiles": tgt_smi,
                "best_candidate": best_smi,
                "best_tanimoto_to_target": best_tan,
                "n_valid": len(valid),
            })
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nPer-pair results saved to {args.output_csv}")


if __name__ == "__main__":
    main()
