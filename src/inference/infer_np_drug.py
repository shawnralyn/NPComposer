"""Generate drug-like candidates from natural product SMILES using a trained checkpoint.

Loads a fine-tuned GP-MolFormer checkpoint and generates k drug candidates for each
input NP SMILES. No ground truth required — pure inference.

Usage:
    # Single SMILES from command line:
    python src/evaluation/infer_np_drug.py \\
        --checkpoint models/np_drug_lora_r16/final \\
        --smiles "CC1=CC2=C(C=C1)C(=O)C3=CC=CC=C3C2=O" \\
        --lora

    # Batch from CSV:
    python src/evaluation/infer_np_drug.py \\
        --checkpoint models/np_drug_lora_r16/final \\
        --input-csv  data/my_nps.csv \\
        --smiles-col canonical_smiles \\
        --output-csv results/candidates.csv \\
        --lora --k 25
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed as rdkit_qed
from rdkit.Chem.rdMolDescriptors import CalcTPSA
from rdkit.Chem import Lipinski
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

MODEL_ID     = "ibm-research/GP-MoLFormer-Uniq"
TOKENIZER_ID = "ibm-research/MoLFormer-XL-both-10pct"
MAX_LENGTH   = 202


# ---------------------------------------------------------------------------
# Frozen embedding (prompt tuning only)
# ---------------------------------------------------------------------------

class FrozenEmbeddingMinusUnk(torch.nn.Module):
    def __init__(self, word_embeddings, unk_token_id):
        super().__init__()
        w = word_embeddings.weight
        self.frozen1 = torch.nn.Parameter(w[:unk_token_id].clone(),      requires_grad=False)
        self.unk     = torch.nn.Parameter(w[[unk_token_id]].clone(),      requires_grad=True)
        self.frozen2 = torch.nn.Parameter(w[unk_token_id + 1:].clone(),  requires_grad=False)

    def forward(self, x):
        return F.embedding(x, torch.cat([self.frozen1, self.unk, self.frozen2]))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(checkpoint_dir, num_virtual_tokens, device, lora=False, lora_rank=16):
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
        model.word_embeddings = FrozenEmbeddingMinusUnk(
            model.word_embeddings, tokenizer.unk_token_id
        )

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
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    print(f"  Model on {device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(model, tokenizer, np_smiles_list, k, batch_size, device):
    """Generate k drug candidates for each NP SMILES. Returns list of lists."""
    all_candidates = []
    n_batches = (len(np_smiles_list) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(np_smiles_list), batch_size), total=n_batches, desc="Generating"):
        batch = np_smiles_list[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Replace last SEP with UNK to form generation prefix
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

        decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
        batch_candidates = []
        for idx in range(len(batch)):
            cands = []
            for seq in decoded[idx * k : (idx + 1) * k]:
                try:
                    smi = seq.split(tokenizer.unk_token)[1].split(tokenizer.sep_token)[0].strip()
                except IndexError:
                    smi = ""
                mol = Chem.MolFromSmiles(smi)
                cands.append(Chem.MolToSmiles(mol) if mol else None)
            batch_candidates.append(cands)
        all_candidates.extend(batch_candidates)

    return all_candidates


# ---------------------------------------------------------------------------
# Property computation
# ---------------------------------------------------------------------------

def mol_props(smi):
    if not smi:
        return {}
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    return {
        "qed":   round(rdkit_qed(mol), 3),
        "mw":    round(Chem.Descriptors.MolWt(mol), 1),
        "logp":  round(Chem.Descriptors.MolLogP(mol), 2),
        "tpsa":  round(CalcTPSA(mol), 1),
        "hbd":   Lipinski.NumHDonors(mol),
        "hba":   Lipinski.NumHAcceptors(mol),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input — either a single SMILES or a CSV
    inp = ap.add_mutually_exclusive_group(required=True)
    inp.add_argument("--smiles",     type=str, help="Single NP SMILES string")
    inp.add_argument("--input-csv",  type=str, help="CSV file with NP SMILES")
    ap.add_argument("--smiles-col",  default="canonical_smiles",
                    help="Column name for NP SMILES in input CSV (default: canonical_smiles)")

    # Model
    ap.add_argument("--checkpoint",         required=True,
                    help="Path to trained checkpoint directory")
    ap.add_argument("--lora",               action="store_true",
                    help="Load a LoRA checkpoint")
    ap.add_argument("--lora-rank",          type=int, default=16)
    ap.add_argument("--num-virtual-tokens", type=int, default=20)

    # Generation
    ap.add_argument("--k",          type=int, default=10,
                    help="Candidates to generate per NP (default: 10)")
    ap.add_argument("--batch-size", type=int, default=32)

    # Output
    ap.add_argument("--output-csv", default=None,
                    help="Save results to CSV (optional)")

    args = ap.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model, tokenizer = load_model(
        args.checkpoint, args.num_virtual_tokens, device,
        lora=args.lora, lora_rank=args.lora_rank,
    )

    # Build input list
    if args.smiles:
        np_smiles = [args.smiles]
    else:
        df_in = pd.read_csv(args.input_csv)
        np_smiles = df_in[args.smiles_col].dropna().tolist()
    print(f"\n{len(np_smiles)} NP(s) to process, generating {args.k} candidates each...")

    candidates = generate(model, tokenizer, np_smiles, args.k, args.batch_size, device)

    # Build results
    rows = []
    for smi, cands in zip(np_smiles, candidates):
        valid = [c for c in cands if c]
        for c in cands:
            props = mol_props(c) if c else {}
            rows.append({"np_smiles": smi, "candidate": c, "valid": c is not None, **props})

        print(f"\nNP: {smi}")
        print(f"  Valid: {len(valid)}/{args.k}")
        for c in valid:
            p = mol_props(c)
            print(f"  {c}  QED={p.get('qed','?')}  MW={p.get('mw','?')}  LogP={p.get('logp','?')}")

    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nSaved to {args.output_csv}")


if __name__ == "__main__":
    main()
