"""Wrapper script to run NPGPT inference with explicit seed control.

NPGPT's inference.py doesn't accept a --seed flag, so this wrapper
sets torch/numpy/python seeds before calling generate_smiles().

Usage:
    python scripts/npgpt_generate.py \
        --npgpt_dir external/npgpt \
        --tokenizer external/npgpt/externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json \
        --checkpoint external/npgpt/checkpoints/smiles-gpt/model.ckpt \
        --num_samples 760 \
        --seed 1 \
        --output outputs/NPGPT/evaluation/npgpt_seed1.smi
"""

import argparse
import os
import sys
import random


def main():
    parser = argparse.ArgumentParser(description="NPGPT generation with seed control")
    parser.add_argument("--npgpt_dir", type=str, default="external/npgpt",
                        help="Path to npgpt root directory")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Path to tokenizer.json")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model.ckpt")
    parser.add_argument("--num_samples", type=int, default=760)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--initial_smiles", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override generation temperature")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Override generation top_p")
    args = parser.parse_args()

    # ---- Set all random seeds BEFORE importing torch ----
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)

    import numpy as np
    np.random.seed(args.seed)

    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- Add npgpt to sys.path ----
    npgpt_src = os.path.join(args.npgpt_dir, "src")
    sys.path.insert(0, npgpt_src)

    # Also need smiles-gpt for tokenizer
    smiles_gpt_path = os.path.join(args.npgpt_dir, "externals", "smiles-gpt")
    sys.path.insert(0, smiles_gpt_path)

    import math
    from tqdm import tqdm
    from transformers import PreTrainedTokenizerFast

    from npgpt import SmilesGptModel, SmilesGptTrainingConfig
    from npgpt.config import SmilesGptGenerationConfig
    from npgpt.tokenizer import get_tokenizer

    # ---- generate_smiles (from npgpt/src/scripts/inference.py) ----
    def generate_smiles(
        model: SmilesGptModel,
        tokenizer: PreTrainedTokenizerFast,
        config: SmilesGptGenerationConfig,
        initial_smiles: str | None = None,
        batch_size: int = 1000,
    ) -> list[str]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        generated_smiles = []
        with torch.no_grad():
            num_batches = math.ceil(config.num_samples / batch_size)
            for batch_idx in tqdm(range(num_batches), desc="Generating SMILES"):
                current_batch_size = min(
                    batch_size, config.num_samples - batch_idx * batch_size
                )
                if initial_smiles is not None:
                    initial_tokens = tokenizer.encode(
                        initial_smiles, add_special_tokens=False
                    )
                    input_ids = torch.tensor(
                        [[tokenizer.bos_token_id] + initial_tokens] * current_batch_size
                    ).to(device)
                else:
                    input_ids = torch.tensor(
                        [[tokenizer.bos_token_id]] * current_batch_size
                    ).to(device)

                outputs = model.model.generate(
                    input_ids,
                    max_length=config.max_length,
                    do_sample=config.do_sample,
                    top_p=config.top_p,
                    temperature=config.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                smiles_list = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                generated_smiles.extend(smiles_list)

        return generated_smiles

    # ---- Load model ----
    training_config = SmilesGptTrainingConfig()
    generation_config = SmilesGptGenerationConfig(
        num_samples=args.num_samples,
    )

    # Override temperature/top_p if specified
    if args.temperature is not None:
        generation_config.temperature = args.temperature
    if args.top_p is not None:
        generation_config.top_p = args.top_p

    tokenizer = get_tokenizer(training_config, args.tokenizer)
    model = SmilesGptModel.load_from_checkpoint(
        args.checkpoint,
        config=training_config,
        tokenizer=tokenizer,
        strict=False,
    )

    smiles_list = generate_smiles(
        model,
        tokenizer,
        generation_config,
        initial_smiles=args.initial_smiles,
        batch_size=args.batch_size,
    )

    # ---- Save output ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(smiles_list))

    print(f"Seed {args.seed}: {len(smiles_list)} SMILES saved to {args.output}")


if __name__ == "__main__":
    main()
