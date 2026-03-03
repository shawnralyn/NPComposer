import argparse
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_yaml(yml):
    """
    Reads a YAML configuration file and returns the parsed configuration as a dictionary.

    Args:
        yml (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary, or None if file not found or parsing fails.
    """
    try:
        with open(yml, "r") as file:
            configs = yaml.safe_load(file)
            return configs
    except FileNotFoundError:
        print(f"Error: The file {yml} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")


def main():
    """
    Main function for running inference with a trained language model.

    - Parses command-line arguments for the YAML configuration file.
    - Loads model and tokenizer from checkpoint specified in the config.
    - Generates a specified number of molecular SMILES strings conditioned on a class.
    - Writes generated SMILES strings to an output file.

    CLI overrides (take precedence over YAML):
        --seed          Random seed for reproducibility
        --np_class      NP class conditioning token, e.g. "<NP:Fatty acids>"
        --output        Output file path
        --num_molecules Number of molecules to generate
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Inference yaml configuration file")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (overrides yaml)")
    ap.add_argument("--np_class", type=str, default=None, help="NP class token (overrides yaml)")
    ap.add_argument("--output", type=str, default=None, help="Output file path (overrides yaml)")
    ap.add_argument("--num_molecules", type=int, default=None, help="Number of molecules (overrides yaml)")
    args = ap.parse_args()

    configs = parse_yaml(args.yaml)

    # CLI overrides
    if args.np_class is not None:
        configs["inference"]["np_class"] = args.np_class
    if args.output is not None:
        configs["inference"]["output_file"] = args.output
    if args.num_molecules is not None:
        configs["inference"]["num_molecules"] = args.num_molecules

    # Set random seed for reproducibility
    seed = args.seed if args.seed is not None else configs["inference"].get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    ckpt = configs["inference"]["ckpt_path"]  # load checkpoint path

    # load tokenizer and model from checkpoint path
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()

    # load output file path and make sure directory exists
    out_file = Path(configs["inference"]["output_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)

    out_file.write_text("")  # overwrite contents of file

    np_class = configs["inference"]["np_class"]
    num_molecules = int(configs["inference"]["num_molecules"])
    print(f"Generating {num_molecules} molecules for class: {np_class}")

    # create starting sequence
    x = tok(np_class, return_tensors="pt", add_special_tokens=False)

    # Prefix length: skip the conditioning token in decoded output
    prompt_len = x["input_ids"].shape[1]

    # generate n number of molecules and append each SMILES string to output file
    generated = 0
    with out_file.open("a") as f:
        for _ in range(num_molecules):
            y = model.generate(
                **x,
                max_new_tokens=200,  # set to max tokens used for training
                do_sample=True,
                top_p=configs["inference"]["top_p"],
                temperature=configs["inference"]["temperature"],
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

            # Decode only the generated tokens (skip conditioning prompt)
            generated_ids = y[0][prompt_len:]
            raw = tok.decode(generated_ids, skip_special_tokens=True).strip()
            # Take first molecule (split on '.')
            smiles = raw.split(".")[0].strip()

            if smiles:
                f.write(smiles + "\n")
                generated += 1

    print(f"Generated {generated}/{num_molecules} molecules -> {out_file}")


if __name__ == "__main__":
    main()
