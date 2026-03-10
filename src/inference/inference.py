import argparse
import torch
import yaml
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_yaml(yml):
    """
    Reads a YAML configuration file and returns the parsed configuration as a dictionary.
    """
    try:
        with open(yml, "r") as file:
            configs = yaml.safe_load(file)
            return configs
    except FileNotFoundError:
        print(f"Error: The file {yml} was not found.")
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")


def build_prompt(pathway=None, superclass=None, is_glycoside=None,
                 aromatic_rings=None, qed_bin=None, sa_bin=None):
    """
    Build conditioning prompt from individual tokens.
    Matches the training format:
        <np_classifier_pathway:X> <np_classifier_superclass:X> <np_classifier_is_glycoside:X>
        <aromatic_rings_count:X> <qed_bin:X> <sa_bin:X>
    Any token can be omitted (set to None) to allow unconditional generation
    for that attribute (matches the conditioning dropout used during training).
    """
    parts = []
    if pathway is not None:
        parts.append(f"<np_classifier_pathway:{pathway}>")
    if superclass is not None:
        parts.append(f"<np_classifier_superclass:{superclass}>")
    if is_glycoside is not None:
        parts.append(f"<np_classifier_is_glycoside:{is_glycoside}>")
    if aromatic_rings is not None:
        parts.append(f"<aromatic_rings_count:{aromatic_rings}>")
    if qed_bin is not None:
        parts.append(f"<qed_bin:{qed_bin}>")
    if sa_bin is not None:
        parts.append(f"<sa_bin:{sa_bin}>")
    return " ".join(parts) + " " if parts else ""


def main():
    """
    Run inference with a trained language model.

    Supports two prompt modes:
      1. Legacy: --np_class "<NP:Fatty acids>" (v1 checkpoint)
      2. v2: --pathway, --superclass, --is_glycoside, --aromatic_rings, --qed_bin, --sa_bin
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Inference yaml configuration file")
    ap.add_argument("--seed", type=int, default=None, help="Random seed (overrides yaml)")
    ap.add_argument("--output", type=str, default=None, help="Output file path (overrides yaml)")
    ap.add_argument("--num_molecules", type=int, default=None, help="Number of molecules (overrides yaml)")

    # Legacy v1 prompt
    ap.add_argument("--np_class", type=str, default=None,
                     help="Legacy v1 prompt, e.g. '<NP:Fatty acids>'")

    # v2 conditioning tokens (each is optional)
    ap.add_argument("--pathway", type=str, default=None,
                     help="NP pathway, e.g. 'Alkaloids'")
    ap.add_argument("--superclass", type=str, default=None,
                     help="NP superclass, e.g. 'Flavonoids'")
    ap.add_argument("--is_glycoside", type=str, default=None,
                     help="Is glycoside: 'True' or 'False'")
    ap.add_argument("--aromatic_rings", type=str, default=None,
                     help="Aromatic ring count, e.g. '2'")
    ap.add_argument("--qed_bin", type=str, default=None,
                     help="QED bin, e.g. '0.5<=qed<0.6'")
    ap.add_argument("--sa_bin", type=str, default=None,
                     help="SA bin, e.g. '3<=sa<4'")

    args = ap.parse_args()
    configs = parse_yaml(args.yaml)

    # CLI overrides for basic settings
    if args.output is not None:
        configs["inference"]["output_file"] = args.output
    if args.num_molecules is not None:
        configs["inference"]["num_molecules"] = args.num_molecules

    # Build prompt: v2 tokens take priority over legacy np_class
    if any([args.pathway, args.superclass, args.is_glycoside,
            args.aromatic_rings, args.qed_bin, args.sa_bin]):
        prompt = build_prompt(
            pathway=args.pathway,
            superclass=args.superclass,
            is_glycoside=args.is_glycoside,
            aromatic_rings=args.aromatic_rings,
            qed_bin=args.qed_bin,
            sa_bin=args.sa_bin,
        )
    elif args.np_class is not None:
        prompt = args.np_class
    else:
        prompt = configs["inference"].get("np_class", "")

    # Set random seed
    seed = args.seed if args.seed is not None else configs["inference"].get("seed", None)
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")

    ckpt = configs["inference"]["ckpt_path"]

    # Load tokenizer and model
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()

    # Output file
    out_file = Path(configs["inference"]["output_file"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("")

    num_molecules = int(configs["inference"]["num_molecules"])
    print(f"Prompt: {prompt!r}")
    print(f"Generating {num_molecules} molecules ...")

    # Tokenize prompt
    x = tok(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = x["input_ids"].shape[1]

    # Generate
    generated = 0
    with out_file.open("a") as f:
        for _ in range(num_molecules):
            y = model.generate(
                **x,
                max_new_tokens=200,
                do_sample=True,
                top_p=configs["inference"]["top_p"],
                temperature=configs["inference"]["temperature"],
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

            # Decode only generated tokens (skip conditioning prompt)
            generated_ids = y[0][prompt_len:]
            raw = tok.decode(generated_ids, skip_special_tokens=True).strip()
            smiles = raw.split(".")[0].strip()

            if smiles:
                f.write(smiles + "\n")
                generated += 1

    print(f"Generated {generated}/{num_molecules} molecules -> {out_file}")


if __name__ == "__main__":
    main()
