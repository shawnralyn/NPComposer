import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_yaml(yml_path):
    """
    Read a YAML configuration file and return it as a dictionary.
    """
    try:
        with open(yml_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find YAML file: {yml_path}")
    except yaml.YAMLError as exc:
        raise ValueError(f"Error parsing YAML file {yml_path}: {exc}")


def load_special_tokens_map(json_path):
    """
    Load the special_tokens_map.json file.
    """
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find JSON file: {json_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"Error parsing JSON file {json_path}: {exc}")


def build_prompt(
    pathway=None,
    superclass=None,
    is_glycoside=None,
    aromatic_rings=None,
    qed_bin=None,
    sa_bin=None,
):
    """
    Build conditioning prompt from individual tokens.
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
    return "".join(parts)


def extract_pathway_and_superclass_tokens(special_tokens_map):
    """
    Extract only special tokens whose content is:
      - <np_classifier_pathway:...>
      - <np_classifier_superclass:...>

    Returns:
        list[str]: raw token strings
    """
    tokens = []
    for token_info in special_tokens_map.get("additional_special_tokens", []):
        content = token_info.get("content", "")
        if (
            content.startswith("<np_classifier_pathway:")
            or content.startswith("<np_classifier_superclass:")
        ):
            tokens.append(content)
    return tokens


def sanitize_token_for_filename(token):
    """
    Convert a special token into a filesystem-friendly filename stem.
    """
    text = token.strip("<>")
    text = text.replace(":", "__")
    text = text.replace("/", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^\w\-\.]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def set_random_seed(seed):
    """
    Set random seed for reproducibility.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed: {seed}")


def load_model_and_tokenizer(ckpt_path):
    """
    Load tokenizer and causal LM model from checkpoint path.
    """
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()
    return tokenizer, model


def generate_smiles_for_prompt(
    model,
    tokenizer,
    prompt,
    num_molecules,
    top_p,
    temperature,
    max_new_tokens=200,
    show_progress=False,
    progress_desc=None,
):
    """
    Generate molecules for a single prompt.

    Returns:
        list[str]: generated SMILES strings
    """
    x = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    prompt_len = x["input_ids"].shape[1]

    if torch.cuda.is_available():
        x = {k: v.to(model.device) for k, v in x.items()}

    generated_smiles = []

    iterator = range(num_molecules)
    if show_progress:
        iterator = tqdm(
            iterator,
            total=num_molecules,
            desc=progress_desc,
            leave=False,
            unit="mol",
        )

    for _ in iterator:
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_ids = y[0][prompt_len:]
        raw = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        smiles = raw.split(".")[0].strip()

        if smiles:
            generated_smiles.append(smiles)

    return generated_smiles


def write_smiles_to_file(smiles_list, out_file):
    """
    Write one SMILES per line.
    """
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w") as f:
        for smiles in smiles_list:
            f.write(smiles + "\n")


def run_single_prompt_mode(args, configs, model, tokenizer):
    """
    Original single-prompt inference mode.
    """
    if any(
        [
            args.pathway,
            args.superclass,
            args.is_glycoside,
            args.aromatic_rings,
            args.qed_bin,
            args.sa_bin,
        ]
    ):
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

    out_file = Path(configs["inference"]["output_file"])
    num_molecules = int(configs["inference"]["num_molecules"])
    top_p = configs["inference"]["top_p"]
    temperature = configs["inference"]["temperature"]

    print(f"Prompt: {prompt!r}")
    print(f"Generating {num_molecules} molecules ...")

    smiles_list = generate_smiles_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        num_molecules=num_molecules,
        top_p=top_p,
        temperature=temperature,
        show_progress=True,
        progress_desc="Generating",
    )

    write_smiles_to_file(smiles_list, out_file)
    print(f"Generated {len(smiles_list)}/{num_molecules} molecules -> {out_file}")


def run_batch_special_token_mode(args, configs, model, tokenizer):
    """
    Batch mode:
    - read special_tokens_map.json
    - find pathway/superclass tokens
    - generate num_molecules for each token
    - write one output file per token
    """
    if args.special_tokens_map is None:
        raise ValueError(
            "--special_tokens_map is required when using "
            "--generate_all_pathway_superclass"
        )

    if args.output_dir is None:
        raise ValueError(
            "--output_dir is required when using "
            "--generate_all_pathway_superclass"
        )

    special_tokens_map = load_special_tokens_map(args.special_tokens_map)
    class_tokens = extract_pathway_and_superclass_tokens(special_tokens_map)

    if not class_tokens:
        raise ValueError(
            "No pathway/superclass tokens were found in the provided "
            "special_tokens_map JSON."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_molecules = int(configs["inference"]["num_molecules"])
    top_p = configs["inference"]["top_p"]
    temperature = configs["inference"]["temperature"]

    print(f"Found {len(class_tokens)} pathway/superclass tokens.")
    print(f"Generating {num_molecules} molecules for each token...")
    print(f"Output directory: {output_dir}")

    token_iterator = tqdm(
        class_tokens,
        desc="Class tokens",
        unit="token",
        leave=True,
    )

    for token in token_iterator:
        token_iterator.set_postfix_str(token)

        filename_stem = sanitize_token_for_filename(token)
        out_file = output_dir / f"{filename_stem}.txt"

        smiles_list = generate_smiles_for_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=token,
            num_molecules=num_molecules,
            top_p=top_p,
            temperature=temperature,
            show_progress=True,
            progress_desc=filename_stem[:40],
        )

        write_smiles_to_file(smiles_list, out_file)


def main():
    """
    Run inference with trained language model.

    Supports:
      1. Single prompt mode
      2. Batch mode over all pathway/superclass special tokens from a JSON file
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Inference YAML configuration file")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides yaml)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (single mode only)",
    )
    ap.add_argument(
        "--num_molecules",
        type=int,
        default=None,
        help="Number of molecules per prompt (overrides yaml)",
    )

    # Legacy v1 prompt
    ap.add_argument(
        "--np_class",
        type=str,
        default=None,
        help="Legacy v1 prompt, e.g. '<NP:Fatty acids>'",
    )

    # v2 conditioning tokens
    ap.add_argument("--pathway", type=str, default=None, help="NP pathway")
    ap.add_argument("--superclass", type=str, default=None, help="NP superclass")
    ap.add_argument(
        "--is_glycoside",
        type=str,
        default=None,
        help="Is glycoside: 'True' or 'False'",
    )
    ap.add_argument(
        "--aromatic_rings",
        type=str,
        default=None,
        help="Aromatic ring count",
    )
    ap.add_argument("--qed_bin", type=str, default=None, help="QED bin")
    ap.add_argument("--sa_bin", type=str, default=None, help="SA bin")

    # Batch mode args
    ap.add_argument(
        "--generate_all_pathway_superclass",
        action="store_true",
        help=(
            "Generate molecules for every pathway/superclass special token "
            "in special_tokens_map JSON"
        ),
    )
    ap.add_argument(
        "--special_tokens_map",
        type=str,
        default=None,
        help="Path to special_tokens_map.json",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for batch-mode output files",
    )

    args = ap.parse_args()
    configs = parse_yaml(args.yaml)

    if args.output is not None:
        configs["inference"]["output_file"] = args.output
    if args.num_molecules is not None:
        configs["inference"]["num_molecules"] = args.num_molecules

    seed = args.seed if args.seed is not None else configs["inference"].get("seed", None)
    set_random_seed(seed)

    ckpt = configs["inference"]["ckpt_path"]
    tokenizer, model = load_model_and_tokenizer(ckpt)

    if torch.cuda.is_available():
        model = model.to("cuda")

    if args.generate_all_pathway_superclass:
        run_batch_special_token_mode(args, configs, model, tokenizer)
    else:
        run_single_prompt_mode(args, configs, model, tokenizer)


if __name__ == "__main__":
    main()