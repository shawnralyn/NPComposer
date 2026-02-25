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
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="Inference yaml configuration file")
    args = ap.parse_args()

    configs = parse_yaml(args.yaml)

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

    # create starting sequence
    x = tok(configs["inference"]["np_class"], return_tensors="pt", add_special_tokens=False)

    # generate n number of molecules and append each SMILES string to output file
    with out_file.open("a") as f:
        for _ in range(int(configs["inference"]["num_molecules"])):
            y = model.generate(
                **x,
                max_new_tokens=200,  # set to max tokens used for training
                do_sample=True,
                top_p=configs["inference"]["top_p"],
                temperature=configs["inference"]["temperature"],
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )

            filtered_tok = tok.decode(y[0], skip_special_tokens=True).split(".")[0]  # grab first SMILES string in output

            if filtered_tok:
                f.write(filtered_tok + "\n")


if __name__ == "__main__":
    main()