"""Generate a single valid molecule and visualize it with RDKit."""

import sys
import os
import base64
import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NPGPT_RL_DIR = PROJECT_ROOT / "src" / "npgpt-rl"
DEFAULT_TOKENIZER = str(
    PROJECT_ROOT / "external/npgpt/externals/smiles-gpt/checkpoints/benchmark-10m/tokenizer.json"
)
MAX_RETRIES = 50


def load_npgpt(ckpt_path, tokenizer_path, is_rl=False):
    """Load NPGPT model.

    Input:
        ckpt_path: path to checkpoint.
        tokenizer_path: path to tokenizer.json.
        is_rl: load as RL checkpoint.
    Output:
        (model, tokenizer)
    """
    sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "src"))
    sys.path.insert(0, str(PROJECT_ROOT / "external" / "npgpt" / "externals" / "smiles-gpt"))
    from npgpt import SmilesGptModel, SmilesGptTrainingConfig, get_tokenizer

    cfg = SmilesGptTrainingConfig()
    tok = get_tokenizer(cfg, tokenizer_path)
    if is_rl:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model = SmilesGptModel(config=cfg, tokenizer=tok)
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model = SmilesGptModel.load_from_checkpoint(
            ckpt_path, config=cfg, tokenizer=tok, strict=False
        )
    model.eval()
    return model, tok


def generate_npgpt(model, tokenizer, temperature=1.5, top_p=1.0):
    """Generate one SMILES with NPGPT.

    Input:
        model, tokenizer: NPGPT model and tokenizer.
        temperature, top_p: sampling parameters.
    Output:
        SMILES string.
    """
    device = torch.device("cpu")
    model.to(device)
    with torch.no_grad():
        ids = torch.tensor([[tokenizer.bos_token_id]]).to(device)
        out = model.model.generate(
            ids, max_length=512, do_sample=True,
            temperature=temperature, top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        return tokenizer.decode(out[0], skip_special_tokens=True).strip()


GPMOLFORMER_TOKENIZER = "ibm-research/MoLFormer-XL-both-10pct"


def load_hf_model(ckpt_path, tokenizer_name=None):
    """Load HuggingFace causal LM.

    Input:
        ckpt_path: model name or path.
        tokenizer_name: tokenizer name/path (defaults to ckpt_path).
    Output:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok_path = tokenizer_name or ckpt_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt_path, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    return model, tokenizer


def generate_hf(model, tokenizer, prompt="", temperature=1.0, top_p=0.95, max_new_tokens=200):
    """Generate one SMILES with HuggingFace model.

    Input:
        model, tokenizer: HF model and tokenizer.
        prompt: conditioning prompt.
        temperature, top_p, max_new_tokens: sampling parameters.
    Output:
        SMILES string.
    """
    device = next(model.parameters()).device
    if prompt:
        x = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    else:
        # Empty prompt: start with BOS token
        bos_id = tokenizer.bos_token_id
        if bos_id is None:
            bos_id = tokenizer.cls_token_id or 0
        x = {"input_ids": torch.tensor([[bos_id]])}
    prompt_len = x["input_ids"].shape[1]
    x = {k: v.to(device) for k, v in x.items()}

    with torch.no_grad():
        y = model.generate(
            **x,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(y[0][prompt_len:], skip_special_tokens=True).strip()
    return raw.split(".")[0].strip()


def generate_valid_molecule(gen_fn, max_retries=MAX_RETRIES):
    """Retry generation until a valid SMILES is produced.

    Input:
        gen_fn: callable returning a SMILES string.
        max_retries: max attempts.
    Output:
        (smiles, mol) tuple.
    """
    for i in range(1, max_retries + 1):
        smi = gen_fn()
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            canonical = Chem.MolToSmiles(mol)
            print(f"Valid molecule found (attempt {i}/{max_retries})")
            print(f"SMILES: {canonical}")
            return canonical, mol
        print(f"  attempt {i}: invalid SMILES '{smi}', retrying...")
    raise RuntimeError(f"Failed to generate a valid molecule after {max_retries} attempts.")



def display_in_terminal(png_path):
    """Display PNG image inline in the terminal.

    Tries iTerm2/Kitty/sixel protocols, then opens with system viewer.

    Input:
        png_path: path to PNG file.
    """
    import platform

    with open(png_path, "rb") as f:
        data = f.read()

    # iTerm2 / WezTerm inline image protocol
    if os.environ.get("TERM_PROGRAM") in ("iTerm.app", "WezTerm"):
        b64 = base64.b64encode(data).decode("ascii")
        sys.stdout.write(f"\033]1337;File=inline=1;size={len(data)}:{b64}\a\n")
        sys.stdout.flush()
        return

    # Kitty terminal graphics protocol
    if os.environ.get("TERM") == "xterm-kitty" or os.environ.get("KITTY_PID"):
        b64 = base64.b64encode(data).decode("ascii")
        chunk_size = 4096
        chunks = [b64[i:i + chunk_size] for i in range(0, len(b64), chunk_size)]
        for i, chunk in enumerate(chunks):
            m = 1 if i < len(chunks) - 1 else 0
            if i == 0:
                sys.stdout.write(f"\033_Ga=T,f=100,t=d,m={m};{chunk}\033\\")
            else:
                sys.stdout.write(f"\033_Gm={m};{chunk}\033\\")
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    # sixel via img2sixel
    if shutil.which("img2sixel"):
        subprocess.run(["img2sixel", str(png_path)])
        return

    # macOS: open with Preview
    if platform.system() == "Darwin":
        subprocess.Popen(["open", str(png_path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  (Opened in Preview)")
        return

    # Linux: xdg-open
    if shutil.which("xdg-open"):
        subprocess.Popen(["xdg-open", str(png_path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("  (Opened in image viewer)")
        return

    print(f"  (View the image at: {png_path})")


def visualize(mol, out_path, size=(500, 400)):
    """Save molecule image and display in terminal.

    Input:
        mol: RDKit Mol object.
        out_path: output PNG path.
        size: image dimensions.
    Output:
        none (saves PNG and prints to terminal).
    """
    # Save clean white-background PNG
    img = Draw.MolToImage(mol, size=size)
    img.save(str(out_path))
    print(f"\nImage saved -> {out_path}\n")

    # Display in terminal (inline protocol or open in viewer)
    display_in_terminal(out_path)


MODEL_CHOICES = ["npgpt", "npgpt-rl", "gpmolformer", "npcomposer"]


def main():
    parser = argparse.ArgumentParser(description="Generate and visualize a single valid molecule.")
    parser.add_argument("model", choices=MODEL_CHOICES, help="Model to use")
    parser.add_argument("--prompt", type=str, default="", help="Conditioning prompt (npcomposer/gpmolformer)")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=None, help="Nucleus sampling p")
    parser.add_argument("--out", type=str, default=None, help="Output image path")
    parser.add_argument("--orig_ckpt", type=str, default=str(NPGPT_RL_DIR / "npgpt.ckpt"))
    parser.add_argument("--rl_ckpt", type=str, default=str(NPGPT_RL_DIR / "npgpt_rl_step_600.ckpt"))
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER)
    parser.add_argument("--npcomposer_ckpt", type=str, default="ralyn/NPComposer-v2")
    parser.add_argument("--gpmolformer_ckpt", type=str, default="ibm-research/GP-MoLFormer-Uniq")
    parser.add_argument("--max_retries", type=int, default=MAX_RETRIES)
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / "results" / "molecules"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"{args.model}_molecule.png"

    if args.model == "npgpt":
        temp = args.temperature if args.temperature is not None else 1.5
        top_p = args.top_p if args.top_p is not None else 1.0
        print(f"Loading NPGPT (pretrained): {args.orig_ckpt}")
        model, tok = load_npgpt(args.orig_ckpt, args.tokenizer, is_rl=False)
        gen_fn = lambda: generate_npgpt(model, tok, temperature=temp, top_p=top_p)

    elif args.model == "npgpt-rl":
        temp = args.temperature if args.temperature is not None else 1.5
        top_p = args.top_p if args.top_p is not None else 1.0
        print(f"Loading NPGPT (RL): {args.rl_ckpt}")
        model, tok = load_npgpt(args.rl_ckpt, args.tokenizer, is_rl=True)
        gen_fn = lambda: generate_npgpt(model, tok, temperature=temp, top_p=top_p)

    elif args.model == "gpmolformer":
        temp = args.temperature if args.temperature is not None else 1.0
        print(f"Loading GP-MoLFormer: {args.gpmolformer_ckpt}")
        model, tok = load_hf_model(args.gpmolformer_ckpt, tokenizer_name=GPMOLFORMER_TOKENIZER)
        if torch.cuda.is_available():
            model = model.to("cuda")
        gen_fn = lambda: generate_hf(model, tok, prompt="", temperature=temp)

    elif args.model == "npcomposer":
        temp = args.temperature if args.temperature is not None else 1.0
        top_p = args.top_p if args.top_p is not None else 0.95
        print(f"Loading NPComposer: {args.npcomposer_ckpt}")
        model, tok = load_hf_model(args.npcomposer_ckpt)
        if torch.cuda.is_available():
            model = model.to("cuda")
        prompt = args.prompt
        print(f"Prompt: {prompt!r}")
        gen_fn = lambda: generate_hf(model, tok, prompt=prompt, temperature=temp, top_p=top_p)

    smi, mol = generate_valid_molecule(gen_fn, max_retries=args.max_retries)
    visualize(mol, out_path)


if __name__ == "__main__":
    main()
