import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ckpt = "" # add path to checkpoint folder
tok=AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True); 
model=AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.float32).eval()

x=tok("<NP:Terpenoids>", return_tensors="pt", add_special_tokens=False)
y = model.generate(**x, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.1,
                   eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
print(tok.decode(y[0], skip_special_tokens=False))