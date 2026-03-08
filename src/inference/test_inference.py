from transformers import AutoTokenizer, AutoModelForCausalLM
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Contrib.SA_Score import sascorer

tok = AutoTokenizer.from_pretrained("ralyn/NPComposer-v2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("ralyn/NPComposer-v2", trust_remote_code=True).eval()

x = tok("<np_classifier_pathway:Terpenoids>", return_tensors="pt", add_special_tokens=False)
y = model.generate(**x, max_new_tokens=200, do_sample=True, top_p=0.95, temperature=0.85)

filtered_tok = tok.decode(y[0], skip_special_tokens=True).split(".")[0]
print(filtered_tok)

smiles = filtered_tok.strip()
mol = Chem.MolFromSmiles(smiles)
if mol is not None:
    qed_score = QED.qed(mol)
    sa_score = sascorer.calculateScore(mol)
    print(f"QED score: {qed_score:.3f}")
    print(f"SA score: {sa_score:.3f}")
else:
    print("Invalid SMILES string:", smiles)