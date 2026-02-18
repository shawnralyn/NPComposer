# Current Components

## 1. Data pipeline component
download_data.sh / download_npass.sh: Download COCONUT and NPASS raw datasets
merge_npass.py: Merge NPASS CSV files into a single CSV
create_subset.py: Filter molecules (atom count<=150, ring count<=10, SA score), embed Morgan fingerprints into Tanimoto space via PCA, select diverse representatives via torch-accelerated K-means with closest-to-centroid selection
merge_training.py: Merge COCONUT and NPASS subsets, deduplicate by SMILES, fill superclass labels
classyfire.py: ClassyFire superclass classification via REST API 


## 2.Analysis components
analyze_distribution.py: Compare raw vs processed distributions (MW, SA, QED, NPL, superclass) & draw histogram for each features
analyze_sdf.py: Inspect SDF file structure and 3D properties

## 3.Evaluation
metrics.py: Evaluate generated molecules (validity, SA, QED, NPL)

## 4.Framework component
Makefile: Build automation (make all runs full pipeline)
config.yaml: Hydra configuration defaults
npcomposer.def: Apptainer container definition
tests: pytest test suite

## 5.Training components
Training components:
Specify inputs and outputs and side effects
1. Pre-trained foundation model - Transformer model trained on 1.1B SMILES strings
2. Foundation model tokenizer - Defines tokenization scheme and maps tokens to ID dictionary for conversion of SMILES information to numerical representation
3. Model/tokenizer loader - methods for loading foundation model and associated tokenizer provided by transformers library
4. Special character tokenizer - function that will read all values in select columns of database and create special tokens using tokenizer.add_special_tokens function
5. Tokenizer - Function that will tokenize SMILES data according to foundation model tokenizer
6. Trainer - Function that will take training arguments and carry out fine-tuning of foundation model
7. Evaluator - Function that uses Evaluator/wandb library to track learning rates and creates plot per epoch