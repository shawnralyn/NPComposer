# Current Components

## 1. Data pipeline component
download_data.sh / download_npass.sh: Download COCONUT and NPASS raw datasets
merge_npass.py: Merge NPASS CSV files into a single CSV
create_subset.py: Filter molecules (atom count<=150, ring count<=10, SA score), embed Morgan fingerprints into Tanimoto space via PCA, select diverse representatives via torch-accelerated K-means with closest-to-centroid selection
merge_training.py: Merge COCONUT and NPASS subsets, deduplicate by SMILES, fill superclass labels
classyfire.py: ClassyFire superclass classification via REST API 


## 2.Analysis componen
analyze_distribution.py: Compare raw vs processed distributions (MW, SA, QED, NPL, superclass) & draw histogram for each features
analyze_sdf.py: Inspect SDF file structure and 3D properties

## 3.Evaluation
metrics.py: Evaluate generated molecules (validity, SA, QED, NPL)

## 4.Framework component
Makefile: Build automation (make all runs full pipeline)
config.yaml: Hydra configuration defaults
npcomposer.def: Apptainer container definition
tests: pytest test suite