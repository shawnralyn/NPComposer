# data/

## raw/

### COCONUT

Download: `bash scripts/download_data.sh`

| Column | Description |
|--------|-------------|
| identifier | COCONUT ID |
| canonical_smiles | Normalized SMILES |
| standard_inchi | InChI |
| standard_inchi_key | InChI Key |
| name | Compound name |
| iupac_name | IUPAC name |
| annotation_level | Annotation level |
| total_atom_count | Total atoms |
| heavy_atom_count | Heavy atoms |
| molecular_weight | Molecular weight |
| exact_molecular_weight | Exact mass |
| molecular_formula | Formula |
| alogp | ALogP |
| topological_polar_surface_area | TPSA |
| rotatable_bond_count | Rotatable bonds |
| hydrogen_bond_acceptors | HBA |
| hydrogen_bond_donors | HBD |
| hydrogen_bond_acceptors_lipinski | HBA (Lipinski) |
| hydrogen_bond_donors_lipinski | HBD (Lipinski) |
| lipinski_rule_of_five_violations | Lipinski violations |
| aromatic_rings_count | Aromatic rings |
| qed_drug_likeliness | QED (COCONUT) |
| formal_charge | Formal charge |
| fractioncsp3 | Fraction Csp3 |
| number_of_minimal_rings | Ring count |
| van_der_walls_volume | VdW volume |
| contains_sugar | Contains sugar |
| contains_ring_sugars | Ring sugars |
| contains_linear_sugars | Linear sugars |
| murcko_framework | Murcko scaffold |
| np_likeness | NP-likeness (COCONUT) |
| chemical_class | Chemical class |
| chemical_sub_class | Chemical subclass |
| chemical_super_class | Chemical superclass |
| direct_parent_classification | Direct parent |
| np_classifier_pathway | NP pathway |
| np_classifier_superclass | NP superclass |
| np_classifier_class | NP class |
| np_classifier_is_glycoside | Is glycoside |
| organisms | Source organisms |
| collections | Collections |
| dois | DOIs |
| synonyms | Synonyms |
| cas | CAS number |

### NPASS

Download: `bash scripts/download_npass.sh && python scripts/merge_npass.py -i data/raw/npass -o data/raw/npass_full.csv`

| Column | Description |
|--------|-------------|
| np_id | NPASS ID |
| inchikey | InChI Key |
| name | Compound name |
| iupac_name | IUPAC name |
| chembl_id | ChEMBL ID |
| pubchem_id | PubChem ID |
| name_initial | Name initial |
| num_of_organism | Organism count |
| num_of_target | Target count |
| num_of_activity | Activity count |
| gene_cluster | Gene cluster |
| ifQuantity | Has quantity |
| canonical_smiles | SMILES |
| standard_inchi | InChI |
| standard_inchi_key | InChI Key |
| activity_count | Activity records |
| activity_types | Activity types |
| activity_value_mean | Mean activity value |
| organism_count | Organism count |
| organisms | Source organisms |
| toxicity_types | Toxicity types |
| toxicity_value_mean | Mean toxicity value |

## processed/

Output from `scripts/create_subset.py`, `scripts/merge_training.py`, and `scripts/analyze_distribution.py`.

Files:

- `coconut_100000.csv` — COCONUT subset (100K)
- `coconut_100000.sdf` — COCONUT 3D structures
- `npass_100000.csv` — NPASS subset (100K)
- `training_data.csv` — Merged training dataset (COCONUT + NPASS, deduplicated)
- `dist_coconut_histograms.png` — COCONUT raw vs processed distribution histograms
- `dist_coconut_summary.csv` — COCONUT distribution summary statistics
- `dist_coconut_superclass.csv` — COCONUT superclass distribution comparison
- `dist_npass_histograms.png` — NPASS raw vs processed distribution histograms
- `dist_npass_summary.csv` — NPASS distribution summary statistics

Added columns (RDKit):

| Column | Description |
|--------|-------------|
| sa_score(RDKit) | Synthetic accessibility (1-10) |
| qed(RDKit) | Drug-likeness (0-1) |
| npl_score(RDKit) | NP-likeness (-3~+3) |
| superclass | NPClassifier superclass |
| source | Dataset origin: coconut or npass (training_data.csv only) |

## splits/

Output from `scripts/split_data.py`

- `train.csv` (80%)
- `val.csv` (10%)
- `test.csv` (10%)
