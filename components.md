# Components

Each component description includes the component name, what it does, its inputs/outputs with type information, and how it interacts with other components.

---

## 1. Data Layer

Vocabulary: SMILES token dictionary with special tokens. SMILESTokenizer: converts SMILES to token sequences. MoleculeDataset: loads molecular data from CSV. DataPipeline: preprocessing, augmentation, batching, splitting.

Inputs:
- smiles_csv (str): path to CSV with SMILES and class labels
- smiles (str): raw SMILES string
- config (dict): YAML configuration

Outputs:
- tokens (List[int]): tokenized SMILES sequence
- dataset (datasets.Dataset): tokenized dataset with input_ids and attention_mask
- special_tokens_dict (dict): special conditioning tokens

Interactions: Feeds tokenized datasets into Training. Provides vocabulary and special tokens used by Generation for decoding.

---

## 2. Transformer Model (Future)

PositionalEncoding: adds position info to sequences. MoleculeEmbedding: maps tokens to vectors. MultiHeadAttention: self-attention mechanism. TransformerEncoder: encodes molecular representations. TransformerDecoder: generates SMILES tokens autoregressively. TransformerModel: encoder-decoder wrapper.

Inputs:
- input_ids (torch.Tensor): batch of tokenized SMILES sequences
- attention_mask (torch.Tensor): mask for valid positions
- labels (torch.Tensor): target token IDs for loss

Outputs:
- logits (torch.Tensor): token probability distributions
- loss (torch.Tensor): cross-entropy loss
- hidden_states (torch.Tensor): encoder representations

Interactions: Receives tokenized inputs from Data Layer. Used by Training for optimization. Serves as backbone for Generation and as policy network for RL Components.

---

## 3. RL Components (Future)

PolicyNetwork: transformer as policy outputting token probabilities. ValueNetwork: critic estimating expected reward. RLAgent: RL algorithm logic (e.g., PPO). ExperienceBuffer: stores trajectories.

Inputs:
- state (torch.Tensor): current partial SMILES sequence
- reward (float): scalar reward signal
- trajectory (List[dict]): (state, action, reward) tuples

Outputs:
- action (int): next token ID from policy
- value_estimate (float): expected return estimate
- policy_loss (torch.Tensor): policy gradient loss

Interactions: Uses Transformer Model as policy network. Receives rewards from Reward and Scoring. Trained via Training component's RLTrainer.

---

## 4. Reward and Scoring

RewardFunction: orchestrates multiple scoring components. DrugLikenessScorer: QED and Lipinski evaluation. ToxicityPredictor: predicts toxicity risk. BindingAffinityPredictor: estimates target binding affinity. SynthesizabilityScorer: SA score. NoveltyChecker: checks novelty against known databases.

Inputs:
- smiles (str): generated SMILES string
- mol (rdkit.Chem.Mol): RDKit molecule object
- known_smiles (Set[str]): known molecules for novelty check

Outputs:
- reward (float): aggregated scalar reward
- qed_score (float): drug-likeness score (0-1)
- sa_score (float): synthetic accessibility score (1-10)
- toxicity_risk (float): toxicity probability
- is_novel (bool): novelty flag

Interactions: Provides rewards to RL Components. Called by Evaluation for quality assessment. Uses RDKit for property computation.

---

## 5. Generation

MoleculeGenerator: generates candidate molecules. SamplingStrategy: greedy, top-k, beam search, etc. MoleculeValidator: RDKit-based validity check.

Inputs:
- prompt (str): conditioning token string
- num_molecules (int): number to generate
- temperature (float): sampling temperature
- top_p (float): nucleus sampling threshold

Outputs:
- generated_smiles (List[str]): generated SMILES list
- valid_smiles (List[str]): valid SMILES only
- output_file (str): text file path

Interactions: Loads trained model from Training. Uses vocabulary from Data Layer. Output passed to Evaluation and Reward and Scoring.

---

## 6. Training

PreTrainer: supervised pre-training with teacher forcing. RLTrainer: generate, score, update policy loop. TrainingConfig: hyperparameter management. CheckpointManager: model save, load, early stopping.

Inputs:
- ds_train (datasets.Dataset): tokenized training dataset
- ds_val (datasets.Dataset): tokenized validation dataset
- model (transformers.AutoModelForCausalLM): model to train
- tokenizer (transformers.AutoTokenizer): tokenizer with special tokens
- training_args (dict): hyperparameters

Outputs:
- checkpoint (str): saved model weights path
- training_logs (dict): loss and metrics logged to W&B
- best_model (transformers.AutoModelForCausalLM): best model by validation loss

Interactions: Receives data from Data Layer. Optimizes Transformer Model. Produces checkpoints for Generation. Coordinates RL Components and Reward and Scoring during RL training.

---

## 7. Evaluation

MoleculeEvaluator: evaluates generated molecules. PropertyPredictor: predicts ADMET properties. DiversityCalculator: measures structural diversity. DistributionAnalyzer: compares training vs generated distributions.

Inputs:
- smiles_list (List[str]): generated SMILES strings
- np_repo_root (Optional[str]): NP-Classifier repo path

Outputs:
- validity (float): fraction of valid molecules
- sa_score (Dict[str, float]): SA statistics
- qed (Dict[str, float]): QED statistics
- np_score (Dict[str, float]): NP-likeness statistics
- results_json (str): JSON output path

Interactions: Takes SMILES from Generation. Calls NPClassifier from Utilities for classification. Provides feedback metrics for Reward and Scoring.

---

## 8. Utilities

MoleculeVisualizer: 2D and 3D structure rendering. Logger: training metrics logging. Config: global configuration management via YAML.

Inputs:
- smiles (str): SMILES string for visualization
- yaml_path (str): YAML config file path
- metrics (dict): metric names and values

Outputs:
- image (PIL.Image or str): rendered structure
- configs (Dict[str, Any]): parsed configuration
- log_entry (dict): formatted log record

Interactions: Config is consumed by Data Layer, Training, and Generation. Logger is used by Training and Evaluation. MoleculeVisualizer is used by Evaluation for visual reports.
