1. Data Layer

Vocabulary: SMILES token dictionary with special tokens
SMILESTokenizer: converts SMILES to token sequences
MoleculeDataset: loads molecular data
DataPipeline: preprocessing, augmentation, batching, splitting

2. Transformer Model. Future

PositionalEncoding: adds position info to sequences
MoleculeEmbedding: maps tokens to vectors
MultiHeadAttention: self attention mechanism
TransformerEncoder: encodes molecular representations
TransformerDecoder: generates SMILES tokens autoregressively
TransformerModel: encoder decoder wrapper

3. RL Components. Future

PolicyNetwork: transformer as policy outputting token probabilities
ValueNetwork: critic estimating expected reward
RLAgent: RL algorithm logic
ExperienceBuffer: stores trajectories

4. Reward and Scoring

RewardFunction: orchestrates multiple scoring components
DrugLikenessScorer: QED and Lipinski evaluation
ToxicityPredictor: predicts toxicity risk
BindingAffinityPredictor: estimates target protein binding affinity
SynthesizabilityScorer: synthetic accessibility score
NoveltyChecker: checks novelty against known databases

5. Generation

MoleculeGenerator: generates candidate molecules
SamplingStrategy: greedy, top k, beam search, etc
MoleculeValidator: RDKit based validity check

6. Training

PreTrainer: supervised pre training with teacher forcing
RLTrainer: generate, score, update policy loop
TrainingConfig: hyperparameter management
CheckpointManager: model save, load, early stopping

7. Evaluation

MoleculeEvaluator: evaluates generated molecules
PropertyPredictor: predicts ADMET properties
DiversityCalculator: measures structural diversity
DistributionAnalyzer: compares training vs generated distributions

8. Utilities

MoleculeVisualizer: 2D and 3D structure rendering
Logger: training metrics logging
Config: global configuration management
