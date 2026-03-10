"""NPClassifier local inference.

Loads Keras .hdf5 models and runs prediction in-process.
No server, no HTTP, no network required.

This module contains code adapted from NP-Classifier
(https://github.com/mwang87/NP-Classifier), which is licensed under
the MIT License. See THIRD_PARTY_LICENSES in the project root for the
full license text.

Reference:
    Kim HW et al. "NPClassifier: A Deep Neural Network-Based Structural
    Classification Tool for Natural Products."
    J. Nat. Prod. 2021, 84, 2795-2807.

Setup:
    1. git clone https://github.com/mwang87/NP-Classifier
    2. cd NP-Classifier/Classifier/models_folder/models
       wget -O models.zip "https://zenodo.org/record/5068687/files/model.zip?download=1"
       unzip models.zip
    3. export NP_CLASSIFIER_ROOT=/path/to/NP-Classifier

Required files under NP_CLASSIFIER_ROOT:
    Classifier/dict/index_v1.json
    Classifier/models_folder/models/NP_classifier_pathway_V1.hdf5
    Classifier/models_folder/models/NP_classifier_superclass_V1.hdf5
    Classifier/models_folder/models/NP_classifier_class_V1.hdf5
"""

import json
import os
import itertools
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

CACHE_FILENAME = "npclassifier_cache.json"


def _load_cache(path):
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache, path):
    try:
        Path(path).write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    except OSError:
        pass


# -- Fingerprint (from NP-Classifier/Classifier/fingerprint_handler.py) ----

def calculate_fingerprint(smiles, radi=2):
    """Generate counted Morgan fingerprints for NPClassifier.

    Input:
        smiles: SMILES string.
        radi: Morgan radius (default 2).
    Output:
        (formula [1x2048], binary [1x4096]) numpy arrays, or None if invalid.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    binary = np.zeros((2048 * radi), int)
    formula = np.zeros(2048, int)
    mol_bi = {}

    for r in range(radi + 1):
        mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, radius=r, bitInfo=mol_bi, nBits=2048
        )
        mol_bi_QC = []
        for i in mol_fp.GetOnBits():
            for j in range(len(mol_bi[i])):
                if mol_bi[i][j][1] == r:
                    mol_bi_QC.append(i)
                    break

        if r == 0:
            for i in mol_bi_QC:
                formula[i] = len([k for k in mol_bi[i] if k[1] == 0])
        else:
            for i in mol_bi_QC:
                binary[(2048 * (r - 1)) + i] = len(
                    [k for k in mol_bi[i] if k[1] == r]
                )

    return formula.reshape(1, 2048), binary.reshape(1, 4096)


def _isglycoside(smiles):
    """Check glycoside substructure."""
    sugar2 = Chem.MolFromSmarts(
        '[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C1),'
        '$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C@C@C1)]'
    )
    sugar3 = Chem.MolFromSmarts(
        '[OX2;$([r5]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C1),'
        '$([r6]1@C(!@[OX2,NX3,SX2,FX1,ClX1,BrX1,IX1])@C@C(O)@C(O)@C1)]'
    )
    sugar4 = Chem.MolFromSmarts(
        '[OX2;$([r5]1@C(!@[OX2H1])@C@C@C1),'
        '$([r6]1@C(!@[OX2H1])@C@C@C@C1)]'
    )
    mol = Chem.MolFromSmiles(smiles)
    try:
        return (mol.HasSubstructMatch(sugar2)
                or mol.HasSubstructMatch(sugar3)
                or mol.HasSubstructMatch(sugar4))
    except Exception:
        return False


# -- Voting (from NP-Classifier/Classifier/prediction_voting.py) -----------

def _vote_classification(n_path, n_class, n_super,
                         pred_class, pred_super,
                         path_from_class, path_from_superclass,
                         isglycoside, index):
    """Run voting algorithm on raw model predictions.

    Input:
        n_path, n_class, n_super: index lists from thresholding.
        pred_class, pred_super: raw prediction arrays.
        path_from_class, path_from_superclass: hierarchy-derived pathway indices.
        isglycoside: bool.
        index: ontology dict (index_v1.json).
    Output:
        (pathway_result, superclass_result, class_result, isglycoside) label lists.
    """
    class_result = []
    superclass_result = []
    pathway_result = []

    index_class = list(index['Class'].keys())
    index_superclass = list(index['Superclass'].keys())
    index_pathway = list(index['Pathway'].keys())

    path_for_vote = n_path + path_from_class + path_from_superclass
    path = list(set([k for k in path_for_vote if path_for_vote.count(k) == 3]))

    if not path:
        path = list(set([k for k in path_for_vote if path_for_vote.count(k) == 2]))
    if not path:
        for w in n_path:
            pathway_result.append(index_pathway[w])
        return pathway_result, superclass_result, class_result, isglycoside

    if set(n_path) & set(path):
        if set(path) & set(path_from_superclass):
            n_super = [l for l in n_super
                       if set(path) & set(index['Super_hierarchy'][str(l)]['Pathway'])]
            if not n_super:
                n_class = [m for m in n_class
                           if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway'])]
                n_super = [index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class]
                n_super = list(set(itertools.chain.from_iterable(n_super)))
            elif len(n_super) > 1:
                n_class = [u for u in n_class
                           if set(path) & set(index['Class_hierarchy'][str(u)]['Pathway'])]
                if n_class:
                    n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
                    n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
                    n_path = list(set(itertools.chain.from_iterable(n_path)))
                    n_super = list(set(itertools.chain.from_iterable(n_super)))
                elif len(path) == 1:
                    n_super = [np.argmax(pred_super)]
                    n_class = [m for m in [np.argmax(pred_class)]
                               if set(n_super) & set(index['Class_hierarchy'][str(m)]['Superclass'])]
            else:
                n_class = [o for o in n_class
                           if set(n_super) & set(index['Class_hierarchy'][str(o)]['Superclass'])]
                if not n_class:
                    n_class = [m for m in [np.argmax(pred_class)]
                               if set(n_super) & set(index['Class_hierarchy'][str(m)]['Superclass'])]
        else:
            n_class = [p for p in n_class
                       if set(path) & set(index['Class_hierarchy'][str(p)]['Pathway'])]
            n_super = [index['Class_hierarchy'][str(q)]['Superclass'] for q in n_class]
            n_super = list(set(itertools.chain.from_iterable(n_super)))
    else:
        n_super = [l for l in n_super
                   if set(path) & set(index['Super_hierarchy'][str(l)]['Pathway'])]
        if not n_super:
            n_class = [m for m in n_class
                       if set(path) & set(index['Class_hierarchy'][str(m)]['Pathway'])]
            n_super = [index['Class_hierarchy'][str(n)]['Superclass'] for n in n_class]
            n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
            n_path = list(set(itertools.chain.from_iterable(n_path)))
            n_super = list(set(itertools.chain.from_iterable(n_super)))
        elif len(n_super) > 1:
            n_class = [u for u in n_class
                       if set(path) & set(index['Class_hierarchy'][str(u)]['Pathway'])]
            n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
            n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
            n_path = list(set(itertools.chain.from_iterable(n_path)))
            n_super = list(set(itertools.chain.from_iterable(n_super)))
        else:
            n_class = [o for o in n_class
                       if set(path) & set(index['Class_hierarchy'][str(o)]['Pathway'])]
            n_super = [index['Class_hierarchy'][str(v)]['Superclass'] for v in n_class]
            n_path = [index['Class_hierarchy'][str(v)]['Pathway'] for v in n_class]
            n_path = list(set(itertools.chain.from_iterable(n_path)))
            n_super = list(set(itertools.chain.from_iterable(n_super)))

    for r in path:
        pathway_result.append(index_pathway[r])
    for s in n_super:
        superclass_result.append(index_superclass[s])
    for t in n_class:
        class_result.append(index_class[t])

    return pathway_result, superclass_result, class_result, isglycoside


# -- Classifier class -------------------------------------------------------

class NPClassifierLocal:
    """Local NPClassifier. Loads Keras .hdf5 models, runs inference in-process."""

    def __init__(self, repo_root=None):
        """Initialize classifier.

        Input:
            repo_root: path to NP-Classifier repo (or set NP_CLASSIFIER_ROOT env).
        """
        if repo_root is None:
            repo_root = os.environ.get("NP_CLASSIFIER_ROOT")
        if repo_root is None:
            raise ValueError(
                "Set NP_CLASSIFIER_ROOT env var or pass repo_root=."
            )
        self.root = Path(repo_root)
        self._validate_files()
        self._load_ontology()
        self._load_models()

    def _validate_files(self):
        required = [
            "Classifier/dict/index_v1.json",
            "Classifier/models_folder/models/NP_classifier_pathway_V1.hdf5",
            "Classifier/models_folder/models/NP_classifier_superclass_V1.hdf5",
            "Classifier/models_folder/models/NP_classifier_class_V1.hdf5",
        ]
        missing = [f for f in required if not (self.root / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing in {self.root}:\n"
                + "\n".join(f"  - {f}" for f in missing)
                + "\nRun get_models.sh to download from Zenodo."
            )

    def _load_ontology(self):
        path = self.root / "Classifier" / "dict" / "index_v1.json"
        with open(path) as f:
            self.ontology = json.load(f)

    def _load_models(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

        d = self.root / "Classifier" / "models_folder" / "models"
        print("Loading NPClassifier models...")
        self.model_pathway = tf.keras.models.load_model(
            str(d / "NP_classifier_pathway_V1.hdf5"), compile=False)
        self.model_superclass = tf.keras.models.load_model(
            str(d / "NP_classifier_superclass_V1.hdf5"), compile=False)
        self.model_class = tf.keras.models.load_model(
            str(d / "NP_classifier_class_V1.hdf5"), compile=False)
        print("NPClassifier models loaded.")

    def classify_one(self, smiles):
        """Classify a single SMILES.

        Input:
            smiles: SMILES string.
        Output:
            dict with pathway, superclass, class_, pathway_all, superclass_all,
            class_all, isglycoside. Returns {'error': ...} on failure.
        """
        results = self.classify_batch_raw([smiles])
        return results[0]

    def classify_batch_raw(self, smiles_list, batch_size=256):
        """Classify a list of SMILES using batched inference.

        Input:
            smiles_list: list of SMILES strings.
            batch_size: number of molecules per GPU/CPU batch (default 256).
        Output:
            list of dicts (same length as input).
        """
        # Pre-compute all fingerprints
        fps = []
        for smi in smiles_list:
            fp = calculate_fingerprint(smi, 2)
            fps.append(fp)

        # Separate valid vs invalid
        valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
        results = [{"error": "invalid SMILES"}] * len(smiles_list)

        if not valid_indices:
            return results

        # Stack fingerprints into batched arrays
        formulas = np.vstack([fps[i][0] for i in valid_indices])  # (N, 2048)
        binaries = np.vstack([fps[i][1] for i in valid_indices])  # (N, 4096)

        # Batched model prediction (much faster than one-by-one)
        print(f"  Batch predicting {len(valid_indices)} molecules (batch_size={batch_size})...")
        inp = {"input_2048": formulas, "input_4096": binaries}
        pred_paths = self.model_pathway.predict(inp, verbose=0, batch_size=batch_size)
        pred_supers = self.model_superclass.predict(inp, verbose=0, batch_size=batch_size)
        pred_classes = self.model_class.predict(inp, verbose=0, batch_size=batch_size)

        # Pre-compute glycoside flags
        glycosides = [_isglycoside(smiles_list[i]) for i in valid_indices]

        # Vote per molecule
        it = range(len(valid_indices))
        if HAS_TQDM:
            it = tqdm(it, desc="NPClassifier voting")

        for j in it:
            idx = valid_indices[j]
            pred_path = pred_paths[j]
            pred_super = pred_supers[j]
            pred_class = pred_classes[j]

            n_path = list(np.where(pred_path >= 0.5)[0])
            n_super = list(np.where(pred_super >= 0.3)[0])
            n_class = list(np.where(pred_class >= 0.1)[0])

            if not n_path:
                n_path = [int(np.argmax(pred_path))]

            path_from_class = []
            for k in n_class:
                path_from_class += self.ontology['Class_hierarchy'][str(k)]['Pathway']
            path_from_class = list(set(path_from_class))

            path_from_superclass = []
            for k in n_super:
                path_from_superclass += self.ontology['Super_hierarchy'][str(k)]['Pathway']
            path_from_superclass = list(set(path_from_superclass))

            pw, sc, cl, gly = _vote_classification(
                n_path, n_class, n_super,
                pred_class, pred_super,
                path_from_class, path_from_superclass,
                glycosides[j], self.ontology
            )

            results[idx] = {
                "pathway": pw[0] if pw else None,
                "superclass": sc[0] if sc else None,
                "class_": cl[0] if cl else None,
                "pathway_all": pw,
                "superclass_all": sc,
                "class_all": cl,
                "isglycoside": gly,
            }

        return results


# -- Module-level API --------------------------------------------------------

_classifier: Optional[NPClassifierLocal] = None


def _get_classifier(repo_root=None):
    global _classifier
    if _classifier is None:
        _classifier = NPClassifierLocal(repo_root=repo_root)
    return _classifier


def classify_batch(smiles_list, cache_dir=".", level="superclass",
                   repo_root=None, **kwargs):
    """Classify a list of SMILES.

    Input:
        smiles_list: list of SMILES strings.
        cache_dir: directory for cache file.
        level: 'pathway', 'superclass', or 'class_'.
        repo_root: path to NP-Classifier repo.
    Output:
        list of label strings (same length as input).
    """
    clf = _get_classifier(repo_root)
    cache_path = str(Path(cache_dir) / CACHE_FILENAME)
    cache = _load_cache(cache_path)

    results = ["Unknown"] * len(smiles_list)
    to_classify = []
    cached_hits = 0

    for i, smi in enumerate(smiles_list):
        if smi in cache:
            results[i] = cache[smi].get(level) or "Unknown"
            cached_hits += 1
        else:
            to_classify.append(i)

    print(f"  Cache hits: {cached_hits:,}, to classify: {len(to_classify):,}")

    if to_classify:
        # Batched inference instead of one-by-one
        batch_smiles = [smiles_list[idx] for idx in to_classify]
        batch_results = clf.classify_batch_raw(batch_smiles)

        classified, failed = 0, 0
        for j, idx in enumerate(to_classify):
            smi = smiles_list[idx]
            result = batch_results[j]
            if "error" in result:
                failed += 1
            else:
                cache[smi] = result
                results[idx] = result.get(level) or "Unknown"
                classified += 1

        print(f"  NPClassifier: {classified:,} ok, {failed:,} failed, {cached_hits:,} cached")

    _save_cache(cache, cache_path)
    return results


def classify_batch_full(smiles_list, cache_dir=".", repo_root=None, **kwargs):
    """Classify SMILES and return full distributions.

    Input:
        smiles_list: list of SMILES strings.
        cache_dir: directory for cache file.
        repo_root: path to NP-Classifier repo.
    Output:
        dict with pathway_distribution, superclass_distribution,
        class_distribution, per_molecule.
    """
    clf = _get_classifier(repo_root)
    cache_path = str(Path(cache_dir) / CACHE_FILENAME)
    cache = _load_cache(cache_path)

    per_molecule = [None] * len(smiles_list)
    to_classify = []
    cached_hits = 0

    for i, smi in enumerate(smiles_list):
        if smi in cache:
            per_molecule[i] = cache[smi]
            cached_hits += 1
        else:
            to_classify.append(i)

    print(f"  Cache hits: {cached_hits:,}, to classify: {len(to_classify):,}")

    if to_classify:
        # Batched inference
        batch_smiles = [smiles_list[idx] for idx in to_classify]
        batch_results = clf.classify_batch_raw(batch_smiles)

        for j, idx in enumerate(to_classify):
            result = batch_results[j]
            per_molecule[idx] = result
            if "error" not in result:
                cache[smiles_list[idx]] = result

    _save_cache(cache, cache_path)

    pw_dist, sc_dist, cl_dist = Counter(), Counter(), Counter()
    for entry in per_molecule:
        if entry is None or "error" in entry:
            continue
        if entry.get("pathway"):
            pw_dist[entry["pathway"]] += 1
        if entry.get("superclass"):
            sc_dist[entry["superclass"]] += 1
        if entry.get("class_"):
            cl_dist[entry["class_"]] += 1

    return {
        "pathway_distribution": dict(pw_dist),
        "superclass_distribution": dict(sc_dist),
        "class_distribution": dict(cl_dist),
        "per_molecule": per_molecule,
    }
