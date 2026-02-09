"""ClassyFire superclass classification via REST API."""

import json
import time
from pathlib import Path
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

API_BASE = "http://classyfire.wishartlab.com"
CACHE_FILENAME = "classyfire_cache.json"


def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """Convert SMILES to InChIKey using RDKit.

    Input:
        smiles: SMILES string.
    Output:
        str InChIKey or None on failure.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        inchi = MolToInchi(mol)
        if inchi is None:
            return None
        return InchiToInchiKey(inchi)
    except Exception:
        return None


def query_entity(inchikey: str, timeout: int = 10) -> Optional[str]:
    """Query ClassyFire API for superclass by InChIKey.

    Input:
        inchikey: InChIKey string.
        timeout: request timeout in seconds.
    Output:
        str superclass name or None if not found.
    """
    if not HAS_REQUESTS:
        return None
    url = f"{API_BASE}/entities/{inchikey}.json"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            data = resp.json()
            sc = data.get("superclass")
            if sc and isinstance(sc, dict):
                return sc.get("name")
        return None
    except (requests.RequestException, json.JSONDecodeError, KeyError):
        return None


def _load_cache(path: str) -> dict:
    """Load classification cache from JSON file."""
    p = Path(path)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict, path: str):
    """Save classification cache to JSON file."""
    try:
        Path(path).write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    except OSError:
        pass


def classify_batch(smiles_list: List[str], cache_dir: str = ".",
                   delay: float = 0.2) -> List[str]:
    """Classify SMILES into ClassyFire superclasses via InChIKey lookup.

    Input:
        smiles_list: list of SMILES strings.
        cache_dir: directory for classyfire_cache.json.
        delay: seconds between API calls (rate limiting).
    Output:
        list of superclass names (same length as input).
        Returns "Unknown" for failed classifications.
    """
    if not HAS_REQUESTS:
        print("  Warning: 'requests' not installed, skipping classification")
        return ["Unknown"] * len(smiles_list)

    cache_path = str(Path(cache_dir) / CACHE_FILENAME)
    cache = _load_cache(cache_path)
    results = []
    api_calls = 0
    cached_hits = 0

    iterator = enumerate(smiles_list)
    if HAS_TQDM:
        iterator = tqdm(list(iterator), desc="ClassyFire")

    for i, smi in iterator:
        inchikey = smiles_to_inchikey(smi)
        if inchikey is None:
            results.append("Unknown")
            continue

        if inchikey in cache:
            results.append(cache[inchikey])
            cached_hits += 1
            continue

        superclass = query_entity(inchikey)
        label = superclass if superclass else "Unknown"
        cache[inchikey] = label
        results.append(label)
        api_calls += 1

        if api_calls % 100 == 0:
            _save_cache(cache, cache_path)

        time.sleep(delay)

    _save_cache(cache, cache_path)
    print(f"  ClassyFire: {api_calls} API calls, {cached_hits} cache hits, "
          f"{len(smiles_list) - api_calls - cached_hits} failed")
    return results
