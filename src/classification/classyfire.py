"""ClassyFire superclass classification via REST API."""

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _classify_one(args):
    """Classify a single SMILES (for thread pool).

    Input:
        args: tuple of (index, smiles, cache, lock).
    Output:
        tuple of (index, inchikey, label).
    """
    idx, smi, cache, lock = args
    inchikey = smiles_to_inchikey(smi)
    if inchikey is None:
        return idx, None, "Unknown"

    with lock:
        if inchikey in cache:
            return idx, inchikey, cache[inchikey]

    label = query_entity(inchikey)
    result = label if label else "Unknown"

    with lock:
        cache[inchikey] = result

    return idx, inchikey, result


def classify_batch(smiles_list: List[str], cache_dir: str = ".",
                   delay: float = 0.2, n_workers: int = 32) -> List[str]:
    """Classify SMILES into ClassyFire superclasses via InChIKey lookup.

    Uses ThreadPoolExecutor for concurrent API requests.

    Input:
        smiles_list: list of SMILES strings.
        cache_dir: directory for classyfire_cache.json.
        delay: seconds between API calls (rate limiting).
        n_workers: number of concurrent threads (default 32).
    Output:
        list of superclass names (same length as input).
        Returns "Unknown" for failed classifications.
    """
    if not HAS_REQUESTS:
        print("  Warning: 'requests' not installed, skipping classification")
        return ["Unknown"] * len(smiles_list)

    cache_path = str(Path(cache_dir) / CACHE_FILENAME)
    cache = _load_cache(cache_path)
    lock = threading.Lock()

    # Separate cached vs uncached
    results = ["Unknown"] * len(smiles_list)
    to_query = []
    cached_hits = 0

    for i, smi in enumerate(smiles_list):
        inchikey = smiles_to_inchikey(smi)
        if inchikey is None:
            continue
        if inchikey in cache:
            results[i] = cache[inchikey]
            cached_hits += 1
        else:
            to_query.append((i, smi, cache, lock))

    print(f"  Cache hits: {cached_hits:,}, to query: {len(to_query):,}")

    if to_query:
        api_calls = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_classify_one, args): args[0]
                       for args in to_query}

            iterator = as_completed(futures)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(futures),
                                desc="ClassyFire")

            for future in iterator:
                idx, inchikey, label = future.result()
                results[idx] = label
                if label != "Unknown":
                    api_calls += 1
                else:
                    failed += 1

        print(f"  ClassyFire: {api_calls:,} classified, "
              f"{failed:,} failed, {cached_hits:,} cached")

    _save_cache(cache, cache_path)
    return results
