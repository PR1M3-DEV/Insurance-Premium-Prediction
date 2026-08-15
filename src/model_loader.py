"""
model_loader.py — loads the trained pipeline (artifacts/model.pkl) once and
exposes it, plus a health-check helper that confirms all required artifacts
are present and loadable.
"""
import json
from pathlib import Path
from functools import lru_cache

import joblib

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / 'artifacts'

REQUIRED_ARTIFACTS = ['model.pkl', 'metrics.json', 'params.json', 'feature_names.json']


@lru_cache(maxsize=1)
def load_pipeline():
    """Loads and caches the fitted sklearn Pipeline. Raises FileNotFoundError
    with a clear message if train.py hasn't been run yet."""
    model_path = ARTIFACTS_DIR / 'model.pkl'
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. Run "
            f"'python model_training/train.py' first."
        )
    return joblib.load(model_path)


@lru_cache(maxsize=1)
def load_metadata() -> dict:
    """Loads metrics.json + params.json + feature_names.json into one dict —
    used by /health and to report which model/version is currently serving."""
    metadata = {}
    for name in ['metrics', 'params', 'feature_names']:
        path = ARTIFACTS_DIR / f'{name}.json'
        if path.exists():
            with open(path) as f:
                metadata[name] = json.load(f)
    return metadata


def artifacts_status() -> dict:
    """Reports which required artifacts are present, for /health."""
    status = {}
    for filename in REQUIRED_ARTIFACTS:
        status[filename] = (ARTIFACTS_DIR / filename).exists()
    status['all_present'] = all(status.values())
    return status