"""
predict.py — the single authoritative preprocessing + inference path.

Both model_training/train.py and app.py (via model_loader.py) import
clean_and_engineer() from here. Nothing else re-implements this logic.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

NUMERIC_FEATURES = [
    'LOG_INSURED_VALUE',
    'INSURED_VALUE_MISSING',
    'SEATS_NUM',
    'CARRYING_CAPACITY',
    'CARRYING_CAPACITY_MISSING',
    'CCM_TON',
    'DURATION_DAYS',
    'VEHICLE_AGE',
]

CATEGORICAL_FEATURES = [
    'SEX',
    'INSR_TYPE',
    'TYPE_VEHICLE',
    'USAGE',
    'MAKE_BUCKET',
]

# Known data-entry duplicates in MAKE (found during dataset inspection).
# Extend this dict if new duplicates are found in future data.
MAKE_ALIASES = {
    'BAJAJI': 'BAJAJ',
}

# Loaded lazily from artifacts/feature_names.json (written by train.py).
# This is the vocabulary of MAKE values seen often enough in training to get
# their own category; anything else collapses to 'OTHER' at both train and
# inference time so a rare/unseen make never breaks prediction.
_known_makes_cache = None


def _load_known_makes() -> set:
    global _known_makes_cache
    if _known_makes_cache is not None:
        return _known_makes_cache
    fn_path = ROOT / 'artifacts' / 'feature_names.json'
    if fn_path.exists():
        with open(fn_path) as f:
            data = json.load(f)
        _known_makes_cache = set(data.get('known_makes', []))
    else:
        _known_makes_cache = set()
    return _known_makes_cache


def clean_and_engineer(df: pd.DataFrame, known_makes: set = None) -> pd.DataFrame:
    """Row-preserving cleaning + feature engineering. Deterministic — no
    fitted statistics computed here (imputation of remaining NaNs happens
    inside the sklearn Pipeline stored in model.pkl, fit only on training
    data). This function is safe to call on a single inference row or the
    full training set and will behave identically either way.

    Does NOT drop rows. Row-dropping (dedup, invalid target) is a
    train-time-only concern and lives in model_training/train.py.
    """
    df = df.copy()

    # --- dates -> duration ---
    begin = pd.to_datetime(df['INSR_BEGIN'], format='%d-%b-%y', errors='coerce')
    end = pd.to_datetime(df['INSR_END'], format='%d-%b-%y', errors='coerce')
    df['DURATION_DAYS'] = (end - begin).dt.days

    # --- vehicle age at policy start ---
    prod_year = pd.to_numeric(df['PROD_YEAR'], errors='coerce')
    df['VEHICLE_AGE'] = begin.dt.year - prod_year
    df['VEHICLE_AGE'] = df['VEHICLE_AGE'].clip(lower=0, upper=60)

    # --- insured value: 0 is treated as "missing", flagged not imputed away.
    # Log-transformed because the raw value is extremely right-skewed (a
    # small number of policies have INSURED_VALUE in the hundreds of
    # millions vs a median in the low hundred-thousands) — left raw, this
    # produced an ill-conditioned design matrix for the linear model
    # (astronomical, unusable coefficients during model comparison).
    insured_value = pd.to_numeric(df['INSURED_VALUE'], errors='coerce').fillna(0)
    df['INSURED_VALUE_MISSING'] = (insured_value == 0).astype(int)
    df['LOG_INSURED_VALUE'] = np.log1p(insured_value)

    # --- carrying capacity: flag missingness, leave NaN for the pipeline imputer ---
    df['CARRYING_CAPACITY'] = pd.to_numeric(df['CARRYING_CAPACITY'], errors='coerce')
    df['CARRYING_CAPACITY_MISSING'] = df['CARRYING_CAPACITY'].isna().astype(int)

    df['SEATS_NUM'] = pd.to_numeric(df['SEATS_NUM'], errors='coerce')
    df['CCM_TON'] = pd.to_numeric(df['CCM_TON'], errors='coerce')

    # --- MAKE cleanup: normalize + fix known duplicates + bucket rare values ---
    make_clean = df['MAKE'].astype(str).str.strip().str.upper()
    make_clean = make_clean.replace(MAKE_ALIASES)

    if known_makes is None:
        known_makes = _load_known_makes()
    if known_makes:
        df['MAKE_BUCKET'] = make_clean.where(make_clean.isin(known_makes), 'OTHER')
    else:
        # train-time first pass: no vocabulary yet, caller (train.py) computes
        # the bucket itself from make_clean before this vocabulary exists
        df['MAKE_BUCKET'] = make_clean

    # --- categorical codes: keep as strings so encoders treat them as categories ---
    df['SEX'] = df['SEX'].astype(str)
    df['INSR_TYPE'] = df['INSR_TYPE'].astype(str)
    df['TYPE_VEHICLE'] = df['TYPE_VEHICLE'].astype(str)
    df['USAGE'] = df['USAGE'].astype(str)

    return df


def predict_premium(raw_df: pd.DataFrame, pipeline) -> np.ndarray:
    """Single entry point for turning raw input rows into premium predictions.
    `raw_df` must contain the original raw columns (same as the training CSV,
    minus PREMIUM). `pipeline` is the fitted sklearn Pipeline loaded from
    artifacts/model.pkl (via model_loader.py).

    Returns predictions on the ORIGINAL premium scale (currency units) —
    the log1p transform is inverted here, once, in the one place predictions
    are produced.
    """
    engineered = clean_and_engineer(raw_df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = engineered[feature_cols]
    log_pred = pipeline.predict(X)
    return np.expm1(log_pred)