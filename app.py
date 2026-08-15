"""
app.py — FastAPI entrypoint. Routing only; no prediction logic lives here.

All preprocessing + inference goes through src.predict.predict_premium(),
the single authoritative prediction path also used by model_training/train.py.

Environment is selected via the APP_ENV variable (dev/staging/prod),
defaulting to dev. Config values (sanity-check bounds, model version,
log level) come from conf/<env>.yaml — never hardcoded here.
"""
import os
from pathlib import Path

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException

from src.model_loader import artifacts_status, load_metadata, load_pipeline
from src.predict import predict_premium
from src.schemas import (
    BatchPolicyInput,
    BatchPredictionResponse,
    HealthResponse,
    PolicyInput,
    PredictionResponse,
)

ROOT = Path(__file__).resolve().parent
APP_ENV = os.getenv("APP_ENV", "dev")


def load_config(env: str) -> dict:
    config_path = ROOT / "conf" / f"{env}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"No config found for environment '{env}' at {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


CONFIG = load_config(APP_ENV)

app = FastAPI(
    title="Motor Insurance Premium Prediction API",
    version=CONFIG["model"]["expected_version"],
)


def _apply_sanity_check(predicted_premium: float) -> bool:
    """Returns True if the prediction falls outside configured bounds —
    a guardrail so wildly implausible predictions get flagged for review
    instead of silently quoted to a customer."""
    sc = CONFIG["sanity_check"]
    if not sc.get("enabled", False):
        return False
    return not (sc["min_premium"] <= predicted_premium <= sc["max_premium"])


@app.get("/health", response_model=HealthResponse)
def health():
    status = artifacts_status()
    metadata = load_metadata() if status["all_present"] else {}
    return HealthResponse(
        status="ok" if status["all_present"] else "degraded",
        environment=APP_ENV,
        model_version=metadata.get("metrics", {}).get("winner", "unknown"),
        artifacts=status,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(policy: PolicyInput):
    if not artifacts_status()["all_present"]:
        raise HTTPException(status_code=503, detail="Model artifacts not available. Run training first.")

    pipeline = load_pipeline()
    df = pd.DataFrame([policy.model_dump()])
    try:
        prediction = float(predict_premium(df, pipeline)[0])
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        predicted_premium=round(prediction, 2),
        model_version=CONFIG["model"]["expected_version"],
        flagged=_apply_sanity_check(prediction),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch: BatchPolicyInput):
    if not artifacts_status()["all_present"]:
        raise HTTPException(status_code=503, detail="Model artifacts not available. Run training first.")
    if not batch.records:
        raise HTTPException(status_code=422, detail="records cannot be empty")

    pipeline = load_pipeline()
    df = pd.DataFrame([r.model_dump() for r in batch.records])
    try:
        predictions = predict_premium(df, pipeline)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}")

    results = [
        PredictionResponse(
            predicted_premium=round(float(p), 2),
            model_version=CONFIG["model"]["expected_version"],
            flagged=_apply_sanity_check(float(p)),
        )
        for p in predictions
    ]
    return BatchPredictionResponse(predictions=results)