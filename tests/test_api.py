"""
test_api.py — tests /health, /predict, /predict/batch against the real
FastAPI app (via TestClient) and the real trained model. No mocking.

Requires artifacts/model.pkl to exist — run model_training/train.py first.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)

VALID_PAYLOAD = {
    "SEX": 0,
    "INSR_BEGIN": "08-AUG-13",
    "INSR_END": "07-AUG-14",
    "INSR_TYPE": 1202,
    "INSURED_VALUE": 519755.22,
    "PROD_YEAR": 2007,
    "SEATS_NUM": 4,
    "CARRYING_CAPACITY": 6,
    "TYPE_VEHICLE": "Pick-up",
    "CCM_TON": 3153,
    "MAKE": "NISSAN",
    "USAGE": "Own Goods",
}


class TestHealth:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_reports_artifacts_present(self):
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"
        assert body["artifacts"]["all_present"] is True
        assert body["model_version"] == "random_forest"


class TestPredict:
    def test_valid_payload_returns_200(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        assert response.status_code == 200

    def test_predicted_premium_is_positive_float(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        body = response.json()
        assert isinstance(body["predicted_premium"], float)
        assert body["predicted_premium"] > 0

    def test_matches_known_verified_prediction(self):
        """This exact payload was verified to predict ~5883.81 via
        predict_premium() directly (see project verification history) —
        confirms the API layer isn't silently transforming the input
        differently from the training/predict path."""
        response = client.post("/predict", json=VALID_PAYLOAD)
        body = response.json()
        assert abs(body["predicted_premium"] - 5883.81) < 1.0

    def test_missing_required_field_returns_422(self):
        bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "MAKE"}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_negative_insured_value_returns_422(self):
        bad_payload = {**VALID_PAYLOAD, "INSURED_VALUE": -100}
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_response_includes_model_version_and_flag(self):
        response = client.post("/predict", json=VALID_PAYLOAD)
        body = response.json()
        assert "model_version" in body
        assert "flagged" in body
        assert isinstance(body["flagged"], bool)


class TestPredictBatch:
    def test_batch_of_three_returns_three_predictions(self):
        response = client.post("/predict/batch", json={"records": [VALID_PAYLOAD] * 3})
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 3

    def test_empty_batch_returns_422(self):
        response = client.post("/predict/batch", json={"records": []})
        assert response.status_code == 422