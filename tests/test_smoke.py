"""
test_smoke.py — hits a live deployed endpoint. Used in CI/CD after a real
deployment. Skips gracefully (not a failure) if no endpoint is configured or
reachable, since there's often nothing deployed yet during local dev.

Configure the target with the SMOKE_TEST_URL environment variable, e.g.:
    $env:SMOKE_TEST_URL = "http://127.0.0.1:8000"     (PowerShell, local)
    SMOKE_TEST_URL=https://your-deployed-host          (CI/CD)
"""
import os

import pytest
import requests

BASE_URL = os.getenv("SMOKE_TEST_URL")


def _skip_if_not_configured():
    if not BASE_URL:
        pytest.skip("SMOKE_TEST_URL not set — no live endpoint to smoke test against")


def _skip_if_unreachable():
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.exceptions.RequestException:
        pytest.skip(f"Could not reach {BASE_URL} — nothing deployed there right now")


class TestSmoke:
    def test_health_endpoint_live(self):
        _skip_if_not_configured()
        _skip_if_unreachable()
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_endpoint_live(self):
        _skip_if_not_configured()
        _skip_if_unreachable()
        payload = {
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
        response = requests.post(f"{BASE_URL}/predict", json=payload, timeout=5)
        assert response.status_code == 200
        assert response.json()["predicted_premium"] > 0