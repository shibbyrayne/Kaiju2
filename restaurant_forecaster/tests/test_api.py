"""Integration tests for the FastAPI dashboard/JSON API (src/api.py).

These exercise the app through TestClient end-to-end (train -> forecast ->
log-actuals -> accuracy -> retrain). api.py intentionally does not
parameterize model_dir/db_path per request -- it's built to be hosted as a
single instance backed by the paths in src.config -- so these tests clean
out those real directories before and after running rather than trying to
inject tmp paths.
"""
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src import config


def _clear_processed_state():
    if config.DB_PATH.exists():
        config.DB_PATH.unlink()
    if config.MODEL_REGISTRY_DIR.exists():
        for f in config.MODEL_REGISTRY_DIR.glob("*"):
            if f.name != ".gitkeep":
                f.unlink()


@pytest.fixture
def client():
    _clear_processed_state()
    from src.api import app
    import src.api as api_module

    api_module._last_training_data_path = None

    with TestClient(app) as c:
        yield c
    _clear_processed_state()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_dashboard_served_at_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Austin Restaurant Forecaster" in resp.text


def test_forecast_before_training_returns_409(client):
    resp = client.post(
        "/api/forecast", json={"start": "2026-10-01", "end": "2026-10-02", "fetch_weather": False}
    )
    assert resp.status_code == 409


def test_full_dashboard_flow(client):
    status = client.get("/api/status").json()
    assert status["models"]["sales"]["trained"] is False
    assert status["sample_data_available"] is True

    train_resp = client.post("/api/train", params={"use_sample_data": True, "fetch_weather": False})
    assert train_resp.status_code == 200
    train_body = train_resp.json()
    assert "sales" in train_body and "guest_count" in train_body
    assert train_body["sales"]["training_rows"] > 0

    status = client.get("/api/status").json()
    assert status["models"]["sales"]["trained"] is True
    assert status["has_training_data_on_file"] is True

    forecast_resp = client.post(
        "/api/forecast", json={"start": "2026-10-01", "end": "2026-10-03", "fetch_weather": False}
    )
    assert forecast_resp.status_code == 200
    forecast_days = forecast_resp.json()["forecast"]
    assert len(forecast_days) == 3
    assert forecast_days[0]["projected_sales"] >= 0
    assert "conf_interval_low" in forecast_days[0]

    actuals_resp = client.post(
        "/api/log-actuals",
        json={"date": "2026-10-01", "sales": forecast_days[0]["projected_sales"], "guests": forecast_days[0]["projected_guests"]},
    )
    assert actuals_resp.status_code == 200
    actuals_body = actuals_resp.json()
    assert len(actuals_body["results"]) == 2
    assert actuals_body["results"][0]["had_forecast"] is True

    accuracy_resp = client.get("/api/accuracy/sales")
    assert accuracy_resp.status_code == 200
    assert accuracy_resp.json()["trailing_7d"]["n_observations"] == 1

    retrain_resp = client.post("/api/retrain")
    assert retrain_resp.status_code == 200
    assert retrain_resp.json()["sales"]["training_rows"] > 0


def test_accuracy_unknown_target_404(client):
    resp = client.get("/api/accuracy/not_a_real_target")
    assert resp.status_code == 404


def test_retrain_without_prior_training_returns_400(client):
    resp = client.post("/api/retrain")
    assert resp.status_code == 400


def test_forecast_end_before_start_returns_400(client):
    resp = client.post(
        "/api/forecast", json={"start": "2026-10-05", "end": "2026-10-01", "fetch_weather": False}
    )
    assert resp.status_code == 400


def test_push_projections_reports_failure_on_bad_endpoint(client, monkeypatch):
    client.post("/api/train", params={"use_sample_data": True, "fetch_weather": False})

    import src.api as api_module

    def fake_push(endpoint, api_key, daily_projections, accuracy_metrics, model_version="x"):
        from src.exporter import PushResult

        return PushResult(success=False, status_code=None, endpoint=endpoint, error="simulated failure")

    monkeypatch.setattr(api_module.exporter, "push_projections", fake_push)

    resp = client.post(
        "/api/push-projections",
        json={
            "endpoint": "https://example.invalid/webhook",
            "api_key": "test-key",
            "start": "2026-10-01",
            "end": "2026-10-02",
            "fetch_weather": False,
        },
    )
    assert resp.status_code == 502
