from pathlib import Path

import pandas as pd
import pytest

from src import config, feedback_loop, storage


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"


def test_log_actuals_without_forecast_on_file(db_path: Path):
    results = feedback_loop.log_actuals("2024-05-01", sales=1000.0, guest_count=50, db_path=db_path)
    assert len(results) == 2
    assert all(not r.had_forecast for r in results)

    actuals = storage.load_actuals(db_path=db_path)
    assert len(actuals) == 1
    assert actuals.loc[0, "sales"] == 1000.0


def test_log_actuals_computes_error_against_stored_forecast(db_path: Path):
    forecast_df = pd.DataFrame({"date": [pd.Timestamp("2024-05-02")], "p10": [900.0], "p50": [1000.0], "p90": [1100.0]})
    storage.save_forecast(forecast_df, target="sales", model_version="v1", generated_at="2024-05-01T00:00:00", db_path=db_path)

    results = feedback_loop.log_actuals("2024-05-02", sales=1200.0, guest_count=60, db_path=db_path)
    sales_result = next(r for r in results if r.target == "sales")

    assert sales_result.had_forecast
    assert sales_result.predicted == 1000.0
    assert sales_result.abs_error == pytest.approx(200.0)
    assert sales_result.bias == pytest.approx(-200.0)  # predicted < actual => under-prediction


def test_detect_drift_flags_sustained_underprediction(db_path: Path):
    for i in range(10):
        d = pd.Timestamp("2024-06-01") + pd.Timedelta(days=i)
        forecast_df = pd.DataFrame({"date": [d], "p10": [800.0], "p50": [900.0], "p90": [1000.0]})
        storage.save_forecast(forecast_df, target="sales", model_version="v1", generated_at=d.isoformat(), db_path=db_path)
        feedback_loop.log_actuals(d.strftime("%Y-%m-%d"), sales=1200.0, guest_count=60, db_path=db_path)

    report = feedback_loop.detect_drift("sales", db_path=db_path)
    assert report.drift_detected
    assert report.mean_bias_pct < 0  # under-predicting


def test_detect_drift_no_drift_when_accurate(db_path: Path):
    for i in range(10):
        d = pd.Timestamp("2024-06-01") + pd.Timedelta(days=i)
        forecast_df = pd.DataFrame({"date": [d], "p10": [950.0], "p50": [1000.0], "p90": [1050.0]})
        storage.save_forecast(forecast_df, target="sales", model_version="v1", generated_at=d.isoformat(), db_path=db_path)
        feedback_loop.log_actuals(d.strftime("%Y-%m-%d"), sales=1000.0, guest_count=50, db_path=db_path)

    report = feedback_loop.detect_drift("sales", db_path=db_path)
    assert not report.drift_detected


def test_trailing_accuracy_summary_empty_when_no_data(db_path: Path):
    summary = feedback_loop.trailing_accuracy_summary("sales", db_path=db_path)
    assert summary["trailing_7d"]["n_observations"] == 0
