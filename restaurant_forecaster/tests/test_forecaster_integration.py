from pathlib import Path

import pandas as pd
import pytest

from src import forecaster


@pytest.fixture
def raw_csv(tmp_path: Path, synthetic_daily_df) -> Path:
    path = tmp_path / "history.csv"
    synthetic_daily_df.to_csv(path, index=False)
    return path


def test_train_and_forecast_end_to_end(tmp_path: Path, raw_csv: Path):
    model_dir = tmp_path / "models"
    db_path = tmp_path / "forecaster.db"

    models = forecaster.train_all_targets(raw_csv, fetch_weather=False, model_dir=model_dir, db_path=db_path)
    assert set(models.keys()) == {"sales", "guest_count"}

    results = forecaster.generate_forecast(
        pd.Timestamp("2025-04-01").date(),
        pd.Timestamp("2025-04-07").date(),
        model_dir=model_dir,
        fetch_weather=False,
        db_path=db_path,
        persist=True,
    )

    assert len(results) == 7
    for day in results:
        assert day.projected_sales >= 0
        assert day.projected_guests >= 0
        assert day.sales_low <= day.projected_sales <= day.sales_high


def test_generate_forecast_without_trained_model_raises(tmp_path: Path):
    with pytest.raises(RuntimeError):
        forecaster.generate_forecast(
            pd.Timestamp("2025-04-01").date(),
            pd.Timestamp("2025-04-02").date(),
            model_dir=tmp_path / "empty_models",
            fetch_weather=False,
            db_path=tmp_path / "forecaster.db",
        )
