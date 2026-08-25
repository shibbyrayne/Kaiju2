from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import feature_engineering
from src.models.ensemble import EnsembleForecastModel


@pytest.fixture
def training_frame(synthetic_daily_df):
    return feature_engineering.build_training_frame(synthetic_daily_df, fetch_weather=False)


def test_ensemble_fit_predict_quantiles_shape_and_monotonicity(training_frame):
    model = EnsembleForecastModel(target="sales", feature_columns=feature_engineering.ALL_FEATURE_COLUMNS)
    model.fit(training_frame, sample_weight=training_frame["sample_weight"].to_numpy())

    future = training_frame.tail(14).reset_index(drop=True)
    preds = model.predict_quantiles(future)

    assert list(preds.columns) == ["p10", "p50", "p90"]
    assert len(preds) == 14
    assert (preds["p10"] <= preds["p50"] + 1e-6).all()
    assert (preds["p50"] <= preds["p90"] + 1e-6).all()
    assert (preds >= 0).all().all()


def test_ensemble_raises_on_too_little_data():
    tiny = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=5, freq="D"), "sales": [100, 110, 90, 105, 95]}
    )
    tiny = feature_engineering.build_training_frame(tiny, fetch_weather=False)
    model = EnsembleForecastModel(target="sales", feature_columns=feature_engineering.ALL_FEATURE_COLUMNS)
    with pytest.raises(ValueError):
        model.fit(tiny)


def test_ensemble_save_and_load_roundtrip(training_frame, tmp_path: Path):
    model = EnsembleForecastModel(target="guest_count", feature_columns=feature_engineering.ALL_FEATURE_COLUMNS)
    model.fit(training_frame, sample_weight=training_frame["sample_weight"].to_numpy())
    model.save(tmp_path)

    loaded = EnsembleForecastModel.load_latest(tmp_path, "guest_count")
    assert loaded is not None
    future = training_frame.tail(5).reset_index(drop=True)
    preds = loaded.predict_quantiles(future)
    assert len(preds) == 5
