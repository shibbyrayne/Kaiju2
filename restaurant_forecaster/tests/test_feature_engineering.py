from datetime import date

import numpy as np
import pandas as pd

from src import config, feature_engineering


def test_recency_weights_increase_toward_present():
    dates = pd.Series(pd.date_range("2024-01-01", periods=200, freq="D"))
    weights = feature_engineering.compute_recency_weights(dates)
    assert weights[-1] > weights[0]
    assert np.all(np.diff(weights) >= -1e-9)  # monotonically non-decreasing


def test_recency_weights_respect_multiplier_bounds():
    dates = pd.Series(pd.date_range("2024-01-01", periods=200, freq="D"))
    weights = feature_engineering.compute_recency_weights(
        dates, min_multiplier=config.RECENCY_MIN_MULTIPLIER, max_multiplier=config.RECENCY_MAX_MULTIPLIER
    )
    assert weights.min() >= config.RECENCY_MIN_MULTIPLIER - 1e-6
    assert weights.max() <= config.RECENCY_MAX_MULTIPLIER + 1e-6


def test_recency_weights_recent_window_much_higher_than_old():
    dates = pd.Series(pd.date_range("2024-01-01", periods=365, freq="D"))
    weights = feature_engineering.compute_recency_weights(dates)
    recent_avg = weights[-90:].mean()
    old_avg = weights[:90].mean()
    assert recent_avg > 2.0 * old_avg


def test_build_future_frame_has_expected_columns():
    frame = feature_engineering.build_future_frame(date(2024, 3, 10), date(2024, 3, 12), fetch_weather=False)
    assert len(frame) == 3
    for col in feature_engineering.CALENDAR_FEATURE_COLUMNS:
        assert col in frame.columns
    for col in feature_engineering.WEATHER_FEATURE_COLUMNS:
        assert col in frame.columns


def test_top_drivers_for_sxsw_day():
    frame = feature_engineering.build_future_frame(date(2024, 3, 12), date(2024, 3, 12), fetch_weather=False)
    drivers = feature_engineering.top_drivers_for_row(frame.iloc[0])
    assert any("Sxsw" in d["description"] for d in drivers)
