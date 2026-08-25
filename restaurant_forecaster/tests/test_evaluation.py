import numpy as np
import pandas as pd
import pytest

from src import evaluation


def test_wape_perfect_prediction():
    actual = np.array([100.0, 200.0, 300.0])
    assert evaluation.compute_wape(actual, actual) == 0.0


def test_wape_known_value():
    actual = np.array([100.0, 100.0])
    predicted = np.array([110.0, 90.0])
    # sum(|error|) = 20, sum(|actual|) = 200 -> WAPE = 0.1
    assert evaluation.compute_wape(actual, predicted) == pytest.approx(0.1)


def test_mape_ignores_zero_actuals():
    actual = np.array([0.0, 100.0])
    predicted = np.array([50.0, 110.0])
    mape = evaluation.compute_mape(actual, predicted)
    assert mape == pytest.approx(0.1)


def test_bias_pct_sign():
    actual = np.array([100.0, 100.0])
    over_predicted = np.array([120.0, 120.0])
    under_predicted = np.array([80.0, 80.0])
    assert evaluation.compute_bias_pct(actual, over_predicted) > 0
    assert evaluation.compute_bias_pct(actual, under_predicted) < 0


def test_accuracy_metrics_accuracy_pct():
    actual = np.array([100.0] * 10)
    predicted = np.array([95.0] * 10)
    metrics = evaluation.compute_accuracy_metrics(actual, predicted)
    assert metrics.wape == pytest.approx(0.05)
    assert metrics.accuracy_pct == pytest.approx(95.0)


def test_interval_coverage():
    actual = np.array([10, 20, 30])
    low = np.array([5, 25, 25])
    high = np.array([15, 30, 35])
    coverage = evaluation.compute_interval_coverage(actual, low, high)
    assert coverage == pytest.approx(2 / 3)


def test_trailing_window_metrics():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    df = pd.DataFrame(
        {
            "date": dates,
            "actual": [100.0] * 20,
            "predicted": [90.0] * 20,
        }
    )
    as_of = dates[-1]
    metrics = evaluation.trailing_window_metrics(df, window_days=7, as_of=as_of)
    assert metrics.n_observations == 7
    assert metrics.wape == pytest.approx(0.1)
