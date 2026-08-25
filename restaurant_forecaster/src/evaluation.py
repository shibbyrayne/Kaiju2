"""Forecast accuracy metrics: WAPE, MAPE, bias, and interval coverage."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AccuracyMetrics:
    wape: float
    mape: float
    bias_pct: float
    mean_absolute_error: float
    n_observations: int

    @property
    def accuracy_pct(self) -> float:
        """1 - WAPE, expressed as a percentage (spec's '90%+ accuracy')."""
        return max(0.0, 1.0 - self.wape) * 100

    def as_dict(self) -> dict:
        return {
            "wape": round(self.wape, 4),
            "mape": round(self.mape, 4),
            "bias_pct": round(self.bias_pct, 4),
            "mean_absolute_error": round(self.mean_absolute_error, 2),
            "accuracy_pct": round(self.accuracy_pct, 2),
            "n_observations": self.n_observations,
        }


def compute_wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted Absolute Percentage Error: sum(|error|) / sum(|actual|)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0 if np.sum(np.abs(predicted)) == 0 else 1.0
    return float(np.sum(np.abs(actual - predicted)) / denom)


def compute_mape(actual: np.ndarray, predicted: np.ndarray, epsilon: float = 1e-6) -> float:
    """Mean Absolute Percentage Error, ignoring rows where actual == 0."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = np.abs(actual) > epsilon
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])))


def compute_bias_pct(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Signed average percentage bias: positive => over-predicting, negative => under-predicting."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return 0.0
    return float(np.sum(predicted - actual) / denom)


def compute_accuracy_metrics(actual: np.ndarray, predicted: np.ndarray) -> AccuracyMetrics:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return AccuracyMetrics(
        wape=compute_wape(actual, predicted),
        mape=compute_mape(actual, predicted),
        bias_pct=compute_bias_pct(actual, predicted),
        mean_absolute_error=float(np.mean(np.abs(actual - predicted))) if len(actual) else 0.0,
        n_observations=len(actual),
    )


def compute_interval_coverage(actual: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    """Fraction of actuals falling within [low, high]."""
    actual = np.asarray(actual, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    if len(actual) == 0:
        return 0.0
    within = (actual >= low) & (actual <= high)
    return float(np.mean(within))


def trailing_window_metrics(error_log: pd.DataFrame, window_days: int, as_of: pd.Timestamp) -> AccuracyMetrics:
    """Compute accuracy metrics for the trailing N days of an error log DataFrame.

    ``error_log`` must have ``date``, ``actual``, and ``predicted`` columns.
    """
    cutoff = as_of - pd.Timedelta(days=window_days)
    window = error_log[(error_log["date"] > cutoff) & (error_log["date"] <= as_of)]
    if window.empty:
        return AccuracyMetrics(wape=0.0, mape=0.0, bias_pct=0.0, mean_absolute_error=0.0, n_observations=0)
    return compute_accuracy_metrics(window["actual"].to_numpy(), window["predicted"].to_numpy())
