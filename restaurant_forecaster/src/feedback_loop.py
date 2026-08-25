"""Actuals logging, drift detection, and adaptive retraining.

Implements the closed feedback loop described in the spec:

1. ``log_actuals`` records an actual outcome, diffs it against the stored
   forecast for that date (absolute error, percentage error, bias), and
   persists the result to the error log.
2. ``detect_drift`` inspects the trailing 7-14 days of error-log bias and
   flags sustained over/under-prediction beyond ``DRIFT_BIAS_THRESHOLD_PCT``.
3. ``maybe_retrain`` retrains the ensemble (recency-weighted, including the
   newly logged actuals) whenever drift is detected, and registers the new
   model version.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config, data_loader, evaluation, feature_engineering, storage
from src.models.ensemble import EnsembleForecastModel

logger = logging.getLogger(__name__)

TARGETS = ["sales", "guest_count"]


@dataclass
class ActualsLogResult:
    date: str
    target: str
    actual: float
    predicted: Optional[float]
    abs_error: Optional[float]
    pct_error: Optional[float]
    bias: Optional[float]
    had_forecast: bool


@dataclass
class DriftReport:
    target: str
    lookback_days: int
    mean_bias_pct: float
    drift_detected: bool
    n_observations: int


def log_actuals(
    date_str: str,
    sales: float,
    guest_count: int,
    db_path: Path = config.DB_PATH,
) -> list[ActualsLogResult]:
    """Record actuals for a date and diff them against any stored forecast."""
    now = datetime.now(timezone.utc).isoformat()
    storage.save_actuals(date_str, sales, guest_count, now, db_path=db_path)

    actual_values = {"sales": sales, "guest_count": guest_count}
    results = []

    for target in TARGETS:
        forecast = storage.get_latest_forecast_for_date(target, date_str, db_path=db_path)
        actual = actual_values[target]

        if forecast is None:
            results.append(
                ActualsLogResult(
                    date=date_str, target=target, actual=actual, predicted=None,
                    abs_error=None, pct_error=None, bias=None, had_forecast=False,
                )
            )
            continue

        predicted = forecast["p50"]
        abs_error = abs(actual - predicted)
        pct_error = (abs_error / actual) if actual else 0.0
        bias = predicted - actual  # positive => over-predicted

        storage.save_error_log_entry(
            date_str=date_str,
            target=target,
            actual=actual,
            predicted=predicted,
            abs_error=abs_error,
            pct_error=pct_error,
            bias=bias,
            model_version=forecast["model_version"],
            logged_at=now,
            db_path=db_path,
        )
        results.append(
            ActualsLogResult(
                date=date_str, target=target, actual=actual, predicted=predicted,
                abs_error=abs_error, pct_error=pct_error, bias=bias, had_forecast=True,
            )
        )

    return results


def detect_drift(
    target: str,
    lookback_days: int = config.DRIFT_LOOKBACK_DAYS,
    min_lookback_days: int = config.DRIFT_MIN_LOOKBACK_DAYS,
    bias_threshold_pct: float = config.DRIFT_BIAS_THRESHOLD_PCT,
    db_path: Path = config.DB_PATH,
) -> DriftReport:
    """Detect sustained bias drift over the trailing lookback window."""
    error_log = storage.load_error_log(target, db_path=db_path)
    if error_log.empty:
        return DriftReport(target=target, lookback_days=lookback_days, mean_bias_pct=0.0, drift_detected=False, n_observations=0)

    as_of = error_log["date"].max()
    cutoff = as_of - pd.Timedelta(days=lookback_days)
    window = error_log[error_log["date"] > cutoff]

    if len(window) < min_lookback_days:
        return DriftReport(
            target=target, lookback_days=lookback_days, mean_bias_pct=0.0,
            drift_detected=False, n_observations=len(window),
        )

    denom = window["actual"].abs().sum()
    mean_bias_pct = float(window["bias"].sum() / denom) if denom else 0.0
    drift_detected = abs(mean_bias_pct) >= bias_threshold_pct

    return DriftReport(
        target=target, lookback_days=lookback_days, mean_bias_pct=mean_bias_pct,
        drift_detected=drift_detected, n_observations=len(window),
    )


def retrain_model(
    raw_data_path: Path,
    target: str,
    fetch_weather: bool = True,
    trigger_reason: str = "manual",
    model_dir: Path = config.MODEL_REGISTRY_DIR,
    db_path: Path = config.DB_PATH,
) -> EnsembleForecastModel:
    """Retrain the ensemble for ``target`` on the latest raw data + logged actuals."""
    df = data_loader.load_and_clean(raw_data_path)
    actuals = storage.load_actuals(db_path=db_path)

    if not actuals.empty:
        actuals_renamed = actuals.rename(columns={"guest_count": "guest_count"})[["date", "sales", "guest_count"]]
        df = pd.concat([df[["date", "sales", "guest_count"]], actuals_renamed], ignore_index=True)
        df = df.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        df = data_loader.clean_data(df, fill_missing_dates=False, clip_outliers=True)

    training_frame = feature_engineering.build_training_frame(df, fetch_weather=fetch_weather)

    model = EnsembleForecastModel(target=target, feature_columns=feature_engineering.ALL_FEATURE_COLUMNS)
    model.fit(training_frame, sample_weight=training_frame["sample_weight"].to_numpy())
    artifact_path = model.save(model_dir)

    storage.register_model(
        target=target,
        model_version=model.metadata.model_version,
        trained_at=model.metadata.trained_at,
        training_rows=model.metadata.training_rows,
        trigger_reason=trigger_reason,
        artifact_path=str(artifact_path),
        db_path=db_path,
    )
    logger.info("Retrained %s model -> version %s (%s)", target, model.metadata.model_version, trigger_reason)
    return model


def maybe_retrain(
    raw_data_path: Path,
    target: str,
    fetch_weather: bool = True,
    model_dir: Path = config.MODEL_REGISTRY_DIR,
    db_path: Path = config.DB_PATH,
) -> tuple[DriftReport, Optional[EnsembleForecastModel]]:
    """Check for drift and retrain automatically if it's detected."""
    report = detect_drift(target, db_path=db_path)
    if not report.drift_detected:
        return report, None

    logger.info(
        "Drift detected for %s: mean_bias_pct=%.3f over trailing %d days (%d obs) -- retraining",
        target, report.mean_bias_pct, report.lookback_days, report.n_observations,
    )
    model = retrain_model(
        raw_data_path, target, fetch_weather=fetch_weather,
        trigger_reason=f"drift(bias={report.mean_bias_pct:.3f})",
        model_dir=model_dir, db_path=db_path,
    )
    return report, model


def trailing_accuracy_summary(target: str, db_path: Path = config.DB_PATH) -> dict:
    """Trailing 7/30/90-day MAPE/WAPE, used in push-projections payloads."""
    error_log = storage.load_error_log(target, db_path=db_path)
    if error_log.empty:
        empty = evaluation.AccuracyMetrics(0, 0, 0, 0, 0).as_dict()
        return {"trailing_7d": empty, "trailing_30d": empty, "trailing_90d": empty}

    as_of = error_log["date"].max()
    return {
        "trailing_7d": evaluation.trailing_window_metrics(error_log, 7, as_of).as_dict(),
        "trailing_30d": evaluation.trailing_window_metrics(error_log, 30, as_of).as_dict(),
        "trailing_90d": evaluation.trailing_window_metrics(error_log, 90, as_of).as_dict(),
    }
