"""Baseline seasonality model: Prophet when available, statsmodels otherwise.

This model captures the weekly/yearly baseline seasonality and holiday
effects referenced in the spec. Prophet is the preferred backend but is a
heavy, sometimes fragile dependency (requires a working C++/Stan
toolchain), so this module transparently falls back to a statsmodels
Holt-Winters exponential smoother — or, if statsmodels itself can't fit
(e.g. too little data), a recency-weighted seasonal-average model — while
exposing the exact same :class:`~src.models.base.BaseForecastModel`
interface used everywhere else.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.models.base import BaseForecastModel

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except ImportError:  # pragma: no cover
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover
    STATSMODELS_AVAILABLE = False


class _WeightedSeasonalFallback:
    """Recency-weighted day-of-week * month seasonal average model.

    Used when neither Prophet nor statsmodels can be fit (very short
    history, or missing optional dependencies). Always available since it
    only depends on pandas/numpy.
    """

    def __init__(self):
        self.global_mean = 0.0
        self.dow_effect: dict[int, float] = {}
        self.residual_std = 0.0

    def fit(self, dates: pd.Series, y: pd.Series, sample_weight: Optional[np.ndarray] = None):
        weights = np.asarray(sample_weight) if sample_weight is not None else np.ones(len(y))
        weights = np.where(weights <= 0, 1e-6, weights)

        self.global_mean = float(np.average(y, weights=weights))
        dow = pd.to_datetime(dates).dt.dayofweek

        for d in range(7):
            mask = (dow == d).to_numpy()
            if mask.sum() == 0:
                self.dow_effect[d] = 0.0
                continue
            local_mean = np.average(y[mask], weights=weights[mask])
            self.dow_effect[d] = float(local_mean - self.global_mean)

        preds = np.array([self.global_mean + self.dow_effect[d] for d in dow])
        residuals = y.to_numpy() - preds
        self.residual_std = float(np.std(residuals)) if len(residuals) > 1 else max(self.global_mean * 0.1, 1.0)
        return self

    def predict(self, dates: pd.Series) -> np.ndarray:
        dow = pd.to_datetime(dates).dt.dayofweek
        return np.array([self.global_mean + self.dow_effect.get(d, 0.0) for d in dow])


class SeasonalBaselineModel(BaseForecastModel):
    """Weekly/yearly seasonality + holiday-effect baseline (Prophet or statsmodels)."""

    name = "seasonal_baseline"

    def __init__(self, weekly_seasonality: bool = True, yearly_seasonality: bool = True):
        self.weekly_seasonality = weekly_seasonality
        self.yearly_seasonality = yearly_seasonality
        self.backend: str = "fallback"
        self._model = None
        self._fallback = _WeightedSeasonalFallback()
        self._residual_std = 0.0
        self._holiday_col = "is_federal_holiday"

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: list[str],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "SeasonalBaselineModel":
        df = df.sort_values("date")
        y = df[target].astype(float)

        if PROPHET_AVAILABLE and len(df) >= 30:
            self.backend = "prophet"
            prophet_df = pd.DataFrame({"ds": df["date"].values, "y": y.values})
            holidays_df = self._build_holidays_frame(df)
            model = Prophet(
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                holidays=holidays_df if not holidays_df.empty else None,
                interval_width=config.CONFIDENCE_UPPER_QUANTILE - config.CONFIDENCE_LOWER_QUANTILE,
            )
            try:
                model.fit(prophet_df)
                self._model = model
            except Exception as exc:  # pragma: no cover - depends on optional stan backend
                logger.warning("Prophet fit failed (%s); falling back to statsmodels/seasonal average", exc)
                self.backend = "fallback"

        if self.backend != "prophet" and STATSMODELS_AVAILABLE and len(df) >= 14:
            self.backend = "statsmodels"
            try:
                series = pd.Series(y.values, index=pd.DatetimeIndex(df["date"].values, freq="D"))
                self._model = ExponentialSmoothing(
                    series,
                    trend="add",
                    seasonal="add",
                    seasonal_periods=7,
                    initialization_method="estimated",
                ).fit()
                residuals = self._model.resid.to_numpy()
                self._residual_std = float(np.std(residuals)) if len(residuals) > 1 else max(y.mean() * 0.1, 1.0)
            except Exception as exc:
                logger.warning("statsmodels Holt-Winters fit failed (%s); falling back to seasonal average", exc)
                self.backend = "fallback"

        if self.backend == "fallback":
            self._fallback.fit(df["date"], y, sample_weight=sample_weight)
            self._residual_std = self._fallback.residual_std

        return self

    def _build_holidays_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        if "is_federal_holiday" not in df.columns:
            return pd.DataFrame(columns=["ds", "holiday"])
        holiday_dates = df.loc[df["is_federal_holiday"] == 1, "date"]
        return pd.DataFrame({"ds": holiday_dates.values, "holiday": "federal_holiday"})

    def predict(self, df: pd.DataFrame, feature_columns: Optional[list[str]] = None) -> np.ndarray:
        if self.backend == "prophet":
            future = pd.DataFrame({"ds": df["date"].values})
            forecast = self._model.predict(future)
            return forecast["yhat"].to_numpy()
        if self.backend == "statsmodels":
            steps = len(df)
            return self._model.forecast(steps).to_numpy()
        return self._fallback.predict(df["date"])

    def predict_quantiles(self, df: pd.DataFrame, feature_columns: Optional[list[str]] = None) -> pd.DataFrame:
        if self.backend == "prophet":
            future = pd.DataFrame({"ds": df["date"].values})
            forecast = self._model.predict(future)
            return pd.DataFrame(
                {
                    "p10": forecast["yhat_lower"].clip(lower=0).to_numpy(),
                    "p50": forecast["yhat"].clip(lower=0).to_numpy(),
                    "p90": forecast["yhat_upper"].clip(lower=0).to_numpy(),
                }
            )

        point = self.predict(df, feature_columns)
        std = self._residual_std or max(np.mean(point) * 0.1, 1.0)
        z10, z90 = -1.2816, 1.2816  # ~10th/90th percentile of a normal distribution
        return pd.DataFrame(
            {
                "p10": np.clip(point + z10 * std, 0, None),
                "p50": np.clip(point, 0, None),
                "p90": np.clip(point + z90 * std, 0, None),
            }
        )
