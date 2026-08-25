"""LightGBM regressor with quantile heads for P10/P50/P90 forecasts.

Captures non-linear interactions between weather, event proximity, and
calendar features. Three separate boosters (one per quantile) are trained
using LightGBM's native ``quantile`` objective, all using the same recency
sample weights so recent observations dominate the fit.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only when lightgbm is absent
    LIGHTGBM_AVAILABLE = False

from src import config
from src.models.base import BaseForecastModel

logger = logging.getLogger(__name__)

QUANTILES = {"p10": 0.10, "p50": 0.50, "p90": 0.90}


class _SklearnGBMFallback:
    """GradientBoostingRegressor-based fallback used when lightgbm isn't installed."""

    def __init__(self, alpha: float, random_state: int):
        from sklearn.ensemble import GradientBoostingRegressor

        self.model = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LGBMQuantileModel(BaseForecastModel):
    """Ensemble of quantile-objective boosters for a single target column."""

    name = "lgbm_quantile"

    def __init__(self, params: Optional[dict] = None, random_state: int = config.RANDOM_SEED):
        self.params = dict(params or config.LGBM_PARAMS)
        self.random_state = random_state
        self.boosters: dict[str, object] = {}
        self.feature_columns: list[str] = []

    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: list[str],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "LGBMQuantileModel":
        self.feature_columns = list(feature_columns)
        X = df[self.feature_columns].fillna(0.0)
        y = df[target].astype(float)

        for q_name, alpha in QUANTILES.items():
            if LIGHTGBM_AVAILABLE:
                params = dict(self.params)
                params.update({"objective": "quantile", "alpha": alpha})
                params.pop("metric", None)
                model = lgb.LGBMRegressor(**params)
                model.fit(X, y, sample_weight=sample_weight)
            else:
                logger.warning("lightgbm not installed; falling back to sklearn GradientBoostingRegressor")
                model = _SklearnGBMFallback(alpha=alpha, random_state=self.random_state).fit(
                    X, y, sample_weight=sample_weight
                )
            self.boosters[q_name] = model

        return self

    def predict(self, df: pd.DataFrame, feature_columns: Optional[list[str]] = None) -> np.ndarray:
        return self.predict_quantiles(df, feature_columns)["p50"].to_numpy()

    def predict_quantiles(self, df: pd.DataFrame, feature_columns: Optional[list[str]] = None) -> pd.DataFrame:
        cols = feature_columns or self.feature_columns
        X = df[cols].fillna(0.0)
        preds = {}
        for q_name, model in self.boosters.items():
            preds[q_name] = model.predict(X)
        result = pd.DataFrame(preds)
        # Enforce monotonicity: p10 <= p50 <= p90, since independently fit
        # quantile models can occasionally cross.
        result["p10"] = np.minimum(result["p10"], result["p50"])
        result["p90"] = np.maximum(result["p90"], result["p50"])
        return result.clip(lower=0.0)
