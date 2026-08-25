"""Shared interfaces and prediction data structures for forecasting models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class QuantilePrediction:
    """Point + interval forecast for a single target on a single date."""

    date: pd.Timestamp
    p10: float
    p50: float
    p90: float

    def as_dict(self) -> dict:
        return {
            "date": self.date.strftime("%Y-%m-%d"),
            "p10": round(float(self.p10), 2),
            "p50": round(float(self.p50), 2),
            "p90": round(float(self.p90), 2),
        }


@dataclass
class ModelMetadata:
    """Bookkeeping attached to every trained model artifact."""

    model_name: str
    model_version: str
    trained_at: str
    target: str
    training_rows: int
    feature_columns: list[str] = field(default_factory=list)


class BaseForecastModel(ABC):
    """Common interface implemented by every model in the ensemble."""

    name: str = "base"

    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        target: str,
        feature_columns: list[str],
        sample_weight: Optional[np.ndarray] = None,
    ) -> "BaseForecastModel":
        """Fit the model on a training frame that contains ``target`` and features."""

    @abstractmethod
    def predict(self, df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
        """Return point predictions for each row of ``df``."""

    def predict_quantiles(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        """Return a DataFrame with p10/p50/p90 columns. Default: point +/- 0."""
        preds = self.predict(df, feature_columns)
        return pd.DataFrame({"p10": preds, "p50": preds, "p90": preds})
