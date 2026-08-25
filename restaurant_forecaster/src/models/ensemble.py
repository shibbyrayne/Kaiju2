"""Ensemble combining the LightGBM quantile model and the seasonal baseline.

Point predictions are a weighted blend (``ENSEMBLE_LGBM_WEIGHT``) of the two
backends' P50 forecasts. Interval width is blended the same way, then
widened slightly to account for combining two imperfectly correlated
estimators.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from src import config
from src.models.base import ModelMetadata
from src.models.lgbm_model import LGBMQuantileModel
from src.models.prophet_model import SeasonalBaselineModel

MODEL_VERSION_FORMAT = "%Y%m%dT%H%M%S"


class EnsembleForecastModel:
    """Trains + blends an LGBM quantile model and a seasonal baseline model for one target."""

    def __init__(
        self,
        target: str,
        feature_columns: list[str],
        lgbm_weight: float = config.ENSEMBLE_LGBM_WEIGHT,
    ):
        self.target = target
        self.feature_columns = list(feature_columns)
        self.lgbm_weight = lgbm_weight
        self.lgbm_model = LGBMQuantileModel()
        self.seasonal_model = SeasonalBaselineModel()
        self.metadata: Optional[ModelMetadata] = None

    def fit(self, df: pd.DataFrame, sample_weight: Optional[np.ndarray] = None) -> "EnsembleForecastModel":
        if len(df) < config.MIN_TRAINING_ROWS:
            raise ValueError(
                f"Need at least {config.MIN_TRAINING_ROWS} rows to train, got {len(df)}."
            )

        self.lgbm_model.fit(df, self.target, self.feature_columns, sample_weight=sample_weight)
        self.seasonal_model.fit(df, self.target, self.feature_columns, sample_weight=sample_weight)

        self.metadata = ModelMetadata(
            model_name="ensemble",
            model_version=datetime.utcnow().strftime(MODEL_VERSION_FORMAT),
            trained_at=datetime.utcnow().isoformat(),
            target=self.target,
            training_rows=len(df),
            feature_columns=self.feature_columns,
        )
        return self

    def predict_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        lgbm_q = self.lgbm_model.predict_quantiles(df, self.feature_columns)
        seasonal_q = self.seasonal_model.predict_quantiles(df, self.feature_columns)

        w = self.lgbm_weight
        blended = pd.DataFrame(
            {
                "p10": w * lgbm_q["p10"] + (1 - w) * seasonal_q["p10"],
                "p50": w * lgbm_q["p50"] + (1 - w) * seasonal_q["p50"],
                "p90": w * lgbm_q["p90"] + (1 - w) * seasonal_q["p90"],
            }
        )
        # Widen the band slightly (~8%) to reflect blended-model uncertainty
        # rather than naively averaging two possibly-correlated intervals.
        median = blended["p50"]
        blended["p10"] = np.clip(median - (median - blended["p10"]) * 1.08, 0, None)
        blended["p90"] = np.clip(median + (blended["p90"] - median) * 1.08, 0, None)
        blended["p10"] = np.minimum(blended["p10"], blended["p50"])
        blended["p90"] = np.maximum(blended["p90"], blended["p50"])
        return blended.reset_index(drop=True)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_quantiles(df)["p50"].to_numpy()

    def save(self, directory: Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        version = self.metadata.model_version if self.metadata else datetime.utcnow().strftime(MODEL_VERSION_FORMAT)
        path = directory / f"{self.target}_{version}.joblib"
        joblib.dump(self, path)

        meta_path = directory / f"{self.target}_{version}.meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata) if self.metadata else {}, f, indent=2)

        latest_path = directory / f"{self.target}_latest.joblib"
        joblib.dump(self, latest_path)
        return path

    @classmethod
    def load(cls, path: Path) -> "EnsembleForecastModel":
        return joblib.load(path)

    @classmethod
    def load_latest(cls, directory: Path, target: str) -> Optional["EnsembleForecastModel"]:
        latest_path = Path(directory) / f"{target}_latest.joblib"
        if not latest_path.exists():
            return None
        return cls.load(latest_path)
