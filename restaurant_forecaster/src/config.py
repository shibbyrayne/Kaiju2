"""Central configuration for the Austin Restaurant Forecasting Engine.

All tunable constants live here so the rest of the codebase can import a
single source of truth instead of scattering magic numbers.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

# PROCESSED_DATA_DIR holds the SQLite DB and trained model artifacts -- the
# only state that needs to survive a restart/redeploy. Overridable via
# FORECASTER_PROCESSED_DIR so a host like Render can point it at a mounted
# persistent disk instead of the (ephemeral, on most plans) app filesystem.
PROCESSED_DATA_DIR = Path(os.getenv("FORECASTER_PROCESSED_DIR", str(DATA_DIR / "processed")))

for _dir in (RAW_DATA_DIR, EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

DB_PATH = PROCESSED_DATA_DIR / "forecaster.db"
WEATHER_CACHE_PATH = EXTERNAL_DATA_DIR / "weather_cache.db"
EVENTS_PATH = EXTERNAL_DATA_DIR / "austin_events.json"
MODEL_REGISTRY_DIR = PROCESSED_DATA_DIR / "models"
MODEL_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Location (Austin, TX metro)
# ---------------------------------------------------------------------------
AUSTIN_LATITUDE = 30.2672
AUSTIN_LONGITUDE = -97.7431
AUSTIN_TIMEZONE = "America/Chicago"

# ---------------------------------------------------------------------------
# External API endpoints
# ---------------------------------------------------------------------------
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_REQUEST_TIMEOUT_SECONDS = 15

# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------
RECENCY_HALF_LIFE_DAYS = 45
RECENCY_WINDOW_DAYS = 90
RECENCY_MIN_MULTIPLIER = 1.0
RECENCY_MAX_MULTIPLIER = 4.0

# ---------------------------------------------------------------------------
# Modeling
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
TARGET_ERROR_RATE = 0.10
CONFIDENCE_LOWER_QUANTILE = 0.10
CONFIDENCE_UPPER_QUANTILE = 0.90
CONFIDENCE_MEDIAN_QUANTILE = 0.50

LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 5,
    "random_state": RANDOM_SEED,
    "verbosity": -1,
}

# Weight given to the LightGBM model vs. the Prophet/seasonal baseline model
# when combining ensemble predictions (0-1, LightGBM weight).
ENSEMBLE_LGBM_WEIGHT = 0.6

# ---------------------------------------------------------------------------
# Adaptive feedback loop
# ---------------------------------------------------------------------------
DRIFT_LOOKBACK_DAYS = 14
DRIFT_MIN_LOOKBACK_DAYS = 7
DRIFT_BIAS_THRESHOLD_PCT = 0.08  # 8% sustained bias triggers retraining
MIN_TRAINING_ROWS = 30

# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------
OUTLIER_ZSCORE_THRESHOLD = 3.5
CLOSURE_SALES_THRESHOLD = 0.0

# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------
API_KEY_ENV_VAR = "RESTAURANT_FORECASTER_API_KEY"


@dataclass
class Settings:
    """Runtime-overridable settings bundle."""

    latitude: float = AUSTIN_LATITUDE
    longitude: float = AUSTIN_LONGITUDE
    timezone: str = AUSTIN_TIMEZONE
    recency_half_life_days: int = RECENCY_HALF_LIFE_DAYS
    recency_window_days: int = RECENCY_WINDOW_DAYS
    target_error_rate: float = TARGET_ERROR_RATE
    db_path: Path = field(default_factory=lambda: DB_PATH)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            latitude=float(os.getenv("FORECASTER_LATITUDE", AUSTIN_LATITUDE)),
            longitude=float(os.getenv("FORECASTER_LONGITUDE", AUSTIN_LONGITUDE)),
            timezone=os.getenv("FORECASTER_TIMEZONE", AUSTIN_TIMEZONE),
        )


settings = Settings.from_env()
