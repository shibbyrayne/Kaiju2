"""Full feature engineering pipeline: calendar/events + weather + recency weights.

This module ties together :mod:`src.austin_calendar` (deterministic
calendar/event features) and :mod:`src.weather_client` (external weather
signals) into a single feature matrix, and computes exponential recency
sample weights used by the modeling layer.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src import austin_calendar, config, weather_client

# Columns that make up the feature matrix handed to the models (excludes
# identifiers/targets). Kept as a module constant so training and inference
# stay in lockstep.
CALENDAR_FEATURE_COLUMNS = [
    "day_of_week_num",
    "day_of_month",
    "week_of_year",
    "month",
    "is_weekend",
    "is_payday_cycle",
    "month_sin",
    "month_cos",
    "dow_sin",
    "dow_cos",
    "is_federal_holiday",
    "is_juneteenth",
    "is_day_before_holiday",
    "is_day_after_holiday",
] + [f"event_{c}_active" for c in austin_calendar.EVENT_CATEGORIES] + [
    f"event_{c}_proximity" for c in austin_calendar.EVENT_CATEGORIES
] + ["any_major_event_active"]

WEATHER_FEATURE_COLUMNS = [
    "temp_max",
    "temp_min",
    "precipitation_mm",
    "is_rainy_day",
    "severe_weather_alert",
]

ALL_FEATURE_COLUMNS = CALENDAR_FEATURE_COLUMNS + WEATHER_FEATURE_COLUMNS


def add_calendar_and_weather_features(
    df: pd.DataFrame,
    fetch_weather: bool = True,
    latitude: float = config.AUSTIN_LATITUDE,
    longitude: float = config.AUSTIN_LONGITUDE,
    timezone: str = config.AUSTIN_TIMEZONE,
) -> pd.DataFrame:
    """Enrich a DataFrame with a ``date`` column with all external features."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = austin_calendar.build_calendar_features(df)

    if fetch_weather:
        start = df["date"].min().date()
        end = df["date"].max().date()
        weather = weather_client.get_weather_features(start, end, latitude, longitude, timezone)
        df = df.merge(weather.drop(columns=["source"], errors="ignore"), on="date", how="left")
        df = weather_client.add_weather_derived_features(df)
    else:
        for col in WEATHER_FEATURE_COLUMNS:
            df[col] = 0.0

    return df


def compute_recency_weights(
    dates: pd.Series,
    as_of: Optional[date] = None,
    half_life_days: int = config.RECENCY_HALF_LIFE_DAYS,
    recent_window_days: int = config.RECENCY_WINDOW_DAYS,
    min_multiplier: float = config.RECENCY_MIN_MULTIPLIER,
    max_multiplier: float = config.RECENCY_MAX_MULTIPLIER,
) -> np.ndarray:
    """Exponential recency sample weights: w_i = exp(-lambda * age_days).

    Weights are rescaled so that the most recent ``recent_window_days`` of
    observations receive, on average, ``max_multiplier``x the weight of the
    oldest observations, per spec (2.5x-4x uplift for the trailing 90 days).
    """
    dates = pd.to_datetime(dates)
    if as_of is None:
        as_of = dates.max().date()
    as_of_ts = pd.Timestamp(as_of)

    age_days = (as_of_ts - dates).dt.days.clip(lower=0).to_numpy(dtype=float)

    lam = np.log(2) / half_life_days
    raw_weights = np.exp(-lam * age_days)

    # Normalize so the oldest observation in the series gets `min_multiplier`
    # and anything inside the recent window approaches `max_multiplier`.
    raw_min = raw_weights.min()
    raw_at_window_edge = np.exp(-lam * recent_window_days)
    if raw_min <= 0 or raw_at_window_edge <= 0:
        return np.full_like(raw_weights, min_multiplier)

    scale = (max_multiplier - min_multiplier) / (1 - raw_at_window_edge / raw_weights.max())
    scale = scale if np.isfinite(scale) and scale > 0 else (max_multiplier - min_multiplier)

    normalized = min_multiplier + (raw_weights - raw_weights.min()) / (
        raw_weights.max() - raw_weights.min() + 1e-9
    ) * (max_multiplier - min_multiplier)

    return normalized


def build_training_frame(
    df: pd.DataFrame,
    fetch_weather: bool = True,
    as_of: Optional[date] = None,
) -> pd.DataFrame:
    """Build the full modeling frame: features + recency weight column."""
    enriched = add_calendar_and_weather_features(df, fetch_weather=fetch_weather)
    enriched["sample_weight"] = compute_recency_weights(enriched["date"], as_of=as_of)
    return enriched


def build_future_frame(
    start: date,
    end: date,
    fetch_weather: bool = True,
    latitude: float = config.AUSTIN_LATITUDE,
    longitude: float = config.AUSTIN_LONGITUDE,
    timezone: str = config.AUSTIN_TIMEZONE,
) -> pd.DataFrame:
    """Build a feature frame for a future date range (no sales/guest_count)."""
    dates = pd.date_range(start, end, freq="D")
    df = pd.DataFrame({"date": dates})
    df = add_calendar_and_weather_features(
        df, fetch_weather=fetch_weather, latitude=latitude, longitude=longitude, timezone=timezone
    )
    return df


def top_drivers_for_row(row: pd.Series, top_n: int = 3) -> list[dict]:
    """Summarize the strongest calendar/event/weather drivers for one forecast day.

    This is a transparent, rule-based explanation layer (not SHAP values) so
    it works identically for any underlying model in the ensemble.
    """
    drivers = []

    for category in austin_calendar.EVENT_CATEGORIES:
        proximity = row.get(f"event_{category}_proximity", 0.0)
        active = row.get(f"event_{category}_active", 0)
        if proximity and proximity > 0.05:
            label = category.replace("_", " ").title()
            state = "active" if active else "nearby"
            drivers.append(
                {
                    "feature": f"event_{category}",
                    "description": f"{label} ({state})",
                    "strength": float(proximity),
                }
            )

    if row.get("is_federal_holiday"):
        drivers.append({"feature": "is_federal_holiday", "description": "Federal holiday", "strength": 0.5})
    if row.get("is_juneteenth"):
        drivers.append({"feature": "is_juneteenth", "description": "Juneteenth", "strength": 0.5})
    if row.get("is_weekend"):
        drivers.append({"feature": "is_weekend", "description": "Weekend", "strength": 0.3})
    if row.get("is_payday_cycle"):
        drivers.append({"feature": "is_payday_cycle", "description": "Payday cycle (1st/15th)", "strength": 0.2})
    if row.get("severe_weather_alert"):
        drivers.append({"feature": "severe_weather_alert", "description": "Severe weather alert", "strength": 0.6})
    elif row.get("is_rainy_day"):
        drivers.append({"feature": "is_rainy_day", "description": "Rainy day", "strength": 0.25})

    drivers.sort(key=lambda d: d["strength"], reverse=True)
    return drivers[:top_n]
