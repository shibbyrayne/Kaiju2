"""Ingestion, schema validation, and cleaning for historical sales data."""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src import config

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["date", "sales", "guest_count"]
OPTIONAL_COLUMNS = ["day_of_week", "meal_period", "promotions_active"]
VALID_MEAL_PERIODS = {"Lunch", "Dinner", "Brunch", "All Day", None}


class SalesRecord(BaseModel):
    """Schema for a single historical daily sales observation."""

    date: date
    sales: float = Field(ge=0)
    guest_count: int = Field(ge=0)
    day_of_week: Optional[str] = None
    meal_period: Optional[str] = None
    promotions_active: Optional[bool] = None

    @field_validator("meal_period")
    @classmethod
    def validate_meal_period(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in VALID_MEAL_PERIODS:
            raise ValueError(f"meal_period must be one of {VALID_MEAL_PERIODS}, got {v!r}")
        return v


class SchemaValidationError(Exception):
    """Raised when the ingested CSV fails required schema checks."""


def _check_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaValidationError(
            f"Missing required column(s): {missing}. Required columns are {REQUIRED_COLUMNS}."
        )


def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a raw CSV and validate that required columns are present."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    _check_required_columns(df)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        logger.warning("Dropping %d row(s) with unparseable dates", bad_dates)
        df = df.dropna(subset=["date"])

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["guest_count"] = pd.to_numeric(df["guest_count"], errors="coerce")

    bad_numeric = df["sales"].isna() | df["guest_count"].isna()
    if bad_numeric.any():
        logger.warning("Dropping %d row(s) with non-numeric sales/guest_count", bad_numeric.sum())
        df = df[~bad_numeric]

    if "promotions_active" in df.columns:
        df["promotions_active"] = df["promotions_active"].astype(bool)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def clean_data(
    df: pd.DataFrame,
    fill_missing_dates: bool = True,
    clip_outliers: bool = True,
    outlier_zscore_threshold: float = config.OUTLIER_ZSCORE_THRESHOLD,
) -> pd.DataFrame:
    """Apply data-cleaning steps: gap-fill dates, flag closures, clip outliers."""
    df = df.copy()
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date").reset_index(drop=True)

    if fill_missing_dates and len(df) > 1:
        full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full_range)
        df.index.name = "date"
        df = df.reset_index()
        # Days absent from the source file are treated as closures (0 sales),
        # not missing data, since restaurants commonly skip logging closed days.
        df["sales"] = df["sales"].fillna(0.0)
        df["guest_count"] = df["guest_count"].fillna(0).astype(int)
        if "promotions_active" in df.columns:
            df["promotions_active"] = df["promotions_active"].fillna(False)

    df["is_closure"] = df["sales"] <= config.CLOSURE_SALES_THRESHOLD

    if clip_outliers:
        open_mask = ~df["is_closure"]
        if open_mask.sum() >= 5:
            mean = df.loc[open_mask, "sales"].mean()
            std = df.loc[open_mask, "sales"].std(ddof=0)
            if std and std > 0:
                z = (df["sales"] - mean) / std
                outliers = open_mask & (z.abs() > outlier_zscore_threshold)
                if outliers.any():
                    lower = mean - outlier_zscore_threshold * std
                    upper = mean + outlier_zscore_threshold * std
                    logger.info("Clipping %d outlier day(s) to [%.2f, %.2f]", outliers.sum(), lower, upper)
                    df.loc[outliers, "sales"] = df.loc[outliers, "sales"].clip(lower=max(lower, 0), upper=upper)

    df["day_of_week"] = df["date"].dt.day_name()
    return df


def load_and_clean(path: str | Path, **clean_kwargs) -> pd.DataFrame:
    """Convenience wrapper: load + validate + clean in one call."""
    df = load_csv(path)
    return clean_data(df, **clean_kwargs)
