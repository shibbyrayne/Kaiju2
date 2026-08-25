"""Open-Meteo weather ingestion (historical archive + forecast) with SQLite caching.

Open-Meteo requires no API key. Historical dates are served from the archive
endpoint; dates within the forecast horizon (today .. +16 days) are served
from the forecast endpoint. Results are cached locally so repeated CLI runs
don't re-hit the network for the same date range.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src import config

logger = logging.getLogger(__name__)

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
]

RAIN_THRESHOLD_MM = 2.5
SEVERE_PRECIP_THRESHOLD_MM = 40.0
SEVERE_TEMP_HIGH_C = 40.0
SEVERE_TEMP_LOW_C = -5.0


def _init_cache(db_path: Path = config.WEATHER_CACHE_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_daily (
            date TEXT PRIMARY KEY,
            temp_max REAL,
            temp_min REAL,
            precipitation_mm REAL,
            source TEXT
        )
        """
    )
    conn.commit()
    return conn


def _empty_weather_frame() -> pd.DataFrame:
    """An empty weather frame with explicit dtypes.

    Critical for a zero-row ``date`` column specifically: pandas infers
    ``object`` dtype for columns of an empty DataFrame/query result, and
    concatenating that against a real ``datetime64`` frame silently
    upcasts the combined column back to ``object`` -- which then blows up
    with "You are trying to merge on datetime64[us] and object columns"
    the moment it's merged against the (correctly-typed) calendar frame.
    """
    return pd.DataFrame(
        {
            "date": pd.Series(dtype="datetime64[ns]"),
            "temp_max": pd.Series(dtype="float64"),
            "temp_min": pd.Series(dtype="float64"),
            "precipitation_mm": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="object"),
        }
    )


def _read_cached(conn: sqlite3.Connection, start: date, end: date) -> pd.DataFrame:
    query = "SELECT * FROM weather_daily WHERE date >= ? AND date <= ?"
    df = pd.read_sql_query(query, conn, params=(start.isoformat(), end.isoformat()))
    df["date"] = pd.to_datetime(df["date"])
    return df


def _write_cache(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    if df.empty:
        return
    records = df.copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    conn.executemany(
        """
        INSERT INTO weather_daily (date, temp_max, temp_min, precipitation_mm, source)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(date) DO UPDATE SET
            temp_max=excluded.temp_max,
            temp_min=excluded.temp_min,
            precipitation_mm=excluded.precipitation_mm,
            source=excluded.source
        """,
        list(
            records[["date", "temp_max", "temp_min", "precipitation_mm", "source"]].itertuples(
                index=False, name=None
            )
        ),
    )
    conn.commit()


def _fetch_open_meteo(
    url: str,
    start: date,
    end: date,
    latitude: float,
    longitude: float,
    timezone: str,
    source_label: str,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": timezone,
    }
    try:
        resp = requests.get(url, params=params, timeout=config.WEATHER_REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("Weather API request failed (%s): %s", source_label, exc)
        return _empty_weather_frame()

    daily = payload.get("daily", {})
    if not daily or "time" not in daily:
        return _empty_weather_frame()

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["time"]),
            "temp_max": daily.get("temperature_2m_max", [None] * len(daily["time"])),
            "temp_min": daily.get("temperature_2m_min", [None] * len(daily["time"])),
            "precipitation_mm": daily.get("precipitation_sum", [None] * len(daily["time"])),
        }
    )
    df["source"] = source_label
    return df


def get_weather(
    start: date,
    end: date,
    latitude: float = config.AUSTIN_LATITUDE,
    longitude: float = config.AUSTIN_LONGITUDE,
    timezone: str = config.AUSTIN_TIMEZONE,
    cache_path: Path = config.WEATHER_CACHE_PATH,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return a daily weather DataFrame for [start, end], using cache-first fetch.

    Historical dates (before today) hit the archive API; dates from today
    through +16 days hit the forecast API; anything further out is left
    blank (weather features fall back to climatological defaults upstream).
    """
    conn = _init_cache(cache_path)
    try:
        cached = _read_cached(conn, start, end) if use_cache else pd.DataFrame()
        have_dates = set(cached["date"].dt.date) if not cached.empty else set()
        all_dates = {start + timedelta(days=i) for i in range((end - start).days + 1)}
        missing_dates = sorted(all_dates - have_dates)

        fetched_frames = []
        if missing_dates:
            today = date.today()
            forecast_horizon = today + timedelta(days=16)

            hist_missing = [d for d in missing_dates if d < today]
            fcst_missing = [d for d in missing_dates if today <= d <= forecast_horizon]

            if hist_missing:
                h_start, h_end = min(hist_missing), max(hist_missing)
                fetched_frames.append(
                    _fetch_open_meteo(
                        config.OPEN_METEO_ARCHIVE_URL, h_start, h_end, latitude, longitude, timezone, "archive"
                    )
                )
            if fcst_missing:
                f_start, f_end = min(fcst_missing), max(fcst_missing)
                fetched_frames.append(
                    _fetch_open_meteo(
                        config.OPEN_METEO_FORECAST_URL, f_start, f_end, latitude, longitude, timezone, "forecast"
                    )
                )

        fetched = pd.concat(fetched_frames, ignore_index=True) if fetched_frames else pd.DataFrame()
        if not fetched.empty:
            _write_cache(conn, fetched)

        combined = pd.concat([cached, fetched], ignore_index=True) if not fetched.empty else cached
        if combined.empty:
            combined = _empty_weather_frame()
        # Belt-and-suspenders: guarantee datetime64 regardless of how
        # `combined` was assembled above (cache hit, fresh fetch, concat of
        # both, or the empty fallback).
        combined["date"] = pd.to_datetime(combined["date"])
        combined = combined.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
        return combined
    finally:
        conn.close()


def add_weather_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_rainy_day / severe_weather_alert flags from raw weather columns."""
    df = df.copy()
    df["temp_max"] = df["temp_max"].fillna(df["temp_max"].mean())
    df["temp_min"] = df["temp_min"].fillna(df["temp_min"].mean())
    df["precipitation_mm"] = df["precipitation_mm"].fillna(0.0)

    df["is_rainy_day"] = (df["precipitation_mm"] >= RAIN_THRESHOLD_MM).astype(int)
    df["severe_weather_alert"] = (
        (df["precipitation_mm"] >= SEVERE_PRECIP_THRESHOLD_MM)
        | (df["temp_max"] >= SEVERE_TEMP_HIGH_C)
        | (df["temp_min"] <= SEVERE_TEMP_LOW_C)
    ).astype(int)
    return df


def get_weather_features(
    start: date,
    end: date,
    latitude: float = config.AUSTIN_LATITUDE,
    longitude: float = config.AUSTIN_LONGITUDE,
    timezone: str = config.AUSTIN_TIMEZONE,
) -> pd.DataFrame:
    """Fetch weather and attach derived rain/severe-weather flags in one call."""
    weather = get_weather(start, end, latitude, longitude, timezone)
    weather["date"] = pd.to_datetime(weather["date"])
    if weather.empty:
        full_range = pd.date_range(start, end, freq="D")
        weather = pd.DataFrame(
            {
                "date": full_range,
                "temp_max": float("nan"),
                "temp_min": float("nan"),
                "precipitation_mm": float("nan"),
                "source": "unavailable",
            }
        )
    return add_weather_derived_features(weather)
