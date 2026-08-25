"""Austin, TX event calendar, holiday, and seasonality feature generation.

All features here are deterministic — no network calls — so they can be
computed identically for historical training data and future forecast
horizons.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from src import config

# Days over which an event's influence decays to (approximately) zero on
# either side of its window.
PROXIMITY_DECAY_WINDOW_DAYS = 10
PROXIMITY_DECAY_RATE = 0.35

EVENT_CATEGORIES = [
    "sxsw",
    "acl",
    "f1",
    "marathon",
    "pecan_street",
    "trail_of_lights",
    "graduation",
    "ut_football",
]


def load_events(path: Path = config.EVENTS_PATH) -> pd.DataFrame:
    """Load the Austin events calendar JSON into a flat DataFrame."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    events = pd.DataFrame(raw["events"])
    events["start"] = pd.to_datetime(events["start"])
    events["end"] = pd.to_datetime(events["end"])
    return events


def _days_from_window(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    """Signed distance (in days) from each date to the nearest edge of [start, end].

    0 while inside the window, negative before it, positive after it.
    """
    d = dates.values.astype("datetime64[D]")
    s = np.datetime64(start.date())
    e = np.datetime64(end.date())
    before = (d < s)
    after = (d > e)
    dist = np.zeros(len(d), dtype=float)
    dist[before] = (d[before] - s).astype("timedelta64[D]").astype(float)
    dist[after] = (d[after] - e).astype("timedelta64[D]").astype(float)
    return dist


def add_event_features(df: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Attach binary flags and proximity-decay features for each event category.

    For each category we add:
      - ``event_<category>_active``: 1 if the date falls inside any event window.
      - ``event_<category>_proximity``: exponential-decay weight in [0, 1] that
        peaks at 1.0 inside the event window and decays toward 0 the farther a
        date is from the nearest occurrence (both before and after).
    """
    df = df.copy()
    if events is None:
        events = load_events()

    dates = df["date"]

    for category in EVENT_CATEGORIES:
        cat_events = events[events["category"] == category]
        active = pd.Series(False, index=df.index)
        proximity = pd.Series(0.0, index=df.index)

        for _, ev in cat_events.iterrows():
            signed_dist = _days_from_window(dates, ev["start"], ev["end"])
            active |= signed_dist == 0
            decay = np.exp(-PROXIMITY_DECAY_RATE * np.abs(signed_dist))
            decay = np.where(np.abs(signed_dist) > PROXIMITY_DECAY_WINDOW_DAYS, 0.0, decay)
            proximity = np.maximum(proximity, decay)

        df[f"event_{category}_active"] = active.astype(int)
        df[f"event_{category}_proximity"] = proximity

    df["any_major_event_active"] = (
        df[[f"event_{c}_active" for c in EVENT_CATEGORIES]].max(axis=1)
    )
    return df


def get_active_event_names(single_date: pd.Timestamp, events: Optional[pd.DataFrame] = None) -> list[str]:
    """Return human-readable names of events active (or within 3 days) of a date."""
    if events is None:
        events = load_events()
    names = []
    for _, ev in events.iterrows():
        dist = _days_from_window(pd.Series([single_date]), ev["start"], ev["end"])[0]
        if abs(dist) <= 3:
            names.append((ev["name"], dist, ev["lift_pct"]))
    return names


def add_holiday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach US federal holiday, Juneteenth, and SXSW-week regional flags."""
    df = df.copy()
    min_year = df["date"].dt.year.min() - 1
    max_year = df["date"].dt.year.max() + 1

    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=f"{min_year}-01-01", end=f"{max_year}-12-31")
    df["is_federal_holiday"] = df["date"].isin(holidays).astype(int)

    juneteenth = pd.to_datetime(
        [f"{y}-06-19" for y in range(min_year, max_year + 1)]
    )
    df["is_juneteenth"] = df["date"].isin(juneteenth).astype(int)

    # Day before / after a federal holiday often shifts dining behavior.
    df["is_day_before_holiday"] = df["date"].isin(holidays - pd.Timedelta(days=1)).astype(int)
    df["is_day_after_holiday"] = df["date"].isin(holidays + pd.Timedelta(days=1)).astype(int)

    return df


def add_seasonality_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach calendar/seasonality features: DOW, DOM, week/month, payday cycle."""
    df = df.copy()
    dt = df["date"].dt

    df["day_of_week_num"] = dt.dayofweek  # Monday=0
    df["day_of_month"] = dt.day
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["month"] = dt.month
    df["is_weekend"] = (dt.dayofweek >= 5).astype(int)
    df["is_payday_cycle"] = df["day_of_month"].isin([1, 15]).astype(int)

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week_num"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week_num"] / 7)

    return df


def build_calendar_features(df: pd.DataFrame, events: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Run the full deterministic feature pipeline: seasonality + holidays + events."""
    df = add_seasonality_features(df)
    df = add_holiday_features(df)
    df = add_event_features(df, events=events)
    return df
