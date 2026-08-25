from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src import weather_client


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("bad status")

    def json(self):
        return self._payload


def _fake_payload(start: date, end: date):
    days = pd.date_range(start, end, freq="D")
    return {
        "daily": {
            "time": [d.strftime("%Y-%m-%d") for d in days],
            "temperature_2m_max": [28.0] * len(days),
            "temperature_2m_min": [15.0] * len(days),
            "precipitation_sum": [0.0] * len(days),
        }
    }


def test_get_weather_uses_cache_on_second_call(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "weather_cache.db"
    call_count = {"n": 0}

    def fake_get(url, params=None, timeout=None):
        call_count["n"] += 1
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        return _FakeResponse(_fake_payload(start, end))

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    # Use a far-past range so it's unambiguously served by the "archive" branch.
    start = date(2023, 1, 1)
    end = date(2023, 1, 5)

    first = weather_client.get_weather(start, end, cache_path=cache_path)
    assert len(first) == 5
    assert call_count["n"] == 1

    second = weather_client.get_weather(start, end, cache_path=cache_path)
    assert len(second) == 5
    assert call_count["n"] == 1  # served entirely from cache, no new HTTP call


def test_get_weather_handles_request_failure_gracefully(tmp_path: Path, monkeypatch):
    cache_path = tmp_path / "weather_cache.db"

    def fake_get(url, params=None, timeout=None):
        raise weather_client.requests.RequestException("network down")

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    df = weather_client.get_weather(date(2023, 1, 1), date(2023, 1, 3), cache_path=cache_path)
    assert df.empty or df["temp_max"].isna().all()


def test_get_weather_date_column_is_datetime_when_request_fails(tmp_path: Path, monkeypatch):
    """Regression: an empty result must still carry a datetime64 'date' column.

    Building the empty frame via plain `pd.DataFrame(columns=[...])` gives
    every column (including 'date') `object` dtype, which upstream code
    can't distinguish from a genuine datetime column until it tries to
    merge -- see the partial-failure test below for the actual crash this
    caused.
    """
    cache_path = tmp_path / "weather_cache.db"

    def fake_get(url, params=None, timeout=None):
        raise weather_client.requests.RequestException("network down")

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    df = weather_client.get_weather(date(2023, 1, 1), date(2023, 1, 3), cache_path=cache_path)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_get_weather_date_dtype_survives_partial_fetch_failure(tmp_path: Path, monkeypatch):
    """Regression for the exact bug reported from a live deploy:

    'You are trying to merge on datetime64[us] and object columns for key
    date'. This happened when one of the two Open-Meteo calls (archive vs.
    forecast) succeeded and the other failed/returned empty -- concatenating
    a real datetime64 column with an empty object-dtype one silently
    upcasts the result back to object.
    """
    cache_path = tmp_path / "weather_cache.db"

    def fake_get(url, params=None, timeout=None):
        if url == weather_client.config.OPEN_METEO_ARCHIVE_URL:
            start = date.fromisoformat(params["start_date"])
            end = date.fromisoformat(params["end_date"])
            return _FakeResponse(_fake_payload(start, end))
        raise weather_client.requests.RequestException("forecast endpoint unreachable")

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    # Spans past (archive, succeeds) and future (forecast, fails) dates so
    # both branches of get_weather() run in the same call.
    start = date.today() - timedelta(days=3)
    end = date.today() + timedelta(days=3)

    df = weather_client.get_weather(start, end, cache_path=cache_path)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])

    # This is the exact operation that raised in production: merging the
    # weather frame's 'date' column against a real datetime64 calendar frame.
    calendar_df = pd.DataFrame({"date": pd.date_range(start, end, freq="D")})
    merged = calendar_df.merge(df.drop(columns=["source"], errors="ignore"), on="date", how="left")
    assert len(merged) == len(calendar_df)


def test_build_training_frame_with_weather_survives_partial_fetch_failure(monkeypatch):
    """End-to-end regression: training with fetch_weather=True must not crash
    when one of the two Open-Meteo endpoints is unreachable (e.g. an
    outbound-network restriction on only one branch's requests).

    get_weather_features()/get_weather() don't take a cache_path override,
    so this exercises the real config.WEATHER_CACHE_PATH -- cleaned up
    below so the test doesn't leave a stray DB in the repo.
    """
    from src import feature_engineering

    def fake_get(url, params=None, timeout=None):
        if url == weather_client.config.OPEN_METEO_ARCHIVE_URL:
            start = date.fromisoformat(params["start_date"])
            end = date.fromisoformat(params["end_date"])
            return _FakeResponse(_fake_payload(start, end))
        raise weather_client.requests.RequestException("forecast endpoint unreachable")

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    dates = pd.date_range(date.today() - timedelta(days=10), date.today() + timedelta(days=3), freq="D")
    df = pd.DataFrame({"date": dates, "sales": 100.0, "guest_count": 10})

    cache_path = weather_client.config.WEATHER_CACHE_PATH
    existed_before = cache_path.exists()
    try:
        result = feature_engineering.build_training_frame(df, fetch_weather=True)
        assert len(result) == len(df)
        for col in feature_engineering.WEATHER_FEATURE_COLUMNS:
            assert col in result.columns
    finally:
        if not existed_before and cache_path.exists():
            cache_path.unlink()


def _fake_payload_with_nulls(start: date, end: date, null_field: str):
    days = pd.date_range(start, end, freq="D")
    payload = {
        "daily": {
            "time": [d.strftime("%Y-%m-%d") for d in days],
            "temperature_2m_max": [28.0] * len(days),
            "temperature_2m_min": [15.0] * len(days),
            "precipitation_sum": [0.0] * len(days),
        }
    }
    payload["daily"][null_field] = [None] * len(days)
    return payload


def test_get_weather_numeric_columns_are_float_when_api_returns_all_nulls(tmp_path: Path, monkeypatch):
    """Regression for the exact bug reported from a live deploy:

    'pandas dtypes must be int, float or bool. Fields with bad pandas
    dtypes: temp_max: object, temp_min: object, precipitation_mm: object'.

    Open-Meteo returns an all-null array for a field near the edges of its
    forecast window; a Python list of all None infers as `object` dtype in
    pandas rather than float64, and that dtype survives all the way to
    LightGBM's .fit() call.
    """
    cache_path = tmp_path / "weather_cache.db"

    def fake_get(url, params=None, timeout=None):
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        return _FakeResponse(_fake_payload_with_nulls(start, end, "temperature_2m_max"))

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    df = weather_client.get_weather(date(2023, 1, 1), date(2023, 1, 5), cache_path=cache_path)
    for col in weather_client.NUMERIC_WEATHER_COLUMNS:
        assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is {df[col].dtype}, expected numeric"

    enriched = weather_client.add_weather_derived_features(df)
    for col in weather_client.NUMERIC_WEATHER_COLUMNS:
        assert pd.api.types.is_numeric_dtype(enriched[col])


def test_build_training_frame_numeric_dtypes_survive_all_null_weather_field(monkeypatch):
    """End-to-end: object-dtype weather columns must not reach the LightGBM fit."""
    from src import feature_engineering

    def fake_get(url, params=None, timeout=None):
        start = date.fromisoformat(params["start_date"])
        end = date.fromisoformat(params["end_date"])
        return _FakeResponse(_fake_payload_with_nulls(start, end, "precipitation_sum"))

    monkeypatch.setattr(weather_client.requests, "get", fake_get)

    dates = pd.date_range(date.today() - timedelta(days=20), date.today() - timedelta(days=5), freq="D")
    df = pd.DataFrame({"date": dates, "sales": 100.0, "guest_count": 10})

    cache_path = weather_client.config.WEATHER_CACHE_PATH
    existed_before = cache_path.exists()
    try:
        result = feature_engineering.build_training_frame(df, fetch_weather=True)
        for col in feature_engineering.WEATHER_FEATURE_COLUMNS:
            assert pd.api.types.is_numeric_dtype(result[col]), f"{col} is {result[col].dtype}"
    finally:
        if not existed_before and cache_path.exists():
            cache_path.unlink()


def test_add_weather_derived_features_flags_rain_and_severe():
    df = pd.DataFrame(
        {
            "temp_max": [30.0, 42.0, 20.0],
            "temp_min": [15.0, 25.0, 10.0],
            "precipitation_mm": [0.0, 5.0, 50.0],
        }
    )
    enriched = weather_client.add_weather_derived_features(df)
    assert enriched.loc[0, "is_rainy_day"] == 0
    assert enriched.loc[1, "is_rainy_day"] == 1
    assert enriched.loc[1, "severe_weather_alert"] == 1  # temp_max >= 40
    assert enriched.loc[2, "severe_weather_alert"] == 1  # precipitation >= 40mm
