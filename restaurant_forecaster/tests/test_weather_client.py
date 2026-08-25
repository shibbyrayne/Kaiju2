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
