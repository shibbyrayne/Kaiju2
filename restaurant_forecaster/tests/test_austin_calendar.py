import pandas as pd

from src import austin_calendar


def test_load_events_parses_dates():
    events = austin_calendar.load_events()
    assert not events.empty
    assert {"name", "category", "start", "end", "lift_pct"}.issubset(events.columns)


def test_event_active_flag_inside_window():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-03-12", "2024-06-01"])})
    enriched = austin_calendar.add_event_features(df)
    assert enriched.loc[0, "event_sxsw_active"] == 1
    assert enriched.loc[1, "event_sxsw_active"] == 0


def test_event_proximity_decays_with_distance():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-10-18", "2024-10-25", "2025-01-01"])})
    enriched = austin_calendar.add_event_features(df)
    # 2024-10-18 is inside F1 window, 10-25 is 5 days after, 1-1 is far away.
    inside = enriched.loc[0, "event_f1_proximity"]
    near = enriched.loc[1, "event_f1_proximity"]
    far = enriched.loc[2, "event_f1_proximity"]
    assert inside == 1.0
    assert 0 < near < inside
    assert far == 0.0


def test_holiday_features_flag_new_years_day():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"])})
    enriched = austin_calendar.add_holiday_features(df)
    assert enriched.loc[0, "is_federal_holiday"] == 1
    assert enriched.loc[1, "is_federal_holiday"] == 0


def test_juneteenth_flagged():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-06-19", "2024-06-20"])})
    enriched = austin_calendar.add_holiday_features(df)
    assert enriched.loc[0, "is_juneteenth"] == 1
    assert enriched.loc[1, "is_juneteenth"] == 0


def test_seasonality_features_weekend_and_payday():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-06", "2024-01-15", "2024-01-08"])})
    enriched = austin_calendar.add_seasonality_features(df)
    assert enriched.loc[0, "is_weekend"] == 1  # Saturday
    assert enriched.loc[1, "is_payday_cycle"] == 1  # 15th
    assert enriched.loc[2, "is_weekend"] == 0  # Monday
