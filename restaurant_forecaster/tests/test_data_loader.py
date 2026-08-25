from pathlib import Path

import pandas as pd
import pytest

from src import data_loader


def test_load_csv_missing_required_column(tmp_path: Path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("date,sales\n2024-01-01,100\n")
    with pytest.raises(data_loader.SchemaValidationError):
        data_loader.load_csv(csv_path)


def test_load_csv_drops_bad_rows(tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text(
        "date,sales,guest_count\n"
        "2024-01-01,100,10\n"
        "not-a-date,50,5\n"
        "2024-01-03,abc,7\n"
        "2024-01-04,200,20\n"
    )
    df = data_loader.load_csv(csv_path)
    assert len(df) == 2
    assert set(df["date"].dt.strftime("%Y-%m-%d")) == {"2024-01-01", "2024-01-04"}


def test_clean_data_fills_missing_dates_as_closures():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
            "sales": [100.0, 200.0],
            "guest_count": [10, 20],
        }
    )
    cleaned = data_loader.clean_data(df)
    assert len(cleaned) == 4
    gap_days = cleaned[cleaned["date"].isin(pd.to_datetime(["2024-01-02", "2024-01-03"]))]
    assert (gap_days["sales"] == 0.0).all()
    assert (gap_days["is_closure"]).all()


def test_clean_data_clips_outliers():
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    sales = [100.0] * 39 + [100000.0]  # one wild outlier
    df = pd.DataFrame({"date": dates, "sales": sales, "guest_count": [10] * 40})
    cleaned = data_loader.clean_data(df, fill_missing_dates=False, clip_outliers=True)
    assert cleaned["sales"].max() < 100000.0


def test_clean_data_flags_closures():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates, "sales": [0, 100, 0, 150, 200], "guest_count": [0, 10, 0, 15, 20]})
    cleaned = data_loader.clean_data(df, fill_missing_dates=False)
    assert cleaned["is_closure"].tolist() == [True, False, True, False, False]
