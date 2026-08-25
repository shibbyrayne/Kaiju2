"""SQLite-backed storage for forecasts, actuals, and error logs.

Single-file database (see ``config.DB_PATH``) that backs the adaptive
feedback loop: every ``forecast`` run persists its projections, every
``log-actuals`` call persists the actual outcome and the resulting error,
and the feedback loop reads both back out to detect drift.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from src import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecasts (
    date TEXT NOT NULL,
    target TEXT NOT NULL,
    p10 REAL,
    p50 REAL,
    p90 REAL,
    model_version TEXT,
    generated_at TEXT NOT NULL,
    PRIMARY KEY (date, target, generated_at)
);

CREATE TABLE IF NOT EXISTS actuals (
    date TEXT PRIMARY KEY,
    sales REAL,
    guest_count INTEGER,
    logged_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS error_log (
    date TEXT NOT NULL,
    target TEXT NOT NULL,
    actual REAL,
    predicted REAL,
    abs_error REAL,
    pct_error REAL,
    bias REAL,
    model_version TEXT,
    logged_at TEXT NOT NULL,
    PRIMARY KEY (date, target)
);

CREATE TABLE IF NOT EXISTS model_registry (
    target TEXT NOT NULL,
    model_version TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    training_rows INTEGER,
    trigger_reason TEXT,
    artifact_path TEXT,
    PRIMARY KEY (target, model_version)
);
"""


@contextmanager
def get_connection(db_path: Path = config.DB_PATH) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_forecast(df: pd.DataFrame, target: str, model_version: str, generated_at: str, db_path: Path = config.DB_PATH) -> None:
    """Persist a forecast DataFrame with date/p10/p50/p90 columns."""
    with get_connection(db_path) as conn:
        rows = [
            (
                pd.Timestamp(r["date"]).strftime("%Y-%m-%d"),
                target,
                float(r["p10"]),
                float(r["p50"]),
                float(r["p90"]),
                model_version,
                generated_at,
            )
            for _, r in df.iterrows()
        ]
        conn.executemany(
            """
            INSERT INTO forecasts (date, target, p10, p50, p90, model_version, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, target, generated_at) DO UPDATE SET
                p10=excluded.p10, p50=excluded.p50, p90=excluded.p90, model_version=excluded.model_version
            """,
            rows,
        )


def get_latest_forecast_for_date(target: str, date_str: str, db_path: Path = config.DB_PATH) -> Optional[dict]:
    with get_connection(db_path) as conn:
        cur = conn.execute(
            """
            SELECT date, target, p10, p50, p90, model_version, generated_at
            FROM forecasts
            WHERE target = ? AND date = ?
            ORDER BY generated_at DESC
            LIMIT 1
            """,
            (target, date_str),
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = ["date", "target", "p10", "p50", "p90", "model_version", "generated_at"]
        return dict(zip(keys, row))


def save_actuals(date_str: str, sales: float, guest_count: int, logged_at: str, db_path: Path = config.DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO actuals (date, sales, guest_count, logged_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET sales=excluded.sales, guest_count=excluded.guest_count, logged_at=excluded.logged_at
            """,
            (date_str, sales, guest_count, logged_at),
        )


def save_error_log_entry(
    date_str: str,
    target: str,
    actual: float,
    predicted: float,
    abs_error: float,
    pct_error: float,
    bias: float,
    model_version: str,
    logged_at: str,
    db_path: Path = config.DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO error_log (date, target, actual, predicted, abs_error, pct_error, bias, model_version, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, target) DO UPDATE SET
                actual=excluded.actual, predicted=excluded.predicted, abs_error=excluded.abs_error,
                pct_error=excluded.pct_error, bias=excluded.bias, model_version=excluded.model_version,
                logged_at=excluded.logged_at
            """,
            (date_str, target, actual, predicted, abs_error, pct_error, bias, model_version, logged_at),
        )


def load_error_log(target: str, db_path: Path = config.DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            "SELECT * FROM error_log WHERE target = ? ORDER BY date", conn, params=(target,)
        )
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def load_actuals(db_path: Path = config.DB_PATH) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM actuals ORDER BY date", conn)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
    return df


def register_model(
    target: str,
    model_version: str,
    trained_at: str,
    training_rows: int,
    trigger_reason: str,
    artifact_path: str,
    db_path: Path = config.DB_PATH,
) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO model_registry (target, model_version, trained_at, training_rows, trigger_reason, artifact_path)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(target, model_version) DO NOTHING
            """,
            (target, model_version, trained_at, training_rows, trigger_reason, artifact_path),
        )
