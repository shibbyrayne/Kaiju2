"""High-level orchestration: train models and generate daily forecasts.

Shared by the CLI (:mod:`src.cli`) and the optional FastAPI server
(:mod:`src.api`) so both interfaces produce identical results.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config, data_loader, feature_engineering, storage
from src.feedback_loop import TARGETS
from src.models.ensemble import EnsembleForecastModel


@dataclass
class ForecastDay:
    date: str
    projected_sales: float
    projected_guests: int
    sales_low: float
    sales_high: float
    guests_low: int
    guests_high: int
    drivers: list[dict]

    def as_dict(self) -> dict:
        return {
            "date": self.date,
            "projected_sales": round(self.projected_sales, 2),
            "projected_guests": self.projected_guests,
            "conf_interval_low": {
                "sales": round(self.sales_low, 2),
                "guest_count": self.guests_low,
            },
            "conf_interval_high": {
                "sales": round(self.sales_high, 2),
                "guest_count": self.guests_high,
            },
            "drivers": self.drivers,
        }


def train_all_targets(
    raw_data_path: Path,
    fetch_weather: bool = True,
    model_dir: Path = config.MODEL_REGISTRY_DIR,
    db_path: Path = config.DB_PATH,
) -> dict[str, EnsembleForecastModel]:
    """Train (or retrain) ensemble models for both sales and guest_count."""
    df = data_loader.load_and_clean(raw_data_path)
    training_frame = feature_engineering.build_training_frame(df, fetch_weather=fetch_weather)

    models = {}
    for target in TARGETS:
        model = EnsembleForecastModel(target=target, feature_columns=feature_engineering.ALL_FEATURE_COLUMNS)
        model.fit(training_frame, sample_weight=training_frame["sample_weight"].to_numpy())
        model.save(model_dir)
        storage.register_model(
            target=target,
            model_version=model.metadata.model_version,
            trained_at=model.metadata.trained_at,
            training_rows=model.metadata.training_rows,
            trigger_reason="initial_train",
            artifact_path=str(model_dir / f"{target}_{model.metadata.model_version}.joblib"),
            db_path=db_path,
        )
        models[target] = model
    return models


def load_models(model_dir: Path = config.MODEL_REGISTRY_DIR) -> dict[str, Optional[EnsembleForecastModel]]:
    return {target: EnsembleForecastModel.load_latest(model_dir, target) for target in TARGETS}


def generate_forecast(
    start: date,
    end: date,
    model_dir: Path = config.MODEL_REGISTRY_DIR,
    fetch_weather: bool = True,
    db_path: Path = config.DB_PATH,
    persist: bool = True,
) -> list[ForecastDay]:
    """Generate day-by-day sales & guest count forecasts for [start, end]."""
    models = load_models(model_dir)
    missing = [t for t, m in models.items() if m is None]
    if missing:
        raise RuntimeError(
            f"No trained model found for target(s): {missing}. Run `train` first."
        )

    future_frame = feature_engineering.build_future_frame(start, end, fetch_weather=fetch_weather)

    sales_q = models["sales"].predict_quantiles(future_frame)
    guests_q = models["guest_count"].predict_quantiles(future_frame)

    generated_at = datetime.now(timezone.utc).isoformat()
    if persist:
        storage.save_forecast(
            pd.DataFrame({"date": future_frame["date"], **sales_q}),
            target="sales", model_version=models["sales"].metadata.model_version,
            generated_at=generated_at, db_path=db_path,
        )
        storage.save_forecast(
            pd.DataFrame({"date": future_frame["date"], **guests_q}),
            target="guest_count", model_version=models["guest_count"].metadata.model_version,
            generated_at=generated_at, db_path=db_path,
        )

    results = []
    for i, row in future_frame.iterrows():
        drivers = feature_engineering.top_drivers_for_row(row)
        results.append(
            ForecastDay(
                date=row["date"].strftime("%Y-%m-%d"),
                projected_sales=float(sales_q.loc[i, "p50"]),
                projected_guests=int(round(guests_q.loc[i, "p50"])),
                sales_low=float(sales_q.loc[i, "p10"]),
                sales_high=float(sales_q.loc[i, "p90"]),
                guests_low=int(round(guests_q.loc[i, "p10"])),
                guests_high=int(round(guests_q.loc[i, "p90"])),
                drivers=drivers,
            )
        )
    return results


def forecasts_to_dataframe(forecasts: list[ForecastDay]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": f.date,
                "projected_sales": round(f.projected_sales, 2),
                "sales_low": round(f.sales_low, 2),
                "sales_high": round(f.sales_high, 2),
                "projected_guests": f.projected_guests,
                "guests_low": f.guests_low,
                "guests_high": f.guests_high,
                "drivers": "; ".join(d["description"] for d in f.drivers) if f.drivers else "",
            }
            for f in forecasts
        ]
    )
