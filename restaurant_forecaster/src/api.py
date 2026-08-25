"""Optional local FastAPI server exposing forecast/log-actuals/push endpoints.

Run with:  uvicorn src.api:app --reload
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src import exporter, feedback_loop, forecaster

app = FastAPI(
    title="Austin Restaurant Forecasting Engine",
    description="Sales & guest count forecasting API for Austin, TX restaurants.",
    version="1.0.0",
)


class ForecastRequest(BaseModel):
    start: date
    end: date
    fetch_weather: bool = True


class ActualsRequest(BaseModel):
    date: date
    sales: float = Field(ge=0)
    guests: int = Field(ge=0)


class PushRequest(BaseModel):
    endpoint: str
    api_key: str
    start: date
    end: date
    fetch_weather: bool = True


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/forecast")
def post_forecast(req: ForecastRequest) -> dict:
    if req.end < req.start:
        raise HTTPException(status_code=400, detail="end must not be before start")
    try:
        results = forecaster.generate_forecast(req.start, req.end, fetch_weather=req.fetch_weather)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"forecast": [r.as_dict() for r in results]}


@app.post("/log-actuals")
def post_log_actuals(req: ActualsRequest) -> dict:
    date_str = req.date.strftime("%Y-%m-%d")
    results = feedback_loop.log_actuals(date_str, req.sales, req.guests)
    drift_reports = [feedback_loop.detect_drift(t).__dict__ for t in feedback_loop.TARGETS]
    return {
        "results": [r.__dict__ for r in results],
        "drift": drift_reports,
    }


@app.get("/accuracy/{target}")
def get_accuracy(target: str) -> dict:
    if target not in feedback_loop.TARGETS:
        raise HTTPException(status_code=404, detail=f"Unknown target '{target}'. Expected one of {feedback_loop.TARGETS}")
    return feedback_loop.trailing_accuracy_summary(target)


@app.post("/push-projections")
def post_push_projections(req: PushRequest) -> dict:
    try:
        results = forecaster.generate_forecast(req.start, req.end, fetch_weather=req.fetch_weather)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    daily_projections = [r.as_dict() for r in results]
    accuracy_metrics = {t: feedback_loop.trailing_accuracy_summary(t) for t in feedback_loop.TARGETS}
    result = exporter.push_projections(req.endpoint, req.api_key, daily_projections, accuracy_metrics)

    if not result.success:
        raise HTTPException(status_code=502, detail=result.error)
    return {"success": True, "status_code": result.status_code}
