"""FastAPI web app: browser dashboard + JSON API for the forecasting engine.

Run locally with:  uvicorn src.api:app --reload
Deployed on Render via the repo's render.yaml (see README for details).
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src import config, exporter, feedback_loop, forecaster

logger = logging.getLogger(__name__)

WEB_DIR = config.PROJECT_ROOT / "web"

app = FastAPI(
    title="Austin Restaurant Forecasting Engine",
    description="Sales & guest count forecasting API for Austin, TX restaurants.",
    version="1.0.0",
)

if (WEB_DIR / "static").exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

# Tracks the CSV last used to (re)train, so the dashboard's "Retrain now"
# action doesn't require re-uploading a file. In-memory only -- fine for the
# single-instance deployment this app targets; a multi-worker/multi-instance
# deployment would need this persisted (e.g. alongside the SQLite DB).
_last_training_data_path: Optional[Path] = None


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


@app.get("/")
def index() -> FileResponse:
    index_path = WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web dashboard not found.")
    return FileResponse(str(index_path))


@app.get("/api/status")
def get_status() -> dict:
    models = forecaster.load_models()
    model_status = {}
    for target, model in models.items():
        if model is None:
            model_status[target] = {"trained": False}
        else:
            model_status[target] = {
                "trained": True,
                "model_version": model.metadata.model_version,
                "trained_at": model.metadata.trained_at,
                "training_rows": model.metadata.training_rows,
            }
    return {
        "models": model_status,
        "has_training_data_on_file": _last_training_data_path is not None,
        "sample_data_available": (config.RAW_DATA_DIR / "sample_sales_data.csv").exists(),
    }


@app.post("/api/train")
async def post_train(
    use_sample_data: bool = True,
    fetch_weather: bool = False,
    file: Optional[UploadFile] = File(None),
) -> dict:
    global _last_training_data_path

    if file is not None:
        suffix = Path(file.filename or "upload.csv").suffix or ".csv"
        tmp_dir = Path(tempfile.mkdtemp(prefix="forecaster_upload_"))
        data_path = tmp_dir / f"history{suffix}"
        with open(data_path, "wb") as out:
            shutil.copyfileobj(file.file, out)
    elif use_sample_data:
        data_path = config.RAW_DATA_DIR / "sample_sales_data.csv"
        if not data_path.exists():
            raise HTTPException(status_code=404, detail="No sample data bundled with this deployment.")
    elif _last_training_data_path is not None:
        data_path = _last_training_data_path
    else:
        raise HTTPException(
            status_code=400,
            detail="No CSV provided, no sample data available, and no prior training data to reuse.",
        )

    try:
        models = forecaster.train_all_targets(data_path, fetch_weather=fetch_weather)
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=422, detail=f"Training failed: {exc}")

    _last_training_data_path = data_path

    return {
        target: {
            "model_version": m.metadata.model_version,
            "training_rows": m.metadata.training_rows,
        }
        for target, m in models.items()
    }


@app.post("/api/forecast")
def post_forecast(req: ForecastRequest) -> dict:
    if req.end < req.start:
        raise HTTPException(status_code=400, detail="end must not be before start")
    try:
        results = forecaster.generate_forecast(req.start, req.end, fetch_weather=req.fetch_weather)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    return {"forecast": [r.as_dict() for r in results]}


@app.post("/api/log-actuals")
def post_log_actuals(req: ActualsRequest) -> dict:
    date_str = req.date.strftime("%Y-%m-%d")
    results = feedback_loop.log_actuals(date_str, req.sales, req.guests)
    drift_reports = [feedback_loop.detect_drift(t).__dict__ for t in feedback_loop.TARGETS]
    return {
        "results": [r.__dict__ for r in results],
        "drift": drift_reports,
    }


@app.post("/api/retrain")
def post_retrain() -> dict:
    if _last_training_data_path is None:
        raise HTTPException(
            status_code=400,
            detail="No training data on file yet -- train once via /api/train first.",
        )
    try:
        models = forecaster.train_all_targets(_last_training_data_path, fetch_weather=False)
    except Exception as exc:
        logger.exception("Retraining failed")
        raise HTTPException(status_code=422, detail=f"Retraining failed: {exc}")
    return {
        target: {
            "model_version": m.metadata.model_version,
            "training_rows": m.metadata.training_rows,
        }
        for target, m in models.items()
    }


@app.get("/api/accuracy/{target}")
def get_accuracy(target: str) -> dict:
    if target not in feedback_loop.TARGETS:
        raise HTTPException(status_code=404, detail=f"Unknown target '{target}'. Expected one of {feedback_loop.TARGETS}")
    return feedback_loop.trailing_accuracy_summary(target)


@app.post("/api/push-projections")
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
