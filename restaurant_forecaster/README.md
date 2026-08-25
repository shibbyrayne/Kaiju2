# Austin Restaurant Sales & Guest Count Forecasting Engine

A modular, production-grade CLI and API tool that ingests historical restaurant
sales and guest count data, enriches it with Austin, TX weather, holidays, and
major-event signals, forecasts daily sales/guest counts with recency-weighted
ensemble models, and adapts over time via a closed feedback loop.

## Features

- **Schema-validated ingestion** of historical CSV data (`pydantic`), with
  automatic gap-filling (missing dates treated as closures), outlier
  clipping, and closure detection.
- **Austin-specific feature engineering**: SXSW, ACL, F1 US Grand Prix, UT
  football home games, graduation weekends, the Austin Marathon, Pecan
  Street Festival, and Trail of Lights — each with a binary "active" flag
  plus an exponential proximity-decay feature. US federal holidays,
  Juneteenth, day-of-week/month/payday-cycle seasonality.
- **Live weather** via the free [Open-Meteo](https://open-meteo.com) API
  (historical archive + 16-day forecast), SQLite-cached locally.
- **Recency-weighted ensemble modeling**: a LightGBM quantile-regression
  model (non-linear feature interactions) blended with a
  Prophet-or-statsmodels seasonal baseline, trained with exponential
  recency sample weights so the trailing 90 days matter up to 4x more than
  older history.
- **P10 / P50 / P90 forecasts** for both sales and guest count, with a
  human-readable list of the day's key demand drivers.
- **Adaptive feedback loop**: log actual results, get instant error/bias
  metrics, and automatically retrain when sustained drift is detected over
  a trailing 7-14 day window.
- **Webhook / REST export** for pushing projections + trailing accuracy
  metrics into a downstream app.
- **CLI** (`typer` + `rich`), a **FastAPI** JSON API, and a **browser
  dashboard** (served by the same FastAPI app) all exposing the same
  functionality — deployable as-is to [Render](https://render.com).

## Project layout

```
Kaiju2/
├── render.yaml                    # Render Blueprint (deploy from repo root)
└── restaurant_forecaster/
    ├── data/
    │   ├── raw/                  # your historical CSVs go here
    │   ├── external/              # austin_events.json, weather_cache.db
    │   └── processed/             # forecaster.db, trained model artifacts
    ├── web/                       # browser dashboard (served by FastAPI)
    │   ├── index.html
    │   └── static/ (app.js, style.css)
    ├── src/
    │   ├── config.py               # paths, constants, tunables
    │   ├── data_loader.py          # schema validation + cleaning
    │   ├── austin_calendar.py      # events, holidays, seasonality
    │   ├── weather_client.py       # Open-Meteo client + SQLite cache
    │   ├── feature_engineering.py  # combines calendar+weather, recency weights
    │   ├── models/
    │   │   ├── base.py
    │   │   ├── lgbm_model.py       # LightGBM quantile regressor
    │   │   ├── prophet_model.py    # Prophet / statsmodels seasonal baseline
    │   │   └── ensemble.py         # blends the two, save/load
    │   ├── evaluation.py           # WAPE, MAPE, bias, interval coverage
    │   ├── feedback_loop.py        # log actuals, drift detection, retraining
    │   ├── storage.py              # SQLite persistence layer
    │   ├── exporter.py             # webhook/REST push client
    │   ├── forecaster.py           # shared train/forecast orchestration
    │   ├── cli.py                  # typer CLI entrypoint
    │   └── api.py                  # FastAPI app: dashboard + JSON API
    ├── tests/
    ├── requirements.txt
    └── README.md
```

## Setup

```bash
cd restaurant_forecaster
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prophet is intentionally **not** a hard dependency — it requires a working
C++/CmdStan build toolchain and can fail to install in constrained
environments. If it's not installed (or fails to fit), the seasonal
baseline model automatically falls back to a statsmodels Holt-Winters
exponential smoother, and finally to a recency-weighted day-of-week average
if even that isn't available. To enable Prophet, uncomment it in
`requirements.txt` and reinstall.

## Data schema

Historical CSVs must include:

| column         | type            | required |
|----------------|-----------------|----------|
| `date`         | `YYYY-MM-DD`    | yes      |
| `sales`        | float           | yes      |
| `guest_count`  | int             | yes      |
| `day_of_week`  | string          | optional (derived automatically if omitted) |
| `meal_period`  | Lunch/Dinner/... | optional |
| `promotions_active` | bool       | optional |

A synthetic ~2.5 year sample dataset is provided at
`data/raw/sample_sales_data.csv` for trying the tool end-to-end.

## CLI usage

```bash
# Train (or retrain from scratch) the sales & guest_count models
python -m src.cli train --data data/raw/sample_sales_data.csv

# Generate a forecast (add --no-weather to skip live weather calls)
python -m src.cli forecast --start 2026-10-01 --end 2026-10-13 --export forecast.csv

# Log actual results and update the feedback loop (auto-retrains on drift
# if --data is supplied)
python -m src.cli log-actuals --date 2026-10-01 --sales 5300 --guests 155 \
    --data data/raw/sample_sales_data.csv

# View trailing 7/30/90-day accuracy
python -m src.cli accuracy --target sales

# Push a forecast + accuracy metrics to a downstream webhook
python -m src.cli push-projections --endpoint https://example.com/webhook \
    --api-key $RESTAURANT_FORECASTER_API_KEY --start 2026-10-01 --end 2026-10-13
```

## Web app / API server

```bash
uvicorn src.api:app --reload
```

Open `http://localhost:8000/` for the dashboard: train models (bundled
sample data or your own CSV upload), generate a forecast with a chart and
driver breakdown, log actuals, watch trailing accuracy, and push
projections to a downstream webhook — all from the browser.

The same functionality is available as JSON under `/api/*`:

| Method | Path                     | Purpose |
|--------|--------------------------|---------|
| GET    | `/health`                | Liveness check (used by Render) |
| GET    | `/api/status`            | Whether models are trained, versions, row counts |
| POST   | `/api/train`             | Train on the bundled sample data or an uploaded CSV |
| POST   | `/api/retrain`           | Retrain on the last dataset used (no re-upload) |
| POST   | `/api/forecast`          | `{start, end, fetch_weather}` → daily P10/P50/P90 forecast |
| POST   | `/api/log-actuals`       | `{date, sales, guests}` → error + drift report |
| GET    | `/api/accuracy/{target}` | Trailing 7/30/90-day WAPE/MAPE/bias |
| POST   | `/api/push-projections`  | Generate + push a forecast to a webhook |

## Deploying to Render

This repo includes a Render [Blueprint](https://render.com/docs/blueprint-spec)
(`render.yaml` at the repo root, `rootDir: restaurant_forecaster`):

1. Push this repo to GitHub (already done if you're reading this on a branch).
2. In the Render dashboard: **New +** → **Blueprint**, select the repo/branch.
3. Render reads `render.yaml` and provisions a free Python web service —
   build command `pip install -r requirements.txt`, start command
   `uvicorn src.api:app --host 0.0.0.0 --port $PORT`, health check `/health`.
4. Once deployed, open the service URL and click **Train models** in the
   dashboard (uses the bundled sample data by default, or upload your own
   CSV) before generating a forecast.

**Persistence caveat:** on Render's free plan the filesystem is rebuilt on
every deploy, so trained models and the accuracy/error-log SQLite DB don't
survive a redeploy — just retrain again afterward. To persist them, upgrade
the service to a paid plan and attach a
[persistent disk](https://render.com/docs/disks) (the commented-out `disk:`
block in `render.yaml` shows how, paired with the `FORECASTER_PROCESSED_DIR`
env var so the app writes its DB/models there instead of the app directory).

## Recency weighting

Sample weights follow `w_i = exp(-ln(2)/half_life * age_days)`, rescaled so
observations from the trailing 90 days receive up to 4x the weight of the
oldest observations in the training set (`src/feature_engineering.py:compute_recency_weights`,
tunable via `config.RECENCY_HALF_LIFE_DAYS` / `RECENCY_MAX_MULTIPLIER`).

## Adaptive feedback loop

`log-actuals` diffs the actual outcome against the forecast stored for that
date, persists the error to `data/processed/forecaster.db`, and checks
trailing bias over the last 7-14 days (`config.DRIFT_LOOKBACK_DAYS`,
`config.DRIFT_BIAS_THRESHOLD_PCT`). If bias exceeds the threshold, the
ensemble is automatically retrained on the latest history + logged actuals.

## Testing

```bash
pytest -q
```

Tests cover schema validation/cleaning, calendar/event feature correctness,
recency weighting, the Open-Meteo client (HTTP calls mocked), the
LightGBM+seasonal ensemble, evaluation metrics, the feedback loop's drift
detection, the webhook exporter, and an end-to-end train→forecast
integration test.

## Notes & caveats

- **UT Austin football / graduation dates** in `data/external/austin_events.json`
  are best-effort estimates for demonstration purposes; swap in a live UT
  Athletics feed for production use.
- **Weather** calls Open-Meteo's free, keyless API; no `.env` configuration
  is required, but network access is needed unless you pass `--no-weather`.
- Confidence intervals are P10/P50/P90 (an 80% interval); adjust
  `config.CONFIDENCE_LOWER_QUANTILE` / `CONFIDENCE_UPPER_QUANTILE` for a
  90% or 95% interval instead.
