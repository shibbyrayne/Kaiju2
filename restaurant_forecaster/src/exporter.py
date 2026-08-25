"""Webhook / REST push client for downstream app integration."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

MODEL_VERSION_LABEL = "restaurant-forecaster-ensemble"
DEFAULT_TIMEOUT_SECONDS = 15
MAX_RETRIES = 3


@dataclass
class PushResult:
    success: bool
    status_code: Optional[int]
    endpoint: str
    error: Optional[str] = None


def build_payload(
    daily_projections: list[dict],
    accuracy_metrics: dict,
    model_version: str,
) -> dict:
    """Build the structured JSON payload described in the spec.

    ``daily_projections`` entries are expected to already contain:
    date, projected_sales, projected_guests, conf_interval_low,
    conf_interval_high, drivers.
    """
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "daily_projections": daily_projections,
        "accuracy_metrics": accuracy_metrics,
    }


def push_projections(
    endpoint: str,
    api_key: str,
    daily_projections: list[dict],
    accuracy_metrics: dict,
    model_version: str = MODEL_VERSION_LABEL,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_retries: int = MAX_RETRIES,
) -> PushResult:
    """POST the forecast payload to a downstream webhook/REST endpoint."""
    payload = build_payload(daily_projections, accuracy_metrics, model_version)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
            if resp.status_code < 300:
                return PushResult(success=True, status_code=resp.status_code, endpoint=endpoint)
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            logger.warning("push-projections attempt %d/%d failed: %s", attempt, max_retries, last_error)
        except requests.RequestException as exc:
            last_error = str(exc)
            logger.warning("push-projections attempt %d/%d raised: %s", attempt, max_retries, last_error)

    return PushResult(success=False, status_code=None, endpoint=endpoint, error=last_error)
