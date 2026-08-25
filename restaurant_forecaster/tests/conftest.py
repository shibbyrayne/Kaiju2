import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_daily_df():
    """~15 months of synthetic daily sales/guest_count data, no external calls needed."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2024-01-01", "2025-03-31", freq="D")
    rows = []
    for d in dates:
        weekend_lift = 1.3 if d.dayofweek >= 4 else 1.0
        base = 4000 * weekend_lift * float(rng.normal(1.0, 0.05))
        sales = max(base, 0)
        guests = int(round(sales / 32))
        rows.append({"date": d.strftime("%Y-%m-%d"), "sales": round(sales, 2), "guest_count": guests})
    return pd.DataFrame(rows)
