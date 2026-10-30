import io

import numpy as np
import pandas as pd
import pytest

from forecast.contracts import CATEGORIES
from forecast.data import Panel


@pytest.fixture
def panel():
    dates = pd.date_range("2024-01-01", "2026-07-31")
    pairs = pd.DataFrame({"warehouse_name": ["w", "w"], "article": ["a", "b"]})
    sales = np.zeros((len(dates), 2), dtype=np.float32)
    sales[:, 0] = 1
    stock = np.full_like(sales, np.nan)
    stock[:, 0] = 10
    meta = pd.DataFrame({"date": [dates[0], dates[0]], "article": ["a", "b"],
                         **{c: ["c", "c"] for c in CATEGORIES}})
    return Panel(dates, pairs, sales, stock, meta, {})


def csv_bytes(frame):
    return io.BytesIO(frame.to_csv(index=False).encode())


@pytest.fixture
def csv_frames():
    dates = pd.date_range("2026-01-01", periods=181)
    sales = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "warehouse_name": "w", "article": "a",
                          "quantity": 1.0, "sales_sum": 1.0, "discount_sum": np.nan,
                          "composite_division": "d", "cities_name": "city", "characteristic": "x",
                          **{c: "cat" for c in CATEGORIES}})
    stock = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "warehouse_name": "w", "article": "a", "stock": -1})
    return sales, stock
