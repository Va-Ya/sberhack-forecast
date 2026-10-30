from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["week", "warehouse_name", "article"]
OUTPUT_COLUMNS = KEYS + ["quantity"]
SALES_COLUMNS = [
    "date", "composite_division", "cities_name", "warehouse_name",
    "category_lvl_1", "category_lvl_2", "category_lvl_3", "article",
    "characteristic", "quantity", "sales_sum", "discount_sum",
]
CATEGORIES = ["category_lvl_1", "category_lvl_2", "category_lvl_3"]
CUTOFF = pd.Timestamp("2026-07-26")
HORIZON = 9


class InputError(ValueError):
    def __init__(self, detail: str, status: int = 422, code: str = "VALIDATION_ERROR"):
        super().__init__(detail)
        self.detail, self.status, self.code = detail, status, code


def next_monday(date: pd.Timestamp) -> pd.Timestamp:
    return date.normalize() - pd.Timedelta(days=date.weekday()) + pd.Timedelta(days=7)


def validate_forecast(frame: pd.DataFrame, pairs: pd.DataFrame, first_week: pd.Timestamp) -> None:
    if list(frame.columns) != OUTPUT_COLUMNS:
        raise InputError(f"Expected output columns: {OUTPUT_COLUMNS}")
    if frame[OUTPUT_COLUMNS].isna().any().any() or frame.duplicated(KEYS).any():
        raise InputError("Forecast has missing values or duplicate keys")
    quantity = pd.to_numeric(frame.quantity, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(quantity).all() or (quantity < 0).any():
        raise InputError("Forecast quantity must be finite and nonnegative")
    expected = pd.MultiIndex.from_tuples([
        (week.strftime("%Y-%m-%d"), warehouse, article)
        for week in pd.date_range(first_week, periods=HORIZON, freq="7D")
        for warehouse, article in pairs.itertuples(index=False, name=None)
    ], names=KEYS)
    actual = pd.MultiIndex.from_frame(frame[KEYS])
    if len(actual) != len(expected) or not actual.sort_values().equals(expected.sort_values()):
        raise InputError("Forecast keys differ from the expected nine-week pair universe")
