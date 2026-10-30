from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .contracts import CATEGORIES, HORIZON, next_monday
from .data import Panel

CATEGORICAL = ["warehouse_id", "article_id", "category_lvl_1_id", "category_lvl_2_id", "category_lvl_3_id"]


def make_encoder(panel: Panel, cutoff: pd.Timestamp) -> dict[str, dict[str, int]]:
    available = panel.metadata[panel.metadata.date <= cutoff]
    result = {col: {v: i for i, v in enumerate(sorted(panel.pairs[col].unique()))}
              for col in ["warehouse_name", "article"]}
    result.update({col: {v: i for i, v in enumerate(sorted(available[col].unique()))} for col in CATEGORIES})
    return result


def complete_week_end(cutoff: pd.Timestamp) -> pd.Timestamp:
    if cutoff.weekday() == 6:
        return cutoff
    return cutoff - pd.Timedelta(days=cutoff.weekday() + 1)


def base_features(panel: Panel, cutoff: pd.Timestamp, encoder: dict) -> pd.DataFrame:
    """Only reads observations <= cutoff, including point-in-time category metadata."""
    cutoff = pd.Timestamp(cutoff).normalize()
    last = panel.index(cutoff)
    if last < 0 or last >= len(panel.dates):
        raise ValueError("Feature cutoff lies outside the panel")
    end = complete_week_end(cutoff)
    features: dict[str, np.ndarray | float] = {}
    for col, name in [("warehouse_name", "warehouse_id"), ("article", "article_id")]:
        features[name] = panel.pairs[col].map(encoder[col]).fillna(-1).to_numpy()
    meta = panel.metadata[panel.metadata.date <= cutoff].sort_values(
        ["date", "article"] + CATEGORIES, kind="stable").drop_duplicates("article", keep="last").set_index("article")
    for col in CATEGORIES:
        features[col + "_id"] = panel.pairs.article.map(meta[col]).map(encoder[col]).fillna(-1).to_numpy()
    for lag in range(1, 14):
        stop = end - pd.Timedelta(weeks=lag - 1)
        features[f"lag_{lag}"] = panel.total(stop - pd.Timedelta(days=6), stop)
    for window in [2, 4, 8, 13, 26]:
        weeks = np.stack([panel.total(end - pd.Timedelta(days=7 * k + 6), end - pd.Timedelta(days=7 * k))
                          for k in range(window)])
        features[f"mean_{window}"] = weeks.mean(axis=0)
        features[f"std_{window}"] = weeks.std(axis=0)
        features[f"zero_fraction_{window}"] = (weeks == 0).mean(axis=0)
    for days in [7, 28, 91, 180]:
        features[f"daily_rate_{days}"] = panel.total(cutoff - pd.Timedelta(days=days - 1), cutoff) * (7 / days)
    history = panel.sales[max(0, last - 179):last + 1]
    sold = history > 0
    positions = np.arange(len(history))[:, None]
    latest = np.where(sold, positions, -1).max(axis=0)
    features["days_since_sale"] = np.where(latest >= 0, len(history) - 1 - latest, 181)
    features["positive_days_180"] = sold.sum(axis=0)
    partial_days = 0 if cutoff.weekday() == 6 else cutoff.weekday() + 1
    features["partial_days"] = partial_days
    features["partial_quantity"] = panel.total(end + pd.Timedelta(days=1), cutoff)
    features["origin_weekday"] = cutoff.weekday()
    stock = panel.stock[max(0, last - 27):last + 1]
    known = ~np.isnan(stock)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        features["stock_mean_28"] = np.nanmean(stock, axis=0)
        features["stock_min_28"] = np.nanmin(stock, axis=0)
    indices = np.where(known, np.arange(len(stock))[:, None], -1).max(axis=0)
    features["stock_last"] = np.where(indices >= 0, stock[np.maximum(indices, 0), np.arange(len(indices))], np.nan)
    features["stock_known_days_28"] = known.sum(axis=0)
    features["stock_nonpositive_days_28"] = ((stock <= 0) & known).sum(axis=0)
    features["trend_4_13"] = features["mean_4"] / (features["mean_13"] + 1)
    for entity in ["warehouse_name", "article"]:
        totals = pd.Series(features["mean_8"]).groupby(panel.pairs[entity]).transform("sum").to_numpy()
        features[f"{entity}_total_8"] = totals
        features[f"{entity}_share_8"] = features["mean_8"] / (totals + 1)
    return pd.DataFrame(features, index=panel.pairs.index).astype(np.float32)


def horizon_features(base: pd.DataFrame, panel: Panel, cutoff: pd.Timestamp, horizon: int) -> pd.DataFrame:
    target = next_monday(cutoff) + pd.Timedelta(weeks=horizon - 1)
    result = base.copy()
    result["horizon"] = np.float32(horizon)
    result["target_month"] = np.float32(target.month)
    result["target_week"] = np.float32(target.isocalendar().week)
    result["target_year"] = np.float32(target.year)
    result["week_sin"] = np.float32(np.sin(2 * np.pi * target.dayofyear / 365.25))
    result["week_cos"] = np.float32(np.cos(2 * np.pi * target.dayofyear / 365.25))
    past = target - pd.Timedelta(weeks=52)
    result["seasonal_52"] = panel.total(past, past + pd.Timedelta(days=6))
    result["seasonal_52_available"] = np.float32(past >= panel.dates[0])
    # Never use an incomplete historical seasonal week (important for very short inputs).
    if past < panel.dates[0] or past + pd.Timedelta(days=6) > cutoff:
        result["seasonal_52"] = np.nan
        result["seasonal_52_available"] = np.float32(0)
    return result.astype(np.float32)


def baselines(panel: Panel, cutoff: pd.Timestamp) -> dict[str, np.ndarray]:
    end = complete_week_end(cutoff)
    weekly = np.stack([panel.total(end - pd.Timedelta(days=k * 7 + 6), end - pd.Timedelta(days=k * 7))
                       for k in range(8)])
    result = {"last": np.tile(weekly[0], (HORIZON, 1)),
              "mean4": np.tile(weekly[:4].mean(axis=0), (HORIZON, 1)),
              "mean8": np.tile(weekly.mean(axis=0), (HORIZON, 1))}
    seasonal = []
    for week in pd.date_range(next_monday(cutoff), periods=HORIZON, freq="7D"):
        old = week - pd.Timedelta(weeks=52)
        seasonal.append(panel.total(old, old + pd.Timedelta(days=6)) if old >= panel.dates[0] else weekly.mean(axis=0))
    result["seasonal"] = np.stack(seasonal)
    return result
