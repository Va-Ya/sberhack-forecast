from __future__ import annotations

import numpy as np
import pandas as pd

WEIGHTS = {"smape": 0.50, "wape": 0.20, "rmse": 0.15, "bias": 0.10, "rmsle": 0.05}


def compute_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """Arrays have shape (nine weeks, series); series receive equal SMAPE/Bias weight."""
    y, p = np.asarray(truth, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if y.shape != p.shape or y.ndim != 2 or not y.size:
        raise ValueError("Expected equal nonempty (weeks, series) arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all() or (y < 0).any() or (p < 0).any():
        raise ValueError("Metrics require finite nonnegative values")
    error = np.abs(y - p)
    denominator = (y + p) / 2
    ratio = np.divide(error, denominator, out=np.zeros_like(y), where=denominator > 0)
    mean_y, mean_p = y.mean(axis=0), p.mean(axis=0)
    nonzero = mean_y > 0
    return {
        "smape": float(ratio.mean(axis=0).mean() * 100),
        "wape": float(error.sum() / y.sum() * 100) if y.sum() else (0.0 if not error.sum() else 100.0),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias": float((np.abs(mean_p[nonzero] - mean_y[nonzero]) / mean_y[nonzero]).mean() * 100)
        if nonzero.any() else 0.0,
        "rmsle": float(np.sqrt(np.mean((np.log1p(y) - np.log1p(p)) ** 2))),
    }


def leaderboard(metrics: dict[str, dict[str, float]], baseline: str = "last") -> pd.DataFrame:
    frame = pd.DataFrame.from_dict(metrics, orient="index")
    rank = (frame.rank(method="average") - 1) / max(len(frame) - 1, 1)
    normalized = frame.copy()
    for col in WEIGHTS:
        base = frame.loc[baseline, col]
        normalized[col] = (frame[col] / base).clip(upper=1) if base else (frame[col] != 0).astype(float)
    frame["rank_score"] = sum(WEIGHTS[col] * rank[col] for col in WEIGHTS)
    frame["baseline_score"] = sum(WEIGHTS[col] * normalized[col] for col in WEIGHTS)
    return frame.sort_values(["rank_score", "smape"], kind="stable")
