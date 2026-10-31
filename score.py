"""Strict competition scorer. Never fills missing predictions or silently clips errors."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from forecast.contracts import KEYS, validate_forecast
from forecast.metrics import compute_metrics, leaderboard


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", required=True)
    parser.add_argument("--submissions", nargs="+", required=True)
    parser.add_argument("--baseline")
    parser.add_argument("--method", choices=["rank", "minmax"], default="minmax")
    parser.add_argument("--output", default="leaderboard.csv")
    args = parser.parse_args()
    truth = pd.read_csv(args.truth, dtype={"week": str, "warehouse_name": str, "article": str})
    pairs = truth[["warehouse_name", "article"]].drop_duplicates().sort_values(["warehouse_name", "article"])
    first = pd.to_datetime(truth.week).min()
    if first.weekday() != 0:
        raise ValueError("Truth weeks must start on Monday")
    validate_forecast(truth, pairs, first)
    order = truth.sort_values(KEYS)
    y = order.quantity.to_numpy().reshape(9, len(pairs))
    metrics = {}
    for path in args.submissions + ([args.baseline] if args.baseline else []):
        frame = pd.read_csv(path, dtype={"week": str, "warehouse_name": str, "article": str})
        validate_forecast(frame, pairs, first)
        name = "last" if path == args.baseline else Path(path).stem
        metrics[name] = compute_metrics(y, frame.sort_values(KEYS).quantity.to_numpy().reshape(y.shape))
    if args.method == "minmax" and not args.baseline:
        parser.error("--method minmax requires --baseline")
    # leaderboard computes both methods; a supplied baseline is explicitly named 'last'.
    fallback = next(iter(metrics))
    result = leaderboard(metrics, "last" if "last" in metrics else fallback)
    if not args.baseline:
        result = result.drop(columns="baseline_score")
    column = "rank_score" if args.method == "rank" else "baseline_score"
    result = result.sort_values(column, kind="stable")
    result.to_csv(args.output, index_label="submission")
    print(result.to_string())


if __name__ == "__main__":
    main()
