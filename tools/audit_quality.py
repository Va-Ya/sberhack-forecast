"""Independent source-data and frozen-backtest audit. Does not promote or tune models."""
from __future__ import annotations

import json
import math
import subprocess
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.contracts import CATEGORIES, CUTOFF
from forecast.data import Panel, sha256_file
from forecast.features import baselines
from forecast.model import Predictor, candidates, model_predictions

DESTINATION = Path("reports/current_audit")
FOLDS = ["2026-01-19", "2026-03-23", "2026-05-25"]


def reference_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Scalar, per-series implementation independent of forecast.metrics."""
    absolute, squared, log_squared, smapes, biases = [], [], [], [], []
    for index in range(y.shape[1]):
        truth = [float(v) for v in y[:, index]]
        prediction = [float(v) for v in p[:, index]]
        ratios = []
        for actual, estimate in zip(truth, prediction, strict=True):
            error = abs(actual - estimate)
            absolute.append(error)
            squared.append(error * error)
            log_squared.append((math.log1p(actual) - math.log1p(estimate)) ** 2)
            ratios.append(2 * error / (actual + estimate) if actual + estimate else 0.0)
        smapes.append(math.fsum(ratios) / len(truth) * 100)
        actual_sum = math.fsum(truth)
        if actual_sum:
            biases.append(abs(math.fsum(prediction) - actual_sum) / actual_sum * 100)
    total = math.fsum(float(v) for v in y.flat)
    error_sum = math.fsum(absolute)
    return {
        "smape": math.fsum(smapes) / len(smapes),
        "wape": error_sum / total * 100 if total else (100.0 if error_sum else 0.0),
        "rmse": math.sqrt(math.fsum(squared) / y.size),
        "bias": math.fsum(biases) / len(biases) if biases else 0.0,
        "rmsle": math.sqrt(math.fsum(log_squared) / y.size),
    }


def read_raw(panel: Panel):
    """Rebuild source matrices via indexed addition, not the production groupby pipeline."""
    pair_index = pd.MultiIndex.from_frame(panel.pairs)
    expected = np.zeros(panel.sales.shape, dtype=np.float64)
    first_sale = np.full(len(panel.pairs), len(panel.dates), dtype=np.int32)
    counts = {}
    metadata_tuples: dict[str, set] = {}
    short_metadata_parts = []
    short_end = pd.Timestamp(FOLDS[-1]) - pd.Timedelta(days=1)
    short_start = short_end - pd.Timedelta(days=179)
    with zipfile.ZipFile("hakaton.zip") as source:
        sales_name = next(n for n in source.namelist() if Path(n).name.startswith("sales_") and n.endswith(".csv"))
        columns = ["date", "warehouse_name", "article", "quantity"] + CATEGORIES
        rows = after = 0
        for chunk in pd.read_csv(source.open(sales_name), usecols=columns, chunksize=100_000,
                                 dtype={c: str for c in CATEGORIES}):
            rows += len(chunk)
            after += int((chunk.date > "2026-07-26").sum())
            chunk = chunk[chunk.date <= "2026-07-26"].copy()
            recent = chunk[(chunk.date >= str(short_start.date())) & (chunk.date <= str(short_end.date()))]
            short_metadata_parts.append(recent[["date", "article"] + CATEGORIES].drop_duplicates())
            d = (pd.to_datetime(chunk.date) - panel.dates[0]).dt.days.to_numpy()
            p = pair_index.get_indexer(pd.MultiIndex.from_frame(chunk[["warehouse_name", "article"]]))
            assert (p >= 0).all() and (d >= 0).all()
            np.add.at(expected, (d, p), chunk.quantity.to_numpy(dtype=np.float64))
            np.minimum.at(first_sale, p, d)
            unique = chunk[["article"] + CATEGORIES].fillna("__missing__").drop_duplicates()
            for article, *categories in unique.itertuples(index=False, name=None):
                metadata_tuples.setdefault(article, set()).add(tuple(categories))
        difference = np.abs(expected - panel.sales)
        np.testing.assert_allclose(expected, panel.sales, rtol=1e-6, atol=1e-6)
        counts["sales"] = {"source_rows": rows, "excluded_after_cutoff": after,
                           "max_abs_cache_difference": float(difference.max()),
                           "category_tuple_conflicts": sum(len(v) > 1 for v in metadata_tuples.values())}
        truths = {}
        for fold in FOLDS:
            values = []
            for week in pd.date_range(fold, periods=9, freq="7D"):
                start = int((week - panel.dates[0]).days)
                values.append(expected[start:start + 7].sum(axis=0))
            truths[fold] = np.stack(values)
        del expected, difference
        expected_stock = np.zeros(panel.stock.shape, dtype=np.float64)
        seen = np.zeros(panel.stock.shape, dtype=bool)
        stock_name = next(n for n in source.namelist() if Path(n).name.startswith("stock_") and n.endswith(".csv"))
        rows = after = valid_rows = negatives = 0
        for chunk in pd.read_csv(source.open(stock_name), chunksize=100_000):
            rows += len(chunk)
            after += int((chunk.date > "2026-07-26").sum())
            chunk = chunk[(chunk.date <= "2026-07-26") & (chunk.date >= str(panel.dates[0].date()))].copy()
            if "article" not in chunk:
                chunk = chunk.rename(columns={"char_article": "article"})
            d = (pd.to_datetime(chunk.date) - panel.dates[0]).dt.days.to_numpy()
            p = pair_index.get_indexer(pd.MultiIndex.from_frame(chunk[["warehouse_name", "article"]]))
            assert (p >= 0).all() and (d >= 0).all()
            np.add.at(expected_stock, (d, p), chunk.stock.to_numpy(dtype=np.float64))
            seen[d, p] = True
            valid_rows += len(chunk)
            negatives += int((chunk.stock < 0).sum())
        np.testing.assert_array_equal(seen, ~np.isnan(panel.stock))
        np.testing.assert_allclose(expected_stock[seen], panel.stock[seen], rtol=1e-6, atol=1e-6)
        counts["stock"] = {"source_rows": rows, "excluded_after_cutoff": after,
                           "negative_rows_within_panel": negatives,
                           "duplicate_rows_within_panel": int(valid_rows - seen.sum()),
                           "max_abs_cache_difference": float(np.abs(expected_stock[seen] - panel.stock[seen]).max()),
                           "missing_mask_equal": True}
    short_metadata = pd.concat(short_metadata_parts, ignore_index=True)
    short_metadata[CATEGORIES] = short_metadata[CATEGORIES].fillna("__missing__").astype(str)
    short_metadata["date"] = pd.to_datetime(short_metadata.date)
    short_metadata = short_metadata.sort_values("date").drop_duplicates(["article"] + CATEGORIES)
    return truths, counts, first_sale, short_metadata


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    panel = Panel.load(Path("data/cache/panel.npz"))
    assert panel.dates[-1] == CUTOFF
    config = json.loads(Path("model/config.json").read_text())
    data_hash = sha256_file(Path("hakaton.zip"))
    assert data_hash == config["audit"]["archive_sha256"]
    print("Rebuilding sales and stock directly from the original archive", flush=True)
    truths, source_audit, first_sale, short_metadata = read_raw(panel)
    saved_summary = json.loads(Path("reports/training_summary.json").read_text())
    selected = config["selected"]
    rows, horizon_rows, slice_rows = [], [], []
    fold_checks = {}
    for fold in FOLDS:
        origin = pd.Timestamp(fold) - pd.Timedelta(days=1)
        frozen = Predictor.load(Path("reports/folds") / fold)
        assert pd.Timestamp(frozen.config["cutoff"]) == origin
        learned = model_predictions(frozen.models, frozen.config["encoder"], frozen.config["features"],
                                    panel, origin, 2)
        fresh = candidates(baselines(panel, origin), learned)
        with np.load(Path("reports/predictions") / f"{fold}.npz") as previous:
            np.testing.assert_allclose(truths[fold], previous["truth"], rtol=1e-6, atol=1e-6)
            maximum = 0.0
            for name, values in fresh.items():
                delta = float(np.max(np.abs(values - previous[name])))
                maximum = max(maximum, delta)
                np.testing.assert_array_equal(values, previous[name])
                metrics = reference_metrics(truths[fold], values)
                for key, value in metrics.items():
                    assert math.isclose(value, saved_summary["folds"][fold][name][key], rel_tol=1e-8, abs_tol=1e-8)
                rows.append({"fold": fold, "candidate": name, **metrics})
        y, p = truths[fold], fresh[selected]
        fold_checks[fold] = {"raw_truth_equal": True, "frozen_predictions_equal": True,
                             "max_abs_prediction_difference": maximum, "independent_metrics_equal": True,
                             "training_cutoff": str(origin.date()),
                             "series_first_sale_after_origin": int((first_sale > panel.index(origin)).sum()),
                             "signed_total_bias_percent": float((p.sum() / y.sum() - 1) * 100),
                             "actual_total": float(y.sum()), "predicted_total": float(p.sum()),
                             "zero_forecast_fraction": float((p == 0).mean())}
        for horizon in range(9):
            horizon_rows.append({"fold": fold, "horizon": horizon + 1,
                                 **reference_metrics(y[horizon:horizon + 1], p[horizon:horizon + 1])})
        masks = {"all_zero": y.sum(axis=0) == 0,
                 "intermittent": ((y == 0).mean(axis=0) >= 0.5) & (y.sum(axis=0) > 0),
                 "regular": (y == 0).mean(axis=0) < 0.5}
        for segment, mask in masks.items():
            if mask.any():
                slice_rows.append({"fold": fold, "segment": segment, "series": int(mask.sum()),
                                   **reference_metrics(y[:, mask], p[:, mask])})
        print(f"Independent metrics and frozen model verified: {fold}", flush=True)
    # Also assess the same frozen control model with only 180 days of input history.
    origin = pd.Timestamp(FOLDS[-1]) - pd.Timedelta(days=1)
    start = max(0, panel.index(origin) - 179)
    end = panel.index(origin) + 1
    short = Panel(panel.dates[start:end], panel.pairs, panel.sales[start:end], panel.stock[start:end],
                  short_metadata, {})
    frozen = Predictor.load(Path("reports/folds") / FOLDS[-1])
    short_values = candidates(baselines(short, origin), model_predictions(frozen.models, frozen.config["encoder"],
                              frozen.config["features"], short, origin, 2))[selected]
    short_metrics = reference_metrics(truths[FOLDS[-1]], short_values)
    rows.append({"fold": FOLDS[-1], "candidate": selected + "_180day_history", **short_metrics})
    pd.DataFrame(rows).to_csv(DESTINATION / "independent_metrics.csv", index=False)
    pd.DataFrame(horizon_rows).to_csv(DESTINATION / "horizon_metrics.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(DESTINATION / "slice_metrics.csv", index=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    audit = {"status": "passed", "time_utc": datetime.now(timezone.utc).isoformat(), "source_commit": commit,
             "elapsed_seconds": time.perf_counter() - started, "archive_sha256": data_hash,
             "model_sha256": sha256_file(Path("model/log.txt")), "source_data": source_audit,
             "folds": fold_checks, "short_history_control_metrics": short_metrics,
             "assumptions": ["fixed final warehouse/article universe across all backtests", "Cartesian pairs, no organizer reference"],
             "competition_test_truth_available": False,
             "note": "No model tuning or artifact promotion in this audit; three historical windows only."}
    (DESTINATION / "independent_quality.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
