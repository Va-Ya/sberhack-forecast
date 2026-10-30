from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import numpy as np
import pandas as pd

from .contracts import CUTOFF, next_monday
from .data import Panel, load_archive, load_inputs, sha256_file
from .features import baselines
from .metrics import compute_metrics, leaderboard
from .model import (
    Predictor,
    TrainingConfig,
    candidates,
    fit_models,
    forecast_frame,
    model_predictions,
    save_bundle,
)

FOLDS = ["2026-01-19", "2026-03-23", "2026-05-25"]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def log_event(event: dict) -> None:
    Path("reports").mkdir(exist_ok=True)
    with Path("reports/experiments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(), **event}) + "\n")


def get_panel(args) -> Panel:
    pairs = pd.read_csv(args.pairs, dtype=str) if args.pairs else None
    archive_hash = sha256_file(Path(args.archive))
    signature = {"archive": archive_hash, "cutoff": str(CUTOFF.date()),
                 "pairs": sha256_file(Path(args.pairs)) if args.pairs else "cartesian", "schema": 2}
    cache = Path("data/cache/panel.npz")
    manifest = cache.with_suffix(".json")
    if cache.exists() and manifest.exists():
        stored = json.loads(manifest.read_text())
        if stored == signature or (not args.pairs and stored == {**signature, "schema": 1}):
            panel = Panel.load(cache)
            if not args.pairs:
                panel.audit.update(pair_policy="cartesian", requested_pairs=None)
            return panel
    print("Reading and validating input CSVs ...", flush=True)
    panel = load_archive(Path(args.archive), CUTOFF, pairs=pairs)
    panel.save(cache)
    write_json(manifest, signature)
    write_json(Path("reports/data_audit.json"), panel.audit)
    return panel


def prepare(args) -> Panel:
    panel = get_panel(args)
    Path("outputs").mkdir(exist_ok=True)
    for name, values in baselines(panel, CUTOFF).items():
        forecast_frame(values[:, panel.output_indices], panel.output_pairs, next_monday(CUTOFF)).to_csv(
            f"outputs/baseline_{name}.csv", index=False)
    print(f"Baselines ready: {len(panel.output_pairs) * 9:,} rows per file", flush=True)
    log_event({"event": "baselines_ready", "pairs": len(panel.pairs), "audit": panel.audit})
    return panel


def train(args) -> None:
    started = time.perf_counter()
    panel = prepare(args)
    config = TrainingConfig(iterations=args.iterations, pairs_per_origin=args.pairs_per_origin, threads=args.threads)
    fold_metrics = {}
    selected = None
    for fold_number, start in enumerate(FOLDS):
        first_week = pd.Timestamp(start)
        origin = first_week - pd.Timedelta(days=1)
        folder = Path("reports/folds") / start
        config_path = folder / "config.json"
        # Resume completed work only when all training parameters and data identities match.
        expected_identity = {"training": config.__dict__, "data_sha256": panel.audit["archive_sha256"]}
        if config_path.exists():
            saved = json.loads(config_path.read_text())
            if saved.get("identity") != expected_identity:
                raise RuntimeError(f"Different run already exists in {folder}; use a new report directory")
            predictor = Predictor.load(folder)
            models, encoder, columns = predictor.models, saved["encoder"], saved["features"]
        else:
            models, encoder, columns, nrows = fit_models(panel, origin, config)
            save_bundle(folder, models, encoder, columns, "log", config, panel,
                        {"identity": expected_identity, "cutoff": str(origin.date()), "training_rows": nrows})
        learned = model_predictions(models, encoder, columns, panel, origin, config.threads)
        predictions = candidates(baselines(panel, origin), learned)
        truth = np.stack([panel.total(week, week + pd.Timedelta(days=6))
                          for week in pd.date_range(first_week, periods=9, freq="7D")])
        metrics = {name: compute_metrics(truth[:, panel.output_indices], values[:, panel.output_indices])
                   for name, values in predictions.items()}
        fold_metrics[start] = metrics
        table = leaderboard(metrics)
        table.to_csv(folder / "metrics.csv", index_label="candidate")
        print(f"Fold {start}\n{table.to_string(float_format=lambda v: f'{v:.4f}')}", flush=True)
        log_event({"event": "fold_complete", "fold": start, "role": "control" if fold_number == 2 else "selection",
                   "metrics": metrics})
        Path("reports/predictions").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path("reports/predictions") / f"{start}.npz", truth=truth, **predictions)
        if fold_number == 1:
            scores = pd.concat([leaderboard(fold_metrics[f]) for f in FOLDS[:2]]).groupby(level=0).mean()
            scores = scores.sort_values(["rank_score", "smape"], kind="stable")
            scores.to_csv("reports/selection.csv", index_label="candidate")
            selected = str(scores.index[0])
            write_json(Path("reports/selection.json"), {"selected": selected,
                       "selection_folds": FOLDS[:2], "control_fold": FOLDS[2],
                       "policy": "mean rank score on first two folds; control not used for tuning"})
            print(f"Locked selection BEFORE control: {selected}", flush=True)
    assert selected is not None
    kinds = ("raw",) if selected.startswith("raw") else (("log",) if selected.startswith("log") else ())
    models, encoder, columns, nrows = fit_models(panel, CUTOFF, config, kinds) if kinds else ({}, {}, [], 0)
    staging = Path("model-next")
    save_bundle(staging, models, encoder, columns, selected, config, panel,
                {"training_rows": nrows, "dependencies": {"python": platform.python_version(),
                 "numpy": np.__version__, "pandas": pd.__version__, "lightgbm": lightgbm.__version__}})
    if args.pairs:
        panel.output_pairs.to_csv(staging / "allowed_pairs.csv", index=False)
    predictor = Predictor.load(staging)
    forecast = predictor.predict_panel(panel)
    repeated = predictor.predict_panel(panel)
    pd.testing.assert_frame_equal(forecast, repeated, check_exact=True)
    forecast.to_csv("outputs/submission.csv", index=False, float_format="%.8f")
    directory = Path("model")
    if directory.exists():
        backup = Path("reports/model-backup") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.copytree(directory, backup)
    directory.mkdir(exist_ok=True)
    for file in staging.iterdir():
        shutil.copy2(file, directory / file.name)
    # Config lists the exact active models; unused old native text files are harmless but excluded from releases.
    summary = {"selected": selected, "folds": fold_metrics,
               "elapsed_seconds": time.perf_counter() - started, "deterministic_repeat": True,
               "submission_sha256": sha256_file(Path("outputs/submission.csv")),
               "rows": len(forecast), "config": config.__dict__}
    write_json(Path("reports/training_summary.json"), summary)
    log_event({"event": "model_promoted", **summary})
    print(f"READY: outputs/submission.csv, selected={selected}, seconds={summary['elapsed_seconds']:.1f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="SberHack reproducible forecasting")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["prepare", "train"]:
        item = sub.add_parser(name)
        item.add_argument("--archive", default="hakaton.zip")
        item.add_argument("--pairs", help="Organizer CSV with warehouse_name,article")
        if name == "train":
            item.add_argument("--iterations", type=int, default=180)
            item.add_argument("--pairs-per-origin", type=int, default=600)
            item.add_argument("--threads", type=int, default=2)
    predict = sub.add_parser("predict")
    predict.add_argument("--sales", required=True)
    predict.add_argument("--stock", required=True)
    predict.add_argument("--model", default="model")
    predict.add_argument("--output", default="forecast.csv")
    predict.add_argument("--pairs")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    else:
        pairs_path = Path(args.pairs) if args.pairs else Path(args.model) / "allowed_pairs.csv"
        pairs = pd.read_csv(pairs_path, dtype=str) if pairs_path.exists() else None
        with open(args.sales, "rb") as sales, open(args.stock, "rb") as stock:
            panel = load_inputs(sales, stock, pairs=pairs)
        Predictor.load(Path(args.model)).predict_panel(panel).to_csv(args.output, index=False, float_format="%.8f")


if __name__ == "__main__":
    main()
