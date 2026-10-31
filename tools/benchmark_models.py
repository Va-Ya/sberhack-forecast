"""Reproduce the LightGBM research on two selection windows, never the control window.

Example (dependencies come from the project's requirements-train.txt):
    python benchmark_portable.py --project ./solution --output ./research-run --threads 2

For the historical 65-feature experiment, also supply its code-only source snapshot:
    --source-snapshot ./lightgbm_source_snapshot.zip
The current project source is used otherwise, and its hashes identify the new run.
No application model, submission or production report is replaced by this script.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

FOLDS = ("2026-01-19", "2026-03-23")
SEED = 2026
PAIRS_PER_ORIGIN = 1200
ORIGIN_STEP_DAYS = 14
PREFIXES = (300, 600, 900)
THRESHOLDS = (0.0, 0.5, 0.75)
FAMILIES = {
    "log_deep": {"objective": "regression", "rounds": 1200, "log": True},
    "log_recent": {"objective": "regression", "rounds": 1200, "log": True, "recent": True},
    "tweedie": {"objective": "tweedie", "tweedie_variance_power": 1.3, "rounds": 1000},
    "poisson": {"objective": "poisson", "rounds": 1000},
    "l2": {"objective": "regression", "rounds": 1000},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def common_parameters(threads: int) -> dict:
    return {
        "learning_rate": 0.035, "num_leaves": 63, "max_bin": 127,
        "min_data_in_leaf": 100, "lambda_l2": 8.0, "num_threads": threads,
        "seed": SEED, "deterministic": True, "force_col_wise": True, "verbosity": -1,
    }


def extract_source(snapshot: Path, destination: Path) -> Path:
    """Extract the expected code-only archive without trusting arbitrary archive paths."""
    with zipfile.ZipFile(snapshot) as archive:
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError("Source snapshot contains a path outside its extraction directory")
            if not info.is_dir() and not (info.filename.startswith("forecast/") and target.suffix == ".py"):
                raise ValueError("Source snapshot must contain only forecast/*.py")
        archive.extractall(destination)
    return destination


def run(args: argparse.Namespace, source: Path) -> None:
    # These imports intentionally follow source selection, so historical feature code is reproducible.
    sys.path.insert(0, str(source))
    from forecast.contracts import CUTOFF, next_monday
    from forecast.data import Panel, load_archive
    from forecast.features import CATEGORICAL, base_features, baselines, horizon_features, make_encoder
    from forecast.metrics import compute_metrics, leaderboard
    from forecast.model import TrainingConfig, training_matrix

    project = args.project.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory to avoid mixing experiment runs")
    panel_path = args.panel.resolve() if args.panel else project / "data/cache/panel.npz"
    if panel_path.exists():
        panel = Panel.load(panel_path)
        data_identity = {"panel_sha256": sha256(panel_path), "kind": "panel"}
    elif args.archive:
        archive = args.archive.resolve()
        panel = load_archive(archive, CUTOFF)
        data_identity = {"archive_sha256": sha256(archive), "kind": "archive"}
    else:
        raise FileNotFoundError("Provide --panel or --archive when project/data/cache/panel.npz is absent")
    data_identity["audit_archive_sha256"] = panel.audit.get("archive_sha256")
    data_identity["dates"] = [str(panel.dates[0].date()), str(panel.dates[-1].date())]
    data_identity["pairs"] = len(panel.pairs)
    if panel.dates[-1] > CUTOFF:
        raise ValueError("Panel contains observations after the competition cutoff")
    config = {
        "selection_folds": list(FOLDS), "control_used": False, "seed": SEED,
        "pairs_per_origin": PAIRS_PER_ORIGIN, "origin_step_days": ORIGIN_STEP_DAYS,
        "common": common_parameters(args.threads),
        "families": {name: FAMILIES[name] for name in args.families},
        "tree_prefixes": list(PREFIXES), "thresholds": list(THRESHOLDS),
        "recent_weight_formula": "exp2((target_week - max(target_week))/52.1775)",
        "target_week_formula": "(target_year-2024)*52.1775 + ISO target_week",
        "source_sha256": {p.relative_to(source).as_posix(): sha256(p)
                          for p in sorted((source / "forecast").rglob("*.py"))},
        "script_sha256": sha256(Path(__file__)), "data_identity": data_identity,
        "source_snapshot_sha256": sha256(args.source_snapshot) if args.source_snapshot else None,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        **{name: importlib.metadata.version(name) for name in
                           ["numpy", "pandas", "scipy", "lightgbm", "scikit-learn"]}},
    }
    write_json(output / "manifest.json", config)
    fold_results = {}
    for fold in FOLDS:
        folder = output / fold
        folder.mkdir()
        origin = pd.Timestamp(fold) - pd.Timedelta(days=1)
        encoder = make_encoder(panel, origin)
        matrix, y = training_matrix(
            panel, origin, encoder,
            TrainingConfig(seed=SEED, iterations=400, pairs_per_origin=PAIRS_PER_ORIGIN,
                           origin_step_days=ORIGIN_STEP_DAYS, threads=args.threads),
        )
        columns = list(matrix.columns)
        base = base_features(panel, origin, encoder)
        tests = [horizon_features(base, panel, origin, h)[columns] for h in range(1, 10)]
        first_week = next_monday(origin)
        truth = np.stack([panel.total(first_week + pd.Timedelta(weeks=h),
                                      first_week + pd.Timedelta(weeks=h, days=6)) for h in range(9)])
        baseline = baselines(panel, origin)["last"]
        predictions = {"truth": truth, "last": baseline}
        metrics = {"last": compute_metrics(truth, baseline)}
        write_json(folder / "encoder.json", encoder)
        write_json(folder / "columns.json", {"columns": columns, "rows": len(y)})
        if args.cache_matrices:
            np.savez_compressed(folder / "matrix.npz", x=matrix.to_numpy(), y=y,
                                columns=np.asarray(columns, dtype=str))
        for name in args.families:
            spec = FAMILIES[name]
            params = {**common_parameters(args.threads),
                      **{k: v for k, v in spec.items() if k not in {"rounds", "log", "recent"}}}
            target = np.log1p(y) if spec.get("log") else y
            weight = None
            if spec.get("recent"):
                week = (matrix.target_year.to_numpy() - 2024) * 52.1775 + matrix.target_week.to_numpy()
                weight = np.exp2((week - week.max()) / 52.1775)
            dataset = lgb.Dataset(matrix, label=target, weight=weight,
                                  categorical_feature=CATEGORICAL, free_raw_data=False)
            started = time.perf_counter()
            model = lgb.train(params, dataset, num_boost_round=spec["rounds"])
            model.save_model(str(folder / f"{name}.txt"))
            for trees in sorted(set([spec["rounds"], *[n for n in PREFIXES if n <= spec["rounds"]]])):
                candidate = f"{name}_n{trees}"
                values = np.stack([model.predict(test, num_iteration=trees, num_threads=args.threads)
                                   for test in tests])
                if spec.get("log"):
                    values = np.expm1(np.clip(values, 0, 30))
                values = np.maximum(values, 0)
                predictions[candidate] = values
                for threshold in THRESHOLDS:
                    metrics[f"{candidate}_t{threshold}"] = compute_metrics(
                        truth, np.where(values < threshold, 0, values),
                    )
            write_json(folder / f"{name}_training.json", {
                "parameters": params, "trees": spec["rounds"], "rows": len(y),
                "seconds": time.perf_counter() - started, "model_sha256": sha256(folder / f"{name}.txt"),
            })
            write_json(folder / "metrics.json", metrics)
            np.savez_compressed(folder / "predictions.npz", **predictions)
            print(f"{fold}: {name} finished", flush=True)
            del model, dataset, target, weight
            gc.collect()
        leaderboard(metrics).to_csv(folder / "metrics.csv", index_label="candidate")
        fold_results[fold] = metrics
        del matrix, y, base, tests, predictions
        gc.collect()
    common = set.intersection(*(set(m) for m in fold_results.values()))
    mean_metrics = {
        name: {metric: float(np.mean([fold_results[f][name][metric] for f in FOLDS]))
               for metric in next(iter(fold_results.values()))[name]}
        for name in sorted(common)
    }
    write_json(output / "mean_selection_metrics.json", mean_metrics)
    print(f"Research complete: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--panel", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--source-snapshot", type=Path,
                        help="Optional code-only forecast/*.py ZIP for the historical feature pipeline")
    parser.add_argument("--families", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--cache-matrices", action="store_true")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.source_snapshot:
        with tempfile.TemporaryDirectory(prefix="sberhack-research-source-") as temp:
            source = extract_source(args.source_snapshot.resolve(), Path(temp))
            run(args, source)
    else:
        run(args, args.project.resolve())


if __name__ == "__main__":
    main()
