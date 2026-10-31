"""Reproduce a fixed, history-only ensemble search on two selection windows.

Run from repository root after its temporal benchmark has saved raw/log/mean8
predictions: python benchmark_calibration.py --root . --output results.json
The previously inspected control window is deliberately never opened.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def run(root: Path, output: Path) -> dict:
    import sys

    sys.path.insert(0, str(root))
    from forecast.metrics import WEIGHTS, compute_metrics

    folds = []
    for fold in ["2026-01-19", "2026-03-23"]:
        with np.load(root / "reports/predictions" / f"{fold}.npz") as archive:
            values = {key: archive[key] for key in ["truth", "log", "raw", "mean8", "last"]}
        reference = compute_metrics(values["truth"], np.where(values["log"] < 0.75, 0, values["log"]))
        naive = compute_metrics(values["truth"], values["last"])
        folds.append((fold, values, reference, naive))

    results = []
    for raw, mean, threshold, scale in itertools.product(
        [0, 0.15, 0.3], [0.15, 0.3, 0.45], [0.7, 0.75, 0.8], [1, 1.025]
    ):
        metrics = {}
        for fold, values, _, naive in folds:
            prediction = (1 - raw - mean) * values["log"] + raw * values["raw"] + mean * values["mean8"]
            prediction *= np.where(values["mean8"] > 2, scale, 1)
            prediction = np.where(prediction < threshold, 0, prediction)
            calculated = compute_metrics(values["truth"], prediction)
            calculated["score"] = sum(
                weight * min(calculated[key] / naive[key], 1) if naive[key]
                else weight * float(calculated[key] != 0)
                for key, weight in WEIGHTS.items()
            )
            metrics[fold] = calculated
        results.append({
            "params": {"raw": raw, "mean8": mean, "threshold": threshold, "scale": scale},
            "metrics": metrics,
            "mean_score": float(np.mean([value["score"] for value in metrics.values()])),
            "improves_every_metric_on_each_fold": all(
                metrics[fold][key] < reference[key] for fold, _, reference, _ in folds for key in WEIGHTS
            ),
        })
    results.sort(key=lambda item: item["mean_score"])
    eligible = [item for item in results if item["improves_every_metric_on_each_fold"]]
    result = {
        "selection_folds": [fold for fold, *_ in folds],
        "control_evaluated": False,
        "reference": {fold: reference for fold, _, reference, _ in folds},
        "candidate_count": len(results),
        "selected": eligible[0] if eligible else None,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("calibration_results.json"))
    arguments = parser.parse_args()
    result = run(arguments.root.resolve(), arguments.output)
    print(json.dumps({"candidate_count": result["candidate_count"], "selected": result["selected"]}, indent=2))
