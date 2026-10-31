"""Repeat the final fit in an audit directory and compare with the promoted model."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.contracts import CUTOFF
from forecast.data import Panel, sha256_file
from forecast.model import Predictor, TrainingConfig, fit_models, save_bundle


def main():
    started = time.perf_counter()
    output = Path("reports/current_audit/refit")
    original = Predictor.load(Path("model"))
    config = TrainingConfig(**original.config["training"])
    panel = Panel.load(Path("data/cache/panel.npz"))
    assert panel.dates[-1] == CUTOFF
    models, encoder, columns, rows = fit_models(panel, CUTOFF, config, tuple(original.config["models"]))
    save_bundle(output, models, encoder, columns, original.config["selected"], config, panel, {})
    hashes = {}
    for kind in original.config["models"]:
        hashes[kind] = {"original": sha256_file(Path("model") / f"{kind}.txt"),
                        "refit": sha256_file(output / f"{kind}.txt")}
    expected = original.predict_panel(panel)
    actual = Predictor.load(output).predict_panel(panel)
    exact = expected.equals(actual)
    max_difference = float(np.max(np.abs(expected.quantity.to_numpy() - actual.quantity.to_numpy())))
    result = {"time_utc": datetime.now(timezone.utc).isoformat(), "status": "passed" if exact else "failed",
              "same_training_data": True, "rows": rows, "native_model_hashes": hashes,
              "native_model_bytes_equal": all(v["original"] == v["refit"] for v in hashes.values()),
              "prediction_exact_equal": exact, "max_abs_prediction_difference": max_difference,
              "elapsed_seconds": time.perf_counter() - started,
              "scope": "same Windows host and pinned installed versions; not a cross-platform guarantee"}
    (output.parent / "refit_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    pd.testing.assert_frame_equal(expected, actual, check_exact=True)


if __name__ == "__main__":
    main()
