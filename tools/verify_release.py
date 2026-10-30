"""Verify final CSV keys and parity with a freshly loaded native model on the original panel."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecast.contracts import CUTOFF, validate_forecast
from forecast.data import Panel, sha256_file
from forecast.model import Predictor


def main():
    panel = Panel.load(Path("data/cache/panel.npz"))
    predictor = Predictor.load(Path("model"))
    assert panel.dates[-1] == CUTOFF
    assert predictor.config["cutoff"] == "2026-07-26"
    stored = pd.read_csv("outputs/submission.csv", dtype={"week": str, "warehouse_name": str, "article": str})
    validate_forecast(stored, panel.output_pairs, pd.Timestamp("2026-07-27"))
    first = predictor.predict_panel(panel)
    second = Predictor.load(Path("model")).predict_panel(panel)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    pd.testing.assert_frame_equal(stored.drop(columns="quantity"), first.drop(columns="quantity"))
    np.testing.assert_allclose(stored.quantity, first.quantity, rtol=0, atol=5.1e-9)
    result = {"status": "passed", "rows": len(stored), "weeks": sorted(stored.week.unique().tolist()),
              "no_duplicates": True, "finite_nonnegative": True, "native_reload_equal": True,
              "fresh_prediction_equal_to_csv": True, "submission_sha256": sha256_file(Path("outputs/submission.csv"))}
    Path("reports/release_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
