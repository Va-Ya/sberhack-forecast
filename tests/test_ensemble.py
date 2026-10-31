import numpy as np
import pandas as pd
import pytest

from forecast.model import Predictor, calibrated_predictions


def test_frozen_ensemble_threshold_and_convex_weights():
    baseline = {"mean8": np.array([[0., 2.]])}
    learned = {"log": np.array([[0.5, 4.]]), "raw": np.array([[1., 6.]])}
    values = calibrated_predictions(baseline, learned,
                                    {"weights": {"log": .55, "raw": .15, "mean8": .3}, "threshold": .75})
    np.testing.assert_allclose(values, [[0., 3.7]])


@pytest.mark.parametrize("recipe", [
    {"weights": {}}, {"weights": {"log": -1}}, {"weights": {"log": float("nan")}},
    {"weights": {"log": .5}}, {"weights": {"log": 1}, "threshold": -1},
])
def test_invalid_frozen_recipe_rejected(recipe):
    with pytest.raises(ValueError):
        calibrated_predictions({}, {"log": np.ones((9, 2))}, recipe)


def test_baseline_only_recipe_native_reload(panel, tmp_path):
    import json

    config = {"models": [], "selected": "calibrated", "features": [], "encoder": {},
              "calibration": {"weights": {"mean8": 1.0}, "threshold": .75}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    first = Predictor.load(tmp_path).predict_panel(panel)
    second = Predictor.load(tmp_path).predict_panel(panel)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert (first.quantity >= 0).all() and len(first) == 18
