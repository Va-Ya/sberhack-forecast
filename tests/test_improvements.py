import numpy as np
import pytest

from forecast.data import load_inputs
from forecast.metrics import leaderboard
from forecast.model import TrainingConfig, candidates
from tests.conftest import csv_bytes


def test_baseline_score_does_not_depend_on_other_candidates():
    names = ["smape", "wape", "rmse", "bias", "rmsle"]
    metrics = {"last": dict.fromkeys(names, 100.0), "model": dict.fromkeys(names, 70.0)}
    before = leaderboard(metrics).loc["model", "baseline_score"]
    metrics["unrelated"] = dict.fromkeys(names, 50.0)
    assert leaderboard(metrics).loc["model", "baseline_score"] == before


def test_mixed_model_is_available_after_bundle_reload():
    baseline = {"mean8": np.ones((9, 2))}
    learned = {"raw": np.ones((9, 2)) * 2, "log": np.ones((9, 2))}
    mixed = candidates(baseline, learned)["mix_0.25"]
    np.testing.assert_array_equal(mixed, np.full((9, 2), 1.25))


@pytest.mark.parametrize("kwargs", [{"iterations": 0}, {"pairs_per_origin": -1}, {"learning_rate": 0}])
def test_invalid_training_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_category_identifiers_preserve_leading_zeros(csv_frames):
    sales, stock = csv_frames
    sales["category_lvl_3"] = "001"
    panel = load_inputs(csv_bytes(sales), csv_bytes(stock), chunksize=70)
    assert set(panel.metadata.category_lvl_3) == {"001"}
