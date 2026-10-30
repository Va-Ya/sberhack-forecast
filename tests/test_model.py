import numpy as np
import pandas as pd

from forecast.data import load_inputs
from forecast.features import base_features, make_encoder
from forecast.model import Predictor, TrainingConfig, fit_models, save_bundle, training_matrix
from tests.conftest import csv_bytes


def test_native_model_roundtrip_determinism(panel, tmp_path):
    cutoff = pd.Timestamp("2025-01-19")
    config = TrainingConfig(iterations=3, pairs_per_origin=2, origin_step_days=90, threads=1)
    models, encoder, columns, _ = fit_models(panel, cutoff, config)
    repeated_models, _, _, _ = fit_models(panel, cutoff, config)
    for kind in models:
        assert models[kind].model_to_string() == repeated_models[kind].model_to_string()
    save_bundle(tmp_path, models, encoder, columns, "log_zero", config, panel, {})
    predictor = Predictor.load(tmp_path)
    first = predictor.predict_panel(panel)
    second = Predictor.load(tmp_path).predict_panel(panel)
    pd.testing.assert_frame_equal(first, second, check_exact=True)
    assert len(first) == 18 and (first.quantity >= 0).all()


def test_requested_pairs_preserve_feature_context(csv_frames):
    sales, stock = csv_frames
    another = sales.copy()
    another["article"] = "b"
    another["quantity"] = 100
    sales = pd.concat([sales, another])
    all_panel = load_inputs(csv_bytes(sales), csv_bytes(stock))
    requested = pd.DataFrame({"warehouse_name": ["w"], "article": ["a"]})
    restricted = load_inputs(csv_bytes(sales), csv_bytes(stock), pairs=requested)
    np.testing.assert_array_equal(all_panel.sales, restricted.sales)
    cutoff = all_panel.dates[-1]
    encoder = make_encoder(all_panel, cutoff)
    pd.testing.assert_frame_equal(base_features(all_panel, cutoff, encoder),
                                  base_features(restricted, cutoff, encoder))
    result = Predictor({"selected": "mean8", "encoder": {}, "features": []}, {}).predict_panel(restricted)
    assert len(result) == 9 and set(result.article) == {"a"}


def test_training_labels_and_features_stop_at_origin(panel):
    cutoff = pd.Timestamp("2025-01-19")
    config = TrainingConfig(iterations=2, pairs_per_origin=2, origin_step_days=90, threads=1)
    encoder = make_encoder(panel, cutoff)
    before, y_before = training_matrix(panel, cutoff, encoder, config)
    panel.sales[panel.index(cutoff) + 1:] = 1e6
    panel.stock[panel.index(cutoff) + 1:] = -1e6
    after, y_after = training_matrix(panel, cutoff, encoder, config)
    pd.testing.assert_frame_equal(before, after)
    np.testing.assert_array_equal(y_before, y_after)
