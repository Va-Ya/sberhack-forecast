import numpy as np
import pandas as pd
import pytest

from forecast.contracts import InputError, next_monday, validate_forecast
from forecast.data import Panel, load_inputs
from forecast.features import base_features, baselines, horizon_features, make_encoder
from forecast.metrics import compute_metrics, leaderboard
from tests.conftest import csv_bytes


def test_metrics_zero_and_bias():
    assert all(v == 0 for v in compute_metrics(np.zeros((9, 2)), np.zeros((9, 2))).values())
    m = compute_metrics(np.zeros((9, 2)), np.ones((9, 2)))
    assert m["smape"] == 200 and m["wape"] == 100 and m["bias"] == 0
    m = compute_metrics(np.array([[0, 2], [0, 2]]), np.array([[100, 4], [100, 4]]))
    assert m["bias"] == 100


def test_ranking_ties():
    values = {"last": dict.fromkeys(["smape", "wape", "rmse", "bias", "rmsle"], 1.0)}
    values["same"] = values["last"].copy()
    assert (leaderboard(values).rank_score == 0.5).all()


def test_daily_duplicates_and_alias(csv_frames):
    sales, stock = csv_frames
    stock = pd.concat([stock, stock]).rename(columns={"article": "char_article"})
    p = load_inputs(csv_bytes(sales), csv_bytes(stock), chunksize=70)
    assert p.stock[0, 0] == -2
    assert p.total(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-11"))[0] == 7


@pytest.mark.parametrize("kind", ["empty", "short", "negative", "inf", "date", "alias", "missing"])
def test_input_errors(csv_frames, kind):
    sales, stock = csv_frames
    if kind == "empty":
        sales = sales.iloc[:0]
    elif kind == "short":
        sales = sales.iloc[:100]
    elif kind == "negative":
        sales.loc[0, "quantity"] = -1
    elif kind == "inf":
        stock["stock"] = stock.stock.astype(float)
        stock.loc[0, "stock"] = np.inf
    elif kind == "date":
        sales.loc[0, "date"] = "2026-1-1"
    elif kind == "alias":
        stock["char_article"] = "different"
    elif kind == "missing":
        sales = sales.drop(columns="cities_name")
    with pytest.raises(InputError):
        load_inputs(csv_bytes(sales), csv_bytes(stock))


def test_logarithmic_money_zero_is_valid(csv_frames):
    sales, stock = csv_frames
    sales.loc[0, "sales_sum"] = -np.inf
    sales.loc[1, "discount_sum"] = -np.inf
    sales.loc[2, "sales_sum"] = np.nan
    p = load_inputs(csv_bytes(sales), csv_bytes(stock))
    assert p.sales.sum() == len(sales)


def test_cutoff_and_no_future_leakage(panel):
    cutoff = pd.Timestamp("2026-07-26")
    enc = make_encoder(panel, cutoff)
    before = horizon_features(base_features(panel, cutoff, enc), panel, cutoff, 1)
    panel.sales[panel.index(cutoff) + 1:] = 10000
    panel.stock[panel.index(cutoff) + 1:] = -999
    panel.metadata.loc[len(panel.metadata)] = [pd.Timestamp("2026-07-27"), "a", "future", "future", "future"]
    after = horizon_features(base_features(panel, cutoff, enc), panel, cutoff, 1)
    pd.testing.assert_frame_equal(before, after)


def test_partial_week(panel):
    cutoff = pd.Timestamp("2026-07-31")
    frame = base_features(panel, cutoff, make_encoder(panel, cutoff))
    assert frame.partial_days.iloc[0] == 5
    assert frame.partial_quantity.iloc[0] == 5
    assert frame.lag_1.iloc[0] == 7
    assert next_monday(cutoff) == pd.Timestamp("2026-08-03")
    assert baselines(panel, cutoff)["mean8"][0, 0] == 7


def test_panel_cache(panel, tmp_path):
    path = tmp_path / "panel.npz"
    panel.save(path)
    other = Panel.load(path)
    np.testing.assert_array_equal(panel.sales, other.sales)
    pd.testing.assert_frame_equal(panel.pairs, other.pairs)


def test_forecast_validation():
    pairs = pd.DataFrame({"warehouse_name": ["w"], "article": ["a"]})
    forecast = pd.DataFrame({"week": pd.date_range("2026-07-27", periods=9, freq="7D").strftime("%Y-%m-%d"),
                             "warehouse_name": "w", "article": "a", "quantity": 0.0})
    validate_forecast(forecast, pairs, pd.Timestamp("2026-07-27"))
    with pytest.raises(InputError):
        validate_forecast(forecast.iloc[:-1], pairs, pd.Timestamp("2026-07-27"))
    forecast.loc[0, "quantity"] = np.inf
    with pytest.raises(InputError):
        validate_forecast(forecast, pairs, pd.Timestamp("2026-07-27"))


@pytest.mark.parametrize("value", [-1, np.inf, np.nan])
def test_metric_input_rejected(value):
    y = np.ones((9, 1))
    p = y.copy()
    p[0, 0] = value
    with pytest.raises(ValueError):
        compute_metrics(y, p)
