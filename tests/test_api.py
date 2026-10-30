import io
import threading
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from forecast.data import load_inputs
from forecast.model import Predictor
from tests.conftest import csv_bytes


@pytest.fixture
def predictor():
    return Predictor({"selected": "mean8", "features": [], "encoder": {}}, {})


def upload(client, sales, stock):
    return client.post("/predict", files={"sales": ("sales.csv", csv_bytes(sales)),
                                           "stock": ("stock.csv", csv_bytes(stock))})


def test_api_parity_and_cache(csv_frames, predictor, tmp_path):
    sales, stock = csv_frames
    expected = predictor.predict_panel(load_inputs(csv_bytes(sales), csv_bytes(stock)))
    with TestClient(create_app(predictor, tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/predict/all").status_code == 409
        assert client.get("/predict").status_code == 400
        response = upload(client, sales, stock)
        assert response.status_code == 200
        actual = pd.read_csv(io.StringIO(response.text))
        pd.testing.assert_frame_equal(actual, expected)
        assert upload(client, sales, stock).text == response.text
        assert client.get("/predict/all").text == response.text
        assert len(client.get("/predict/all?format=json").json()) == 9
        assert client.get("/predict?article=a").status_code == 200
        assert client.get("/predict?warehouse=w").status_code == 200
        assert client.get("/predict?article=a&warehouse=w").status_code == 200
        assert client.get("/predict?article=unknown").status_code == 404
        assert client.get("/predict/all?format=xml").status_code == 400
        assert upload(client, sales.iloc[:100], stock).status_code == 422
        assert client.get("/predict/all").text == response.text
        assert client.post("/predict", files={"sales": ("s.csv", b""), "stock": ("s.csv", b"")}).status_code == 400
        missing = client.post("/predict")
        assert missing.status_code == 422 and missing.json()["code"] == "VALIDATION_ERROR"


def test_future_stock_ignored(csv_frames, predictor):
    sales, stock = csv_frames
    future = stock.iloc[[-1]].copy()
    future["date"] = "2026-12-31"
    future["stock"] = 999
    a = predictor.predict_panel(load_inputs(csv_bytes(sales), csv_bytes(stock)))
    b = predictor.predict_panel(load_inputs(csv_bytes(sales), csv_bytes(pd.concat([stock, future]))))
    pd.testing.assert_frame_equal(a, b)


def test_concurrent_post(csv_frames, predictor, tmp_path):
    entered, release = threading.Event(), threading.Event()
    real_predict = predictor.predict_panel

    def blocking(panel):
        entered.set()
        assert release.wait(10)
        return real_predict(panel)

    predictor.predict_panel = blocking
    with TestClient(create_app(predictor, tmp_path)) as client, ThreadPoolExecutor(2) as pool:
        first = pool.submit(upload, client, *csv_frames)
        assert entered.wait(10)
        try:
            second = upload(client, *csv_frames)
            assert second.status_code == 409
            assert second.json()["code"] == "PREDICTION_IN_PROGRESS"
        finally:
            release.set()
        assert first.result().status_code == 200


def test_model_failure_keeps_previous_cache(csv_frames, predictor, tmp_path):
    with TestClient(create_app(predictor, tmp_path)) as client:
        previous = upload(client, *csv_frames)
        assert previous.status_code == 200

        def broken(_):
            raise RuntimeError("synthetic native-model failure")

        predictor.predict_panel = broken
        failed = upload(client, *csv_frames)
        assert failed.status_code == 500
        assert failed.json()["code"] == "PREDICTION_ERROR"
        assert client.get("/predict/all").text == previous.text
