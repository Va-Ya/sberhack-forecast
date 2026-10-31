import sys

import pytest

from forecast import cli
from forecast.model import Predictor


@pytest.mark.parametrize("invalid_target", ["missing", "directory"])
def test_explicit_pairs_path_must_be_a_file(csv_frames, tmp_path, monkeypatch, invalid_target):
    sales, stock = csv_frames
    sales_path, stock_path = tmp_path / "sales.csv", tmp_path / "stock.csv"
    sales.to_csv(sales_path, index=False)
    stock.to_csv(stock_path, index=False)
    pairs_path = tmp_path / "organizer_pairs.csv"
    if invalid_target == "directory":
        pairs_path.mkdir()
    fallback = Predictor({"selected": "mean8", "features": [], "encoder": {}}, {})
    monkeypatch.setattr(cli.Predictor, "load", lambda _: fallback)
    monkeypatch.setattr(sys, "argv", ["forecast", "predict", "--sales", str(sales_path),
                                     "--stock", str(stock_path), "--pairs", str(pairs_path),
                                     "--output", str(tmp_path / "forecast.csv")])
    with pytest.raises(SystemExit) as failure:
        cli.main()
    assert failure.value.code == 2
    assert not (tmp_path / "forecast.csv").exists()
