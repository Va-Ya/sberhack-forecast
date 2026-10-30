"""Full real-data CLI/API parity smoke; no network is needed with --in-process."""
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from forecast.data import load_inputs, sha256_file
from forecast.model import Predictor


def write_inputs(archive: Path, directory: Path, cutoff: str = "2026-07-26", days: int = 180):
    directory.mkdir(parents=True, exist_ok=True)
    last = pd.Timestamp(cutoff)
    first = last - pd.Timedelta(days=days - 1)
    with zipfile.ZipFile(archive) as z:
        for kind in ["sales", "stock"]:
            name = next(n for n in z.namelist() if Path(n).name.startswith(kind + "_") and n.endswith(".csv"))
            destination = directory / f"{kind}.csv"
            header = True
            for chunk in pd.read_csv(z.open(name), chunksize=50_000):
                selected = chunk[(chunk.date >= str(first.date())) & (chunk.date <= str(last.date()))]
                if not selected.empty:
                    selected.to_csv(destination, index=False, mode="w" if header else "a", header=header)
                    header = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", default="hakaton.zip")
    parser.add_argument("--input-dir", default="data/smoke")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    args = parser.parse_args()
    directory = Path(args.input_dir)
    if not args.skip_prepare:
        write_inputs(Path(args.archive), directory)
    if args.prepare_only:
        return
    predictor = Predictor.load(Path("model"))
    with (directory / "sales.csv").open("rb") as sales, (directory / "stock.csv").open("rb") as stock:
        panel = load_inputs(sales, stock)
    cli = predictor.predict_panel(panel)
    cli_csv = cli.to_csv(index=False, float_format="%.8f")
    del panel
    started = time.perf_counter()
    with TestClient(create_app(predictor)) as client:
        assert client.get("/health").status_code == 200
        with (directory / "sales.csv").open("rb") as sales, (directory / "stock.csv").open("rb") as stock:
            response = client.post("/predict", files={"sales": ("sales.csv", sales), "stock": ("stock.csv", stock)})
        assert response.status_code == 200, response.text[:500]
        assert response.text == cli_csv, "CLI/API serialization differs"
        assert client.get("/predict/all").text == response.text
        records = client.get("/predict/all?format=json").json()
        assert len(records) == len(cli)
        article, warehouse = cli.article.iloc[0], cli.warehouse_name.iloc[0]
        filtered = client.get("/predict", params={"article": article, "warehouse": warehouse})
        assert len(pd.read_csv(io.StringIO(filtered.text))) == 9
    Path("reports").mkdir(exist_ok=True)
    (directory / "forecast.csv").write_text(cli_csv, encoding="utf-8", newline="")
    (Path("reports") / "api_smoke.json").write_text(json.dumps({"status": "passed", "rows": len(cli),
        "cli_api_equal": True, "input_days": 180, "elapsed_seconds": time.perf_counter() - started,
        "forecast_sha256": sha256_file(directory / "forecast.csv")}, indent=2), encoding="utf-8")
    print("Real-data API/CLI smoke passed", flush=True)


if __name__ == "__main__":
    main()
