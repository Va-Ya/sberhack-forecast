"""Create first-100-row schema samples and a valid 181-day synthetic API example."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def main():
    directory = Path("data")
    directory.mkdir(exist_ok=True)
    with ZipFile("hakaton.zip") as archive:
        for kind in ["sales", "stock"]:
            name = next(n for n in archive.namelist() if Path(n).name.startswith(kind + "_") and n.endswith(".csv"))
            sample = pd.read_csv(archive.open(name), nrows=100)
            sample.to_csv(directory / f"sample_{kind}.csv", index=False)
    dates = pd.date_range("2026-01-27", "2026-07-26")
    template = pd.read_csv(directory / "sample_sales.csv").iloc[0]
    sales = pd.DataFrame([template.to_dict()] * len(dates))
    sales["date"] = dates.strftime("%Y-%m-%d")
    sales["quantity"] = np.where(dates.weekday < 5, 1.0, 0.0)
    sales["sales_sum"] = 0.0
    sales["discount_sum"] = np.nan
    stock = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "warehouse_name": template.warehouse_name,
                          "char_article": template.article, "stock": 10.0})
    sales.to_csv(directory / "demo_sales.csv", index=False)
    stock.to_csv(directory / "demo_stock.csv", index=False)


if __name__ == "__main__":
    main()
