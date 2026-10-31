"""Exercise the real HTTP service and preserve a dated, machine-readable result."""
from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd


def main():
    destination = Path("reports/current_audit")
    destination.mkdir(parents=True, exist_ok=True)
    checks = []
    sales = pd.read_csv("data/demo_sales.csv")
    stock = pd.read_csv("data/demo_stock.csv")
    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=3600, trust_env=False)

    def request(name, method, url, expected, **kwargs):
        started = time.perf_counter()
        response = client.request(method, url, **kwargs)
        record = {"check": name, "status": response.status_code, "expected": expected,
                  "seconds": time.perf_counter() - started, "passed": response.status_code == expected}
        if expected >= 400:
            body = response.json()
            record["error_envelope_valid"] = isinstance(body.get("detail"), str) and isinstance(body.get("code"), str)
            record["passed"] &= record["error_envelope_valid"]
        checks.append(record)
        assert record["passed"], record
        return response

    def upload(name, left, right, expected=200):
        return request(name, "POST", "/predict", expected,
                       files={"sales": ("sales.csv", left.to_csv(index=False).encode()),
                              "stock": ("stock.csv", right.to_csv(index=False).encode())})

    request("health", "GET", "/health", 200)
    request("openapi", "GET", "/openapi.json", 200)
    request("missing_get_filter", "GET", "/predict", 400)
    original = upload("valid_181_day_demo", sales, stock)
    demo = pd.read_csv(io.StringIO(original.text))
    assert len(demo) == 9 and demo.week.iloc[0] == "2026-07-27"
    invalid = sales.copy()
    invalid.loc[0, "quantity"] = -1
    upload("negative_sales", invalid, stock, 422)
    upload("short_history", sales.iloc[:100], stock, 422)
    invalid = sales.copy()
    invalid.loc[0, "date"] = "2026-2-31"
    upload("bad_date", invalid, stock, 422)
    upload("missing_column", sales.drop(columns="article"), stock, 422)
    request("empty_files", "POST", "/predict", 400,
            files={"sales": ("s.csv", b""), "stock": ("t.csv", b"")})
    request("unreadable_csv", "POST", "/predict", 400,
            files={"sales": ("s.csv", b'bad,header\n"unterminated'),
                   "stock": ("t.csv", stock.to_csv(index=False).encode())})
    request("missing_multipart_fields", "POST", "/predict", 422)
    assert request("failed_posts_keep_cache", "GET", "/predict/all", 200).content == original.content
    negative_stock = stock.copy()
    negative_stock.loc[0, "stock"] = -3
    upload("negative_stock_allowed", sales, negative_stock)
    both_aliases = stock.copy()
    both_aliases["article"] = both_aliases.char_article
    upload("matching_stock_aliases", sales, both_aliases)
    both_aliases.loc[0, "article"] = "conflicting_article"
    upload("conflicting_stock_aliases", sales, both_aliases, 422)
    shifted_sales, shifted_stock = sales.copy(), stock.copy()
    shifted_sales["date"] = (pd.to_datetime(sales.date) + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%d")
    shifted_stock["date"] = (pd.to_datetime(stock.date) + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%d")
    friday = upload("friday_origin", shifted_sales, shifted_stock)
    assert pd.read_csv(io.StringIO(friday.text)).week.iloc[0] == "2026-08-03"
    unknown_sales, unknown_stock = sales.copy(), stock.copy()
    unknown_sales["article"], unknown_sales["warehouse_name"] = "unseen_article", "unseen_warehouse"
    unknown_stock["char_article"], unknown_stock["warehouse_name"] = "unseen_article", "unseen_warehouse"
    unknown = upload("unseen_entities", unknown_sales, unknown_stock)
    unknown_frame = pd.read_csv(io.StringIO(unknown.text))
    assert len(unknown_frame) == 9 and np.isfinite(unknown_frame.quantity).all()
    extra = stock.iloc[[-1]].copy()
    extra["date"], extra["char_article"], extra["warehouse_name"] = "2026-12-31", "future_article", "future_warehouse"
    future = upload("future_stock_ignored", sales, pd.concat([stock, extra]))
    assert future.content == original.content
    request("unknown_filter", "GET", "/predict?article=__not_in_cache__", 404)
    request("invalid_format", "GET", "/predict/all?format=xml", 400)
    real = []
    for repetition in range(2):
        with open("data/smoke/sales.csv", "rb") as sale_file, open("data/smoke/stock.csv", "rb") as stock_file:
            real.append(request(f"real_180_day_post_{repetition + 1}", "POST", "/predict", 200,
                                files={"sales": ("sales.csv", sale_file), "stock": ("stock.csv", stock_file)}))
    assert real[0].content == real[1].content == Path("data/smoke/forecast.csv").read_bytes()
    frame = pd.read_csv(io.BytesIO(real[-1].content))
    assert len(frame) == 79101 and np.isfinite(frame.quantity).all() and (frame.quantity >= 0).all()
    assert not frame.duplicated(["week", "warehouse_name", "article"]).any()
    assert request("full_cached_csv", "GET", "/predict/all?format=csv", 200).content == real[-1].content
    json_frame = pd.DataFrame(request("full_json", "GET", "/predict/all?format=json", 200).json())
    pd.testing.assert_frame_equal(frame.drop(columns="quantity"), json_frame.drop(columns="quantity"))
    np.testing.assert_allclose(frame.quantity, json_frame.quantity, rtol=0, atol=5.1e-9)
    article, warehouse = frame.article.iloc[0], frame.warehouse_name.iloc[0]
    for name, query, mask in [
        ("article_filter", f"article={article}", frame.article == article),
        ("warehouse_filter", f"warehouse={warehouse}", frame.warehouse_name == warehouse),
        ("pair_filter", f"article={article}&warehouse={warehouse}",
         (frame.article == article) & (frame.warehouse_name == warehouse)),
    ]:
        filtered = pd.read_csv(io.BytesIO(request(name, "GET", "/predict?" + query, 200).content))
        pd.testing.assert_frame_equal(filtered, frame.loc[mask].reset_index(drop=True))
    result = {"status": "passed", "time_utc": datetime.now(timezone.utc).isoformat(),
              "url": str(client.base_url), "checks": checks, "real_rows": len(frame),
              "repeated_post_exact_equal": True, "existing_cli_csv_exact_equal": True,
              "csv_json_numeric_equal": True, "container_test": False}
    (destination / "http_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    client.close()
    print(json.dumps({"status": "passed", "checks": len(checks), "real_rows": len(frame),
                      "real_post_seconds": [c["seconds"] for c in checks if c["check"].startswith("real_180")]}), flush=True)


if __name__ == "__main__":
    main()
