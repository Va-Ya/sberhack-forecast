from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd

from .contracts import CATEGORIES, SALES_COLUMNS, InputError


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    pairs: pd.DataFrame
    sales: np.ndarray
    stock: np.ndarray
    metadata: pd.DataFrame
    audit: dict

    @property
    def output_indices(self) -> np.ndarray:
        requested = self.audit.get("requested_pairs")
        if requested is None:
            return np.arange(len(self.pairs))
        return pd.MultiIndex.from_frame(self.pairs).get_indexer(pd.MultiIndex.from_tuples(
            [tuple(pair) for pair in requested], names=["warehouse_name", "article"]))

    @property
    def output_pairs(self) -> pd.DataFrame:
        return self.pairs.iloc[self.output_indices].reset_index(drop=True)

    def index(self, date: pd.Timestamp) -> int:
        return int((pd.Timestamp(date) - self.dates[0]).days)

    def total(self, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
        lo, hi = max(0, self.index(start)), min(len(self.dates), self.index(end) + 1)
        return self.sales[lo:hi].sum(axis=0) if hi > lo else np.zeros(len(self.pairs), dtype=np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dates=self.dates.to_numpy(), sales=self.sales, stock=self.stock,
                            pairs=self.pairs.to_numpy(dtype=str), metadata=self.metadata.to_json(orient="split",
                            date_format="iso"), audit=json.dumps(self.audit))

    @classmethod
    def load(cls, path: Path) -> Panel:
        with np.load(path, allow_pickle=False) as a:
            metadata = pd.read_json(io.StringIO(str(a["metadata"])), orient="split")
            metadata["date"] = pd.to_datetime(metadata["date"])
            return cls(pd.DatetimeIndex(a["dates"]), pd.DataFrame(a["pairs"],
                       columns=["warehouse_name", "article"]), a["sales"], a["stock"],
                       metadata, json.loads(str(a["audit"])))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(stream: BinaryIO, kind: str, cutoff: pd.Timestamp | None, min_days: int,
          chunksize: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    groups, metadata = [], []
    rows = excluded = negatives = 0
    minimum = maximum = None
    required = SALES_COLUMNS if kind == "sales" else ["date", "warehouse_name", "stock"]
    try:
        reader = pd.read_csv(stream, chunksize=chunksize, dtype={"date": str, "article": str,
                            "char_article": str, "warehouse_name": str, **dict.fromkeys(CATEGORIES, str)})
        for chunk in reader:
            missing = set(required) - set(chunk.columns)
            if kind == "stock" and not {"article", "char_article"}.intersection(chunk.columns):
                missing.add("article or char_article")
            if missing:
                raise InputError(f"Missing {kind} columns: {sorted(missing)}")
            if chunk.empty:
                continue
            rows += len(chunk)
            if not chunk.date.str.fullmatch(r"\d{4}-\d{2}-\d{2}", na=False).all():
                raise InputError(f"{kind}: dates must be YYYY-MM-DD")
            chunk["date"] = pd.to_datetime(chunk.date, format="%Y-%m-%d", errors="raise")
            if cutoff is not None:
                excluded += int((chunk.date > cutoff).sum())
                chunk = chunk[chunk.date <= cutoff].copy()
            if chunk.empty:
                continue
            if kind == "stock":
                if "char_article" in chunk:
                    if "article" in chunk and not chunk.article.equals(chunk.char_article):
                        raise InputError("article and char_article disagree")
                    chunk["article"] = chunk.char_article
            if chunk[["warehouse_name", "article"]].isna().any().any():
                raise InputError(f"{kind}: missing warehouse or article")
            minimum = min(minimum or chunk.date.min(), chunk.date.min())
            maximum = max(maximum or chunk.date.max(), chunk.date.max())
            numeric = ["quantity", "sales_sum", "discount_sum"] if kind == "sales" else ["stock"]
            for col in numeric:
                if col == "discount_sum":
                    chunk[col] = chunk[col].fillna(0)
                chunk[col] = pd.to_numeric(chunk[col], errors="raise")
                values = chunk[col].to_numpy(dtype=float)
                # Money is supplied as np.log(amount): log(0) is legitimately -inf.
                valid = np.isfinite(values)
                if col in {"sales_sum", "discount_sum"}:
                    valid |= np.isneginf(values)
                if col == "sales_sum":
                    # The supplied archive has three missing monetary values. They are
                    # unknown, never zero sales; money is excluded from model features.
                    valid |= np.isnan(values)
                if not valid.all():
                    raise InputError(f"{kind}: {col} must contain finite numbers")
            value = "quantity" if kind == "sales" else "stock"
            negatives += int((chunk[value] < 0).sum())
            if kind == "sales" and negatives:
                raise InputError("sales: negative quantity")
            keys = ["date", "warehouse_name", "article"]
            groups.append(chunk.groupby(keys, as_index=False, sort=False)[value].sum())
            if kind == "sales":
                meta = chunk[["date", "article"] + CATEGORIES].copy()
                meta[CATEGORIES] = meta[CATEGORIES].fillna("__missing__").astype(str)
                metadata.append(meta.sort_values("date").drop_duplicates(["article"] + CATEGORIES))
    except InputError:
        raise
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeError) as exc:
        raise InputError(f"Cannot parse {kind} CSV", 400, "PARSE_ERROR") from exc
    except (ValueError, TypeError, OverflowError) as exc:
        raise InputError(f"Invalid {kind} values: {exc}") from exc
    if not rows:
        raise InputError(f"Empty {kind} file", 400, "EMPTY_FILE")
    if minimum is None:
        raise InputError(f"No {kind} observations on or before cutoff")
    if (maximum - minimum).days + 1 < min_days:
        raise InputError(f"{kind}: at least {min_days} days of history required")
    result = pd.concat(groups, ignore_index=True).groupby(
        ["date", "warehouse_name", "article"], as_index=False, sort=False)[value].sum()
    if not np.isfinite(result[value]).all():
        raise InputError(f"{kind}: aggregated {value} is not finite")
    meta_result = pd.concat(metadata, ignore_index=True) if metadata else pd.DataFrame()
    if not meta_result.empty:
        meta_result = meta_result.sort_values("date").drop_duplicates(["article"] + CATEGORIES)
    return result, meta_result, dict(rows=rows, excluded_after_cutoff=excluded,
                                     start=str(minimum.date()), end=str(maximum.date()), negative_values=negatives)


def load_inputs(sales: BinaryIO, stock: BinaryIO, cutoff: pd.Timestamp | None = None,
                min_days: int = 180, chunksize: int = 200_000,
                pairs: pd.DataFrame | None = None) -> Panel:
    cutoff = pd.Timestamp(cutoff) if cutoff is not None else None
    daily_sales, metadata, sales_audit = _read(sales, "sales", cutoff, min_days, chunksize)
    # The prediction origin is determined only by sales, never by later stock.
    last = cutoff if cutoff is not None else daily_sales.date.max()
    daily_stock, _, stock_audit = _read(stock, "stock", last, min_days, chunksize)
    pair_policy = "provided" if pairs is not None else "cartesian"
    requested = None
    if pairs is not None:
        if not {"warehouse_name", "article"}.issubset(pairs.columns):
            raise InputError("Pairs CSV requires warehouse_name and article")
        if pairs[["warehouse_name", "article"]].isna().any().any() or pairs.empty:
            raise InputError("Pair universe is empty or has missing identifiers")
        pairs = pairs[["warehouse_name", "article"]].astype(str).drop_duplicates().sort_values(
            ["warehouse_name", "article"]).reset_index(drop=True)
        requested = pairs.to_numpy().tolist()
    warehouses = set(daily_sales.warehouse_name) | set(daily_stock.warehouse_name)
    articles = set(daily_sales.article) | set(daily_stock.article)
    if pairs is not None:
        warehouses.update(pairs.warehouse_name)
        articles.update(pairs.article)
    # Retain all observed series for cross-series features; restrict only the output.
    pairs = pd.MultiIndex.from_product([sorted(warehouses), sorted(articles)],
            names=["warehouse_name", "article"]).to_frame(index=False)
    dates = pd.date_range(daily_sales.date.min().normalize(), last, freq="D")
    pair_index = pd.MultiIndex.from_frame(pairs)
    shape = (len(dates), len(pairs))
    y = np.zeros(shape, dtype=np.float32)
    s = np.full(shape, np.nan, dtype=np.float32)
    for frame, target, value in [(daily_sales, y, "quantity"), (daily_stock, s, "stock")]:
        p = pair_index.get_indexer(pd.MultiIndex.from_frame(frame[["warehouse_name", "article"]]))
        d = (frame.date - dates[0]).dt.days.to_numpy()
        valid = (p >= 0) & (d >= 0) & (d < len(dates))
        target[d[valid], p[valid]] = frame.loc[valid, value].to_numpy(dtype=np.float32)
    return Panel(dates, pairs, y, s, metadata, {"sales": sales_audit, "stock": stock_audit,
                 "pairs": len(pairs), "origin": str(last.date()), "pair_policy": pair_policy,
                 "requested_pairs": requested})


def load_archive(path: Path, cutoff: pd.Timestamp | None = None,
                 pairs: pd.DataFrame | None = None) -> Panel:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        sales_name = next(n for n in names if Path(n).name.startswith("sales_") and n.endswith(".csv"))
        stock_name = next(n for n in names if Path(n).name.startswith("stock_") and n.endswith(".csv"))
        with archive.open(sales_name) as sales, archive.open(stock_name) as stock:
            panel = load_inputs(sales, stock, cutoff, pairs=pairs)
    panel.audit["archive_sha256"] = sha256_file(path)
    return panel
