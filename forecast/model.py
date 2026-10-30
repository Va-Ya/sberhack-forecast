from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .contracts import HORIZON, OUTPUT_COLUMNS, next_monday, validate_forecast
from .data import Panel
from .features import CATEGORICAL, base_features, baselines, horizon_features, make_encoder


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2026
    iterations: int = 180
    pairs_per_origin: int = 600
    origin_step_days: int = 14
    threads: int = 2
    learning_rate: float = 0.055
    num_leaves: int = 31


def training_matrix(panel: Panel, cutoff: pd.Timestamp, encoder: dict, config: TrainingConfig):
    first = panel.dates[0] + pd.Timedelta(days=182)
    first += pd.Timedelta(days=(6 - first.weekday()) % 7)
    origins = list(pd.date_range(first, cutoff - pd.Timedelta(days=7), freq=f"{config.origin_step_days}D"))
    # Add Fridays to teach the same model how partial weeks affect the next Monday.
    origins += [d - pd.Timedelta(days=2) for d in origins[::2]]
    origins = sorted(set(origins))
    rng = np.random.default_rng(config.seed)
    matrices, targets = [], []
    for origin in origins:
        ids = np.sort(rng.choice(len(panel.pairs), size=min(config.pairs_per_origin, len(panel.pairs)), replace=False))
        base = base_features(panel, origin, encoder)
        for horizon in range(1, HORIZON + 1):
            week = next_monday(origin) + pd.Timedelta(weeks=horizon - 1)
            if week + pd.Timedelta(days=6) > cutoff:
                continue
            matrices.append(horizon_features(base, panel, origin, horizon).iloc[ids])
            targets.append(panel.total(week, week + pd.Timedelta(days=6))[ids])
    if not matrices:
        raise ValueError("Training requires history and at least one complete future target week")
    return pd.concat(matrices, ignore_index=True), np.concatenate(targets).astype(np.float32)


def fit_models(panel: Panel, cutoff: pd.Timestamp, config: TrainingConfig,
               kinds: tuple[str, ...] = ("raw", "log")) -> tuple[dict, dict, list[str], int]:
    encoder = make_encoder(panel, cutoff)
    matrix, y = training_matrix(panel, cutoff, encoder, config)
    print(f"Training at {cutoff.date()}: {len(y):,} examples, {matrix.shape[1]} features", flush=True)
    models = {}
    for kind in kinds:
        model = lgb.LGBMRegressor(
            objective="regression_l1" if kind == "raw" else "regression",
            n_estimators=config.iterations, learning_rate=config.learning_rate,
            num_leaves=config.num_leaves, max_bin=127, min_child_samples=100,
            reg_lambda=3.0, n_jobs=config.threads, random_state=config.seed,
            deterministic=True, force_col_wise=True, verbosity=-1,
        )
        model.fit(matrix, np.log1p(y) if kind == "log" else y,
                  categorical_feature=CATEGORICAL)
        models[kind] = model.booster_
        print(f"Finished {kind} model at {cutoff.date()}", flush=True)
    return models, encoder, list(matrix.columns), len(y)


def model_predictions(models: dict, encoder: dict, columns: list[str], panel: Panel,
                      cutoff: pd.Timestamp, threads: int = 2) -> dict[str, np.ndarray]:
    base = base_features(panel, cutoff, encoder)
    result = {kind: [] for kind in models}
    for horizon in range(1, HORIZON + 1):
        matrix = horizon_features(base, panel, cutoff, horizon)[columns]
        for kind, model in models.items():
            values = model.predict(matrix, num_threads=threads)
            if kind == "log":
                values = np.expm1(np.clip(values, 0, 30))
            result[kind].append(np.maximum(values, 0))
    return {kind: np.stack(value) for kind, value in result.items()}


def candidates(baseline: dict, learned: dict) -> dict:
    result = dict(baseline)
    for kind, prediction in learned.items():
        result[kind] = prediction
        result[kind + "_zero"] = np.where(prediction < 0.5, 0, prediction)
        result[kind + "_blend"] = 0.8 * prediction + 0.2 * baseline["mean8"]
    return result


def forecast_frame(values: np.ndarray, pairs: pd.DataFrame, first_week: pd.Timestamp) -> pd.DataFrame:
    frames = []
    for i, week in enumerate(pd.date_range(first_week, periods=HORIZON, freq="7D")):
        frame = pairs.copy()
        frame["week"] = week.strftime("%Y-%m-%d")
        frame["quantity"] = np.asarray(values[i], dtype=float)
        frames.append(frame[OUTPUT_COLUMNS])
    result = pd.concat(frames, ignore_index=True)
    validate_forecast(result, pairs, first_week)
    return result


class Predictor:
    def __init__(self, config: dict, models: dict):
        self.config, self.models = config, models

    @classmethod
    def load(cls, directory: Path) -> Predictor:
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        models = {kind: lgb.Booster(model_file=str(directory / f"{kind}.txt")) for kind in config["models"]}
        return cls(config, models)

    def predict_panel(self, panel: Panel) -> pd.DataFrame:
        cutoff = panel.dates[-1]
        baseline = baselines(panel, cutoff)
        learned = model_predictions(self.models, self.config["encoder"], self.config["features"],
                                    panel, cutoff, self.config.get("threads", 2)) if self.models else {}
        values = candidates(baseline, learned)[self.config["selected"]]
        return forecast_frame(values[:, panel.output_indices], panel.output_pairs, next_monday(cutoff))


def save_bundle(directory: Path, models: dict, encoder: dict, columns: list[str], selected: str,
                config: TrainingConfig, panel: Panel, extras: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for kind, model in models.items():
        model.save_model(str(directory / f"{kind}.txt"))
    metadata = {"version": "1.0.0", "selected": selected, "models": list(models), "encoder": encoder,
                "features": columns, "threads": config.threads, "training": asdict(config),
                "cutoff": str(panel.dates[-1].date()), "audit": panel.audit, **extras}
    (directory / "config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    panel.pairs.to_csv(directory / "pairs.csv", index=False)
