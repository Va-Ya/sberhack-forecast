from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from forecast.contracts import InputError
from forecast.data import load_inputs
from forecast.model import Predictor

LOGGER = logging.getLogger(__name__)


def csv_response(frame: pd.DataFrame, filename: str) -> StreamingResponse:
    def chunks():
        for start in range(0, len(frame), 10_000):
            yield frame.iloc[start:start + 10_000].to_csv(index=False, header=start == 0, float_format="%.8f")
    return StreamingResponse(chunks(), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{filename}"'})


def create_app(predictor: Predictor | None = None, model_dir: Path | None = None) -> FastAPI:
    model_dir = model_dir or Path(os.environ.get("MODEL_DIR", "model"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = predictor or Predictor.load(model_dir)
        application.state.cache = None
        application.state.post_lock = threading.Lock()
        application.state.allowed_pairs = pd.read_csv(model_dir / "allowed_pairs.csv", dtype=str) \
            if (model_dir / "allowed_pairs.csv").exists() else None
        yield

    application = FastAPI(title="SberHack Forecast API", version="1.0.0", lifespan=lifespan)

    @application.exception_handler(InputError)
    async def input_error(_: Request, exc: InputError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail, "code": exc.code})

    @application.exception_handler(RequestValidationError)
    async def request_error(_: Request, exc: RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": str(exc), "code": "VALIDATION_ERROR"})

    @application.get("/health")
    def health(request: Request):
        config = request.app.state.predictor.config
        return {"status": "ok", "model_version": config.get("version", "test"),
                "model": config["selected"], "forecast_ready": request.app.state.cache is not None}

    def cached(request: Request) -> pd.DataFrame:
        frame = request.app.state.cache
        if frame is None:
            raise InputError("Сначала вызовите POST /predict", 409, "FORECAST_NOT_READY")
        return frame

    @application.post("/predict")
    def predict_post(request: Request, sales: UploadFile = File(...), stock: UploadFile = File(...)):
        state = request.app.state
        if not state.post_lock.acquire(blocking=False):
            raise InputError("Другой прогноз ещё рассчитывается", 409, "PREDICTION_IN_PROGRESS")
        try:
            panel = load_inputs(sales.file, stock.file, pairs=state.allowed_pairs)
            frame = state.predictor.predict_panel(panel)
            response = csv_response(frame, "forecast.csv")
            state.cache = frame
            return response
        except InputError:
            raise
        except Exception as exc:
            LOGGER.exception("Prediction failed; previous successful cache retained")
            raise InputError("Ошибка расчёта прогноза", 500, "PREDICTION_ERROR") from exc
        finally:
            state.post_lock.release()

    @application.get("/predict/all")
    def predict_all(request: Request, format: str = Query("csv")):
        if format not in {"csv", "json"}:
            raise InputError("format должен быть csv или json", 400, "INVALID_FORMAT")
        frame = cached(request)
        return csv_response(frame, "full_forecast.csv") if format == "csv" else JSONResponse(frame.to_dict("records"))

    @application.get("/predict")
    def predict_get(request: Request, article: str | None = None, warehouse: str | None = None):
        if not article and not warehouse:
            raise InputError("Укажите article или warehouse", 400, "MISSING_FILTER")
        frame = cached(request)
        mask = pd.Series(True, index=frame.index)
        if article:
            mask &= frame.article == article
        if warehouse:
            mask &= frame.warehouse_name == warehouse
        selected = frame.loc[mask]
        if selected.empty:
            raise InputError("Склад, товар или их сочетание не найдены", 404, "PAIR_NOT_FOUND")
        return csv_response(selected, "filtered_forecast.csv")

    return application


app = create_app()
