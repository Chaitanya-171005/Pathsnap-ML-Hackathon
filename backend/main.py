"""Run with python -m uvicorn backend.main:app."""

import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from math import isfinite
import os
from pathlib import Path
from threading import BoundedSemaphore, Lock
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException

from frontend.models.phase5 import Phase5Result
from frontend.models.schemas import Capabilities, Geometry, Health, PredictionJob, PredictionRequest
from frontend.models.schemas import TripList, TripSummary
from src.decoding import MODELS

from .service import ApiError, InferenceService, MODES


LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.0.0"


def create_app(*, data_root=None, model_paths=None, timeout_s=None):
    root = Path(data_root) if data_root is not None else Path(
        os.environ.get("PATHSNAP_DATA_ROOT", ROOT / "data/raw"))
    if model_paths is None:
        model_paths = {phase: ROOT / path for phase, path in MODELS.items()}
        for phase in model_paths:
            override = os.getenv(f"PATHSNAP_{phase.upper()}_MODEL")
            if override:
                model_paths[phase] = Path(override)
    if timeout_s is None:
        timeout_s = float(os.getenv("PATHSNAP_PREDICTION_TIMEOUT_S", "12"))
    if not isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Prediction timeout must be finite and positive")

    @asynccontextmanager
    async def lifespan(application):
        application.state.inference = InferenceService(root, model_paths)
        application.state.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pathsnap")
        application.state.prediction_slot = BoundedSemaphore(1)
        application.state.results = OrderedDict()
        application.state.results_lock = Lock()
        try:
            yield
        finally:
            application.state.executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="PathSnap", version=VERSION, lifespan=lifespan)

    @app.exception_handler(ApiError)
    async def api_error(request: Request, error: ApiError):
        return JSONResponse(dict(code=error.code, message=error.message), status_code=error.status)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: RequestValidationError):
        return JSONResponse(dict(code="invalid_request", message="Request does not match API v1"),
                            status_code=422)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException):
        return JSONResponse(dict(code="http_error", message="Endpoint unavailable"),
                            status_code=error.status_code, headers=error.headers)

    @app.exception_handler(OSError)
    async def data_error(request: Request, error: OSError):
        LOGGER.warning("Local data access failed", exc_info=error)
        return JSONResponse(dict(code="data_unavailable", message="Local data is unavailable"),
                            status_code=503)

    @app.exception_handler(Exception)
    async def server_error(request: Request, error: Exception):
        LOGGER.error("PathSnap request failed", exc_info=error)
        return JSONResponse(dict(code="server_error", message="PathSnap could not complete the request"),
                            status_code=500)

    @app.get("/health", response_model=Health)
    def health():
        service = app.state.inference
        ready = "phase3" in service.bundles and bool(service.partitions())
        return dict(status="ok" if ready else "degraded", version=VERSION)

    @app.get("/api/v1/capabilities", response_model=Capabilities)
    def capabilities():
        return app.state.inference.capabilities()

    @app.get("/api/v1/trips", response_model=TripList)
    def trips(partition: str):
        return dict(trips=app.state.inference.trips(partition))

    @app.get("/api/v1/trips/{trip_id}", response_model=TripSummary)
    def trip(trip_id: str, partition: str):
        service = app.state.inference
        return service.summary(service.files(partition, trip_id))

    @app.get("/api/v1/trips/{trip_id}/geometry", response_model=Geometry)
    def geometry(trip_id: str, partition: str):
        service = app.state.inference
        return JSONResponse(service.geometry(service.files(partition, trip_id)))

    @app.post("/api/v1/predictions", response_model=Phase5Result | PredictionJob)
    async def predict(body: PredictionRequest):
        if body.mode not in MODES:
            raise ApiError(409, "mode_unavailable", "Prediction mode is unavailable")
        service = app.state.inference
        slot = app.state.prediction_slot
        if not slot.acquire(blocking=False):
            raise ApiError(503, "prediction_busy", "Server is processing another prediction")
        result_id = uuid4().hex

        def run():
            files = service.files(body.partition, body.trip_id)
            return JSONResponse(service.predict(files, body.mode, result_id))

        try:
            future = app.state.executor.submit(run)
        except BaseException:
            slot.release()
            raise
        future.add_done_callback(lambda _: slot.release())
        try:
            response = await asyncio.wait_for(asyncio.wrap_future(future), timeout=timeout_s)
        except TimeoutError as error:
            raise ApiError(504, "prediction_timeout", "Prediction exceeded the server deadline") from error

        if body.mode != "hmm_viterbi" and len(response.body) <= 4 * 1024 * 1024:
            with app.state.results_lock:
                results = app.state.results
                results[result_id] = response.body
                if len(results) > 8:
                    results.popitem(last=False)
        return response

    @app.get("/api/v1/results/{result_id}", response_model=PredictionJob)
    def result(result_id: str):
        with app.state.results_lock:
            body = app.state.results.get(result_id)
        if body is None:
            raise ApiError(404, "result_not_found", "Result unavailable")
        return Response(body, media_type="application/json")

    return app


app = create_app()
