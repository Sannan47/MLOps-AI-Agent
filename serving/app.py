"""Inference service (deployed to Kubernetes)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field


def _logger() -> logging.Logger:
    log = logging.getLogger("serving")
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(
            logging.Formatter(
                '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s",'
                '"message":%(message)s}'
            )
        )
        log.addHandler(h)
    log.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    log.propagate = False
    return log


def _msg(text: str) -> str:
    return json.dumps(text)


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1, max_length=100)


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str


def create_app() -> FastAPI:
    app = FastAPI(title="Prediction API")
    log = _logger()
    state: dict = {"model": None, "scaler": None, "meta": {}, "error": None}

    model_path = Path(os.environ.get("MODEL_PATH", "serving/model/model.joblib"))
    try:
        state["model"] = joblib.load(model_path)
        scaler_path = model_path.with_name("scaler.joblib")
        if scaler_path.exists():
            state["scaler"] = joblib.load(scaler_path)
        meta_path = model_path.with_name("metadata.json")
        if meta_path.exists():
            state["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        log.info(_msg(f"model loaded from {model_path}"))
    except Exception as e:  # noqa: BLE001 - readiness must report, not crash
        state["error"] = f"{type(e).__name__}: {e}"
        log.error(_msg(f"model load failed: {state['error']}"))

    threshold = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))
    app_version = os.environ.get("APP_VERSION", "dev")
    model_version = os.environ.get(
        "MODEL_VERSION", str(state["meta"].get("model_version", "unknown"))
    )

    @app.get("/health/live")
    def live():
        return {"status": "alive"}

    @app.get("/health/ready")
    def ready(response: Response):
        if state["model"] is None:
            response.status_code = 503
            return {"status": "not ready", "reason": state["error"] or "model not loaded"}
        return {"status": "ready"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        model = state["model"]
        if model is None:
            raise HTTPException(503, "model not loaded")
        n = getattr(model, "n_features_in_", len(req.features))
        if len(req.features) != n:
            raise HTTPException(422, f"expected {n} features, got {len(req.features)}")
        x = [req.features]
        if state["scaler"] is not None:
            x = state["scaler"].transform(x)
        proba = float(model.predict_proba(x)[0][1])
        return PredictResponse(
            prediction=int(proba >= threshold), probability=proba, model_version=model_version
        )

    @app.get("/version")
    def version():
        meta = state["meta"]
        return {
            "app_version": app_version,
            "model_name": meta.get("model_name", "breast_cancer_clf"),
            "model_version": model_version,
            "run_id": meta.get("run_id"),
            "data_version": meta.get("data_version"),
        }

    return app


app = create_app()
