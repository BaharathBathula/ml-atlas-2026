from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model_service import ModelService

app = FastAPI(
    title="ML Atlas 2026 Inference API",
    description="Production-style inference API for ML Atlas 2026.",
    version="1.0.0",
)

MODEL_PATH = os.getenv("ML_ATLAS_MODEL_PATH", "artifacts/demo_iris_pipeline.pkl")
service = ModelService(MODEL_PATH)


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    model_path: str
    predictions: List[Any]
    probabilities: List[Dict[str, float]] | None = None


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": service.is_loaded,
        "model_path": MODEL_PATH,
    }


@app.get("/model-info")
def model_info():
    return service.info()


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail=(
                "No model artifact is loaded. Run `python scripts/train_demo_model.py` "
                "or set ML_ATLAS_MODEL_PATH to a valid fitted pipeline."
            ),
        )

    try:
        df = pd.DataFrame(payload.records)
        pred, probs = service.predict(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictionResponse(
        model_path=MODEL_PATH,
        predictions=pred,
        probabilities=probs,
    )
