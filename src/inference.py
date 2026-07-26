"""
Production-style inference module.

Loads a saved model artifact + preprocessing pipeline,
validates the feature schema, and returns a structured prediction.
A minimal FastAPI app is included at the bottom.

Usage (CLI):
    python -m src.inference --model artifacts/lgbm_personalized.pkl \\
                            --features '{"PLV_FP_Alpha": 0.42, ...}'

Usage (API):
    uvicorn src.inference:app --port 8000
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_SCHEMA_PATH = Path("artifacts/feature_schema.json")
MODEL_VERSION = "1.0.0"


# ── Core inference logic ───────────────────────────────────────────────────────

def load_artifact(model_path: Path) -> tuple[Any, Any, list[str]]:
    """
    Load (model, preprocessing_pipeline, feature_names) from a joblib file.

    The artifact is expected to be saved as:
        {"model": ..., "pipeline": ..., "feature_names": [...], "version": "..."}
    """
    import joblib
    artifact = joblib.load(model_path)
    model         = artifact["model"]
    pipeline      = artifact["pipeline"]
    feature_names = artifact["feature_names"]
    logger.info("Loaded artifact from %s (version=%s)", model_path, artifact.get("version", "?"))
    return model, pipeline, feature_names


def save_artifact(
    model: Any,
    pipeline: Any,
    feature_names: list[str],
    output_path: Path,
    version: str = MODEL_VERSION,
) -> None:
    """Save model + pipeline + feature names as a single joblib artifact."""
    import joblib
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model":         model,
        "pipeline":      pipeline,
        "feature_names": feature_names,
        "version":       version,
    }
    joblib.dump(artifact, output_path)
    logger.info("Saved artifact → %s", output_path)


def predict_window(
    feature_window: dict[str, float],
    model: Any,
    pipeline: Any,
    feature_names: list[str],
    low_confidence_threshold: float = 0.6,
) -> dict[str, Any]:
    """
    Classify one feature window (one 2-second EEG epoch).

    Parameters
    ----------
    feature_window : dict[str, float]
        Feature name → value. Must contain all names in feature_names.
    model : fitted classifier
    pipeline : fitted sklearn Pipeline (imputer + scaler)
    feature_names : list[str]
        Expected feature names in order.
    low_confidence_threshold : float
        P(predicted_class) below this triggers a low-confidence flag.

    Returns
    -------
    dict with keys: predicted_class, confidence, low_confidence, model_version
    """
    # Validate schema
    missing = [f for f in feature_names if f not in feature_window]
    if missing:
        raise ValueError(f"Missing features: {missing[:5]}{'...' if len(missing)>5 else ''}")

    # Assemble feature vector in the correct column order
    x = np.array([[feature_window.get(f, np.nan) for f in feature_names]], dtype=float)
    x = pipeline.transform(x)

    t0 = time.perf_counter()
    y_pred = model.predict(x)[0]
    proba  = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
    latency_us = (time.perf_counter() - t0) * 1e6

    confidence = float(np.max(proba)) if proba is not None else float("nan")
    classes    = list(model.classes_) if hasattr(model, "classes_") else None
    class_name = str(classes[int(y_pred)]) if classes is not None else str(y_pred)

    return {
        "predicted_class":    class_name,
        "predicted_label":    int(y_pred),
        "confidence":         round(confidence, 4),
        "low_confidence":     bool(confidence < low_confidence_threshold),
        "latency_us":         round(latency_us, 2),
        "model_version":      MODEL_VERSION,
        "probabilities":      {str(c): round(float(p), 4) for c, p in zip(classes or range(len(proba or [])), proba or [])}
                              if proba is not None else {},
    }


# ── FastAPI application ────────────────────────────────────────────────────────

def _build_app():
    """Build and return the FastAPI app. Imported lazily to keep inference.py lightweight."""
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(
        title="EEG Meditation Classifier",
        description=(
            "Classifies a pre-extracted EEG feature window as meditation or thinking. "
            "This endpoint expects feature vectors already extracted from raw EEG. "
            "It does NOT accept raw EEG signals."
        ),
        version=MODEL_VERSION,
    )

    # Default artifact path — override with env var MODEL_PATH
    import os
    _model_path = Path(os.environ.get("MODEL_PATH", "artifacts/lgbm_personalized.pkl"))
    _model, _pipeline, _feature_names = None, None, None

    def _load():
        nonlocal _model, _pipeline, _feature_names
        if not _model_path.exists():
            raise RuntimeError(
                f"Model artifact not found at {_model_path}. "
                "Run the personalized evaluation script first."
            )
        _model, _pipeline, _feature_names = load_artifact(_model_path)

    class PredictRequest(BaseModel):
        features: dict[str, float]
        low_confidence_threshold: float = 0.6

    @app.on_event("startup")
    def startup():
        _load()

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": _model is not None}

    @app.get("/model-info")
    def model_info():
        return {
            "version":        MODEL_VERSION,
            "n_features":     len(_feature_names) if _feature_names else 0,
            "feature_names":  _feature_names or [],
            "model_path":     str(_model_path),
            "disclaimer":     (
                "This model is a research proof-of-concept trained on a small dataset (20 subjects). "
                "It is NOT validated for clinical use. "
                "The confidence score is not a measure of meditation depth."
            ),
        }

    @app.post("/predict")
    def predict(req: PredictRequest):
        if _model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            result = predict_window(
                req.features,
                _model,
                _pipeline,
                _feature_names,
                req.low_confidence_threshold,
            )
            return result
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    return app


# Build app at module level so `uvicorn src.inference:app` works
try:
    app = _build_app()
except ImportError:
    app = None  # FastAPI not installed
    logger.info("FastAPI not available — inference API disabled. Install fastapi and uvicorn to enable.")
