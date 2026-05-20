"""
FastAPI application for real-time credit card fraud detection inference.
"""

import logging
import os
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import PredictionResponse, TransactionFeatures

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Module-level model/scaler references populated during startup
_model = None
_scaler = None

# Feature order must match training data
FEATURE_ORDER = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "xgboost.joblib")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and scaler on startup; release on shutdown."""
    global _model, _scaler

    model_path = os.path.abspath(MODEL_PATH)
    scaler_path = os.path.abspath(SCALER_PATH)

    if os.path.exists(model_path):
        _model = joblib.load(model_path)
        logger.info("XGBoost model loaded from %s", model_path)
    else:
        _model = None
        logger.warning("Model file not found at %s — /predict will return 503", model_path)

    if os.path.exists(scaler_path):
        _scaler = joblib.load(scaler_path)
        logger.info("Scaler loaded from %s", scaler_path)
    else:
        _scaler = None
        logger.warning("Scaler file not found at %s — /predict will return 503", scaler_path)

    yield

    # Cleanup (nothing to release for joblib objects)
    _model = None
    _scaler = None


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Real-time fraud detection using an XGBoost model trained on the Kaggle Credit Card Fraud dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health check")
def health_check():
    """Return API status and whether the model is loaded."""
    return {
        "status": "ok",
        "model_loaded": _model is not None and _scaler is not None,
    }


@app.post("/predict", response_model=PredictionResponse, summary="Predict fraud probability")
def predict(transaction: TransactionFeatures) -> PredictionResponse:
    """
    Accept a single transaction and return fraud probability.

    - Converts input fields to a numpy array in training feature order
    - Scales using the fitted StandardScaler
    - Returns predict_proba from XGBoost; threshold = 0.5
    """
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model or scaler not loaded. Train the model first and restart the API.",
        )

    tx_dict = transaction.model_dump()
    feature_vector = np.array([[tx_dict[col] for col in FEATURE_ORDER]], dtype=np.float64)

    scaled_vector = _scaler.transform(feature_vector)
    fraud_proba = float(_model.predict_proba(scaled_vector)[0, 1])
    is_fraud = fraud_proba >= 0.5

    return PredictionResponse(
        fraud_probability=fraud_proba,
        is_fraud=is_fraud,
        model_used="xgboost",
    )
