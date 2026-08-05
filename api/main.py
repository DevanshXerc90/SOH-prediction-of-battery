"""
api/main.py
-----------
FastAPI serving layer for the Lithium-ion Battery SoH LSTM model.

The LSTM was trained on per-cycle State of Health (SoH) values using a
look-back window of 3 cycles.  So to predict the SoH of the NEXT cycle,
you supply the SoH readings of the last 3 discharge cycles.

Run with:
    uvicorn api.main:app --reload
"""

import os
import json
import numpy as np
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

import tensorflow as tf
from tensorflow.keras.models import model_from_json

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the saved model — we use the B05 battery trained at the 50% split
# as a general representative model.  Adjust BATTERY / SPLIT to swap models.
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT       = "50%"
BATTERY     = "B05"
MODEL_JSON  = os.path.join(BASE_DIR, "4_LSTM_with_SoH", SPLIT, "model", f"{BATTERY}_model.json")
MODEL_WGTS  = os.path.join(BASE_DIR, "4_LSTM_with_SoH", SPLIT, "model", f"{BATTERY}_weights.weights.h5")

LOOK_BACK   = 3          # must match what was used during training
SOH_GOOD    = 0.8        # threshold for "Good" vs "Degraded"

# ---------------------------------------------------------------------------
# Global model holder — loaded once at startup, reused for every request
# ---------------------------------------------------------------------------

model = None   # will be set inside the lifespan handler


# ---------------------------------------------------------------------------
# Lifespan: load model before the first request is served
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LSTM model weights once when the server starts."""
    global model

    if not os.path.exists(MODEL_JSON):
        raise RuntimeError(f"Model JSON not found: {MODEL_JSON}")
    if not os.path.exists(MODEL_WGTS):
        raise RuntimeError(f"Model weights not found: {MODEL_WGTS}")

    with open(MODEL_JSON, "r") as f:
        model = model_from_json(f.read())

    model.load_weights(MODEL_WGTS)
    model.compile(loss="mae", optimizer="adam")   # needed to restore state
    print(f"[startup] Loaded model  →  {BATTERY} @ {SPLIT} split")

    yield   # <-- server is running and serving requests from here

    # anything after yield runs on shutdown
    print("[shutdown] Server stopped.")


from fastapi.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Battery SoH Prediction API",
    description=(
        "Predicts State of Health (SoH) for the next discharge cycle "
        "using an LSTM trained on NASA Prognostics battery data."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static dashboard files (index.html mounted at root)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# ---------------------------------------------------------------------------
# Pydantic input schema
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    soh_history: the SoH readings of the last 3 discharge cycles, in order
                 from oldest to most recent.

    SoH is a ratio: 1.0 = fully healthy, 0.0 = completely degraded.
    Typical range for usable batteries: 0.60 – 1.00.

    Example:
        {"soh_history": [0.917, 0.912, 0.907]}
    """
    soh_history: list[float] = Field(
        ...,
        min_length=LOOK_BACK,
        max_length=LOOK_BACK,
        description=f"Exactly {LOOK_BACK} consecutive per-cycle SoH readings (oldest first).",
        examples=[[0.917, 0.912, 0.907]],
    )

    @field_validator("soh_history")
    @classmethod
    def values_in_range(cls, v):
        for val in v:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"Each SoH value must be between 0.0 and 1.0, got {val}"
                )
        return v


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class PredictResponse(BaseModel):
    predicted_soh: float
    health_status: str   # "Good" or "Degraded"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Health check")
def health():
    """
    Returns {"status": "ok"} when the model is loaded and ready.
    If the model failed to load at startup the server won't reach this point.
    """
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, summary="Predict next-cycle SoH")
def predict(request: PredictRequest):
    """
    Predicts the State of Health (SoH) for the NEXT discharge cycle.

    **Input**
    - `soh_history`: list of exactly 3 SoH readings (oldest → most recent)

    **Output**
    - `predicted_soh`: float in [0, 1]
    - `health_status`: "Good" (SoH >= 0.8) or "Degraded" (SoH < 0.8)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # 1. Build the input array: shape (1, look_back, 1)
    #    — 1 sample, look_back time-steps, 1 feature (SoH)
    x = np.array(request.soh_history, dtype=np.float32)
    x = x.reshape(1, LOOK_BACK, 1)

    # 2. Run inference
    prediction = model.predict(x, verbose=0)   # shape: (1, 1)
    predicted_soh = float(prediction[0][0])

    # 3. Clamp to [0, 1] — model can occasionally overshoot slightly
    predicted_soh = max(0.0, min(1.0, predicted_soh))

    # 4. Classify
    health_status = "Good" if predicted_soh >= SOH_GOOD else "Degraded"

    return PredictResponse(
        predicted_soh=round(predicted_soh, 4),
        health_status=health_status,
    )
