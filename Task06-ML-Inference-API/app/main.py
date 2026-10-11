from pathlib import Path
import math

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"

model: Pipeline = joblib.load(MODEL_DIR / "champion_model.joblib")
metadata = joblib.load(MODEL_DIR / "champion_metadata.joblib")
FEATURES = metadata["feature_columns"]

app = FastAPI(
    title="RabTech ML Inference API",
    description=(
        "Educational REST API for a saved breast-cancer classification model. "
        "Not intended for clinical diagnosis or treatment decisions."
    ),
    version="1.0.0",
)




@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
    # Return safe error details without echoing invalid input values.
    errors = [
        {
            "loc": list(error.get("loc", ())),
            "msg": error.get("msg", "Invalid input"),
            "type": error.get("type", "value_error"),
        }
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    features: dict[str, float] = Field(
        ...,
        description="Exactly the 30 named numerical features expected by the model.",
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, value):
        expected = set(FEATURES)
        supplied = set(value)
        missing = sorted(expected - supplied)
        extra = sorted(supplied - expected)

        if missing:
            raise ValueError(f"Missing feature names: {missing}")
        if extra:
            raise ValueError(f"Unexpected feature names: {extra}")

        for name, number in value.items():
            if isinstance(number, bool) or not math.isfinite(number):
                raise ValueError(f"Feature '{name}' must be a finite number.")
        return value


class PredictionResponse(BaseModel):
    prediction: str
    predicted_class: int
    probabilities: dict[str, float]
    model_name: str
    disclaimer: str


@app.get("/")
def root():
    return {
        "service": "RabTech ML Inference API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": metadata["champion_model"],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    ordered_values = [request.features[name] for name in FEATURES]
    frame = pd.DataFrame([ordered_values], columns=FEATURES)

    predicted_class = int(model.predict(frame)[0])
    raw_probabilities = model.predict_proba(frame)[0]
    classes = model.classes_

    probability_map = {
        ("malignant" if int(label) == 1 else "benign"): float(probability)
        for label, probability in zip(classes, raw_probabilities)
    }

    return PredictionResponse(
        prediction="malignant" if predicted_class == 1 else "benign",
        predicted_class=predicted_class,
        probabilities=probability_map,
        model_name=metadata["champion_model"],
        disclaimer=(
            "Educational demonstration only. Not for clinical diagnosis "
            "or treatment decisions."
        ),
    )
