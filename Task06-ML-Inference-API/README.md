# Task 06 — Real-Time ML Inference REST API

## Overview

This project exposes the saved machine-learning model from Task 04 through a REST API built with FastAPI.

The API accepts exactly 30 numerical features in JSON format and returns a predicted class together with class probabilities.

> **Educational demonstration only. This model is not intended for clinical diagnosis or treatment decisions.**

## Architecture

Client → FastAPI REST API → Pydantic Validation → Scikit-learn Pipeline → Prediction + Probabilities → JSON Response

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Health and model status |
| POST | `/predict` | Generate ML prediction |
| GET | `/docs` | Swagger/OpenAPI documentation |

## Example Request

Send a POST request to `/predict` with all 30 expected feature names and numerical values.

The request must contain a `features` object. Use the exact feature names provided by the saved model metadata. A complete example can be tested using the Swagger UI at `/docs`.

## Example Response

The API returns the predicted class, class probabilities, model name, and an educational-use disclaimer.

```json
{
  "prediction": "benign",
  "predicted_class": 0,
  "probabilities": {
    "benign": 0.98,
    "malignant": 0.02
  },
  "model_name": "Logistic Regression",
  "disclaimer": "Educational demonstration only. Not for clinical diagnosis or treatment decisions."
}

```

Note: This is an illustrative response; actual predictions and probabilities depend on the input.
## Model

- Algorithm: Logistic Regression
- Pipeline: StandardScaler + Logistic Regression
- Input: 30 numerical features
- Class 0: Benign
- Class 1: Malignant

## Input Validation

- Exactly 30 expected feature names are required.
- Unexpected feature names are rejected.
- Values must be finite numerical values.
- Invalid requests return HTTP 422.

## Local Setup

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

Swagger documentation: `http://127.0.0.1:8000/docs`

Health check: `http://127.0.0.1:8000/health`

## Testing

Run:

```bash
python -m pytest -v
```

Tests cover health checks, valid predictions, missing features, unexpected features, missing feature objects, and non-finite values.

## Docker

A Dockerfile is included for containerized deployment.

```bash
docker build -t rabtech-ml-api .
docker run -p 8000:8000 rabtech-ml-api
```

Docker was not locally verified because Docker is not installed in the development environment.

## Project Structure

```text
Task06-ML-Inference-API/
├── app/
│   ├── __init__.py
│   └── main.py
├── models/
│   ├── champion_model.joblib
│   └── champion_metadata.joblib
├── tests/
│   └── test_api.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## Technology Stack

Python, FastAPI, Pydantic, Uvicorn, Scikit-learn, Pandas, NumPy, Joblib, Pytest, Docker.

## Disclaimer

This project is created for educational and internship demonstration purposes. The predictions must not be used for clinical diagnosis, treatment decisions, or medical advice.
