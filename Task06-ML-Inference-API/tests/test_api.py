import json
from fastapi.testclient import TestClient
from sklearn.datasets import load_breast_cancer

from app.main import app, FEATURES

client = TestClient(app, raise_server_exceptions=False)


def valid_payload():
    dataset = load_breast_cancer()
    sample = dataset.data[0]
    # Map sklearn's dataset feature names to the exact saved-model schema.
    values = dict(zip(FEATURES, sample.tolist()))
    return {"features": values}


def test_root_and_health():
    assert client.get("/").status_code == 200

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_valid_prediction_schema():
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"benign", "malignant"}
    assert body["predicted_class"] in {0, 1}
    assert set(body["probabilities"]) == {"benign", "malignant"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"].values())
    assert body["model_name"]
    assert "Educational demonstration only" in body["disclaimer"]
    expected_label = "malignant" if body["predicted_class"] == 1 else "benign"
    assert body["prediction"] == expected_label

def test_missing_feature_returns_422():
    payload = valid_payload()
    payload["features"].pop(FEATURES[0])

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unexpected_feature_returns_422():
    payload = valid_payload()
    payload["features"]["unexpected_feature"] = 1.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_features_object_returns_422():
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_non_finite_value_returns_422():
    payload = valid_payload()
    payload["features"][FEATURES[0]] = float("nan")

    # Send raw JSON so the test client does not reject NaN before the API.
    response = client.post(
        "/predict",
        content=json.dumps(payload, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422

def test_top_level_extra_field_returns_422():
    payload = valid_payload()
    payload["unexpected"] = "not allowed"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
