from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)

MODEL_FILES = {
    "baseline": Path("models/baseline.joblib"),
    "lstm": Path("models/lstm.pt"),
    "distilbert": Path("models/distilbert/model.safetensors"),
}


def needs(model):
    """Skip a test when that model's file isn't on this machine (they're not in git)."""
    return pytest.mark.skipif(not MODEL_FILES[model].exists(), reason=f"{model} model not available")


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@needs("distilbert")
def test_predict_positive_review():
    response = client.post("/predict", json={"text": "This medicine changed my life, no side effects at all!"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert 0.5 <= body["confidence"] <= 1.0


@needs("distilbert")
def test_predict_negative_review():
    response = client.post("/predict", json={"text": "Worst three weeks of my life. It did nothing for my pain."})
    assert response.json()["label"] == "negative"


@pytest.mark.parametrize("model", ["baseline", "lstm", "distilbert"])
def test_every_model_answers(model):
    if not MODEL_FILES[model].exists():
        pytest.skip(f"{model} model not available")
    response = client.post("/predict", json={"text": "It helped a lot.", "model": model})
    assert response.status_code == 200
    assert response.json()["model"] == model


def test_empty_text_is_rejected():
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_unknown_model_is_rejected():
    assert client.post("/predict", json={"text": "Fine.", "model": "gpt"}).status_code == 422
