"""Predict the sentiment of a single review with any of the three trained models.

Shared by the API (app/api.py) and the web page (app/streamlit_app.py).

    from src.predict import predict
    predict("This medicine changed my life", model="distilbert")
    # {'model': 'distilbert', 'label': 'positive', 'confidence': 0.98}
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import joblib
import torch

from src import lstm

MODEL_DIR = Path("models")
MODELS = ("baseline", "lstm", "distilbert")
LABELS = ("negative", "positive")

# The trained models are too big for GitHub, so they live on Hugging Face Hub.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "temiloluwadeboye/drug-review-sentiment-models")
REQUIRED_FILES = ("baseline.joblib", "lstm.pt", "lstm_vocab.json", "distilbert/model.safetensors")


@lru_cache(maxsize=None)
def ensure_models():
    """Download the models from Hugging Face Hub if they aren't on this machine yet.

    On a laptop that trained the models this does nothing. On a fresh server
    (like the online demo) it downloads them once into models/.
    """
    if all((MODEL_DIR / name).exists() for name in REQUIRED_FILES):
        return
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=HF_MODEL_REPO, local_dir=MODEL_DIR)


# ---------------------------------------------------------------------------
# Loading: each function runs once; @lru_cache hands back the same loaded
# model on every later call instead of reading it from disk again.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def load_baseline():
    ensure_models()
    return joblib.load(MODEL_DIR / "baseline.joblib")


@lru_cache(maxsize=None)
def load_lstm():
    ensure_models()
    with open(MODEL_DIR / "lstm_vocab.json") as f:
        vocab = json.load(f)
    model = lstm.LSTMClassifier(len(vocab))
    model.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location="cpu"))
    model.eval()
    return model, vocab


@lru_cache(maxsize=None)
def load_distilbert():
    # Imported here, not at the top: the transformers library takes ~20 seconds
    # to import on a laptop CPU, so only pay that cost when DistilBERT is used.
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ensure_models()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "distilbert")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "distilbert")
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Predicting: every model ends with "probability that the review is positive"
# ---------------------------------------------------------------------------

def positive_probability(text, model):
    if model == "baseline":
        pipeline = load_baseline()
        return float(pipeline.predict_proba([text])[0][1])   # column 1 = positive

    if model == "lstm":
        net, vocab = load_lstm()
        ids = torch.tensor([lstm.encode(text, vocab)])       # a batch of one review
        with torch.no_grad():
            score = net(ids)[0]
        return float(torch.sigmoid(score))                   # any number -> 0..1

    if model == "distilbert":
        net, tokenizer = load_distilbert()
        inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = net(**inputs).logits[0]                 # [negative score, positive score]
        return float(torch.softmax(logits, dim=0)[1])        # two scores -> two probabilities

    raise ValueError(f"Unknown model {model!r}; choose one of {MODELS}")


def predict(text, model="distilbert"):
    """Return the model's label and how confident it is in that label."""
    p_positive = positive_probability(text, model)
    label = LABELS[int(p_positive >= 0.5)]
    confidence = p_positive if label == "positive" else 1 - p_positive
    return {"model": model, "label": label, "confidence": round(confidence, 4)}
