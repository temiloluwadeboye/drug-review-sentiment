"""REST API for the drug review sentiment models.

Run from the project root:
    uvicorn app.api:app --reload

Then open http://127.0.0.1:8000/docs to try it in the browser.
"""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(
    title="Drug Review Sentiment API",
    description="Predicts whether a patient's drug review is positive or negative.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# The shape of the request and the response. FastAPI checks every request
# against ReviewIn and rejects bad ones (empty text, unknown model) with a
# clear error before our code runs.
# ---------------------------------------------------------------------------

class ReviewIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000, examples=["This medicine changed my life!"])
    model: Literal["baseline", "lstm", "distilbert"] = "distilbert"


class PredictionOut(BaseModel):
    model: str
    label: Literal["negative", "positive"]
    confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Simple check that the service is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict_review(review: ReviewIn):
    """Predict the sentiment of one review."""
    return predict(review.text, review.model)
