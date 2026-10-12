"""Final exam: score all three trained models once on the untouched test set.

Run from the project root (DistilBERT on CPU takes ~10-15 minutes):
    python -m src.evaluate_test
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

from src import distilbert, lstm

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_baseline(texts):
    """Load models/baseline.joblib and return its predictions (a list of 0/1)."""
    model = joblib.load(MODEL_DIR / "baseline.joblib")
    return list(model.predict(texts))


def predict_lstm(texts, labels):
    """Rebuild the LSTM, pour in the saved weights, and predict."""
    with open(MODEL_DIR / "lstm_vocab.json") as f:
        vocab = json.load(f)

    # 1. build an empty model with the same shape as the trained one
    model = lstm.LSTMClassifier(len(vocab))
    # 2. load the saved weights (the state_dict) into it
    model.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location=DEVICE))
    # 3. move it to the CPU/GPU
    model.to(DEVICE)

    loader = DataLoader(lstm.ReviewDataset(texts, labels, vocab), batch_size=256)
    _, y_pred = lstm.evaluate(model, loader, DEVICE)
    return y_pred


def predict_distilbert(texts, labels):
    """Load the fine-tuned model and its tokenizer, and predict."""
    # Hugging Face saves the whole model, so there's no "build it first" step
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "distilbert")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR / "distilbert").to(DEVICE)

    loader = DataLoader(
        distilbert.ReviewDataset(texts, labels, tokenizer),
        batch_size=64,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    _, y_pred = distilbert.evaluate(model, loader, DEVICE)
    return y_pred


def score(y_true, y_pred):
    """Return a dict with accuracy, macro F1 and negative recall
    (recall_score with pos_label=0), each rounded to 4 decimals."""
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "negative_recall": round(recall_score(y_true, y_pred, pos_label=0), 4),
    }


def plot_comparison(results):
    """Bar chart of macro F1 per model, saved to results/model_comparison.png."""
    names = list(results)
    scores = [results[name]["macro_f1"] for name in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, scores, color=["#9aa5b1", "#4c78a8", "#f58518"])
    ax.bar_label(bars, fmt="%.3f")              # write the score on top of each bar
    ax.set_ylim(0.75, 0.9)                      # zoom in so the differences are visible
    ax.set_ylabel("Macro F1 (test set)")
    ax.set_title("Drug review sentiment: model comparison")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def main():
    # Step 1: load the test set
    test = pd.read_csv(DATA_DIR / "test.csv")
    texts, labels = test["text"], test["label"]
    print(f"Test set: {len(test):,} reviews  (device: {DEVICE})")

    # Step 2: predict with each model
    predictions = {}
    print("Predicting with TF-IDF + Logistic Regression...")
    predictions["TF-IDF + LogReg"] = predict_baseline(texts)
    print("Predicting with LSTM...")
    predictions["LSTM"] = predict_lstm(texts, labels)
    print("Predicting with DistilBERT (slow on CPU, ~10-15 minutes)...")
    predictions["DistilBERT"] = predict_distilbert(texts, labels)

    # Step 3: score each model, print a small table, save the scores
    results = {name: score(labels, preds) for name, preds in predictions.items()}
    print(f"\n{'Model':<18}{'Accuracy':>10}{'Macro F1':>10}{'Neg. recall':>13}")
    for name, s in results.items():
        print(f"{name:<18}{s['accuracy']:>10.4f}{s['macro_f1']:>10.4f}{s['negative_recall']:>13.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "test_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Step 4: chart for the README
    plot_comparison(results)

    # Step 5: save DistilBERT's mistakes for error analysis
    errors = test.assign(prediction=predictions["DistilBERT"])
    errors = errors[errors["label"] != errors["prediction"]]
    errors[["text", "label", "prediction", "drug", "condition"]].to_csv(
        RESULTS_DIR / "distilbert_errors.csv", index=False
    )

    print(f"\nSaved {RESULTS_DIR / 'test_metrics.json'}, {RESULTS_DIR / 'model_comparison.png'}")
    print(f"Saved {len(errors):,} DistilBERT mistakes to {RESULTS_DIR / 'distilbert_errors.csv'}")


if __name__ == "__main__":
    main()
