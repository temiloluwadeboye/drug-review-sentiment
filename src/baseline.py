"""Baseline model: TF-IDF + Logistic Regression.

Run from the project root:
    python -m src.baseline
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")


def evaluate(name, y_true, y_pred):
    """Print the scores for one model and return the headline numbers."""
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=3, zero_division=0))
    print("Confusion matrix (rows = true, columns = predicted; negative first):")
    print(confusion_matrix(y_true, y_pred))
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def top_words(pipeline, n=15):
    """The words/phrases that push hardest toward each label."""
    words = pipeline.named_steps["tfidf"].get_feature_names_out()
    weights = pipeline.named_steps["logreg"].coef_[0]
    ranked = sorted(zip(weights, words))
    return [w for _, w in ranked[:n]], [w for _, w in ranked[-n:][::-1]]


def main():
    # Step 1: load train.csv and val.csv
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")

    # Step 2: "dumb" comparison model (DummyClassifier, always guesses the most common label)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train["text"], train["label"])
    dummy_preds = dummy.predict(val["text"])

    # Step 3: build the Pipeline (TfidfVectorizer -> LogisticRegression)
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=5)),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )

    # Step 4: train on the training set only
    pipeline.fit(train["text"], train["label"])
    baseline_preds = pipeline.predict(val["text"])

    # Step 5: compare both models on the validation set
    results = {
        "dummy": evaluate("Dummy (always positive)", val["label"], dummy_preds),
        "baseline": evaluate("TF-IDF + Logistic Regression", val["label"], baseline_preds),
    }

    negative_words, positive_words = top_words(pipeline)
    print("\nMost negative words:", ", ".join(negative_words))
    print("Most positive words:", ", ".join(positive_words))

    # Step 6: save the whole pipeline (it turns raw text into a prediction,
    # so the app can use it directly) and the scores
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / "baseline.joblib")
    with open(RESULTS_DIR / "baseline_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR / 'baseline.joblib'}")
    print(f"Saved scores to {RESULTS_DIR / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
