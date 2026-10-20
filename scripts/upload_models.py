"""Upload the trained models to Hugging Face Hub, so the online demo can download them.

First log in once (your token is stored on your computer, never in this repo):
    hf auth login

Then, from the project root:
    python scripts/upload_models.py
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import HF_MODEL_REPO, MODEL_DIR, REQUIRED_FILES  # noqa: E402

RESULTS_DIR = Path("results")


def model_card():
    """The README shown on the model's Hugging Face page."""
    with open(RESULTS_DIR / "test_metrics.json") as f:
        scores = json.load(f)
    rows = "\n".join(
        f"| {name} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['negative_recall']:.3f} |"
        for name, s in scores.items()
    )
    return f"""---
license: apache-2.0
language: en
pipeline_tag: text-classification
base_model: distilbert-base-uncased
tags:
- sentiment-analysis
- drug-reviews
---

# Drug review sentiment models

Three models that predict whether a patient's drug review is **positive** (rating 7-10)
or **negative** (rating 1-4), trained for the project
[github.com/temiloluwadeboye/drug-review-sentiment](https://github.com/temiloluwadeboye/drug-review-sentiment).

| File | Model |
|---|---|
| `baseline.joblib` | TF-IDF + Logistic Regression (scikit-learn pipeline) |
| `lstm.pt` + `lstm_vocab.json` | LSTM trained from scratch in PyTorch (weights + vocabulary) |
| `distilbert/` | Fine-tuned `distilbert-base-uncased` (Hugging Face format) |

## Test-set results (22,630 held-out reviews)

| Model | Accuracy | Macro F1 | Negative recall |
|---|---|---|---|
{rows}

## Training data

[DrugRev](https://doi.org/10.17632/64cc5w5dxy.1) (Ahmed et al., 2023, Mendeley Data, CC BY 4.0):
patient reviews from Drugs.com, deduplicated to 226,291 reviews; neutral ratings (5-6) removed.

## Limitations

- Struggles with mixed reviews (benefits and side effects in the same review) and very short reviews.
- Labels come from star ratings, which sometimes disagree with the review text.
- English only. For demonstration and research, **not for medical decisions**.
"""


def main():
    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).exists()]
    if missing:
        sys.exit(f"Missing model files in {MODEL_DIR}/: {missing}. Train the models first.")

    api = HfApi()
    print(f"Logged in as: {api.whoami()['name']}")

    api.create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(
        path_or_fileobj=model_card().encode(), path_in_repo="README.md", repo_id=HF_MODEL_REPO
    )
    print("Uploading models (about 290 MB, this can take a few minutes)...")
    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=HF_MODEL_REPO,
        ignore_patterns=[".gitkeep"],
        commit_message="Upload trained models",
    )
    print(f"Done: https://huggingface.co/{HF_MODEL_REPO}")


if __name__ == "__main__":
    main()
