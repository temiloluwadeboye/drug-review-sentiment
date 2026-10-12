"""Fine-tune a pretrained DistilBERT transformer for review sentiment.

Run from the project root (needs a GPU; use the Colab notebook):
    python -m src.distilbert
"""

import json
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models/distilbert")
RESULTS_DIR = Path("results")

SEED = 42
MODEL_NAME = "distilbert-base-uncased"   # the pretrained model downloaded from Hugging Face
MAX_LEN = 256          # word pieces per review; longer reviews are cut
BATCH_SIZE = 32
EPOCHS = 2             # it already knows English, so 2 passes are usually enough
LEARNING_RATE = 2e-5   # small: gently adjust what it knows instead of overwriting it
WARMUP_SHARE = 0.06    # start with tiny steps for the first 6% of training


# ---------------------------------------------------------------------------
# Block 1: the data
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Turns each review into DistilBERT's word-piece IDs with its own tokenizer."""

    def __init__(self, texts, labels, tokenizer):
        # truncation=True cuts reviews longer than MAX_LEN. No padding here:
        # each batch is padded only to its own longest review (faster).
        self.encodings = tokenizer(list(texts), truncation=True, max_length=MAX_LEN)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.labels[i]
        return item


# ---------------------------------------------------------------------------
# Block 2: training and evaluating
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scheduler, scaler, device):
    """Same four steps as the LSTM: forward, loss, backward, step."""
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        # autocast uses 16-bit numbers on the GPU where it's safe: about 2x faster
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            outputs = model(**batch)        # 1. forward (the model computes the loss itself
            loss = outputs.loss             # 2.          when it is given the labels)
        scaler.scale(loss).backward()       # 3. backward
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)              # 4. step
        scaler.update()
        scheduler.step()                    # adjust the learning rate for the next step
        total_loss += loss.item() * len(batch["labels"])
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Predict on every batch. Returns (true labels, predicted labels)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**batch).logits
            y_pred.extend(logits.argmax(dim=1).tolist())   # pick the label with the higher score
            y_true.extend(batch["labels"].tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Block 3: putting it all together
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    if device == "cpu":
        print("Warning: no GPU found. This will be extremely slow; use the Colab notebook.")

    # Step 1: load the same split as the other models
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    print(f"Train: {len(train):,} reviews   Val: {len(val):,} reviews")

    # Step 2: download the pretrained tokenizer and model.
    # num_labels=2 adds a new, randomly started output layer for negative/positive.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} weights")

    # Step 3: datasets and batch loaders (the collator pads each batch)
    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        ReviewDataset(train["text"], train["label"], tokenizer),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator,
    )
    val_loader = DataLoader(
        ReviewDataset(val["text"], val["label"], tokenizer),
        batch_size=BATCH_SIZE * 2, collate_fn=collator,
    )

    # Step 4: optimizer (AdamW = Adam made for transformers) and a learning-rate
    # schedule that warms up, then shrinks steadily to zero by the end
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * WARMUP_SHARE), total_steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    # Step 5: fine-tune, checking validation macro F1 after each epoch and keeping the best
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    best_f1, history = 0.0, []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_macro_f1": round(val_f1, 4)})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}   val macro F1 {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(MODEL_DIR)       # saves weights + settings
            tokenizer.save_pretrained(MODEL_DIR)   # the app needs the same tokenizer

    # Step 6: reload the best version, report and save the scores
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    y_true, y_pred = evaluate(model, val_loader, device)
    print("\n=== DistilBERT (best epoch) ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=3))

    results = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "train_reviews": len(train),
        "history": history,
    }
    with open(RESULTS_DIR / "distilbert_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model to {MODEL_DIR} and scores to {RESULTS_DIR / 'distilbert_metrics.json'}")


if __name__ == "__main__":
    main()
