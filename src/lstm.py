"""LSTM sentiment classifier, built and trained from scratch in PyTorch.

Run from the project root:
    python -m src.lstm --sample 5000     # quick test on your laptop (CPU)
    python -m src.lstm                   # full run (use Google Colab with a GPU)
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

SEED = 42
VOCAB_SIZE = 20_000   # keep the 20,000 most common words
MAX_LEN = 200         # cut/pad every review to 200 words
BATCH_SIZE = 64
EMBED_DIM = 128       # size of each word's vector
HIDDEN_DIM = 128      # size of the LSTM's memory
EPOCHS = 5
LEARNING_RATE = 1e-3
PATIENCE = 2          # stop if validation F1 doesn't improve for 2 epochs

PAD, UNK = "<pad>", "<unk>"


# ---------------------------------------------------------------------------
# Block 1: turning text into numbers
# ---------------------------------------------------------------------------

def tokenize(text):
    """Lowercase the text and split it into a list of words."""
    return re.findall(r"[a-z']+", str(text).lower())


def build_vocab(texts):
    """Give the VOCAB_SIZE most common training words an ID: {"<pad>": 0, "<unk>": 1, "the": 2, ...}."""
    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))
    vocab = {PAD: 0, UNK: 1}
    for word, _ in counts.most_common(VOCAB_SIZE - len(vocab)):
        vocab[word] = len(vocab)
    return vocab


def encode(text, vocab):
    """Turn one review into exactly MAX_LEN word IDs.

    Padding goes at the START ("pre-padding"), so the LSTM reads the padding
    first and the real words last. Its memory at the final step then reflects
    the actual review instead of a long run of padding.
    """
    ids = [vocab.get(word, vocab[UNK]) for word in tokenize(text)][:MAX_LEN]
    return [vocab[PAD]] * (MAX_LEN - len(ids)) + ids


# ---------------------------------------------------------------------------
# Block 2: serving the data in batches
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    """Holds the encoded reviews and labels so a DataLoader can serve them in batches."""

    def __init__(self, texts, labels, vocab):
        # Encode everything once up front, so each epoch doesn't redo the work.
        self.x = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long)
        self.y = torch.tensor(list(labels), dtype=torch.float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


# ---------------------------------------------------------------------------
# Block 3: the neural network
# ---------------------------------------------------------------------------

class LSTMClassifier(nn.Module):
    """Embedding -> LSTM -> Linear (one output score per review)."""

    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        # x: (batch, MAX_LEN) word IDs
        vectors = self.embedding(x)              # (batch, MAX_LEN, EMBED_DIM)
        _, (hidden, _) = self.lstm(vectors)      # hidden: (1, batch, HIDDEN_DIM)
        memory = self.dropout(hidden[-1])        # the memory after the last word
        return self.out(memory).squeeze(1)       # (batch,) one score per review


# ---------------------------------------------------------------------------
# Block 4: training and evaluating
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """One pass over the training data. Returns the average loss."""
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()          # clear the previous batch's corrections
        scores = model(x)              # 1. forward: make predictions
        loss = loss_fn(scores, y)      # 2. loss: how wrong were they?
        loss.backward()                # 3. backward: work out each weight's share of the blame
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keep corrections from blowing up
        optimizer.step()               # 4. step: nudge every weight to reduce the loss
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Predict on every batch. Returns (true labels, predicted labels)."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():              # no learning here, so skip the blame bookkeeping
        for x, y in loader:
            scores = model(x.to(device))
            y_pred.extend((scores > 0).long().tolist())   # score > 0 means probability > 50%
            y_true.extend(y.long().tolist())
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Block 5: putting it all together
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="train on only this many reviews")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Step 1: load the data (a smaller sample for quick tests)
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    if args.sample:
        train = train.sample(args.sample, random_state=SEED)
        val = val.sample(min(args.sample // 4, len(val)), random_state=SEED)
    print(f"Train: {len(train):,} reviews   Val: {len(val):,} reviews")

    # Step 2: vocabulary from the TRAINING texts only (no peeking at validation)
    vocab = build_vocab(train["text"])
    print(f"Vocabulary: {len(vocab):,} words")

    # Step 3: datasets and batch loaders
    train_loader = DataLoader(
        ReviewDataset(train["text"], train["label"], vocab), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(ReviewDataset(val["text"], val["label"], vocab), batch_size=BATCH_SIZE)

    # Step 4: model, loss and optimizer
    model = LSTMClassifier(len(vocab)).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} weights to learn")

    # Step 5: train, checking validation F1 after every epoch
    MODEL_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    best_f1, epochs_without_improvement, history = 0.0, 0, []
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        y_true, y_pred = evaluate(model, val_loader, device)
        val_f1 = f1_score(y_true, y_pred, average="macro")
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4), "val_macro_f1": round(val_f1, 4)})
        print(f"Epoch {epoch}: train loss {train_loss:.4f}   val macro F1 {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, epochs_without_improvement = val_f1, 0
            torch.save(model.state_dict(), MODEL_DIR / "lstm.pt")   # keep the best version
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Validation F1 stopped improving, stopping early.")
                break

    # Step 6: reload the best version, report and save
    model.load_state_dict(torch.load(MODEL_DIR / "lstm.pt", map_location=device))
    y_true, y_pred = evaluate(model, val_loader, device)
    print("\n=== LSTM (best epoch) ===")
    print(classification_report(y_true, y_pred, target_names=["negative", "positive"], digits=3))

    with open(MODEL_DIR / "lstm_vocab.json", "w") as f:
        json.dump(vocab, f)
    results = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "train_reviews": len(train),
        "history": history,
    }
    name = "lstm_metrics_sample.json" if args.sample else "lstm_metrics.json"
    with open(RESULTS_DIR / name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved model to {MODEL_DIR / 'lstm.pt'} and scores to {RESULTS_DIR / name}")


if __name__ == "__main__":
    main()
