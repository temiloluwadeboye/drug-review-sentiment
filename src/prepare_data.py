"""Clean the raw DrugRev data and split it into train / validation / test sets.

Run from the project root:
    python -m src.prepare_data

Input:  data/raw/DrugReviews.csv
Output: data/processed/train.csv, val.csv, test.csv  (columns: text, label, drug, condition, rating)
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/DrugReviews.csv")
OUT_DIR = Path("data/processed")

# Fixed seed so the split is identical every time the script runs.
seed = 42

# Ratings 7-10 count as positive, 1-4 as negative. 5-6 are left out because
# "mixed" reviews make the labels unclear.
POSITIVE_MIN = 7
NEGATIVE_MAX = 4


def clean_text(text: str) -> str:
    """Collapse repeated whitespace and trim the ends."""
    return " ".join(str(text).split())


def rating_to_label(rating: int):
    """1 = positive, 0 = negative, None = neutral (dropped)."""
    if rating >= POSITIVE_MIN:
        return 1
    if rating <= NEGATIVE_MAX:
        return 0
    return None


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one copy of each review text.

    Drugs.com shows the same review on both the brand-name and generic-name
    pages, so the raw data contains many exact copies. If copies landed in
    both the training and test sets, the model would be tested on reviews it
    had already seen (data leakage) and its scores would look too good.

    A few copies have different ratings; those labels are unreliable, so
    every copy of those texts is removed.
    """
    conflicting = df.groupby("text")["rating"].nunique()
    conflicting_texts = conflicting[conflicting > 1].index
    df = df[~df["text"].isin(conflicting_texts)]
    return df.drop_duplicates(subset="text", keep="first")


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn the raw table into a clean table of (text, label) examples."""
    df = pd.DataFrame(
        {
            "text": raw["Reviews"].map(clean_text),
            "rating": raw["Rating"],
            "drug": raw["MedicineName"],
            "condition": raw["MedicineFor"],
        }
    )
    # UserName is deliberately not carried over: it is not useful for
    # prediction and keeping reviewers' names out of our outputs is good practice.
    df = df[df["text"] != ""]
    df = remove_duplicates(df)
    df["label"] = df["rating"].map(rating_to_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label", "drug", "condition", "rating"]].reset_index(drop=True)


def split(df: pd.DataFrame):
    """80% train, 10% validation, 10% test, keeping the positive/negative mix equal in each."""
    train, rest = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=0.5, stratify=rest["label"], random_state=seed
    )
    return train, val, test


def main():
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    df = prepare(raw)
    train, val, test = split(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(OUT_DIR / f"{name}.csv", index=False)

    print(f"Raw rows:            {len(raw):,}")
    print(f"After cleaning:      {len(df):,}")
    print(f"Positive / negative: {df['label'].sum():,} / {(df['label'] == 0).sum():,}")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        print(f"{name:<6} {len(part):>8,} rows  ({part['label'].mean():.1%} positive)")


if __name__ == "__main__":
    main()
