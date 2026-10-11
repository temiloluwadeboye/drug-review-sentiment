import pandas as pd

from src.prepare_data import clean_text, prepare, rating_to_label, split


def make_raw(rows):
    return pd.DataFrame(
        rows, columns=["MedicineName", "MedicineFor", "UserName", "Reviews", "Rating"]
    )


def test_clean_text_collapses_whitespace():
    assert clean_text("  works   great\n\ttoday ") == "works great today"


def test_rating_to_label():
    assert rating_to_label(10) == 1
    assert rating_to_label(7) == 1
    assert rating_to_label(4) == 0
    assert rating_to_label(1) == 0
    assert rating_to_label(5) is None
    assert rating_to_label(6) is None


def test_prepare_removes_duplicates_and_neutral_reviews():
    raw = make_raw(
        [
            ["Monistat 3", "For Yeast", "a", "Did not work", 1],
            ["Miconazole", "For Yeast", "b", "Did not work", 1],  # same text, other drug name
            ["Drug X", "For Pain", "c", "It was okay", 5],  # neutral
            ["Drug Y", "For Pain", "d", "Helped a lot", 9],
        ]
    )
    df = prepare(raw)
    assert sorted(df["text"]) == ["Did not work", "Helped a lot"]
    assert "UserName" not in df.columns


def test_prepare_drops_texts_with_conflicting_ratings():
    raw = make_raw(
        [
            ["Drug A", "For Pain", "a", "Same words", 10],
            ["Drug B", "For Pain", "b", "Same words", 1],
            ["Drug C", "For Pain", "c", "Helped a lot", 9],
        ]
    )
    df = prepare(raw)
    assert list(df["text"]) == ["Helped a lot"]


def test_split_has_no_overlap_and_keeps_every_row():
    df = pd.DataFrame(
        {"text": [f"review {i}" for i in range(100)], "label": [i % 2 for i in range(100)]}
    )
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == 100
    assert set(train["text"]).isdisjoint(test["text"])
    assert set(train["text"]).isdisjoint(val["text"])
    assert set(val["text"]).isdisjoint(test["text"])
