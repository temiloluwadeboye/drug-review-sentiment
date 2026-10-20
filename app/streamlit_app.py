"""Web demo for the drug review sentiment models.

Run from the project root:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit runs this file directly, so make the project root importable (for src/).
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.predict import predict  # noqa: E402

MODEL_NAMES = {
    "distilbert": "DistilBERT (fine-tuned transformer)",
    "lstm": "LSTM (trained from scratch)",
    "baseline": "TF-IDF + Logistic Regression",
}
EXAMPLES = {
    "Positive": "This medicine changed my life. My migraines are gone and I have had no side effects at all.",
    "Negative": "I would not say this helped. Worst three weeks of my life and the pain never went away.",
    "Mixed": "It worked for my pain but the nausea was unbearable, so I had to stop taking it.",
}

st.set_page_config(page_title="Drug Review Sentiment", page_icon="💊")
st.title("💊 Drug Review Sentiment")
st.write(
    "Type a patient's review of a medication and the model predicts whether "
    "the patient was satisfied (positive) or not (negative)."
)

# --- Input -------------------------------------------------------------------
# Clicking an example button stores its text, and the text box shows it.
if "review" not in st.session_state:
    st.session_state.review = ""

st.caption("Try an example:")
columns = st.columns(len(EXAMPLES))
for column, (name, text) in zip(columns, EXAMPLES.items()):
    if column.button(name, use_container_width=True):
        st.session_state.review = text

review = st.text_area("Review", key="review", height=130, placeholder="e.g. It helped my sleep but...")
compare = st.toggle("Compare all three models")

# --- Prediction ----------------------------------------------------------------
if st.button("Predict", type="primary", disabled=not review.strip()):
    models = list(MODEL_NAMES) if compare else ["distilbert"]
    with st.spinner("Thinking... (the first run loads the models, which takes a few seconds)"):
        results = [predict(review, model) for model in models]

    for result in results:
        icon = "🟢" if result["label"] == "positive" else "🔴"
        st.subheader(f"{icon} {result['label'].capitalize()}")
        st.progress(result["confidence"], text=f"{MODEL_NAMES[result['model']]}: {result['confidence']:.0%} confident")

    if any(r["confidence"] < 0.65 for r in results):
        st.info("Low confidence usually means the review is mixed: good and bad in the same review.")

# --- About -----------------------------------------------------------------------
with st.expander("About this project"):
    st.markdown(
        """
Trained on 226,000 deduplicated patient reviews from Drugs.com
([DrugRev dataset](https://doi.org/10.17632/64cc5w5dxy.1), CC BY 4.0).
Ratings 7–10 = positive, 1–4 = negative.

| Model | Test macro F1 |
|---|---|
| TF-IDF + Logistic Regression | 0.828 |
| LSTM (trained from scratch) | 0.835 |
| DistilBERT (fine-tuned) | **0.874** |

Code: [github.com/temiloluwadeboye/drug-review-sentiment](https://github.com/temiloluwadeboye/drug-review-sentiment)

*For demonstration only. Not medical advice.*
"""
    )
