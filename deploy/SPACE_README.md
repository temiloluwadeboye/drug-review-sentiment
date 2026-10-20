---
title: Drug Review Sentiment
emoji: 💊
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Is a patient's drug review positive or negative?
---

# Drug Review Sentiment

Type a patient's review of a medication and see whether three different models think the
patient was satisfied: a TF-IDF + Logistic Regression baseline, an LSTM trained from scratch,
and a fine-tuned DistilBERT transformer.

The first visit after a quiet period takes about a minute while the Space wakes up and loads the models.

- Code, results and error analysis: [github.com/temiloluwadeboye/drug-review-sentiment](https://github.com/temiloluwadeboye/drug-review-sentiment)
- Models: [drug-review-sentiment-models](https://huggingface.co/temiloluwadeboye/drug-review-sentiment-models)

*For demonstration only. Not medical advice.*
