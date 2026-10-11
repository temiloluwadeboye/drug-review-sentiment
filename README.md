# Drug Review Sentiment Classifier

Predicts whether a patient was satisfied with a medication from the text of their review, and compares three approaches:

1. TF-IDF + Logistic Regression (baseline)
2. LSTM neural network trained from scratch in PyTorch
3. Fine-tuned DistilBERT transformer

🚧 Work in progress. Results, demo link and setup instructions coming soon.

## Data

**DrugRev: A comprehensive customers reviews on drugs purchasing and satisfaction** (Mendeley Data, version 1, 2023)

- Dataset page: https://data.mendeley.com/datasets/64cc5w5dxy/1
- DOI: [10.17632/64cc5w5dxy.1](https://doi.org/10.17632/64cc5w5dxy.1)
- Direct download (`DrugReviews.csv`, ~170 MB): https://data.mendeley.com/public-files/datasets/64cc5w5dxy/files/288e24c1-4605-4c4b-9a88-c789f94dbe35/file_downloaded
- License: [Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

392,510 patient reviews collected from Drugs.com, each with the drug name, the condition it was taken for, a 1–10 rating, and more. The data file is not stored in this repository; download it from the link above into `data/raw/`.

**Changes made to the data:** ratings are converted into sentiment labels (7–10 = positive, 1–4 = negative; 5–6 = neutral, excluded from the main task). Any further cleaning steps will be listed here.

**Citation:**
Ahmed, T., Mia, M. M., Mazrur, M. B., Rana, M. S., Ekfa, R. I., & Rahman, W. (2023). *DrugRev: A comprehensive customers reviews on drugs purchasing and satisfaction* (Version 1) [Data set]. Mendeley Data. https://doi.org/10.17632/64cc5w5dxy.1

This project is not affiliated with or endorsed by the dataset authors or Drugs.com.
