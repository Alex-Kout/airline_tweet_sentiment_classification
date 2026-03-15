# Airline Tweet Sentiment Classification with ANN

Sentiment analysis on the [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset using a Artificial Neural Network (ANN) built with PyTorch. Three vectorisation methods were compared: TF-IDF, Word2Vec and FastText.

---


## Pipeline

1. **Text Cleaning** — remove URLs, mentions, hashtags, punctuation (NLTK)
2. **Tokenisation & Lemmatisation** — stopword removal + WordNet lemmatiser
3. **Vectorisation** — TF-IDF (1–2 grams, 15k features), Word2Vec, FastText
4. **Model** — 3-layer feedforward ANN: 512 → 256 → 128 → 3
5. **Imbalance Handling** — weighted CrossEntropyLoss
6. **Regularisation** — BatchNorm + Dropout (0.4 / 0.3 / 0.2)
7. **Training** — Adam optimiser, early stopping (patience=7), best weights restored

---

## Results

| Method | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro F1 | Epochs |
|--------|----------|-------------|------------|-------------|----------|--------|
| TF-IDF | **79.34%** | **0.87** | **0.62** | **0.74** | **0.74** | 8 |
| Word2Vec | 68.47% | 0.78 | 0.49 | 0.62 | 0.63 | 57 |
| FastText | 61.29% | 0.71 | 0.48 | 0.52 | 0.57 | 26 |

**TF-IDF is the clear winner**, outperforming Word2Vec by +11% and FastText by +18%. Word2Vec and FastText were trained from scratch on ~14k tweets — too small a corpus to learn competitive embeddings, resulting in noisy and unstable validation performance across epochs.

Notable: the TF-IDF best model was saved at **epoch 1** (val loss: 0.6950). After that, training loss dropped to near zero while validation loss kept rising — a sign of severe overfitting on the sparse high-dimensional features.

---

## Tech Stack

- **Deep Learning** — PyTorch
- **NLP** — NLTK, Gensim (Word2Vec, FastText)
- **ML** — Scikit-learn (TF-IDF, class weights, metrics)
- **Data** — Pandas, NumPy
- **Visualisation** — Matplotlib, Seaborn

---

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/DL_NLP_exercise_1.ipynb
```

---

## Dataset

[Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) — 14,640 tweets labelled as positive, neutral or negative across 6 US airlines.

| Class | Count | % |
|-------|-------|---|
| Negative | 9,178 | 63% |
| Neutral | 3,099 | 21% |
| Positive | 2,363 | 16% |


