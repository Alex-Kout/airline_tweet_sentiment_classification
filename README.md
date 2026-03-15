# Airline Tweet Sentiment Classification with ANN

Sentiment analysis on the [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment) dataset using a Artificial Neural Network (ANN) built with PyTorch. Three vectorisation methods were compared: TF-IDF, Word2Vec and FastText.

---

## Project Structure
```
airline-tweet-sentiment-ann/
│
├── data/
│   └── Tweets.csv
│
├── notebooks/
│   └── DL_NLP_exercise_1.ipynb
│
├── outputs/
│   ├── figures/
│   │   ├── sentiment_distribution.png
│   │   ├── training_loss_curve.png
│   │   └── confusion_matrix.png
│   └── results/
│       └── classification_report.txt
│
├── requirements.txt
└── README.md
```

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

| Method | Accuracy | Negative F1 | Neutral F1 | Positive F1 | Macro F1 |
|--------|----------|-------------|------------|-------------|----------|
| TF-IDF | **79.34%** | **0.86** | **0.63** | **0.73** | **0.74** |
| Word2Vec | 68.47% | 0.71 | 0.48 | 0.52 | 0.57 |
| FastText | 61.29% | — | — | — | — |

TF-IDF outperformed both embedding methods. Word2Vec and FastText were trained from scratch on ~14k tweets — too small a corpus to learn competitive embeddings, leading to severe overfitting.

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


