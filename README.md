# Bag of Words (BoW) — Text to Feature Vectors

A demonstration project showing how to convert text data into numerical feature vectors using the **Bag of Words** model. This is a fundamental NLP technique where each text document is represented as a vector of word counts from a shared vocabulary.

## What this project shows

| Concept | Description |
|---------|-------------|
| **Dataset** | 8 dummy movie reviews (4 positive, 4 negative) |
| **Vocabulary** | All unique words extracted from the dataset |
| **Feature Vectors** | Each review → a numerical vector of word counts |
| **Classifier** | Naive Bayes trained on BoW vectors → sentiment prediction |

## How Bag of Words Works

```
Raw Text:          "the movie was fantastic"
                         │
                         ▼
Vocabulary:  [amazing, bad, boring, fantastic, great, ...]
                         │
                         ▼
Feature Vector:  [0, 0, 0, 1, 0, ...]   ← count of each word
                         │
                         ▼
Machine Learning Model  ← learns patterns from vectors
```

## Files

| File | Description |
|------|-------------|
| `bow_demo.py` | Main script — runs the full BoW pipeline |
| `dataset.csv` | The dummy dataset (text + sentiment labels) |
| `feature_vectors.csv` | The generated BoW feature vectors |
| `vocabulary.json` | Vocabulary words and metadata |
| `SUMMARY.md` | Auto-generated summary report |

## Requirements

- Python 3.6+
- pandas
- numpy
- scikit-learn

Install with:

```bash
pip install pandas numpy scikit-learn
```

## Run the Demo

```bash
python bow_demo.py
```

## Sample Output

```
STEP 3 — Feature Vectors (Manual)
                 acting  amazing  avoid  bad  boring  brilliant  ...
Doc 1                0        1      0    0       0          0    ...
Doc 2                0        0      0    0       0          0    ...
Doc 3                1        0      0    0       0          1    ...
...
```

## Why this matters for GitHub

This project demonstrates:
- ✅ Understanding of **text preprocessing** and **feature extraction**
- ✅ Ability to convert **unstructured text → structured numerical data**
- ✅ Knowledge of **CountVectorizer** (industry-standard tool)
- ✅ **End-to-end ML pipeline**: data → vectors → model → prediction
- ✅ Clean, documented, reproducible Python code

## Future Improvements

- Add **TF-IDF** (Term Frequency-Inverse Document Frequency)
- Use **n-grams** (bigrams, trigrams) instead of single words
- Handle **stop words**, stemming, lemmatization
- Scale to larger real-world datasets