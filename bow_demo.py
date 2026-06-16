"""
========================================================================
BAG OF WORDS (BoW) — From Text to Feature Vectors
========================================================================
A complete demonstration of converting text data into numerical
feature vectors using the Bag of Words model.

What this script does:
  1. Creates a dummy dataset of text documents (movie reviews)
  2. Builds a vocabulary from all unique words
  3. Converts each document into a feature vector
  4. Displays results in a readable format
  5. Shows how BoW vectors feed into machine learning
========================================================================
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import json

# ======================================================================
# STEP 1 — Create a dummy dataset (text documents with labels)
# ======================================================================

DATA = [
    "the movie was fantastic amazing loved it",
    "terrible movie waste of time hated it",
    "great acting brilliant performance enjoyed",
    "boring film slow waste of money",
    "loved the story amazing characters great",
    "worst movie ever disappointed horrible",
    "excellent visuals stunning direction loved",
    "bad acting pathetic story avoid it",
]

LABELS = [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative

print("=" * 70)
print("STEP 1 — Raw Dataset (Documents)")
print("=" * 70)
for i, (doc, label) in enumerate(zip(DATA, LABELS)):
    sentiment = "POSITIVE" if label == 1 else "NEGATIVE"
    print(f"  Doc {i+1}: [{sentiment}] \"{doc}\"")

# ======================================================================
# STEP 2 — Build Vocabulary (custom implementation for clarity)
# ======================================================================

print("\n" + "=" * 70)
print("STEP 2 — Build Vocabulary (all unique words across documents)")
print("=" * 70)

word_counts = Counter()
for doc in DATA:
    word_counts.update(doc.lower().split())

# Sort alphabetically for consistency
vocabulary = sorted(word_counts.keys())
print(f"\n  Vocabulary size: {len(vocabulary)} words")
print(f"  Words: {vocabulary}")

# ======================================================================
# STEP 3 — Convert Documents to Feature Vectors (Manual BoW)
# ======================================================================

print("\n" + "=" * 70)
print("STEP 3 — Converting Documents to Feature Vectors (Manual)")
print("=" * 70)

# Create word → index mapping
word_to_idx = {word: i for i, word in enumerate(vocabulary)}

bow_vectors_manual = []
for doc in DATA:
    vec = [0] * len(vocabulary)
    for word in doc.lower().split():
        idx = word_to_idx[word]
        vec[idx] += 1
    bow_vectors_manual.append(vec)

# Display as DataFrame for readability
df_manual = pd.DataFrame(
    bow_vectors_manual,
    columns=vocabulary,
    index=[f"Doc {i+1}" for i in range(len(DATA))]
)
print("\nFeature Vectors (rows = documents, columns = words):\n")
print(df_manual.to_string())

# ======================================================================
# STEP 4 — Using sklearn's CountVectorizer (production-ready)
# ======================================================================

print("\n" + "=" * 70)
print("STEP 4 — Using sklearn CountVectorizer (industry standard)")
print("=" * 70)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(DATA)

feature_names = vectorizer.get_feature_names_out()
print(f"\n  Vocabulary size: {len(feature_names)} words")
print(f"  Words: {list(feature_names)}\n")

df_sklearn = pd.DataFrame(
    X.toarray(),
    columns=feature_names,
    index=[f"Doc {i+1}" for i in range(len(DATA))]
)
print(df_sklearn.to_string())

# ======================================================================
# STEP 5 — Show how BoW feeds into a classifier
# ======================================================================

print("\n" + "=" * 70)
print("STEP 5 — Quick ML demo: Train a classifier on BoW features")
print("=" * 70)

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, LABELS, test_size=0.25, random_state=42
)

# Train classifier
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

print(f"\n  Test accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"\n  Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

# ======================================================================
# STEP 6 — Demo on new unseen text
# ======================================================================

print("\n" + "=" * 70)
print("STEP 6 — Predict sentiment on new reviews")
print("=" * 70)

new_reviews = [
    "amazing movie loved the story",
    "boring and terrible waste of time",
    "great acting and brilliant direction",
]

new_vectors = vectorizer.transform(new_reviews)
predictions = clf.predict(new_vectors)
probabilities = clf.predict_proba(new_vectors)

for review, pred, prob in zip(new_reviews, predictions, probabilities):
    sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
    confidence = max(prob) * 100
    print(f"\n  Review: \"{review}\"")
    print(f"  → Predicted: {sentiment} (confidence: {confidence:.1f}%)")
    print(f"  → Feature vector: {vectorizer.transform([review]).toarray().tolist()[0]}")

# ======================================================================
# STEP 7 — Save the dataset and results to files
# ======================================================================

print("\n" + "=" * 70)
print("STEP 7 — Saving outputs to files")
print("=" * 70)

# Save dataset as CSV
dataset_df = pd.DataFrame({
    "text": DATA,
    "label": LABELS,
    "sentiment": ["Positive" if l == 1 else "Negative" for l in LABELS]
})
dataset_df.to_csv("dataset.csv", index=False)
print("  ✓ Saved dataset.csv")

# Save feature vectors as CSV
df_sklearn.to_csv("feature_vectors.csv")
print("  ✓ Saved feature_vectors.csv")

# Save vocabulary as JSON
vocab_dict = {
    "vocabulary_size": len(feature_names),
    "words": feature_names.tolist(),
    "vectorizer_config": str(vectorizer)
}
with open("vocabulary.json", "w") as f:
    json.dump(vocab_dict, f, indent=2)
print("  ✓ Saved vocabulary.json")

# Save a summary report
with open("SUMMARY.md", "w") as f:
    f.write("# Bag of Words Project — Summary\n\n")
    f.write(f"- **Dataset size:** {len(DATA)} documents\n")
    f.write(f"- **Vocabulary size:** {len(feature_names)} unique words\n")
    f.write(f"- **Feature vector dimension:** {len(feature_names)}\n")
    f.write(f"- **Classifier accuracy:** {accuracy_score(y_test, y_pred):.2f}\n")
    f.write(f"- **Labels:** 0 = Negative, 1 = Positive\n\n")
    f.write("## How Bag of Words Works\n\n")
    f.write("1. Take a collection of text documents.\n")
    f.write("2. Build a vocabulary of all unique words across all documents.\n")
    f.write("3. For each document, count how many times each vocabulary word appears.\n")
    f.write("4. Each document becomes a numerical vector of word counts.\n")
    f.write("5. These vectors can be used as input to machine learning models.\n")
print("  ✓ Saved SUMMARY.md")

print("\n" + "=" * 70)
print("DONE! All files generated successfully.")
print("=" * 70)