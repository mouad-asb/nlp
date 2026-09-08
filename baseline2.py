"""
Only TF-IDF + Logistic Regression
Naive NLP baseline.
"""

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)
df["target"] = (df["label"] == "opinion").astype(int)

X = df["text"]
y = df["target"]

model = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=300, binary=True,
                               ngram_range=(1, 2), min_df=3,
                               stop_words="english")),
    ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
])

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"])

print("Baseline 2 — TF-IDF + Logistic Regression (no KB, no hand-crafted features)")
print(f"  Accuracy : {results['test_accuracy'].mean():.4f} ± {results['test_accuracy'].std():.4f}")
print(f"  F1 macro : {results['test_f1_macro'].mean():.4f} ± {results['test_f1_macro'].std():.4f}")