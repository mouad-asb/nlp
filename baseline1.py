"""
Majority class classifier that always predicts the most frequent label.
This is the absolute floor.
"""

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)
df["target"] = (df["label"] == "opinion").astype(int)

X = df["text"]
y = df["target"]

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model   = DummyClassifier(strategy="most_frequent")
results = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"])

print("Baseline 1 — Majority Class")
print(f"  Accuracy : {results['test_accuracy'].mean():.4f} ± {results['test_accuracy'].std():.4f}")
print(f"  F1 macro : {results['test_f1_macro'].mean():.4f} ± {results['test_f1_macro'].std():.4f}")