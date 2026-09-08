"""
Stage 1 classifiers with binary KB
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from knowledge_base import KnowledgeBase
from features import extract_all

df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)
df["target"] = (df["label"] == "opinion").astype(int)

kb = KnowledgeBase(weights_path=None)
X, tfidf, scaler = extract_all(df["text"], kb, fit=True)
y = df["target"].values

cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Linear SVM":          LinearSVC(C=1.0, max_iter=2000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
}

print("Model 1 — Binary KB\n")
print(f"  {'Model':<25} {'Accuracy':>10} {'F1 macro':>10}")
print(f"  {'-'*25} {'-'*10} {'-'*10}")

for name, model in models.items():
    res = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"])
    acc = res["test_accuracy"].mean()
    f1  = res["test_f1_macro"].mean()
    print(f"  {name:<25} {acc:.4f} ± {res['test_accuracy'].std():.4f}"
          f"    {f1:.4f} ± {res['test_f1_macro'].std():.4f}")