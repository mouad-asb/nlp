"""
Stage 1 ablation study: we isolates the contribution of each KB component.

  Variant A : no KB features 
  Variant B : binary KB, no cancellation rules
  Variant C : binary KB, with cancellation rules
  Variant D : PMI-weighted KB, with cancellation rules

B - C isolates cancellation rules.
C - D isolates PMI calibration.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from knowledge_base import KnowledgeBase
from features import pos_features, sentiment_features

df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)
y  = df["target"] = (df["label"] == "opinion").astype(int)

cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

# Shared TF-IDF fitted once on full data
tfidf = TfidfVectorizer(max_features=300, binary=True,
                        ngram_range=(1, 2), min_df=3,
                        stop_words="english")
tfidf_mat = tfidf.fit_transform(df["text"])


def build_matrix(df, kb=None, use_cancellation=True):
    """Build feature matrix with optional KB features."""
    records = []
    for text in df["text"]:
        row = {}
        row.update(pos_features(text))
        row.update(sentiment_features(text))
        if kb is not None:
            row.update(kb.score(text, use_cancellation=use_cancellation))
        records.append(row)
    dense  = pd.DataFrame(records).fillna(0)
    scaled = StandardScaler().fit_transform(dense.values)
    return hstack([csr_matrix(scaled), tfidf_mat])


def evaluate(name, X):
    res = cross_validate(model, X, y, cv=cv,
                         scoring=["accuracy", "f1_macro"])
    acc = res["test_accuracy"].mean()
    f1  = res["test_f1_macro"].mean()
    print(f"  {name:<45} Acc: {acc:.4f} ± {res['test_accuracy'].std():.4f}"
          f"  F1: {f1:.4f} ± {res['test_f1_macro'].std():.4f}")
    return f1


print("Ablation Study : Logistic Regression, 5-fold CV\n")
print(f"  {'Variant':<45} {'Accuracy':>15} {'F1 macro':>15}")
print(f"  {'-'*45} {'-'*15} {'-'*15}")

print("\nExtracting features for Variant A ...")
X_A = build_matrix(df, kb=None)
f1_A = evaluate("A : POS + Sentiment + TF-IDF (no KB)", X_A)

kb_binary = KnowledgeBase(weights_path=None)

print("\nExtracting features for Variant B ...")
X_B = build_matrix(df, kb=kb_binary, use_cancellation=False)
f1_B = evaluate("B : + Binary KB, no cancellation", X_B)

print("\nExtracting features for Variant C ...")
X_C = build_matrix(df, kb=kb_binary, use_cancellation=True)
f1_C = evaluate("C : + Binary KB, with cancellation", X_C)

kb_pmi = KnowledgeBase(weights_path="kb_weights.json")

print("\nExtracting features for Variant D ...")
X_D = build_matrix(df, kb=kb_pmi, use_cancellation=True)
f1_D = evaluate("D : + PMI KB, with cancellation (full model)", X_D)

print(f"  Cancellation rules contribution (C - B) : {f1_C - f1_B:+.4f}")
print(f"  PMI calibration contribution   (D - C) : {f1_D - f1_C:+.4f}")
print(f"  Total KB contribution          (D - A) : {f1_D - f1_A:+.4f}")