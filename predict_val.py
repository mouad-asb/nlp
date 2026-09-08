"""
Trains the best Stage 1 model on full training data,
then predicts on the validation set and saves the submission CSV.
"""

import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

from knowledge_base import KnowledgeBase
from features import extract_all

def clean_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text

train = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
train["label"] = train["label"].str.strip().str.lower()
train = train[train["label"].isin(["fact", "opinion"])].reset_index(drop=True)
train["text"]   = train["text"].apply(clean_text)
train["target"] = (train["label"] == "opinion").astype(int)

# Val data
val = pd.read_csv("validationset.csv", header=0, names=["text"])
val["text"] = val["text"].apply(clean_text)

print(f"Train: {len(train)} samples | Val: {len(val)} samples")

kb = KnowledgeBase(weights_path="kb_weights.json")
X_train, tfidf, scaler = extract_all(train["text"], kb, fit=True)
y_train = train["target"].values


cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = cross_validate(
    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    X_train, y_train, cv=cv,
    scoring=["accuracy", "f1_macro"]
)
print(f"\nCV on training set (PMI-weighted KB + RF):")
print(f"  Accuracy : {res['test_accuracy'].mean():.4f} ± {res['test_accuracy'].std():.4f}")
print(f"  F1 macro : {res['test_f1_macro'].mean():.4f} ± {res['test_f1_macro'].std():.4f}")

# Train on full training set
model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Extract validation features
print("\nExtracting validation features...")
X_val, _, _ = extract_all(val["text"], kb, tfidf=tfidf, scaler=scaler, fit=False)

# Predict
preds = model.predict(X_val)
pred_labels = ["Opinion" if p == 1 else "Fact" for p in preds]

# Prediction distribution
n_fact    = pred_labels.count("Fact")
n_opinion = pred_labels.count("Opinion")
print(f"\nPrediction distribution on validation set:")
print(f"  Fact:    {n_fact}  ({n_fact/len(pred_labels)*100:.1f}%)")
print(f"  Opinion: {n_opinion}  ({n_opinion/len(pred_labels)*100:.1f}%)")

# Somes examples
print("\nSample predictions:")
for i in [0, 1, 2, 4, 8]:
    print(f"  [{pred_labels[i]:>7}] {val['text'].iloc[i][:90]}...")

out = pd.DataFrame({
    "Content":    val["text"],
    "Prediction": pred_labels,
})
out.to_csv("group54_classifications_1.csv", index=False)
print(f"\nSaved")