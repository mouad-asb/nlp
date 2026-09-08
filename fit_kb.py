"""
Computes PMI weights for each lexicon term against the training data
Saves results to kb_weights.json.

PMI(term, class) = log[ P(term|class) / P(term) ]
Applying add-k smoothing to prevent log(0) for unseen terms
min_count=5 to filter terms too rare to trust
"""

import json, math, re
from collections import defaultdict
import pandas as pd
from nltk import word_tokenize
import nltk
from knowledge_base import LEXICON

for r in ["punkt", "punkt_tab"]:
    nltk.download(r, quiet=True)

df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)
print(f"Loaded {len(df)} samples — fact: {(df['label']=='fact').sum()}, opinion: {(df['label']=='opinion').sum()}")

# Count term occurences per class
terms = list({t.lower() for terms in LEXICON.values() for t in terms})
n     = len(df)
n_cls = df["label"].value_counts().to_dict()

counts = defaultdict(lambda: defaultdict(int))  # counts[term][class]
total  = defaultdict(int)                        # total[term]

for _, row in df.iterrows():
    text_lower = row["text"].lower()
    tokens     = set(word_tokenize(text_lower))
    for term in terms:
        hit = (term in tokens) if " " not in term else (term in text_lower)
        if hit:
            counts[term][row["label"]] += 1
            total[term] += 1

# Compute PMI
SMOOTHING  = 0.1
MIN_COUNT  = 5
MIN_PMI    = 0.1
weights    = {"opinion": {}, "fact": {}}

for term in terms:
    if total[term] < MIN_COUNT:
        continue
    p_term = total[term] / n
    for cls in ["opinion", "fact"]:
        p_term_given_cls = (counts[term][cls] + SMOOTHING) / (n_cls[cls] + SMOOTHING * 2)
        pmi = math.log(p_term_given_cls / p_term)
        if abs(pmi) >= MIN_PMI:
            weights[cls][term] = round(pmi, 4)

# Save
json.dump(weights, open("kb_weights.json", "w"), indent=2)
print(f"Saved kb_weights.json — opinion terms: {len(weights['opinion'])}, fact terms: {len(weights['fact'])}")

for cls in ["opinion", "fact"]:
    top = sorted(weights[cls].items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"  Top 5 → {cls}: {top}")