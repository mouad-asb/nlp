"""
Runs the Stage 2 RAG pipeline on the validation set
Saves group54_classifications_2.csv
"""

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from openai import OpenAI

from knowledge_base import KnowledgeBase
from retriever import retrieve
from embedder import get_embeddings, embed
from s2_rag import classify_rag, SYSTEM_PROMPT

API_KEY  = "sk-llmduckt-bfd23b838cc588fbf11aca79dfef6faaf3be48e3490d3cc1443f44d020292b27"
BASE_URL = "https://flaait.tugraz.at/llmduckt/llmapi/v1/"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# Load training data 
df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
df["label"] = df["label"].str.strip().str.lower()
df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)

train, _ = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
train = train.reset_index(drop=True)

# Load validation data
val = pd.read_csv("validationset.csv", header=0, names=["text"])
val["text"] = val["text"].astype(str).str.strip()

print(f"Train: {len(train)} | Val: {len(val)}")

# Precompute embeddings + load KB 
train_embeddings = get_embeddings(train["text"].tolist())
kb               = KnowledgeBase(weights_path="kb_weights.json")

# Classify validation set
preds = []
print("Classifying validation set...")
for i, row in val.iterrows():
    if i % 20 == 0:
        print(f"  {i}/{len(val)}")
    pred, reasoning, used_retrieval, best_sim = classify_rag(
        row["text"],
        train_embeddings,
        train["text"].tolist(),
        train["label"].tolist(),
        kb
    )
    preds.append(pred.lower())


out = pd.DataFrame({
    "Content":    val["text"],
    "Prediction": preds,
})
out.to_csv("groupXX_classifications_2.csv", index=False)

n_fact    = preds.count("fact")
n_opinion = preds.count("opinion")
print(f"\nSaved group54_classifications_2.csv")
print(f"  Fact: {n_fact} ({n_fact/len(preds)*100:.1f}%)")
print(f"  Opinion: {n_opinion} ({n_opinion/len(preds)*100:.1f}%)")