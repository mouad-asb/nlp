"""
Stage 2 Baseline: Zero-shot classification with Ministral-3 14B.

Evaluation: 80/20 stratified split
The 80% train split is used to build the RAG retrieval index in stage2_rag.py
The 20% held-out split is used to evaluate both baseline and RAG approach
"""

import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

API_KEY  = "sk-llmduckt-bfd23b838cc588fbf11aca79dfef6faaf3be48e3490d3cc1443f44d020292b27"
BASE_URL = "https://flaait.tugraz.at/llmduckt/llmapi/v1/"
MODEL    = "tugrazflaait/ministral-3:14b"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert linguist specializing in distinguishing factual statements from opinions.

a FACT is a statement that:
- Can be verified against external sources
- References specific entities, numbers, dates, or events
- Uses neutral, objective language
- Attributes claims to named sources

an OPINION is a statement that:
- Expresses a personal belief, judgment, or evaluation
- Uses evaluative or emotionally charged language
- Makes normative claims (what should or must happen)
- Cannot be objectively verified as true or false

Respond with exactly one word: either Fact or Opinion."""


def classify_zero_shot(text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f'Classify this sentence:\n\n"{text}"'},
        ],
        temperature=0,
        max_tokens=10,
    )
    raw = response.choices[0].message.content.strip().lower()
    return "opinion" if "opinion" in raw else "fact"


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)

    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    test_df = test_df.reset_index(drop=True)

    print(f"Stage 2 Baseline")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}\n")

    preds, trues = [], []
    for i, row in test_df.iterrows():
        if i % 20 == 0:
            print(f"  {i}/{len(test_df)}")
        pred = classify_zero_shot(row["text"])
        preds.append(pred)
        trues.append(row["label"])

    accuracy = sum(p == t for p, t in zip(preds, trues)) / len(preds)
    f1 = f1_score(
        [1 if t == "opinion" else 0 for t in trues],
        [1 if p == "opinion" else 0 for p in preds],
        average="macro"
    )

    print(f"\nZero-Shot Results (n={len(test_df)}, held-out 20%):")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1 macro : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=["fact", "opinion"]))