"""
Stage 2 Full Pipeline: RAG + Few-Shot + Chain-of-Thought with Ministral-3 14B

Evaluation: 80/20 stratified split
Retrieval index built exclusively from the 80% train split => no leakage
"""

import re
import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

from knowledge_base import KnowledgeBase
from embedder import get_embeddings, embed

API_KEY  = ""
BASE_URL = ""
MODEL    = ""

K       = 4      # total examples: K/2 facts + K/2 opinions
MIN_SIM = 0.45   # fallback to zero-shot if best similarity below this treshhold

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = """You are an expert linguist specializing in distinguishing factual statements from opinions.
 
a FACT is a statement that:
- Can be verified against external sources
- References specific entities, numbers, dates, or events
- Uses neutral, objective language
- Attributes claims to named sources (e.g. "according to", "researchers found")
 
an OPINION is a statement that:
- Expresses a personal belief, judgment, or evaluation
- Uses evaluative or emotionally charged language (e.g. "extremely", "should", "believe")
- Makes normative claims about what ought to happen
- Cannot be objectively verified as true or false
 
Very important: if an evaluative word appears inside quotation marks or next to a number,
it may not signal opinion because the author could be reporting someone else's view.
 
Think step by step through the linguistic signals briefly (keep it under 3 sentences). Then on the very last line of your response, write your final answer as exactly one word, either Fact or Opinion. Do not write anything after the final answer."""
 
 
def balanced_retrieve(query_embedding, train_embeddings,
                      train_texts, train_labels, k=4):
    """Retrieve top k/2 facts and top k/2 opinions by cosine similarity."""
    sims = cosine_similarity(query_embedding.reshape(1, -1), train_embeddings)[0]
 
    fact_idx    = [i for i, l in enumerate(train_labels) if l == "fact"]
    opinion_idx = [i for i, l in enumerate(train_labels) if l == "opinion"]
 
    half         = k // 2
    top_facts    = sorted(fact_idx,    key=lambda i: sims[i], reverse=True)[:half]
    top_opinions = sorted(opinion_idx, key=lambda i: sims[i], reverse=True)[:half]
 
    examples = [
        {"text": train_texts[i], "label": train_labels[i], "similarity": float(sims[i])}
        for i in top_facts + top_opinions
    ]
    best_sim = max(e["similarity"] for e in examples) if examples else 0.0
    return examples, best_sim
 
 
def parse_label(reasoning: str) -> str:
    """
    Extract the final classification label from the model's CoT response
    """
    # Step 1 : explicit conclusion marker
    conclusion = re.search(
        r'(?:final answer|therefore|conclusion|classify as|my answer is|answer:)'
        r'[:\s]*\**\s*(Fact|Opinion)',
        reasoning, re.IGNORECASE
    )
    if conclusion:
        return conclusion.group(1).capitalize()
 
    # Step 2: scan for Fact/Opinion on its own line
    for line in reversed(reasoning.strip().split('\n')):
        m = re.match(r'^\s*\*?\*?\s*(Fact|Opinion)\s*\*?\*?\s*$', line, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
 
    # Step 3: search for last occurrence of Fact/Opinion anywhere
    matches = re.findall(r'\b(Fact|Opinion)\b', reasoning, re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
 
    return "Fact"  # absolute fallback
 
 
def build_prompt(text, retrieved, kb_signals, use_retrieval=True):
    kb_lines = []
    if kb_signals["kb_adj_opinion"] > 0:
        kb_lines.append(f"- Opinion signals detected (score: {kb_signals['kb_adj_opinion']:.2f})")
    if kb_signals["kb_adj_fact"] > 0:
        kb_lines.append(f"- Fact signals detected (score: {kb_signals['kb_adj_fact']:.2f})")
    if kb_signals["kb_cancel_delta"] > 0:
        kb_lines.append(f"- Some opinion signals cancelled by objectifying context (delta: {kb_signals['kb_cancel_delta']:.2f})")
    if kb_signals["kb_first_person"] > 0:
        kb_lines.append(f"- First-person pronouns present")
    kb_summary = "\n".join(kb_lines) if kb_lines else "No strong signals detected"
 
    if use_retrieval:
        sorted_ex = sorted(retrieved, key=lambda x: x["label"])
        examples  = "\n".join([
            f"  [{r['label'].upper()}]: \"{r['text'][:120]}\""
            for r in sorted_ex
        ])
        retrieval_block = f"Similar labeled sentences (2 Facts, 2 Opinions):\n{examples}\n\n"
    else:
        retrieval_block = "(No similar examples available... Classifying from linguistic signals only)\n\n"
 
    return f"""{retrieval_block}Linguistic signals detected:
{kb_summary}
 
Sentence to classify:
"{text}"
 
Think step by step, then write your final answer on the last line as a single word (Fact or Opinion)."""
 
 
def classify_rag(text, train_embeddings, train_texts, train_labels, kb):
    query_emb             = embed([text])[0]
    retrieved, best_sim   = balanced_retrieve(
        query_emb, train_embeddings, train_texts, train_labels, k=K
    )
    use_retrieval = best_sim >= MIN_SIM
    kb_signals    = kb.score(text)
    user_prompt   = build_prompt(text, retrieved, kb_signals, use_retrieval)
 
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
        max_tokens=512,
    )
    reasoning = response.choices[0].message.content.strip()
    label     = parse_label(reasoning)
    return label, reasoning, use_retrieval, best_sim
 
 
if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)
 
    train_df, test_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_df = train_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
 
    print(f"Stage 2 RAG (K={K}, balanced, MIN_SIM={MIN_SIM})")
    print(f"Train: {len(train_df)} | Test: {len(test_df)}\n")
 
    train_embeddings = get_embeddings(train_df["text"].tolist(), force_recompute=True)
    kb               = KnowledgeBase(weights_path="kb_weights.json")
 
    preds, trues, reasonings = [], [], []
    n_retrieval_used = 0
    similarities     = []
 
    for i, row in test_df.iterrows():
        if i % 20 == 0:
            print(f"  {i}/{len(test_df)}")
        pred, reasoning, used_retrieval, best_sim = classify_rag(
            row["text"], train_embeddings,
            train_df["text"].tolist(), train_df["label"].tolist(), kb
        )
        preds.append(pred.lower())
        trues.append(row["label"])
        reasonings.append(reasoning)
        similarities.append(best_sim)
        if used_retrieval:
            n_retrieval_used += 1
 
    accuracy = sum(p == t for p, t in zip(preds, trues)) / len(preds)
    f1 = f1_score(
        [1 if t == "opinion" else 0 for t in trues],
        [1 if p == "opinion" else 0 for p in preds],
        average="macro"
    )
 
    print(f"\nRAG Results (n={len(test_df)}, K={K}, balanced):")
    print(f"  Accuracy      : {accuracy:.4f}")
    print(f"  F1 macro      : {f1:.4f}")
    print(f"  Retrieval used: {n_retrieval_used}/{len(test_df)} "
          f"({n_retrieval_used/len(test_df)*100:.1f}%)")
    print(f"  Avg similarity: {np.mean(similarities):.3f}")
    print("\nClassification Report:")
    print(classification_report(trues, preds, target_names=["fact", "opinion"]))
 
    print("\n An Example of CoT Reasoning: ")
    print(f"Sentence: {test_df['text'].iloc[0][:120]}...")
    print(f"True label: {test_df['label'].iloc[0]}")
    print(f"Predicted:  {preds[0]}")
    print(f"Reasoning:\n{reasonings[0]}")
