"""
Given a query sentence and precomputed training embeddings,
returns the top-K most similar training examples with their labels.
Similarity metric: cosine similarity
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query_embedding: np.ndarray,
             train_embeddings: np.ndarray,
             train_texts: list[str],
             train_labels: list[str],
             k: int = 3) -> list[dict]:
    """
    Retrieve the top-K most similar training examples for a query

    Input
    ----------
    query_embedding  : 1D array of shape (dim,)
    train_embeddings : 2D array of shape (n_train, dim)
    train_texts      : list of training sentences
    train_labels     : list of training labels ('fact' or 'opinion')
    k                : number of examples to retrieve

    Returns
    -------
    List of dicts with keys: text, label, similarity
    Ordered by descending similarity
    """
    sims     = cosine_similarity(query_embedding.reshape(1, -1), train_embeddings)[0]
    top_idx  = np.argsort(sims)[::-1][:k]

    return [
        {
            "text":       train_texts[i],
            "label":      train_labels[i],
            "similarity": float(sims[i]),
        }
        for i in top_idx
    ]


if __name__ == "__main__":
    import pandas as pd
    from embedder import get_embeddings, embed

    df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)

    train_embeddings = get_embeddings(df["text"].tolist())
    query = "I think this policy is extremely dangerous and should be reconsidered."
    query_embedding  = embed([query])[0]

    results = retrieve(query_embedding, train_embeddings,
                       df["text"].tolist(), df["label"].tolist(), k=3)

    print(f"Query: {query}\n")
    print("Top-3 retrieved examples:")
    for i, r in enumerate(results):
        print(f"  [{i+1}] [{r['label'].upper()}] (sim={r['similarity']:.3f})")
        print(f"       {r['text'][:100]}...")