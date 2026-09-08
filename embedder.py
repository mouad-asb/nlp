"""
Loads a sentence transformer model and embeds a list of sentences.
Saves embeddings locally to avoid recomputing on every run.
"""

import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME   = "all-MiniLM-L6-v2"
EMBEDDINGS_PATH = Path("train_embeddings.npy")


def get_embeddings(sentences: list[str], force_recompute: bool = False) -> np.ndarray:
    """
    Returns embeddings for a list of sentences.
    Loads already existing embeddings if available, otherwise computes and saves.
    """
    if EMBEDDINGS_PATH.exists() and not force_recompute:
        print(f"Loading embeddings from {EMBEDDINGS_PATH}")
        return np.load(EMBEDDINGS_PATH)

    print(f"Computing embeddings for {len(sentences)} sentences...")
    model      = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(sentences, show_progress_bar=True,
                              convert_to_numpy=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_PATH} — shape: {embeddings.shape}")
    return embeddings


def embed(sentences: list[str]) -> np.ndarray:
    """Embed a list of sentences without caching (for query sentences)."""
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(sentences, convert_to_numpy=True)


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)

    embeddings = get_embeddings(df["text"].tolist(), force_recompute=True)
    print(f"Done — {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")