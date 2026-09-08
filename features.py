"""
Extracts all features and TF-IDF for Stage 1

Feature groups:
  - POS ratios: adjective, adverb, proper noun, verb ratios
  - Sentiment: TextBlob polarity/subjectivity, VADER compound
  - KB features: scores from KnowledgeBase.score(), binary or PMI-weighted
  - TF-IDF: binary unigrams + bigrams, top 300
"""

import re
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix

from knowledge_base import KnowledgeBase

for r in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
          "averaged_perceptron_tagger_eng"]:
    nltk.download(r, quiet=True)

VADER = SentimentIntensityAnalyzer()


# ── Individual feature extractors ─────────────────────────────────────────────

def pos_features(text):
    tokens = [t for t in word_tokenize(text.lower()) if t.isalpha()]
    total  = max(len(tokens), 1)
    tags   = pos_tag(tokens)
    counts = {}
    for _, tag in tags:
        counts[tag] = counts.get(tag, 0) + 1

    return {
        "pos_adj_ratio":  sum(counts.get(t, 0) for t in ["JJ","JJR","JJS"]) / total,
        "pos_adv_ratio":  sum(counts.get(t, 0) for t in ["RB","RBR","RBS"]) / total,
        "pos_prop_ratio": sum(counts.get(t, 0) for t in ["NNP","NNPS"])     / total,
        "pos_verb_ratio": sum(counts.get(t, 0) for t in ["VB","VBD","VBG","VBN","VBP","VBZ"]) / total,
    }


def sentiment_features(text):
    blob  = TextBlob(text)
    vader = VADER.polarity_scores(text)
    return {
        "sent_polarity":    blob.sentiment.polarity,
        "sent_subjectivity": blob.sentiment.subjectivity,
        "vader_compound":   vader["compound"],
        "vader_abs":        abs(vader["compound"]),
    }


def extract_all(texts, kb, tfidf=None, scaler=None, fit=False):
    """
    Build the full feature matrix for a list of texts.
    If fit=True: fits the TF-IDF and scaler on this data (training set only).
    If fit=False: transforms using pre-fitted objects (validation/test set).
    Returns X (sparse matrix), tfidf, scaler.
    """
    print("Extracting features...")
    records = []
    for i, text in enumerate(texts):
        if i % 200 == 0:
            print(f"  {i}/{len(texts)}")
        row = {}
        row.update(pos_features(text))
        row.update(sentiment_features(text))
        row.update(kb.score(text))
        records.append(row)
    print(f"  {len(texts)}/{len(texts)} done.")

    dense_df = pd.DataFrame(records).fillna(0)

    if fit:
        scaler       = StandardScaler()
        dense_scaled = scaler.fit_transform(dense_df.values)
        tfidf        = TfidfVectorizer(max_features=300, binary=True,
                                       ngram_range=(1, 2), min_df=3,
                                       stop_words="english")
        tfidf_mat    = tfidf.fit_transform(texts)
    else:
        dense_scaled = scaler.transform(dense_df.values)
        tfidf_mat    = tfidf.transform(texts)

    X = hstack([csr_matrix(dense_scaled), tfidf_mat])
    return X, tfidf, scaler


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv", header=0, names=["text", "label"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["fact", "opinion"])].reset_index(drop=True)

    kb = KnowledgeBase()
    X, tfidf, scaler = extract_all(df["text"], kb, fit=True)
    print(f"\nFeature matrix: {X.shape[0]} samples x {X.shape[1]} features")
    print(f"  Hand-crafted: {len(pd.DataFrame([pos_features('test') | sentiment_features('test') | kb.score('test')]).columns)}")
    print(f"  TF-IDF: {len(tfidf.get_feature_names_out())}")