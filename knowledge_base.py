"""
Defines the lexicon and KnowledgeBase class for Stage 1.
"""

import re
import json
import math
from pathlib import Path
from nltk import word_tokenize

# ── Lexicon ───────────────────────────────────────────────────────────────────

LEXICON = {
    "epistemic_verbs": [
        "think", "believe", "feel", "suppose", "suspect", "doubt",
        "assume", "reckon", "guess", "imagine", "expect", "hope",
        "wonder", "seem", "appear", "suggest",
    ],
    "evaluative_adjectives": [
        "good", "bad", "great", "terrible", "excellent", "poor",
        "wonderful", "awful", "amazing", "horrible", "fantastic",
        "outstanding", "mediocre", "brilliant", "stupid", "important",
        "dangerous", "useful", "useless", "effective", "misleading",
        "biased", "fair", "unfair", "reasonable", "unreasonable",
        "absurd", "crucial", "beneficial", "harmful",
    ],
    "hedging_adverbs": [
        "probably", "possibly", "perhaps", "maybe", "likely", "unlikely",
        "apparently", "presumably", "arguably", "seemingly", "supposedly",
        "allegedly", "conceivably", "potentially",
    ],
    "deontic_modals": [
        "should", "must", "ought", "shall",
    ],
    "intensifiers": [
        "very", "extremely", "incredibly", "absolutely", "totally",
        "completely", "utterly", "highly", "deeply", "truly",
        "remarkably", "exceptionally", "immensely", "profoundly",
    ],
    "attribution_verbs": [
        "found", "showed", "demonstrated", "measured", "recorded",
        "reported", "published", "announced", "stated", "confirmed",
        "revealed", "indicated", "noted", "observed", "documented",
        "established", "determined", "calculated", "estimated",
        "concluded", "discovered", "identified", "verified",
    ],
    "quantitative_markers": [
        "percent", "million", "billion", "trillion", "thousand",
        "data", "study", "research", "survey", "statistics",
        "evidence", "result", "analysis", "experiment",
        "measurement", "rate", "ratio",
    ],
    "citation_markers": [
        "according to", "based on", "as reported by",
        "as stated by", "as found by", "as per",
    ],
}

_QUOTE_RE       = re.compile(r'["\u201c\u201d]([^"]+)["\u201c\u201d]')
_NUMBER_RE      = re.compile(r'\b\d+([.,]\d+)?\s*%?\b')
_ATTRIBUTION_RE = re.compile(
    r'\b(according to|based on|found that|reported that|stated that|confirmed that)\b',
    re.IGNORECASE
)


class KnowledgeBase:
    """
    Scores a text against the lexicon.
    Optionally loads PMI weights from kb_weights.json
    Without weights, all terms count as 1
    """

    def __init__(self, weights_path="kb_weights.json"):
        self.weights = {}
        if weights_path is not None and Path(weights_path).exists():
            self.weights = json.load(open(weights_path))

    def _w(self, term, polarity):
        """Return PMI weight for a term, defaulting to 1.0"""
        return self.weights.get(polarity, {}).get(term, 1.0)

    def _cancelled_spans(self, text):
        """Character spans where opinion signals should be ignored"""
        spans = []
        for m in _QUOTE_RE.finditer(text):
            spans.append((m.start(), m.end()))
        for m in _NUMBER_RE.finditer(text):
            spans.append((max(0, m.start() - 30), m.end() + 30))
        for m in _ATTRIBUTION_RE.finditer(text):
            spans.append((m.start(), m.end() + 60))
        return spans

    def score(self, text: str, use_cancellation: bool = True):
        """Return a flat dict of KB-derived features for a text"""
        text_lower = text.lower()
        tokens     = word_tokenize(text_lower)
        total      = max(len(tokens), 1)

        opinion_set = set(
            t for cat in ["epistemic_verbs", "evaluative_adjectives",
                          "hedging_adverbs", "deontic_modals", "intensifiers"]
            for t in LEXICON[cat]
        )
        fact_set = set(
            t for cat in ["attribution_verbs", "quantitative_markers"]
            for t in LEXICON[cat]
        )

        # Raw scores
        raw_opinion = sum(self._w(t, "opinion") for t in tokens if t in opinion_set)
        raw_fact    = sum(self._w(t, "fact")    for t in tokens if t in fact_set)
        for phrase in LEXICON["citation_markers"]:
            if phrase in text_lower:
                raw_fact += self._w(phrase, "fact")

        # Cancellation-adjusted scores
        spans = self._cancelled_spans(text_lower) if use_cancellation else []
        cancelled = lambda pos: any(s <= pos <= e for s, e in spans)
        adj_opinion = sum(
            self._w(m.group(), "opinion")
            for m in re.finditer(r'\b\w+\b', text_lower)
            if m.group() in opinion_set and not cancelled(m.start())
        )
        adj_fact = raw_fact  # fact signals are not cancelled

        return {
            "kb_raw_opinion":       raw_opinion / total,
            "kb_raw_fact":          raw_fact    / total,
            "kb_raw_net":           (raw_opinion - raw_fact) / total,
            "kb_adj_opinion":       adj_opinion / total,
            "kb_adj_fact":          adj_fact    / total,
            "kb_adj_net":           (adj_opinion - adj_fact) / total,
            "kb_cancel_delta":      (raw_opinion - adj_opinion) / total,
            "kb_first_person":      sum(1 for t in tokens if t in {"i","me","my","we","our","us"}) / total,
            "kb_question_marks":    float(text.count("?")),
        }


if __name__ == "__main__":
    kb = KnowledgeBase()
    tests = {
        "FACT":    'According to a 2023 study, researchers found that 73% of participants improved.',
        "OPINION": 'I think this is an extremely dangerous approach that should be reconsidered.',
        "MIXED":   'The report stated the policy was "incredibly harmful", though officials confirmed a 12% improvement.',
    }
    for label, text in tests.items():
        s = kb.score(text)
        print(f"[{label}]  raw_net={s['kb_raw_net']:+.3f}  adj_net={s['kb_adj_net']:+.3f}  delta={s['kb_cancel_delta']:.3f}")