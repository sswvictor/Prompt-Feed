from __future__ import annotations

from typing import Any

_analyzer: Any = None


def _get_analyzer() -> Any:
    global _analyzer
    if _analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_sentiment(text: str) -> float:
    """Return VADER compound score in [-1, 1]."""
    if not text or not text.strip():
        return 0.0
    try:
        scores = _get_analyzer().polarity_scores(text)
        return float(scores.get("compound", 0.0))
    except ImportError:
        return _fallback_compound(text)


def is_negative_turn(text: str, threshold: float = -0.3) -> bool:
    return score_sentiment(text) <= threshold


def _fallback_compound(text: str) -> float:
    """Lightweight lexicon fallback when vaderSentiment is unavailable."""
    negative = {
        "bad",
        "terrible",
        "awful",
        "hate",
        "wrong",
        "frustrated",
        "annoyed",
        "useless",
        "disappointed",
        "angry",
        "no",
        "not",
    }
    positive = {
        "good",
        "great",
        "thanks",
        "perfect",
        "love",
        "nice",
        "helpful",
        "yes",
    }
    tokens = {t.strip(".,!?").lower() for t in text.split()}
    neg_hits = len(tokens & negative)
    pos_hits = len(tokens & positive)
    if neg_hits == pos_hits == 0:
        return 0.0
    raw = (pos_hits - neg_hits) / max(neg_hits + pos_hits, 1)
    return max(-1.0, min(1.0, raw))


def reset_analyzer() -> None:
    """Test helper: drop cached VADER instance."""
    global _analyzer
    _analyzer = None
