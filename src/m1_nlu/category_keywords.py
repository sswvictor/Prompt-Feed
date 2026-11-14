from __future__ import annotations

import re
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from src.m1_nlu.types import StructuredTaskVariable

SERVICE_CATEGORIES = ("fitness", "wellness", "beauty", "medical", "dining", "travel")

CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "travel": ("travel", "flight", "hotel", "trip", "airport", "fly", "vacation"),
    "dining": (
        "eat",
        "eating",
        "hungry",
        "hunger",
        "starving",
        "restaurant",
        "food",
        "lunch",
        "dinner",
        "meal",
        "dining",
        "buffet",
        "brunch",
        "cuisine",
        "sushi",
        "thai",
        "pizza",
        "ramen",
        "noodles",
        "cafe",
        "bistro",
    ),
    "beauty": ("nail", "nails", "manicure", "skin", "salon", "beauty", "cosmetic"),
    "fitness": ("gym", "workout", "fitness", "yoga", "exercise", "class"),
    "wellness": ("spa", "massage", "relax", "unwind", "wellness", "calm", "retreat"),
    "medical": ("doctor", "clinic", "checkup", "medical", "health", "hospital", "appointment"),
}

# Extra multi-word cues (word-boundary safe via substring-on-normalized).
CATEGORY_PHRASES: Dict[str, Tuple[str, ...]] = {
    "medical": ("health check", "routine checkup", "doctor appointment", "see a doctor"),
    "wellness": ("wellness retreat", "spa day"),
    "fitness": ("gym session", "personal training"),
    "dining": ("book a table", "make a reservation"),
    "travel": ("book a flight", "hotel booking"),
    "beauty": ("nail appointment", "hair salon"),
}

# Strong medical lexical cues — may override conflicting M1 labels.
MEDICAL_STRONG_CUES = frozenset(
    {"doctor", "clinic", "checkup", "hospital", "appointment", "health check", "routine checkup"}
)

CATEGORY_SYNONYMS: Dict[str, str] = {
    "travel": "trip or flight",
    "dining": "food or restaurant",
    "beauty": "beauty or nail care",
    "fitness": "workout or gym",
    "wellness": "spa or relaxation",
    "medical": "clinic or checkup",
}

_NEGATION_WINDOW = re.compile(
    r"\b(?:not|no|never|without)\s+(?:a\s+|an\s+|the\s+)?([a-z]+(?:\s+[a-z]+)?)",
    re.I,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _word_hit(lower: str, word: str) -> bool:
    """Whole-word match only (avoids eat⊂retreat)."""
    return re.search(rf"\b{re.escape(word)}\b", lower) is not None


def _negated_terms(lower: str) -> set[str]:
    terms: set[str] = set()
    for m in _NEGATION_WINDOW.finditer(lower):
        phrase = m.group(1).strip()
        terms.add(phrase)
        for tok in phrase.split():
            terms.add(tok)
    return terms


def score_categories(text: str) -> Dict[str, float]:
    """Lexical category scores with phrase bonuses, strong medical weight, negation."""
    lower = _normalize(text)
    negated = _negated_terms(lower)
    scores: Dict[str, float] = {}

    for cat, words in CATEGORY_KEYWORDS.items():
        score = 0.0
        for w in words:
            if not _word_hit(lower, w):
                continue
            if w in negated or any(w in n for n in negated):
                score -= 1.5
                continue
            weight = 2.0 if (cat == "medical" and w in MEDICAL_STRONG_CUES) else 1.0
            score += weight
        for phrase in CATEGORY_PHRASES.get(cat, ()):
            if phrase in lower:
                if any(tok in negated for tok in phrase.split()):
                    score -= 1.5
                else:
                    bonus = 2.5 if cat == "medical" else 1.5
                    if phrase in MEDICAL_STRONG_CUES:
                        bonus = 3.0
                    score += bonus
        if score > 0:
            scores[cat] = score
        elif score < 0 and cat in scores:
            del scores[cat]
    # Drop non-positive after negation-only hits
    return {c: s for c, s in scores.items() if s > 0}


def detect_categories_from_text(text: str) -> List[str]:
    scores = score_categories(text)
    return [c for c, _ in sorted(scores.items(), key=lambda x: (-x[1], x[0]))]


def detect_primary_category(text: str) -> Optional[str]:
    scores = score_categories(text)
    if not scores:
        return None
    return max(scores.items(), key=lambda x: (x[1], x[0]))[0]


def supports_category(text: str, category: str) -> bool:
    if not category:
        return False
    return category in score_categories(text)


def extract_user_keywords(text: str, category: Optional[str] = None) -> List[str]:
    lower = _normalize(text)
    found: List[str] = []
    pools = [CATEGORY_KEYWORDS[category]] if category and category in CATEGORY_KEYWORDS else []
    for words in pools or CATEGORY_KEYWORDS.values():
        for w in words:
            if _word_hit(lower, w) and w not in found:
                found.append(w)
    if not found:
        tokens = [t.strip(",.?!") for t in lower.split() if len(t) > 2]
        found.extend(tokens[:3])
    return found[:5]


def apply_domain_guards(
    stv: StructuredTaskVariable,
    text: str,
    *,
    enabled: bool = True,
    conf_override_max: float = 0.85,
) -> StructuredTaskVariable:
    """Correct M1 category when strong lexical evidence disagrees (S2)."""
    if not enabled:
        return stv

    scores = score_categories(text)
    if not scores:
        return stv

    winner, win_score = max(scores.items(), key=lambda x: x[1])
    medical_strong = winner == "medical" and win_score >= 2.0
    ranked = sorted(scores.values(), reverse=True)
    or_ambiguous = (
        " or " in _normalize(text)
        and len(scores) >= 2
        and (ranked[0] - ranked[1]) < 0.75
        and not medical_strong
    )

    if or_ambiguous:
        # Keep M1 if it already chose one of the lexical candidates; only soft-cap conf.
        if stv.service_category in scores:
            capped = min(max(stv.category_confidence, 0.60), 0.75)
            return replace(stv, category_confidence=capped, extraction_confidence=capped)
        primary = max(scores.items(), key=lambda x: x[1])[0]
        return replace(
            stv,
            service_category=primary,
            category_confidence=0.60,
            extraction_confidence=0.60,
        )

    if winner == stv.service_category:
        if win_score >= 2.0:
            boosted = max(stv.category_confidence, min(0.92, stv.category_confidence + 0.08))
            return replace(stv, category_confidence=boosted, extraction_confidence=boosted)
        return stv

    should_override = medical_strong or stv.category_confidence < conf_override_max or win_score >= 2.0
    if not should_override:
        return stv

    new_conf = max(stv.category_confidence, min(0.93, 0.55 + 0.12 * win_score))
    return replace(
        stv,
        service_category=winner,
        category_confidence=new_conf,
        extraction_confidence=new_conf,
    )


def vote_category_from_history(
    history: Sequence[StructuredTaskVariable],
    current: StructuredTaskVariable,
    *,
    enabled: bool = True,
    decay: float = 0.75,
    min_share: float = 0.55,
) -> StructuredTaskVariable:
    """Weighted multi-turn M1 vote so a bad last turn cannot erase earlier evidence (S2)."""
    if not enabled or not history:
        return current

    hist_w: Dict[str, float] = {}
    n = len(history)
    for i, past in enumerate(history):
        cat = past.service_category
        if cat in {"ambiguous", "other"}:
            continue
        age = n - i
        hist_w[cat] = hist_w.get(cat, 0.0) + float(past.category_confidence) * (decay**age)

    # Do not erase a current label that has on-turn lexical support.
    current_scores = score_categories(current.raw_input or "")
    current_lexically_supported = current.service_category in current_scores

    # History consensus can override a weak disagreeing current label.
    if hist_w and not current_lexically_supported:
        h_winner, h_win = max(hist_w.items(), key=lambda x: x[1])
        h_share = h_win / (sum(hist_w.values()) or 1.0)
        if (
            h_winner != current.service_category
            and h_share >= 0.5
            and current.category_confidence < 0.8
        ):
            new_conf = max(current.category_confidence, min(0.9, h_share))
            return replace(
                current,
                service_category=h_winner,
                category_confidence=new_conf,
                extraction_confidence=new_conf,
            )

    weights = dict(hist_w)
    if current.service_category not in {"ambiguous", "other"}:
        weights[current.service_category] = (
            weights.get(current.service_category, 0.0) + float(current.category_confidence)
        )
    if not weights:
        return current

    winner, win_w = max(weights.items(), key=lambda x: x[1])
    share = win_w / (sum(weights.values()) or 1.0)
    if (
        winner != current.service_category
        and share >= min_share
        and not current_lexically_supported
    ):
        new_conf = max(current.category_confidence, min(0.9, share))
        return replace(
            current,
            service_category=winner,
            category_confidence=new_conf,
            extraction_confidence=new_conf,
        )
    return current
