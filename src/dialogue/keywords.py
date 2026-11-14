from __future__ import annotations

from typing import List, Optional

from src.m1_nlu.category_keywords import (
    CATEGORY_KEYWORDS,
    CATEGORY_SYNONYMS,
    SERVICE_CATEGORIES,
    detect_categories_from_text,
    detect_primary_category,
    extract_user_keywords,
    supports_category,
)

YES_REPLIES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "correct",
        "right",
        "对",
        "是的",
        "好",
        "可以",
    }
)

NO_REPLIES = frozenset({"no", "nope", "not", "nah", "不", "不是"})

SHORT_FOLLOWUP_MAX_TOKENS = 4


def _normalize(text: str) -> str:
    import re

    return re.sub(r"\s+", " ", text.lower().strip())


def is_confirmation(text: str) -> bool:
    token = _normalize(text).rstrip("!.,")
    return token in YES_REPLIES


def is_rejection(text: str) -> bool:
    token = _normalize(text).rstrip("!.,")
    return token in NO_REPLIES


def is_short_followup(text: str) -> bool:
    lower = _normalize(text)
    if not lower:
        return False
    if is_confirmation(lower) or is_rejection(lower):
        return True
    return len(lower.split()) <= SHORT_FOLLOWUP_MAX_TOKENS
