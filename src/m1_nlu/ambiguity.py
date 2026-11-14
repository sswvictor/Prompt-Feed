from __future__ import annotations

import re
from typing import Dict


def ambiguity_bucket(text: str, row: Dict[str, object]) -> str:
    """Rule-based vagueness bucket for eval slices (no manual vague labels)."""
    text_l = text.lower()
    if row.get("service_category") == "ambiguous":
        return "high"
    uncertain_terms = ["not sure", "maybe", "don't know", "any", " or ", "i don't know"]
    conflict_patterns = [
        r"\bnow\b.*\bweekend\b",
        r"\bweekend\b.*\bnow\b",
        r"\burgent\b.*\bnot urgent\b",
        r"\bnot urgent\b.*\burgent\b",
    ]
    if any(term in text_l for term in uncertain_terms):
        return "high"
    if any(re.search(pattern, text_l) for pattern in conflict_patterns):
        return "high"

    explicit_slots = 0
    if row.get("temporal_preference") is not None:
        explicit_slots += 1
    if row.get("location_context") is not None:
        explicit_slots += 1
    if str(row.get("urgency_level", "medium")) != "medium":
        explicit_slots += 1
    if len(list(row.get("sentiment_tags", []))) > 0:
        explicit_slots += 1
    return "low" if explicit_slots >= 2 else "medium"
