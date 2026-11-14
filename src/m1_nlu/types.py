from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


VALID_SERVICE_CATEGORIES = {
    "fitness",
    "wellness",
    "beauty",
    "medical",
    "dining",
    "travel",
    "other",
    "ambiguous",
}
VALID_URGENCY_LEVELS = {"low", "medium", "high"}


# M3 privacy memory must not write into STV; history adjusts M2 prior only.


@dataclass
class StructuredTaskVariable:
    service_category: str
    category_confidence: float
    temporal_preference: Optional[str]
    urgency_level: str
    time_window_days: Optional[int]
    location_context: Optional[str]
    sentiment_tags: List[str]
    extraction_confidence: float
    raw_input: str
    clarification_needed: bool = False


def normalize_stv(stv: StructuredTaskVariable) -> StructuredTaskVariable:
    # Preserve extractor-predicted category; vague inputs are handled by M2 multi-turn belief.
    # clarification_needed defaults to False; dialogue policy decides follow-up.
    if stv.service_category not in VALID_SERVICE_CATEGORIES:
        stv.service_category = "other"
    if stv.urgency_level not in VALID_URGENCY_LEVELS:
        stv.urgency_level = "medium"
    stv.category_confidence = min(1.0, max(0.0, stv.category_confidence))
    stv.extraction_confidence = min(1.0, max(0.0, stv.extraction_confidence))
    return stv
