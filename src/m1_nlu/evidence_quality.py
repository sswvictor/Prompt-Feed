"""Evidence-quality controls that do NOT rewrite service_category via keywords.

- Cross-turn category conflict → dampen current confidence (belief mixes; policy may clarify)
- Optional multi-turn STV vote remains in category_keywords.vote_category_from_history
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from src.m1_nlu.types import StructuredTaskVariable


def dampen_conflicting_turn(
    history: Sequence[StructuredTaskVariable],
    current: StructuredTaskVariable,
    *,
    enabled: bool = True,
    factor: float = 0.65,
    min_prev_conf: float = 0.45,
) -> StructuredTaskVariable:
    """If the new label disagrees with the previous turn, soften confidence.

    Does not change service_category. Keeps v2 hypothesis/belief machinery intact
    while reducing over-trust in a sudden noisy M1 flip.
    """
    if not enabled or not history:
        return current
    cat = current.service_category
    if cat in {"ambiguous", "other"}:
        return current
    prev = history[-1]
    if prev.service_category in {"ambiguous", "other"}:
        return current
    if prev.service_category == cat:
        return current
    if float(prev.category_confidence) < min_prev_conf:
        return current
    new_conf = max(0.15, float(current.category_confidence) * factor)
    return replace(
        current,
        category_confidence=new_conf,
        extraction_confidence=min(float(current.extraction_confidence), new_conf),
    )
