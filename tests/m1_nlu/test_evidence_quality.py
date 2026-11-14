from __future__ import annotations

from src.m1_nlu.evidence_quality import dampen_conflicting_turn
from src.m1_nlu.types import StructuredTaskVariable


def _stv(cat: str, conf: float) -> StructuredTaskVariable:
    return StructuredTaskVariable(
        service_category=cat,
        category_confidence=conf,
        temporal_preference=None,
        urgency_level="medium",
        time_window_days=None,
        location_context=None,
        sentiment_tags=[],
        extraction_confidence=conf,
        raw_input="x",
        clarification_needed=False,
    )


def test_conflict_dampens_confidence_not_label() -> None:
    hist = [_stv("medical", 0.9)]
    cur = _stv("fitness", 0.8)
    out = dampen_conflicting_turn(hist, cur, factor=0.65)
    assert out.service_category == "fitness"
    assert out.category_confidence < cur.category_confidence
    assert abs(out.category_confidence - 0.8 * 0.65) < 1e-9


def test_same_category_unchanged() -> None:
    hist = [_stv("dining", 0.7)]
    cur = _stv("dining", 0.85)
    out = dampen_conflicting_turn(hist, cur)
    assert out.category_confidence == 0.85
