from __future__ import annotations

from src.m1_nlu.category_keywords import (
    apply_domain_guards,
    detect_primary_category,
    score_categories,
    vote_category_from_history,
)
from src.m1_nlu.types import StructuredTaskVariable


def _stv(category: str, confidence: float, text: str = "x") -> StructuredTaskVariable:
    return StructuredTaskVariable(
        service_category=category,
        category_confidence=confidence,
        temporal_preference=None,
        urgency_level="medium",
        time_window_days=None,
        location_context=None,
        sentiment_tags=[],
        extraction_confidence=confidence,
        raw_input=text,
        clarification_needed=False,
    )


def test_retreat_does_not_trigger_dining() -> None:
    scores = score_categories("Health check or wellness retreat, I'm not sure")
    assert "dining" not in scores
    assert "wellness" in scores
    assert "medical" in scores


def test_clinic_checkup_primary_medical() -> None:
    assert detect_primary_category("Any clinic nearby tomorrow morning for checkup") == "medical"


def test_not_spa_prefers_medical() -> None:
    text = "Doctor appointment for a routine checkup, not spa"
    assert detect_primary_category(text) == "medical"
    guarded = apply_domain_guards(_stv("wellness", 0.89, text), text)
    assert guarded.service_category == "medical"


def test_guard_overrides_fitness_on_clinic() -> None:
    text = "Any clinic nearby tomorrow morning for checkup"
    guarded = apply_domain_guards(_stv("fitness", 0.54, text), text)
    assert guarded.service_category == "medical"


def test_nail_still_beauty() -> None:
    assert detect_primary_category("fix my nail") == "beauty"


def test_cuisine_names_map_to_dining() -> None:
    text = "Maybe Thai or sushi, tomorrow afternoon"
    assert detect_primary_category(text) == "dining"
    guarded = apply_domain_guards(_stv("wellness", 0.88, text), text)
    assert guarded.service_category == "dining"


def test_vote_recovers_from_bad_last_turn() -> None:
    history = [
        _stv("wellness", 0.7, "something vague"),
        _stv("medical", 0.9, "doctor checkup please"),
    ]
    # No lexical fitness cues — history medical may override.
    current = _stv("fitness", 0.55, "any center nearby tomorrow")
    voted = vote_category_from_history(history, current)
    assert voted.service_category == "medical"


def test_vote_respects_lexical_current_turn() -> None:
    history = [_stv("wellness", 0.7, "vague"), _stv("wellness", 0.5, "vague2")]
    current = _stv("fitness", 0.62, "Fitness or wellness, quick session")
    voted = vote_category_from_history(history, current)
    assert voted.service_category == "fitness"
