from __future__ import annotations

from src.m1_nlu.ambiguity import ambiguity_bucket


def test_ambiguity_bucket_high_uncertainty() -> None:
    row = {"service_category": "wellness", "urgency_level": "medium", "sentiment_tags": []}
    assert ambiguity_bucket("I am not sure, maybe spa or gym", row) == "high"


def test_ambiguity_bucket_high_conflict() -> None:
    row = {"service_category": "fitness", "urgency_level": "high", "sentiment_tags": ["quick"]}
    assert ambiguity_bucket("Need it now but also this weekend", row) == "high"


def test_ambiguity_bucket_low_explicit() -> None:
    row = {
        "service_category": "fitness",
        "temporal_preference": "evening",
        "location_context": "nearby",
        "urgency_level": "high",
        "sentiment_tags": ["quick"],
    }
    assert ambiguity_bucket("Book workout tonight near me urgent", row) == "low"
