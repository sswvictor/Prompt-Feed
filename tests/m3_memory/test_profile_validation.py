from __future__ import annotations

import pytest

from src.m3_memory.profile import AggregatedUserProfile, empty_profile, validate_profile_dict


def test_empty_profile_has_zero_sessions() -> None:
    p = empty_profile("u_test")
    assert p.user_id == "u_test"
    assert p.total_sessions == 0
    assert p.category_counts == {}


def test_from_dict_roundtrip() -> None:
    data = {
        "user_id": "u1",
        "category_counts": {"fitness": 10, "wellness": 3},
        "temporal_counts": {"morning": 5},
        "price_tier_counts": {"budget": 2, "mid": 8},
        "last_updated": 1_700_000_000.0,
        "total_sessions": 13,
    }
    p = AggregatedUserProfile.from_dict(data)
    assert p.category_counts["fitness"] == 10
    assert p.total_sessions == 13


def test_forbidden_pii_fields_rejected() -> None:
    with pytest.raises(ValueError, match="Forbidden"):
        validate_profile_dict(
            {
                "user_id": "u1",
                "email": "a@b.com",
                "category_counts": {},
                "temporal_counts": {},
                "price_tier_counts": {},
                "last_updated": 0.0,
                "total_sessions": 0,
            }
        )
