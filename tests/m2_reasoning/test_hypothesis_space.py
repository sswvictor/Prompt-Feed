from __future__ import annotations

from collections import Counter

from src.m2_reasoning.hypothesis import SERVICE_CATEGORIES, generate_hypothesis_space


def test_hypothesis_space_size_and_ids() -> None:
    hs = generate_hypothesis_space(max_size=80)
    assert len(hs) == 80
    assert hs[0].hypothesis_id.startswith("h")
    assert hs[-1].hypothesis_id.startswith("h")


def test_even_category_coverage_at_legacy_360() -> None:
    """Regression: sequential fill used to omit dining/travel at max_size=360."""
    hs = generate_hypothesis_space(max_size=360)
    counts = Counter(h.service_category for h in hs)
    assert set(counts) == set(SERVICE_CATEGORIES)
    assert min(counts.values()) >= 50
    assert max(counts.values()) - min(counts.values()) <= 1


def test_compact_120_covers_all_domains() -> None:
    hs = generate_hypothesis_space(max_size=120)
    counts = Counter(h.service_category for h in hs)
    assert set(counts) == set(SERVICE_CATEGORIES)
    assert all(v == 20 for v in counts.values())
