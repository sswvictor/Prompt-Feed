from __future__ import annotations

from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.belief import (
    compute_likelihood,
    initialize_belief,
    uniform_prior,
    update_belief,
)
from src.m2_reasoning.hypothesis import GoalHypothesis, generate_hypothesis_space


def _fitness_hypothesis() -> GoalHypothesis:
    return GoalHypothesis(
        hypothesis_id="h_fit",
        service_category="fitness",
        service_subtype="gym",
        temporal_slot="asap",
        location_zone="nearby",
        price_tier="budget",
    )


def _wellness_hypothesis() -> GoalHypothesis:
    return GoalHypothesis(
        hypothesis_id="h_well",
        service_category="wellness",
        service_subtype="spa",
        temporal_slot="asap",
        location_zone="nearby",
        price_tier="budget",
    )


def _minimal_stv(service_category: str, category_confidence: float) -> StructuredTaskVariable:
    return StructuredTaskVariable(
        service_category=service_category,
        category_confidence=category_confidence,
        temporal_preference=None,
        urgency_level="medium",
        time_window_days=None,
        location_context=None,
        sentiment_tags=[],
        extraction_confidence=0.8,
        raw_input="x",
        clarification_needed=False,
    )


def test_category_confidence_softens_match() -> None:
    """Low confidence pulls matching/mismatching likelihoods toward uniform."""
    h_fit = _fitness_hypothesis()
    stv_low = _minimal_stv("fitness", 0.3)
    stv_high = _minimal_stv("fitness", 0.9)

    like_match_low = compute_likelihood(stv_low, h_fit)
    like_match_high = compute_likelihood(stv_high, h_fit)
    assert like_match_high > like_match_low

    h_well = _wellness_hypothesis()
    like_mismatch_low = compute_likelihood(stv_low, h_well)
    like_mismatch_high = compute_likelihood(stv_high, h_well)
    assert like_mismatch_low > like_mismatch_high

    # conf=0.3 → match=0.32, mismatch=0.68; conf=0.9 → match=0.86, mismatch=0.14
    assert abs(like_match_low - 0.32) < 1e-9
    assert abs(like_match_high - 0.86) < 1e-9


def test_ambiguous_skips_category_evidence() -> None:
    h_fit = _fitness_hypothesis()
    h_well = _wellness_hypothesis()
    stv = _minimal_stv("ambiguous", 0.2)

    assert compute_likelihood(stv, h_fit) == compute_likelihood(stv, h_well)


def test_other_applies_weak_category_penalty() -> None:
    h_fit = _fitness_hypothesis()
    stv = _minimal_stv("other", 0.9)
    # capped at 0.4 → mismatch = 0.05 + 0.90 * 0.6 = 0.59
    assert abs(compute_likelihood(stv, h_fit) - 0.59) < 1e-9


def test_high_confidence_sharpens_fitness_posterior() -> None:
    # Need >90 hypotheses so wellness entries exist alongside fitness (90 per category).
    hs = generate_hypothesis_space(max_size=100)
    prior = uniform_prior(hs)
    belief = initialize_belief(hs)

    stv_low = _minimal_stv("fitness", 0.3)
    stv_high = _minimal_stv("fitness", 0.9)

    post_low = update_belief(belief, stv_low, prior, hs)
    post_high = update_belief(belief, stv_high, prior, hs)

    def fitness_mass(d: dict[str, float]) -> float:
        return sum(p for h in hs if h.service_category == "fitness" for p in [d[h.hypothesis_id]])

    assert fitness_mass(post_high) > fitness_mass(post_low)


def test_belief_normalization_after_update() -> None:
    hs = generate_hypothesis_space(max_size=50)
    belief = initialize_belief(hs)
    prior = uniform_prior(hs)
    stv = StructuredTaskVariable(
        service_category="fitness",
        category_confidence=0.9,
        temporal_preference="asap",
        urgency_level="high",
        time_window_days=1,
        location_context="nearby",
        sentiment_tags=["quick"],
        extraction_confidence=0.9,
        raw_input="x",
        clarification_needed=False,
    )
    updated = update_belief(belief, stv, prior, hs)
    s = sum(updated.values())
    assert abs(s - 1.0) < 1e-8


def test_category_flip_follows_recent_m1() -> None:
    """After fitness evidence, a high-conf dining STV should move mass to dining."""
    hs = generate_hypothesis_space(max_size=120)
    prior = uniform_prior(hs)
    belief = initialize_belief(hs)

    stv_fit = _minimal_stv("fitness", 0.85)
    mid = update_belief(belief, stv_fit, prior, hs)
    stv_din = _minimal_stv("dining", 0.9)
    post = update_belief(mid, stv_din, prior, hs)

    def mass(cat: str, d: dict[str, float]) -> float:
        return sum(d[h.hypothesis_id] for h in hs if h.service_category == cat)

    assert mass("dining", post) > mass("fitness", post)
    assert mass("dining", post) > 0.35
