from __future__ import annotations

import time

from src.m2_reasoning.belief import entropy
from src.m2_reasoning.hypothesis import generate_hypothesis_space
from src.m3_memory.prior import blend_priors, compute_prior, normalize_with_smoothing
from src.m3_memory.profile import AggregatedUserProfile


def test_normalize_with_smoothing_sums_to_one() -> None:
    probs = normalize_with_smoothing({"fitness": 10, "wellness": 5}, decay=1.0, alpha=0.1)
    assert abs(sum(probs.values()) - 1.0) < 1e-8
    assert probs["fitness"] > probs["wellness"]


def test_compute_prior_fitness_heavy_profile() -> None:
    hs = generate_hypothesis_space(max_size=80)
    profile = AggregatedUserProfile(
        user_id="u_fitness",
        category_counts={"fitness": 20, "wellness": 2},
        temporal_counts={"asap": 10, "morning": 5},
        price_tier_counts={"budget": 8, "mid": 10},
        last_updated=time.time(),
        total_sessions=22,
    )
    prior = compute_prior(profile, hs)
    assert abs(sum(prior.values()) - 1.0) < 1e-8
    fitness_mass = sum(p for h in hs if h.service_category == "fitness" for p in [prior[h.hypothesis_id]])
    wellness_mass = sum(p for h in hs if h.service_category == "wellness" for p in [prior[h.hypothesis_id]])
    assert fitness_mass > wellness_mass


def test_cold_start_returns_uniform() -> None:
    hs = generate_hypothesis_space(max_size=50)
    from src.m2_reasoning.belief import uniform_prior

    profile = AggregatedUserProfile(user_id="cold", total_sessions=0)
    prior = compute_prior(profile, hs)
    uniform = uniform_prior(hs)
    for h in hs:
        assert abs(prior[h.hypothesis_id] - uniform[h.hypothesis_id]) < 1e-8


def test_blend_priors_reduces_entropy_when_history_peaked() -> None:
    hs = generate_hypothesis_space(max_size=40)
    from src.m2_reasoning.belief import uniform_prior

    uniform = uniform_prior(hs)
    profile = AggregatedUserProfile(
        user_id="u",
        category_counts={"fitness": 50},
        temporal_counts={"asap": 30},
        price_tier_counts={"budget": 20},
        last_updated=time.time(),
        total_sessions=50,
    )
    history = compute_prior(profile, hs)
    blended = blend_priors(uniform, history, history_weight=1.0)
    assert entropy(blended) < entropy(uniform)
