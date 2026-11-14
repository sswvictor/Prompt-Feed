from __future__ import annotations

from src.dialogue.intention_context import resolve_dialogue_policy
from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.hypothesis import GoalHypothesis, generate_hypothesis_space
from src.pipeline.runner import PipelineResult


def _pipeline_result(
    *,
    top1_prob: float = 0.1,
    top1_category_prob: float | None = None,
    converged: bool = False,
    posterior_entropy: float = 4.0,
    hypothesis_count: int = 540,
    service_category: str = "fitness",
    clarification_needed: bool = False,
) -> PipelineResult:
    hyps = generate_hypothesis_space(max_size=hypothesis_count)
    cat_mass = top1_category_prob if top1_category_prob is not None else top1_prob
    other_mass = max(1e-6, (1.0 - cat_mass) / max(1, len({h.service_category for h in hyps}) - 1))
    posterior: dict[str, float] = {}
    for h in hyps:
        posterior[h.hypothesis_id] = cat_mass if h.service_category == service_category else other_mass
    total = sum(posterior.values())
    posterior = {k: v / total for k, v in posterior.items()}
    from src.m2_reasoning.belief import category_entropy as cat_ent_fn
    from src.m2_reasoning.belief import top1_category_probability as top1_cat_fn

    stv = StructuredTaskVariable(
        service_category=service_category,
        category_confidence=0.9,
        temporal_preference="evening",
        urgency_level="medium",
        time_window_days=None,
        location_context="nearby",
        sentiment_tags=[],
        extraction_confidence=0.9,
        raw_input="test",
        clarification_needed=clarification_needed,
    )
    return PipelineResult(
        stv=stv,
        prior=posterior,
        posterior=posterior,
        top_k_ids=list(posterior.keys())[:3],
        hypotheses=hyps,
        posterior_entropy=posterior_entropy,
        top1_prob=top1_prob,
        top1_category_prob=top1_cat_fn(posterior, hyps),
        category_entropy=cat_ent_fn(posterior, hyps),
        converged=converged,
    )


def test_force_clarify_overrides_confident_belief() -> None:
    result = _pipeline_result(top1_prob=0.95, converged=True, posterior_entropy=0.5)
    assert resolve_dialogue_policy(result, force_clarify=True) == "clarify"


def test_suggest_when_converged() -> None:
    result = _pipeline_result(top1_category_prob=0.9, converged=True, posterior_entropy=0.8)
    assert resolve_dialogue_policy(result) == "suggest"


def test_suggest_when_top1_above_threshold() -> None:
    result = _pipeline_result(top1_category_prob=0.55, converged=False, posterior_entropy=1.2)
    assert resolve_dialogue_policy(result) == "suggest"


def test_clarify_when_not_converged_and_low_top1() -> None:
    result = _pipeline_result(top1_category_prob=0.2, converged=False, posterior_entropy=1.5)
    assert resolve_dialogue_policy(result) == "clarify"


def test_clarify_budget_forces_suggest() -> None:
    result = _pipeline_result(top1_category_prob=0.2, converged=False, posterior_entropy=1.5)
    assert resolve_dialogue_policy(result, clarify_turns_used=0, clarify_budget=1) == "clarify"
    assert resolve_dialogue_policy(result, clarify_turns_used=1, clarify_budget=1) == "suggest"


def test_clarify_on_high_posterior_entropy() -> None:
    # Near-uniform over 6 categories: log(6) ≈ 1.79; 0.95 * log(6) ≈ 1.70
    hyps = generate_hypothesis_space(max_size=120)
    posterior = {h.hypothesis_id: 1.0 / len(hyps) for h in hyps}
    from src.m2_reasoning.belief import category_entropy as cat_ent_fn
    from src.m2_reasoning.belief import top1_category_probability as top1_cat_fn

    stv = StructuredTaskVariable(
        service_category="fitness",
        category_confidence=0.9,
        temporal_preference="evening",
        urgency_level="medium",
        time_window_days=None,
        location_context="nearby",
        sentiment_tags=[],
        extraction_confidence=0.9,
        raw_input="test",
        clarification_needed=False,
    )
    result = PipelineResult(
        stv=stv,
        prior=posterior,
        posterior=posterior,
        top_k_ids=list(posterior.keys())[:3],
        hypotheses=hyps,
        posterior_entropy=4.0,
        top1_prob=0.55,
        top1_category_prob=top1_cat_fn(posterior, hyps),
        category_entropy=cat_ent_fn(posterior, hyps),
        converged=False,
    )
    assert resolve_dialogue_policy(result) == "clarify"


def test_m1_clarification_flags_do_not_drive_policy() -> None:
    # M1 ambiguous/clarification flags must not override converged M2 belief.
    # Use a valid hypothesis-space category so posterior mass is non-uniform.
    result = _pipeline_result(
        top1_category_prob=0.9,
        converged=True,
        posterior_entropy=0.6,
        service_category="fitness",
        clarification_needed=True,
    )
    assert resolve_dialogue_policy(result) == "suggest"
