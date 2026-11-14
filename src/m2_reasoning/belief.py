from __future__ import annotations

from typing import Dict, List, Optional

from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.hypothesis import GoalHypothesis

# Defaults for S0/S1 belief redesign (overridable via kwargs / PipelineConfig).
DEFAULT_SLOT_EVIDENCE_POWER = 0.30
DEFAULT_BELIEF_POWER_DECAY = 0.80
DEFAULT_RECENCY_ALPHA = 0.70
DEFAULT_PRIOR_REINJECTION = 0.15


def normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(dist.values()) or 1.0
    return {k: v / total for k, v in dist.items()}


def initialize_belief(hypotheses: List[GoalHypothesis]) -> Dict[str, float]:
    if not hypotheses:
        return {}
    p = 1.0 / len(hypotheses)
    return {h.hypothesis_id: p for h in hypotheses}


def uniform_prior(hypotheses: List[GoalHypothesis]) -> Dict[str, float]:
    return initialize_belief(hypotheses)


def _category_match_scores(confidence: float) -> tuple[float, float]:
    """Map category_confidence to soft match/mismatch likelihoods.

    Linear in [0, 1]: at conf=1 → 0.95/0.05 (legacy hard values); at conf=0 → 0.05/0.95.
    """
    conf = min(1.0, max(0.0, confidence))
    match = 0.05 + 0.90 * conf
    mismatch = 0.05 + 0.90 * (1.0 - conf)
    return match, mismatch


def _slot_multiplier(stv: StructuredTaskVariable, h: GoalHypothesis) -> float:
    score = 1.0
    if stv.temporal_preference:
        temporal = str(stv.temporal_preference).lower()
        if temporal in {"asap", "immediately", "now"}:
            score *= 0.9 if h.temporal_slot == "asap" else 0.35
        elif temporal == "weekend":
            score *= 0.9 if h.temporal_slot == "weekend_anytime" else 0.5
        else:
            score *= 0.85 if temporal in h.temporal_slot else 0.6

    if stv.location_context:
        loc = str(stv.location_context).lower()
        if loc in {"nearby", "near me"}:
            score *= 0.85 if h.location_zone == "nearby" else 0.65
        elif loc in {"city_center", "downtown"}:
            score *= 0.85 if h.location_zone == "city_center" else 0.65

    if stv.urgency_level == "high":
        score *= 0.9 if h.temporal_slot == "asap" else 0.4
    elif stv.urgency_level == "low":
        score *= 1.1 if h.temporal_slot != "asap" else 0.8

    if "affordable" in stv.sentiment_tags:
        score *= 1.15 if h.price_tier == "budget" else 0.9
    if "premium" in stv.sentiment_tags:
        score *= 1.15 if h.price_tier == "premium" else 0.9

    return max(score, 1e-6)


def compute_likelihood(
    stv: StructuredTaskVariable,
    h: GoalHypothesis,
    *,
    slot_evidence_power: float = DEFAULT_SLOT_EVIDENCE_POWER,
) -> float:
    """Category-dominant likelihood; slot terms are down-weighted (S0)."""
    score = 1.0
    cat = stv.service_category
    if cat == "ambiguous":
        pass
    elif cat == "other":
        _, mismatch = _category_match_scores(min(stv.category_confidence, 0.4))
        score *= mismatch
    else:
        match, mismatch = _category_match_scores(stv.category_confidence)
        score *= match if h.service_category == cat else mismatch

    power = min(1.0, max(0.0, slot_evidence_power))
    score *= _slot_multiplier(stv, h) ** power
    return max(score, 1e-6)


def update_belief(
    belief: Dict[str, float],
    stv: StructuredTaskVariable,
    prior: Dict[str, float],
    hypotheses: List[GoalHypothesis],
    *,
    slot_evidence_power: float = DEFAULT_SLOT_EVIDENCE_POWER,
    power_decay: float = DEFAULT_BELIEF_POWER_DECAY,
    recency_alpha: float = DEFAULT_RECENCY_ALPHA,
    prior_reinjection: float = DEFAULT_PRIOR_REINJECTION,
) -> Dict[str, float]:
    """Bayesian update on previous posterior with optional prior soft pull + recency mix.

    p_base(h) ∝ ((1-λ) p_{t-1}(h)^γ + λ prior(h))
    p_bayes(h) ∝ L(stv|h) * p_base(h)
    p_t = normalize(α p_bayes + (1-α) p_{t-1}^γ)
    """
    by_id = {h.hypothesis_id: h for h in hypotheses}
    gamma = min(1.0, max(0.05, power_decay))
    lam = min(1.0, max(0.0, prior_reinjection))
    alpha = min(1.0, max(0.0, recency_alpha))

    decayed = normalize({h_id: max(p, 1e-12) ** gamma for h_id, p in belief.items()})
    base: Dict[str, float] = {}
    for h_id, p_dec in decayed.items():
        prior_p = prior.get(h_id, p_dec)
        base[h_id] = (1.0 - lam) * p_dec + lam * prior_p
    base = normalize(base)

    bayes: Dict[str, float] = {}
    for h_id, p_base in base.items():
        h = by_id[h_id]
        bayes[h_id] = (
            compute_likelihood(stv, h, slot_evidence_power=slot_evidence_power) * p_base
        )
    bayes = normalize(bayes)

    mixed = {
        h_id: alpha * bayes[h_id] + (1.0 - alpha) * decayed.get(h_id, 0.0)
        for h_id in bayes
    }
    return normalize(mixed)


def entropy(belief: Dict[str, float]) -> float:
    import math

    return -sum(p * math.log(p + 1e-12) for p in belief.values())


def resolve_prior(
    hypotheses: List[GoalHypothesis],
    profile_prior: Dict[str, float] | None = None,
    blend_lambda: float = 1.0,
) -> Dict[str, float]:
    """Resolve belief prior: uniform when M3 off; blend when history prior provided."""
    base = uniform_prior(hypotheses)
    if not profile_prior:
        return base
    lam = min(1.0, max(0.0, blend_lambda))
    if lam <= 0.0:
        return base
    if lam >= 1.0:
        aligned = {h.hypothesis_id: profile_prior.get(h.hypothesis_id, 0.0) for h in hypotheses}
        total = sum(aligned.values())
        if total <= 0:
            return base
        return normalize(aligned)
    blended = {
        h.hypothesis_id: (1.0 - lam) * base[h.hypothesis_id] + lam * profile_prior.get(h.hypothesis_id, 0.0)
        for h in hypotheses
    }
    return normalize(blended)


def apply_category_penalty(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
    target_category: str,
    *,
    penalty: float = 0.5,
    other_bonus: float = 1.1,
) -> Dict[str, float]:
    """Down-weight hypotheses in target_category; small boost to others (emotion escape)."""
    by_id = {h.hypothesis_id: h for h in hypotheses}
    adjusted: Dict[str, float] = {}
    for h_id, p in belief.items():
        h = by_id.get(h_id)
        if h is None:
            adjusted[h_id] = p
        elif h.service_category == target_category:
            adjusted[h_id] = p * penalty
        else:
            adjusted[h_id] = p * other_bonus
    return normalize(adjusted)


def category_masses(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
) -> Dict[str, float]:
    by_id = {h.hypothesis_id: h for h in hypotheses}
    masses: Dict[str, float] = {}
    for h_id, p in belief.items():
        h = by_id.get(h_id)
        if h is None:
            continue
        masses[h.service_category] = masses.get(h.service_category, 0.0) + p
    return masses


def top1_category_probability(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
) -> float:
    masses = category_masses(belief, hypotheses)
    return max(masses.values()) if masses else 0.0


def category_entropy(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
) -> float:
    import math

    masses = category_masses(belief, hypotheses)
    return -sum(p * math.log(p + 1e-12) for p in masses.values())


def top1_category(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
) -> Optional[str]:
    if not belief:
        return None
    masses = category_masses(belief, hypotheses)
    if not masses:
        return None
    return max(masses.items(), key=lambda x: x[1])[0]


def belief_step(
    stv: StructuredTaskVariable,
    hypotheses: List[GoalHypothesis],
    prior: Dict[str, float],
    turn: int = 0,
) -> tuple[Dict[str, float], List[str]]:
    """Single M2 update: init belief from prior, apply STV evidence, return posterior + top-k ids."""
    from src.m2_reasoning.convergence import top_k_hypothesis_ids

    belief = dict(prior)
    posterior = update_belief(belief, stv, prior, hypotheses)
    top_ids = top_k_hypothesis_ids(posterior, turn)
    return posterior, top_ids
