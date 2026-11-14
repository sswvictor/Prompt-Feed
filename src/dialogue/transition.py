from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.dialogue.keywords import (
    SERVICE_CATEGORIES,
    detect_primary_category,
    supports_category,
)
from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.belief import (
    apply_category_penalty,
    category_entropy,
    category_masses,
    normalize,
    top1_category,
    top1_category_probability,
    update_belief,
)
from src.m2_reasoning.convergence import is_category_converged, top_k_hypothesis_ids, top1_probability
from src.m2_reasoning.hypothesis import GoalHypothesis
from src.m2_reasoning.belief import entropy
from src.pipeline.runner import PipelineResult, PipelineRunner
from src.pipeline.session import SessionState

ABANDON_PATTERNS = (
    r"\bforget it\b",
    r"\bnever mind\b",
    r"\bnevermind\b",
    r"\bskip that\b",
    r"\bnot that\b",
    r"算了",
    r"不要了",
    r"不用了",
    r"换个别的话题",
)

PIVOT_PATTERNS = (
    r"\binstead\b",
    r"\bwhat about\b",
    r"\brather\b",
    r"\bswitch to\b",
    r"还是",
    r"换成",
    r"不如",
)

TRANSITION_OBSERVE_TURNS = 3
DECAY_FACTOR = 0.75
THETA_PROMOTE = 0.45
DELTA_PROMOTE = 0.15
EARLY_PROMOTE_THRESHOLD = 0.55


@dataclass
class TransitionDecision:
    action: str  # continue | promote | abort | fallback_clarify
    reason: str


def detect_hard_transition(text: str) -> bool:
    lower = text.lower()
    for pat in ABANDON_PATTERNS + PIVOT_PATTERNS:
        if re.search(pat, lower):
            return True
    return False


def open_transition(
    session: SessionState,
    user_input: str,
    hypotheses: List[GoalHypothesis],
) -> Optional[str]:
    seed = detect_primary_category(user_input)
    session.mode = "TRANSITION"
    session.observe_turn = 0
    session.transition_trigger = "hard"
    session.clarify_active = False
    session.pending_confirmation = None
    session.clarify_offered_categories = []
    session.surface_policy = "observe"
    session.candidate_seed_category = seed
    session.candidate_belief = None
    session.candidate_top1_history = []
    session.active_top1_history = []
    if session.belief:
        session.active_category = top1_category(session.belief, hypotheses)
    if seed:
        session.candidate_seed_category = seed
    return seed


def init_candidate_belief(
    runner: PipelineRunner,
    user_input: str,
    user_id: str,
    seed_category: Optional[str],
) -> Dict[str, float]:
    result = runner.process_text(user_input, turn=0, user_id=user_id, current_belief=None)
    belief = dict(result.posterior)
    if seed_category and seed_category in SERVICE_CATEGORIES:
        belief = _boost_category(belief, result.hypotheses, seed_category, boost=1.35)
    return belief


def _boost_category(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
    category: str,
    *,
    boost: float,
) -> Dict[str, float]:
    by_id = {h.hypothesis_id: h for h in hypotheses}
    adjusted = {}
    for h_id, p in belief.items():
        h = by_id.get(h_id)
        if h and h.service_category == category:
            adjusted[h_id] = p * boost
        else:
            adjusted[h_id] = p
    return normalize(adjusted)


def decay_active_belief(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
    active_category: Optional[str],
    *,
    decay_factor: float = DECAY_FACTOR,
) -> Dict[str, float]:
    if not active_category:
        return belief
    return apply_category_penalty(
        belief,
        hypotheses,
        active_category,
        penalty=decay_factor,
        other_bonus=1.0,
    )


def _top1_for_belief(belief: Dict[str, float], hypotheses: List[GoalHypothesis]) -> float:
    if not belief:
        return 0.0
    return top1_category_probability(belief, hypotheses)


def _build_result_from_belief(
    runner: PipelineRunner,
    stv: StructuredTaskVariable,
    prior: Dict[str, float],
    posterior: Dict[str, float],
    hypotheses: List[GoalHypothesis],
    turn: int,
) -> PipelineResult:
    top_k_ids = top_k_hypothesis_ids(posterior, turn)
    return PipelineResult(
        stv=stv,
        prior=prior,
        posterior=posterior,
        top_k_ids=top_k_ids,
        hypotheses=hypotheses,
        prior_entropy=entropy(prior),
        posterior_entropy=entropy(posterior),
        top1_prob=top1_probability(posterior),
        top1_category_prob=top1_category_probability(posterior, hypotheses),
        category_entropy=category_entropy(posterior, hypotheses),
        converged=is_category_converged(
            posterior, hypotheses, threshold=runner.config.converge_threshold
        ),
    )


def process_transition_turn(
    runner: PipelineRunner,
    session: SessionState,
    user_input: str,
) -> Tuple[PipelineResult, TransitionDecision]:
    hypotheses = runner.hypotheses
    stv = runner._extractor.extract(user_input)
    prior = runner.resolve_prior_for_user(session.user_id, stv=stv)

    active = dict(session.belief) if session.belief else dict(prior)
    active_support = supports_category(user_input, session.active_category or "")
    if not active_support and session.active_category:
        active = decay_active_belief(active, hypotheses, session.active_category)

    active = update_belief(active, stv, prior, hypotheses)

    seed = session.candidate_seed_category or detect_primary_category(user_input)
    if session.candidate_belief is None:
        session.candidate_belief = init_candidate_belief(runner, user_input, session.user_id, seed)
    else:
        cand = dict(session.candidate_belief)
        session.candidate_belief = update_belief(cand, stv, prior, hypotheses)
        if seed and seed != session.candidate_seed_category and detect_hard_transition(user_input):
            session.candidate_seed_category = seed
            session.candidate_belief = init_candidate_belief(runner, user_input, session.user_id, seed)
            session.observe_turn = 0

    session.observe_turn += 1
    active_top1 = _top1_for_belief(active, hypotheses)
    cand_top1 = _top1_for_belief(session.candidate_belief or {}, hypotheses)
    session.active_top1_history.append(active_top1)
    session.candidate_top1_history.append(cand_top1)
    session.transition_traces.append(
        {
            "turn": session.observe_turn,
            "active_top1_cat": round(active_top1, 4),
            "candidate_top1_cat": round(cand_top1, 4),
            "m1_category": stv.service_category,
            "seed": session.candidate_seed_category,
        }
    )

    session.belief = active
    result = _build_result_from_belief(runner, stv, prior, active, hypotheses, session.turn)

    decision = evaluate_transition(session, user_input, active_top1, cand_top1)
    if decision.action == "promote":
        _promote_candidate(session, hypotheses)
    elif decision.action == "abort":
        _abort_transition(session)
    elif decision.action == "fallback_clarify":
        _fallback_clarify(session, hypotheses)

    session.stv_history.append(stv)
    session.turn += 1
    return result, decision


def evaluate_transition(
    session: SessionState,
    user_input: str,
    active_top1: float,
    cand_top1: float,
) -> TransitionDecision:
    seed = detect_primary_category(user_input)
    if seed and supports_category(user_input, seed) and cand_top1 >= EARLY_PROMOTE_THRESHOLD:
        return TransitionDecision("promote", "early_strong_keyword")

    if (
        cand_top1 >= THETA_PROMOTE
        and cand_top1 >= active_top1 + DELTA_PROMOTE
        and _not_falling(session.candidate_top1_history)
    ):
        if session.observe_turn >= 1:
            return TransitionDecision("promote", "relative_advantage")

    if session.active_category and supports_category(user_input, session.active_category):
        if len(session.active_top1_history) >= 2 and _rising(session.active_top1_history):
            return TransitionDecision("abort", "active_support")

    if session.observe_turn >= TRANSITION_OBSERVE_TURNS:
        if cand_top1 > active_top1:
            return TransitionDecision("promote", "timeout_candidate_leads")
        if active_top1 > cand_top1 + DELTA_PROMOTE:
            return TransitionDecision("abort", "timeout_active_leads")
        return TransitionDecision("fallback_clarify", "timeout_tied")

    return TransitionDecision("continue", "observing")


def _not_falling(history: List[float]) -> bool:
    if len(history) < 2:
        return True
    return history[-1] >= history[-2] - 0.05


def _rising(history: List[float]) -> bool:
    if len(history) < 2:
        return False
    return history[-1] > history[-2]


def _promote_candidate(session: SessionState, hypotheses: List[GoalHypothesis]) -> None:
    if session.candidate_belief:
        session.belief = dict(session.candidate_belief)
    session.active_category = top1_category(session.belief, hypotheses)
    session.mode = "TRACKING"
    session.surface_policy = "suggest"
    session.candidate_belief = None
    session.candidate_seed_category = None
    session.observe_turn = 0
    session.transition_traces = []


def _abort_transition(session: SessionState) -> None:
    session.mode = "TRACKING"
    session.surface_policy = "suggest" if session.belief else "clarify"
    session.candidate_belief = None
    session.candidate_seed_category = None
    session.observe_turn = 0


def _fallback_clarify(session: SessionState, hypotheses: List[GoalHypothesis]) -> None:
    masses = category_masses(session.belief, hypotheses)
    cand_masses = (
        category_masses(session.candidate_belief, hypotheses) if session.candidate_belief else {}
    )
    merged: Dict[str, float] = {}
    for cat in set(masses) | set(cand_masses):
        merged[cat] = masses.get(cat, 0.0) + cand_masses.get(cat, 0.0)
    top = sorted(merged.items(), key=lambda x: -x[1])[:2]
    session.mode = "TRACKING"
    session.clarify_active = True
    session.clarify_offered_categories = [c for c, _ in top]
    session.pending_confirmation = None
    session.surface_policy = "clarify"
    session.candidate_belief = None
    session.observe_turn = 0
