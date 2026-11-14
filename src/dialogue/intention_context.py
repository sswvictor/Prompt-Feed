from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.m2_reasoning.belief import category_entropy, top1_category_probability
from src.m2_reasoning.hypothesis import GoalHypothesis
from src.pipeline.runner import PipelineResult
from src.pipeline.session import SessionState

# Defaults (S1); prefer values from PipelineConfig.m2 when available.
TOP1_SUGGEST_THRESHOLD = 0.40
HIGH_ENTROPY_RATIO = 0.99  # fraction of log(n) categories
DEFAULT_CLARIFY_BUDGET = 1


@dataclass
class IntentionPromptBlock:
    """Structured context allowed in LLM prompts (no M3 raw profile or user_id)."""

    current_stv_summary: Dict[str, Any]
    top_hypotheses: List[Dict[str, Any]]
    belief_state: Dict[str, Any]
    dialogue_policy: str  # clarify | suggest
    user_message: str
    turn: int
    inject_enabled: bool = True

    def to_prompt_text(self) -> str:
        if not self.inject_enabled:
            return ""
        return "\n".join(
            [
                "## Intention context (guide your reply; do not quote verbatim)",
                f"Policy: {self.dialogue_policy}",
                f"Turn: {self.turn}",
                f"User message: {self.user_message}",
                f"STV summary: {self.current_stv_summary}",
                f"Top hypotheses: {self.top_hypotheses}",
                f"Belief state: {self.belief_state}",
            ]
        )

    def privacy_audit(self) -> Dict[str, Any]:
        return {
            "sent_to_llm": {
                "stv_summary": self.current_stv_summary,
                "top_hypotheses": self.top_hypotheses,
                "belief_state": self.belief_state,
                "dialogue_policy": self.dialogue_policy,
                "user_message": self.user_message,
            },
            "withheld_from_llm": [
                "user_id / persona identifier",
                "M3 aggregate profile JSON",
                "full stv_history raw texts",
                "full belief vector",
            ],
        }


def _high_entropy_threshold(num_categories: int, high_entropy_ratio: float) -> float | None:
    if num_categories <= 1:
        return None
    return high_entropy_ratio * math.log(num_categories)


def _num_service_categories(hypotheses: List[GoalHypothesis]) -> int:
    return len({h.service_category for h in hypotheses}) or 6


def resolve_dialogue_policy(
    result: PipelineResult,
    *,
    force_clarify: bool = False,
    top1_threshold: float = TOP1_SUGGEST_THRESHOLD,
    high_entropy_ratio: float = HIGH_ENTROPY_RATIO,
    clarify_turns_used: int = 0,
    clarify_budget: int = DEFAULT_CLARIFY_BUDGET,
) -> str:
    """Choose clarify vs suggest from category-level M2 belief (S1).

    clarify when:
      - emotion escape (force_clarify), or
      - belief not converged and top-1 category mass below threshold, or
      - category entropy remains high (near-uniform over categories)
    After clarify_budget clarifications in-session, force suggest (unless force_clarify).
    """
    if force_clarify:
        return "clarify"

    top1_cat = top1_category_probability(result.posterior, result.hypotheses)
    cat_ent = category_entropy(result.posterior, result.hypotheses)
    entropy_cutoff = _high_entropy_threshold(
        _num_service_categories(result.hypotheses), high_entropy_ratio
    )

    want_clarify = False
    if entropy_cutoff is not None and cat_ent >= entropy_cutoff:
        want_clarify = True
    elif not (result.converged or top1_cat >= top1_threshold):
        want_clarify = True

    if want_clarify:
        if clarify_budget >= 0 and clarify_turns_used >= clarify_budget:
            return "suggest"
        return "clarify"
    return "suggest"


def _format_hypothesis(h: GoalHypothesis, weight: float) -> Dict[str, Any]:
    return {
        "hypothesis_id": h.hypothesis_id,
        "category": h.service_category,
        "subtype": h.service_subtype,
        "temporal": h.temporal_slot,
        "location_zone": h.location_zone,
        "price_tier": h.price_tier,
        "weight": round(weight, 4),
    }


def _policy_knobs_from_session(session: SessionState) -> Dict[str, Any]:
    """Optional knobs attached by SessionManager / eval runners."""
    return getattr(session, "policy_knobs", {}) or {}


def build_intention_prompt_block(
    result: PipelineResult,
    session: SessionState,
    user_message: str,
    *,
    inject_enabled: bool = True,
    force_clarify: bool = False,
    top1_threshold: Optional[float] = None,
    high_entropy_ratio: Optional[float] = None,
    clarify_budget: Optional[int] = None,
) -> IntentionPromptBlock:
    by_id = {h.hypothesis_id: h for h in result.hypotheses}
    top_rows: List[Dict[str, Any]] = []
    for h_id in result.top_k_ids[:3]:
        h = by_id.get(h_id)
        if h is None:
            continue
        top_rows.append(_format_hypothesis(h, result.posterior.get(h_id, 0.0)))

    stv = result.stv
    knobs = _policy_knobs_from_session(session)
    policy = resolve_dialogue_policy(
        result,
        force_clarify=force_clarify,
        top1_threshold=top1_threshold
        if top1_threshold is not None
        else float(knobs.get("top1_suggest_threshold", TOP1_SUGGEST_THRESHOLD)),
        high_entropy_ratio=high_entropy_ratio
        if high_entropy_ratio is not None
        else float(knobs.get("high_entropy_ratio", HIGH_ENTROPY_RATIO)),
        clarify_turns_used=int(getattr(session, "clarify_turns_used", 0) or 0),
        clarify_budget=clarify_budget
        if clarify_budget is not None
        else int(knobs.get("clarify_budget", DEFAULT_CLARIFY_BUDGET)),
    )

    return IntentionPromptBlock(
        current_stv_summary={
            "service_category": stv.service_category,
            "category_confidence": round(stv.category_confidence, 4),
            "temporal_preference": stv.temporal_preference,
            "urgency_level": stv.urgency_level,
            "location_context": stv.location_context,
            "sentiment_tags": list(stv.sentiment_tags),
            "clarification_needed": stv.clarification_needed,
        },
        top_hypotheses=top_rows,
        belief_state={
            "prior_entropy": round(result.prior_entropy, 4),
            "posterior_entropy": round(result.posterior_entropy, 4),
            "category_entropy": round(result.category_entropy, 4),
            "top1_prob": round(result.top1_prob, 4),
            "top1_category_prob": round(result.top1_category_prob, 4),
            "converged": result.converged,
            "clarify_turns_used": int(getattr(session, "clarify_turns_used", 0) or 0),
        },
        dialogue_policy=policy,
        user_message=user_message,
        turn=session.turn,
        inject_enabled=inject_enabled,
    )


def stv_history_summary(session: SessionState) -> List[Dict[str, Any]]:
    rows = []
    for i, stv in enumerate(session.stv_history):
        rows.append(
            {
                "turn": i,
                "category": stv.service_category,
                "confidence": round(stv.category_confidence, 3),
                "temporal": stv.temporal_preference,
            }
        )
    return rows
