from __future__ import annotations

from typing import List, Optional

from src.dialogue.keywords import (
    detect_primary_category,
    is_confirmation,
    is_rejection,
    supports_category,
)
from src.m2_reasoning.belief import category_masses, top1_category
from src.pipeline.runner import PipelineResult
from src.pipeline.session import SessionState

CLARIFY_ENUMERATE_K = 3


def top_categories_for_clarify(result: PipelineResult, k: int = CLARIFY_ENUMERATE_K) -> List[str]:
    masses = category_masses(result.posterior, result.hypotheses)
    ranked = sorted(masses.items(), key=lambda x: -x[1])
    return [cat for cat, _ in ranked[:k] if cat not in {"other", "ambiguous"}]


def enter_clarify(session: SessionState, offered: List[str]) -> None:
    session.clarify_active = True
    session.clarify_offered_categories = offered
    session.pending_confirmation = None
    session.clarify_last_prompt_type = "enumerate"
    session.surface_policy = "clarify"


def exit_clarify_to_suggest(session: SessionState, category: Optional[str] = None) -> None:
    session.clarify_active = False
    session.pending_confirmation = None
    session.clarify_offered_categories = []
    session.clarify_last_prompt_type = ""
    session.surface_policy = "suggest"
    if category:
        session.active_category = category


def update_clarify_after_turn(
    session: SessionState,
    result: PipelineResult,
    user_input: str,
) -> str:
    """Update clarify sub-state; return prompt_type: enumerate | confirm | resolved."""
    keyword_cat = detect_primary_category(user_input)

    if session.pending_confirmation:
        if is_confirmation(user_input):
            exit_clarify_to_suggest(session, session.pending_confirmation)
            return "resolved"
        if is_rejection(user_input):
            session.pending_confirmation = None
            session.clarify_last_prompt_type = "enumerate"
            return "enumerate"

    if keyword_cat:
        session.pending_confirmation = keyword_cat
        session.clarify_last_prompt_type = "confirm"
        return "confirm"

    if result.converged or result.top1_category_prob >= 0.5:
        cat = top1_category(result.posterior, result.hypotheses)
        if cat and not keyword_cat:
            exit_clarify_to_suggest(session, cat)
            return "resolved"

    if not session.clarify_offered_categories:
        enter_clarify(session, top_categories_for_clarify(result))
        return "enumerate"

    session.clarify_last_prompt_type = "enumerate"
    return "enumerate"


def should_enter_clarify(session: SessionState, result: PipelineResult, force_clarify: bool) -> bool:
    if session.mode == "TRANSITION":
        return False
    if force_clarify:
        return True
    if session.clarify_active:
        return True
    from src.dialogue.intention_context import resolve_dialogue_policy

    knobs = getattr(session, "policy_knobs", {}) or {}
    return (
        resolve_dialogue_policy(
            result,
            force_clarify=force_clarify,
            top1_threshold=float(knobs.get("top1_suggest_threshold", 0.45)),
            high_entropy_ratio=float(knobs.get("high_entropy_ratio", 0.95)),
            clarify_turns_used=int(getattr(session, "clarify_turns_used", 0) or 0),
            clarify_budget=int(knobs.get("clarify_budget", 2)),
        )
        == "clarify"
    )


def active_supports_category(session: SessionState, text: str) -> bool:
    return bool(session.active_category and supports_category(text, session.active_category))
