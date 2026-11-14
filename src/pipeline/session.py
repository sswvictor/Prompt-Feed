from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from uuid import uuid4

from src.m1_nlu.sentiment import is_negative_turn
from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.belief import (
    apply_category_penalty,
    category_entropy,
    entropy,
    top1_category,
    top1_category_probability,
)
from src.m2_reasoning.convergence import is_category_converged, top1_probability, top_k_hypothesis_ids
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineResult, PipelineRunner


@dataclass
class SessionState:
    session_id: str
    user_id: str
    turn: int = 0
    belief: Dict[str, float] = field(default_factory=dict)
    stv_history: List[StructuredTaskVariable] = field(default_factory=list)
    consecutive_negative_turns: int = 0
    force_clarify: bool = False

    mode: str = "TRACKING"  # TRACKING | TRANSITION
    surface_policy: str = "suggest"  # clarify | suggest | observe | chitchat
    active_category: Optional[str] = None

    clarify_active: bool = False
    clarify_offered_categories: List[str] = field(default_factory=list)
    pending_confirmation: Optional[str] = None
    clarify_last_prompt_type: str = ""
    clarify_turns_used: int = 0
    policy_knobs: Dict[str, object] = field(default_factory=dict)

    candidate_belief: Optional[Dict[str, float]] = None
    candidate_seed_category: Optional[str] = None
    observe_turn: int = 0
    transition_trigger: str = ""
    transition_traces: List[dict] = field(default_factory=list)
    candidate_top1_history: List[float] = field(default_factory=list)
    active_top1_history: List[float] = field(default_factory=list)


class SessionManager:
    def __init__(self, config: PipelineConfig, runner: Optional[PipelineRunner] = None) -> None:
        self.config = config
        self.runner = runner or PipelineRunner(config)

    def new_session(self, user_id: str) -> SessionState:
        m2 = self.config.m2
        return SessionState(
            session_id=str(uuid4()),
            user_id=user_id,
            policy_knobs={
                "top1_suggest_threshold": m2.top1_suggest_threshold,
                "high_entropy_ratio": m2.high_entropy_ratio,
                "clarify_budget": m2.clarify_budget,
            },
        )

    def _apply_emotion_escape(
        self,
        session: SessionState,
        result: PipelineResult,
        user_input: str,
    ) -> PipelineResult:
        sent_cfg = self.config.sentiment
        if not sent_cfg.enabled:
            return result

        if is_negative_turn(user_input, sent_cfg.negative_threshold):
            session.consecutive_negative_turns += 1
        else:
            session.consecutive_negative_turns = 0
            session.force_clarify = False
            return result

        if session.consecutive_negative_turns < sent_cfg.consecutive_negative_turns:
            return result

        target_category = top1_category(result.posterior, result.hypotheses)
        if not target_category:
            session.force_clarify = True
            return result

        posterior = apply_category_penalty(
            result.posterior,
            result.hypotheses,
            target_category,
            penalty=sent_cfg.top_category_penalty,
            other_bonus=sent_cfg.other_category_bonus,
        )
        top_k_ids = top_k_hypothesis_ids(posterior, session.turn)
        top1 = top1_probability(posterior)

        session.force_clarify = True
        if session.mode == "TRANSITION":
            session.mode = "TRACKING"
            session.candidate_belief = None
            session.observe_turn = 0

        return PipelineResult(
            stv=result.stv,
            prior=result.prior,
            posterior=posterior,
            top_k_ids=top_k_ids,
            hypotheses=result.hypotheses,
            prior_entropy=result.prior_entropy,
            posterior_entropy=entropy(posterior),
            top1_prob=top1,
            top1_category_prob=top1_category_probability(posterior, result.hypotheses),
            category_entropy=category_entropy(posterior, result.hypotheses),
            converged=is_category_converged(
                posterior, result.hypotheses, threshold=self.config.converge_threshold
            ),
            emotion_escape_triggered=True,
        )

    def process_turn(self, session: SessionState, user_input: str) -> PipelineResult:
        self.runner.config.user_id = session.user_id
        current = session.belief if session.turn > 0 and session.belief else None
        result = self.runner.process_text(
            user_input,
            turn=session.turn,
            user_id=session.user_id,
            current_belief=current,
            stv_history=list(session.stv_history),
        )
        result = self._apply_emotion_escape(session, result, user_input)

        session.stv_history.append(result.stv)
        session.belief = result.posterior
        session.active_category = top1_category(result.posterior, result.hypotheses)
        session.turn += 1
        return result
