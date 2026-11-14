from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.m1_nlu.category_keywords import apply_domain_guards, vote_category_from_history
from src.m1_nlu.evidence_quality import dampen_conflicting_turn
from src.m1_nlu.factory import get_extractor
from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.belief import (
    category_entropy,
    entropy,
    resolve_prior,
    top1_category_probability,
    uniform_prior,
    update_belief,
)
from src.m2_reasoning.convergence import is_category_converged, top_k_hypothesis_ids, top1_probability
from src.m2_reasoning.hypothesis import GoalHypothesis, generate_hypothesis_space
from src.m3_memory.prior import compute_prior, load_m3_config
from src.m3_memory.store import MemoryStore, create_store
from src.pipeline.config import PipelineConfig


@dataclass
class PipelineResult:
    stv: StructuredTaskVariable
    prior: Dict[str, float]
    posterior: Dict[str, float]
    top_k_ids: List[str]
    hypotheses: List[GoalHypothesis] = field(repr=False)
    prior_entropy: float = 0.0
    posterior_entropy: float = 0.0
    top1_prob: float = 0.0
    top1_category_prob: float = 0.0
    category_entropy: float = 0.0
    converged: bool = False
    emotion_escape_triggered: bool = False


class PipelineRunner:
    """M1 → M3 prior → M2 belief update (intention tracking core)."""

    def __init__(self, config: PipelineConfig, store: Optional[MemoryStore] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.config = config.resolve_paths(root)
        self._extractor = get_extractor(self.config.extractor_mode)
        self._hypotheses = generate_hypothesis_space(max_size=self.config.hypothesis_max_size)
        self._by_id = {h.hypothesis_id: h for h in self._hypotheses}
        self._store = store
        if store is None and self.config.enable_m3:
            self._store = create_store(self.config.memory_backend, self.config.sqlite_path)

    @property
    def store(self) -> Optional[MemoryStore]:
        return self._store

    @property
    def hypotheses(self) -> List[GoalHypothesis]:
        return self._hypotheses

    def hypothesis_by_id(self, hypothesis_id: str) -> Optional[GoalHypothesis]:
        return self._by_id.get(hypothesis_id)

    def _m3_blend_lambda(self, stv: StructuredTaskVariable) -> float:
        if not self.config.enable_m3:
            return 0.0
        if stv.service_category == "ambiguous" or stv.clarification_needed:
            return self.config.m3_weight_when_ambiguous
        return 1.0

    def resolve_prior_for_user(
        self,
        user_id: Optional[str],
        stv: Optional[StructuredTaskVariable] = None,
    ) -> Dict[str, float]:
        uniform = uniform_prior(self._hypotheses)
        if not self.config.enable_m3 or self._store is None:
            return uniform

        uid = user_id or self.config.user_id
        if not uid:
            return uniform

        profile = self._store.get_profile(uid)
        history_prior = compute_prior(profile, self._hypotheses, self.config.config_path)
        lam = self._m3_blend_lambda(stv) if stv is not None else 1.0
        return resolve_prior(self._hypotheses, history_prior, blend_lambda=lam)

    def _belief_update(
        self,
        stv: StructuredTaskVariable,
        prior: Dict[str, float],
        turn: int,
        current_belief: Optional[Dict[str, float]] = None,
    ) -> tuple[Dict[str, float], List[str]]:
        if turn == 0 or not current_belief:
            belief = dict(prior)
        else:
            belief = dict(current_belief)
        m2 = self.config.m2
        posterior = update_belief(
            belief,
            stv,
            prior,
            self._hypotheses,
            slot_evidence_power=m2.slot_evidence_power,
            power_decay=m2.belief_power_decay,
            recency_alpha=m2.belief_recency_alpha,
            prior_reinjection=m2.prior_reinjection,
        )
        top_ids = top_k_hypothesis_ids(posterior, turn)
        return posterior, top_ids

    def process_text(
        self,
        text: str,
        turn: int = 0,
        user_id: Optional[str] = None,
        current_belief: Optional[Dict[str, float]] = None,
        stv_history: Optional[List[StructuredTaskVariable]] = None,
    ) -> PipelineResult:
        stv = self._extractor.extract(text)
        m2 = self.config.m2
        # Hard lexical overrides only if explicitly enabled (not default).
        if m2.enable_domain_guards:
            stv = apply_domain_guards(
                stv,
                text,
                enabled=True,
                conf_override_max=m2.guard_confidence_cap,
            )
        # Soft: dampen confidence on cross-turn flips (never rewrites category).
        if m2.enable_conflict_dampen and stv_history:
            stv = dampen_conflicting_turn(
                stv_history,
                stv,
                enabled=True,
                factor=m2.conflict_dampen_factor,
            )
        # Multi-turn STV vote (label from history consensus only when current is weak).
        if m2.enable_m1_voting and stv_history:
            stv = vote_category_from_history(
                stv_history,
                stv,
                enabled=True,
                decay=m2.vote_recency_decay,
            )
        prior = self.resolve_prior_for_user(user_id or self.config.user_id, stv=stv)
        posterior, top_k_ids = self._belief_update(stv, prior, turn, current_belief)
        top1 = top1_probability(posterior)
        top1_cat = top1_category_probability(posterior, self._hypotheses)
        cat_ent = category_entropy(posterior, self._hypotheses)

        return PipelineResult(
            stv=stv,
            prior=prior,
            posterior=posterior,
            top_k_ids=top_k_ids,
            hypotheses=self._hypotheses,
            prior_entropy=entropy(prior),
            posterior_entropy=entropy(posterior),
            top1_prob=top1,
            top1_category_prob=top1_cat,
            category_entropy=cat_ent,
            converged=is_category_converged(
                posterior, self._hypotheses, threshold=self.config.converge_threshold
            ),
        )

    def prior_comparison(self, user_id: str) -> Dict[str, object]:
        uniform = uniform_prior(self._hypotheses)
        cfg = load_m3_config(self.config.config_path)
        if not self.config.enable_m3 or self._store is None:
            return {
                "uniform_entropy": entropy(uniform),
                "history_entropy": entropy(uniform),
                "user_id": user_id,
            }
        profile = self._store.get_profile(user_id)
        history = compute_prior(profile, self._hypotheses, self.config.config_path)
        blended = resolve_prior(self._hypotheses, history, blend_lambda=1.0)
        return {
            "user_id": user_id,
            "uniform_entropy": entropy(uniform),
            "history_entropy": entropy(blended),
            "profile_sessions": profile.total_sessions,
            "category_counts": dict(profile.category_counts),
            "m3_config": cfg,
        }
