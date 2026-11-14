from __future__ import annotations

from typing import Dict, List

from src.m2_reasoning.belief import category_masses, top1_category_probability
from src.m2_reasoning.hypothesis import GoalHypothesis


def k_schedule(turn: int) -> int:
    table = {0: 10, 1: 6, 2: 4, 3: 3}
    return table.get(turn, 3)


def top_k_hypothesis_ids(belief: Dict[str, float], turn: int) -> List[str]:
    k = k_schedule(turn)
    sorted_ids = sorted(belief.items(), key=lambda x: x[1], reverse=True)
    return [h_id for h_id, _ in sorted_ids[:k]]


def top1_probability(belief: Dict[str, float]) -> float:
    return max(belief.values()) if belief else 0.0


def is_converged(belief: Dict[str, float], threshold: float = 0.85) -> bool:
    return top1_probability(belief) > threshold


def is_category_converged(
    belief: Dict[str, float],
    hypotheses: List[GoalHypothesis],
    threshold: float = 0.55,
) -> bool:
    return top1_category_probability(belief, hypotheses) >= threshold


def incorporate_feedback(
    belief: Dict[str, float],
    selected_hypothesis_id: str,
    boost: float = 1.5,
    decay: float = 0.8,
) -> Dict[str, float]:
    updated: Dict[str, float] = {}
    for h_id, p in belief.items():
        updated[h_id] = p * boost if h_id == selected_hypothesis_id else p * decay
    total = sum(updated.values()) or 1.0
    return {k: v / total for k, v in updated.items()}
