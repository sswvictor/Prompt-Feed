from __future__ import annotations

from src.m2_reasoning.convergence import incorporate_feedback, is_converged, k_schedule, top_k_hypothesis_ids


def test_k_schedule() -> None:
    assert k_schedule(0) == 10
    assert k_schedule(1) == 6
    assert k_schedule(2) == 4
    assert k_schedule(3) == 3
    assert k_schedule(9) == 3


def test_feedback_boosts_selected() -> None:
    belief = {"h001": 0.4, "h002": 0.3, "h003": 0.3}
    out = incorporate_feedback(belief, "h002", boost=1.5, decay=0.8)
    assert out["h002"] > belief["h002"]


def test_top_k_and_convergence() -> None:
    belief = {"h001": 0.9, "h002": 0.05, "h003": 0.05}
    top = top_k_hypothesis_ids(belief, turn=0)
    assert top[0] == "h001"
    assert is_converged(belief, threshold=0.85)
