from __future__ import annotations

from src.m2_reasoning.belief import entropy
from src.pipeline.config import PipelineConfig
from src.pipeline.session import SessionManager


def test_multiturn_belief_continues() -> None:
    mgr = SessionManager(PipelineConfig(enable_m3=False, hypothesis_max_size=80))
    session = mgr.new_session("u1")
    r1 = mgr.process_turn(session, "Book a quick workout class nearby tonight")
    e1 = entropy(r1.posterior)
    r2 = mgr.process_turn(session, "Actually make it this weekend instead")
    e2 = entropy(r2.posterior)
    assert session.turn == 2
    assert session.belief == r2.posterior
    assert r1.posterior != r2.posterior
