from __future__ import annotations

import pytest

from src.dialogue.keywords import detect_primary_category, is_confirmation
from src.dialogue.router import DialogueRouter
from src.dialogue.transition import detect_hard_transition
from src.dialogue.types import DialogueTurnType
from src.pipeline.config import PipelineConfig
from src.pipeline.session import SessionManager, SessionState


def test_hard_transition_forget_it_nail() -> None:
    assert detect_hard_transition("Forget it, maybe I want to fix my nail") is True


def test_hard_transition_instead() -> None:
    assert detect_hard_transition("Let's do weekend instead") is True


def test_no_hard_transition_casual() -> None:
    assert detect_hard_transition("How are you doing today?") is False
    assert detect_hard_transition("something vague maybe") is False


def test_keyword_detect_nail() -> None:
    assert detect_primary_category("fix my nail") == "beauty"


def test_router_yes_after_clarify_context() -> None:
    router = DialogueRouter(gpt_router_fallback=False, enable_gpt=False)
    session = SessionState(session_id="s", user_id="u", clarify_active=True)
    assert router.classify("Yes", session=session) == DialogueTurnType.TASK


def test_router_transition_always_task() -> None:
    router = DialogueRouter(gpt_router_fallback=False, enable_gpt=False)
    session = SessionState(session_id="s", user_id="u", mode="TRANSITION")
    assert router.classify("yes", session=session) == DialogueTurnType.TASK


def test_confirmation() -> None:
    assert is_confirmation("Yes") is True
    assert is_confirmation("对") is True


def test_open_transition_sets_mode() -> None:
    from src.dialogue.transition import open_transition
    from src.pipeline.runner import PipelineRunner

    cfg = PipelineConfig()
    runner = PipelineRunner(cfg)
    session = SessionManager(cfg, runner=runner).new_session("u")
    seed = open_transition(session, "Forget it, nail instead", runner.hypotheses)
    assert session.mode == "TRANSITION"
    assert seed == "beauty"
    assert session.clarify_active is False
