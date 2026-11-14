from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.dialogue.intention_context import build_intention_prompt_block
from src.m1_nlu.sentiment import is_negative_turn, reset_analyzer, score_sentiment
from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.belief import apply_category_penalty, top1_category
from src.m2_reasoning.hypothesis import GoalHypothesis, generate_hypothesis_space
from src.pipeline.config import PipelineConfig, SentimentConfig
from src.pipeline.runner import PipelineResult
from src.pipeline.session import SessionManager, SessionState


def _mock_compound(value: float):
    analyzer = MagicMock()
    analyzer.polarity_scores.return_value = {"compound": value}
    return analyzer


def test_score_sentiment_returns_compound_in_range() -> None:
    reset_analyzer()
    with patch("src.m1_nlu.sentiment._get_analyzer", return_value=_mock_compound(0.42)):
        assert score_sentiment("great service") == pytest.approx(0.42)


def test_is_negative_turn_respects_threshold() -> None:
    reset_analyzer()
    with patch("src.m1_nlu.sentiment._get_analyzer", return_value=_mock_compound(-0.5)):
        assert is_negative_turn("this is terrible") is True
        assert is_negative_turn("this is terrible", threshold=-0.6) is False


def test_apply_category_penalty_reduces_top_category_mass() -> None:
    hyps = [
        GoalHypothesis("h1", "fitness", "gym", "evening", "nearby", "mid"),
        GoalHypothesis("h2", "wellness", "spa", "evening", "nearby", "mid"),
    ]
    belief = {"h1": 0.55, "h2": 0.45}
    adjusted = apply_category_penalty(belief, hyps, "fitness", penalty=0.5, other_bonus=1.1)
    assert adjusted["h1"] < belief["h1"]
    assert adjusted["h2"] > belief["h2"]
    assert top1_category(adjusted, hyps) == "wellness"


def test_build_intention_prompt_block_force_clarify() -> None:
    hyps = generate_hypothesis_space(max_size=20)
    posterior = {h.hypothesis_id: 1.0 / len(hyps) for h in hyps}
    stv = StructuredTaskVariable(
        service_category="fitness",
        category_confidence=0.9,
        temporal_preference="evening",
        urgency_level="medium",
        time_window_days=None,
        location_context="nearby",
        sentiment_tags=[],
        extraction_confidence=0.9,
        raw_input="book gym",
        clarification_needed=False,
    )
    result = PipelineResult(
        stv=stv,
        prior=posterior,
        posterior=posterior,
        top_k_ids=list(posterior.keys())[:3],
        hypotheses=hyps,
        top1_prob=0.9,
        top1_category_prob=0.9,
        category_entropy=0.5,
        converged=True,
    )
    session = SessionState(session_id="s1", user_id="u1", turn=2)
    block = build_intention_prompt_block(result, session, "wrong suggestion", force_clarify=True)
    assert block.dialogue_policy == "clarify"


def test_emotion_escape_triggers_after_consecutive_negative_turns() -> None:
    reset_analyzer()
    cfg = PipelineConfig(
        enable_m3=False,
        hypothesis_max_size=80,
        sentiment=SentimentConfig(consecutive_negative_turns=2, negative_threshold=-0.3),
    )
    mgr = SessionManager(cfg)
    session = mgr.new_session("u1")

    with patch("src.m1_nlu.sentiment._get_analyzer", return_value=_mock_compound(-0.5)):
        r1 = mgr.process_turn(session, "Book a gym class nearby")
        assert r1.emotion_escape_triggered is False
        assert session.consecutive_negative_turns == 1
        assert session.force_clarify is False

        r2 = mgr.process_turn(session, "No that's wrong, terrible suggestion")
        assert session.consecutive_negative_turns == 2
        assert session.force_clarify is True
        assert r2.emotion_escape_triggered is True

        block = build_intention_prompt_block(r2, session, "No that's wrong", force_clarify=session.force_clarify)
        assert block.dialogue_policy == "clarify"


def test_positive_turn_resets_negative_streak() -> None:
    reset_analyzer()
    cfg = PipelineConfig(enable_m3=False, hypothesis_max_size=80)
    mgr = SessionManager(cfg)
    session = mgr.new_session("u1")

    with patch("src.m1_nlu.sentiment._get_analyzer") as mock_get:
        mock_get.return_value = _mock_compound(-0.5)
        mgr.process_turn(session, "This is awful")
        assert session.consecutive_negative_turns == 1

        mock_get.return_value = _mock_compound(0.6)
        mgr.process_turn(session, "Actually thanks, gym is fine")
        assert session.consecutive_negative_turns == 0
        assert session.force_clarify is False
