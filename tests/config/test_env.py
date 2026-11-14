from __future__ import annotations

from src.config.env import apply_env_overrides, env_bool, env_int, env_str, gpt_connection_status
from src.pipeline.config import DialogueConfig, PipelineConfig


def test_env_str_bool_int_helpers(monkeypatch) -> None:
    monkeypatch.delenv("TEST_ENV_STR", raising=False)
    assert env_str("TEST_ENV_STR") is None

    monkeypatch.setenv("TEST_ENV_STR", "  hello  ")
    assert env_str("TEST_ENV_STR") == "hello"

    monkeypatch.setenv("TEST_ENV_BOOL", "true")
    assert env_bool("TEST_ENV_BOOL") is True
    monkeypatch.setenv("TEST_ENV_BOOL", "off")
    assert env_bool("TEST_ENV_BOOL") is False

    monkeypatch.setenv("TEST_ENV_INT", "42")
    assert env_int("TEST_ENV_INT") == 42
    monkeypatch.setenv("TEST_ENV_INT", "not-a-number")
    assert env_int("TEST_ENV_INT") is None


def test_apply_env_overrides_dialogue(monkeypatch) -> None:
    monkeypatch.setenv("DIALOGUE_MODEL", "gpt-4o")
    monkeypatch.setenv("DIALOGUE_ENABLE_GPT", "false")
    monkeypatch.setenv("DIALOGUE_INJECT_INTENTION_BLOCK", "false")
    monkeypatch.setenv("DIALOGUE_MAX_HISTORY_TURNS", "4")
    monkeypatch.setenv("PIPELINE_EXTRACTOR_MODE", "llm")
    monkeypatch.setenv("PIPELINE_HYPOTHESIS_MAX_SIZE", "120")
    monkeypatch.setenv("M3_MEMORY_BACKEND", "sqlite")

    base = PipelineConfig(dialogue=DialogueConfig())
    cfg = apply_env_overrides(base)

    assert cfg.dialogue.model == "gpt-4o"
    assert cfg.dialogue.enable_gpt is False
    assert cfg.dialogue.inject_intention_block is False
    assert cfg.dialogue.max_history_turns == 4
    assert cfg.extractor_mode == "llm"
    assert cfg.hypothesis_max_size == 120
    assert cfg.memory_backend == "sqlite"


def test_gpt_connection_status_with_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    status = gpt_connection_status(model="gpt-4o-mini", enable_gpt=True)
    assert status["connected"] is True
    assert status["mode"] == "gpt"
    assert status["model"] == "gpt-4o-mini"


def test_gpt_connection_status_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    status = gpt_connection_status(model="gpt-4o-mini", enable_gpt=True)
    assert status["connected"] is False
    assert status["mode"] == "mock"
    assert "OPENAI_API_KEY" in (status["reason"] or "")


def test_gpt_connection_status_disabled(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    status = gpt_connection_status(model="gpt-4o-mini", enable_gpt=False)
    assert status["connected"] is False
    assert status["mode"] == "mock"
