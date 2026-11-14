from __future__ import annotations

from src.pipeline.config import pipeline_config_from_yaml


def test_yaml_loads_dialogue_section() -> None:
    cfg = pipeline_config_from_yaml()
    assert cfg.dialogue.enable_gpt is True
    assert cfg.dialogue.model == "gpt-4o-mini"
    assert cfg.dialogue.max_history_turns == 6
    assert cfg.dialogue.gpt_router_fallback is True
