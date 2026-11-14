from __future__ import annotations

from src.m1_nlu.llm_provider import MockLLMProvider


def test_mock_llm_provider_output_contract() -> None:
    p = MockLLMProvider()
    out = p.extract_json("I need relaxing spa nearby this weekend")
    assert "service_category" in out
    assert "category_confidence" in out
    assert isinstance(out.get("sentiment_tags"), list)
