from __future__ import annotations

from src.m1_nlu.factory import get_extractor
from src.m1_nlu.llm_extractor import LLMExtractor
from src.m1_nlu.llm_provider import MockLLMProvider
from src.m1_nlu.types import StructuredTaskVariable, normalize_stv


def test_low_confidence_preserves_predicted_category() -> None:
    stv = StructuredTaskVariable(
        service_category="fitness",
        category_confidence=0.2,
        temporal_preference=None,
        urgency_level="medium",
        time_window_days=7,
        location_context=None,
        sentiment_tags=[],
        extraction_confidence=0.3,
        raw_input="something",
    )
    out = normalize_stv(stv)
    assert out.service_category == "fitness"
    assert out.clarification_needed is False


def test_pipeline_extractor_returns_stv() -> None:
    ext = get_extractor("pipeline")
    out = ext.extract("I need fitness workout nearby")
    assert out.service_category in {"fitness", "other", "ambiguous"}
    assert 0.0 <= out.extraction_confidence <= 1.0
    assert out.raw_input


def test_llm_extractor_returns_stv() -> None:
    ext = LLMExtractor(provider=MockLLMProvider())
    out = ext.extract("I need a relaxing spa")
    assert out.service_category in {"wellness", "other", "ambiguous"}
    assert 0.0 <= out.category_confidence <= 1.0
