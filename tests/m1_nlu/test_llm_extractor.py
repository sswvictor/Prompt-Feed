from __future__ import annotations

from src.m1_nlu.llm_extractor import LLMExtractor
from src.m1_nlu.llm_provider import LLMProvider


class _NullConfidenceProvider(LLMProvider):
    def extract_json(self, user_input: str) -> dict[str, object]:
        return {
            "service_category": "wellness",
            "category_confidence": None,
            "extraction_confidence": 0.72,
        }


class _MissingConfidenceProvider(LLMProvider):
    def extract_json(self, user_input: str) -> dict[str, object]:
        return {"service_category": "fitness"}


def test_llm_extractor_falls_back_when_category_confidence_none() -> None:
    ext = LLMExtractor(provider=_NullConfidenceProvider())
    stv = ext.extract("spa tonight")
    assert stv.service_category == "wellness"
    assert stv.category_confidence == 0.72


def test_llm_extractor_default_confidence_when_missing() -> None:
    ext = LLMExtractor(provider=_MissingConfidenceProvider())
    stv = ext.extract("gym class")
    assert stv.service_category == "fitness"
    assert stv.category_confidence == 0.35
