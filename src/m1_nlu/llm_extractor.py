from __future__ import annotations

from src.m1_nlu.llm_provider import LLMProvider, OpenAICompatProvider
from src.m1_nlu.types import StructuredTaskVariable, normalize_stv


class LLMExtractor:
    # JSON-mode extraction with provider abstraction + fallback.
    def __init__(self, provider: LLMProvider | None = None) -> None:
        self.provider = provider or OpenAICompatProvider()

    def extract(self, user_input: str) -> StructuredTaskVariable:
        payload = self.provider.extract_json(user_input)
        category = str(payload.get("service_category", "other"))
        raw_conf = payload.get("category_confidence")
        if raw_conf is None:
            raw_conf = payload.get("extraction_confidence", 0.35)
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = 0.35

        stv = StructuredTaskVariable(
            service_category=category,
            category_confidence=conf,
            temporal_preference=payload.get("temporal_preference"),
            urgency_level=str(payload.get("urgency_level", "medium")),
            time_window_days=1 if str(payload.get("urgency_level", "medium")) == "high" else 7,
            location_context=payload.get("location_context"),
            sentiment_tags=[str(x) for x in payload.get("sentiment_tags", [])],
            extraction_confidence=conf,
            raw_input=user_input,
            clarification_needed=False,
        )
        return normalize_stv(stv)
