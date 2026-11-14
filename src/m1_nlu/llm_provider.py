from __future__ import annotations

import json
import os
from typing import Dict

from src.config.env import load_project_env


class LLMProvider:
    def extract_json(self, user_input: str) -> Dict[str, object]:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def extract_json(self, user_input: str) -> Dict[str, object]:
        text = user_input.lower()
        category = "other"
        if "fitness" in text or "workout" in text:
            category = "fitness"
        elif "relax" in text or "spa" in text:
            category = "wellness"
        elif "beauty" in text:
            category = "beauty"
        elif "doctor" in text or "medical" in text:
            category = "medical"
        elif "restaurant" in text or "dining" in text or "dinner" in text:
            category = "dining"
        elif "flight" in text or "hotel" in text or "travel" in text:
            category = "travel"
        return {
            "service_category": category,
            "temporal_preference": "weekend" if "weekend" in text else None,
            "urgency_level": "high" if ("urgent" in text or "asap" in text) else "medium",
            "location_context": "nearby" if ("nearby" in text or "near me" in text) else None,
            "sentiment_tags": [x for x in ["relaxing", "affordable", "quick"] if x in text],
            "category_confidence": 0.86 if category != "other" else 0.35,
        }


class OpenAICompatProvider(LLMProvider):
    """
    Optional provider. Expects OPENAI_API_KEY.
    Uses mock fallback when key not present or package unavailable.
    """

    def __init__(self) -> None:
        load_project_env()
        self.api_key = os.getenv("OPENAI_API_KEY", "")

    def extract_json(self, user_input: str) -> Dict[str, object]:
        if not self.api_key:
            return MockLLMProvider().extract_json(user_input)
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return MockLLMProvider().extract_json(user_input)

        client = OpenAI(api_key=self.api_key)
        prompt = (
            "Extract booking intent from user input as JSON. "
            "Return ONLY valid JSON with keys: service_category, temporal_preference, urgency_level, "
            "location_context, sentiment_tags, category_confidence. "
            f"User input: {user_input}"
        )
        resp = client.responses.create(model="gpt-4o-mini", input=prompt)
        text = resp.output_text
        try:
            data = json.loads(text)
        except Exception:
            return MockLLMProvider().extract_json(user_input)
        return data
