from __future__ import annotations

import os
from typing import List, Protocol

from src.config.env import load_project_env


class ChatCompletionProvider(Protocol):
    def complete(self, system: str, messages: List[dict[str, str]]) -> str:
        ...


class MockChatProvider:
    """Deterministic replies for tests and offline fallback."""

    def complete(self, system: str, messages: List[dict[str, str]]) -> str:
        last_user = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "").lower()
                break
        if "clarify" in system.lower() or "clarification" in system.lower():
            if "relax" in last_user:
                return "Would you prefer a spa, massage, or another wellness service?"
            return "Could you tell me which type of service you are looking for?"
        if "hello" in last_user or "hi" in last_user:
            return "Hello! I can help you book fitness, wellness, beauty, medical, dining, or travel services. What would you like?"
        return "I'm here to help with bookings. What service are you looking for?"


class OpenAIChatProvider:
    """OpenAI-compatible chat completion; falls back to MockChatProvider."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        load_project_env()
        self.model = model
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        self._fallback = MockChatProvider()

    def complete(self, system: str, messages: List[dict[str, str]]) -> str:
        if not self.api_key:
            return self._fallback.complete(system, messages)
        try:
            from openai import OpenAI  # type: ignore
        except Exception:
            return self._fallback.complete(system, messages)

        client = OpenAI(api_key=self.api_key)
        payload = [{"role": "system", "content": system}, *messages]
        try:
            resp = client.responses.create(model=self.model, input=payload)
            text = (resp.output_text or "").strip()
            return text or self._fallback.complete(system, messages)
        except Exception:
            return self._fallback.complete(system, messages)


def get_chat_provider(
    enable_gpt: bool = True,
    model: str = "gpt-4o-mini",
    provider: ChatCompletionProvider | None = None,
) -> ChatCompletionProvider:
    if provider is not None:
        return provider
    if enable_gpt:
        return OpenAIChatProvider(model=model)
    return MockChatProvider()
