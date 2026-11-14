from __future__ import annotations

import re
from typing import List, Optional, TYPE_CHECKING

from src.dialogue.gpt import ChatCompletionProvider, MockChatProvider
from src.dialogue.keywords import is_short_followup
from src.dialogue.types import ChatMessage, DialogueTurnType

if TYPE_CHECKING:
    from src.pipeline.session import SessionState

_CHITCHAT_PATTERNS = (
    r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
    r"^how are you",
    r"^(thanks|thank you|thx)\b",
    r"^what can you do",
    r"^who are you",
)

_TASK_KEYWORDS = (
    "book",
    "booking",
    "need",
    "want",
    "looking for",
    "schedule",
    "appointment",
    "reserve",
    "spa",
    "massage",
    "fitness",
    "workout",
    "gym",
    "doctor",
    "medical",
    "beauty",
    "salon",
    "restaurant",
    "dining",
    "dinner",
    "lunch",
    "flight",
    "hotel",
    "travel",
    "airport",
    "relax",
    "weekend",
    "tonight",
    "tomorrow",
    "nearby",
    "near me",
    "class",
    "session",
    "nail",
    "hungry",
    "eat",
    "eating",
)

_ROUTER_SYSTEM = (
    "Classify the user message for a booking assistant. "
    "Reply with exactly one word: task or chitchat. "
    "task = user wants to find or book a service, or is answering a prior booking question. "
    "chitchat = greeting, small talk, or off-topic chat."
)


def _last_assistant_message(chat_history: List[ChatMessage]) -> Optional[str]:
    for msg in reversed(chat_history):
        if msg.role == "assistant":
            return msg.content
    return None


def _assistant_asked_question(chat_history: List[ChatMessage]) -> bool:
    last = _last_assistant_message(chat_history)
    if not last:
        return False
    lower = last.lower()
    return "?" in last or "would you" in lower or "are you looking" in lower


class DialogueRouter:
    def __init__(
        self,
        gpt_router_fallback: bool = True,
        enable_gpt: bool = True,
        model: str = "gpt-4o-mini",
        provider: Optional[ChatCompletionProvider] = None,
    ) -> None:
        self.gpt_router_fallback = gpt_router_fallback
        self.enable_gpt = enable_gpt
        self.model = model
        self._provider = provider

    def _provider_instance(self) -> ChatCompletionProvider:
        if self._provider is not None:
            return self._provider
        if self.enable_gpt:
            from src.dialogue.gpt import OpenAIChatProvider

            return OpenAIChatProvider(model=self.model)
        return MockChatProvider()

    def classify(
        self,
        user_input: str,
        chat_history: Optional[List[ChatMessage]] = None,
        session: Optional["SessionState"] = None,
    ) -> DialogueTurnType:
        text = user_input.strip()
        lower = text.lower()
        history = chat_history or []

        if not lower:
            return DialogueTurnType.CHITCHAT

        if session is not None:
            if session.mode == "TRANSITION":
                return DialogueTurnType.TASK
            if session.clarify_active or session.pending_confirmation:
                if is_short_followup(text):
                    return DialogueTurnType.TASK

        if history and is_short_followup(text) and _assistant_asked_question(history):
            return DialogueTurnType.TASK

        for pattern in _CHITCHAT_PATTERNS:
            if re.search(pattern, lower):
                return DialogueTurnType.CHITCHAT

        if any(kw in lower for kw in _TASK_KEYWORDS):
            return DialogueTurnType.TASK

        if self.gpt_router_fallback and self.enable_gpt:
            context_note = ""
            if history:
                last_asst = _last_assistant_message(history)
                if last_asst:
                    context_note = f"\nPrior assistant message: {last_asst[:200]}"
            reply = self._provider_instance().complete(
                _ROUTER_SYSTEM,
                [{"role": "user", "content": text + context_note}],
            )
            label = reply.strip().lower().split()[0] if reply.strip() else "task"
            if label.startswith("chitchat"):
                return DialogueTurnType.CHITCHAT

        return DialogueTurnType.TASK
