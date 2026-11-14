from __future__ import annotations

from typing import Optional

from src.dialogue.gpt import ChatCompletionProvider, MockChatProvider

_CHITCHAT_SYSTEM = (
    "You are a friendly booking assistant. Reply briefly in English (1-3 sentences). "
    "If the user is off-topic, politely steer them toward booking fitness, wellness, "
    "beauty, medical, dining, or travel services. Do not claim to have made a booking."
)

_TEMPLATE_REPLIES = {
    "hello": "Hello! I can help you find and book local services. What are you looking for today?",
    "thanks": "You're welcome! Let me know if you'd like help booking a service.",
    "how are you": "I'm doing well, thank you! How can I help you with a booking today?",
}


def _template_chitchat(user_input: str) -> str:
    lower = user_input.lower().strip()
    for key, reply in _TEMPLATE_REPLIES.items():
        if key in lower:
            return reply
    return "I'm here to help with bookings. What service would you like to find?"


def generate_chitchat(
    user_input: str,
    *,
    enable_gpt: bool = True,
    model: str = "gpt-4o-mini",
    provider: Optional[ChatCompletionProvider] = None,
) -> str:
    if not enable_gpt:
        return _template_chitchat(user_input)

    chat_provider = provider
    if chat_provider is None:
        from src.dialogue.gpt import OpenAIChatProvider

        chat_provider = OpenAIChatProvider(model=model)

    try:
        reply = chat_provider.complete(
            _CHITCHAT_SYSTEM,
            [{"role": "user", "content": user_input}],
        ).strip()
        if reply:
            return reply
    except Exception:
        pass

    return MockChatProvider().complete(
        _CHITCHAT_SYSTEM,
        [{"role": "user", "content": user_input}],
    )
