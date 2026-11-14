from __future__ import annotations

from src.dialogue.gpt import MockChatProvider
from src.dialogue.router import DialogueRouter
from src.dialogue.types import DialogueTurnType


def test_router_chitchat_greeting() -> None:
    router = DialogueRouter(gpt_router_fallback=False, enable_gpt=False)
    assert router.classify("hello there") == DialogueTurnType.CHITCHAT
    assert router.classify("How are you?") == DialogueTurnType.CHITCHAT
    assert router.classify("thanks!") == DialogueTurnType.CHITCHAT


def test_router_task_booking() -> None:
    router = DialogueRouter(gpt_router_fallback=False, enable_gpt=False)
    assert router.classify("I need a relaxing spa nearby this weekend") == DialogueTurnType.TASK
    assert router.classify("Book a quick workout class tonight") == DialogueTurnType.TASK


def test_router_gpt_fallback() -> None:
    class ChitchatOnlyProvider(MockChatProvider):
        def complete(self, system: str, messages):  # type: ignore[no-untyped-def]
            return "chitchat"

    router = DialogueRouter(
        gpt_router_fallback=True,
        enable_gpt=True,
        provider=ChitchatOnlyProvider(),
    )
    assert router.classify("something vague maybe") == DialogueTurnType.CHITCHAT
