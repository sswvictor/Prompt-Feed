from __future__ import annotations

from typing import List, Optional

from src.dialogue.gpt import ChatCompletionProvider, MockChatProvider
from src.dialogue.intention_context import IntentionPromptBlock
from src.dialogue.keywords import CATEGORY_SYNONYMS, extract_user_keywords
from src.dialogue.types import ChatMessage

_CLARIFY_SYSTEM = (
    "You are a helpful booking assistant for fitness, wellness, beauty, medical, dining, and travel services. "
    "The user's intent is still unclear. Ask exactly one short clarifying question in English. "
    "Do not invent specific bookable services, prices, or availability. "
    "Do not reference stored user preferences or history."
)

_CLARIFY_ENUMERATE_SYSTEM = (
    "You are a helpful booking assistant. The user's intent is unclear. "
    "Ask which service category they want. Mention ONLY these options: {options}. "
    "One short question, English, no bullet list longer than 3 items."
)

_CLARIFY_CONFIRM_SYSTEM = (
    "You are a helpful booking assistant. Briefly confirm whether the user wants {category} services. "
    "One short sentence ending with a yes/no style question. English only."
)

_OBSERVE_SYSTEM = (
    "You are a helpful booking assistant. The user may be shifting topic. "
    "Reply in 2-3 English sentences. Include the user keywords: {user_keywords}. "
    "Gently mention {category_hint} without interrogating. "
    "Do NOT ask 'which do you mean', do NOT force A/B/C choice, do NOT say 'please confirm'. "
    "End with an open invitation to continue."
)

_SUGGEST_SYSTEM = (
    "You are a helpful booking assistant for fitness, wellness, beauty, medical, dining, and travel services. "
    "Use the intention context to offer concise, prioritized suggestions in natural language (2-4 sentences). "
    "You may list up to 3 options as bullet points. "
    "Do not invent specific business names unless implied by context. "
    "Do not reference stored user preferences or raw history."
)

_NO_INTENT_SYSTEM = (
    "You are a helpful booking assistant. Respond naturally in 2-4 sentences. "
    "Ask a clarifying question if the request is vague."
)


def _history_to_messages(chat_history: List[ChatMessage], max_turns: int) -> List[dict[str, str]]:
    recent = chat_history[-(max_turns * 2) :]
    return [{"role": msg.role, "content": msg.content} for msg in recent]


def _complete(
    system: str,
    messages: List[dict[str, str]],
    *,
    enable_gpt: bool,
    model: str,
    provider: Optional[ChatCompletionProvider],
) -> str:
    if provider is not None:
        chat_provider = provider
    elif enable_gpt:
        from src.dialogue.gpt import OpenAIChatProvider

        chat_provider = OpenAIChatProvider(model=model)
    else:
        chat_provider = MockChatProvider()
    try:
        reply = chat_provider.complete(system, messages).strip()
        if reply:
            return reply
    except Exception:
        pass
    return MockChatProvider().complete(system, messages)


def generate_task_reply(
    block: IntentionPromptBlock,
    chat_history: List[ChatMessage],
    *,
    enable_gpt: bool = True,
    model: str = "gpt-4o-mini",
    provider: Optional[ChatCompletionProvider] = None,
    max_history_turns: int = 6,
    clarify_prompt_type: str = "",
    offered_categories: Optional[List[str]] = None,
    pending_category: Optional[str] = None,
) -> str:
    if clarify_prompt_type == "enumerate" and offered_categories:
        base_system = _CLARIFY_ENUMERATE_SYSTEM.format(options=", ".join(offered_categories))
    elif clarify_prompt_type == "confirm" and pending_category:
        base_system = _CLARIFY_CONFIRM_SYSTEM.format(category=pending_category)
    elif block.dialogue_policy == "clarify":
        base_system = _CLARIFY_SYSTEM
    else:
        base_system = _SUGGEST_SYSTEM

    system = (
        f"{base_system}\n\n{block.to_prompt_text()}"
        if block.inject_enabled
        else _NO_INTENT_SYSTEM
    )

    messages = _history_to_messages(chat_history, max_history_turns)
    messages.append({"role": "user", "content": block.user_message})
    return _complete(system, messages, enable_gpt=enable_gpt, model=model, provider=provider)


def generate_observe_reply(
    user_message: str,
    candidate_category: Optional[str],
    chat_history: List[ChatMessage],
    *,
    enable_gpt: bool = True,
    model: str = "gpt-4o-mini",
    provider: Optional[ChatCompletionProvider] = None,
    max_history_turns: int = 6,
) -> str:
    cat = candidate_category or "services"
    hint = CATEGORY_SYNONYMS.get(cat, cat)
    keywords = ", ".join(extract_user_keywords(user_message, candidate_category)) or "your request"
    system = _OBSERVE_SYSTEM.format(user_keywords=keywords, category_hint=hint)
    messages = _history_to_messages(chat_history, max_history_turns)
    messages.append({"role": "user", "content": user_message})
    reply = _complete(system, messages, enable_gpt=enable_gpt, model=model, provider=provider)
    if not enable_gpt or not reply:
        return (
            f"If you're thinking about {hint}, we can explore options around {keywords}. "
            f"Feel free to share a bit more whenever you're ready."
        )
    return reply
