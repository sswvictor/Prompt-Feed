from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from src.dialogue.intention_context import IntentionPromptBlock
from src.pipeline.runner import PipelineResult


class DialogueTurnType(str, Enum):
    TASK = "task"
    CLARIFY = "clarify"
    CHITCHAT = "chitchat"


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str
    turn_type: Optional[DialogueTurnType] = None
    user_query: Optional[str] = None
    pipeline_result: Optional[PipelineResult] = field(default=None, repr=False)
    intention_block: Optional[IntentionPromptBlock] = field(default=None, repr=False)
    session_mode: str = "TRACKING"
    surface_policy: str = "suggest"


@dataclass
class DialogueTurnResult:
    turn_type: DialogueTurnType
    assistant_message: str
    pipeline_result: Optional[PipelineResult] = None
    intention_block: Optional[IntentionPromptBlock] = None
    session_turn: int = 0
    routed_as: str = ""
    session_mode: str = "TRACKING"
    surface_policy: str = "suggest"

    def to_chat_messages(self, user_input: str) -> List[ChatMessage]:
        return [
            ChatMessage(role="user", content=user_input),
            ChatMessage(
                role="assistant",
                content=self.assistant_message,
                turn_type=self.turn_type,
                user_query=user_input,
                pipeline_result=self.pipeline_result,
                intention_block=self.intention_block,
                session_mode=self.session_mode,
                surface_policy=self.surface_policy,
            ),
        ]
