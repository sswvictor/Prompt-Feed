from __future__ import annotations



from src.dialogue.gpt import MockChatProvider

from src.dialogue.orchestrator import DialogueOrchestrator

from src.dialogue.types import DialogueTurnType

from src.pipeline.config import DialogueConfig, PipelineConfig

from src.pipeline.session import SessionManager





def _orchestrator(inject: bool = True) -> DialogueOrchestrator:

    cfg = PipelineConfig(

        enable_m3=False,

        hypothesis_max_size=540,

        dialogue=DialogueConfig(

            enable_gpt=False,

            gpt_router_fallback=False,

            inject_intention_block=inject,

        ),

    )

    mgr = SessionManager(cfg)

    return DialogueOrchestrator(cfg, session_manager=mgr, chat_provider=MockChatProvider())





def test_chitchat_does_not_increment_turn() -> None:

    orch = _orchestrator()

    session = orch.session_manager.new_session("u1")

    result = orch.handle_turn(session, "hello!")

    assert result.turn_type == DialogueTurnType.CHITCHAT

    assert session.turn == 0

    assert result.pipeline_result is None





def test_multiturn_task_updates_belief() -> None:

    orch = _orchestrator()

    session = orch.session_manager.new_session("u1")

    r1 = orch.handle_turn(session, "Book a quick workout class nearby tonight")

    r2 = orch.handle_turn(session, "weekend yoga class please")

    assert r1.turn_type in (DialogueTurnType.TASK, DialogueTurnType.CLARIFY)

    assert r2.turn_type in (DialogueTurnType.TASK, DialogueTurnType.CLARIFY)

    assert session.turn == 2

    assert r1.pipeline_result is not None

    assert r2.pipeline_result is not None

    assert r1.pipeline_result.posterior != r2.pipeline_result.posterior

    assert r1.intention_block is not None





def test_ambiguous_input_uses_intention_block() -> None:

    orch = _orchestrator()

    session = orch.session_manager.new_session("u1")

    result = orch.handle_turn(session, "something vague")

    assert result.turn_type == DialogueTurnType.CLARIFY

    assert result.pipeline_result is not None

    assert result.intention_block is not None

    assert result.intention_block.dialogue_policy == "clarify"

    assert session.turn == 1

    assert "?" in result.assistant_message or len(result.assistant_message) > 0





def test_without_intention_block_still_replies() -> None:

    orch = _orchestrator(inject=False)

    session = orch.session_manager.new_session("u1")

    result = orch.handle_turn(session, "I need a relaxing spa this weekend")

    assert result.intention_block is not None

    assert result.intention_block.inject_enabled is False

    assert result.assistant_message


