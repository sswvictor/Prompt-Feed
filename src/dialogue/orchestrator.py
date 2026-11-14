from __future__ import annotations

from typing import List, Optional

from src.dialogue.chitchat import generate_chitchat
from src.dialogue.clarify import (
    enter_clarify,
    exit_clarify_to_suggest,
    should_enter_clarify,
    top_categories_for_clarify,
    update_clarify_after_turn,
)
from src.dialogue.gpt import ChatCompletionProvider, get_chat_provider
from src.dialogue.intention_context import IntentionPromptBlock, build_intention_prompt_block
from src.dialogue.keywords import detect_primary_category, is_confirmation
from src.dialogue.router import DialogueRouter
from src.dialogue.task_reply import generate_observe_reply, generate_task_reply
from src.dialogue.transition import detect_hard_transition, open_transition, process_transition_turn
from src.dialogue.types import ChatMessage, DialogueTurnResult, DialogueTurnType
from src.m2_reasoning.convergence import top_k_hypothesis_ids
from src.m3_memory.writeback import record_session_outcome
from src.pipeline.config import DialogueConfig, PipelineConfig
from src.pipeline.session import SessionManager, SessionState


class DialogueOrchestrator:
    def __init__(
        self,
        config: PipelineConfig,
        session_manager: Optional[SessionManager] = None,
        dialogue_config: Optional[DialogueConfig] = None,
        chat_provider: Optional[ChatCompletionProvider] = None,
        inject_intention_block: Optional[bool] = None,
    ) -> None:
        self.config = config
        self.dialogue = dialogue_config or config.dialogue
        self.inject_intention_block = (
            inject_intention_block
            if inject_intention_block is not None
            else self.dialogue.inject_intention_block
        )
        self.session_manager = session_manager or SessionManager(config)
        self.chat_provider = get_chat_provider(
            enable_gpt=self.dialogue.enable_gpt,
            model=self.dialogue.model,
            provider=chat_provider,
        )
        self.router = DialogueRouter(
            gpt_router_fallback=self.dialogue.gpt_router_fallback,
            enable_gpt=self.dialogue.enable_gpt,
            model=self.dialogue.model,
            provider=self.chat_provider,
        )

    def handle_turn(
        self,
        session: SessionState,
        user_input: str,
        chat_history: Optional[List[ChatMessage]] = None,
    ) -> DialogueTurnResult:
        history = chat_history or []
        runner = self.session_manager.runner
        hypotheses = runner.hypotheses

        if detect_hard_transition(user_input):
            open_transition(session, user_input, hypotheses)

        route = self.router.classify(user_input, chat_history=history, session=session)

        if route == DialogueTurnType.CHITCHAT:
            session.surface_policy = "chitchat"
            message = generate_chitchat(
                user_input,
                enable_gpt=self.dialogue.enable_gpt,
                model=self.dialogue.model,
                provider=self.chat_provider,
            )
            return DialogueTurnResult(
                turn_type=DialogueTurnType.CHITCHAT,
                assistant_message=message,
                pipeline_result=None,
                intention_block=None,
                session_turn=session.turn,
                routed_as="chitchat",
                session_mode=session.mode,
                surface_policy=session.surface_policy,
            )

        clarify_prompt_type = ""
        transition_decision = None

        if session.mode == "TRANSITION":
            result, transition_decision = process_transition_turn(
                self.session_manager.runner, session, user_input
            )
            if transition_decision.action == "continue":
                session.surface_policy = "observe"
            elif transition_decision.action == "promote":
                session.surface_policy = "suggest"
            elif transition_decision.action == "abort":
                session.surface_policy = "suggest"
            elif transition_decision.action == "fallback_clarify":
                session.surface_policy = "clarify"
                clarify_prompt_type = "enumerate"
        else:
            result = self.session_manager.process_turn(session, user_input)

            if result.emotion_escape_triggered:
                enter_clarify(session, top_categories_for_clarify(result))
                session.surface_policy = "clarify"
                clarify_prompt_type = "enumerate"
            elif session.clarify_active or should_enter_clarify(
                session, result, force_clarify=session.force_clarify
            ):
                clarify_prompt_type = update_clarify_after_turn(session, result, user_input)
                session.surface_policy = "clarify" if clarify_prompt_type != "resolved" else "suggest"
                if clarify_prompt_type == "resolved":
                    clarify_prompt_type = ""
            elif session.pending_confirmation and is_confirmation(user_input):
                exit_clarify_to_suggest(session, session.pending_confirmation)
                session.surface_policy = "suggest"
            elif detect_primary_category(user_input):
                session.surface_policy = "suggest"
                session.active_category = detect_primary_category(user_input)
            else:
                from src.dialogue.intention_context import resolve_dialogue_policy

                knobs = getattr(session, "policy_knobs", {}) or {}
                pol = resolve_dialogue_policy(
                    result,
                    force_clarify=session.force_clarify,
                    top1_threshold=float(knobs.get("top1_suggest_threshold", 0.45)),
                    high_entropy_ratio=float(knobs.get("high_entropy_ratio", 0.95)),
                    clarify_turns_used=int(getattr(session, "clarify_turns_used", 0) or 0),
                    clarify_budget=int(knobs.get("clarify_budget", 2)),
                )
                if pol == "clarify" and not session.force_clarify:
                    session.clarify_turns_used += 1
                session.surface_policy = pol

        block = build_intention_prompt_block(
            result,
            session,
            user_input,
            inject_enabled=self.inject_intention_block,
            force_clarify=session.force_clarify and session.surface_policy == "clarify",
        )
        if session.surface_policy == "observe":
            block = IntentionPromptBlock(
                current_stv_summary=block.current_stv_summary,
                top_hypotheses=block.top_hypotheses,
                belief_state={**block.belief_state, "mode": "transition_observe"},
                dialogue_policy="observe",
                user_message=user_input,
                turn=session.turn,
                inject_enabled=block.inject_enabled,
            )
        elif session.surface_policy == "suggest":
            block = IntentionPromptBlock(
                current_stv_summary=block.current_stv_summary,
                top_hypotheses=block.top_hypotheses,
                belief_state=block.belief_state,
                dialogue_policy="suggest",
                user_message=user_input,
                turn=session.turn,
                inject_enabled=block.inject_enabled,
            )

        if session.surface_policy == "observe":
            message = generate_observe_reply(
                user_input,
                session.candidate_seed_category,
                history,
                enable_gpt=self.dialogue.enable_gpt,
                model=self.dialogue.model,
                provider=self.chat_provider,
                max_history_turns=self.dialogue.max_history_turns,
            )
            turn_type = DialogueTurnType.TASK
            routed_as = "transition_observe"
        else:
            turn_type = (
                DialogueTurnType.CLARIFY
                if session.surface_policy == "clarify"
                else DialogueTurnType.TASK
            )
            message = generate_task_reply(
                block,
                history,
                enable_gpt=self.dialogue.enable_gpt,
                model=self.dialogue.model,
                provider=self.chat_provider,
                max_history_turns=self.dialogue.max_history_turns,
                clarify_prompt_type=clarify_prompt_type,
                offered_categories=session.clarify_offered_categories,
                pending_category=session.pending_confirmation,
            )
            routed_as = "task_clarify" if turn_type == DialogueTurnType.CLARIFY else "task"

        return DialogueTurnResult(
            turn_type=turn_type,
            assistant_message=message,
            pipeline_result=result,
            intention_block=block,
            session_turn=session.turn,
            routed_as=routed_as,
            session_mode=session.mode,
            surface_policy=session.surface_policy,
        )

    def end_session_with_outcome(self, session: SessionState) -> bool:
        runner = self.session_manager.runner
        store = runner.store
        if store is None or not session.belief:
            return False
        turn_idx = max(0, session.turn - 1)
        top_ids = top_k_hypothesis_ids(session.belief, turn_idx)
        if not top_ids:
            return False
        hypothesis = runner.hypothesis_by_id(top_ids[0])
        if hypothesis is None:
            return False
        record_session_outcome(store, session.user_id, hypothesis)
        return True
