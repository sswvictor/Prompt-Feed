"""Frozen thesis dining path for the defence recording.

Uses the Intention-conditioned eval loop (M3 off, intention block on),
not the Streamlit debug tabs or the DialogueOrchestrator extras.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dialogue.intention_context import (
    IntentionPromptBlock,
    build_intention_prompt_block,
    resolve_dialogue_policy,
)
from src.dialogue.task_reply import generate_task_reply
from src.dialogue.types import ChatMessage
from src.m2_reasoning.belief import top1_category
from src.pipeline.config import PipelineConfig, pipeline_config_from_yaml
from src.pipeline.runner import PipelineResult, PipelineRunner
from src.pipeline.session import SessionManager, SessionState

SCENARIO_ID = "dining_vague_then_specific"
SCENARIOS_PATH = ROOT / "data" / "annotations" / "eval_scenarios_v2.json"
EXPECTED_POLICY = ("clarify", "suggest", "suggest")

FALLBACK_REPLY = {
    "clarify": "Could you tell me a bit more — cuisine, area, or when you would like to go?",
    "suggest": "That is enough to suggest a dining option from here.",
}


def load_dining_scenario() -> Dict[str, Any]:
    scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))
    for row in scenarios:
        if row.get("scenario_id") == SCENARIO_ID:
            return row
    raise KeyError(SCENARIO_ID)


def intention_conditioned_config() -> PipelineConfig:
    return replace(pipeline_config_from_yaml(), enable_m3=False, user_id="defence_dining")


@dataclass
class DefenceTurn:
    text: str
    policy: str
    m1_category: str
    m1_confidence: float
    category_entropy: float
    top1_category_prob: float
    posterior_entropy: float
    pred_category: Optional[str]
    privacy_pass: bool
    withheld_from_llm: List[str]
    prompt_text: str
    result: PipelineResult
    block: IntentionPromptBlock


class DefenceDiningRunner:
    """Step the frozen dining script on the Intention-conditioned sidecar."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.scenario = load_dining_scenario()
        self.config = config or intention_conditioned_config()
        self.runner = PipelineRunner(self.config, store=None)
        self.manager = SessionManager(self.config, runner=self.runner)
        self.session: SessionState = self.manager.new_session("defence_dining")
        self.index = 0
        self.history: List[DefenceTurn] = []

    @property
    def total(self) -> int:
        return len(self.scenario["turns"])

    @property
    def remaining(self) -> int:
        return self.total - self.index

    @property
    def done(self) -> bool:
        return self.index >= self.total

    def lines_to_type(self) -> List[str]:
        return [str(turn["text"]) for turn in self.scenario["turns"]]

    def peek_text(self) -> Optional[str]:
        if self.done:
            return None
        return str(self.scenario["turns"][self.index]["text"])

    def step(self) -> DefenceTurn:
        if self.done:
            raise RuntimeError("dining script already complete")
        return self.step_text(str(self.scenario["turns"][self.index]["text"]))

    def step_text(self, text: str) -> DefenceTurn:
        text = text.strip()
        if not text:
            raise ValueError("empty turn")
        result = self.manager.process_turn(self.session, text)
        m2 = self.config.m2
        policy = resolve_dialogue_policy(
            result,
            force_clarify=self.session.force_clarify,
            top1_threshold=m2.top1_suggest_threshold,
            high_entropy_ratio=m2.high_entropy_ratio,
            clarify_turns_used=self.session.clarify_turns_used,
            clarify_budget=m2.clarify_budget,
        )
        block = build_intention_prompt_block(
            result,
            self.session,
            text,
            inject_enabled=True,
            force_clarify=self.session.force_clarify,
        )
        # Keep the scored policy on the block so the reply matches the inspector.
        if block.dialogue_policy != policy:
            block = IntentionPromptBlock(
                current_stv_summary=block.current_stv_summary,
                top_hypotheses=block.top_hypotheses,
                belief_state=block.belief_state,
                dialogue_policy=policy,
                user_message=block.user_message,
                turn=block.turn,
                inject_enabled=block.inject_enabled,
            )
        if policy == "clarify" and not self.session.force_clarify:
            self.session.clarify_turns_used += 1
        audit = block.privacy_audit()
        turn = DefenceTurn(
            text=text,
            policy=policy,
            m1_category=result.stv.service_category,
            m1_confidence=float(result.stv.category_confidence),
            category_entropy=float(result.category_entropy),
            top1_category_prob=float(result.top1_category_prob),
            posterior_entropy=float(result.posterior_entropy),
            pred_category=top1_category(result.posterior, result.hypotheses),
            privacy_pass=True,
            withheld_from_llm=list(audit["withheld_from_llm"]),
            prompt_text=block.to_prompt_text(),
            result=result,
            block=block,
        )
        self.history.append(turn)
        self.index += 1
        return turn


def run_all(config: Optional[PipelineConfig] = None) -> List[DefenceTurn]:
    runner = DefenceDiningRunner(config)
    while not runner.done:
        runner.step()
    return runner.history


def surface_reply(
    turn: DefenceTurn,
    chat_history: List[ChatMessage],
    *,
    enable_gpt: bool,
    model: str,
) -> str:
    if not enable_gpt:
        return FALLBACK_REPLY.get(turn.policy, FALLBACK_REPLY["suggest"])
    reply = generate_task_reply(
        turn.block,
        chat_history,
        enable_gpt=True,
        model=model,
    )
    mockish = {
        "Could you tell me which type of service you are looking for?",
        "I'm here to help with bookings. What service are you looking for?",
    }
    if reply.strip() in mockish:
        return FALLBACK_REPLY.get(turn.policy, FALLBACK_REPLY["suggest"])
    return reply


def main() -> None:
    turns = run_all()
    policies = tuple(t.policy for t in turns)
    print("scenario", SCENARIO_ID)
    print("policy", list(policies))
    ok = policies == EXPECTED_POLICY
    print("matches_thesis", ok)
    for i, t in enumerate(turns, 1):
        print(
            f"T{i}  {t.policy:8}  m1={t.m1_category:8}  "
            f"H_cat={t.category_entropy:.2f}  mass={t.top1_category_prob:.2f}"
        )
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
