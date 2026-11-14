"""Optional GPT surface smoke: few scenarios × thesis inject on/off.

Does not replace Table B1. Writes qualitative notes for Results/Discussion.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dialogue.orchestrator import DialogueOrchestrator
from src.dialogue.types import ChatMessage
from src.pipeline.config import pipeline_config_from_yaml
from src.pipeline.session import SessionManager


DEFAULT_SCENARIOS = [
    "vague_wellness_weekend",
    "clear_fitness_single",
    "dining_vague_then_specific",
]


def _load_scenarios(path: Path) -> Dict[str, Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["scenario_id"]): r for r in rows}


def _run_one(
    scenario: Dict[str, Any],
    *,
    inject: bool,
    label: str,
) -> Dict[str, Any]:
    base = pipeline_config_from_yaml()
    dialogue = replace(base.dialogue, enable_gpt=True, inject_intention_block=inject)
    cfg = replace(base, enable_m3=False, dialogue=dialogue)
    mgr = SessionManager(cfg)
    orch = DialogueOrchestrator(cfg, session_manager=mgr, inject_intention_block=inject)
    session = mgr.new_session(str(scenario.get("user_id") or "smoke_user"))
    history: List[ChatMessage] = []
    turns_out: List[Dict[str, Any]] = []

    for turn_spec in scenario["turns"]:
        text = str(turn_spec["text"])
        result = orch.handle_turn(session, text, chat_history=history)
        history.append(ChatMessage(role="user", content=text))
        history.append(ChatMessage(role="assistant", content=result.assistant_message))
        turns_out.append(
            {
                "user": text,
                "assistant": result.assistant_message,
                "turn_type": getattr(result.turn_type, "value", str(result.turn_type)),
                "surface_policy": session.surface_policy,
                "inject_intention_block": inject,
            }
        )

    return {
        "condition": label,
        "scenario_id": scenario["scenario_id"],
        "gold_category": scenario.get("gold_category"),
        "turns": turns_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/annotations/eval_scenarios.json")
    parser.add_argument("--out", default="artifacts/thesis_eval/v1/llm_surface_smoke.json")
    parser.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_SCENARIOS),
        help="Comma-separated scenario_ids",
    )
    args = parser.parse_args()

    by_id = _load_scenarios(ROOT / args.dataset)
    ids = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    rows: List[Dict[str, Any]] = []
    for sid in ids:
        if sid not in by_id:
            raise SystemExit(f"Unknown scenario_id: {sid}")
        sc = by_id[sid]
        rows.append(_run_one(sc, inject=False, label="LLM-only"))
        rows.append(_run_one(sc, inject=True, label="Intention-conditioned"))

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows),
        "scenario_ids": ids,
        "note": "Qualitative GPT surface smoke; not Table B1.",
        "runs": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = out.with_suffix(".md")
    lines = [
        "# LLM surface smoke (qualitative)",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- scenarios: {', '.join(ids)}",
        "",
    ]
    for run in rows:
        lines.append(f"## {run['condition']} · `{run['scenario_id']}` (gold={run['gold_category']})")
        lines.append("")
        for i, t in enumerate(run["turns"], 1):
            lines.append(f"**Turn {i} user:** {t['user']}")
            lines.append("")
            lines.append(f"**Assistant:** {t['assistant']}")
            lines.append("")
            lines.append(f"_policy={t['surface_policy']} · type={t['turn_type']}_")
            lines.append("")
        lines.append("---")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
