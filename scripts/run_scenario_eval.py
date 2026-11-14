from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dialogue.intention_context import resolve_dialogue_policy
from src.m1_nlu.factory import get_extractor
from src.m2_reasoning.belief import top1_category
from src.pipeline.config import PipelineConfig, pipeline_config_from_yaml
from src.pipeline.runner import PipelineRunner
from src.pipeline.session import SessionManager

# C0 baseline: M1-only policy without M2 belief (legacy confidence gate for comparison).
C0_CONFIDENCE_SUGGEST_THRESHOLD = 0.5


def _load_scenarios(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _c0_policy(stv) -> str:
    """M1-only dialogue policy baseline (no M2 posterior)."""
    cat = stv.service_category
    conf = stv.category_confidence
    if cat in {"ambiguous", "other"} or conf < C0_CONFIDENCE_SUGGEST_THRESHOLD:
        return "clarify"
    return "suggest"


def _is_resolved(policy: str, result) -> bool:
    return policy == "suggest" and (
        result.converged or result.top1_category_prob >= 0.5
    )


def _turns_to_resolution(policies: List[str], results: List) -> Optional[int]:
    for i, (policy, result) in enumerate(zip(policies, results), start=1):
        if _is_resolved(policy, result):
            return i
    return None


def _run_c1_scenario(
    scenario: Dict[str, Any],
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Full M1+M2 multi-turn session with M2-driven dialogue policy (C1)."""
    user_id = str(scenario.get("user_id") or "persona_cold_start")
    turns = scenario["turns"]
    mgr = SessionManager(config)
    session = mgr.new_session(user_id)

    turn_rows: List[Dict[str, Any]] = []
    policies: List[str] = []
    results = []

    for turn_spec in turns:
        text = str(turn_spec["text"])
        result = mgr.process_turn(session, text)
        policy = resolve_dialogue_policy(result, force_clarify=session.force_clarify)
        pred_category = top1_category(result.posterior, result.hypotheses)

        policies.append(policy)
        results.append(result)
        turn_rows.append(
            {
                "turn": session.turn,
                "text": text,
                "m1_category": result.stv.service_category,
                "m1_confidence": round(result.stv.category_confidence, 4),
                "posterior_entropy": round(result.posterior_entropy, 6),
                "top1_prob": round(result.top1_prob, 4),
                "top1_category_prob": round(result.top1_category_prob, 4),
                "category_entropy": round(result.category_entropy, 4),
                "converged": result.converged,
                "pred_category": pred_category,
                "dialogue_policy": policy,
                "emotion_escape": result.emotion_escape_triggered,
                "gold_hints": turn_spec.get("gold_hints"),
            }
        )

    final_pred = top1_category(results[-1].posterior, results[-1].hypotheses) if results else None
    gold = str(scenario.get("gold_category", ""))
    entropies = [r.posterior_entropy for r in results]

    return {
        "condition": "C1_pipeline_m2",
        "scenario_id": scenario["scenario_id"],
        "gold_category": gold,
        "n_turns": len(turns),
        "turns": turn_rows,
        "entropy_per_turn": [round(e, 6) for e in entropies],
        "entropy_drop": round(entropies[0] - entropies[-1], 6) if len(entropies) > 1 else 0.0,
        "turns_to_resolution": _turns_to_resolution(policies, results),
        "final_pred_category": final_pred,
        "gold_match": final_pred == gold if gold else None,
        "clarify_turns": sum(1 for p in policies if p == "clarify"),
        "suggest_turns": sum(1 for p in policies if p == "suggest"),
        "policy_sequence": policies,
    }


def _run_c0_scenario(
    scenario: Dict[str, Any],
    config: PipelineConfig,
) -> Dict[str, Any]:
    """Per-turn M1 extraction without multi-turn belief; M1-only policy (C0)."""
    user_id = str(scenario.get("user_id") or "persona_cold_start")
    turns = scenario["turns"]
    runner = PipelineRunner(config)
    extractor = get_extractor(config.extractor_mode)

    turn_rows: List[Dict[str, Any]] = []
    policies: List[str] = []
    results = []

    for turn_spec in turns:
        text = str(turn_spec["text"])
        stv = extractor.extract(text)
        policy = _c0_policy(stv)
        # Independent per-turn pipeline result (turn=0, no belief carry-over).
        result = runner.process_text(text, turn=0, user_id=user_id)

        policies.append(policy)
        results.append(result)
        turn_rows.append(
            {
                "turn": len(turn_rows) + 1,
                "text": text,
                "m1_category": stv.service_category,
                "m1_confidence": round(stv.category_confidence, 4),
                "posterior_entropy": round(result.posterior_entropy, 6),
                "top1_prob": round(result.top1_prob, 4),
                "converged": result.converged,
                "pred_category": top1_category(result.posterior, result.hypotheses),
                "dialogue_policy": policy,
                "gold_hints": turn_spec.get("gold_hints"),
            }
        )

    final_pred = results[-1].stv.service_category if results else None
    gold = str(scenario.get("gold_category", ""))
    entropies = [r.posterior_entropy for r in results]

    return {
        "condition": "C0_m1_only",
        "scenario_id": scenario["scenario_id"],
        "gold_category": gold,
        "n_turns": len(turns),
        "turns": turn_rows,
        "entropy_per_turn": [round(e, 6) for e in entropies],
        "entropy_drop": round(entropies[0] - entropies[-1], 6) if len(entropies) > 1 else 0.0,
        "turns_to_resolution": _turns_to_resolution(policies, results),
        "final_pred_category": final_pred,
        "gold_match": final_pred == gold if gold else None,
        "clarify_turns": sum(1 for p in policies if p == "clarify"),
        "suggest_turns": sum(1 for p in policies if p == "suggest"),
        "policy_sequence": policies,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}

    resolved = [r["turns_to_resolution"] for r in rows if r["turns_to_resolution"] is not None]
    gold_hits = [int(r["gold_match"]) for r in rows if r["gold_match"] is not None]
    total_turns = sum(r["n_turns"] for r in rows)
    total_clarify = sum(r["clarify_turns"] for r in rows)

    return {
        "n_scenarios": len(rows),
        "total_turns": total_turns,
        "gold_accuracy": round(mean(gold_hits), 4) if gold_hits else 0.0,
        "mean_entropy_drop": round(mean(r["entropy_drop"] for r in rows), 6),
        "mean_turns_to_resolution": round(mean(resolved), 4) if resolved else None,
        "resolution_rate": round(len(resolved) / len(rows), 4),
        "clarify_rate": round(total_clarify / total_turns, 4) if total_turns else 0.0,
        "suggest_rate": round(1.0 - total_clarify / total_turns, 4) if total_turns else 0.0,
    }


def _policy_comparison(c0_rows: List[Dict[str, Any]], c1_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id_c0 = {r["scenario_id"]: r for r in c0_rows}
    by_id_c1 = {r["scenario_id"]: r for r in c1_rows}
    paired: List[Dict[str, Any]] = []

    for sid, c1 in by_id_c1.items():
        c0 = by_id_c0.get(sid)
        if c0 is None:
            continue
        paired.append(
            {
                "scenario_id": sid,
                "c0_turns_to_resolution": c0["turns_to_resolution"],
                "c1_turns_to_resolution": c1["turns_to_resolution"],
                "c1_faster": (
                    c1["turns_to_resolution"] is not None
                    and (
                        c0["turns_to_resolution"] is None
                        or c1["turns_to_resolution"] < c0["turns_to_resolution"]
                    )
                ),
                "c0_clarify_turns": c0["clarify_turns"],
                "c1_clarify_turns": c1["clarify_turns"],
                "c0_gold_match": c0["gold_match"],
                "c1_gold_match": c1["gold_match"],
                "c0_final_entropy": c0["entropy_per_turn"][-1] if c0["entropy_per_turn"] else None,
                "c1_final_entropy": c1["entropy_per_turn"][-1] if c1["entropy_per_turn"] else None,
            }
        )

    c1_faster_count = sum(1 for p in paired if p["c1_faster"])
    c1_gold_wins = sum(
        1
        for p in paired
        if p["c1_gold_match"] and not p["c0_gold_match"]
    )
    c0_gold_wins = sum(
        1
        for p in paired
        if p["c0_gold_match"] and not p["c1_gold_match"]
    )

    return {
        "paired_scenarios": len(paired),
        "c1_faster_resolution_count": c1_faster_count,
        "c1_gold_match_wins": c1_gold_wins,
        "c0_gold_match_wins": c0_gold_wins,
        "mean_c0_clarify_per_scenario": round(mean(p["c0_clarify_turns"] for p in paired), 4) if paired else 0.0,
        "mean_c1_clarify_per_scenario": round(mean(p["c1_clarify_turns"] for p in paired), 4) if paired else 0.0,
        "pairs": paired,
    }


def run_eval(
    scenarios_path: Path,
    config: PipelineConfig,
) -> Dict[str, Any]:
    scenarios = _load_scenarios(scenarios_path)
    c0_rows = [_run_c0_scenario(s, config) for s in scenarios]
    c1_rows = [_run_c1_scenario(s, config) for s in scenarios]

    return {
        "dataset": str(scenarios_path),
        "extractor_mode": config.extractor_mode,
        "n_scenarios": len(scenarios),
        "categories_covered": sorted({s.get("gold_category") for s in scenarios if s.get("gold_category")}),
        "conditions": {
            "C0_m1_only": {
                "description": "Per-turn M1 extraction; M1-confidence policy baseline (no multi-turn M2)",
                "aggregate": _aggregate(c0_rows),
                "scenarios": c0_rows,
            },
            "C1_pipeline_m2": {
                "description": "Multi-turn M1+M2 session; M2-driven resolve_dialogue_policy()",
                "aggregate": _aggregate(c1_rows),
                "scenarios": c1_rows,
            },
        },
        "policy_comparison": _policy_comparison(c0_rows, c1_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn scenario evaluation (M1+M2)")
    parser.add_argument(
        "--dataset",
        default="data/annotations/eval_scenarios.json",
        help="Path to eval scenarios JSON",
    )
    parser.add_argument(
        "--out",
        default="artifacts/m1/scenario_eval_report.json",
        help="Output report path",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional config.yaml path (default: project config.yaml)",
    )
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    out_path = ROOT / args.out
    config = pipeline_config_from_yaml(args.config)
    config.enable_m3 = False

    report = run_eval(dataset, config)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
