"""Compare thesis_eval packages (e.g. v1 vs v2) and write an audit-friendly report."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NAIL_SCENARIOS = [
    "dining_vague_then_specific",
    "travel_flight_vague",
    "medical_wellness_checkup",
    "vague_wellness_weekend",
]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _b1(report: Dict[str, Any]) -> Dict[str, Any]:
    return report.get("table_b1") or {}


def _scenario_map(report: Dict[str, Any], condition: str) -> Dict[str, Dict[str, Any]]:
    rows = report.get("conditions", {}).get(condition, {}).get("scenarios") or []
    return {r["scenario_id"]: r for r in rows}


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(b - a, 4)


def compare(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    metrics = [
        "category_task_match",
        "last_m1_match_gold",
        "belief_matches_last_m1",
        "clarify_rate",
        "mean_entropy_drop",
        "mean_category_entropy_drop",
        "resolution_rate",
        "privacy_audit_pass_rate",
    ]
    table: Dict[str, Any] = {}
    for cond in ["LLM-only", "Intention-conditioned", "Memory-augmented"]:
        b = _b1(before).get(cond) or {}
        a = _b1(after).get(cond) or {}
        table[cond] = {
            m: {
                "before": b.get(m),
                "after": a.get(m),
                "delta": _delta(b.get(m), a.get(m)),
            }
            for m in metrics
        }

    nails: List[Dict[str, Any]] = []
    for sid in NAIL_SCENARIOS:
        row: Dict[str, Any] = {"scenario_id": sid}
        for cond in ["Intention-conditioned", "Memory-augmented"]:
            bm = _scenario_map(before, cond)
            am = _scenario_map(after, cond)
            b = bm.get(sid) or {}
            a = am.get(sid) or {}
            row[cond] = {
                "before": {
                    "gold": b.get("gold_category"),
                    "pred": b.get("final_pred_category"),
                    "match": b.get("gold_match"),
                    "last_m1": b.get("last_m1_category"),
                    "policy": b.get("policy_sequence"),
                },
                "after": {
                    "gold": a.get("gold_category"),
                    "pred": a.get("final_pred_category"),
                    "match": a.get("gold_match"),
                    "last_m1": a.get("last_m1_category"),
                    "belief_matches_last_m1": a.get("belief_matches_last_m1"),
                    "policy": a.get("policy_sequence"),
                },
            }
        nails.append(row)

    # Full C1 scenario flip table
    flips: List[Dict[str, Any]] = []
    cond = "Intention-conditioned"
    bm = _scenario_map(before, cond)
    am = _scenario_map(after, cond)
    for sid in sorted(set(bm) | set(am)):
        b = bm.get(sid) or {}
        a = am.get(sid) or {}
        flips.append(
            {
                "scenario_id": sid,
                "gold": a.get("gold_category") or b.get("gold_category"),
                "before_pred": b.get("final_pred_category"),
                "after_pred": a.get("final_pred_category"),
                "before_match": b.get("gold_match"),
                "after_match": a.get("gold_match"),
                "after_last_m1": a.get("last_m1_category"),
                "after_belief_eq_m1": a.get("belief_matches_last_m1"),
                "before_clarify": b.get("clarify_turns"),
                "after_clarify": a.get("clarify_turns"),
                "before_policy": b.get("policy_sequence"),
                "after_policy": a.get("policy_sequence"),
            }
        )

    return {
        "before_created_at": before.get("created_at"),
        "after_created_at": after.get("created_at"),
        "before_freeze": before.get("freeze"),
        "after_freeze": after.get("freeze"),
        "table_b1_delta": table,
        "nail_scenarios": nails,
        "intention_conditioned_scenario_flips": flips,
    }


def _write_md(rep: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Thesis eval before/after comparison",
        "",
        f"- before: `{rep.get('before_created_at')}`",
        f"- after: `{rep.get('after_created_at')}`",
        "",
        "## Freeze",
        "",
        "### Before",
        "```json",
        json.dumps(rep.get("before_freeze"), indent=2, ensure_ascii=False),
        "```",
        "",
        "### After",
        "```json",
        json.dumps(rep.get("after_freeze"), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Table B1 deltas",
        "",
    ]
    for cond, metrics in (rep.get("table_b1_delta") or {}).items():
        lines.append(f"### {cond}")
        lines.append("")
        lines.append("| Metric | Before | After | Δ |")
        lines.append("|---|---:|---:|---:|")
        for m, cell in metrics.items():
            lines.append(
                f"| {m} | {cell.get('before')} | {cell.get('after')} | {cell.get('delta')} |"
            )
        lines.append("")

    lines.extend(["## Nail scenarios (C1/C2)", ""])
    for nail in rep.get("nail_scenarios") or []:
        lines.append(f"### `{nail['scenario_id']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps({k: nail[k] for k in nail if k != "scenario_id"}, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    lines.extend(
        [
            "## Intention-conditioned flips (all scenarios)",
            "",
            "| Scenario | Gold | Before | After | Before✓ | After✓ | After M1 | Belief=M1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in rep.get("intention_conditioned_scenario_flips") or []:
        lines.append(
            "| {scenario_id} | {gold} | {before_pred} | {after_pred} | {before_match} | "
            "{after_match} | {after_last_m1} | {after_belief_eq_m1} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, default=ROOT / "artifacts/thesis_eval/v1/thesis_condition_eval.json")
    ap.add_argument("--after", type=Path, default=ROOT / "artifacts/thesis_eval/v2/thesis_condition_eval.json")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/thesis_eval/v2")
    args = ap.parse_args()

    before = _load(args.before)
    after = _load(args.after)
    rep = compare(before, after)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "before_after_comparison.json"
    md_path = args.out_dir / "before_after_comparison.md"
    json_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_md(rep, md_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    # Compact console summary
    c1 = rep["table_b1_delta"]["Intention-conditioned"]
    print(
        "C1 match {before} → {after} (Δ {delta}); clarify {cb} → {ca}".format(
            before=c1["category_task_match"]["before"],
            after=c1["category_task_match"]["after"],
            delta=c1["category_task_match"]["delta"],
            cb=c1["clarify_rate"]["before"],
            ca=c1["clarify_rate"]["after"],
        )
    )


if __name__ == "__main__":
    main()
