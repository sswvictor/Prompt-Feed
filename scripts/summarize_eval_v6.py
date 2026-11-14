"""Summarize thesis_eval/v6 vs v4 (legacy-14 slice + full N=42)."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]


def _agg(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    matches = [x["gold_match"] for x in rows if x.get("gold_match") is not None]
    hits = sum(1 for m in matches if m)
    total_turns = sum(int(x["n_turns"]) for x in rows) or 1
    clarify = sum(int(x["clarify_turns"]) for x in rows)
    return {
        "n": n,
        "hits": hits,
        "category_task_match": round(hits / len(matches), 4) if matches else None,
        "match_frac": f"{hits}/{len(matches)}" if matches else None,
        "clarify_rate": round(clarify / total_turns, 4),
        "mean_entropy_drop": round(mean(float(x["entropy_drop"]) for x in rows), 6) if rows else None,
    }


def main() -> None:
    v6 = json.loads((ROOT / "artifacts/thesis_eval/v6/thesis_condition_eval.json").read_text(encoding="utf-8"))
    v4 = json.loads((ROOT / "artifacts/thesis_eval/v4/thesis_condition_eval.json").read_text(encoding="utf-8"))
    legacy_ids = {
        s["scenario_id"]
        for s in json.loads((ROOT / "data/annotations/eval_scenarios.json").read_text(encoding="utf-8"))
    }
    new_ids = {
        s["scenario_id"]
        for s in json.loads((ROOT / "data/annotations/eval_scenarios_v2.json").read_text(encoding="utf-8"))
        if s.get("pack") == "v2_enlarge"
    }

    report: Dict[str, Any] = {
        "full_n42": {},
        "legacy14_on_v6": {},
        "new28_on_v6": {},
        "v4_freeze_n14": {},
        "c1_misses_full": [],
    }
    for cond in ["LLM-only", "Intention-conditioned", "Memory-augmented"]:
        sc = v6["conditions"][cond]["scenarios"]
        report["full_n42"][cond] = _agg(sc)
        report["legacy14_on_v6"][cond] = _agg([x for x in sc if x["scenario_id"] in legacy_ids])
        report["new28_on_v6"][cond] = _agg([x for x in sc if x["scenario_id"] in new_ids])
        b = v4["table_b1"][cond]
        report["v4_freeze_n14"][cond] = {
            "category_task_match": b.get("category_task_match"),
            "clarify_rate": b.get("clarify_rate"),
            "mean_entropy_drop": b.get("mean_entropy_drop"),
        }

    report["c1_misses_full"] = [
        {
            "scenario_id": x["scenario_id"],
            "pred": x["final_pred_category"],
            "gold": x["gold_category"],
            "last_m1": x.get("last_m1_category"),
        }
        for x in v6["conditions"]["Intention-conditioned"]["scenarios"]
        if not x.get("gold_match")
    ]

    out = ROOT / "artifacts/thesis_eval/v6/expanded_vs_legacy_summary.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Expanded eval summary (v6, N=42)",
        "",
        "## Full pack (6×7)",
        "",
        "| Condition | Match | Frac | Clarify | Entropy drop |",
        "|---|---:|---:|---:|---:|",
    ]
    for cond, a in report["full_n42"].items():
        lines.append(
            f"| {cond} | {a['category_task_match']} | {a['match_frac']} | {a['clarify_rate']} | {a['mean_entropy_drop']} |"
        )
    lines.extend(
        [
            "",
            "## Legacy-14 slice (same IDs as v4) vs v4 freeze",
            "",
            "| Condition | v6 legacy match | v4 freeze match | v6 clarify | v4 clarify |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for cond in report["legacy14_on_v6"]:
        a = report["legacy14_on_v6"][cond]
        b = report["v4_freeze_n14"][cond]
        lines.append(
            f"| {cond} | {a['category_task_match']} ({a['match_frac']}) | {b['category_task_match']} | "
            f"{a['clarify_rate']} | {b['clarify_rate']} |"
        )
    lines.extend(["", "## New-28 only", "", "| Condition | Match | Frac | Clarify |", "|---|---:|---:|---:|"])
    for cond, a in report["new28_on_v6"].items():
        lines.append(f"| {cond} | {a['category_task_match']} | {a['match_frac']} | {a['clarify_rate']} |")
    lines.extend(["", f"## C1 misses on full N=42 ({len(report['c1_misses_full'])})", ""])
    for m in report["c1_misses_full"]:
        lines.append(f"- `{m['scenario_id']}`: pred={m['pred']} gold={m['gold']} m1={m['last_m1']}")
    md = ROOT / "artifacts/thesis_eval/v6/expanded_vs_legacy_summary.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["full_n42"], indent=2))
    print(f"Wrote {out}")
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
