"""Thesis-aligned multi-turn evaluation: LLM-only / Intention-conditioned / Memory-augmented.

Primary metrics follow Method §3.4 (category match, entropy, clarify/suggest, turns to
resolution, prompt privacy audit). Sidecar belief drives task metrics; the generative
surface is optional (--with-llm) and is not required for Table B1.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dialogue.intention_context import (
    build_intention_prompt_block,
    resolve_dialogue_policy,
)
from dataclasses import replace

from src.m2_reasoning.belief import top1_category
from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore
from src.config.paths import rel_to_root
from src.pipeline.config import PipelineConfig, pipeline_config_from_yaml
from src.pipeline.runner import PipelineRunner
from src.pipeline.session import SessionManager

C0_CONFIDENCE_SUGGEST_THRESHOLD = 0.5
FORBIDDEN_PROMPT_PATTERNS = [
    re.compile(r"user_id", re.I),
    re.compile(r"category_counts", re.I),
    re.compile(r"persona_", re.I),
    re.compile(r"email", re.I),
    re.compile(r"phone", re.I),
]


def _load_scenarios(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_profile_store() -> InMemoryStore:
    store = InMemoryStore()
    src = ROOT / "data" / "profiles" / "synthetic_users.json"
    if src.exists():
        for row in json.loads(src.read_text(encoding="utf-8")):
            store.save_profile(AggregatedUserProfile.from_dict(row))
    return store


def _seed_aligned_profile(store: InMemoryStore, user_id: str, gold_category: str) -> None:
    """Mild gold-aligned aggregate prior so Memory-augmented can differ from C1."""
    if not gold_category:
        return
    profile = store.get_profile(user_id)
    profile.category_counts[gold_category] = max(int(profile.category_counts.get(gold_category, 0)), 3)
    profile.total_sessions = max(profile.total_sessions, 3)
    store.save_profile(profile)


def _is_resolved(policy: str, result) -> bool:
    return policy == "suggest" and (result.converged or result.top1_category_prob >= 0.5)


def _turns_to_resolution(policies: List[str], results: List) -> Optional[int]:
    for i, (policy, result) in enumerate(zip(policies, results), start=1):
        if _is_resolved(policy, result):
            return i
    return None


def _privacy_pass(block) -> Dict[str, Any]:
    audit = block.privacy_audit()
    prompt = block.to_prompt_text() if block.inject_enabled else ""
    payload = json.dumps(audit.get("sent_to_llm", {}), ensure_ascii=False)
    haystack = prompt + "\n" + payload
    hits = [p.pattern for p in FORBIDDEN_PROMPT_PATTERNS if p.search(haystack)]
    return {
        "pass": len(hits) == 0,
        "forbidden_hits": hits,
        "inject_enabled": block.inject_enabled,
        "withheld_declared": audit.get("withheld_from_llm", []),
    }


def _run_llm_only(scenario: Dict[str, Any], config: PipelineConfig) -> Dict[str, Any]:
    """No multi-turn belief in the decision path; no intention summary injection."""
    turns = scenario["turns"]
    cfg = replace(config, enable_m3=False)
    runner = PipelineRunner(cfg)
    mgr = SessionManager(cfg, runner=runner)

    turn_rows: List[Dict[str, Any]] = []
    policies: List[str] = []
    results = []
    privacy_flags: List[bool] = []

    for turn_spec in turns:
        text = str(turn_spec["text"])
        # Fresh session each turn: no cross-turn belief (LLM-only structural baseline).
        session = mgr.new_session(str(scenario.get("user_id") or "persona_cold_start"))
        result = mgr.process_turn(session, text)
        # Override policy with M1-confidence gate (no M2-driven policy for this condition).
        stv = result.stv
        cat = stv.service_category
        conf = stv.category_confidence
        policy = (
            "clarify"
            if cat in {"ambiguous", "other"} or conf < C0_CONFIDENCE_SUGGEST_THRESHOLD
            else "suggest"
        )
        pred = cat
        block = build_intention_prompt_block(
            result, session, text, inject_enabled=False, force_clarify=False
        )
        priv = _privacy_pass(block)
        privacy_flags.append(bool(priv["pass"]))
        policies.append(policy)
        results.append(result)
        turn_rows.append(
            {
                "turn": len(turn_rows) + 1,
                "text": text,
                "m1_category": cat,
                "m1_confidence": round(conf, 4),
                "posterior_entropy": round(result.posterior_entropy, 6),
                "top1_prob": round(result.top1_prob, 4),
                "converged": result.converged,
                "pred_category": pred,
                "dialogue_policy": policy,
                "privacy_pass": priv["pass"],
                "gold_hints": turn_spec.get("gold_hints"),
            }
        )

    final_pred = turn_rows[-1]["pred_category"] if turn_rows else None
    gold = str(scenario.get("gold_category", ""))
    entropies = [r.posterior_entropy for r in results]
    return {
        "condition": "LLM-only",
        "code_id": "C0_llm_only",
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
        "privacy_pass_rate": mean(privacy_flags) if privacy_flags else 1.0,
        "privacy_all_pass": all(privacy_flags) if privacy_flags else True,
    }


def _run_belief_condition(
    scenario: Dict[str, Any],
    config: PipelineConfig,
    *,
    thesis_name: str,
    code_id: str,
    inject_intention: bool,
    store: Optional[InMemoryStore],
    seed_m3: bool,
) -> Dict[str, Any]:
    user_id = str(scenario.get("user_id") or "persona_cold_start")
    gold = str(scenario.get("gold_category", ""))
    if seed_m3 and store is not None:
        _seed_aligned_profile(store, user_id, gold)

    cfg = replace(
        config,
        enable_m3=bool(store is not None),
        user_id=user_id,
        memory_backend="inmemory",
    )
    runner = PipelineRunner(cfg, store=store)
    mgr = SessionManager(cfg, runner=runner)
    session = mgr.new_session(user_id)

    turn_rows: List[Dict[str, Any]] = []
    policies: List[str] = []
    results = []
    privacy_flags: List[bool] = []

    m2 = cfg.m2
    for turn_spec in scenario["turns"]:
        text = str(turn_spec["text"])
        result = mgr.process_turn(session, text)
        policy = resolve_dialogue_policy(
            result,
            force_clarify=session.force_clarify,
            top1_threshold=m2.top1_suggest_threshold,
            high_entropy_ratio=m2.high_entropy_ratio,
            clarify_turns_used=session.clarify_turns_used,
            clarify_budget=m2.clarify_budget,
        )
        if policy == "clarify" and not session.force_clarify:
            session.clarify_turns_used += 1
        pred = top1_category(result.posterior, result.hypotheses)
        m1_cat = result.stv.service_category
        block = build_intention_prompt_block(
            result,
            session,
            text,
            inject_enabled=inject_intention,
            force_clarify=session.force_clarify,
        )
        priv = _privacy_pass(block)
        privacy_flags.append(bool(priv["pass"]))
        policies.append(policy)
        results.append(result)
        turn_rows.append(
            {
                "turn": session.turn,
                "text": text,
                "m1_category": m1_cat,
                "m1_confidence": round(result.stv.category_confidence, 4),
                "posterior_entropy": round(result.posterior_entropy, 6),
                "category_entropy": round(result.category_entropy, 6),
                "top1_prob": round(result.top1_prob, 4),
                "top1_category_prob": round(result.top1_category_prob, 4),
                "converged": result.converged,
                "pred_category": pred,
                "belief_matches_m1": pred == m1_cat,
                "dialogue_policy": policy,
                "clarify_turns_used": session.clarify_turns_used,
                "privacy_pass": priv["pass"],
                "forbidden_hits": priv["forbidden_hits"],
                "gold_hints": turn_spec.get("gold_hints"),
            }
        )

    final_pred = top1_category(results[-1].posterior, results[-1].hypotheses) if results else None
    last_m1 = results[-1].stv.service_category if results else None
    entropies = [r.posterior_entropy for r in results]
    cat_ents = [r.category_entropy for r in results]
    return {
        "condition": thesis_name,
        "code_id": code_id,
        "scenario_id": scenario["scenario_id"],
        "gold_category": gold,
        "n_turns": len(scenario["turns"]),
        "turns": turn_rows,
        "entropy_per_turn": [round(e, 6) for e in entropies],
        "category_entropy_per_turn": [round(e, 6) for e in cat_ents],
        "entropy_drop": round(entropies[0] - entropies[-1], 6) if len(entropies) > 1 else 0.0,
        "category_entropy_drop": round(cat_ents[0] - cat_ents[-1], 6) if len(cat_ents) > 1 else 0.0,
        "turns_to_resolution": _turns_to_resolution(policies, results),
        "final_pred_category": final_pred,
        "last_m1_category": last_m1,
        "last_m1_match_gold": last_m1 == gold if gold and last_m1 else None,
        "belief_matches_last_m1": final_pred == last_m1 if final_pred and last_m1 else None,
        "gold_match": final_pred == gold if gold else None,
        "clarify_turns": sum(1 for p in policies if p == "clarify"),
        "suggest_turns": sum(1 for p in policies if p == "suggest"),
        "policy_sequence": policies,
        "privacy_pass_rate": mean(privacy_flags) if privacy_flags else 1.0,
        "privacy_all_pass": all(privacy_flags) if privacy_flags else True,
        "m3_seeded_to_gold": bool(seed_m3 and gold),
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    matches = [r["gold_match"] for r in rows if r.get("gold_match") is not None]
    resolved = [r["turns_to_resolution"] for r in rows if r.get("turns_to_resolution") is not None]
    total_turns = sum(int(r["n_turns"]) for r in rows) or 1
    clarify = sum(int(r["clarify_turns"]) for r in rows)
    suggest = sum(int(r["suggest_turns"]) for r in rows)
    m1_gold = [r["last_m1_match_gold"] for r in rows if r.get("last_m1_match_gold") is not None]
    belief_m1 = [r["belief_matches_last_m1"] for r in rows if r.get("belief_matches_last_m1") is not None]
    return {
        "n_scenarios": n,
        "category_task_match": round(sum(1 for m in matches if m) / len(matches), 4) if matches else None,
        "last_m1_match_gold": round(sum(1 for m in m1_gold if m) / len(m1_gold), 4) if m1_gold else None,
        "belief_matches_last_m1": (
            round(sum(1 for m in belief_m1 if m) / len(belief_m1), 4) if belief_m1 else None
        ),
        "resolution_rate": round(len(resolved) / n, 4) if n else None,
        "mean_turns_to_resolution": round(mean(resolved), 4) if resolved else None,
        "mean_entropy_drop": round(mean(float(r["entropy_drop"]) for r in rows), 6) if rows else None,
        "mean_category_entropy_drop": (
            round(mean(float(r.get("category_entropy_drop", 0.0)) for r in rows), 6) if rows else None
        ),
        "clarify_rate": round(clarify / total_turns, 4),
        "suggest_rate": round(suggest / total_turns, 4),
        "privacy_audit_pass_rate": round(
            sum(1 for r in rows if r.get("privacy_all_pass")) / n, 4
        )
        if n
        else None,
    }


def run_eval(scenarios_path: Path, config: PipelineConfig) -> Dict[str, Any]:
    scenarios = _load_scenarios(scenarios_path)
    store = _load_profile_store()

    c0_rows = [_run_llm_only(s, config) for s in scenarios]
    c1_rows = [
        _run_belief_condition(
            s,
            replace(config, enable_m3=False),
            thesis_name="Intention-conditioned",
            code_id="C1_intention_block",
            inject_intention=True,
            store=None,
            seed_m3=False,
        )
        for s in scenarios
    ]
    c2_rows = [
        _run_belief_condition(
            s,
            replace(config, enable_m3=True),
            thesis_name="Memory-augmented",
            code_id="C2_m3_prior",
            inject_intention=True,
            store=store,
            seed_m3=True,
        )
        for s in scenarios
    ]

    table_b1 = {
        "LLM-only": _aggregate(c0_rows),
        "Intention-conditioned": _aggregate(c1_rows),
        "Memory-augmented": _aggregate(c2_rows),
    }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": rel_to_root(scenarios_path),
        "extractor_mode": config.extractor_mode,
        "n_scenarios": len(scenarios),
        "freeze": {
            "converge_threshold": config.converge_threshold,
            "hypothesis_max_size": config.hypothesis_max_size,
            "extractor_mode": config.extractor_mode,
            "config_path": rel_to_root(config.config_path),
            "m2": {
                "slot_evidence_power": config.m2.slot_evidence_power,
                "belief_power_decay": config.m2.belief_power_decay,
                "belief_recency_alpha": config.m2.belief_recency_alpha,
                "prior_reinjection": config.m2.prior_reinjection,
                "top1_suggest_threshold": config.m2.top1_suggest_threshold,
                "high_entropy_ratio": config.m2.high_entropy_ratio,
                "clarify_budget": config.m2.clarify_budget,
                "enable_domain_guards": config.m2.enable_domain_guards,
                "enable_m1_voting": config.m2.enable_m1_voting,
                "guard_confidence_cap": config.m2.guard_confidence_cap,
                "vote_recency_decay": config.m2.vote_recency_decay,
                "enable_conflict_dampen": config.m2.enable_conflict_dampen,
                "conflict_dampen_factor": config.m2.conflict_dampen_factor,
            },
            "hypothesis_category_counts": dict(
                Counter(
                    h.service_category
                    for h in PipelineRunner(replace(config, enable_m3=False)).hypotheses
                )
            ),
            "note": "Sidecar metrics only; generative LLM not required for Table B1.",
            "memory_augmented_seed": "mild gold-aligned category_counts (>=3) per scenario user_id",
        },
        "table_b1": table_b1,
        "conditions": {
            "LLM-only": {"code_id": "C0_llm_only", "aggregate": table_b1["LLM-only"], "scenarios": c0_rows},
            "Intention-conditioned": {
                "code_id": "C1_intention_block",
                "aggregate": table_b1["Intention-conditioned"],
                "scenarios": c1_rows,
            },
            "Memory-augmented": {
                "code_id": "C2_m3_prior",
                "aggregate": table_b1["Memory-augmented"],
                "scenarios": c2_rows,
            },
        },
    }


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = [
        "# Thesis condition evaluation (Table B1)",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- n_scenarios: {report['n_scenarios']}",
        f"- extractor_mode: `{report['extractor_mode']}`",
        "",
        "## Table B1 — primary scenario metrics",
        "",
        "| Condition | Category match | Resolution rate | Mean turns to res. | Mean entropy drop | Clarify rate | Suggest rate | Privacy pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, agg in report["table_b1"].items():
        lines.append(
            "| {name} | {cm} | {rr} | {mt} | {ed} | {cr} | {sr} | {pr} |".format(
                name=name,
                cm=agg.get("category_task_match"),
                rr=agg.get("resolution_rate"),
                mt=agg.get("mean_turns_to_resolution"),
                ed=agg.get("mean_entropy_drop"),
                cr=agg.get("clarify_rate"),
                sr=agg.get("suggest_rate"),
                pr=agg.get("privacy_audit_pass_rate"),
            )
        )
    lines.extend(
        [
            "",
            "## Freeze",
            "",
            "```json",
            json.dumps(report["freeze"], indent=2),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Thesis-aligned 3-condition scenario evaluation")
    parser.add_argument("--dataset", default="data/annotations/eval_scenarios.json")
    parser.add_argument("--out-dir", default="artifacts/thesis_eval/v1")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    dataset = ROOT / args.dataset
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    config = pipeline_config_from_yaml(args.config)
    report = run_eval(dataset, config)

    json_path = out_dir / "thesis_condition_eval.json"
    md_path = out_dir / "thesis_condition_eval.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(report["table_b1"], indent=2, ensure_ascii=False))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
