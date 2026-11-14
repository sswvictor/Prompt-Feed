from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.ambiguity import ambiguity_bucket
from src.m1_nlu.factory import get_extractor
from src.m2_reasoning.belief import entropy, initialize_belief, update_belief, uniform_prior
from src.m2_reasoning.convergence import k_schedule, top1_probability, top_k_hypothesis_ids
from src.m2_reasoning.hypothesis import GoalHypothesis, generate_hypothesis_space
from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner


def _load_rows() -> List[Dict[str, object]]:
    files = [
        ROOT / "data" / "annotations" / "m1_eval_samples.json",
        ROOT / "data" / "annotations" / "m1_hard_cases.json",
    ]
    rows: List[Dict[str, object]] = []
    for fp in files:
        rows.extend(json.loads(fp.read_text(encoding="utf-8")))
    return rows



def _top_category(belief: Dict[str, float], hypotheses: List[GoalHypothesis]) -> str:
    by_id = {h.hypothesis_id: h for h in hypotheses}
    cat_probs: Dict[str, float] = {}
    for h_id, p in belief.items():
        cat = by_id[h_id].service_category
        cat_probs[cat] = cat_probs.get(cat, 0.0) + p
    return max(cat_probs.items(), key=lambda kv: kv[1])[0]


def _run_condition(
    rows: List[Dict[str, object]],
    extractor_mode: str,
    use_m2: bool,
    label: str,
    hypotheses: List[GoalHypothesis],
) -> Dict[str, object]:
    extractor = get_extractor(extractor_mode)
    correct = 0
    bucket_scores: Dict[str, List[int]] = {"high": [], "medium": [], "low": []}
    entropy_drops: List[float] = []
    top1_gains: List[float] = []
    candidate_reductions: List[float] = []
    error_taxonomy = {
        "m1_category_mismatch": 0,
        "m2_not_recovered": 0,
        "ambiguous_input_error": 0,
    }
    sample_errors: List[Dict[str, object]] = []

    prior = uniform_prior(hypotheses)
    init = initialize_belief(hypotheses)
    prior_top1 = top1_probability(init)
    prior_entropy = entropy(init)
    total_h = len(hypotheses)
    k0 = k_schedule(0)

    for row in rows:
        text = str(row["text"])
        gold = str(row["service_category"])
        bucket = ambiguity_bucket(text, row)
        stv = extractor.extract(text)
        m1_pred = stv.service_category

        if use_m2:
            posterior = update_belief(init, stv, prior, hypotheses)
            final_pred = _top_category(posterior, hypotheses)
            entropy_drops.append(prior_entropy - entropy(posterior))
            top1_gains.append(top1_probability(posterior) - prior_top1)
            top_k = top_k_hypothesis_ids(posterior, 0)
            candidate_reductions.append((total_h - len(top_k)) / total_h if total_h else 0.0)
        else:
            final_pred = m1_pred

        hit = int(final_pred == gold)
        correct += hit
        bucket_scores[bucket].append(hit)

        if hit == 0:
            if gold == "ambiguous":
                error_taxonomy["ambiguous_input_error"] += 1
            if m1_pred != gold:
                error_taxonomy["m1_category_mismatch"] += 1
            if use_m2 and m1_pred != gold and final_pred != gold:
                error_taxonomy["m2_not_recovered"] += 1
            if len(sample_errors) < 10:
                sample_errors.append(
                    {
                        "text": text,
                        "gold": gold,
                        "m1_pred": m1_pred,
                        "final_pred": final_pred,
                        "bucket": bucket,
                    }
                )

    def _acc(v: List[int]) -> float:
        return sum(v) / len(v) if v else 0.0

    entropy_drop_mean = mean(entropy_drops) if entropy_drops else 0.0
    top1_gain_mean = mean(top1_gains) if top1_gains else 0.0
    reduction_mean = mean(candidate_reductions) if candidate_reductions else ((total_h - k0) / total_h if total_h else 0.0)

    return {
        "condition": label,
        "extractor_mode": extractor_mode,
        "use_m2": use_m2,
        "use_m3": False,
        "n": len(rows),
        "intent_accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "bucket_accuracy": {k: round(_acc(v), 4) for k, v in bucket_scores.items()},
        "m2_metrics": {
            "belief_entropy_drop_mean": round(entropy_drop_mean, 6),
            "belief_entropy_drop_std": round(pstdev(entropy_drops), 6) if len(entropy_drops) > 1 else 0.0,
            "top1_prob_gain_mean": round(top1_gain_mean, 6),
            "candidate_reduction_rate_mean": round(reduction_mean, 6),
        },
        "error_taxonomy": error_taxonomy,
        "error_samples": sample_errors,
    }


def _dominant_category(profile: AggregatedUserProfile) -> str | None:
    if not profile.category_counts:
        return None
    return max(profile.category_counts.items(), key=lambda kv: kv[1])[0]


def _load_profile_store() -> InMemoryStore:
    store = InMemoryStore()
    src = ROOT / "data" / "profiles" / "synthetic_users.json"
    if src.exists():
        rows = json.loads(src.read_text(encoding="utf-8"))
        for row in rows:
            store.save_profile(AggregatedUserProfile.from_dict(row))
    return store


def _run_condition_c3(
    rows: List[Dict[str, object]],
    hypotheses: List[GoalHypothesis],
    store: InMemoryStore,
) -> Dict[str, object]:
    correct = 0
    profile_aligned_hits = 0
    profile_aligned_n = 0
    bucket_scores: Dict[str, List[int]] = {"high": [], "medium": [], "low": []}
    entropy_drops: List[float] = []
    top1_gains: List[float] = []
    candidate_reductions: List[float] = []
    error_taxonomy = {
        "m1_category_mismatch": 0,
        "m2_not_recovered": 0,
        "ambiguous_input_error": 0,
    }
    sample_errors: List[Dict[str, object]] = []
    total_h = len(hypotheses)
    k0 = k_schedule(0)

    for row in rows:
        text = str(row["text"])
        gold = str(row["service_category"])
        bucket = ambiguity_bucket(text, row)
        user_id = str(row.get("user_id") or "persona_cold_start")

        cfg = PipelineConfig(
            extractor_mode="pipeline",
            enable_m3=True,
            user_id=user_id,
            memory_backend="inmemory",
            hypothesis_max_size=len(hypotheses),
        )
        runner = PipelineRunner(cfg, store=store)
        result = runner.process_text(text, turn=0, user_id=user_id)
        stv = result.stv
        m1_pred = stv.service_category
        final_pred = _top_category(result.posterior, hypotheses)

        entropy_drops.append(result.prior_entropy - result.posterior_entropy)
        top1_gains.append(result.top1_prob - top1_probability(result.prior))
        candidate_reductions.append((total_h - len(result.top_k_ids)) / total_h if total_h else 0.0)

        hit = int(final_pred == gold)
        correct += hit
        bucket_scores[bucket].append(hit)

        profile = store.get_profile(user_id)
        dom = _dominant_category(profile)
        if dom and gold == dom:
            profile_aligned_n += 1
            profile_aligned_hits += hit

        if hit == 0:
            if gold == "ambiguous":
                error_taxonomy["ambiguous_input_error"] += 1
            if m1_pred != gold:
                error_taxonomy["m1_category_mismatch"] += 1
            if m1_pred != gold and final_pred != gold:
                error_taxonomy["m2_not_recovered"] += 1
            if len(sample_errors) < 10:
                sample_errors.append(
                    {
                        "text": text,
                        "gold": gold,
                        "m1_pred": m1_pred,
                        "final_pred": final_pred,
                        "user_id": user_id,
                        "bucket": bucket,
                    }
                )

    def _acc(v: List[int]) -> float:
        return sum(v) / len(v) if v else 0.0

    return {
        "condition": "C3",
        "extractor_mode": "pipeline",
        "use_m2": True,
        "use_m3": True,
        "n": len(rows),
        "intent_accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "profile_aligned_accuracy": round(profile_aligned_hits / profile_aligned_n, 4)
        if profile_aligned_n
        else None,
        "profile_aligned_n": profile_aligned_n,
        "bucket_accuracy": {k: round(_acc(v), 4) for k, v in bucket_scores.items()},
        "m2_metrics": {
            "belief_entropy_drop_mean": round(mean(entropy_drops), 6) if entropy_drops else 0.0,
            "belief_entropy_drop_std": round(pstdev(entropy_drops), 6) if len(entropy_drops) > 1 else 0.0,
            "top1_prob_gain_mean": round(mean(top1_gains), 6) if top1_gains else 0.0,
            "candidate_reduction_rate_mean": round(mean(candidate_reductions), 6)
            if candidate_reductions
            else ((total_h - k0) / total_h if total_h else 0.0),
        },
        "error_taxonomy": error_taxonomy,
        "error_samples": sample_errors,
    }


def _write_report(report: Dict[str, object]) -> Tuple[Path, Path]:
    out_dir = ROOT / "artifacts" / "m1_m2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "m1_m2_attribution_report.json"
    out_md = out_dir / "m1_m2_attribution_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    c0 = report["conditions"]["C0_llm_only"]
    c1 = report["conditions"]["C1_pipeline_m2"]
    c2 = report["conditions"]["C2_llm_m2"]
    md_lines = [
        "# M1M2 Attribution Report",
        "",
        "## Conditions",
        "- C0: LLM-only (no M2)",
        "- C1: M1(pipeline) + M2",
        "- C2: M1(llm) + M2",
        "",
        "## Core Metrics",
        f"- C0 intent_accuracy: {c0['intent_accuracy']}",
        f"- C1 intent_accuracy: {c1['intent_accuracy']}",
        f"- C2 intent_accuracy: {c2['intent_accuracy']}",
        f"- C1 entropy_drop_mean: {c1['m2_metrics']['belief_entropy_drop_mean']}",
        f"- C2 entropy_drop_mean: {c2['m2_metrics']['belief_entropy_drop_mean']}",
        f"- C1 candidate_reduction_rate_mean: {c1['m2_metrics']['candidate_reduction_rate_mean']}",
        f"- C2 candidate_reduction_rate_mean: {c2['m2_metrics']['candidate_reduction_rate_mean']}",
        "",
        "## Attribution Reading",
        "- If C1 > C0, structured extraction + belief update contributes beyond LLM-only.",
        "- If C2 ~= C1, M2 is likely the dominant gain source.",
        "- If C2 >> C1, upstream extraction quality remains a bottleneck.",
        "",
        "## Notes",
        "- LLM mode may fallback to mock provider if no API key is configured.",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")
    return out_json, out_md


def _write_m3_report(report: Dict[str, object]) -> Tuple[Path, Path]:
    out_dir = ROOT / "artifacts" / "m1_m2_m3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "m1_m2_m3_attribution_report.json"
    out_md = out_dir / "m1_m2_m3_attribution_report.md"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    c1 = report["conditions"]["C1_pipeline_m2"]
    c3 = report["conditions"]["C3_pipeline_m2_m3"]
    md_lines = [
        "# M1M2M3 Attribution Report",
        "",
        "## Conditions",
        "- C0: LLM-only (no M2)",
        "- C1: M1(pipeline) + M2 (uniform prior)",
        "- C2: M1(llm) + M2",
        "- C3: M1(pipeline) + M2 + M3 (history prior) — maps to thesis report Condition C2",
        "",
        "## Core Metrics",
        f"- C1 intent_accuracy: {c1['intent_accuracy']}",
        f"- C3 intent_accuracy: {c3['intent_accuracy']}",
        f"- C3 profile_aligned_accuracy: {c3.get('profile_aligned_accuracy')}",
        f"- C1 entropy_drop_mean: {c1['m2_metrics']['belief_entropy_drop_mean']}",
        f"- C3 entropy_drop_mean: {c3['m2_metrics']['belief_entropy_drop_mean']}",
        "",
        "## Attribution Reading",
        "- If C3 > C1 on profile-aligned subset, privacy memory helps when history matches intent.",
        "- If C3 ~= C1, history prior may be washed out by STV likelihood or small sample size.",
        "",
        "## Privacy",
        "- M3 stores aggregate counts only; no raw dialogue in profile store.",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    analysis = [
        "# M1M2M3 运行分析（M3 阶段）",
        "",
        f"- C1 intent_accuracy: {c1['intent_accuracy']}",
        f"- C3 intent_accuracy: {c3['intent_accuracy']}",
        f"- C3 profile_aligned_accuracy: {c3.get('profile_aligned_accuracy')} (n={c3.get('profile_aligned_n')})",
        "",
        "## 结论（pilot）",
        "- C3 在 PipelineConfig(enable_m3=True) 下自动串联 M1→M3→M2。",
        "- 样本量较小，结论仅供阶段验收；扩样后重跑 attribution。",
    ]
    (out_dir / "m3_analysis_cn.md").write_text("\n".join(analysis), encoding="utf-8")
    return out_json, out_md


def main() -> None:
    rows = _load_rows()
    # Use full hypothesis space to avoid category prior bias.
    hypotheses = generate_hypothesis_space(max_size=1000)
    store = _load_profile_store()
    report = {
        "dataset": {"total_samples": len(rows), "sources": ["m1_eval_samples", "m1_hard_cases"]},
        "conditions": {
            "C0_llm_only": _run_condition(rows, extractor_mode="llm", use_m2=False, label="C0", hypotheses=hypotheses),
            "C1_pipeline_m2": _run_condition(rows, extractor_mode="pipeline", use_m2=True, label="C1", hypotheses=hypotheses),
            "C2_llm_m2": _run_condition(rows, extractor_mode="llm", use_m2=True, label="C2", hypotheses=hypotheses),
            "C3_pipeline_m2_m3": _run_condition_c3(rows, hypotheses=hypotheses, store=store),
        },
    }
    out_json, out_md = _write_report(report)
    m3_json, m3_md = _write_m3_report(report)
    print(
        json.dumps(
            {
                "report_json": str(out_json),
                "report_md": str(out_md),
                "m3_report_json": str(m3_json),
                "m3_report_md": str(m3_md),
                "conditions": report["conditions"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
