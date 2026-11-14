from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.evaluator import evaluate_dataset


def _latency(mode: str, rounds: int = 30) -> dict:
    from src.m1_nlu.factory import get_extractor

    ext = get_extractor(mode)
    sample = "Need quick affordable wellness nearby this weekend"
    vals = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        _ = ext.extract(sample)
        vals.append((time.perf_counter() - t0) * 1000.0)
    vals.sort()
    return {
        "mean_ms": sum(vals) / len(vals),
        "p50_ms": vals[len(vals) // 2],
        "p95_ms": vals[int((len(vals) - 1) * 0.95)],
    }


def main() -> None:
    normal = "data/annotations/m1_eval_samples.json"
    hard = "data/annotations/m1_hard_cases.json"
    report = {
        "pipeline_normal": evaluate_dataset(normal, "pipeline"),
        "llm_normal": evaluate_dataset(normal, "llm"),
        "pipeline_hard": evaluate_dataset(hard, "pipeline"),
        "llm_hard": evaluate_dataset(hard, "llm"),
        "latency_pipeline": _latency("pipeline"),
        "latency_llm": _latency("llm"),
    }

    out_json = Path("artifacts/m1/m1_completion_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# M1 Completion Report",
        "",
        "## Normal Set",
        f"- pipeline intent_acc: {report['pipeline_normal']['intent_accuracy']}",
        f"- llm intent_acc: {report['llm_normal']['intent_accuracy']}",
        f"- pipeline slot_f1_macro: {report['pipeline_normal']['slot_f1_macro']}",
        f"- llm slot_f1_macro: {report['llm_normal']['slot_f1_macro']}",
        "",
        "## Hard Set",
        f"- pipeline intent_acc: {report['pipeline_hard']['intent_accuracy']}",
        f"- llm intent_acc: {report['llm_hard']['intent_accuracy']}",
        "",
        "## Latency",
        f"- pipeline p50: {report['latency_pipeline']['p50_ms']:.4f} ms",
        f"- llm p50: {report['latency_llm']['p50_ms']:.4f} ms",
    ]
    Path("artifacts/m1/m1_completion_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
