from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from src.config.paths import rel_to_root
from src.m1_nlu.factory import get_extractor

DEFAULT_ATIS_PATH = "data/benchmarks/atis_intent_mapped.json"


def load_public_benchmark(path: str | Path) -> List[Dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("benchmark file must be a JSON list")
    return data


def evaluate_public_benchmark(
    benchmark_path: str,
    mode: str = "pipeline",
) -> Dict[str, object]:
    rows = load_public_benchmark(benchmark_path)
    ext = get_extractor(mode)

    total = len(rows)
    intent_hit = 0
    cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_source: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "hit": 0})

    for row in rows:
        text = row["text"]
        gold = row["service_category"]
        stv = ext.extract(text)
        pred = stv.service_category
        if pred == gold:
            intent_hit += 1
        cm[gold][pred] += 1
        src = row.get("source_intent", "unknown")
        by_source[src]["total"] += 1
        if pred == gold:
            by_source[src]["hit"] += 1

    atis_only = [r for r in rows if r.get("source_intent") != "in_domain"]
    in_domain = [r for r in rows if r.get("source_intent") == "in_domain"]

    def _acc(subset: List[Dict[str, str]]) -> float:
        if not subset:
            return 0.0
        hits = sum(1 for r in subset if ext.extract(r["text"]).service_category == r["service_category"])
        return round(hits / len(subset), 4)

    return {
        "benchmark": rel_to_root(benchmark_path),
        "mode": mode,
        "total": total,
        "intent_accuracy": round(intent_hit / total, 4) if total else 0.0,
        "atis_travel_subset_accuracy": _acc(atis_only),
        "in_domain_subset_accuracy": _acc(in_domain),
        "confusion_matrix": {k: dict(v) for k, v in cm.items()},
        "by_source_intent": {
            k: {
                "count": v["total"],
                "accuracy": round(v["hit"] / v["total"], 4) if v["total"] else 0.0,
            }
            for k, v in sorted(by_source.items())
        },
    }


def format_report_md(report: Dict[str, object]) -> str:
    lines = [
        "# M1 Public Benchmark Report",
        "",
        f"- Benchmark: `{report['benchmark']}`",
        f"- Mode: `{report['mode']}`",
        f"- Overall intent accuracy: **{report['intent_accuracy']}**",
        f"- ATIS travel subset: **{report['atis_travel_subset_accuracy']}**",
        f"- In-domain sanity subset: **{report['in_domain_subset_accuracy']}**",
        "",
        "See `docs/m1_atis_mapping.md` for mapping policy.",
    ]
    return "\n".join(lines) + "\n"
