from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List

from src.config.paths import rel_to_root
from src.m1_nlu.ambiguity import ambiguity_bucket
from src.m1_nlu.dataset import M1Sample, build_bio_labels
from src.m1_nlu.factory import get_extractor


def _slot_f1(pred: List[str], gold: List[str], positive_prefix: str) -> Dict[str, float]:
    tp = fp = fn = 0
    for p, g in zip(pred, gold):
        # OLD:
        # p_pos = p.startswith(positive_prefix)
        # g_pos = g.startswith(positive_prefix)
        # NEW: count both B- and I- for each slot family
        slot_name = positive_prefix.replace("B-", "")
        p_pos = p.endswith(slot_name) and p[0] in {"B", "I"}
        g_pos = g.endswith(slot_name) and g[0] in {"B", "I"}
        if p_pos and g_pos:
            tp += 1
        elif p_pos and not g_pos:
            fp += 1
        elif (not p_pos) and g_pos:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _predict_bio_from_stv(text: str, stv: Dict[str, object]) -> List[str]:
    sample = M1Sample(
        text=text,
        service_category=str(stv.get("service_category", "other")),
        temporal_preference=stv.get("temporal_preference"),  # type: ignore[arg-type]
        urgency_level=str(stv.get("urgency_level", "medium")),
        location_context=stv.get("location_context"),  # type: ignore[arg-type]
        sentiment_tags=list(stv.get("sentiment_tags", [])),  # type: ignore[arg-type]
    )
    _, labels = build_bio_labels(sample)
    return labels


def _extract_error_cases(rows: List[Dict[str, str]], predictions: List[Dict[str, str]], limit: int = 20) -> List[Dict[str, str]]:
    errors: List[Dict[str, str]] = []
    for row, pred in zip(rows, predictions):
        if row["service_category"] != pred["service_category"]:
            errors.append(
                {
                    "text": row["text"],
                    "gold_service_category": row["service_category"],
                    "pred_service_category": pred["service_category"],
                }
            )
        if len(errors) >= limit:
            break
    return errors


def evaluate_dataset(dataset_path: str, mode: str, *, by_ambiguity: bool = False) -> Dict[str, object]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        rows: List[Dict[str, str]] = json.load(f)
    ext = get_extractor(mode)

    total = len(rows)
    intent_hit = 0
    schema_valid = 0
    ambiguous_count = 0
    cm: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    temporal_scores = []
    urgency_scores = []
    location_scores = []
    predictions: List[Dict[str, str]] = []
    bucket_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "intent_hit": 0, "ambiguous_pred": 0}
    )

    for row in rows:
        stv = ext.extract(row["text"])
        predictions.append({"service_category": stv.service_category})
        if stv.service_category == row["service_category"]:
            intent_hit += 1
        cm[row["service_category"]][stv.service_category] += 1
        if stv.raw_input and 0.0 <= stv.extraction_confidence <= 1.0:
            schema_valid += 1
        if stv.service_category == "ambiguous":
            ambiguous_count += 1

        if by_ambiguity:
            bucket = ambiguity_bucket(row["text"], row)
            bucket_stats[bucket]["total"] += 1
            if stv.service_category == row["service_category"]:
                bucket_stats[bucket]["intent_hit"] += 1
            if stv.service_category == "ambiguous":
                bucket_stats[bucket]["ambiguous_pred"] += 1

        gold_sample = M1Sample(
            text=row["text"],
            service_category=row["service_category"],
            temporal_preference=row.get("temporal_preference"),
            urgency_level=row.get("urgency_level", "medium"),
            location_context=row.get("location_context"),
            sentiment_tags=row.get("sentiment_tags", []),
        )
        _, gold_bio = build_bio_labels(gold_sample)
        pred_bio = _predict_bio_from_stv(
            row["text"],
            {
                "service_category": stv.service_category,
                "temporal_preference": stv.temporal_preference,
                "urgency_level": stv.urgency_level,
                "location_context": stv.location_context,
                "sentiment_tags": stv.sentiment_tags,
            },
        )
        temporal_scores.append(_slot_f1(pred_bio, gold_bio, "B-TEMPORAL"))
        urgency_scores.append(_slot_f1(pred_bio, gold_bio, "B-URGENCY"))
        location_scores.append(_slot_f1(pred_bio, gold_bio, "B-LOCATION"))

    def avg_f1(items: List[Dict[str, float]]) -> float:
        return sum(x["f1"] for x in items) / len(items) if items else 0.0

    slot_macro_f1 = (avg_f1(temporal_scores) + avg_f1(urgency_scores) + avg_f1(location_scores)) / 3.0

    confusion_matrix = {k: dict(v) for k, v in cm.items()}
    error_cases = _extract_error_cases(rows, predictions, limit=20)

    result: Dict[str, object] = {
        "dataset": rel_to_root(dataset_path),
        "mode": mode,
        "intent_accuracy": round(intent_hit / total, 4) if total else 0.0,
        "schema_valid_rate": round(schema_valid / total, 4) if total else 0.0,
        "ambiguous_ratio": round(ambiguous_count / total, 4) if total else 0.0,
        "slot_f1_temporal": round(avg_f1(temporal_scores), 4),
        "slot_f1_urgency": round(avg_f1(urgency_scores), 4),
        "slot_f1_location": round(avg_f1(location_scores), 4),
        "slot_f1_macro": round(slot_macro_f1, 4),
        "confusion_matrix": confusion_matrix,
        "error_cases": error_cases,
    }
    if by_ambiguity:
        by_bucket: Dict[str, Dict[str, float]] = {}
        for bucket, stats in sorted(bucket_stats.items()):
            n = stats["total"] or 1
            by_bucket[bucket] = {
                "count": stats["total"],
                "intent_accuracy": round(stats["intent_hit"] / n, 4),
                "ambiguous_pred_rate": round(stats["ambiguous_pred"] / n, 4),
            }
        result["by_ambiguity_bucket"] = by_bucket
    return result
