"""Template + paraphrase augmentation for fitness/wellness (balance short classes).

Does not copy eval_scenarios utterances. Used by build_m1_composite_train.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_m1_composite_train import (  # noqa: E402
    CompositeRow,
    infer_slots_from_text,
    is_filler,
    normalize_text,
)

DEFAULT_SEED_PATH = ROOT / "data" / "annotations" / "m1_train.json"
DEFAULT_OUT_PATH = ROOT / "data" / "annotations" / "m1_train_augmented.json"
EVAL_SCENARIOS_PATH = ROOT / "data" / "annotations" / "eval_scenarios.json"

FITNESS_ACTIVITIES = (
    "workout class",
    "gym session",
    "HIIT class",
    "cardio class",
    "strength training",
    "personal training",
    "yoga class",
    "pilates session",
    "fitness coaching",
    "group exercise",
    "bootcamp",
    "interval training",
    "spin class",
    "crossfit session",
    "gym workout",
    "training session",
    "exercise class",
    "weight training",
    "running group",
    "boxing class",
)

WELLNESS_ACTIVITIES = (
    "spa session",
    "massage",
    "wellness therapy",
    "recovery massage",
    "body care",
    "stress relief treatment",
    "spa package",
    "wellness appointment",
    "soothing massage",
    "relaxation treatment",
    "sauna session",
    "hot stone massage",
    "calm spa visit",
    "gentle massage",
    "wellness recovery",
    "aromatherapy massage",
    "relaxation spa",
    "decompress session",
)

OPENERS = (
    "Book",
    "Find",
    "Need",
    "Schedule",
    "I want",
    "Looking for",
    "Any",
    "Can I get",
    "Please book",
    "Help me find",
    "I'd like",
)

TEMPORAL_PHRASES: Dict[str, str] = {
    "ASAP": "ASAP",
    "evening": "tonight",
    "morning": "tomorrow morning",
    "afternoon": "this afternoon",
    "weekend": "this weekend",
    "tomorrow": "tomorrow",
}

LOCATION_PHRASES: Dict[str, str] = {
    "nearby": "nearby",
    "city_center": "in city center",
    "near_office": "near the office",
    "not_far": "not too far",
}

SENTIMENT_BY_CATEGORY: Dict[str, tuple[str, ...]] = {
    "fitness": ("quick", "affordable", "intense", "urgent", "premium", "low-cost"),
    "wellness": ("relaxing", "affordable", "quick", "premium", "calming", "soothing"),
}

URGENCY_BY_SENTIMENT: Dict[str, str] = {
    "urgent": "high",
    "quick": "high",
    "affordable": "low",
    "low-cost": "low",
    "premium": "low",
    "intense": "medium",
    "relaxing": "medium",
    "calming": "medium",
    "soothing": "medium",
}


@dataclass
class AugmentConfig:
    # Keep moderate defaults: heavy 350–400 template fill regressed scenario C1 (see thesis_eval/v5).
    fitness_target: int = 200
    wellness_target: int = 200
    seed: int = 42


def _eval_blocked_texts(path: Path = EVAL_SCENARIOS_PATH) -> Set[str]:
    paths = [
        path,
        ROOT / "data" / "annotations" / "eval_scenarios_v2.json",
    ]
    blocked: Set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        for s in rows:
            for t in s.get("turns") or []:
                key = normalize_text(str(t.get("text", "")))
                if key:
                    blocked.add(key)
    return blocked


def _load_seeds(path: Path, category: str) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [r for r in rows if r.get("service_category") == category]


def _row(text: str, category: str, source_intent: str, **kwargs: Any) -> CompositeRow:
    temporal, urgency, location = infer_slots_from_text(text)
    return CompositeRow(
        text=text,
        service_category=category,
        source="augment",
        source_intent=source_intent,
        temporal_preference=kwargs.get("temporal_preference", temporal),
        urgency_level=kwargs.get("urgency_level", urgency),
        location_context=kwargs.get("location_context", location),
        sentiment_tags=list(kwargs.get("sentiment_tags") or []),
    )


def _vague_templates(category: str) -> Iterable[CompositeRow]:
    """Vague / boundary utterances — paraphrases only, not eval copies."""
    if category == "wellness":
        texts = (
            "I don't know maybe something for body and mood tonight maybe",
            "Something to unwind not sure what near me whenever",
            "Feeling wiped out, need a calming spa nearby soon",
            "Need help recovering this weekend somehow",
            "Maybe downtown spa this afternoon if affordable",
            "Looking for relaxation nearby not sure what exactly",
            "Need mood relief and recovery near me soon",
            "Something calming for body tonight maybe not too far",
            "Maybe massage or spa weekend downtown",
            "Not sure what I need but want to relax nearby",
            "Exhausted need wellness recovery asap near me",
            "Body care and mood relief this weekend nearby",
            "Unwind and recover somewhere close tonight",
            "A calm unwind session this weekend sounds perfect",
            "Prefer a calm spot this weekend, not too far away",
            "Feeling drained, spa or gentle massage please",
            "Need to decompress with a spa visit, not dinner",
            "Quick wellness session tonight to relax, not food",
            "Something relaxing after work would really help",
            "I want to unwind tonight, massage if possible",
        )
    else:
        texts = (
            "Need exercise now and also later this weekend near office",
            "Gym or yoga I can't decide asap",
            "Looking for an affordable workout tomorrow evening",
            "I don't know gym or class maybe tonight",
            "Need quick training near office not sure which",
            "Maybe workout this weekend downtown not sure",
            "Looking for exercise nearby whenever works",
            "Need gym slot now or tomorrow morning",
            "Not sure yoga or HIIT but need it soon",
            "Want a proper training session, not a spa",
            "Quick fitness session near the office please",
            "Book gym time this weekend for a real workout",
            "Need to move my body, gym preferred tonight",
            "Looking for HIIT or strength training nearby",
        )
    for text in texts:
        yield _row(text, category, f"vague_{category}")


def _contrastive_templates(category: str) -> Iterable[CompositeRow]:
    if category == "fitness":
        texts = (
            "Actually I want a proper workout, not a massage",
            "Gym session this weekend, skip the spa",
            "Looking for HIIT or gym, not wellness treatments",
            "I need exercise today, not relaxation spa",
            "Book a gym slot for training, not spa therapy",
            "Fitness class nearby, intense workout please",
            "Personal training at the gym this evening",
            "Workout first; keep it fitness focused not massage",
            "Need strength training, please no spa packages",
            "Cardio class tomorrow, not a wellness retreat",
        )
    else:
        texts = (
            "Spa afternoon to unwind, not a gym workout",
            "Book a relaxing massage, not a clinic checkup",
            "Wellness massage tonight, nothing medical",
            "I want spa recovery, not intense fitness",
            "Calm massage nearby, skip the workout options",
            "Need relaxation treatment, not HIIT",
            "Soothing spa package this weekend, not gym",
            "Decompress with massage, not exercise class",
            "Hot stone massage please, not personal training",
            "Wellness appointment to unwind, not medical",
        )
    for text in texts:
        yield _row(text, category, f"contrast_{category}")


def _template_variants(
    category: str,
    *,
    include_location: bool = True,
    include_sentiment: bool = True,
) -> Iterable[CompositeRow]:
    activities = FITNESS_ACTIVITIES if category == "fitness" else WELLNESS_ACTIVITIES
    sentiments = SENTIMENT_BY_CATEGORY.get(category, ())

    for opener in OPENERS:
        for activity in activities:
            for temporal_key, temporal_phrase in TEMPORAL_PHRASES.items():
                location_opts: List[Optional[str]] = [None]
                if include_location:
                    location_opts.extend(list(LOCATION_PHRASES.keys()))

                sentiment_opts: List[Optional[str]] = [None]
                if include_sentiment:
                    sentiment_opts.extend(list(sentiments))

                for location_key in location_opts:
                    for sentiment in sentiment_opts:
                        parts = [opener, "a" if opener not in {"Need", "Find", "Any"} else "", activity]
                        text = " ".join(p for p in parts if p)
                        if sentiment:
                            text = f"{sentiment.capitalize()} {text.lower()}"
                        text = f"{text} {temporal_phrase}"
                        if location_key:
                            text = f"{text} {LOCATION_PHRASES[location_key]}"
                        text = " ".join(text.split())

                        urgency = "medium"
                        if temporal_key == "ASAP":
                            urgency = "high"
                        if sentiment and sentiment in URGENCY_BY_SENTIMENT:
                            urgency = URGENCY_BY_SENTIMENT[sentiment]

                        tags: List[str] = []
                        if sentiment:
                            tags.append(sentiment)

                        yield CompositeRow(
                            text=text,
                            service_category=category,
                            source="augment",
                            source_intent=f"template_{category}",
                            temporal_preference=temporal_key,
                            urgency_level=urgency,
                            location_context=location_key,
                            sentiment_tags=tags,
                        )


def _seed_paraphrases(seeds: List[Dict[str, Any]], category: str) -> Iterable[CompositeRow]:
    temporal_swap = {
        "tonight": ("evening", "evening"),
        "tomorrow morning": ("morning", "morning"),
        "this afternoon": ("afternoon", "afternoon"),
        "this weekend": ("weekend", "weekend"),
        "ASAP": ("ASAP", "ASAP"),
        "now": ("ASAP", "ASAP"),
        "tomorrow": ("tomorrow", "tomorrow"),
        "tomorrow evening": ("evening", "evening"),
    }
    location_swap = {
        "nearby": "nearby",
        "near me": "nearby",
        "city center": "city_center",
        "downtown": "city_center",
        "in downtown": "city_center",
    }

    for seed in seeds:
        base = seed["text"].strip()
        for old, (temporal_key, phrase) in temporal_swap.items():
            if old not in base.lower():
                continue
            text = base
            for src, dst in location_swap.items():
                if src in text.lower():
                    text = text.replace(src, dst.replace("_", " "))
            text = text.replace(old, phrase)
            yield CompositeRow(
                text=text,
                service_category=category,
                source="augment",
                source_intent="seed_paraphrase",
                temporal_preference=temporal_key,
                urgency_level=seed.get("urgency_level", "medium"),
                location_context=seed.get("location_context"),
                sentiment_tags=list(seed.get("sentiment_tags", [])),
            )


def generate_augmented_rows(
    seed_path: Path,
    *,
    fitness_target: int = 200,
    wellness_target: int = 200,
    seed: int = 42,
    blocked_texts: Optional[Set[str]] = None,
) -> List[CompositeRow]:
    fitness_seeds = _load_seeds(seed_path, "fitness")
    wellness_seeds = _load_seeds(seed_path, "wellness")
    blocked = blocked_texts if blocked_texts is not None else _eval_blocked_texts()

    rnd = random.Random(seed)

    tiers: List[List[CompositeRow]] = [
        list(_contrastive_templates("fitness")),
        list(_contrastive_templates("wellness")),
        list(_vague_templates("fitness")),
        list(_vague_templates("wellness")),
        list(_seed_paraphrases(fitness_seeds, "fitness")),
        list(_seed_paraphrases(wellness_seeds, "wellness")),
        list(_template_variants("fitness", include_sentiment=False)),
        list(_template_variants("wellness", include_sentiment=False)),
        list(_template_variants("fitness", include_sentiment=True)),
        list(_template_variants("wellness", include_sentiment=True)),
    ]
    for tier in tiers:
        rnd.shuffle(tier)

    kept: Dict[str, CompositeRow] = {}
    counts = {"fitness": 0, "wellness": 0}
    targets = {"fitness": fitness_target, "wellness": wellness_target}
    # Limit near-duplicate slot fills: max per (cat, activity-ish intent bucket)
    bucket_counts: Dict[str, int] = {}
    bucket_cap = 24

    for tier in tiers:
        for row in tier:
            cat = row.service_category
            if counts[cat] >= targets[cat]:
                continue
            if is_filler(row.text):
                continue
            key = normalize_text(row.text)
            if not key or key in kept or key in blocked:
                continue
            bucket = f"{cat}:{row.source_intent}:{row.temporal_preference}:{row.location_context}"
            if row.source_intent.startswith("template_"):
                if bucket_counts.get(bucket, 0) >= bucket_cap:
                    continue
            kept[key] = row
            counts[cat] += 1
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
            if all(counts[c] >= targets[c] for c in targets):
                break
        if all(counts[c] >= targets[c] for c in targets):
            break

    return list(kept.values())


def write_augmented_json(rows: List[CompositeRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment fitness/wellness training rows from seeds")
    parser.add_argument("--seed-file", default=str(DEFAULT_SEED_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--fitness-target", type=int, default=200)
    parser.add_argument("--wellness-target", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_augmented_rows(
        Path(args.seed_file),
        fitness_target=args.fitness_target,
        wellness_target=args.wellness_target,
        seed=args.seed,
    )
    write_augmented_json(rows, Path(args.out))

    from collections import Counter

    counts = Counter(r.service_category for r in rows)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n": len(rows),
                "counts": dict(counts),
                "blocked_eval_utterances": len(_eval_blocked_texts()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
