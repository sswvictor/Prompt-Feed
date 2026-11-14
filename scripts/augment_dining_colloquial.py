from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_m1_composite_train import (  # noqa: E402
    CompositeRow,
    is_filler,
    normalize_text,
)

DEFAULT_OUT_PATH = ROOT / "data" / "annotations" / "m1_train_augmented_dining.json"

# Hand-authored colloquial dining seeds (demo + hard-case coverage).
COLLOQUIAL_SEEDS: List[str] = [
    "I'm a bit hungry actually",
    "I'm hungry, need something to eat nearby",
    "Getting pretty hungry, any food options?",
    "I want to grab a bite",
    "Craving something to eat this afternoon",
    "Starving, what's good around here for lunch?",
    "I want to have my lunch",
    "Need dinner tonight, nothing too fancy",
    "Brunch this weekend somewhere casual",
    "Quick lunch near the office",
    "Late night snack or meal nearby",
    "Buffet, I'm hungry",
    "Seafood buffet sounds good",
    "Any buffet lunch near me?",
    "All you can eat place for tonight",
    "I'm hungry, not sure what cuisine yet",
    "Maybe Italian or Thai, I'm pretty hungry",
    "Something to eat, don't know what though",
    "Food-wise I'm open, just hungry now",
    "Book lunch for two nearby tomorrow",
    "Find a casual restaurant, I'm hungry",
    "Reserve a table, we're hungry after work",
    "What about eating? A bit hungry now",
    "I want some food near me",
    "Looking for a place to eat tonight",
    "Need a bite to eat ASAP",
    "Fancy a meal downtown this evening",
    "Anywhere for dinner that's affordable",
    "I could go for pizza or sushi, pretty hungry",
    "Hungry and want something quick nearby",
]

OPENERS = (
    "I'm",
    "Getting",
    "Feeling",
    "Kind of",
    "Pretty",
)

HUNGER_PHRASES = (
    "hungry",
    "peckish",
    "starving",
    "in the mood for food",
    "craving a meal",
)

MEAL_PHRASES = (
    "lunch",
    "dinner",
    "brunch",
    "a bite",
    "something to eat",
    "food",
    "a meal",
)

TEMPORAL = {
    "tonight": "evening",
    "tomorrow": "tomorrow",
    "this weekend": "weekend",
    "this afternoon": "afternoon",
    "now": "ASAP",
    "ASAP": "ASAP",
}

LOCATION = {
    "nearby": "nearby",
    "near me": "nearby",
    "downtown": "city_center",
    "close to office": "city_center",
}


def _seed_rows() -> Iterable[CompositeRow]:
    for text in COLLOQUIAL_SEEDS:
        yield CompositeRow(
            text=text,
            service_category="dining",
            source="augment",
            source_intent="colloquial_seed",
            temporal_preference=None,
            urgency_level="medium",
            location_context=None,
            sentiment_tags=[],
        )


def _template_rows_fixed() -> Iterable[CompositeRow]:
    rnd = random.Random(0)
    for hunger in HUNGER_PHRASES:
        for meal in MEAL_PHRASES:
            for loc_phrase, loc_key in LOCATION.items():
                opener = rnd.choice(OPENERS)
                text = f"{opener} {hunger}, want {meal} {loc_phrase}"
                yield CompositeRow(
                    text=text,
                    service_category="dining",
                    source="augment",
                    source_intent="colloquial_template",
                    temporal_preference=None,
                    urgency_level="medium",
                    location_context=loc_key,
                    sentiment_tags=[],
                )
    for temp_phrase, temp_key in TEMPORAL.items():
        yield CompositeRow(
            text=f"Hungry for dinner {temp_phrase}",
            service_category="dining",
            source="augment",
            source_intent="colloquial_template",
            temporal_preference=temp_key,
            urgency_level="high" if temp_key == "ASAP" else "medium",
            location_context=None,
            sentiment_tags=[],
        )
    for style in ("buffet", "seafood buffet", "Italian", "Thai", "casual cafe"):
        yield CompositeRow(
            text=f"I'm hungry, maybe {style} tonight",
            service_category="dining",
            source="augment",
            source_intent="colloquial_template",
            temporal_preference="evening",
            urgency_level="medium",
            location_context=None,
            sentiment_tags=[],
        )


def generate_dining_colloquial_rows(*, target: int = 80, seed: int = 42) -> List[CompositeRow]:
    rnd = random.Random(seed)
    tiers = [list(_seed_rows()), list(_template_rows_fixed())]
    for tier in tiers:
        rnd.shuffle(tier)

    kept: Dict[str, CompositeRow] = {}
    for tier in tiers:
        for row in tier:
            if len(kept) >= target:
                break
            if is_filler(row.text):
                continue
            key = normalize_text(row.text)
            if not key or key in kept:
                continue
            kept[key] = row
        if len(kept) >= target:
            break
    return list(kept.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment colloquial dining rows")
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))
    parser.add_argument("--target", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_dining_colloquial_rows(target=args.target, seed=args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.to_dict() for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "total": len(rows), "target": args.target}, indent=2))


if __name__ == "__main__":
    main()
