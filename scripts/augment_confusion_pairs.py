"""Generate confusion-pair training rows for M1 (evidence-quality retrain).

Templates are hand-authored paraphrases of known failure *families*, not copies of
eval_scenarios.json utterances. Do not add eval gold texts here.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DEFAULT = ROOT / "data" / "annotations" / "m1_confusion_supplement.json"

# (text, service_category) — diverse surface forms for NB to learn separations.
SEED_ROWS: List[Dict[str, str]] = [
    # medical vs wellness / fitness
    {"text": "I need a doctor visit for a routine checkup, not a spa day", "service_category": "medical"},
    {"text": "Book a clinic slot tomorrow morning for a health checkup", "service_category": "medical"},
    {"text": "Looking for a medical center nearby for checkup, not massage", "service_category": "medical"},
    {"text": "GP appointment please, this is medical not wellness", "service_category": "medical"},
    {"text": "Hospital outpatient booking for bloodwork next week", "service_category": "medical"},
    {"text": "Urgent care clinic open now for a medical issue", "service_category": "medical"},
    {"text": "Schedule an annual physical with a physician", "service_category": "medical"},
    {"text": "Need a doctor appointment, skip the retreat options", "service_category": "medical"},
    {"text": "Health screening at a clinic, not a wellness package", "service_category": "medical"},
    {"text": "Any clinic for checkup near the office tomorrow", "service_category": "medical"},
    {"text": "Medical consultation downtown, routine checkup", "service_category": "medical"},
    {"text": "See a doctor about my symptoms, not spa therapy", "service_category": "medical"},
    {"text": "Book lab tests at the medical center this afternoon", "service_category": "medical"},
    {"text": "I want a clinical checkup rather than a wellness retreat", "service_category": "medical"},
    {"text": "Find a walk-in clinic for a quick medical check", "service_category": "medical"},
    # dining vs fitness/wellness
    {"text": "Any quick lunch spot nearby that is affordable", "service_category": "dining"},
    {"text": "Maybe Thai food or sushi tomorrow afternoon", "service_category": "dining"},
    {"text": "Hungry for pizza or ramen near me tonight", "service_category": "dining"},
    {"text": "Book a casual restaurant for lunch downtown", "service_category": "dining"},
    {"text": "Looking for a cafe or bistro for brunch this weekend", "service_category": "dining"},
    {"text": "I want Italian dinner, not a workout", "service_category": "dining"},
    {"text": "Food reservation for two, sushi place if possible", "service_category": "dining"},
    {"text": "Need somewhere to eat lunch close to the office", "service_category": "dining"},
    {"text": "Craving noodles or Thai, any dining options nearby", "service_category": "dining"},
    {"text": "Affordable meal spot for a quick lunch please", "service_category": "dining"},
    {"text": "Dinner reservation tonight, nothing fancy", "service_category": "dining"},
    {"text": "Grab a bite, maybe sushi, tomorrow afternoon", "service_category": "dining"},
    # travel vs dining/wellness/medical
    {"text": "Planning a trip with hotel and transport next month", "service_category": "travel"},
    {"text": "Flying out Friday morning, need a hotel near the airport", "service_category": "travel"},
    {"text": "Book a flight and hotel package for vacation", "service_category": "travel"},
    {"text": "I need travel options to get somewhere next week", "service_category": "travel"},
    {"text": "Airport transfer and hotel booking for my trip", "service_category": "travel"},
    {"text": "Looking at flights for a weekend getaway", "service_category": "travel"},
    {"text": "Hotel near the airport plus a return flight", "service_category": "travel"},
    {"text": "Trip planning: flights first, then lodging", "service_category": "travel"},
    {"text": "Vacation travel booking, not a restaurant", "service_category": "travel"},
    {"text": "Need to fly somewhere next week, hotel optional", "service_category": "travel"},
    # fitness vs wellness
    {"text": "Actually I want a proper workout, not a massage", "service_category": "fitness"},
    {"text": "Gym session this weekend, skip the spa", "service_category": "fitness"},
    {"text": "Looking for HIIT or gym, not wellness treatments", "service_category": "fitness"},
    {"text": "Fitness class nearby, intense workout please", "service_category": "fitness"},
    {"text": "I need exercise today, not relaxation spa", "service_category": "fitness"},
    {"text": "Book a gym slot for a quick training session", "service_category": "fitness"},
    {"text": "Workout first; yoga only if it is fitness focused", "service_category": "fitness"},
    {"text": "Personal training at the gym this evening", "service_category": "fitness"},
    # wellness (clear, not beauty / not dining)
    {"text": "I need something relaxing like a spa or massage", "service_category": "wellness"},
    {"text": "Spa afternoon to unwind, gentle massage preferred", "service_category": "wellness"},
    {"text": "Feeling exhausted, want a calm spa session nearby", "service_category": "wellness"},
    {"text": "Wellness massage tonight, nothing medical", "service_category": "wellness"},
    {"text": "Book a relaxing spa, not a clinic checkup", "service_category": "wellness"},
    {"text": "Unwind with a massage this weekend near me", "service_category": "wellness"},
    {"text": "Feeling wiped out, need a calming spa nearby soon", "service_category": "wellness"},
    {"text": "A relaxing weekend plan to help me unwind would help", "service_category": "wellness"},
    {"text": "Prefer a calm spot this weekend, not too far away", "service_category": "wellness"},
    {"text": "Need to decompress with a spa visit, not dinner", "service_category": "wellness"},
    {"text": "Quick wellness session tonight to relax, not food", "service_category": "wellness"},
    {"text": "Exhausted after work, book a massage nearby asap", "service_category": "wellness"},
    {"text": "A calm unwind session this weekend sounds perfect", "service_category": "wellness"},
    {"text": "Feeling drained, spa or gentle massage please", "service_category": "wellness"},
    # fitness vague (no dining cues)
    {"text": "Need a place for exercise now and also this weekend near office", "service_category": "fitness"},
    {"text": "Let's do a weekend workout session instead", "service_category": "fitness"},
    {"text": "Quick fitness session, gym preferred", "service_category": "fitness"},
]

TEMPLATES = {
    "medical": [
        "Need a {place} for a {check} {when}",
        "Book a {check} at a {place} {when}",
        "{place} appointment for {check}, not spa",
        "Looking for {place} nearby for {check}",
    ],
    "dining": [
        "Any {meal} {place} nearby, affordable",
        "Maybe {food} {when}",
        "Hungry for {food} near me",
        "Book {meal} at a {place} {when}",
    ],
    "travel": [
        "Book a {mode} and {stay} for my trip",
        "{mode} on {when}, {stay} near airport",
        "Planning travel with {stay} and {mode}",
    ],
    "fitness": [
        "{activity} session {when}, not massage",
        "Need a proper {activity}, skip spa",
        "Book {activity} at the gym {when}",
    ],
    "wellness": [
        "Spa or massage to {goal} {when}",
        "Want a relaxing {tx} nearby {when}",
    ],
}

SLOTS = {
    "medical": {
        "place": ["clinic", "doctor", "medical center", "GP office", "hospital outpatient"],
        "check": ["checkup", "health check", "consultation", "physical exam", "screening"],
        "when": ["tomorrow morning", "next week", "this afternoon", "asap", "nearby"],
    },
    "dining": {
        "meal": ["lunch", "dinner", "brunch", "quick lunch"],
        "place": ["spot", "restaurant", "cafe", "bistro"],
        "food": ["Thai", "sushi", "pizza", "ramen", "noodles", "Italian"],
        "when": ["tomorrow afternoon", "tonight", "this weekend", "near the office"],
    },
    "travel": {
        "mode": ["flight", "flights"],
        "stay": ["hotel", "airport hotel"],
        "when": ["Friday morning", "next week", "this weekend"],
    },
    "fitness": {
        "activity": ["workout", "gym", "HIIT", "training"],
        "when": ["this weekend", "tonight", "tomorrow morning"],
    },
    "wellness": {
        "goal": ["unwind", "relax", "calm down"],
        "tx": ["spa", "massage", "wellness session"],
        "when": ["tonight", "this weekend", "this afternoon"],
    },
}


def _fill(template: str, slots: Dict[str, List[str]], rng: random.Random) -> str:
    out = template
    for key, options in slots.items():
        token = "{" + key + "}"
        while token in out:
            out = out.replace(token, rng.choice(options), 1)
    return out


def generate_rows(extra_per_class: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    rows = [dict(r) for r in SEED_ROWS]
    for cat, templates in TEMPLATES.items():
        slots = SLOTS[cat]
        for _ in range(extra_per_class):
            text = _fill(rng.choice(templates), slots, rng)
            rows.append({"text": text, "service_category": cat, "source": "confusion_augment"})
    # tag seeds
    for r in rows:
        r.setdefault("source", "confusion_seed")
    # dedupe by normalized text
    seen = set()
    unique: List[Dict[str, str]] = []
    for r in rows:
        key = " ".join(r["text"].lower().split())
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    rng.shuffle(unique)
    return unique


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _eval_scenario_texts(scenarios_path: Path | None = None) -> set[str]:
    paths = [
        ROOT / "data" / "annotations" / "eval_scenarios.json",
        ROOT / "data" / "annotations" / "eval_scenarios_v2.json",
    ]
    if scenarios_path is not None:
        paths.append(scenarios_path)
    blocked: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for s in rows:
            for t in s.get("turns") or []:
                blocked.add(_norm(str(t.get("text", ""))))
    return blocked


def merge_into_composite(
    supplement: List[Dict[str, str]],
    composite_path: Path,
    out_path: Path,
    *,
    blocked_texts: set[str] | None = None,
) -> int:
    base = json.loads(composite_path.read_text(encoding="utf-8"))
    blocked = blocked_texts or set()
    # Drop any eval-scenario utterances already present in the base composite.
    cleaned = [x for x in base if _norm(x.get("text", "")) not in blocked]
    removed = len(base) - len(cleaned)
    existing = {_norm(x["text"]) for x in cleaned}
    added = 0
    for row in supplement:
        key = _norm(row["text"])
        if key in existing or key in blocked:
            continue
        cleaned.append(
            {
                "text": row["text"],
                "service_category": row["service_category"],
                "source": row.get("source", "confusion_augment"),
                "source_intent": row["service_category"],
                "temporal_preference": None,
                "urgency_level": "medium",
                "location_context": None,
                "sentiment_tags": [],
            }
        )
        existing.add(key)
        added += 1
    out_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")
    if removed:
        print(f"Removed {removed} eval-scenario leaks from composite base")
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--extra-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--merge-composite", action="store_true")
    ap.add_argument(
        "--composite-in",
        type=Path,
        default=ROOT / "data" / "annotations" / "m1_train_composite.json",
    )
    ap.add_argument(
        "--composite-out",
        type=Path,
        default=ROOT / "data" / "annotations" / "m1_train_composite_eq.json",
    )
    args = ap.parse_args()

    rows = generate_rows(args.extra_per_class, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} confusion rows -> {args.out}")

    if args.merge_composite:
        blocked = _eval_scenario_texts(ROOT / "data" / "annotations" / "eval_scenarios.json")
        added = merge_into_composite(
            rows,
            args.composite_in,
            args.composite_out,
            blocked_texts=blocked,
        )
        print(f"Merged +{added} into {args.composite_out} (from {args.composite_in})")
        print(f"Blocked eval utterances: {len(blocked)}")


if __name__ == "__main__":
    main()
