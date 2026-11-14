"""Build eval_scenarios_v2.json: keep legacy 14 + enlarge each class to ~7.

Does not modify eval_scenarios.json (v4 freeze). New utterances must stay out of train.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT / "data" / "annotations" / "eval_scenarios.json"
OUT_DEFAULT = ROOT / "data" / "annotations" / "eval_scenarios_v2.json"

TARGET_PER_CLASS = 7


def _sc(
    scenario_id: str,
    description: str,
    user_id: str,
    gold: str,
    turns: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "scenario_id": scenario_id,
        "description": description,
        "user_id": user_id,
        "gold_category": gold,
        "turns": turns,
        "pack": "v2_enlarge",
    }


def _t(text: str, **hints: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {"text": text}
    if hints:
        row["gold_hints"] = hints
    return row


NEW_SCENARIOS: List[Dict[str, Any]] = [
    # --- beauty (+6 → 7) ---
    _sc(
        "beauty_nail_weekend",
        "Nail care request clarified over two turns",
        "persona_beauty_budget",
        "beauty",
        [
            _t("My nails look rough, need something this weekend", service_category="beauty", temporal_preference="weekend"),
            _t("Book a manicure nearby Saturday afternoon", service_category="beauty", temporal_preference="afternoon", location_context="nearby"),
        ],
    ),
    _sc(
        "beauty_salon_hair",
        "Hair salon booking with location",
        "persona_beauty_budget",
        "beauty",
        [
            _t("Looking for a salon for a haircut tomorrow", service_category="beauty", temporal_preference="tomorrow"),
            _t("Somewhere downtown if possible", service_category="beauty", location_context="city_center"),
        ],
    ),
    _sc(
        "beauty_facial_clear",
        "Clear single-turn facial booking",
        "persona_beauty_budget",
        "beauty",
        [_t("Book a facial appointment near me this evening", service_category="beauty", temporal_preference="evening", location_context="nearby")],
    ),
    _sc(
        "beauty_vs_medical_skin",
        "Skin concern resolves to cosmetic beauty care",
        "persona_beauty_budget",
        "beauty",
        [
            _t("Skin looks dull, not sure if I need a doctor", service_category="beauty"),
            _t("Just cosmetic care, maybe a facial, not a clinic", service_category="beauty"),
            _t("Schedule beauty facial tomorrow morning nearby", service_category="beauty", temporal_preference="morning", location_context="nearby"),
        ],
    ),
    _sc(
        "beauty_makeup_event",
        "Makeup / beauty prep before an event",
        "persona_beauty_budget",
        "beauty",
        [
            _t("Need help looking presentable before a party tonight", service_category="beauty", temporal_preference="evening"),
            _t("Beauty makeup session near me if available", service_category="beauty", location_context="nearby"),
        ],
    ),
    _sc(
        "beauty_brows_lash",
        "Brow/lash beauty service",
        "persona_beauty_budget",
        "beauty",
        [
            _t("Can I get my brows done this afternoon", service_category="beauty", temporal_preference="afternoon"),
            _t("Salon beauty service, not medical", service_category="beauty"),
        ],
    ),
    # --- medical (+5 → 7) ---
    _sc(
        "medical_gp_followup",
        "GP follow-up appointment",
        "persona_medical_cautious",
        "medical",
        [
            _t("Need a follow-up with my doctor next week", service_category="medical"),
            _t("Clinic visit for review of test results", service_category="medical"),
        ],
    ),
    _sc(
        "medical_urgent_care",
        "Urgent care / walk-in clinic",
        "persona_medical_cautious",
        "medical",
        [
            _t("I feel unwell and need a medical clinic today", service_category="medical", urgency_level="high"),
            _t("Walk-in doctor appointment nearby asap", service_category="medical", location_context="nearby", urgency_level="high"),
        ],
    ),
    _sc(
        "medical_vaccination",
        "Vaccination / shot booking",
        "persona_medical_cautious",
        "medical",
        [_t("Book a vaccination appointment at a clinic this week", service_category="medical")],
    ),
    _sc(
        "medical_specialist_referral",
        "Specialist consultation booking",
        "persona_medical_cautious",
        "medical",
        [
            _t("Looking for a specialist consultation downtown", service_category="medical", location_context="city_center"),
            _t("Medical appointment, not a spa treatment", service_category="medical"),
        ],
    ),
    _sc(
        "medical_lab_bloodwork",
        "Lab / bloodwork scheduling",
        "persona_medical_cautious",
        "medical",
        [
            _t("Need lab bloodwork scheduled tomorrow morning", service_category="medical", temporal_preference="morning"),
            _t("Any medical center nearby for tests", service_category="medical", location_context="nearby"),
        ],
    ),
    # --- dining (+5 → 7) ---
    _sc(
        "dining_dinner_date",
        "Dinner reservation for two",
        "persona_mixed_explorer",
        "dining",
        [
            _t("Want to eat out tonight, somewhere nice", service_category="dining", temporal_preference="evening"),
            _t("Reserve a restaurant table for two downtown", service_category="dining", location_context="city_center"),
        ],
    ),
    _sc(
        "dining_brunch_weekend",
        "Weekend brunch plan",
        "persona_mixed_explorer",
        "dining",
        [
            _t("Brunch ideas for this weekend?", service_category="dining", temporal_preference="weekend"),
            _t("Cafe or bistro nearby Saturday morning", service_category="dining", temporal_preference="morning", location_context="nearby"),
        ],
    ),
    _sc(
        "dining_hungry_takeout",
        "Hungry → takeout / meal",
        "persona_mixed_explorer",
        "dining",
        [
            _t("I'm starving, need food near me now", service_category="dining", urgency_level="high", location_context="nearby"),
            _t("Maybe pizza takeout for dinner", service_category="dining", temporal_preference="evening"),
        ],
    ),
    _sc(
        "dining_ramen_night",
        "Specific cuisine dinner",
        "persona_mixed_explorer",
        "dining",
        [_t("Find a ramen place for dinner tomorrow evening", service_category="dining", temporal_preference="evening")],
    ),
    _sc(
        "dining_vs_wellness_food",
        "Clarify food not spa",
        "persona_mixed_explorer",
        "dining",
        [
            _t("I need something tonight but not sure what", service_category="dining", temporal_preference="evening"),
            _t("Actually just dinner, a restaurant, not a spa", service_category="dining"),
        ],
    ),
    # --- travel (+5 → 7) ---
    _sc(
        "travel_weekend_getaway",
        "Weekend trip planning",
        "persona_cold_start",
        "travel",
        [
            _t("Thinking about a short getaway this weekend", service_category="travel", temporal_preference="weekend"),
            _t("Maybe a train or flight plus a hotel", service_category="travel"),
        ],
    ),
    _sc(
        "travel_hotel_only",
        "Hotel booking focus",
        "persona_cold_start",
        "travel",
        [
            _t("Need a hotel in the city center next Friday", service_category="travel", location_context="city_center"),
            _t("Two nights, travel lodging only", service_category="travel"),
        ],
    ),
    _sc(
        "travel_airport_transfer",
        "Flight + airport logistics",
        "persona_cold_start",
        "travel",
        [
            _t("Flying in Sunday evening, need airport hotel", service_category="travel", temporal_preference="evening"),
            _t("Travel booking near the airport please", service_category="travel", location_context="nearby"),
        ],
    ),
    _sc(
        "travel_vacation_vague",
        "Vague vacation → travel",
        "persona_cold_start",
        "travel",
        [
            _t("I want to go away for a few days next month", service_category="travel"),
            _t("Book flights first, hotel can wait", service_category="travel"),
        ],
    ),
    _sc(
        "travel_vs_dining_trip",
        "Trip not restaurant",
        "persona_cold_start",
        "travel",
        [
            _t("Planning something for next week, not sure yet", service_category="travel"),
            _t("It's a trip with flights, not a dinner plan", service_category="travel"),
        ],
    ),
    # --- wellness (+4 → 7) ---
    _sc(
        "wellness_massage_clear",
        "Clear massage booking",
        "persona_premium_wellness",
        "wellness",
        [_t("Book a deep tissue massage nearby tomorrow afternoon", service_category="wellness", temporal_preference="afternoon", location_context="nearby")],
    ),
    _sc(
        "wellness_sauna_relax",
        "Sauna / relax session",
        "persona_premium_wellness",
        "wellness",
        [
            _t("I need to unwind after work tonight", service_category="wellness", temporal_preference="evening"),
            _t("Sauna or spa session, not a gym workout", service_category="wellness"),
        ],
    ),
    _sc(
        "wellness_hot_stone",
        "Hot stone spa package",
        "persona_premium_wellness",
        "wellness",
        [
            _t("Any calming spa treatments this weekend", service_category="wellness", temporal_preference="weekend"),
            _t("Hot stone massage if available downtown", service_category="wellness", location_context="city_center"),
        ],
    ),
    _sc(
        "wellness_vs_fitness_recover",
        "Recovery wellness not training",
        "persona_wellness_weekend",
        "wellness",
        [
            _t("Sore and tired, need recovery not more exercise", service_category="wellness"),
            _t("Gentle spa recovery session near me asap", service_category="wellness", location_context="nearby", urgency_level="high"),
        ],
    ),
    # --- fitness (+3 → 7) ---
    _sc(
        "fitness_hiit_evening",
        "HIIT class evening",
        "persona_fitness_heavy",
        "fitness",
        [_t("Book a HIIT class near the office tonight", service_category="fitness", temporal_preference="evening", location_context="city_center")],
    ),
    _sc(
        "fitness_pt_session",
        "Personal training session",
        "persona_fitness_heavy",
        "fitness",
        [
            _t("Looking for personal training this weekend", service_category="fitness", temporal_preference="weekend"),
            _t("Gym PT session, not massage therapy", service_category="fitness"),
        ],
    ),
    _sc(
        "fitness_strength_morning",
        "Morning strength training",
        "persona_urgent_fitness",
        "fitness",
        [
            _t("Need a strength workout tomorrow morning nearby", service_category="fitness", temporal_preference="morning", location_context="nearby"),
            _t("Weight training at the gym please", service_category="fitness"),
        ],
    ),
]


def build(legacy_path: Path) -> List[Dict[str, Any]]:
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    out: List[Dict[str, Any]] = []
    for row in legacy:
        r = dict(row)
        r["pack"] = "v1_legacy"
        out.append(r)
    seen = {r["scenario_id"] for r in out}
    for row in NEW_SCENARIOS:
        if row["scenario_id"] in seen:
            raise ValueError(f"duplicate id {row['scenario_id']}")
        out.append(row)
        seen.add(row["scenario_id"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--legacy", type=Path, default=LEGACY)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = ap.parse_args()

    rows = build(args.legacy)
    counts = Counter(r["gold_category"] for r in rows)
    short = {c: n for c, n in counts.items() if n < TARGET_PER_CLASS}
    if short:
        print("WARNING under target:", short, file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_scenarios": len(rows),
                "n_turns": sum(len(r["turns"]) for r in rows),
                "by_category": dict(sorted(counts.items())),
                "legacy": sum(1 for r in rows if r.get("pack") == "v1_legacy"),
                "new": sum(1 for r in rows if r.get("pack") == "v2_enlarge"),
                "target_per_class": TARGET_PER_CLASS,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
