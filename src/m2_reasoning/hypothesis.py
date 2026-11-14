from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List


@dataclass(frozen=True)
class GoalHypothesis:
    hypothesis_id: str
    service_category: str
    service_subtype: str
    temporal_slot: str
    location_zone: str
    price_tier: str


SERVICE_CATEGORIES: Dict[str, List[str]] = {
    "fitness": ["gym", "hiit"],
    "wellness": ["spa", "massage"],
    "beauty": ["facial", "nail"],
    "medical": ["checkup", "consultation"],
    "dining": ["restaurant_reservation", "takeout"],
    "travel": ["flight", "hotel"],
}


def _slot_grids_for_budget(per_category: int) -> tuple[List[str], List[str], List[str]]:
    """Shrink slot cardinalities so each category still gets subtype coverage."""
    # Full grid: 2 subtypes × 5 × 3 × 3 = 90 per category.
    if per_category >= 90:
        return (
            ["morning", "afternoon", "evening", "weekend_anytime", "asap"],
            ["nearby", "city_center", "any"],
            ["budget", "mid", "premium"],
        )
    if per_category >= 36:
        return (
            ["asap", "evening", "weekend_anytime", "afternoon"],
            ["nearby", "city_center", "any"],
            ["budget", "mid", "premium"],
        )
    if per_category >= 24:
        return (
            ["asap", "evening", "weekend_anytime"],
            ["nearby", "any"],
            ["budget", "mid"],
        )
    # Compact grid for ~120 total (~20/cat): 2×3×2×2 = 24, then truncate.
    return (
        ["asap", "evening", "weekend_anytime"],
        ["nearby", "any"],
        ["budget", "mid"],
    )


def generate_hypothesis_space(max_size: int = 120) -> List[GoalHypothesis]:
    """Build hypothesis space with even coverage across all service categories.

    Truncation must never drop whole categories (legacy sequential fill omitted
    dining/travel when max_size ≤ 360).
    """
    if max_size <= 0:
        return []

    n_cats = len(SERVICE_CATEGORIES)
    per_cat_budget = max(1, (max_size + n_cats - 1) // n_cats)
    temporal_slots, location_zones, price_tiers = _slot_grids_for_budget(per_cat_budget)

    per_cat_lists: List[List[GoalHypothesis]] = []
    for category, subtypes in SERVICE_CATEGORIES.items():
        cat_hs: List[GoalHypothesis] = []
        for subtype, temporal_slot, location_zone, price_tier in product(
            subtypes, temporal_slots, location_zones, price_tiers
        ):
            cat_hs.append(
                GoalHypothesis(
                    hypothesis_id="",  # assigned after round-robin
                    service_category=category,
                    service_subtype=subtype,
                    temporal_slot=temporal_slot,
                    location_zone=location_zone,
                    price_tier=price_tier,
                )
            )
        per_cat_lists.append(cat_hs)

    # Round-robin so residual slots (max_size % n_cats) stay balanced and size hits max_size.
    space: List[GoalHypothesis] = []
    idx = 0
    pointers = [0] * n_cats
    while len(space) < max_size:
        progressed = False
        for c_i, cat_hs in enumerate(per_cat_lists):
            if pointers[c_i] >= len(cat_hs):
                continue
            if len(space) >= max_size:
                break
            h = cat_hs[pointers[c_i]]
            pointers[c_i] += 1
            idx += 1
            space.append(
                GoalHypothesis(
                    hypothesis_id=f"h{idx:03d}",
                    service_category=h.service_category,
                    service_subtype=h.service_subtype,
                    temporal_slot=h.temporal_slot,
                    location_zone=h.location_zone,
                    price_tier=h.price_tier,
                )
            )
            progressed = True
        if not progressed:
            break
    return space
