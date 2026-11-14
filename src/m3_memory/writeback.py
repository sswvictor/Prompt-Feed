from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.m2_reasoning.hypothesis import GoalHypothesis
from src.m3_memory.profile import AggregatedUserProfile

if TYPE_CHECKING:
    from src.m3_memory.store import MemoryStore


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = int(counts.get(key, 0)) + 1


def record_session_outcome(
    store: MemoryStore,
    user_id: str,
    hypothesis: GoalHypothesis,
) -> AggregatedUserProfile:
    """Update aggregate counts after a completed booking session (no raw text)."""
    profile = store.get_profile(user_id)
    _increment(profile.category_counts, hypothesis.service_category)
    _increment(profile.temporal_counts, hypothesis.temporal_slot)
    _increment(profile.price_tier_counts, hypothesis.price_tier)
    profile.total_sessions += 1
    profile.last_updated = time.time()
    store.save_profile(profile)
    return profile
