from __future__ import annotations

import time

from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore
from src.m3_memory.writeback import record_session_outcome
from src.m2_reasoning.hypothesis import generate_hypothesis_space


def test_record_session_outcome_increments_counts() -> None:
    store = InMemoryStore()
    uid = "u_test"
    store.save_profile(AggregatedUserProfile(user_id=uid))
    h = generate_hypothesis_space(max_size=10)[0]
    updated = record_session_outcome(store, uid, h)
    assert updated.total_sessions == 1
    assert updated.category_counts.get(h.service_category, 0) >= 1
    again = store.get_profile(uid)
    assert again.total_sessions == 1
