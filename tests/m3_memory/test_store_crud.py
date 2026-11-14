from __future__ import annotations

import time
from pathlib import Path

from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore, SqliteStore


def _sample_profile(user_id: str = "u_store") -> AggregatedUserProfile:
    return AggregatedUserProfile(
        user_id=user_id,
        category_counts={"wellness": 7},
        temporal_counts={"weekend_anytime": 4},
        price_tier_counts={"mid": 5},
        last_updated=time.time(),
        total_sessions=7,
    )


def test_inmemory_crud() -> None:
    store = InMemoryStore()
    assert store.get_profile("missing").total_sessions == 0
    store.save_profile(_sample_profile())
    assert store.get_profile("u_store").category_counts["wellness"] == 7
    summary = store.get_summary("u_store")
    assert summary["total_sessions"] == 7
    assert "raw_input" not in summary
    assert store.delete_profile("u_store")
    assert store.get_profile("u_store").total_sessions == 0


def test_sqlite_crud(tmp_path: Path) -> None:
    db = tmp_path / "profiles.db"
    store = SqliteStore(db)
    store.save_profile(_sample_profile("u_sqlite"))
    assert store.list_user_ids() == ["u_sqlite"]
    loaded = store.get_profile("u_sqlite")
    assert loaded.category_counts["wellness"] == 7
    assert store.delete_profile("u_sqlite")
    assert store.get_profile("u_sqlite").total_sessions == 0
