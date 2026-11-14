from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Protocol, runtime_checkable

from src.m3_memory.profile import AggregatedUserProfile, empty_profile, profile_summary


@runtime_checkable
class MemoryStore(Protocol):
    def get_profile(self, user_id: str) -> AggregatedUserProfile: ...

    def save_profile(self, profile: AggregatedUserProfile) -> None: ...

    def delete_profile(self, user_id: str) -> bool: ...

    def list_user_ids(self) -> List[str]: ...

    def get_summary(self, user_id: str) -> Dict[str, object]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._profiles: Dict[str, AggregatedUserProfile] = {}

    def get_profile(self, user_id: str) -> AggregatedUserProfile:
        return self._profiles.get(user_id, empty_profile(user_id))

    def save_profile(self, profile: AggregatedUserProfile) -> None:
        self._profiles[profile.user_id] = profile

    def delete_profile(self, user_id: str) -> bool:
        return self._profiles.pop(user_id, None) is not None

    def list_user_ids(self) -> List[str]:
        return sorted(self._profiles.keys())

    def get_summary(self, user_id: str) -> Dict[str, object]:
        return profile_summary(self.get_profile(user_id))

    def load_many(self, profiles: List[AggregatedUserProfile]) -> None:
        for p in profiles:
            self.save_profile(p)


class SqliteStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def get_profile(self, user_id: str) -> AggregatedUserProfile:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if row is None:
            return empty_profile(user_id)
        return AggregatedUserProfile.from_dict(json.loads(row["payload"]))

    def save_profile(self, profile: AggregatedUserProfile) -> None:
        payload = json.dumps(profile.to_dict(), ensure_ascii=False)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, payload)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET payload = excluded.payload
                """,
                (profile.user_id, payload),
            )
            conn.commit()

    def delete_profile(self, user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_user_ids(self) -> List[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id FROM user_profiles ORDER BY user_id").fetchall()
        return [str(r["user_id"]) for r in rows]

    def get_summary(self, user_id: str) -> Dict[str, object]:
        return profile_summary(self.get_profile(user_id))


def create_store(backend: str = "inmemory", sqlite_path: str = "data/memory/profiles.db") -> MemoryStore:
    if backend == "inmemory":
        return InMemoryStore()
    if backend == "sqlite":
        return SqliteStore(sqlite_path)
    raise ValueError(f"Unsupported memory backend: {backend}")
