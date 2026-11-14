from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore, SqliteStore, create_store


def load_synthetic(path: Path) -> list[AggregatedUserProfile]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [AggregatedUserProfile.from_dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed M3 user profiles from synthetic JSON")
    parser.add_argument(
        "--source",
        default=str(ROOT / "data" / "profiles" / "synthetic_users.json"),
        help="Path to synthetic_users.json",
    )
    parser.add_argument("--backend", choices=["inmemory", "sqlite"], default="sqlite")
    parser.add_argument(
        "--sqlite-path",
        default=str(ROOT / "data" / "memory" / "profiles.db"),
    )
    args = parser.parse_args()

    profiles = load_synthetic(Path(args.source))
    store = create_store(args.backend, args.sqlite_path)
    for p in profiles:
        store.save_profile(p)

    out = {
        "backend": args.backend,
        "count": len(profiles),
        "user_ids": store.list_user_ids(),
    }
    if isinstance(store, InMemoryStore):
        out["note"] = "inmemory store is process-local"
    else:
        out["sqlite_path"] = args.sqlite_path
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
