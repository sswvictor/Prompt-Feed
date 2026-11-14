from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Set

FORBIDDEN_PROFILE_KEYS: Set[str] = {
    "raw_input",
    "raw_text",
    "email",
    "phone",
    "name",
    "address",
    "dialogue",
    "conversation",
    "query_text",
    "geo_coordinates",
    "latitude",
    "longitude",
}

REQUIRED_PROFILE_KEYS: Set[str] = {
    "user_id",
    "category_counts",
    "temporal_counts",
    "price_tier_counts",
    "last_updated",
    "total_sessions",
}


@dataclass
class AggregatedUserProfile:
    """Privacy-preserving aggregate user statistics (no raw dialogue or PII)."""

    user_id: str
    category_counts: Dict[str, int] = field(default_factory=dict)
    temporal_counts: Dict[str, int] = field(default_factory=dict)
    price_tier_counts: Dict[str, int] = field(default_factory=dict)
    last_updated: float = field(default_factory=time.time)
    total_sessions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AggregatedUserProfile:
        validate_profile_dict(data)
        return cls(
            user_id=str(data["user_id"]),
            category_counts={str(k): int(v) for k, v in dict(data.get("category_counts", {})).items()},
            temporal_counts={str(k): int(v) for k, v in dict(data.get("temporal_counts", {})).items()},
            price_tier_counts={str(k): int(v) for k, v in dict(data.get("price_tier_counts", {})).items()},
            last_updated=float(data.get("last_updated", time.time())),
            total_sessions=int(data.get("total_sessions", 0)),
        )


def empty_profile(user_id: str) -> AggregatedUserProfile:
    return AggregatedUserProfile(user_id=user_id)


def validate_profile_dict(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Profile must be a dict")
    forbidden = FORBIDDEN_PROFILE_KEYS & set(data.keys())
    if forbidden:
        raise ValueError(f"Forbidden profile fields (PII/raw): {sorted(forbidden)}")
    for key in data:
        if key.startswith("raw_") or key.endswith("_text"):
            raise ValueError(f"Forbidden profile field pattern: {key}")
    missing = REQUIRED_PROFILE_KEYS - set(data.keys())
    if missing and "user_id" not in data:
        raise ValueError(f"Missing required profile keys: {sorted(missing)}")


def profile_summary(profile: AggregatedUserProfile) -> Dict[str, Any]:
    """Transparency API: aggregated view only (no raw history)."""
    return {
        "user_id": profile.user_id,
        "category_counts": dict(profile.category_counts),
        "temporal_counts": dict(profile.temporal_counts),
        "price_tier_counts": dict(profile.price_tier_counts),
        "total_sessions": profile.total_sessions,
        "last_updated": profile.last_updated,
    }
