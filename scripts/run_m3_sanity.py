from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m3_memory.prior import compute_prior
from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore, SqliteStore
from src.m2_reasoning.belief import entropy, uniform_prior
from src.m2_reasoning.hypothesis import generate_hypothesis_space
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner


def main() -> None:
    hs = generate_hypothesis_space(max_size=80)
    store = InMemoryStore()
    profile = AggregatedUserProfile(
        user_id="sanity_user",
        category_counts={"wellness": 15, "fitness": 3},
        temporal_counts={"weekend_anytime": 10, "evening": 5},
        price_tier_counts={"mid": 12, "premium": 6},
        last_updated=time.time(),
        total_sessions=18,
    )
    store.save_profile(profile)

    uniform = uniform_prior(hs)
    history = compute_prior(profile, hs)
    deleted = False
    store.delete_profile("sanity_user")
    deleted = store.get_profile("sanity_user").total_sessions == 0
    store.save_profile(profile)

    runner = PipelineRunner(
        PipelineConfig(enable_m3=True, user_id="sanity_user", hypothesis_max_size=80),
        store=store,
    )
    result = runner.process_text("I need a relaxing spa nearby this weekend", turn=0)

    sqlite_path = ROOT / "data" / "memory" / "profiles_sanity.db"
    sqlite_store = SqliteStore(sqlite_path)
    sqlite_store.save_profile(profile)
    sqlite_loaded = sqlite_store.get_profile("sanity_user").total_sessions == 18

    output = {
        "uniform_entropy": round(entropy(uniform), 6),
        "history_prior_entropy": round(entropy(history), 6),
        "delete_reset_ok": deleted,
        "sqlite_roundtrip_ok": sqlite_loaded,
        "pipeline_top1_prob": round(result.top1_prob, 6),
        "pipeline_converged": result.converged,
        "prior_comparison": runner.prior_comparison("sanity_user"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
