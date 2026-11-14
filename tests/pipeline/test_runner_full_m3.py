from __future__ import annotations

import time

from src.m2_reasoning.belief import entropy, uniform_prior
from src.m2_reasoning.hypothesis import generate_hypothesis_space
from src.m3_memory.profile import AggregatedUserProfile
from src.m3_memory.store import InMemoryStore
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner


def test_m3_prior_differs_from_uniform() -> None:
    store = InMemoryStore()
    store.save_profile(
        AggregatedUserProfile(
            user_id="persona_fitness_heavy",
            category_counts={"fitness": 30, "wellness": 2},
            temporal_counts={"asap": 15, "morning": 10},
            price_tier_counts={"budget": 12, "mid": 15},
            last_updated=time.time(),
            total_sessions=32,
        )
    )
    cfg = PipelineConfig(
        enable_m3=True,
        user_id="persona_fitness_heavy",
        hypothesis_max_size=80,
        memory_backend="inmemory",
    )
    runner = PipelineRunner(cfg, store=store)
    cmp = runner.prior_comparison("persona_fitness_heavy")
    assert cmp["history_entropy"] < cmp["uniform_entropy"]


def test_full_m3_pipeline_deterministic() -> None:
    store = InMemoryStore()
    store.save_profile(
        AggregatedUserProfile(
            user_id="u1",
            category_counts={"wellness": 20},
            temporal_counts={"weekend_anytime": 12},
            price_tier_counts={"mid": 10},
            last_updated=time.time(),
            total_sessions=20,
        )
    )
    cfg = PipelineConfig(enable_m3=True, user_id="u1", hypothesis_max_size=80)
    runner = PipelineRunner(cfg, store=store)
    r1 = runner.process_text("I need a relaxing spa nearby this weekend", turn=0)
    r2 = runner.process_text("I need a relaxing spa nearby this weekend", turn=0)
    assert r1.top_k_ids == r2.top_k_ids
    assert abs(r1.top1_prob - r2.top1_prob) < 1e-12
