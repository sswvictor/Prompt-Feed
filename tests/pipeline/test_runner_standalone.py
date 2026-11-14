from __future__ import annotations

from src.m2_reasoning.belief import belief_step, uniform_prior
from src.m2_reasoning.hypothesis import generate_hypothesis_space
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner


def test_enable_m3_false_matches_manual_m2() -> None:
    text = "Book a quick workout class nearby tonight"
    runner = PipelineRunner(PipelineConfig(enable_m3=False, hypothesis_max_size=80))
    result = runner.process_text(text, turn=0)

    hs = runner.hypotheses
    prior = uniform_prior(hs)
    manual_post, manual_top = belief_step(result.stv, hs, prior, turn=0)

    for h in hs:
        assert abs(result.posterior[h.hypothesis_id] - manual_post[h.hypothesis_id]) < 1e-8
    assert set(result.top_k_ids) == set(manual_top)
