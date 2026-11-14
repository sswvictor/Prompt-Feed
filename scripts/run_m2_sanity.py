from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.types import StructuredTaskVariable
from src.m2_reasoning.belief import entropy, initialize_belief, uniform_prior, update_belief
from src.m2_reasoning.convergence import incorporate_feedback, is_converged, top1_probability, top_k_hypothesis_ids
from src.m2_reasoning.hypothesis import generate_hypothesis_space


def make_stv(
    service_category: str,
    temporal_preference: str | None,
    urgency_level: str,
    location_context: str | None,
    tags: list[str],
) -> StructuredTaskVariable:
    return StructuredTaskVariable(
        service_category=service_category,
        category_confidence=0.8,
        temporal_preference=temporal_preference,
        urgency_level=urgency_level,
        time_window_days=1 if urgency_level == "high" else 7,
        location_context=location_context,
        sentiment_tags=tags,
        extraction_confidence=0.8,
        raw_input="synthetic",
        clarification_needed=False,
    )


def run_case(stv: StructuredTaskVariable, turn: int = 0) -> dict:
    H = generate_hypothesis_space(max_size=80)
    belief = initialize_belief(H)
    prior = uniform_prior(H)
    updated = update_belief(belief, stv, prior, H)
    top_ids = top_k_hypothesis_ids(updated, turn)
    return {
        "top1_prob": round(top1_probability(updated), 6),
        "entropy": round(entropy(updated), 6),
        "top_k_ids": top_ids,
        "converged": is_converged(updated, threshold=0.85),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--with-m3", action="store_true", help="Run optional M3 pipeline branch")
    parser.add_argument("--user-id", default="persona_fitness_heavy")
    args = parser.parse_args()

    if args.with_m3:
        from src.m3_memory.profile import AggregatedUserProfile
        from src.m3_memory.store import InMemoryStore
        from src.pipeline.config import PipelineConfig
        from src.pipeline.runner import PipelineRunner
        import time

        store = InMemoryStore()
        store.save_profile(
            AggregatedUserProfile(
                user_id=args.user_id,
                category_counts={"fitness": 20, "wellness": 4},
                temporal_counts={"asap": 10},
                price_tier_counts={"budget": 8, "mid": 10},
                last_updated=time.time(),
                total_sessions=24,
            )
        )
        runner = PipelineRunner(
            PipelineConfig(enable_m3=True, user_id=args.user_id, hypothesis_max_size=80),
            store=store,
        )
        r = runner.process_text("Book a quick workout class nearby tonight", turn=0)
        print(
            json.dumps(
                {
                    "m3_branch": True,
                    "user_id": args.user_id,
                    "top1_prob": r.top1_prob,
                    "prior_entropy": r.prior_entropy,
                    "posterior_entropy": r.posterior_entropy,
                    "converged": r.converged,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    case_clear = make_stv("fitness", "asap", "high", "nearby", ["quick"])
    case_ambiguous = make_stv("ambiguous", None, "medium", None, [])
    case_conflict = make_stv("medical", "weekend", "high", "city_center", ["affordable"])

    clear_result = run_case(case_clear, turn=0)
    ambiguous_result = run_case(case_ambiguous, turn=0)
    conflict_result = run_case(case_conflict, turn=1)

    # feedback test on clear case
    H = generate_hypothesis_space(max_size=80)
    belief = initialize_belief(H)
    prior = uniform_prior(H)
    updated = update_belief(belief, case_clear, prior, H)
    chosen = top_k_hypothesis_ids(updated, turn=0)[0]
    after_feedback = incorporate_feedback(updated, chosen, boost=1.5, decay=0.8)

    output = {
        "clear_case": clear_result,
        "ambiguous_case": ambiguous_result,
        "conflict_case": conflict_result,
        "feedback_case": {
            "selected_id": chosen,
            "top1_before": round(top1_probability(updated), 6),
            "top1_after": round(top1_probability(after_feedback), 6),
            "converged_after": is_converged(after_feedback, threshold=0.85),
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
