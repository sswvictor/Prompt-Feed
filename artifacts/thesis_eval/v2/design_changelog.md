# Design changelog — thesis_eval v1 → v2

**Date:** 2026-08-01  
**Scope:** S0 + S1 engineering fixes (not thesis-constrained).  
**Package:** `artifacts/thesis_eval/v2/` (v1 frozen, not overwritten)

## Root cause confirmed in audit

Runtime `PIPELINE_HYPOTHESIS_MAX_SIZE=360` + sequential hypothesis fill produced:

| Category | Count (v1) |
|----------|------------|
| fitness / wellness / beauty / medical | 90 each |
| dining / travel | **0** |

Missing dining/travel hypotheses forced travel/dining STVs into residual slot competition among the first four domains → fitness attractor. Freeze note claimed 540 from yaml, but `.env` overrode to 360.

## Changes

### S0 — quick knobs

| Change | Where | Before → After |
|--------|--------|----------------|
| Even category coverage | `src/m2_reasoning/hypothesis.py` | sequential fill → round-robin / per-category budgets |
| Hypothesis size | `config.yaml` + `.env` | 360/540 → **120** (20×6) |
| Slot evidence power | `m2.slot_evidence_power` | full multiply → **0.30** exponent |
| Belief on previous posterior | `update_belief` | mostly `L × static prior` → `L × decayed previous` + soft prior reinjection |
| Power decay / recency | `belief_power_decay` / `belief_recency_alpha` | n/a → **0.80** / **0.70** |
| Prior reinjection λ | `prior_reinjection` | n/a → **0.15** |
| Suggest threshold | `top1_suggest_threshold` | 0.50 → **0.40** |
| Entropy clarify gate | `high_entropy_ratio` | 0.85 → **0.99** |
| Clarify budget | `clarify_budget` | unlimited → **1**/session |

### S1 — structure

| Change | Where |
|--------|--------|
| Policy reads **category** mass / category entropy only (already primary; thresholds retuned) | `intention_context.resolve_dialogue_policy` |
| Clarify budget (default 2 / session) | `SessionState.clarify_turns_used` + policy gate |
| Knobs on session from `PipelineConfig.m2` | `SessionManager.new_session` |

### Logging / audit

- Eval freeze block now records full `m2` knobs + `hypothesis_category_counts`
- Per-turn: `belief_matches_m1`, `category_entropy`, `clarify_turns_used`
- Per-scenario: `last_m1_category`, `belief_matches_last_m1`, `category_entropy_drop`
- Comparison: `scripts/compare_thesis_eval_versions.py` → `before_after_comparison.{json,md}`

## Reproduce

```bash
cd Code
python scripts/run_thesis_condition_eval.py --out-dir artifacts/thesis_eval/v2
python scripts/compare_thesis_eval_versions.py \
  --before artifacts/thesis_eval/v1/thesis_condition_eval.json \
  --after artifacts/thesis_eval/v2/thesis_condition_eval.json \
  --out-dir artifacts/thesis_eval/v2
```

## Success criteria (engineering)

1. Belief match ≥ last-M1 match on Intention-conditioned  
2. Clarify rate ~0.15–0.35  
3. Nail scenarios improve: `dining_vague_then_specific`, `travel_flight_vague`, `medical_wellness_checkup`, `vague_wellness_weekend`  
4. Category entropy mostly decreasing across multi-turn scenarios  
