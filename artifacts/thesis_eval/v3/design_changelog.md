# Design changelog — thesis_eval v2 → v3

**Date:** 2026-08-01  
**Scope:** S2 domain keyword guards + multi-turn M1 voting  
**Package:** `artifacts/thesis_eval/v3/` (v1/v2 frozen)

## Motivation

v2 fixed missing dining/travel hypotheses and sticky belief. Remaining nail failure:

- `medical_wellness_checkup`: M1 still labels clinic/checkup turns as fitness/wellness

Root causes: NB mislabels + keyword substring bugs (`eat`⊂`retreat`) + belief following last M1.

## Changes

| Change | Where |
|--------|--------|
| Word-boundary-only keyword hits; phrase cues; medical strong weights | `src/m1_nlu/category_keywords.py` |
| Negation (`not spa`) down-weights wellness | same |
| `apply_domain_guards(stv, text)` override when lexical evidence disagrees | same; wired in `PipelineRunner.process_text` |
| `vote_category_from_history` — history consensus overrides weak bad last turn | same; `SessionManager` passes `stv_history` |
| Config knobs | `m2.enable_domain_guards`, `enable_m1_voting`, `guard_confidence_cap`, `vote_recency_decay` |

## Not changed

- Frozen NB artefact (S4 retrain still later)
- Memory gold seed (S3)
- v1 / v2 artefact packages

## Reproduce

```bash
cd Code
python scripts/run_thesis_condition_eval.py --out-dir artifacts/thesis_eval/v3
python scripts/compare_thesis_eval_versions.py \
  --before artifacts/thesis_eval/v2/thesis_condition_eval.json \
  --after artifacts/thesis_eval/v3/thesis_condition_eval.json \
  --out-dir artifacts/thesis_eval/v3
```

## Iteration notes (same day)

1. First S2 pass fixed medical/travel but regressed `conflict_time_fitness` / `exhausted_wellness_urgent` via history vote overriding lexically supported current turns.  
2. Fix: skip history override when current `raw_input` has lexical support for its label; soften `X or Y` handling.  
3. Last miss `dining_budget_lunch`: add cuisine keywords (`sushi`, `thai`, …).

## Success criteria (S2) — outcome

| Criterion | Result |
|-----------|--------|
| `medical_wellness_checkup` C1 match | **pass** |
| C1 match ≥ v2 (0.71) | **1.00** |
| Clarify ~0.15–0.35 | **0.29** |
| belief_matches_last_m1 | **1.00** |
| No dining/travel regression | **pass** (travel_hotel_combo also fixed) |
