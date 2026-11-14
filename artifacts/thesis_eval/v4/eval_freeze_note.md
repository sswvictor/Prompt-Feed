# Eval freeze note (v4)

**Created:** 2026-08-01  
**Package:** `artifacts/thesis_eval/v4/`  
**Compared to:** `v2/` (architecture fix; no hard guards)

## Claim (keep honest)

v4 improves **extractor evidence** via confusion-pair retrain on top of the v2 belief/hypothesis design.  
Hard domain guards remain **disabled**. Perfect scores from v3 are **not** the main result.

## Frozen settings

| Item | Value |
|------|--------|
| Hypothesis max size | 120 (even coverage) |
| Domain guards | **false** |
| M1 voting | true (ablation: no C1 delta on N=14) |
| Conflict dampen | false |
| Train data | `m1_train_composite_eq.json` (composite + confusion; eval texts blocked) |
| Model | `artifacts/m1/model.json` (backup `model_pre_eq_backup.json`) |

## Table B1

| Condition | Match | Clarify | Belief=M1 |
|-----------|------:|--------:|----------:|
| LLM-only | 0.86 | 0.17 | — |
| Intention-conditioned | **0.86** | **0.26** | 1.00 |
| Memory-augmented | 0.93 | 0.17 | 0.93 |

## vs v2 (C1)

0.71 → **0.86** match; 0.31 → **0.26** clarify. No v2 substrate rollback.

## Caveats for Ch5

1. N=14; gains are engineering, not population inference.  
2. Supplement targets known confusion *families*; eval utterances excluded (0 exact leaks after filter).  
3. Two wellness-vague scenarios still fail — limitation of lexical NB + sparse cues.  
4. Prefer v2 for “architecture” story; v4 for “better evidence / retrain” story; v3 only as rejected heuristic upper bound.
