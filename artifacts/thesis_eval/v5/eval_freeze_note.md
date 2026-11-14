# Eval freeze note (v5) — balanced aug attempt

**Created:** 2026-08-01  
**Verdict:** Data balancing **succeeded in counts**; Intention-conditioned match **did not improve** (0.86→0.71 vs v4).

## Runtime recommendation

Keep **`artifacts/m1/model.json` = v4-era weights** (`model_pre_v5_backup.json`).  
v5 weights archived at `artifacts/m1/model_v5_balanced_attempt.json` for audit only.

## Table B1 (v5 model, guards false)

| Condition | Match | Clarify |
|-----------|------:|--------:|
| LLM-only | 0.86 | 0.14 |
| Intention-conditioned | **0.71** | 0.29 |
| Memory-augmented | 0.86 | 0.17 |

## Logging

- `design_changelog.md` — what was tried / why it failed  
- `data_card.json` — class/source counts  
- `before_after_comparison.md` — v4→v5  
- `v2_to_v5_comparison.md` — architecture baseline vs this attempt  
