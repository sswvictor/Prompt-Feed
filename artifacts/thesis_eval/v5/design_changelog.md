# Design changelog — balanced template aug + retrain (v5)

**Date:** 2026-08-01  
**Status:** **Negative / mixed result** — class counts balanced, scenario C1 **regressed** vs v4.

## What we did

1. Expanded `scripts/augment_fitness_wellness.py`
   - Larger activity/opener/location lists
   - Contrastive + vague tiers (no eval utterance copies)
   - Eval-scenario blocklist
   - Bucket cap to limit near-duplicate slot fills
2. Extended LUCID intent map (`find_massages`, `find_workouts`, …)
3. Rebuilt `m1_train_composite.json` with `per_class_cap=500`
4. Merged confusion supplement → `m1_train_composite_eq.json` (0 eval leaks)
5. Retrained NB → archived as `artifacts/m1/model_v5_balanced_attempt.json`

## Data card (final v5 train file)

See `data_card.json`. Approx after moderate aug (200/200) + confusion:

| Category | N |
|----------|--:|
| beauty | 500 |
| dining | ~548 |
| fitness | ~390 |
| medical | ~560 |
| travel | ~526 |
| wellness | ~426 |
| **total** | **~2950** |

Heavy-aug attempt (fitness≈400 templates) first hit ~500/class but **hurt** dining/medical discrimination.

## Metrics (guards OFF)

| Pack | C1 match | Notes |
|------|---------:|-------|
| v2 | 0.71 | architecture only |
| v4 | **0.86** | confusion retrain (best so far) |
| v5 | **0.71** | balanced template aug retrain |

v5 does **not** beat v4. Active runtime model restored to **v4-era** (`model_pre_v5_backup.json` → `model.json`).

## Lesson

Equalizing N via slot templates is not free: NB overfits template collocations (`relaxing/nearby/quick`) and pulls dining/medical toward wellness. Prefer **contrastive + confusion** over mass template fill for this extractor.

## Artefacts

- `thesis_condition_eval.{json,md}` — v5 model scores  
- `before_after_comparison.*` — v4→v5  
- `v2_to_v5_comparison.*`  
- `data_card.json`  
- `model_v5_balanced_attempt.json` — keep for audit; do not use as default  
