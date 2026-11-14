# Design changelog — thesis_eval v2 → v4 (evidence quality)

**Date:** 2026-08-01  
**Goal:** Improve M1 evidence quality **without** hard lexical label overrides, and **without regressing v2**.

## What stayed (v2 substrate)

- Even hypothesis coverage (120 = 6×20)
- Belief update on previous posterior + slot down-weight + clarify budget
- **`enable_domain_guards: false`** (hard keyword category rewrite OFF)

## What changed (evidence quality)

1. **Confusion-pair supplement** (`m1_confusion_supplement.json`, ~200 rows)  
   - Families: medical↔wellness/fitness, dining colloquial/cuisine, travel, fitness≠spa  
   - Template paraphrases — **not** copies of `eval_scenarios`  
2. **Train merge** → `m1_train_composite_eq.json`  
   - Explicitly **strips any eval-scenario utterance** found in the base composite (found 2 pre-existing leaks; removed)  
3. **Retrain** NB → `artifacts/m1/model.json` (backup: `model_pre_eq_backup.json`)  
4. Multi-turn STV voting left on; conflict-confidence dampen left **off** (ablation: no C1 gain on N=14)

## Ablation (guards always OFF)

| Setting | C1 match |
|---------|---------:|
| v2 model era (pack v2) | 0.7143 |
| Retrain + vote on/off | **0.8571** (vote tied) |
| Hard guards (v3, not used here) | 1.00 (rejected as main path) |

## Outcome vs v2

| Metric (Intention-conditioned) | v2 | v4 |
|--------------------------------|---:|---:|
| category_task_match | 0.71 | **0.86** |
| clarify_rate | 0.31 | **0.26** |
| enable_domain_guards | n/a | **false** |

Remaining misses (honest): `vague_wellness_weekend`, `exhausted_wellness_urgent` — still hard for bag-of-words without leaking eval text into train.

## Reproduce

```bash
cd Code
python scripts/augment_confusion_pairs.py --merge-composite --extra-per-class 50 --seed 44
python scripts/run_m1_train.py --train_file data/annotations/m1_train_composite_eq.json --out_file artifacts/m1/model_stub_eq.json
python scripts/run_thesis_condition_eval.py --out-dir artifacts/thesis_eval/v4
python scripts/compare_thesis_eval_versions.py \
  --before artifacts/thesis_eval/v2/thesis_condition_eval.json \
  --after artifacts/thesis_eval/v4/thesis_condition_eval.json \
  --out-dir artifacts/thesis_eval/v4
```
