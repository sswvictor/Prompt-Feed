# Eval freeze note (v6) — expanded scenarios

**Created:** 2026-08-01  
**Dataset:** `data/annotations/eval_scenarios_v2.json`  
**Model:** v4-era `artifacts/m1/model.json` (guards OFF)

## Package definition

| Item | Value |
|------|--------|
| Total scenarios | **42** (6 classes × **7**) |
| Turns | **87** |
| Train leakage of eval texts | **0** (checked) |
| Builder | `scripts/build_eval_scenarios_v2.py` |

Thesis-facing evaluation uses this file only.

## Table B1 — full N=42

| Condition | Match | Frac | Clarify | Entropy drop |
|-----------|------:|------|--------:|-------------:|
| LLM-only | 0.8095 | 34/42 | 0.1609 | 0.0335 |
| Intention-conditioned | 0.7619 | 32/42 | 0.3103 | 0.2505 |
| Memory-augmented | 0.8333 | 35/42 | 0.2299 | 0.2194 |

Step size ≈ 1/42 ≈ 0.0238 (finer than 1/14).

## How to reproduce

```bash
cd Code
python scripts/build_eval_scenarios_v2.py
python scripts/run_thesis_condition_eval.py \
  --dataset data/annotations/eval_scenarios_v2.json \
  --out-dir artifacts/thesis_eval/v6
python scripts/summarize_eval_v6.py
```

## Thesis usage

- **Canonical scenario package (thesis-facing):** **v6 / N=42** (`eval_scenarios_v2.json`) with **v4 model/weights**.  
- Do not present N=14 as a current evaluation package in the thesis.
