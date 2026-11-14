# Eval freeze note (v2)

**Created:** 2026-08-01  
**Package:** `artifacts/thesis_eval/v2/`  
**Baseline:** `artifacts/thesis_eval/v1/` (unchanged)

## Frozen parameters

| Parameter | Value |
|-----------|--------|
| Extractor | `pipeline` (frozen composite artefact) |
| Config | `config.yaml` via `pipeline_config_from_yaml` |
| Converge threshold | `0.55` |
| Hypothesis max size | **120** (even 20×6 categories; `.env` aligned) |
| Slot evidence power | `0.30` |
| Belief power decay / recency α | `0.80` / `0.70` |
| Prior reinjection λ | `0.15` |
| Top-1 suggest threshold | `0.40` |
| High entropy ratio | `0.99` |
| Clarify budget | `1` / session |
| Scenarios | `data/annotations/eval_scenarios.json` (N=14) |
| Memory-augmented prior | mild gold-aligned `category_counts >= 3` (disclose) |
| Generative LLM | **not** used for Table B1 |

## Table B1 (v2)

| Condition | Category match | Clarify rate | Entropy drop | Belief=last M1 | Privacy |
|-----------|---------------:|-------------:|-------------:|---------------:|--------:|
| LLM-only | 0.64 | 0.14 | 0.10 | — | 1.0 |
| Intention-conditioned | **0.71** | **0.31** | 0.21 | **0.86** | 1.0 |
| Memory-augmented | **0.86** | **0.20** | 0.21 | 0.71 | 1.0 |

## vs v1 (Intention-conditioned)

| Metric | v1 | v2 | Δ |
|--------|---:|---:|--:|
| category_task_match | 0.50 | 0.71 | +0.21 |
| clarify_rate | 0.54 | 0.31 | −0.23 |
| mean_entropy_drop | 0.08 | 0.21 | +0.13 |

Full audit: `before_after_comparison.md`, design notes: `design_changelog.md`.

## How to reproduce

```bash
cd Code
python scripts/run_thesis_condition_eval.py --out-dir artifacts/thesis_eval/v2
python scripts/compare_thesis_eval_versions.py
```

## Remaining gaps (S2+)

- `medical_wellness_checkup`: M1 still → fitness; needs domain guards / retrain  
- Memory seed still gold-aligned (disclose until S3)  
- Do not overwrite `v1/` or `v2/`; bump to `v3/` for next parameter change  
