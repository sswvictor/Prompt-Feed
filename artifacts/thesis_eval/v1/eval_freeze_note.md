# Eval freeze note (v1)

**Created:** 2026-07-26  
**Package:** `artifacts/thesis_eval/v1/`

## Frozen parameters

| Parameter | Value |
|-----------|--------|
| Extractor | `pipeline` (frozen composite artefact) |
| Config | `config.yaml` via `pipeline_config_from_yaml` |
| Converge threshold | from config (`m2.converge_threshold`, currently 0.55) |
| Hypothesis max size | from config (`pipeline.hypothesis_max_size`, currently 540) |
| Scenarios | `data/annotations/eval_scenarios.json` (N=14) |
| Memory-augmented prior | mild gold-aligned `category_counts >= 3` per scenario `user_id` (+ `synthetic_users.json` if present) |
| Generative LLM | **not** used for Table B1 (sidecar + intention-block privacy audit only) |

## Thesis conditions

| Thesis name | Code id | Operationalisation in this package |
|-------------|---------|--------------------------------------|
| LLM-only | `C0_llm_only` | Fresh session each turn; final category = M1; M1-confidence policy; intention inject **off** |
| Intention-conditioned | `C1_intention_block` | Multi-turn M1+M2; category = belief top-1 category; M2 policy; inject **on**; M3 off |
| Memory-augmented | `C2_m3_prior` | As Intention-conditioned + M3 prior; inject **on** |

## How to reproduce

```bash
cd Code
python scripts/run_thesis_condition_eval.py --out-dir artifacts/thesis_eval/v1
python scripts/run_m1_eval.py --compare-hard-cases
python scripts/run_m1_public_benchmark.py
python -m pytest tests/dialogue/test_privacy_audit.py -q
```

## Caveats (for Ch5)

1. Table B1 measures **structured task / belief path**, not end-to-end LLM reply quality.  
2. LLM-only here is a **structural** baseline (no multi-turn belief, no intention injection), aligned with Method naming but not a full GPT dialogue ablation.  
3. Memory-augmented uses a **seeded** gold-aligned prior so C2 can differ from C1; disclose in Results.  
4. Do not overwrite `v1/`; bump to `v2/` if parameters change.
