# Eval freeze note (v3)

**Created:** 2026-08-01  
**Package:** `artifacts/thesis_eval/v3/`  
**Baselines:** `v1/` (pre-redesign), `v2/` (S0/S1), this pack = S2 on top of v2

## Frozen parameters (delta vs v2)

| Parameter | Value |
|-----------|--------|
| Hypothesis max size | 120 (even 6×20) — unchanged from v2 |
| Belief / clarify knobs | same as v2 |
| `enable_domain_guards` | **true** |
| `enable_m1_voting` | **true** |
| `guard_confidence_cap` | 0.85 |
| `vote_recency_decay` | 0.75 |
| Lexical guards | word-boundary keywords + phrases + medical strong cues + negation + cuisine names |
| Memory seed | still mild gold-aligned (disclose; S3 pending) |

## Table B1 (v3)

| Condition | Category match | Clarify rate | Entropy drop | Belief=last M1 | Privacy |
|-----------|---------------:|-------------:|-------------:|---------------:|--------:|
| LLM-only | 0.93 | 0.11 | 0.08 | — | 1.0 |
| Intention-conditioned | **1.00** | **0.29** | 0.40 | **1.00** | 1.0 |
| Memory-augmented | **1.00** | **0.17** | 0.35 | **1.00** | 1.0 |

## Trajectory

| Pack | C1 match | C1 clarify | Notes |
|------|---------:|-----------:|-------|
| v1 | 0.50 | 0.54 | dining/travel hyps missing; sticky belief |
| v2 | 0.71 | 0.31 | S0/S1 space + belief + clarify budget |
| v3 | **1.00** | 0.29 | S2 guards + M1 voting |

Comparisons: `before_after_comparison.md` (v2→v3), `v1_to_v3_comparison.md` (full arc).

## Caveats (keep for Ch5)

1. N=14 scenarios — perfect match is **engineering success on this pack**, not a claim of solved NLU.  
2. Domain guards are **transparent lexical rules**; disclose alongside NB extractor.  
3. Memory-augmented still uses gold-aligned seed until S3.  
4. B1 remains sidecar/belief-path metrics (not full GPT reply quality).  
5. Do not overwrite v1/v2/v3; next change → `v4/`.

## Reproduce

```bash
cd Code
python scripts/run_thesis_condition_eval.py --out-dir artifacts/thesis_eval/v3
python scripts/compare_thesis_eval_versions.py \
  --before artifacts/thesis_eval/v2/thesis_condition_eval.json \
  --after artifacts/thesis_eval/v3/thesis_condition_eval.json \
  --out-dir artifacts/thesis_eval/v3
```
