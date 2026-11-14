# Thesis condition evaluation (Table B1)

- created_at: `2026-07-31T23:34:56.264040+00:00`
- n_scenarios: 14
- extractor_mode: `pipeline`

## Table B1 — primary scenario metrics

| Condition | Category match | Resolution rate | Mean turns to res. | Mean entropy drop | Clarify rate | Suggest rate | Privacy pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM-only | 0.6429 | 0.5714 | 2 | 0.096365 | 0.1429 | 0.8571 | 1.0 |
| Intention-conditioned | 0.7143 | 0.6429 | 1.8889 | 0.212138 | 0.3143 | 0.6857 | 1.0 |
| Memory-augmented | 0.8571 | 0.7143 | 1.6 | 0.208104 | 0.2 | 0.8 | 1.0 |

## Freeze

```json
{
  "converge_threshold": 0.55,
  "hypothesis_max_size": 120,
  "extractor_mode": "pipeline",
  "config_path": "config.yaml",
  "m2": {
    "slot_evidence_power": 0.3,
    "belief_power_decay": 0.8,
    "belief_recency_alpha": 0.7,
    "prior_reinjection": 0.15,
    "top1_suggest_threshold": 0.4,
    "high_entropy_ratio": 0.99,
    "clarify_budget": 1
  },
  "hypothesis_category_counts": {
    "fitness": 20,
    "wellness": 20,
    "beauty": 20,
    "medical": 20,
    "dining": 20,
    "travel": 20
  },
  "note": "Sidecar metrics only; generative LLM not required for Table B1.",
  "memory_augmented_seed": "mild gold-aligned category_counts (>=3) per scenario user_id"
}
```
