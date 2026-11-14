# Thesis condition evaluation (Table B1)

- created_at: `2026-08-01T05:01:50.239923+00:00`
- n_scenarios: 14
- extractor_mode: `pipeline`

## Table B1 — primary scenario metrics

| Condition | Category match | Resolution rate | Mean turns to res. | Mean entropy drop | Clarify rate | Suggest rate | Privacy pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM-only | 0.9286 | 0.7143 | 1.6 | 0.082412 | 0.1429 | 0.8571 | 1.0 |
| Intention-conditioned | 0.9286 | 0.7857 | 1.6364 | 0.407682 | 0.2571 | 0.7429 | 1.0 |
| Memory-augmented | 0.9286 | 0.7143 | 1.4 | 0.347219 | 0.1143 | 0.8857 | 1.0 |

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
    "clarify_budget": 1,
    "enable_domain_guards": false,
    "enable_m1_voting": true,
    "guard_confidence_cap": 0.85,
    "vote_recency_decay": 0.75,
    "enable_conflict_dampen": false,
    "conflict_dampen_factor": 0.65
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
