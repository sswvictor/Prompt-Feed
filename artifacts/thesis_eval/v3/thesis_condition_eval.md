# Thesis condition evaluation (Table B1)

- created_at: `2026-07-31T23:39:53.342771+00:00`
- n_scenarios: 14
- extractor_mode: `pipeline`

## Table B1 — primary scenario metrics

| Condition | Category match | Resolution rate | Mean turns to res. | Mean entropy drop | Clarify rate | Suggest rate | Privacy pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM-only | 0.9286 | 0.5 | 2 | 0.081015 | 0.1143 | 0.8857 | 1.0 |
| Intention-conditioned | 1.0 | 0.7143 | 2.1 | 0.400534 | 0.2857 | 0.7143 | 1.0 |
| Memory-augmented | 1.0 | 0.7857 | 1.4545 | 0.351648 | 0.1714 | 0.8286 | 1.0 |

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
    "enable_domain_guards": true,
    "enable_m1_voting": true,
    "guard_confidence_cap": 0.85,
    "vote_recency_decay": 0.75
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
