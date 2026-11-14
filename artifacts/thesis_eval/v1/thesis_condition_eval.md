# Thesis condition evaluation (Table B1)

- created_at: `2026-07-26T01:49:34.610018+00:00`
- n_scenarios: 14
- extractor_mode: `pipeline`

## Table B1 — primary scenario metrics

| Condition | Category match | Resolution rate | Mean turns to res. | Mean entropy drop | Clarify rate | Suggest rate | Privacy pass |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLM-only | 0.6429 | 0.7143 | 1.5 | 0.080178 | 0.1429 | 0.8571 | 1.0 |
| Intention-conditioned | 0.5 | 0.7143 | 1.5 | 0.080178 | 0.5429 | 0.4571 | 1.0 |
| Memory-augmented | 0.6429 | 0.7857 | 1.2727 | 0.139449 | 0.3429 | 0.6571 | 1.0 |

## Freeze

```json
{
  "converge_threshold": 0.55,
  "hypothesis_max_size": 360,
  "extractor_mode": "pipeline",
  "config_path": "config.yaml",
  "note": "Sidecar metrics only; generative LLM not required for Table B1.",
  "memory_augmented_seed": "mild gold-aligned category_counts (>=3) per scenario user_id"
}
```
