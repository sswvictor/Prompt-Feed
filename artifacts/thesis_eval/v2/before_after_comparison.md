# Thesis eval before/after comparison

- before: `2026-07-26T01:49:34.610018+00:00`
- after: `2026-07-31T23:34:56.264040+00:00`

## Freeze

### Before
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

### After
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

## Table B1 deltas

### LLM-only

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.6429 | 0.6429 | 0.0 |
| last_m1_match_gold | None | None | None |
| belief_matches_last_m1 | None | None | None |
| clarify_rate | 0.1429 | 0.1429 | 0.0 |
| mean_entropy_drop | 0.080178 | 0.096365 | 0.0162 |
| mean_category_entropy_drop | None | 0.0 | None |
| resolution_rate | 0.7143 | 0.5714 | -0.1429 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

### Intention-conditioned

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.5 | 0.7143 | 0.2143 |
| last_m1_match_gold | None | 0.6429 | None |
| belief_matches_last_m1 | None | 0.8571 | None |
| clarify_rate | 0.5429 | 0.3143 | -0.2286 |
| mean_entropy_drop | 0.080178 | 0.212138 | 0.132 |
| mean_category_entropy_drop | None | 0.211811 | None |
| resolution_rate | 0.7143 | 0.6429 | -0.0714 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

### Memory-augmented

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.6429 | 0.8571 | 0.2142 |
| last_m1_match_gold | None | 0.6429 | None |
| belief_matches_last_m1 | None | 0.7143 | None |
| clarify_rate | 0.3429 | 0.2 | -0.1429 |
| mean_entropy_drop | 0.139449 | 0.208104 | 0.0687 |
| mean_category_entropy_drop | None | 0.2241 | None |
| resolution_rate | 0.7857 | 0.7143 | -0.0714 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

## Nail scenarios (C1/C2)

### `dining_vague_then_specific`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "dining",
      "pred": "fitness",
      "match": false,
      "last_m1": null,
      "policy": [
        "clarify",
        "clarify",
        "clarify"
      ]
    },
    "after": {
      "gold": "dining",
      "pred": "dining",
      "match": true,
      "last_m1": "dining",
      "belief_matches_last_m1": true,
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    }
  },
  "Memory-augmented": {
    "before": {
      "gold": "dining",
      "pred": "fitness",
      "match": false,
      "last_m1": null,
      "policy": [
        "clarify",
        "clarify",
        "clarify"
      ]
    },
    "after": {
      "gold": "dining",
      "pred": "dining",
      "match": true,
      "last_m1": "dining",
      "belief_matches_last_m1": true,
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    }
  }
}
```

### `travel_flight_vague`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "travel",
      "pred": "fitness",
      "match": false,
      "last_m1": null,
      "policy": [
        "clarify",
        "clarify",
        "clarify"
      ]
    },
    "after": {
      "gold": "travel",
      "pred": "travel",
      "match": true,
      "last_m1": "travel",
      "belief_matches_last_m1": true,
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    }
  },
  "Memory-augmented": {
    "before": {
      "gold": "travel",
      "pred": "fitness",
      "match": false,
      "last_m1": null,
      "policy": [
        "clarify",
        "clarify",
        "clarify"
      ]
    },
    "after": {
      "gold": "travel",
      "pred": "travel",
      "match": true,
      "last_m1": "travel",
      "belief_matches_last_m1": true,
      "policy": [
        "suggest",
        "suggest",
        "suggest"
      ]
    }
  }
}
```

### `medical_wellness_checkup`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "medical",
      "pred": "fitness",
      "match": false,
      "last_m1": null,
      "policy": [
        "clarify",
        "suggest",
        "clarify"
      ]
    },
    "after": {
      "gold": "medical",
      "pred": "wellness",
      "match": false,
      "last_m1": "fitness",
      "belief_matches_last_m1": false,
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    }
  },
  "Memory-augmented": {
    "before": {
      "gold": "medical",
      "pred": "medical",
      "match": true,
      "last_m1": null,
      "policy": [
        "clarify",
        "suggest",
        "clarify"
      ]
    },
    "after": {
      "gold": "medical",
      "pred": "wellness",
      "match": false,
      "last_m1": "fitness",
      "belief_matches_last_m1": false,
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    }
  }
}
```

### `vague_wellness_weekend`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "wellness",
      "pred": "fitness",
      "match": false,
      "last_m1": null,
      "policy": [
        "suggest",
        "clarify",
        "clarify"
      ]
    },
    "after": {
      "gold": "wellness",
      "pred": "wellness",
      "match": true,
      "last_m1": "beauty",
      "belief_matches_last_m1": false,
      "policy": [
        "suggest",
        "suggest",
        "clarify"
      ]
    }
  },
  "Memory-augmented": {
    "before": {
      "gold": "wellness",
      "pred": "wellness",
      "match": true,
      "last_m1": null,
      "policy": [
        "suggest",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "wellness",
      "pred": "wellness",
      "match": true,
      "last_m1": "beauty",
      "belief_matches_last_m1": false,
      "policy": [
        "suggest",
        "suggest",
        "suggest"
      ]
    }
  }
}
```

## Intention-conditioned flips (all scenarios)

| Scenario | Gold | Before | After | Before✓ | After✓ | After M1 | Belief=M1 |
|---|---|---|---|---|---|---|---|
| beauty_skin_uncertain | beauty | beauty | beauty | True | True | beauty | True |
| clear_fitness_single | fitness | fitness | fitness | True | True | fitness | True |
| conflict_time_fitness | fitness | fitness | fitness | True | True | fitness | True |
| dining_budget_lunch | dining | wellness | wellness | False | False | wellness | True |
| dining_vague_then_specific | dining | fitness | dining | False | True | dining | True |
| exhausted_wellness_urgent | wellness | wellness | wellness | True | True | wellness | True |
| fitness_wellness_yoga_spa | fitness | fitness | fitness | True | True | fitness | True |
| gym_yoga_massage_conflict | fitness | fitness | fitness | True | True | fitness | True |
| medical_wellness_ambiguous | medical | fitness | fitness | False | False | fitness | True |
| medical_wellness_checkup | medical | fitness | wellness | False | False | fitness | False |
| premium_spa_afternoon | wellness | wellness | wellness | True | True | wellness | True |
| travel_flight_vague | travel | fitness | travel | False | True | travel | True |
| travel_hotel_combo | travel | wellness | wellness | False | False | wellness | True |
| vague_wellness_weekend | wellness | fitness | wellness | False | True | beauty | False |
