# Thesis eval before/after comparison

- before: `2026-08-01T00:09:09.912397+00:00`
- after: `2026-08-01T00:24:40.691223+00:00`

## Freeze

### Before
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

## Table B1 deltas

### LLM-only

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.8571 | 0.8571 | 0.0 |
| last_m1_match_gold | None | None | None |
| belief_matches_last_m1 | None | None | None |
| clarify_rate | 0.1714 | 0.1429 | -0.0285 |
| mean_entropy_drop | 0.068827 | 0.02948 | -0.0393 |
| mean_category_entropy_drop | 0.0 | 0.0 | 0.0 |
| resolution_rate | 0.6429 | 0.5714 | -0.0715 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

### Intention-conditioned

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.8571 | 0.7143 | -0.1428 |
| last_m1_match_gold | 0.8571 | 0.8571 | 0.0 |
| belief_matches_last_m1 | 1.0 | 0.7143 | -0.2857 |
| clarify_rate | 0.2571 | 0.2857 | 0.0286 |
| mean_entropy_drop | 0.410523 | 0.297467 | -0.1131 |
| mean_category_entropy_drop | 0.410252 | 0.297218 | -0.113 |
| resolution_rate | 0.7143 | 0.7143 | 0.0 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

### Memory-augmented

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.9286 | 0.8571 | -0.0715 |
| last_m1_match_gold | 0.8571 | 0.8571 | 0.0 |
| belief_matches_last_m1 | 0.9286 | 1.0 | 0.0714 |
| clarify_rate | 0.1714 | 0.1714 | 0.0 |
| mean_entropy_drop | 0.381309 | 0.258442 | -0.1229 |
| mean_category_entropy_drop | 0.397861 | 0.274946 | -0.1229 |
| resolution_rate | 0.8571 | 0.7143 | -0.1428 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

## Nail scenarios (C1/C2)

### `dining_vague_then_specific`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "dining",
      "pred": "dining",
      "match": true,
      "last_m1": "dining",
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "dining",
      "pred": "dining",
      "match": true,
      "last_m1": "dining",
      "belief_matches_last_m1": true,
      "policy": [
        "suggest",
        "suggest",
        "suggest"
      ]
    }
  },
  "Memory-augmented": {
    "before": {
      "gold": "dining",
      "pred": "dining",
      "match": true,
      "last_m1": "dining",
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "dining",
      "pred": "dining",
      "match": true,
      "last_m1": "dining",
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

### `travel_flight_vague`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "travel",
      "pred": "travel",
      "match": true,
      "last_m1": "travel",
      "policy": [
        "clarify",
        "suggest",
        "suggest"
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
      "pred": "travel",
      "match": true,
      "last_m1": "travel",
      "policy": [
        "suggest",
        "suggest",
        "suggest"
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
      "pred": "medical",
      "match": true,
      "last_m1": "medical",
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "medical",
      "pred": "medical",
      "match": true,
      "last_m1": "medical",
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
      "gold": "medical",
      "pred": "medical",
      "match": true,
      "last_m1": "medical",
      "policy": [
        "clarify",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "medical",
      "pred": "medical",
      "match": true,
      "last_m1": "medical",
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

### `vague_wellness_weekend`

```json
{
  "Intention-conditioned": {
    "before": {
      "gold": "wellness",
      "pred": "dining",
      "match": false,
      "last_m1": "dining",
      "policy": [
        "suggest",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "wellness",
      "pred": "dining",
      "match": false,
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
      "gold": "wellness",
      "pred": "dining",
      "match": false,
      "last_m1": "dining",
      "policy": [
        "suggest",
        "suggest",
        "suggest"
      ]
    },
    "after": {
      "gold": "wellness",
      "pred": "dining",
      "match": false,
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

## Intention-conditioned flips (all scenarios)

| Scenario | Gold | Before | After | Before✓ | After✓ | After M1 | Belief=M1 |
|---|---|---|---|---|---|---|---|
| beauty_skin_uncertain | beauty | beauty | wellness | True | False | beauty | False |
| clear_fitness_single | fitness | fitness | fitness | True | True | fitness | True |
| conflict_time_fitness | fitness | fitness | wellness | True | False | fitness | False |
| dining_budget_lunch | dining | dining | dining | True | True | wellness | False |
| dining_vague_then_specific | dining | dining | dining | True | True | dining | True |
| exhausted_wellness_urgent | wellness | fitness | wellness | False | True | wellness | True |
| fitness_wellness_yoga_spa | fitness | fitness | fitness | True | True | fitness | True |
| gym_yoga_massage_conflict | fitness | fitness | fitness | True | True | fitness | True |
| medical_wellness_ambiguous | medical | medical | wellness | True | False | medical | False |
| medical_wellness_checkup | medical | medical | medical | True | True | medical | True |
| premium_spa_afternoon | wellness | wellness | wellness | True | True | wellness | True |
| travel_flight_vague | travel | travel | travel | True | True | travel | True |
| travel_hotel_combo | travel | travel | travel | True | True | travel | True |
| vague_wellness_weekend | wellness | dining | dining | False | False | dining | True |
