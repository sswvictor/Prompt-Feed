# Thesis eval before/after comparison

- before: `2026-08-01T00:09:09.912397+00:00`
- after: `2026-08-01T00:36:42.578509+00:00`

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
| category_task_match | 0.8571 | 0.8095 | -0.0476 |
| last_m1_match_gold | None | None | None |
| belief_matches_last_m1 | None | None | None |
| clarify_rate | 0.1714 | 0.1609 | -0.0105 |
| mean_entropy_drop | 0.068827 | 0.033537 | -0.0353 |
| mean_category_entropy_drop | 0.0 | 0.0 | 0.0 |
| resolution_rate | 0.6429 | 0.619 | -0.0239 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

### Intention-conditioned

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.8571 | 0.7619 | -0.0952 |
| last_m1_match_gold | 0.8571 | 0.8095 | -0.0476 |
| belief_matches_last_m1 | 1.0 | 0.9286 | -0.0714 |
| clarify_rate | 0.2571 | 0.3103 | 0.0532 |
| mean_entropy_drop | 0.410523 | 0.250512 | -0.16 |
| mean_category_entropy_drop | 0.410252 | 0.249618 | -0.1606 |
| resolution_rate | 0.7143 | 0.6667 | -0.0476 |
| privacy_audit_pass_rate | 1.0 | 1.0 | 0.0 |

### Memory-augmented

| Metric | Before | After | Δ |
|---|---:|---:|---:|
| category_task_match | 0.9286 | 0.8333 | -0.0953 |
| last_m1_match_gold | 0.8571 | 0.8095 | -0.0476 |
| belief_matches_last_m1 | 0.9286 | 0.9048 | -0.0238 |
| clarify_rate | 0.1714 | 0.2299 | 0.0585 |
| mean_entropy_drop | 0.381309 | 0.21938 | -0.1619 |
| mean_category_entropy_drop | 0.397861 | 0.231555 | -0.1663 |
| resolution_rate | 0.8571 | 0.7381 | -0.119 |
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
        "clarify",
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
        "suggest",
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
| beauty_brows_lash | beauty | None | beauty | None | True | beauty | True |
| beauty_facial_clear | beauty | None | beauty | None | True | beauty | True |
| beauty_makeup_event | beauty | None | beauty | None | True | beauty | True |
| beauty_nail_weekend | beauty | None | medical | None | False | medical | True |
| beauty_salon_hair | beauty | None | beauty | None | True | beauty | True |
| beauty_skin_uncertain | beauty | beauty | beauty | True | True | beauty | True |
| beauty_vs_medical_skin | beauty | None | medical | None | False | medical | True |
| clear_fitness_single | fitness | fitness | fitness | True | True | fitness | True |
| conflict_time_fitness | fitness | fitness | fitness | True | True | fitness | True |
| dining_brunch_weekend | dining | None | wellness | None | False | fitness | False |
| dining_budget_lunch | dining | dining | dining | True | True | dining | True |
| dining_dinner_date | dining | None | dining | None | True | dining | True |
| dining_hungry_takeout | dining | None | dining | None | True | dining | True |
| dining_ramen_night | dining | None | dining | None | True | dining | True |
| dining_vague_then_specific | dining | dining | dining | True | True | dining | True |
| dining_vs_wellness_food | dining | None | wellness | None | False | dining | False |
| exhausted_wellness_urgent | wellness | fitness | fitness | False | False | fitness | True |
| fitness_hiit_evening | fitness | None | fitness | None | True | fitness | True |
| fitness_pt_session | fitness | None | fitness | None | True | fitness | True |
| fitness_strength_morning | fitness | None | fitness | None | True | fitness | True |
| fitness_wellness_yoga_spa | fitness | fitness | fitness | True | True | fitness | True |
| gym_yoga_massage_conflict | fitness | fitness | fitness | True | True | fitness | True |
| medical_gp_followup | medical | None | medical | None | True | medical | True |
| medical_lab_bloodwork | medical | None | medical | None | True | medical | True |
| medical_specialist_referral | medical | None | wellness | None | False | wellness | True |
| medical_urgent_care | medical | None | medical | None | True | medical | True |
| medical_vaccination | medical | None | medical | None | True | medical | True |
| medical_wellness_ambiguous | medical | medical | medical | True | True | medical | True |
| medical_wellness_checkup | medical | medical | medical | True | True | medical | True |
| premium_spa_afternoon | wellness | wellness | wellness | True | True | wellness | True |
| travel_airport_transfer | travel | None | travel | None | True | travel | True |
| travel_flight_vague | travel | travel | travel | True | True | travel | True |
| travel_hotel_combo | travel | travel | travel | True | True | travel | True |
| travel_hotel_only | travel | None | travel | None | True | travel | True |
| travel_vacation_vague | travel | None | travel | None | True | travel | True |
| travel_vs_dining_trip | travel | None | medical | None | False | medical | True |
| travel_weekend_getaway | travel | None | fitness | None | False | travel | False |
| vague_wellness_weekend | wellness | dining | dining | False | False | dining | True |
| wellness_hot_stone | wellness | None | wellness | None | True | wellness | True |
| wellness_massage_clear | wellness | None | wellness | None | True | wellness | True |
| wellness_sauna_relax | wellness | None | fitness | None | False | fitness | True |
| wellness_vs_fitness_recover | wellness | None | wellness | None | True | wellness | True |
