# Expanded eval summary (v6, N=42)

## Full pack (6×7)

| Condition | Match | Frac | Clarify | Entropy drop |
|---|---:|---:|---:|---:|
| LLM-only | 0.8095 | 34/42 | 0.1609 | 0.033537 |
| Intention-conditioned | 0.7619 | 32/42 | 0.3103 | 0.250512 |
| Memory-augmented | 0.8333 | 35/42 | 0.2299 | 0.21938 |

## Legacy-14 slice (same IDs as v4) vs v4 freeze

| Condition | v6 legacy match | v4 freeze match | v6 clarify | v4 clarify |
|---|---:|---:|---:|---:|
| LLM-only | 0.8571 (12/14) | 0.8571 | 0.1714 | 0.1714 |
| Intention-conditioned | 0.8571 (12/14) | 0.8571 | 0.2571 | 0.2571 |
| Memory-augmented | 0.9286 (13/14) | 0.9286 | 0.1714 | 0.1714 |

## New-28 only

| Condition | Match | Frac | Clarify |
|---|---:|---:|---:|
| LLM-only | 0.7857 | 22/28 | 0.1538 |
| Intention-conditioned | 0.7143 | 20/28 | 0.3462 |
| Memory-augmented | 0.7857 | 22/28 | 0.2692 |

## C1 misses on full N=42 (10)

- `vague_wellness_weekend`: pred=dining gold=wellness m1=dining
- `exhausted_wellness_urgent`: pred=fitness gold=wellness m1=fitness
- `beauty_nail_weekend`: pred=medical gold=beauty m1=medical
- `beauty_vs_medical_skin`: pred=medical gold=beauty m1=medical
- `medical_specialist_referral`: pred=wellness gold=medical m1=wellness
- `dining_brunch_weekend`: pred=wellness gold=dining m1=fitness
- `dining_vs_wellness_food`: pred=wellness gold=dining m1=dining
- `travel_weekend_getaway`: pred=fitness gold=travel m1=travel
- `travel_vs_dining_trip`: pred=medical gold=travel m1=medical
- `wellness_sauna_relax`: pred=fitness gold=wellness m1=fitness
