# M1 Composite Corpus — Public Dataset → `service_category` Mapping

Canonical label set: [`m1_label_schema.md`](m1_label_schema.md).

## Purpose

Fixmeapp M1 intent training uses a **composite corpus**: user utterances from multiple public NLU/dialogue datasets, each mapped to the Fixmeapp 6-class `service_category` taxonomy (`fitness`, `wellness`, `beauty`, `medical`, `dining`, `travel`) plus `other` for out-of-scope noise. Slot fields (`temporal_preference`, `urgency_level`, etc.) are filled separately (lexicon / rules); this document covers **intent mapping only**.

Build script (Item 3): `scripts/build_m1_composite_train.py` → `data/annotations/m1_train_composite.json`.

Related:

- ATIS benchmark subset policy (upgraded): [`m1_atis_mapping.md`](m1_atis_mapping.md)
- In-domain vague eval: [`m1_vague_input_contract.md`](m1_vague_input_contract.md)

## Fixmeapp 6-class taxonomy (training targets)

| `service_category` | Meaning | Primary public sources (Phase 1) |
|--------------------|---------|----------------------------------|
| `fitness` | Gym, classes, personal training | In-domain `m1_train.json` |
| `wellness` | Spa, massage, recovery | In-domain `m1_train.json` |
| `beauty` | Salon, facial, nail, skin care | In-domain `m1_train.json` |
| `medical` | Checkups, clinic, doctor visits | In-domain `m1_train.json`; Phase 2: HASD |
| `dining` | Restaurant reservation, table booking | SNIPS `BookRestaurant`; Phase 2: Restaurants-8K, SGD |
| `travel` | Flights, hotels, ground transport | ATIS travel intents; Phase 2: SGD, LUCID |
| `other` | Out-of-scope / noise | Residual SNIPS intents (skipped in training) |

`ambiguous` is **not** a training label; it remains an eval-only or runtime signal (see vague input contract).

---

## Dataset summary — which source feeds which class

| Dataset | Phase | `fitness` | `wellness` | `beauty` | `medical` | `dining` | `travel` | `other` / skip |
|---------|-------|-----------|------------|----------|-----------|----------|----------|----------------|
| **In-domain** (`m1_train.json`, `m1_hard_cases.json`) | 1 | ✓ | ✓ | ✓ | ✓ | — | — | rare |
| **ATIS** | 1 | — | — | — | — | — | ✓ | — |
| **SNIPS** | 1 | — | — | — | — | ✓ (`BookRestaurant`) | — | ✓ (6 intents skipped) |
| **SGD** | 2 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **LUCID** | 2 | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| **Restaurants-8K** | 2 | — | — | — | — | ✓ | — | — |
| **HASD** | 2 | — | — | — | ✓ | — | — | ✓ |

---

## Phase 1 — In-domain annotations

**Files:** `data/annotations/m1_train.json`, `data/annotations/m1_hard_cases.json`

| source_intent / origin | `service_category` | Notes |
|------------------------|-------------------|--------|
| *(row field `service_category`)* | pass-through | Rows are already labeled to the 6-class schema (currently `fitness`, `wellness`, `beauty`, `medical`; no `dining`/`travel` rows yet) |
| `ambiguous` (eval rows only) | pass-through | Used in `m1_hard_cases.json` for vague-slice eval; **excluded from composite training** |

**Policy:** No remapping. Merge in-domain rows as-is after validating labels against [`m1_label_schema.md`](m1_label_schema.md). `m1_hard_cases.json` is for eval / attribution, not composite training unless explicitly included in the build script.

---

## Phase 1 — ATIS (Air Travel Information System)

**Source:** Hugging Face `datasets` (`atis`) or vendored benchmark [`data/benchmarks/atis_intent_mapped.json`](../data/benchmarks/atis_intent_mapped.json).

**Role:** External NLU sanity benchmark **and** composite training source for `travel`.

**Upgrade (Day 2):** All ATIS travel-domain intents map to `travel`, **not** `other`. ATIS is now aligned with the Fixmeapp `travel` category rather than treated as out-of-taxonomy noise.

| source_intent | `service_category` | Notes |
|---------------|-------------------|--------|
| `flight` | `travel` | Flight search / availability |
| `airfare` | `travel` | Fare queries |
| `airline` | `travel` | Carrier lookup |
| `airport` | `travel` | Airport lists / codes |
| `city` | `travel` | Cities served by carriers |
| `ground_service` | `travel` | Ground transport at airport |
| `flight_no` | `travel` | Flight number lookup |
| `aircraft` | `travel` | Aircraft type |
| `arrival_time` | `travel` | Arrival schedule |
| `departure_time` | `travel` | Departure schedule |
| `distance` | `travel` | Route distance |
| `meal` | `travel` | In-flight meal info |
| `capacity` | `travel` | Seat capacity |
| `restriction` | `travel` | Flight restrictions |
| `abbreviation` | `travel` | Airline / airport codes (still travel-domain) |
| `in_domain` *(synthetic sanity rows)* | pass-through | Booking-like utterances appended to benchmark file; categories `fitness`, `wellness`, `beauty`, `medical`, `ambiguous` |

**Slot alignment:** Out of scope for M1 intent mapping; only `service_category` is evaluated on the public benchmark.

**Benchmark metrics:** `atis_travel_subset_accuracy` in `run_m1_public_benchmark.py` expects gold `travel` on all non-`in_domain` ATIS rows.

See also: [`m1_atis_mapping.md`](m1_atis_mapping.md).

---

## Phase 1 — SNIPS (built-in intents)

**Source:** Hugging Face `datasets` (e.g. `snips_built_in_intents` / Coucke et al. 2018).

**Role:** Composite training for `dining`; other intents mapped to `other` and **skipped in training** (not merged into composite JSON).

| source_intent | `service_category` | Notes |
|---------------|-------------------|--------|
| `BookRestaurant` | `dining` | **Include** in composite training |
| `AddToPlaylist` | `other` | **Skip** — music playlist; no Fixmeapp proxy |
| `GetWeather` | `other` | **Skip** — weather queries outside booking taxonomy |
| `PlayMusic` | `other` | **Skip** — media playback |
| `RateBook` | `other` | **Skip** — book rating; not local-life booking |
| `SearchCreativeWork` | `other` | **Skip** — media search |
| `SearchScreeningEvent` | `other` | **Skip** — cinema/events; too coarse to map reliably to `fitness`/`wellness`; defer to Phase 2 if needed |

### SNIPS ambiguous-intent decisions

| Intent | Decision | Rationale |
|--------|----------|-------------|
| `SearchScreeningEvent` | `other`, skip | Utterances refer to movies/shows, not bookable Fixmeapp services; mapping to `fitness` (class/event) would add label noise |
| `GetWeather` | `other`, skip | No booking intent; including would dilute 6-class classifier |
| Remaining 5 non-restaurant intents | `other`, skip | Voice-assistant tasks with no stable proxy in Fixmeapp taxonomy |

**Training policy:** Only `BookRestaurant` rows enter `m1_train_composite.json`. Other intents may appear in external benchmark reports as `other` accuracy / skip counts.

---

## Phase 2 — Deferred datasets (planned mappings)

These sources expand composite volume and under-represented classes. **Not wired in Item 3 MVP** (ATIS + SNIPS + in-domain only).

### SGD (Schema-Guided Dialogue)

**Source:** `sgd` / Multi-domain task-oriented dialogues.

| source domain / service | `service_category` | Notes |
|-------------------------|-------------------|--------|
| `Restaurants_*`, `Restaurants_ReserveRestaurant` | `dining` | Restaurant find/book flows — extract user turns |
| `Hotels_*`, `Flights_*`, `Trains_*`, `Buses_*`, `RentalCars_*`, `Travel_*` | `travel` | Travel booking domains |
| `Services_*` (gym, spa, salon, doctor, …) | `fitness` / `wellness` / `beauty` / `medical` | Map by service name keyword table (e.g. `gym` → `fitness`, `spa` → `wellness`, `salon` → `beauty`, `doctor` → `medical`) |
| `Media_*`, `Calendar_*`, `Messaging_*`, … | `other` | **Skip** — not Fixmeapp booking |

### LUCID (Apple ml-lucid-datagen, ~2024)

Local file: `data/external/lucid/ml-lucid-datagen-main/lucid_v1.0/LUCID_data.json`. User turns paired with system `expression` intents:

| LUCID `expression` intent | `service_category` | Notes |
|---------------------------|-------------------|--------|
| `book_swimming_lesson`, `book_triathlon`, `create_workout_regime`, `log_exercise` | `fitness` | 228 raw rows → 92 after dedupe/cap |
| `book_massage`, `book_spa_appointment` | `wellness` | 196 raw → 116 lucid + 4 sgd in balanced |
| salon / beauty / travel / dining tags | `beauty` / `travel` / `dining` | If present; current build uses fitness/wellness intents only |
| other / chitchat | `other` | **Skip** |

### Seeded augmentation (`augment_fitness_wellness.py`, Day 4)

Template synthesis from `m1_train.json` seeds; `source=augment`, same dedupe priority as `in_domain`:

| Method | `service_category` | Notes |
|--------|-------------------|--------|
| Vague hard-case-style templates | `fitness`, `wellness` | Mirrors `m1_hard_cases.json` phrasing (body/mood, unwind, exhausted) |
| Slot/time/location cross-product | `fitness`, `wellness` | Fills cap when LUCID dedupe leaves <200/class |
| Default targets | fitness 120, wellness 60 | Integrated in `build_m1_composite_train.py` |

**Policy:** Disclose seeded augmentation in thesis methods; not mixed into `m1_hard_cases.json` eval set.

### Restaurants-8K (PolyAI)

| source_intent / domain | `service_category` | Notes |
|------------------------|-------------------|--------|
| restaurant booking utterances (all) | `dining` | Table reservation / availability requests |

### HASD (Hybrid Appointment Scheduling Dataset)

| source_intent | `service_category` | Notes |
|---------------|-------------------|--------|
| `schedule_appointment`, `cancel_appointment`, clinic/doctor scheduling | `medical` | Filter to healthcare scheduling context only |
| non-healthcare scheduling | `other` | **Skip** or manual review |

**Medical gap:** MedDialog / MediTOD (symptom triage) are **not** mapped to `medical` for M1 intent training — task mismatch (diagnosis vs booking).

---

## Composite build rules (for Item 3)

1. **Extract** user single-turn utterances from each Phase 1 source.
2. **Map** via tables above; attach metadata: `source_dataset`, `source_intent`.
3. **Exclude** `ambiguous` gold rows and all SNIPS rows except `BookRestaurant`.
4. **Dedupe** on normalized text (lowercase, strip).
5. **Balance** per-class cap (e.g. ~2k per class in MVP); prefer in-domain rows when duplicates conflict.
6. **Split** train/val/test per [`m1_label_schema.md`](m1_label_schema.md) (seed 42) → `m1_split_manifest.json`.

Output schema (minimal training row):

```json
{
  "text": "book a table for two tomorrow evening",
  "service_category": "dining",
  "source_dataset": "snips",
  "source_intent": "BookRestaurant"
}
```

Optional slot fields may be copied from in-domain rows or left null for public-source rows.

---

## Evaluation crosswalk

| Eval track | Data | Expected mapping |
|------------|------|------------------|
| In-domain intent | `m1_train.json` holdout, `m1_eval_samples.json` | Native 6-class labels |
| Vague slice | `m1_hard_cases.json` | Native labels + `ambiguity_bucket` (not a class) |
| External sanity | `atis_intent_mapped.json` | ATIS → `travel`; synthetic `in_domain` rows |
| External sanity (future) | SNIPS test subset | `BookRestaurant` → `dining` only |

Run: `python scripts/run_m1_public_benchmark.py` (see [`m1_atis_mapping.md`](m1_atis_mapping.md)).

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-04 | Initial composite mapping doc; ATIS travel intents upgraded from `other` to `travel`; SNIPS dining-only inclusion policy; Phase 2 planned mappings for SGD, LUCID, Restaurants-8K, HASD |
| 2026-07-10 | LUCID Apple ml-lucid-datagen integrated; seeded `augment` source for fitness/wellness; balanced counts fitness 262 / wellness 230 |
