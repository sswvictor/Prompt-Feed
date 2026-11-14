# ATIS → Fixmeapp `service_category` mapping

Vendored subset: [`data/benchmarks/atis_intent_mapped.json`](../data/benchmarks/atis_intent_mapped.json)

Full composite mapping (all public datasets): [`m1_composite_mapping.md`](m1_composite_mapping.md)

## Purpose

ATIS is used as an **external NLU sanity benchmark** and as a **composite training source** for the `travel` category. ATIS utterances describe flight and travel queries; with the Day 2 6-class schema they map to `travel`, not `other`.

## Mapping policy

| ATIS / source intent family | `service_category` | Notes |
|----------------------------|-------------------|--------|
| `flight`, `airfare`, `airline`, `airport`, `city`, `ground_service`, `flight_no`, `aircraft`, `arrival_time`, `departure_time`, `distance`, `meal`, `capacity`, `restriction`, `abbreviation` | `travel` | Full ATIS travel-domain intent set |
| `in_domain` (synthetic rows appended) | pass-through | Sanity check on booking-like utterances: `fitness`, `wellness`, `beauty`, `medical`, `ambiguous` |

**Day 2 upgrade:** Previously all ATIS intents mapped to `other` (out-of-taxonomy). They now map to `travel` because `travel` is a first-class Fixmeapp category.

Slot alignment with ATIS is **out of scope**; only `service_category` accuracy is reported.

## Run

```bash
python scripts/run_m1_public_benchmark.py
python scripts/run_m1_public_benchmark.py --mode llm
```

Output: `artifacts/m1/public_benchmark_report.json` and `.md`

**Note:** The vendored JSON file still contains legacy `other` labels until Item 3 rebuilds it from this mapping. Re-run the benchmark after updating `atis_intent_mapped.json`.
