# Table B4 — failure notes (v1)

Source: `thesis_condition_eval.json` (gold_match = false).  
N_scenarios = 14; total miss rows = 17 (a scenario can fail under multiple conditions).

## Summary by failure type

| Type | Pattern | Typical conditions |
|------|---------|--------------------|
| A Wellness / beauty–fitness swap | gold `wellness` → `beauty` or `fitness` | LLM-only, Intention-conditioned |
| B Medical → fitness | gold `medical` → `fitness` | all three (persistent) |
| C Dining collapse | gold `dining` → `fitness` / `wellness` | all three |
| D Travel collapse | gold `travel` → `fitness` / `wellness` | Intention / Memory; LLM-only on hotel combo |
| E Over-clarify | all or most turns `clarify`, still wrong final | Intention-conditioned (and some Memory) |

## Miss table

| Condition | Scenario | Gold | Pred | Clarify/N | Type |
|-----------|----------|------|------|-----------|------|
| LLM-only | vague_wellness_weekend | wellness | beauty | 2/3 | A |
| LLM-only | medical_wellness_ambiguous | medical | fitness | 0/3 | B |
| LLM-only | dining_budget_lunch | dining | wellness | 0/2 | C |
| LLM-only | travel_hotel_combo | travel | wellness | 1/2 | D |
| LLM-only | medical_wellness_checkup | medical | fitness | 0/3 | B |
| Intention-conditioned | vague_wellness_weekend | wellness | fitness | 2/3 | A+E |
| Intention-conditioned | medical_wellness_ambiguous | medical | fitness | 0/3 | B |
| Intention-conditioned | dining_vague_then_specific | dining | fitness | 3/3 | C+E |
| Intention-conditioned | travel_flight_vague | travel | fitness | 3/3 | D+E |
| Intention-conditioned | dining_budget_lunch | dining | wellness | 1/2 | C |
| Intention-conditioned | travel_hotel_combo | travel | wellness | 2/2 | D+E |
| Intention-conditioned | medical_wellness_checkup | medical | fitness | 2/3 | B+E |
| Memory-augmented | medical_wellness_ambiguous | medical | fitness | 0/3 | B |
| Memory-augmented | dining_vague_then_specific | dining | fitness | 3/3 | C+E |
| Memory-augmented | travel_flight_vague | travel | fitness | 3/3 | D+E |
| Memory-augmented | dining_budget_lunch | dining | wellness | 1/2 | C |
| Memory-augmented | travel_hotel_combo | travel | wellness | 2/2 | D+E |

## Cross-condition notes (for Discussion)

1. **Memory recovers two Intention misses:** `vague_wellness_weekend` and `medical_wellness_checkup` match under Memory-augmented but not under Intention-conditioned (seeded gold-aligned prior; disclose in Results).  
2. **Intention-conditioned has the lowest category match (0.50)** with the highest clarify rate (0.54): belief policy is more conservative and does not convert that caution into better finals on this set.  
3. **Shared hard families:** medical→fitness and dining/travel→fitness|wellness dominate; likely extractor / mapping confusability rather than privacy.  
4. **Privacy:** no miss row is a privacy failure (audit pass rate 1.0 on all conditions).

## Raw extract

Machine-readable miss list: `_b4_raw.json` (same folder).
