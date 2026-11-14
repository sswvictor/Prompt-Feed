# Technical Architecture Roadmap (English)

**Canonical source:** [`Technical_Architecture_Roadmap_ShuweiSong.md`](../../Technical_Architecture_Roadmap_ShuweiSong.md) (Version 2.0, LLM-centric).

This copy summarizes the **current** architecture. Full module specs (including archived M4/M5 sections) remain in the root roadmap file.

## Active modules

| Module | Role |
|--------|------|
| M1 | STV extraction from natural language |
| M2 | Belief tracking over bookable goals |
| M3 | Privacy-preserving aggregate memory (prior only) |
| Dialogue | Intention prompt block + GPT surface |

## Removed

- **M4** mock catalog recommendation engine
- **M5** template formatter / NDCG@K eval tied to M4

## Runtime flow

See [`full_pipeline_contract.md`](full_pipeline_contract.md) and [`dialogue_contract.md`](dialogue_contract.md).

## Evaluation (current plan)

1. **M1 component:** ATIS/SNIPS intent mapping + in-domain vague slice (`m1_hard_cases.json`)
2. **Multi-turn RQ1:** scenario scripts, entropy / turns-to-converge (phase B)
3. **RQ2 privacy:** prompt audit + user perception (phase B–C)
4. **Ablation:** C0 pure LLM vs C1 intention block vs C2 + M3 prior
