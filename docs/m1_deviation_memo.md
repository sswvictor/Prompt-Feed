# M1 Deviation Memo

Date: 2026-06-16  
Status: **Active (Day 2)** — M1 evidence path defined; JointBERT deferred.

## Context

This memo records gaps between the original thesis route (JointBERT + slot BIO) and the current M1 implementation, plus the **accepted** path for thesis evidence.

## Known deviation points

1. **Pipeline mainline mismatch**
   - Planned: JointBERT + slot BIO with optional CRF.
   - Current: lightweight `JointNLUModel` (multinomial NB intent + token-tag slot) + heuristics in `PipelineExtractor`.

2. **Slot BIO learning fidelity**
   - Planned: model-trained token-level BIO.
   - Current: rule-assisted BIO construction dominates; slot F1 below original 0.80 target on small in-domain set.

3. **Evaluation scale**
   - In-domain `m1_train.json` / eval sets are small; conclusions need external benchmarks and vague slices.

## Decision (Day 2)

**Thesis M1 evidence = Route A + B (not C):**

| Route | Action | Role in thesis |
|-------|--------|----------------|
| **A** | Strengthen NB baseline + expand `m1_hard_cases.json` vague slice | In-domain vague input handling |
| **B** | `run_m1_public_benchmark.py` on ATIS (SNIPS optional) with intent mapping | Component credibility |
| **C** | JointBERT/HF full training | **Deferred** → Limitations / future work |

**Production default:** `extractor_mode: pipeline` (config.yaml / `.env`). LLM extractor used for ablation (`--mode llm`), not as sole M1 claim.

## Re-entry criteria for JointBERT (optional)

- Stable HF training path integrated with STV schema.
- Slot F1 ≥ 0.80 on held-out in-domain test with reproducible runs.
- Justified time budget before thesis submission.

## Related docs

- [`m1_vague_input_contract.md`](m1_vague_input_contract.md) — operational vague definition (6.15: M2-driven policy, no M1 auto-gate)
- [`m1_composite_mapping.md`](m1_composite_mapping.md) — composite training corpus (6-class)
- [`m1_label_schema.md`](m1_label_schema.md) — labels and BIO tags
