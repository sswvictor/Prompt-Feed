# M1 Vague Input Contract

Operational definition of **vague input** for the Fixmeapp booking assistant thesis. No manual per-utterance vague labels are required for evaluation slices.

**Consensus (2026-06-15):** Vague input is resolved primarily by **M2 multi-turn belief**, not by an M1 confidence gate that auto-emits `ambiguous`. M1 reports `service_category` + `category_confidence` per turn; dialogue policy is driven by M2 posterior state (and optional emotion escape), not by `stv.ambiguous` or `stv.clarification_needed`.

## 1. What counts as vague?

Vague input is **underspecified natural language** where the user's situational booking intention cannot be committed to a single service category and/or slots without further dialogue.

We use **four linked mechanisms** (not subjective annotation):

| Layer | Mechanism | Output signal |
|-------|-----------|---------------|
| **M1 STV** | Per-turn extraction | `service_category` + `category_confidence` (6-class taxonomy; no auto ambiguous gate) |
| **M2 belief** | Soft category likelihood + multi-turn update | Posterior entropy, top-1 mass, `converged` |
| **Eval slice** | `ambiguity_bucket(text, row)` rules | `high` / `medium` / `low` buckets for reporting |
| **Dialogue** | `resolve_dialogue_policy()` | `dialogue_policy=clarify` vs `suggest` |

## 2. M1 / M2 / GPT division of labour

| Component | Role in vague input | Does **not** do |
|-----------|---------------------|-----------------|
| **M1** | Extract STV each turn: category, confidence, slots, sentiment tags | Auto-gate to `ambiguous`; decide clarify vs suggest |
| **M2** | Accumulate evidence across turns; soft match/mismatch from `category_confidence`; narrow hypothesis posterior | Generate user-facing text |
| **GPT (dialogue layer)** | Realize clarify questions or suggestions from `IntentionPromptBlock` | Update belief; override structural policy |

### M1 extraction (single turn)

`normalize_stv()` in `src/m1_nlu/types.py` **preserves** the extractor-predicted category and clamps confidence to `[0, 1]`. It does **not** rewrite low-confidence turns to `ambiguous` or set `clarification_needed`.

```python
# src/m1_nlu/types.py — normalize_stv() (current)
# Preserve extractor-predicted category; vague inputs handled by M2 multi-turn belief.
# clarification_needed defaults to False; dialogue policy decides follow-up.
```

**6-class taxonomy** (plus `other`, `ambiguous`): `fitness`, `wellness`, `beauty`, `medical`, `dining`, `travel`. See [`m1_label_schema.md`](m1_label_schema.md).

`ambiguous` may still appear when the extractor explicitly predicts it, but it is **not** a training label and is **not** a dialogue-policy trigger.

### M2 belief (multi-turn — main vague battlefield)

Implemented in `src/m2_reasoning/belief.py`:

- **Soft category likelihood:** `_category_match_scores(confidence)` maps `category_confidence` linearly to match/mismatch weights (conf=1 → 0.95/0.05; conf=0 → 0.05/0.95).
- **`cat == "ambiguous"`:** category term skipped (weak/no category evidence).
- **`cat == "other"`:** softened mismatch applied to all concrete hypotheses.
- **Multi-turn:** `SessionManager.process_turn()` carries posterior forward; repeated partial evidence narrows belief even when single-turn M1 confidence is medium.

| | M1 per-turn STV | M2 belief spread |
|--|-----------------|------------------|
| **Question** | What does this utterance suggest? | Which bookable goal hypothesis is most likely? |
| **When** | Every turn (language layer) | Accumulates over session turns |
| **Output** | `service_category`, `category_confidence`, slots | Posterior entropy, top-1 prob, top-K ids, `converged` |

## 3. Dialogue policy (LLM surface)

Policy is resolved by `resolve_dialogue_policy()` in `src/dialogue/intention_context.py`. **M1 STV flags are not primary triggers** (verified in `tests/dialogue/test_intention_policy.py`).

| Condition | Policy | Notes |
|-----------|--------|-------|
| `force_clarify=True` (emotion escape) | `clarify` | Overrides all other signals |
| `posterior_entropy ≥ 0.85 × log(n)` hypotheses | `clarify` | Near-uniform posterior |
| `converged` **or** `top1_prob ≥ 0.5` | `suggest` | Either sufficient |
| Otherwise (not converged and low top-1) | `clarify` | Default under uncertainty |

Constants: `TOP1_SUGGEST_THRESHOLD = 0.5`, `HIGH_ENTROPY_RATIO = 0.85`.

`build_intention_prompt_block()` passes `force_clarify=session.force_clarify` from the orchestrator. GPT generates the actual question or suggestion; the system only sets **policy**.

## 4. Emotion escape (VADER)

When users express sustained negative sentiment, the system backs off the current top category and forces clarification.

**Flow** (`src/pipeline/session.py` → `_apply_emotion_escape()`):

1. VADER compound score via `is_negative_turn()` (`src/m1_nlu/sentiment.py`); default threshold `-0.3`.
2. Track `consecutive_negative_turns` on `SessionState`.
3. After **2 consecutive** negative turns (configurable via `SentimentConfig`):
   - `apply_category_penalty()` down-weights the current top-1 **category** (`penalty=0.5`, `other_bonus=1.1`).
   - Set `session.force_clarify = True` → `resolve_dialogue_policy()` returns `clarify`.
   - Set `emotion_escape_triggered=True` on `PipelineResult`.
4. A non-negative turn resets the streak and clears `force_clarify`.

This is independent of M1 category confidence and M2 convergence; it handles user frustration when suggestions miss intent.

## 5. Ambiguity buckets (evaluation only)

Implemented in `src/m1_nlu/ambiguity.py` — used by `run_m1_m2_attribution.py` and `run_m1_eval.py --by-ambiguity`. **Not used for runtime dialogue policy.**

### `high` (vague slice)

Any of:

- Gold `service_category == "ambiguous"` (if present in row)
- Uncertainty cues: `not sure`, `maybe`, `don't know`, `any`, ` or `, `i don't know`
- Temporal conflict patterns: `now` + `weekend` in same utterance; `urgent` + `not urgent`

### `medium`

Not `high`, and fewer than two explicit slot hints (temporal, location, non-medium urgency, sentiment tags).

### `low`

Not `high`, and at least two explicit slot hints.

**Note:** Gold intent in `m1_hard_cases.json` may still be a single category (e.g. wellness) while the **utterance** is vague — bucket is computed from **text + row metadata**, not from whether the model predicted `ambiguous`.

## 6. Composite training data

M1 intent training uses a **composite corpus** built by `scripts/build_m1_composite_train.py` → `data/annotations/m1_train_composite.json`. Public datasets (ATIS, SNIPS, in-domain) map to the 6-class taxonomy; `ambiguous` gold rows are **excluded from training**.

See [`m1_composite_mapping.md`](m1_composite_mapping.md) for source→category tables and build rules.

## 7. Data assets

| File | Role |
|------|------|
| `data/annotations/m1_hard_cases.json` | In-domain vague / conflict utterances (≥15 target in high bucket) |
| `data/annotations/m1_eval_samples.json` | General in-domain eval |
| `data/annotations/m1_train_composite.json` | Composite training corpus (6-class, no `ambiguous`) |
| Demo `FF_SCENARIOS` in `demo_pre_streamlit.py` | Multi-turn vague demos |

Eval focus (Day 2): **scenario / multi-turn** sessions where M2 belief narrows over turns, plus **by-ambiguity** single-turn slices.

## 8. Metrics (Day 2+)

- **Intent accuracy** overall and **by ambiguity bucket** (`run_m1_eval.py --by-ambiguity`)
- **Multi-turn convergence** — posterior entropy drop, turns-to-converge on demo scenarios
- **Clarify vs suggest rate** — policy distribution under vague vs low-bucket utterances
- **Emotion escape rate** — `emotion_escape_triggered` on hard-case frustration turns
- **External:** ATIS/SNIPS mapped intent accuracy (`run_m1_public_benchmark.py`)

## 9. Paper wording (draft)

> Vague inputs are operationalized without manual vague labels. M1 extracts a service category and confidence each turn without an automatic ambiguous gate; multi-turn resolution is handled by M2 belief updating with soft category likelihoods. Evaluation slices use rule-based ambiguity buckets (uncertainty cues and slot conflicts). Clarification versus suggestion is decided by M2 posterior state (entropy, top-1 mass, convergence) and an optional VADER-based emotion escape; natural-language realization is delegated to the LLM.

## Related docs

- [`m1_label_schema.md`](m1_label_schema.md) — 6-class labels and STV fields
- [`m1_composite_mapping.md`](m1_composite_mapping.md) — composite training mapping
- [`m1_deviation_memo.md`](m1_deviation_memo.md) — M1 evidence path (NB + benchmarks)
- [`dialogue_contract.md`](dialogue_contract.md) — LLM prompt block and privacy
