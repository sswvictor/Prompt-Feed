# M1 Method & Evaluation Snippets (Thesis)

Draft paragraphs for the thesis **Method** and **Evaluation** sections. Aligns with Day 3 scenario eval, composite corpus, and public benchmarks.

---

## Method — Intent extraction and training data

M1 maps each user utterance to a **Structured Task Variable (STV)** comprising a six-class `service_category` (`fitness`, `wellness`, `beauty`, `medical`, `dining`, `travel`), optional slot fields, and a scalar `category_confidence`. Unlike a hard ambiguous gate, M1 preserves the extractor-predicted category each turn; multi-turn disambiguation is delegated to M2 belief updating (see [`m1_vague_input_contract.md`](m1_vague_input_contract.md)).

**Composite corpus.** Training data is assembled by `scripts/build_m1_composite_train.py` into `data/annotations/m1_train_composite.json`. The composite merges (i) in-domain Fixmeapp utterances, (ii) **seeded augmentation** for fitness/wellness (`scripts/augment_fitness_wellness.py`), (iii) Apple **LUCID** intents, (iv) ATIS flight/travel → `travel`, (v) SNIPS `BookRestaurant` → `dining`, (vi) HASD → `medical`, (vii) SGD service keywords → beauty/medical/dining/travel. Rows labelled `ambiguous` in evaluation sets are excluded from training. Class balancing uses `--per-class-cap` (default 400); split manifests in `data/annotations/m1_split_manifest.json`.

**Extractor.** The production path uses a lightweight pipeline extractor (multinomial Naive Bayes intent head + rule-assisted slot tagging) configured via `config.yaml` (`extractor_mode: pipeline`). An LLM extractor is retained for ablation only.

---

## Method — Multi-turn belief and dialogue policy

M2 maintains a posterior over a generated hypothesis space (category × temporal × location × price combinations). Each turn, M1 STV evidence updates belief via soft category likelihoods; session state carries the posterior forward (`SessionManager.process_turn`). Dialogue policy (`clarify` vs `suggest`) is resolved from M2 posterior entropy, top-1 mass, and convergence—not from M1 `clarification_needed` flags. Optional VADER-based emotion escape forces clarification after sustained negative sentiment.

---

## Evaluation — Single-turn and vague slices

**In-domain accuracy.** `scripts/run_m1_eval.py` reports intent accuracy on `m1_eval_samples.json` and, with `--by-ambiguity`, stratifies by rule-based ambiguity buckets (`high` / `medium` / `low`) defined in `src/m1_nlu/ambiguity.py`. The `m1_hard_cases.json` slice targets high-bucket vague utterances (uncertainty cues, temporal conflicts).

**Public benchmarks.** `scripts/run_m1_public_benchmark.py` evaluates mapped ATIS and SNIPS test sets (`data/benchmarks/atis_intent_mapped.json`, etc.) to demonstrate component credibility on external NLU data. Mapped categories follow [`m1_composite_mapping.md`](m1_composite_mapping.md).

**Attribution (C0–C3).** `scripts/run_m1_m2_attribution.py` compares C0 (LLM-only, no M2), C1 (pipeline + M2), C2 (LLM + M2), and C3 (pipeline + M2 + M3 prior). Day 5 pilot (N=22): C0 **50%** → C1 **68.2%** → C3 **86.4%** intent accuracy; M2 entropy drop positive on single-turn updates. This complements pipeline-level scenario C0/C1 in `run_scenario_eval.py` (not full LLM ablation from `ablation_conditions.md`).

---

## Evaluation — Multi-turn scenario sessions

**Scenario corpus.** `data/annotations/eval_scenarios.json` defines ≥10 multi-turn sessions spanning all six service categories. Each scenario lists ordered user turns with optional `gold_hints` (category, temporal, location) for analysis; resolution gold is `gold_category` at session level.

**Scenario runner.** `scripts/run_scenario_eval.py` replays each scenario through the M1+M2 pipeline:

| Metric | Description |
|--------|-------------|
| `entropy_per_turn` | Posterior entropy after each belief update |
| `turns_to_resolution` | First turn where policy = `suggest` and (converged ∨ top-1 ≥ 0.5) |
| `entropy_drop` | Turn-1 minus final-turn entropy |
| C0 vs C1 policy | C0: per-turn M1-confidence baseline; C1: multi-turn M2 `resolve_dialogue_policy()` |

Reports are written to `artifacts/m1/scenario_eval_report.json`. Scenario eval complements single-turn buckets by measuring whether belief narrows across turns and whether M2-driven clarification reduces premature suggestions relative to an M1-only policy baseline.

---

## Suggested Results wording (template)

> On the composite corpus (N=2074, cap 400/class), M1 achieved **97.4%** intent accuracy on held-in composite rows and **91.5%** on in-domain `m1_train.json`. ATIS-mapped travel accuracy was **99.7%**. On `m1_hard_cases.json`, pipeline accuracy was **64.7%** (high-bucket **64.3%**). Multi-turn scenario evaluation (N=14) under C1 showed **42.9%** resolution rate and mean entropy drop **−0.013**; C0 M1-only baseline matched resolution with lower clarify rate (18% vs 71%). **Attribution (N=22):** C0 LLM-only **50.0%** → C1 pipeline+M2 **68.2%** (entropy drop **+0.34**) → C3 +M3 **86.4%** (entropy drop **+0.53**). Fitness/wellness composite counts **262 / 230** after seeded augmentation plus LUCID.

Replace bracketed values from `artifacts/m1/composite_build_report.json`, `artifacts/m1/public_benchmark_report.json`, and `artifacts/m1/scenario_eval_report.json`.
