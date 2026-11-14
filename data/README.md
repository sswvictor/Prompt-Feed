# Data

Published files are **synthetic or mapped public-benchmark subsets**. They are not production logs.

| Path | Use |
|------|-----|
| `annotations/m1_train.json` | In-domain M1 labels |
| `annotations/eval_scenarios.json` / `eval_scenarios_v2.json` | Multi-turn condition eval |
| `annotations/m1_train_composite*.json` | Composite train set (derived; see `NOTICE.md`) |
| `profiles/synthetic_users.json` | Aggregate M3 personas (`persona_*`) |
| `benchmarks/atis_intent_mapped.json` | ATIS mapped subset |
| `external/` | Local-only dumps (gitignored) |
