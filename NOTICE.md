# Copyright and third-party notices

Copyright (c) 2026 Shuwei Song, KTH Royal Institute of Technology.

Code is MIT (`LICENSE`). Thesis: *Ethical AI for Task-Oriented User Intention Inference and Planning under Vague Input* (KTH EECS, 2026). Cite with `CITATION.cff` if you use this code.

## Data in this repository

| Path | Notes |
|------|--------|
| `data/annotations/*.json` | Thesis labels and a composite set derived from public datasets. Not real user chats. |
| `data/profiles/synthetic_users.json` | Aggregate counts, ids like `persona_*`. |
| `data/benchmarks/atis_intent_mapped.json` | Mapped ATIS subset. |
| `data/external/` | Not in git. Put SGD / LUCID / etc. here if you rebuild the composite train set (`scripts/build_m1_composite_train.py`). |

M3 only stores category / time / price-tier counts. See `docs/m3_privacy_contract.md`.

## Third-party datasets (not shipped as dumps)

- ATIS (mapped subset included)
- SNIPS (dining source when rebuilding the composite set)
- Schema-Guided Dialogue (SGD), typically CC BY 4.0: https://github.com/google-research-datasets/dstc8-schema-guided-dialogue
- Apple LUCID (`ml-lucid-datagen`): https://github.com/apple/ml-lucid-datagen

`artifacts/demo/` is defence UI artwork only. Python packages keep their own licences.
