# v4 data restore + verification (2026-08-01)

## Policy

- **Canonical:** v4 (data recipe + frozen `model.json` / `model_pre_v5_backup.json` + B1 pack `thesis_eval/v4/`).
- **Rejected:** v5 balanced-template expansion. Files under `data/annotations/_archive_v5_failed/`.

## What was restored

1. Archived overwritten v5-era `m1_train_composite*.json`.
2. Rebuilt / downsampled `m1_train_composite.json` to Day-5 balanced counts (**N=2074**, cap 400) using `artifacts/m1/composite_build_report.json` targets.
3. Regenerated confusion merge → `m1_train_composite_eq.json` (**N=2264**; seed 44, `--extra-per-class 50`). Original v4 merge was ~2274–2276; exact utterance identity cannot be recovered after the v5 LUCID/cap overwrite.

## Verification

| Check | Result |
|-------|--------|
| `model.json` == `model_pre_v5_backup.json` | Yes (byte-identical weights) |
| B1 with frozen model vs `thesis_eval/v4/` | Match on match/clarify (C1 **0.8571**); Memory entropy_drop differs only at 1e-5 |
| Fresh full-fit on reconstructed eq (2264) | **Not** identical to frozen v4 (C1 0.9286) — texts ≠ original train set |

## Decision

Keep **frozen v4 `model.json`** as the runtime/thesis model. Restored json corpora are the **declared v4 training resources** for the thesis table; do not replace the model with a retrain on the reconstructed file unless deliberately opening a new eval pack.
