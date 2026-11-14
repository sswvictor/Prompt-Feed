# Archived v5-era training files (FAILED — do not use)

**Status:** Negative experiment. Canonical thesis train/model is **v4**.

| File | Notes |
|------|--------|
| `m1_train_composite_v5era.json` | Cap≈500 rebuild (~2761) |
| `m1_train_composite_eq_v5era.json` | Cap500 + confusion (~2950) |

Active paths (restored 2026-08-01):

- `../m1_train_composite.json` → Day-5 style **N=2074** (cap 400)
- `../m1_train_composite_eq.json` → composite + confusion (**N=2264**)
- `artifacts/m1/model.json` → **v4 weights** (`model_pre_v5_backup.json`), not a fresh retrain on the reconstructed texts
