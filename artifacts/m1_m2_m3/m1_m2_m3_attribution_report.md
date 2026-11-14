# M1M2M3 Attribution Report

## Conditions
- C0: LLM-only (no M2)
- C1: M1(pipeline) + M2 (uniform prior)
- C2: M1(llm) + M2
- C3: M1(pipeline) + M2 + M3 (history prior) — maps to thesis report Condition C2

## Core Metrics
- C1 intent_accuracy: 0.6364
- C3 intent_accuracy: 0.8636
- C3 profile_aligned_accuracy: 1.0
- C1 entropy_drop_mean: 0.472264
- C3 entropy_drop_mean: 0.533946

## Attribution Reading
- If C3 > C1 on profile-aligned subset, privacy memory helps when history matches intent.
- If C3 ~= C1, history prior may be washed out by STV likelihood or small sample size.

## Privacy
- M3 stores aggregate counts only; no raw dialogue in profile store.