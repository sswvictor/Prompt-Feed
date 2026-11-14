# M1/M2 run tests and attribution (stages A+B)

## 1) Commands

- `pytest tests/m1_nlu -q`
- `pytest tests/m2_reasoning -q`
- `python scripts/run_m1_eval.py --mode pipeline`
- `python scripts/run_m2_sanity.py`
- `python scripts/run_m1_ab_robustness.py`
- `python scripts/run_m1_m2_attribution.py`

## 2) Baseline (no GPT)

- M1 unit tests: 7/7
- M2 unit tests: 5/5
- M1 pipeline eval: `intent_accuracy=0.8`, `slot_f1_macro=0.2` (small eval set; slot scores are low)
- M2 sanity: clear / ambiguous / conflict cases show different entropy; after feedback `top1_prob` rises (0.055715 -> 0.09961)

The first method point can be checked locally without GPT.

## 3) C0 / C1 / C2 (same 15 inputs)

- C0 (LLM-only, no M2): `intent_accuracy=0.6667`
- C1 (M1 pipeline + M2): `intent_accuracy=0.8`
- C2 (M1 llm + M2): `intent_accuracy=0.6667`

M2 metrics (means):

- C1: `belief_entropy_drop_mean=0.877516`, `top1_prob_gain_mean=0.012398`, `candidate_reduction_rate_mean=0.972222`
- C2: `belief_entropy_drop_mean=0.523676`, `top1_prob_gain_mean=0.006527`, `candidate_reduction_rate_mean=0.972222`

Reading:

- C1 > C0: structured extraction plus belief updates help.
- C2 does not beat C1: the gain is from this pipeline, not from a stronger LLM extractor.

## 4) Ambiguity buckets

- C1 accuracy: `high=0.7692`, `low=1.0`, `medium=0.0`
- C2 accuracy: `high=0.6923`, `low=0.5`, `medium=0.0`

The `medium` bucket has 0 items, so that number is not meaningful.

## 5) Error types

- C1: `m1_category_mismatch=3`, `m2_not_recovered=3`, `ambiguous_input_error=1`
- C2: `m1_category_mismatch=4`, `m2_not_recovered=4`, `ambiguous_input_error=1`

Most errors are wrong M1 category plus M2 not recovering. Next steps would be better ambiguous detection (`category_confidence < 0.4`) and more mid-ambiguity samples.

## 6) When to use GPT

- Not needed to check the first method point.
- Use a real API when comparing against a live general-purpose LLM (not the mock).

## 7) Outputs

- `artifacts/m1_m2/m1_m2_attribution_report.json`
- `artifacts/m1_m2/m1_m2_attribution_report.md`
- `artifacts/m1_m2/m1_m2_analysis.md`
