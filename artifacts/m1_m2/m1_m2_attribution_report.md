# M1M2 Attribution Report

## Conditions
- C0: LLM-only (no M2)
- C1: M1(pipeline) + M2
- C2: M1(llm) + M2

## Core Metrics
- C0 intent_accuracy: 0.5
- C1 intent_accuracy: 0.6364
- C2 intent_accuracy: 0.5455
- C1 entropy_drop_mean: 0.472264
- C2 entropy_drop_mean: 0.168741
- C1 candidate_reduction_rate_mean: 0.981481
- C2 candidate_reduction_rate_mean: 0.981481

## Attribution Reading
- If C1 > C0, structured extraction + belief update contributes beyond LLM-only.
- If C2 ~= C1, M2 is likely the dominant gain source.
- If C2 >> C1, upstream extraction quality remains a bottleneck.

## Notes
- LLM mode may fallback to mock provider if no API key is configured.