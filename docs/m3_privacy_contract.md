# M3 Privacy Contract

## Purpose

M3 (Privacy-Preserving Memory) stores **aggregate statistics only** and converts them into a **belief prior** for M2. It does not store raw dialogue, PII, or service-level behavioral logs.

## Allowed fields (`AggregatedUserProfile`)

- `user_id` — anonymous identifier
- `category_counts`, `temporal_counts`, `price_tier_counts`
- `last_updated`, `total_sessions`

## Forbidden

- Raw user text, email, phone, name, address, geo coordinates
- Passing history into LLM prompts or M1 extraction

## User controls

- `MemoryStore.get_summary(user_id)` — transparency view
- `MemoryStore.delete_profile(user_id)` — reset / erasure

## Pipeline integration

Enable full M1→M3→M2 flow via:

```python
PipelineConfig(enable_m3=True, user_id="persona_fitness_heavy")
PipelineRunner(config).process_text(utterance)
```

With `enable_m3=False` (default), behavior matches standalone M1+M2 with uniform prior.
