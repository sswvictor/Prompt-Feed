# M1 Resource Gap Review

## Current status

- M1 contract and dual extractor paths are runnable.
- Evaluation script and tests are in place.
- Training pipeline is a placeholder stub (for architecture closure).

## Data gaps

- Target annotation size from report: 200-500 in phase-1, 2000 after augmentation.
- Current project seed data is only a small bootstrap set for pipeline verification.
- Next action: scale `data/annotations/m1_train.json` with label guidelines.

## API gaps

- Real LLM extraction path requires `OPENAI_API_KEY`.
- Without API key, code falls back to `MockLLMProvider`.
- Next action: provide key and run llm latency/quality benchmarks.

## Compute gaps

- Joint model real training (HF/PyTorch + CRF) not yet enabled.
- Current `model.py` is a stable interface placeholder.
- Next action: integrate transformer backbone when dataset reaches minimum viable size.
