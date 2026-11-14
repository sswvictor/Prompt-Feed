# Table B2 — component background (v1)

Collected alongside thesis condition freeze. Sources: hard-case / public benchmark runners + Day-5 composite snapshot for in-domain train check.

| Split | Metric | Value | Source |
|-------|--------|------:|--------|
| In-domain `m1_train.json` (Day-5 snapshot) | Intent accuracy | 0.915 | `composite_build_report.json` eval_snapshot |
| Hard cases (pipeline) | Intent accuracy | 0.5882 | `artifacts/m1/hard_cases_vague_slice.json` |
| Hard cases high-ambiguity bucket | Intent accuracy | 0.3571 | same |
| ATIS travel mapped subset | Intent accuracy | 0.997 | `public_benchmark_report.md` |
| ATIS overall (mapped file) | Intent accuracy | 0.9968 | same |

**Role:** background only — does not answer RQ1/RQ2.
