# Ablation Condition Mapping

| Thesis report | Code condition | Description |
|---------------|----------------|-------------|
| C0 (Baseline) | `C0_llm_only` | Pure LLM dialogue; no intention block / no M2 tracking in prompt |
| C1 | `C1_intention_block` | LLM + `build_intention_prompt_block()` (M1/M2 summary in prompt) |
| C2 | `C2_m3_prior` | C1 + M3 history prior on belief (aggregate profile still withheld from LLM) |

Legacy M4/M5 rule-based ranking and template formatting have been removed; recommendations are LLM-generated under the intention block.

## Run scripts

- M1–M3 attribution: `python scripts/run_m1_m2_attribution.py`
- Multi-turn demo: `streamlit run scripts/demo_pre_streamlit.py`
