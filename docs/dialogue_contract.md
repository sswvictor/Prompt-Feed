# Dialogue layer contract (LLM multi-turn)

## Scope

The `src/dialogue/` module wraps M1→M3→M2 with routing, an **intention prompt block**, and LLM-generated task/clarify/chitchat replies. M2 belief updates still run on every task turn; the user-facing text is produced by the LLM.

## Turn types

| Type | Pipeline | Belief update |
|------|----------|---------------|
| `task` | M1→M3→M2 → intention block → LLM | Yes (`SessionManager.process_turn`) |
| `clarify` | Same; policy `clarify` steers a short question | Yes |
| `chitchat` | Bypassed | No |

## Privacy constraints (LLM prompts)

**Allowed in LLM prompts (via `IntentionPromptBlock`):**

- Current user utterance
- STV summary (category, confidence, temporal/location slots)
- Top-3 hypotheses + weights
- Belief state summary (entropy, top-1 prob, converged flag)
- Dialogue policy (`clarify` | `suggest`)

**Forbidden in LLM prompts:**

- `user_id` or persona identifiers
- M3 aggregated profile JSON or category counts
- Full `stv_history` raw texts
- Full belief vector

Use `intention_block.privacy_audit()` in the Streamlit demo to inspect sent vs withheld fields.

## Configuration

Structural defaults live in [`config.yaml`](../config.yaml). Secrets and local overrides use [`.env`](../.env) (copy from [`.env.example`](../.env.example)).

**Priority:** system environment variables > `.env` > `config.yaml`.

| Variable | Overrides |
|----------|-----------|
| `OPENAI_API_KEY` | GPT API access (required for real dialogue) |
| `DIALOGUE_ENABLE_GPT` | `dialogue.enable_gpt` |
| `DIALOGUE_MODEL` | `dialogue.model` |
| `DIALOGUE_INJECT_INTENTION_BLOCK` | `dialogue.inject_intention_block` |
| `DIALOGUE_MAX_HISTORY_TURNS` | `dialogue.max_history_turns` |
| `DIALOGUE_GPT_ROUTER_FALLBACK` | `dialogue.gpt_router_fallback` |
| `PIPELINE_EXTRACTOR_MODE` | `pipeline.extractor_mode` |
| `PIPELINE_HYPOTHESIS_MAX_SIZE` | `pipeline.hypothesis_max_size` |
| `M3_MEMORY_BACKEND` | `pipeline.memory_backend` |
| `M3_SQLITE_PATH` | `pipeline.sqlite_path` |

`pipeline_config_from_yaml()` calls `load_project_env()` automatically.

When `enable_gpt: false` or `OPENAI_API_KEY` is missing, replies fall back to fixed English templates via `MockChatProvider`.

## Configuration (`config.yaml` → `dialogue`)

```yaml
dialogue:
  enable_gpt: true
  model: gpt-4o-mini
  max_history_turns: 6
  gpt_router_fallback: true
  inject_intention_block: true
```

## Streamlit demo

The **Multi-turn Assistant** tab uses `st.session_state` for:

- `ff_chat_messages` — rendered chat history
- `ff_session` — `SessionState` (belief, turn, STV history)
- `ff_entropy_history` — belief entropy curve
- Sidebar: GPT connection status, M3 profile (local only), intention block preview, **End session** → M3 writeback

Toggles: **M3 memory**, **Intention block** (ablation C0 vs C1). Changing persona or toggles resets the chat.

## Entry point

```python
from src.dialogue import DialogueOrchestrator

orch = DialogueOrchestrator(pipeline_config, session_manager=mgr)
turn = orch.handle_turn(session, user_input, chat_history=messages)
orch.end_session_with_outcome(session)  # optional M3 writeback
```
