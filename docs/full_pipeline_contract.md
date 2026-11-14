# Full Pipeline Contract (LLM-centric)

## Config combinations

| Mode | enable_m3 | inject_intention_block | Flow |
|------|-----------|------------------------|------|
| Standalone M1+M2 | false | — | extract → uniform prior → belief |
| + M3 prior | true | — | extract → history prior → belief |
| + LLM dialogue | * | true/false | above → intention block → LLM reply |

Natural-language recommendations and clarifications are **LLM-generated**, not a separate rank-and-template module.

## Entry points

```python
from src.pipeline.config import PipelineConfig
from src.pipeline.runner import PipelineRunner
from src.dialogue import DialogueOrchestrator
from src.pipeline.session import SessionManager

cfg = PipelineConfig(enable_m3=True, user_id="persona_wellness_weekend")
runner = PipelineRunner(cfg)
mgr = SessionManager(cfg, runner=runner)
session = mgr.new_session("persona_wellness_weekend")

orch = DialogueOrchestrator(cfg, session_manager=mgr)
turn = orch.handle_turn(session, "relaxing spa weekend nearby")
# turn.assistant_message — LLM reply
# turn.intention_block — what was allowed in the prompt
# turn.pipeline_result — M1/M2/M3 tracking
```

## Session (multi-turn)

```python
r1 = mgr.process_turn(session, "workout tonight")
r2 = mgr.process_turn(session, "actually weekend instead")
orch.end_session_with_outcome(session)  # M3 aggregate writeback
```

## Scripts

- `streamlit run scripts/demo_pre_streamlit.py` → **Multi-turn Assistant** tab
