"""Headless smoke test for demo_pre_streamlit pipeline paths (no Streamlit UI)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.demo_pre_streamlit import FF_SCENARIOS  # noqa: E402
from src.dialogue import DialogueOrchestrator  # noqa: E402
from src.pipeline.config import pipeline_config_from_yaml  # noqa: E402


def main() -> None:
    cfg = pipeline_config_from_yaml()
    cfg.enable_m3 = False
    orch = DialogueOrchestrator(cfg)

    rows = []
    for name, turns in FF_SCENARIOS.items():
        session = orch.session_manager.new_session("persona_smoke")
        last = None
        for text in turns:
            last = orch.handle_turn(session, text)
        assert last is not None
        pr = last.pipeline_result
        rows.append(
            {
                "scenario": name,
                "turns": len(turns),
                "final_category": pr.stv.service_category if pr else None,
                "converged": pr.converged if pr else None,
                "policy": last.intention_block.dialogue_policy if last.intention_block else None,
            }
        )

    print("demo_streamlit_smoke OK")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
