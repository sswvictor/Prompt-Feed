from __future__ import annotations

import json

from src.dialogue.intention_context import IntentionPromptBlock


def test_intention_block_withholds_m3_and_full_history() -> None:
    block = IntentionPromptBlock(
        current_stv_summary={"service_category": "wellness", "category_confidence": 0.8},
        top_hypotheses=[{"category": "wellness", "weight": 0.4}],
        belief_state={"category_entropy": 1.2, "converged": False},
        dialogue_policy="clarify",
        user_message="maybe spa tonight",
        turn=1,
    )
    audit = block.privacy_audit()
    withheld = audit["withheld_from_llm"]
    assert any("M3" in item for item in withheld)
    assert any("user_id" in item for item in withheld)
    assert any("full belief" in item for item in withheld)

    prompt = block.to_prompt_text()
    assert "persona_" not in prompt
    assert "category_counts" not in prompt
    payload = json.dumps(audit["sent_to_llm"])
    assert "user_id" not in payload
    assert "category_counts" not in payload
