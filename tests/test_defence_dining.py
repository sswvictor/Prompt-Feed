from scripts.defence_dining import EXPECTED_POLICY, SCENARIO_ID, DefenceDiningRunner, run_all


def test_defence_dining_matches_thesis_table():
    turns = run_all()
    assert tuple(t.policy for t in turns) == EXPECTED_POLICY
    assert [round(t.category_entropy, 2) for t in turns] == [1.67, 1.31, 1.01]
    assert [round(t.top1_category_prob, 2) for t in turns] == [0.37, 0.60, 0.73]
    assert all(t.m1_category == "dining" for t in turns)
    assert turns[-1].pred_category == "dining"
    assert turns[0].text.startswith("I'm hungry")
    joined = "\n".join(t.prompt_text for t in turns).lower()
    assert "user_id" not in joined
    assert "category_counts" not in joined
    assert SCENARIO_ID == "dining_vague_then_specific"


def test_typed_lines_drive_the_same_path():
    runner = DefenceDiningRunner()
    lines = runner.lines_to_type()
    assert len(lines) == 3
    turns = [runner.step_text(line) for line in lines]
    assert tuple(t.policy for t in turns) == EXPECTED_POLICY
    assert [t.text for t in turns] == lines
