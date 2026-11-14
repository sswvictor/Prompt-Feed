from __future__ import annotations

from src.m1_nlu.evaluator import evaluate_dataset


def test_evaluator_outputs_slot_metrics() -> None:
    out = evaluate_dataset("data/annotations/m1_eval_samples.json", "pipeline")
    assert "intent_accuracy" in out
    assert "slot_f1_macro" in out
    assert "confusion_matrix" in out
