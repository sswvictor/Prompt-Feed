from __future__ import annotations

from pathlib import Path

from src.m1_nlu.dataset import category_distribution, load_samples, split_samples
from src.m1_nlu.train import train_m1


def test_dataset_load_and_split() -> None:
    samples = load_samples("data/annotations/m1_train.json")
    assert len(samples) >= 5
    train, val, test = split_samples(samples, seed=1)
    assert len(train) + len(val) + len(test) == len(samples)
    dist = category_distribution(train)
    assert isinstance(dist, dict)
    assert len(dist) >= 1


def test_train_stub_produces_artifact(tmp_path: Path) -> None:
    out = tmp_path / "model_stub.json"
    model_out = tmp_path / "model.json"
    artifact = train_m1(
        "data/annotations/m1_train.json",
        str(out),
        model_path=str(model_out),
    )
    assert artifact["train_size"] > 0
    assert artifact["model_type"] == "multinomial_nb_intent + token_tag_slot"
    assert out.exists()
    assert model_out.exists()
