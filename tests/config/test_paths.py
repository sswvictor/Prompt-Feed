from __future__ import annotations

from src.config.env import project_root
from src.config.paths import rel_to_root


def test_rel_to_root_strips_absolute_prefix() -> None:
    root = project_root()
    abs_cfg = root / "config.yaml"
    assert rel_to_root(abs_cfg) == "config.yaml"
    assert rel_to_root("data/annotations/m1_train.json") == "data/annotations/m1_train.json"
