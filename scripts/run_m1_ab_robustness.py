from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.evaluator import evaluate_dataset


def main() -> None:
    normal_set = "data/annotations/m1_eval_samples.json"
    hard_set = "data/annotations/m1_hard_cases.json"

    result = {
        "pipeline_normal": evaluate_dataset(normal_set, "pipeline"),
        "llm_normal": evaluate_dataset(normal_set, "llm"),
        "pipeline_hard": evaluate_dataset(hard_set, "pipeline"),
        "llm_hard": evaluate_dataset(hard_set, "llm"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
