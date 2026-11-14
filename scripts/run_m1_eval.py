from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.paths import rel_to_root
from src.m1_nlu.evaluator import evaluate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pipeline", "llm"], default="pipeline")
    parser.add_argument("--dataset", default="data/annotations/m1_eval_samples.json")
    parser.add_argument("--by-ambiguity", action="store_true")
    parser.add_argument(
        "--compare-hard-cases",
        action="store_true",
        help="Run pipeline vs llm on m1_hard_cases.json with ambiguity buckets",
    )
    args = parser.parse_args()

    if args.compare_hard_cases:
        dataset = str(ROOT / "data" / "annotations" / "m1_hard_cases.json")
        report = {
            "dataset": rel_to_root(dataset),
            "pipeline": evaluate_dataset(dataset, "pipeline", by_ambiguity=True),
            "llm": evaluate_dataset(dataset, "llm", by_ambiguity=True),
        }
        out_dir = ROOT / "artifacts" / "m1"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "hard_cases_vague_slice.json"
        out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"\nWrote {out_json}")
        return

    out = evaluate_dataset(args.dataset, args.mode, by_ambiguity=args.by_ambiguity)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
