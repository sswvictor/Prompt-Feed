from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.public_benchmark import (
    DEFAULT_ATIS_PATH,
    evaluate_public_benchmark,
    format_report_md,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 external benchmark (ATIS mapped subset)")
    parser.add_argument("--mode", choices=["pipeline", "llm"], default="pipeline")
    parser.add_argument("--benchmark", default=DEFAULT_ATIS_PATH)
    args = parser.parse_args()

    benchmark_path = str(ROOT / args.benchmark) if not Path(args.benchmark).is_absolute() else args.benchmark
    report = evaluate_public_benchmark(benchmark_path, mode=args.mode)

    out_dir = ROOT / "artifacts" / "m1"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "public_benchmark_report.json"
    md_path = out_dir / "public_benchmark_report.md"

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(format_report_md(report), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
