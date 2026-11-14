from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.factory import get_extractor


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int((len(xs) - 1) * q)
    return xs[idx]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pipeline", "llm"], default="pipeline")
    parser.add_argument("--rounds", type=int, default=50)
    args = parser.parse_args()

    ext = get_extractor(args.mode)
    sample = "I need a relaxing spa nearby this weekend and affordable options"
    latencies = []
    for _ in range(args.rounds):
        t0 = time.perf_counter()
        _ = ext.extract(sample)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    report = {
        "mode": args.mode,
        "rounds": args.rounds,
        "latency_ms_mean": round(mean(latencies), 3),
        "latency_ms_p50": round(percentile(latencies, 0.5), 3),
        "latency_ms_p95": round(percentile(latencies, 0.95), 3),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
