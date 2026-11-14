from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.m1_nlu.train import train_m1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", default="data/annotations/m1_train.json")
    parser.add_argument("--out_file", default="artifacts/m1/model_stub.json")
    args = parser.parse_args()
    out = train_m1(args.train_file, args.out_file)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
