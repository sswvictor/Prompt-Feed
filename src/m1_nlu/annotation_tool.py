from __future__ import annotations

import argparse
import json
from pathlib import Path


TEMPLATE = {
    "text": "",
    "service_category": "other",
    "temporal_preference": None,
    "urgency_level": "medium",
    "location_context": None,
    "sentiment_tags": [],
}


def init_file(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("[]", encoding="utf-8")


def add_sample(path: str, sample: dict) -> None:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    rows.append(sample)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="M1 annotation helper")
    parser.add_argument("--file", default="data/annotations/m1_train.json")
    parser.add_argument("--init", action="store_true")
    parser.add_argument("--text")
    parser.add_argument("--service_category")
    parser.add_argument("--temporal_preference", default=None)
    parser.add_argument("--urgency_level", default="medium")
    parser.add_argument("--location_context", default=None)
    parser.add_argument("--sentiment_tags", default="")
    args = parser.parse_args()

    init_file(args.file)
    if args.init:
        print(f"Initialized {args.file}")
        return

    if not args.text or not args.service_category:
        print("Provide --text and --service_category to add sample.")
        return

    sample = dict(TEMPLATE)
    sample["text"] = args.text
    sample["service_category"] = args.service_category
    sample["temporal_preference"] = None if args.temporal_preference in {"", "none", "null"} else args.temporal_preference
    sample["urgency_level"] = args.urgency_level
    sample["location_context"] = None if args.location_context in {"", "none", "null"} else args.location_context
    sample["sentiment_tags"] = [x.strip() for x in args.sentiment_tags.split(",") if x.strip()]
    add_sample(args.file, sample)
    print(f"Added sample to {args.file}")


if __name__ == "__main__":
    main()
