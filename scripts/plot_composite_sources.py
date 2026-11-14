"""Plot composite training data sources by service category (stacked bar)."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORIES = ["fitness", "wellness", "beauty", "medical", "dining", "travel"]


def main() -> None:
    data_path = ROOT / "data" / "annotations" / "m1_train_composite.json"
    out_dir = ROOT / "artifacts" / "m1" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        cat = row.get("service_category", "other")
        src = row.get("source", "unknown")
        counts[cat][src] += 1

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        summary = {cat: dict(counts.get(cat, {})) for cat in CATEGORIES}
        out_json = out_dir / "composite_sources_summary.json"
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"matplotlib unavailable; wrote {out_json}")
        return

    sources = sorted({s for cat in counts for s in counts[cat]})
    bottom = [0] * len(CATEGORIES)
    fig, ax = plt.subplots(figsize=(10, 5))
    for src in sources:
        vals = [counts[cat].get(src, 0) for cat in CATEGORIES]
        ax.bar(CATEGORIES, vals, bottom=bottom, label=src)
        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_title("Composite corpus by category and source")
    ax.set_ylabel("count")
    ax.legend(title="source", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    out_png = out_dir / "composite_sources_stacked.png"
    fig.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
