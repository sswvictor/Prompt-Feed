"""Generate academic system-overview figure for thesis Chapter 1."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[2] / "Thesis_new" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_TEXT = "#1a1a1a"
COLOR_EDGE = "#333333"
COLOR_SIDEcar = "#e8eef7"
COLOR_LLM = "#fff4e6"
COLOR_BOUNDARY = "#7a2e2e"
FONT = "DejaVu Sans"


def _box(ax, x, y, w, h, title, body, facecolor, title_size=10, body_size=8.8):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=COLOR_EDGE,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=title_size, fontweight="bold", color=COLOR_TEXT)
    ax.text(x + 0.16, y + h - 0.62, body, ha="left", va="top", fontsize=body_size, color=COLOR_TEXT, linespacing=1.35)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.0,
            color=COLOR_EDGE,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.8), dpi=300)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.5)
    ax.axis("off")

    cx = 5.0
    x0 = 1.15
    bw = 7.7

    ax.text(cx, 11.85, "User utterance", ha="center", va="center", fontsize=11.2, fontweight="bold", color=COLOR_TEXT)
    _arrow(ax, cx, 11.55, cx, 11.05)

    # Sidecar container
    sy, sh = 4.35, 6.55
    ax.add_patch(
        Rectangle(
            (x0 - 0.12, sy - 0.12),
            bw + 0.24,
            sh + 0.24,
            linewidth=1.1,
            edgecolor=COLOR_BOUNDARY,
            facecolor="none",
            linestyle=(0, (4, 3)),
        )
    )
    ax.text(x0 + bw + 0.08, sy + sh - 0.05, "Local intention sidecar", ha="left", va="top", fontsize=8.6, color=COLOR_BOUNDARY)

    h = 1.05
    g = 0.18
    y = sy + sh - h - 0.22
    modules = [
        ("Natural language understanding", "Structured task variables per turn"),
        ("Aggregate memory", "Local personalisation priors only"),
        ("Belief tracker", "Posterior over goals and dialogue policy"),
        ("Dialogue controller", "Clarification and topic-transition control"),
        ("Intention summary", "Structured prompt block for generation"),
    ]
    for title, body in modules:
        _box(ax, x0, y, bw, h, title, body, COLOR_SIDEcar)
        y -= h + g

    _arrow(ax, cx, sy, cx, sy - 0.42)

    llm_h = 1.05
    llm_y = 2.75
    _box(
        ax,
        x0,
        llm_y,
        bw,
        llm_h,
        "Large language model",
        "User-visible natural-language reply",
        COLOR_LLM,
        body_size=9,
    )

    _arrow(ax, cx, llm_y, cx, llm_y - 0.42)
    ax.text(cx, 2.05, "Assistant reply", ha="center", va="center", fontsize=11.2, fontweight="bold", color=COLOR_TEXT)

    ax.text(
        0.35,
        0.45,
        "Only the intention summary is passed to the LLM;\nchat logs and raw memory remain outside the model prompt.",
        ha="left",
        va="bottom",
        fontsize=8.2,
        color=COLOR_BOUNDARY,
    )

    fig.savefig(OUT_DIR / "system_overview.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "system_overview.png", bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_DIR / 'system_overview.pdf'}")


if __name__ == "__main__":
    main()
