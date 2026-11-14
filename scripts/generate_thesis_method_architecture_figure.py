"""Method architecture: one horizontal sidecar row; M3 left of M2 (prior)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parents[2] / "Thesis_new" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
EDGE = "#2f2f2f"
MUTED = "#5a5a5a"
C_MAIN = "#e9eef6"
C_MEM = "#e6f1e8"
C_SUM = "#f0e9dd"
C_LLM = "#f6efe6"
C_IO = "#ffffff"
C_BOUND = "#8a3b3b"
C_PRIOR = "#2f6b4a"
C_INJ = "#2c5a86"
FONT = "DejaVu Sans"


def rbox(ax, x, y, w, h, fc, ec=EDGE, lw=1.15):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.07",
            linewidth=lw,
            edgecolor=ec,
            facecolor=fc,
            zorder=2,
        )
    )


def arr(ax, a, b, color=EDGE, lw=1.25, ms=10):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=ms,
            linewidth=lw,
            color=color,
            shrinkA=1.5,
            shrinkB=1.5,
            zorder=3,
        )
    )


def t(ax, x, y, s, size=9, weight="normal", color=INK, ha="center", va="center"):
    ax.text(
        x,
        y,
        s,
        ha=ha,
        va=va,
        fontsize=size,
        fontweight=weight,
        color=color,
        fontfamily=FONT,
        zorder=4,
        linespacing=1.18,
    )


def main() -> None:
    fig, ax = plt.subplots(figsize=(10.6, 6.4), dpi=300)
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 11.5)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    t(ax, 10.5, 11.0, "Layered booking-assistant architecture", size=13.2, weight="bold")
    t(ax, 10.5, 10.5, "Local intention sidecar with controlled generative surface", size=9.0, color=MUTED)

    y, h = 6.55, 2.15
    # M1 | M3 | M2 | Dialogue | Intention  (single row)
    boxes = [
        (0.55, 3.35, "M1 · STV extraction", "Six-class category\nslots + confidence\nno ambiguous gate", C_MAIN, EDGE),
        (4.2, 2.85, "M3 · Aggregate\nmemory", "Aggregate counts\nprior only\nnot in prompt", C_MEM, "#3d6b4f"),
        (7.35, 3.55, "M2 · Belief tracking", "Multi-turn posterior\nentropy / top-1\nclarify vs suggest", C_MAIN, EDGE),
        (11.25, 3.7, "Dialogue control", "Clarify while tracking\ndual-track on hard pivot", C_MAIN, EDGE),
        (15.3, 4.5, "Intention prompt\nblock", "Allow-listed STV,\nhypotheses, belief,\npolicy", C_SUM, "#6a5840"),
    ]

    # Sidecar
    ax.add_patch(
        Rectangle(
            (0.3, 6.15),
            20.4,
            3.55,
            linewidth=1.4,
            edgecolor=C_BOUND,
            facecolor="#f7f9fc",
            linestyle=(0, (5, 3)),
            zorder=0,
        )
    )
    t(ax, 0.55, 9.4, "Local intention sidecar", size=8.8, weight="bold", color=C_BOUND, ha="left")

    # User → M1
    m1_cx = boxes[0][0] + boxes[0][1] / 2
    rbox(ax, m1_cx - 1.9, 9.7, 3.8, 0.58, C_IO, lw=1.2)
    t(ax, m1_cx, 9.99, "User utterance", size=10.5, weight="bold")
    arr(ax, (m1_cx, 9.7), (m1_cx, y + h))

    # Draw boxes
    for x, w, title, body, fc, ec in boxes:
        rbox(ax, x, y, w, h, fc, ec=ec)
        t(ax, x + w / 2, y + 1.5, title, size=9.0, weight="bold")
        t(ax, x + w / 2, y + 0.6, body, size=7.3, color=MUTED)

    # Arrows: M1→M2 (over M3), M3→M2 prior, M2→Dial→Int
    m1 = boxes[0]
    m3 = boxes[1]
    m2 = boxes[2]
    dial = boxes[3]
    inten = boxes[4]
    # M1 to M2: arrow along upper path
    arr(ax, (m1[0] + m1[1], y + h * 0.72), (m2[0], y + h * 0.72), lw=1.2)
    # M3 to M2 prior
    arr(ax, (m3[0] + m3[1], y + h * 0.35), (m2[0], y + h * 0.35), color=C_PRIOR, lw=1.35)
    t(ax, (m3[0] + m3[1] + m2[0]) / 2, y + h * 0.35 + 0.28, "prior", size=7.4, color=C_PRIOR, weight="bold")
    arr(ax, (m2[0] + m2[1], y + h / 2), (dial[0], y + h / 2))
    arr(ax, (dial[0] + dial[1], y + h / 2), (inten[0], y + h / 2))

    t(ax, 17.5, 5.85, "Chitchat bypasses the sidecar.", size=7.4, color=MUTED, ha="center")

    # Boundary
    by = 5.35
    ax.plot([0.5, 20.5], [by, by], color=C_BOUND, lw=1.5, ls=(0, (3, 2)), zorder=1)
    t(
        ax,
        10.5,
        by - 0.28,
        "Prompt boundary: sidecar state enters via the intention block; profiles and identifiers remain local",
        size=7.5,
        weight="bold",
        color=C_BOUND,
    )

    int_cx = inten[0] + inten[1] / 2
    llm_w, llm_h = 5.2, 0.9
    llm_x = int_cx - llm_w / 2
    llm_y = 3.2
    arr(ax, (int_cx, y), (int_cx, by + 0.04), color=C_INJ, lw=1.3)
    arr(ax, (int_cx, by - 0.5), (int_cx, llm_y + llm_h), color=C_INJ, lw=1.3)
    rbox(ax, llm_x, llm_y, llm_w, llm_h, C_LLM, ec="#7a5a28")
    t(ax, int_cx, llm_y + 0.58, "Large language model", size=10.0, weight="bold")
    t(ax, int_cx, llm_y + 0.24, "Surface realisation of clarify / suggest / observe", size=7.4, color=MUTED)

    rx, rw, rh = llm_x + llm_w + 0.4, 3.3, 0.65
    arr(ax, (llm_x + llm_w, llm_y + llm_h / 2), (rx, llm_y + llm_h / 2))
    rbox(ax, rx, llm_y + 0.12, rw, rh, C_IO, lw=1.2)
    t(ax, rx + rw / 2, llm_y + 0.12 + rh / 2, "Assistant reply", size=10.2, weight="bold")

    # Compact legend
    rbox(ax, 0.4, 0.25, 20.2, 1.5, "#fafafa", ec="#b5b5b5", lw=0.9)
    t(ax, 0.75, 1.4, "Comparative conditions", size=8.0, weight="bold", ha="left")
    t(
        ax,
        0.75,
        0.75,
        "LLM-only: no intention-block injection   ·   Intention-conditioned: inject intention prompt block   ·   "
        "Memory-augmented: enable M3→M2 prior (profile withheld)",
        size=7.4,
        color=MUTED,
        ha="left",
    )

    out_pdf = OUT_DIR / "method_architecture.pdf"
    out_png = OUT_DIR / "method_architecture.png"
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, bbox_inches="tight", facecolor="white", dpi=300)
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
