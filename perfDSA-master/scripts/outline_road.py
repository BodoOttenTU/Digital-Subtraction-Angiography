"""Outline graphic for the final presentation: a vertical road with one drawn icon per sub-question.

    python scripts/outline_road.py

No data is used. Written to results/presentation/outline_road.png (transparent background).
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, FancyArrow, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parents[1] / "results" / "presentation" / "outline_road.png"
SCARLET, NAVY, CYAN, ORANGE, GREY, LIGHT = "#c81919", "#101073", "#0092b5", "#e86a00", "#6a6a7e", "#e4e4ec"

STATIONS = [
    ("Sampling segment", "ICA-top vs ICA-full"),
    ("Projection view", "frontal vs lateral"),
    ("Patient", "patient vs measurement setup"),
    ("Perfusion", "effect on CBF, CBV, MTT"),
]
R = 0.42          # icon circle radius
X = 0.6           # road x
YS = [3.3, 2.2, 1.1, 0.0]


def icon_segment(ax, x, y):
    t = np.linspace(-1, 1, 50)
    vx, vy = x + 0.22 * np.sin(2.2 * t), y + 0.3 * t
    ax.plot(vx, vy, color=GREY, lw=5, solid_capstyle="round", zorder=4)
    k = t > 0.35
    ax.plot(vx[k], vy[k], color=SCARLET, lw=5, solid_capstyle="round", zorder=5)


def icon_view(ax, x, y):
    ax.add_patch(Circle((x + 0.06, y - 0.06), 0.15, fc="white", ec=GREY, lw=1.6, zorder=4))
    ax.add_patch(FancyArrow(x + 0.06, y + 0.33, 0, -0.15, width=0.05, head_width=0.13, head_length=0.08,
                            color=NAVY, zorder=5, length_includes_head=True))
    ax.add_patch(FancyArrow(x - 0.33, y - 0.06, 0.15, 0, width=0.05, head_width=0.13, head_length=0.08,
                            color=ORANGE, zorder=5, length_includes_head=True))


def icon_patient(ax, x, y):
    ax.add_patch(Circle((x, y + 0.13), 0.1, fc=NAVY, ec="none", zorder=4))
    ax.add_patch(FancyBboxPatch((x - 0.17, y - 0.27), 0.34, 0.25, boxstyle="round,pad=0,rounding_size=0.1",
                                fc=NAVY, ec="none", zorder=4))


def icon_perfusion(ax, x, y):
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:4, 0:4]
    v = np.clip(1 - np.hypot(xx - 1.2, yy - 1.4) / 3 + rng.normal(0, 0.08, xx.shape), 0, 1)
    cmap = plt.get_cmap("magma")
    s = 0.12
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((x - 2 * s + j * s, y - 2 * s + i * s), s * 0.92, s * 0.92,
                                   fc=cmap(0.25 + 0.7 * v[i, j]), ec="none", zorder=4))


ICONS = [icon_segment, icon_view, icon_patient, icon_perfusion]


def main():
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 4.6)
    ax.set_ylim(-0.6, 3.85)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.plot([X, X], [YS[-1], YS[0]], color=LIGHT, lw=6, solid_capstyle="round", zorder=1)
    for i, ((head, sub), y, icon) in enumerate(zip(STATIONS, YS, ICONS)):
        ax.add_patch(Circle((X, y), R, fc="white", ec=SCARLET, lw=2.5, zorder=3))
        icon(ax, X, y)
        ax.add_patch(Circle((X + R * 0.78, y + R * 0.78), 0.14, fc=SCARLET, ec="white", lw=1.5, zorder=6))
        ax.text(X + R * 0.78, y + R * 0.78, str(i + 1), color="white", fontsize=9, fontweight="bold",
                ha="center", va="center", zorder=7)
        ax.text(X + R + 0.25, y + 0.1, f"SQ{i + 1}  {head}", color=NAVY, fontsize=13, fontweight="bold",
                va="center")
        ax.text(X + R + 0.25, y - 0.2, sub, color=GREY, fontsize=11, va="center")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=300, transparent=True)
    plt.close(fig)
    print("written", OUT)


if __name__ == "__main__":
    main()
