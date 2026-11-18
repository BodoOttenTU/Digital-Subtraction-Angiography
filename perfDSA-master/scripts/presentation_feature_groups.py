"""Synthetic AIF with the three feature groups marked (amplitude, timing, shape), for the presentation.

    python scripts/presentation_feature_groups.py

No patient data: a gamma-variate bolus with a small recirculation tail.
Only the features the later slides talk about are drawn: peak height (amplitude), arrival time and
width / FWHM (timing), and the tail as the overall shape. Colours match the three boxes on the slide.
Written to results/presentation/
  feature_groups.png          5.2 x 3.3 in (earlier version, kept for other slides)
  feature_groups_slide10.png  8.35 x 2.75 in, the figure box of slide 10 in the final presentation
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "presentation"
SCARLET, NAVY, CYAN, INK = "#c81919", "#101073", "#0092b5", "#222222"
ARRIVAL_FRAC = 0.15  # aif_features_config.json

plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})


def bolus(t, t0=1.0, a=3.0, b=0.35):
    s = np.clip(t - t0, 0, None)
    y = s ** a * np.exp(-s / b)
    y /= y.max()
    s2 = np.clip(t - t0 - 2.0, 0, None)
    r = s2 ** a * np.exp(-s2 / (2.2 * b))
    y = y + 0.13 * r / r.max()
    return y / y.max()


def draw(ax, t, y, fs, label_peak, label_arr, label_width, label_tail, tail_xy):
    i_pk = int(np.argmax(y))
    t_arr = t[np.nonzero(y >= ARRIVAL_FRAC)[0][0]]
    half = np.nonzero(y >= 0.5)[0]
    t_l, t_r = t[half[0]], t[half[-1]]
    tail = t > t_r + 0.7  # starts right of the peak-height label
    ax.fill_between(t[tail], 0, y[tail], color=CYAN, alpha=0.25, lw=0)
    ax.text(*tail_xy, label_tail, color=CYAN, fontsize=fs, fontweight="bold", ha="center")
    ax.plot(t, y, color=INK, lw=2.8, zorder=3)
    # amplitude
    ax.annotate("", xy=(t[i_pk], 1.0), xytext=(t[i_pk], 0), arrowprops=dict(arrowstyle="<->", color=SCARLET, lw=2.2))
    ax.text(t[i_pk] + 0.1, 0.06, label_peak, color=SCARLET, fontsize=fs, fontweight="bold")  # under the curve
    # timing
    ax.axvline(t_arr, color=NAVY, lw=1.6, ls="--")
    ax.text(t_arr + 0.08, 1.04, label_arr, color=NAVY, fontsize=fs, fontweight="bold", ha="left", va="bottom")
    ax.annotate("", xy=(t_l, 0.5), xytext=(t_r, 0.5), arrowprops=dict(arrowstyle="<->", color=NAVY, lw=2.2))
    ax.text(t_r + 0.12, 0.5, label_width, color=NAVY, fontsize=fs, fontweight="bold", va="center")
    ax.set_ylim(-0.03, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("AIF")
    return t_arr, t_l, t_r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, 8, 400)
    y = bolus(t)

    # earlier square-ish version
    fig, ax = plt.subplots(figsize=(5.2, 3.3), constrained_layout=True)
    draw(ax, t, y, 12, "peak height", "arrival", "width", "tail / shape", (5.3, 0.27))
    ax.set_xlim(0, 8)
    fig.savefig(OUT / "feature_groups.png", dpi=300, facecolor="white")
    plt.close(fig)

    # slide 10: wide box, the curve stretched over the whole width
    fig, ax = plt.subplots(figsize=(8.35, 2.75), constrained_layout=True)
    t_arr, t_l, t_r = draw(ax, t, y, 14, "peak height", "arrival time", "width (FWHM)", "tail = shape",
                           (5.6, 0.25))
    ax.set_xlim(0, 8)
    path = OUT / "feature_groups_slide10.png"
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"arrival {t_arr:.2f} s, FWHM {t_r - t_l:.2f} s ({t_l:.2f}-{t_r:.2f}); written to {OUT}")


if __name__ == "__main__":
    main()
