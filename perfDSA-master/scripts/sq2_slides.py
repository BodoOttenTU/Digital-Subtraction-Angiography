"""SQ2 slide figures.

    python scripts/sq2_slides.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sq1_summary_figure import PLANE_COL, PLANE_NAME, box, fmt_p, overlay  # noqa: E402
from paths import DATA, SQ2

EXAMPLE = "PRAISE055_01_I01"  # typical pair: peak B/A 1.30, CBV B/A close to the cohort median, ICA injection
CURVE_WINDOW = (-1.0, 8.0)    # s after arrival, as in sq2_plane.py
AIF_COL, PERF_COL = "#555555", "#7b3fb5"

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ratio_axis(ax, ticks=(0.25, 0.5, 1, 2, 4)):
    ax.axhline(1, color="k", lw=1)
    ax.set_yscale("log")
    ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()


def slide1(out, pairs, path):
    """One injection: the frontal and lateral image with the ica_top mask, and the AIF from each."""
    q = pairs[pairs.ab_pair_id == EXAMPLE].iloc[0]
    fig = plt.figure(figsize=(15, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.75, 0.75, 1.2], wspace=0.08)
    curves = {}
    for k, plane in enumerate("AB"):
        d = np.load(out / "npz" / f"{q[f'run_{plane}']}.npz")
        top = d["mask_ica_top"]
        ys, xs = np.nonzero(top)
        h, w = top.shape
        cy, cx, half = int(ys.mean()), int(xs.mean()), 190
        crop = (slice(max(cy - half, 0), min(cy + half, h)), slice(max(cx - half, 0), min(cx + half, w)))
        img = d["minip"][crop]
        ax = fig.add_subplot(gs[k])
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        overlay(ax, top[crop], PLANE_COL[plane], 0.8)
        ax.set_axis_off()
        name = PLANE_NAME[plane]
        ax.set_title(name[0].upper() + name[1:], color=PLANE_COL[plane])
        curves[plane] = (d["time_s"], d["aif_ica_top"])
    ax = fig.add_subplot(gs[2])
    for plane, (t, y) in curves.items():
        ax.plot(t, y, color=PLANE_COL[plane], lw=3)
        i = int(np.argmax(y))
        ax.text(t[i] + 0.4, y[i], PLANE_NAME[plane], color=PLANE_COL[plane], fontsize=14, fontweight="bold",
                va="center")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("AIF (a.u.)")
    ax.set_yticks([])
    ax.set_title("Same bolus, two AIFs", pad=12)
    ax.text(0.98, 0.97, "same injection\nsame frame times\nonly the view differs", transform=ax.transAxes,
            ha="right", va="top", fontsize=13, color="#444444")
    fig.subplots_adjust(left=0.02, right=0.98)
    save(fig, path)


def slide2(pairs, tests, path):
    """Cohort: lateral ÷ frontal per injection, for the AIF peak and the perfusion parameters."""
    t = tests[(tests.subset == "all pairs") & (tests.variant == "ica_top")].set_index("feature")
    items = [("aif_peak", "AIF\npeak", AIF_COL), ("CBV_median", "CBV", PERF_COL),
             ("CBF_median", "CBF", PERF_COL), ("MTT_median", "MTT", PERF_COL)]
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (feat, label, col) in enumerate(items):
        box(ax, i, np.exp(pairs[f"d_{feat}"]), col, width=0.6)
        ax.text(i, 1.0, f"{float(t.loc[feat, 'effect'].rstrip('%')):+.0f}%\n{fmt_p(t.loc[feat, 'p_holm'])}", ha="center", va="bottom",
                fontsize=14, fontweight="bold", color=col, transform=ax.get_xaxis_transform())
    ratio_axis(ax, (0.1, 0.25, 0.5, 1, 2, 4))
    ax.set_ylim(0.06, 6)
    ax.axvline(0.5, color="#bbbbbb", lw=1, ls=":")
    ax.set_xticks(range(len(items)), [lab for _, lab, _ in items])
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_ylabel("lateral ÷ frontal (same injection)")
    ax.text(0.55, 1.06, "no view effect", ha="left", va="bottom", fontsize=12, color="k")
    ax.set_title(f"Lateral vs frontal view, n = {len(pairs)} injections", pad=58)
    fig.tight_layout()
    save(fig, path)


def slide3(pairs, curves, feat, rmse, path):
    """(a) normalising the AIF makes the views coincide; (b) but CBV does not follow."""
    fig, (ax_c, ax_r) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw=dict(width_ratios=[1.2, 0.9]))

    c = curves[(curves.variant == "ica_top") & curves.t_from_arrival_s.between(*CURVE_WINDOW)]
    peak = feat[feat.variant == "ica_top"].set_index("run_id").aif_peak
    norm = c[c.run_id.isin(pairs.run_A)].run_id.map(peak).median()  # frontal median peak = 1
    for plane in "AB":
        sub = c[c.run_id.isin(pairs[f"run_{plane}"])].copy()
        sub["raw"] = sub.c_peak_norm * sub.run_id.map(peak) / norm
        for col, ls, lw in (("raw", "-", 2.5), ("c_peak_norm", "--", 2.5)):
            m = sub.groupby("t_from_arrival_s")[col].median()
            ax_c.plot(m.index, m, color=PLANE_COL[plane], ls=ls, lw=lw, alpha=1 if col == "raw" else 0.9)
    ax_c.text(1.35, 1.2, "lateral", fontsize=13, color=PLANE_COL["B"], fontweight="bold")
    ax_c.text(-0.1, 0.45, "frontal", fontsize=13, color=PLANE_COL["A"], fontweight="bold", ha="right")
    ax_c.text(2.3, 0.75, "solid: raw\ndashed: peak-normalised\n→ curves coincide", fontsize=13, color="#444444")
    med = rmse.median(numeric_only=True)
    ax_c.text(0.98, 0.97, f"A–B mismatch\nraw {100 * med.rmse_raw:.0f}% → normalised {100 * med.rmse_peak_norm:.0f}%",
              transform=ax_c.transAxes, ha="right", va="top", fontsize=13)
    ax_c.set_xlabel("time after arrival (s)")
    ax_c.set_ylabel("AIF (frontal peak = 1)")
    ax_c.set_ylim(-0.05, 1.6)
    ax_c.set_title("AIF differs only in scale", pad=12)

    items = [(np.exp(pairs.d_rel_peak), "AIF ÷ whole-\nimage contrast", AIF_COL),
             (np.exp(pairs.d_CBV_median), "CBV", PERF_COL)]
    for i, (vals, label, col) in enumerate(items):
        m = box(ax_r, i, vals, col, width=0.6)
        ax_r.text(i, 0.98, f"median {m:.2f}", ha="center", va="top", fontsize=15, fontweight="bold", color=col,
                  transform=ax_r.get_xaxis_transform())
    ratio_axis(ax_r, (0.1, 0.25, 0.5, 1, 2, 4))
    ax_r.set_ylim(0.06, 6)
    ax_r.set_xticks([0, 1], [lab for _, lab, _ in items])
    ax_r.set_xlim(-0.6, 1.6)
    ax_r.set_ylabel("lateral ÷ frontal")
    ax_r.set_title("…but CBV does not cancel", pad=12)
    fig.tight_layout(w_pad=4)
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    out = p.parse_args().out
    d = SQ2
    dst = d / "slides"
    dst.mkdir(exist_ok=True)
    pairs = pd.read_csv(d / "sq2_pairs.csv")
    pairs = pairs[pairs.variant == "ica_top"]
    tests = pd.read_csv(d / "sq2_tests.csv")
    rmse = pd.read_csv(d / "sq2_curve_rmse.csv")
    feat = pd.read_csv(out / "features" / "features_aif.csv")
    curves = pd.read_csv(out / "features" / "curves_shape.csv")

    slide1(out, pairs, dst / "sq2_slide1_comparison.png")
    slide2(pairs, tests, dst / "sq2_slide2_result.png")
    slide3(pairs, curves, feat, rmse, dst / "sq2_slide3_conclusion.png")


if __name__ == "__main__":
    main()
