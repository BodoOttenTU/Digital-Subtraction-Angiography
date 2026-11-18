"""SQ1 report figures (example run and summary panels).

    python scripts/sq1_summary_figure.py
"""
import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy import stats
from paths import DATA, PLANE, SQ1

EXAMPLE = "PRAISE054_01_I01"  # typical pair: top ring 0.26 (A) vs 0.59 (B), grown ring ~0.25 in both
RING_PX = 15                  # as in plane_difference.py
PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}
PLANE_NAME = {"A": "frontal (A)", "B": "lateral (B)"}
SEG_COL = {"ica_top": "#e0115f", "ica_full_grown": "#00b3c7"}
SEG_NAME = {"ica_top": "ica_top (supraclinoid)", "ica_full_grown": "ica_full_grown (cervical-petrous)"}
RING_COL = "#ffd400"
RNG = np.random.default_rng(0)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def pct(x):
    return 100 * (np.exp(x) - 1)


def fmt_p(p):
    return "n.s." if p >= 0.05 else ("p < 0.001" if p < 0.001 else f"p = {p:.3f}")


def stars(p):
    return "n.s." if p >= 0.05 else "*" if p >= 0.01 else "**" if p >= 0.001 else "***"


def box(ax, x, vals, col, width=0.55):
    """Box (IQR + median) with the individual runs as jittered dots."""
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    ax.boxplot(vals, positions=[x], widths=width, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=col, alpha=0.25, edgecolor=col), medianprops=dict(color=col, lw=2.2),
               whiskerprops=dict(color=col), capprops=dict(color=col))
    ax.scatter(x + RNG.uniform(-width / 3, width / 3, len(vals)), vals, s=10, color=col, alpha=0.55, lw=0, zorder=3)
    return np.median(vals)


def ring(mask):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
    return cv.dilate(mask.astype(np.uint8), k).astype(bool) & ~mask


def overlay(ax, mask, col, alpha):
    ax.imshow(np.ma.masked_where(~mask, mask), cmap=ListedColormap([col]), alpha=alpha, interpolation="nearest")


def plot_example(out, pairs, runs, path):
    q = pairs[pairs.ab_pair_id == EXAMPLE].set_index("plane")
    rr = runs.set_index(["run_id", "variant"]).ring_vessel_frac
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.6), gridspec_kw=dict(height_ratios=[1.35, 1]))
    for k, plane in enumerate("AB"):
        run = q.loc[plane, "run_id"]
        d = np.load(out / "npz" / f"{run}.npz")
        top, grown = d["mask_ica_top"], d["mask_ica_full_grown"]
        ys, xs = np.nonzero(top | grown)
        pad = 60
        edge = 8
        crop = (slice(max(ys.min() - pad, edge), min(ys.max() + pad, top.shape[0] - edge)),
                slice(max(xs.min() - pad, edge), min(xs.max() + pad, top.shape[1] - edge)))

        ax = axes[0, k]
        img = d["minip"][crop]
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        overlay(ax, ring(top)[crop], RING_COL, 0.35)
        for v, m in (("ica_full_grown", grown), ("ica_top", top)):
            overlay(ax, m[crop], SEG_COL[v], 0.6)
            ax.contour(m[crop].astype(float), levels=[0.5], colors=[SEG_COL[v]], linewidths=1.2)
        ax.set_axis_off()
        ax.set_title(f"{'ab'[k]}  {PLANE_NAME[plane]}", color=PLANE_COL[plane])
        ax.text(0.02, 0.02, f"vessel fraction in ring\naround ica_top: {rr[(run, 'ica_top')]:.2f}",
                transform=ax.transAxes, color="white", fontsize=9, va="bottom",
                bbox=dict(fc="black", alpha=0.55, ec="none", boxstyle="round,pad=0.3"))

        ax = axes[1, k]
        for v in ("ica_top", "ica_full_grown"):
            ax.plot(d["time_s"], d[f"aif_{v}"], color=SEG_COL[v], lw=2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("AIF (a.u.)")
        ax.text(0.97, 0.95, f"grown vs top\npeak {pct(q.loc[plane, 'd_aif_peak']):+.0f}%\n"
                            f"arrival {q.loc[plane, 'd_aif_arrival_s']:+.2f} s",
                transform=ax.transAxes, ha="right", va="top", fontsize=9)
        ax.set_title(f"{'cd'[k]}  AIF, {PLANE_NAME[plane]}", color=PLANE_COL[plane])
    handles = [Patch(color=SEG_COL[v], alpha=0.7, label=SEG_NAME[v]) for v in SEG_COL]
    handles.append(Patch(color=RING_COL, alpha=0.5, label=f"{RING_PX}-px ring around ica_top"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def panel_perfusion(ax, pairs, tests):
    params = [("CBV_median", "CBV"), ("CBF_median", "CBF"), ("MTT_median", "MTT")]
    t = tests.set_index(["plane", "feature"])
    for i, (feat, name) in enumerate(params):
        for k, plane in enumerate("AB"):
            x = i * 2.6 + k
            med = box(ax, x, pct(pairs.loc[pairs.plane == plane, f"d_{feat}"]), PLANE_COL[plane])
            ax.text(x, 1.0, f"{med:+.0f}%\n{fmt_p(t.loc[(plane, feat), 'p_holm'])}", ha="center", va="bottom",
                    fontsize=8, color=PLANE_COL[plane], transform=ax.get_xaxis_transform())
    ax.set_xticks([i * 2.6 + 0.5 for i in range(3)], [n for _, n in params])
    ax.axhline(0, color="k", lw=0.7)
    ax.set_ylim(-60, 155)
    ax.set_ylabel("ica_full_grown vs ica_top (%)")
    ax.set_title("a  Perfusion: only the lateral view shifts", pad=30)


def panel_shape(axs, pairs, tests, inter):
    feats = [("aif_arrival_s", "arrival time (s)", False), ("aif_fwhm_s", "FWHM (s)", False),
             ("aif_peak", "peak height (%)", True)]
    t = tests.set_index(["plane", "feature"])
    for ax, (feat, label, is_pct) in zip(axs, feats):
        for k, plane in enumerate("AB"):
            v = pairs.loc[pairs.plane == plane, f"d_{feat}"]
            med = box(ax, k, pct(v) if is_pct else v, PLANE_COL[plane], width=0.6)
            ax.text(k, 1.0, (f"{med:+.0f}%" if is_pct else f"{med:+.2f}") + "\n" + stars(t.loc[(plane, feat), "p_holm"]),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=PLANE_COL[plane])
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks([0, 1], ["A", "B"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylabel(label, fontsize=9)
    p_flip = inter.set_index(["comparison", "feature"]).loc[("grown", "aif_peak"), "p_holm"]
    axs[2].plot([0, 1], [38, 38], color="k", lw=0.8)
    axs[2].text(0.5, 39, f"flip {stars(p_flip)}", ha="center", va="bottom", fontsize=8)
    axs[2].set_ylim(-40, 48)
    axs[0].set_title("b  AIF shape: earlier, wider, peak flips", pad=30)


def panel_why(ax, pairs, runs):
    r = runs.set_index(["run_id", "variant"]).ring_vessel_frac
    q = pairs.copy()
    q["d_ring"] = (r.loc[list(zip(q.run_id, ["ica_full_grown"] * len(q)))].to_numpy()
                   - r.loc[list(zip(q.run_id, ["ica_top"] * len(q)))].to_numpy())
    q["peak_pct"] = pct(q.d_aif_peak)
    for plane in "AB":
        s = q[q.plane == plane]
        ax.scatter(s.d_ring, s.peak_pct, s=18, color=PLANE_COL[plane], alpha=0.75, lw=0, label=PLANE_NAME[plane])
    rho = stats.spearmanr(q.d_ring, q.peak_pct)
    b, a = np.polyfit(q.d_ring, q.peak_pct, 1)
    xs = np.linspace(q.d_ring.min(), q.d_ring.max(), 2)
    ax.plot(xs, a + b * xs, color="0.3", lw=1, ls="--")
    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.text(0.03, 0.97, f"Spearman ρ = {rho.statistic:.2f}\nn = {len(q)}, {fmt_p(rho.pvalue)}",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlabel("vessel fraction around mask, grown − top\n← ica_top more overlapped")
    ax.set_ylabel("peak height, grown vs top (%)")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_title("c  Why: overlap around ica_top", pad=30)


def panel_which(ax_ring, ax_ratio, pairs, runs, plane_pairs, plane_tests):
    rr = runs.set_index(["run_id", "variant"]).ring_vessel_frac
    variants = ["ica_top", "ica_full_grown"]
    for k, plane in enumerate("AB"):
        ids = pairs.loc[pairs.plane == plane, "run_id"]
        meds = [np.median(rr.loc[list(zip(ids, [v] * len(ids)))]) for v in variants]
        xs = np.arange(2) + (k - 0.5) * 0.38
        ax_ring.bar(xs, meds, 0.36, color=PLANE_COL[plane])
        for x, m in zip(xs, meds):
            ax_ring.text(x, m + 0.015, f"{m:.2f}", ha="center", fontsize=8)
    ax_ring.set_xticks([0, 1], ["top", "grown"])
    ax_ring.set_ylim(0, 0.75)
    ax_ring.set_ylabel("vessel fraction around mask (median)")
    ax_ring.set_title("d  Which curve moves?", pad=30)

    t = plane_tests.set_index(["variant", "feature"])
    for i, v in enumerate(variants):
        vals = np.exp(plane_pairs.loc[plane_pairs.variant == v, "aif_rel_peak_cmp"].dropna())
        med = box(ax_ratio, i, vals, SEG_COL[v], width=0.55)
        ax_ratio.text(i, 1.0, f"{med:.2f}\n{fmt_p(t.loc[(v, 'aif_rel_peak'), 'p_holm'])}",
                      transform=ax_ratio.get_xaxis_transform(), ha="center", va="bottom", fontsize=8,
                      color=SEG_COL[v])
    ax_ratio.axhline(1, color="k", lw=0.7)
    ax_ratio.set_yscale("log")
    ax_ratio.set_yticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
    ax_ratio.minorticks_off()
    ax_ratio.set_xticks([0, 1], ["top", "grown"])
    ax_ratio.set_xlim(-0.6, 1.6)
    ax_ratio.set_ylabel("AIF peak / whole-image peak,\nlateral ÷ frontal")


def plot_summary(pairs, runs, tests, inter, plane_pairs, plane_tests, path):
    fig = plt.figure(figsize=(12, 9.5))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.28)
    panel_perfusion(fig.add_subplot(gs[0, 0]), pairs, tests)
    sub = gs[0, 1].subgridspec(1, 3, wspace=0.8)
    panel_shape([fig.add_subplot(sub[i]) for i in range(3)], pairs, tests, inter)
    panel_why(fig.add_subplot(gs[1, 0]), pairs, runs)
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.6)
    panel_which(fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), pairs, runs, plane_pairs, plane_tests)
    handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=PLANE_COL[k], label=PLANE_NAME[k]) for k in "AB"]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    out = p.parse_args().out
    d = SQ1
    pairs = pd.read_csv(d / "sq1_pairs.csv")
    pairs = pairs[pairs.comparison == "grown"]
    tests = pd.read_csv(d / "sq1_tests.csv")
    tests = tests[tests.comparison == "grown"]
    inter = pd.read_csv(d / "sq1_interaction.csv")
    runs = pd.read_csv(PLANE / "plane_runs.csv")
    plane_pairs = pd.read_csv(PLANE / "plane_pairs.csv")
    plane_tests = pd.read_csv(PLANE / "plane_tests.csv")

    plot_example(out, pairs, runs, d / "sq1_example.png")
    plot_summary(pairs, runs, tests, inter, plane_pairs, plane_tests, d / "sq1_summary.png")


if __name__ == "__main__":
    main()
