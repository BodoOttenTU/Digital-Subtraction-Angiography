"""SQ1 slide figures.

    python scripts/sq1_slides.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sq1_summary_figure import EXAMPLE, PLANE_COL, SEG_COL, box, fmt_p, overlay, pct
from paths import DATA, PLANE, SQ1

PLANE_NAME = {"A": "frontal (A)", "B": "lateral (B)"}
SEG_SHORT = {"ica_top": "ICA-top", "ica_full_grown": "ICA-full"}

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def slide1(out, pairs, path):
    """One injection, frontal view: where each AIF is sampled and the curves that result."""
    run = pairs[(pairs.ab_pair_id == EXAMPLE) & (pairs.plane == "A")].run_id.iloc[0]
    d = np.load(out / "npz" / f"{run}.npz")
    top, grown = d["mask_ica_top"], d["mask_ica_full_grown"]
    ys, xs = np.nonzero(top | grown)
    pad, edge = 50, 8
    crop = (slice(max(ys.min() - pad, edge), min(ys.max() + pad, top.shape[0] - edge)),
            slice(max(xs.min() - pad, edge), min(xs.max() + pad, top.shape[1] - edge)))

    fig, (ax_img, ax_aif) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw=dict(width_ratios=[0.8, 1.2]))
    img = d["minip"][crop]
    ax_img.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
    grown_only = grown & ~top
    overlay(ax_img, grown_only[crop], SEG_COL["ica_full_grown"], 0.6)
    overlay(ax_img, top[crop], SEG_COL["ica_top"], 0.75)
    ax_img.set_axis_off()
    ax_img.set_title("Where the AIF is sampled")
    ty, tx = np.nonzero(top[crop])
    gy, gx = np.nonzero(grown_only[crop])
    h, w = img.shape
    ax_img.annotate("ICA-top\n(supraclinoid,\n~1 100 px)", xy=(tx.min(), ty.mean()), xytext=(0.03, 0.97),
                    textcoords="axes fraction", va="top", fontsize=13, fontweight="bold", color=SEG_COL["ica_top"],
                    bbox=dict(fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=SEG_COL["ica_top"], lw=1.5))
    low = gy > np.percentile(gy, 85)
    ax_img.annotate("ICA-full\n(petrous + cervical,\n~12 000 px)", xy=(gx[low].mean(), gy[low].mean()),
                    xytext=(0.97, 0.12), textcoords="axes fraction", ha="right", va="bottom", fontsize=13,
                    fontweight="bold", color=SEG_COL["ica_full_grown"], bbox=dict(fc="white", ec="none", alpha=0.85),
                    arrowprops=dict(arrowstyle="-", color=SEG_COL["ica_full_grown"], lw=1.5))

    for v in ("ica_top", "ica_full_grown"):
        ax_aif.plot(d["time_s"], d[f"aif_{v}"], color=SEG_COL[v], lw=3, label=SEG_SHORT[v])
    ax_aif.set_xlabel("time (s)")
    ax_aif.set_ylabel("AIF (a.u.)")
    ax_aif.set_yticks([])
    ax_aif.legend(frameon=False, loc="upper right")
    ax_aif.set_title("Same injection, two different AIFs")
    fig.tight_layout(w_pad=3)
    save(fig, path)


def slide2(pairs, tests, path):
    """Cohort: peak height and CBV, ICA-full vs ICA-top, per view."""
    t = tests.set_index(["plane", "feature"])
    n = pairs.groupby("plane").size()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, (feat, title, ylim) in zip(axes, [("aif_peak", "AIF peak height", (-40, 45)),
                                              ("CBV_median", "CBV (median per run)", (-60, 155))]):
        for k, plane in enumerate("AB"):
            med = box(ax, k, pct(pairs.loc[pairs.plane == plane, f"d_{feat}"]), PLANE_COL[plane], width=0.6)
            ax.text(k, 1.0, f"{med:+.0f}%\n{fmt_p(t.loc[(plane, feat), 'p_holm'])}", ha="center", va="bottom",
                    fontsize=14, fontweight="bold", color=PLANE_COL[plane], transform=ax.get_xaxis_transform())
        ax.axhline(0, color="k", lw=1)
        ax.set_xticks([0, 1], [f"{PLANE_NAME[p]}\nn = {n[p]}" for p in "AB"])
        ax.set_xlim(-0.6, 1.6)
        ax.set_ylim(*ylim)
        ax.set_title(title, pad=58)
    axes[0].set_ylabel("ICA-full vs ICA-top (%)")
    fig.tight_layout(w_pad=4)
    save(fig, path)


def slide3(tests, plane_pairs, plane_tests, path):
    """(a) which AIF is the same in both views; (b) how far CBV can differ for one run."""
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(13, 5.8), gridspec_kw=dict(width_ratios=[0.85, 1.15]))

    pt = plane_tests.set_index(["variant", "feature"])
    for i, v in enumerate(["ica_top", "ica_full_grown"]):
        vals = np.exp(plane_pairs.loc[plane_pairs.variant == v, "aif_rel_peak_cmp"].dropna())
        med = box(ax_r, i, vals, SEG_COL[v], width=0.6)
        ax_r.text(i, 1.0, f"{med:.2f}\n{fmt_p(pt.loc[(v, 'aif_rel_peak'), 'p_holm'])}", ha="center", va="bottom",
                  fontsize=14, fontweight="bold", color=SEG_COL[v], transform=ax_r.get_xaxis_transform())
    ax_r.axhline(1, color="k", lw=1)
    ax_r.set_yscale("log")
    ax_r.set_yticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
    ax_r.minorticks_off()
    ax_r.set_xticks([0, 1], [SEG_SHORT["ica_top"], SEG_SHORT["ica_full_grown"]])
    ax_r.set_xlim(-0.6, 1.6)
    ax_r.set_ylabel("AIF lateral ÷ frontal\n(normalised to whole-image contrast)")
    ax_r.set_title("Is the AIF the same in both views?", pad=58)

    t = tests.set_index(["plane", "feature"])
    for k, plane in enumerate("AB"):
        r = t.loc[(plane, "CBV_median")]
        lo, hi = pct(r.ba_loa_low), pct(r.ba_loa_high)
        y = 1 - k
        ax_l.plot([lo, hi], [y, y], color=PLANE_COL[plane], lw=14, alpha=0.45, solid_capstyle="butt")
        ax_l.text(lo, y + 0.2, f"{lo:+.0f}%", ha="center", fontsize=14, fontweight="bold", color=PLANE_COL[plane])
        ax_l.text(hi, y + 0.2, f"{hi:+.0f}%", ha="center", fontsize=14, fontweight="bold", color=PLANE_COL[plane])
    ax_l.axvline(0, color="k", lw=1)
    ax_l.set_yticks([1, 0], [PLANE_NAME[p] for p in "AB"])
    ax_l.set_ylim(-0.7, 1.6)
    ax_l.set_xlim(-90, 200)
    ax_l.spines["left"].set_visible(False)
    ax_l.tick_params(axis="y", length=0)
    ax_l.set_xlabel("CBV with ICA-full vs ICA-top (%)")
    ax_l.set_title("How much can CBV change for one run?\n(95% limits of agreement)", pad=20)
    fig.tight_layout(w_pad=4)
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    out = p.parse_args().out
    d = SQ1
    dst = d / "slides"
    dst.mkdir(exist_ok=True)
    pairs = pd.read_csv(d / "sq1_pairs.csv")
    pairs = pairs[pairs.comparison == "grown"]
    tests = pd.read_csv(d / "sq1_tests.csv")
    tests = tests[tests.comparison == "grown"]
    plane_pairs = pd.read_csv(PLANE / "plane_pairs.csv")
    plane_tests = pd.read_csv(PLANE / "plane_tests.csv")

    slide1(out, pairs, dst / "sq1_slide1_comparison.png")
    slide2(pairs, tests, dst / "sq1_slide2_result.png")
    slide3(tests, plane_pairs, plane_tests, dst / "sq1_slide3_conclusion.png")


if __name__ == "__main__":
    main()
