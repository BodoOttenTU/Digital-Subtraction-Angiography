"""
One 16:9 figure for the progress-update slide, from results already computed (no new analysis):
    (a) SQ1  ica_full_grown vs ica_top per plane: peak and CBV (median, IQR)     sq1_full_vs_top/sq1_pairs.csv
    (b) SQ2  ica_top AIF of plane A and B, arrival-aligned (median, IQR)        aif_features/curves_shape.csv
    (c) SQ2  B/A ratios: AIF peak, whole image, AIF / whole image, CBV, MTT     sq2_plane/sq2_pairs.csv

Example:
    python scripts/update_slide_figure.py results/combined_v1 report/figures/update_slide.png
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}


def panel_sq1(ax, pairs):
    p = pairs[pairs.comparison == "grown"]
    feats = [("d_aif_peak", "AIF peak"), ("d_CBV_median", "CBV")]
    x = np.arange(len(feats))
    for k, plane in enumerate(["A", "B"]):
        q = p[p.plane == plane]
        med = [np.expm1(q[c].median()) * 100 for c, _ in feats]
        lo = [np.expm1(q[c].quantile(0.25)) * 100 for c, _ in feats]
        hi = [np.expm1(q[c].quantile(0.75)) * 100 for c, _ in feats]
        xx = x + (k - 0.5) * 0.35
        ax.bar(xx, med, 0.35, color=PLANE_COL[plane], label=f"plane {plane} (n={len(q)})")
        ax.errorbar(xx, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor="k", lw=1)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x, [n for _, n in feats])
    ax.set_ylabel("full (grown) vs top, % (median, IQR)")
    ax.set_title("SQ1  Longer ICA segment:\neffect flips between views", fontsize=11)
    ax.set_ylim(-25, 80)
    ax.legend(fontsize=8, frameon=False, loc="upper left")


def panel_curves(ax, sq2, curves, feat):
    p = sq2[sq2.variant == "ica_top"]
    c = curves[(curves.variant == "ica_top") & curves.t_from_arrival_s.between(-1, 8)]
    peak = feat[feat.variant == "ica_top"].set_index("run_id").aif_peak
    for plane in ("A", "B"):
        sub = c[c.run_id.isin(p[f"run_{plane}"])].copy()
        sub["raw"] = sub.c_peak_norm * sub.run_id.map(peak)
        g = sub.groupby("t_from_arrival_s").raw
        ax.plot(g.median().index, g.median(), color=PLANE_COL[plane],
                label="A frontal" if plane == "A" else "B lateral")
        ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=PLANE_COL[plane], alpha=0.2)
    ax.set_xlabel("s after arrival")
    ax.set_ylabel("AIF (baseline-corrected)")
    ax.set_title(f"SQ2  Same bolus, two views:\nlateral AIF higher (n={len(p)})", fontsize=11)
    ax.legend(fontsize=8, frameon=False)


def panel_ratios(ax, sq2):
    p = sq2[sq2.variant == "ica_top"]
    items = [("d_aif_peak", "AIF peak"), ("d_global", "whole image"), ("d_rel_peak", "AIF / whole\nimage"),
             ("d_CBV_median", "CBV"), ("d_MTT_median", "MTT")]
    x = np.arange(len(items))
    med = [np.exp(p[c].median()) for c, _ in items]
    lo = [np.exp(p[c].quantile(0.25)) for c, _ in items]
    hi = [np.exp(p[c].quantile(0.75)) for c, _ in items]
    col = ["#eb6834", "#eb6834", "0.55", "#7b3fa0", "#7b3fa0"]
    ax.bar(x, np.subtract(med, 1), bottom=1, color=col, width=0.6)
    ax.errorbar(x, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor="k", lw=1)
    for xi, m in zip(x, med):
        ax.text(xi + 0.33, m, f"{m:.2f}", ha="left", va="center", fontsize=9, fontweight="bold")
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x, [n for _, n in items], fontsize=9)
    ax.set_ylim(0, 1.9)
    ax.set_ylabel("ratio B / A (median, IQR)")
    ax.set_title("SQ2  AIF difference = exposure;\nperfusion maps differ more", fontsize=11)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("png", type=Path)
    args = p.parse_args()
    sq1 = pd.read_csv(args.out / "sq1_full_vs_top" / "sq1_pairs.csv")
    sq2 = pd.read_csv(args.out / "sq2_plane" / "sq2_pairs.csv")
    curves = pd.read_csv(args.out / "aif_features" / "curves_shape.csv")
    feat = pd.read_csv(args.out / "aif_features" / "features_aif.csv")

    fig, axes = plt.subplots(1, 3, figsize=(13.33, 4.6), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1.1]))
    panel_sq1(axes[0], sq1)
    panel_curves(axes[1], sq2, curves, feat)
    panel_ratios(axes[2], sq2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=200)


if __name__ == "__main__":
    main()
