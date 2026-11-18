"""Methods figures: AIF masks on one injection and an annotated AIF curve with its features.

    python scripts/methods_figures.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch, Rectangle
from scipy import ndimage as ndi
from skimage.morphology import disk

from sq1_summary_figure import SEG_COL
from paths import DATA, ROOT

# trusted in both planes and both masks; chosen by eye from the candidates with one main ICA-full piece
MASK_EXAMPLE = "PRAISE046_01_I01"
OPEN_RADIUS = 3  # px, removes spurs thinner than this from the displayed masks
SMOOTH_SIGMA = 1.5  # px, smooths the displayed outline
CURVE_EXAMPLE = "PRAISE058_01_I01"  # ICA-top only: smooth curve, labels stay clear
PLANE_NAME = {"A": "Frontal", "B": "Lateral"}
SEG_LABEL = {"ica_top": "ICA-top", "ica_full_grown": "ICA-full"}
INK, GUIDE, FILL = "#222222", "#8a8a8a", "#dcdcdc"
FIRST_PASS_REF = 0.15  # arrival fraction, aif_features_config.json
EDGE = 8


def style():
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix", "pdf.fonttype": 42, "font.size": 8, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titlesize": 8, "axes.titleweight": "normal",
                         "axes.titlelocation": "left"})


def save(fig, out_dir, name):
    for ext, kw in (("pdf", {}), ("png", dict(dpi=300))):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, facecolor="white", **kw)
    plt.close(fig)


def square_crop(mask, pad=40):
    ys, xs = np.nonzero(mask)
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) // 2 + pad
    half = min(half, mask.shape[0] // 2 - EDGE)
    cy = np.clip((ys.max() + ys.min()) // 2, half + EDGE, mask.shape[0] - half - EDGE)
    cx = np.clip((xs.max() + xs.min()) // 2, half + EDGE, mask.shape[1] - half - EDGE)
    return slice(cy - half, cy + half), slice(cx - half, cx + half)


def clean_mask(mask):
    """Display only: drop thin spurs and stray pieces, fill holes. The saved masks and all results are unchanged."""
    m = ndi.binary_opening(ndi.binary_fill_holes(mask), structure=disk(OPEN_RADIUS))
    lab, _ = ndi.label(m, structure=np.ones((3, 3)))
    return ndi.binary_fill_holes(lab == np.argmax(np.bincount(lab[lab > 0])))


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def fig_masks(data, feat, out_dir):
    """ICA-top and ICA-full on the same injection, both views, cleaned for display."""
    runs = feat[(feat.ab_pair_id == MASK_EXAMPLE) & (feat.variant == "ica_top")].set_index("plane").run_id
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.95))
    for ax, plane in zip(axes, "AB"):
        d = np.load(data / "npz" / f"{runs[plane]}.npz")
        raw = {v: d[f"mask_{v}"].astype(bool) for v in SEG_LABEL}
        clean = {v: clean_mask(m) for v, m in raw.items()}
        for v in SEG_LABEL:
            print(f"{plane} {v}: raw {raw[v].sum()} px, {ndi.label(raw[v], structure=np.ones((3, 3)))[1]} pieces; "
                  f"cleaned {clean[v].sum()} px, 1 piece; Dice {dice(raw[v], clean[v]):.3f}; "
                  f"holes {(ndi.binary_fill_holes(clean[v]).sum() - clean[v].sum())} px")
        crop = square_crop(clean["ica_top"] | clean["ica_full_grown"])
        img = d["minip"][crop]
        ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5))
        shape = ax.get_xlim(), ax.get_ylim()
        for v, alpha in (("ica_full_grown", 0.85), ("ica_top", 0.95)):
            ax.contourf(ndi.gaussian_filter(clean[v][crop].astype(float), SMOOTH_SIGMA), levels=[0.5, 2],
                        colors=[SEG_COL[v]], alpha=alpha)
        ax.set_xlim(shape[0])
        ax.set_ylim(shape[1])
        ax.set_axis_off()
        ax.set_title(PLANE_NAME[plane], pad=2)
    handles = [Patch(color=SEG_COL[v], label=SEG_LABEL[v]) for v in ("ica_top", "ica_full_grown")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, handlelength=1.2, handleheight=0.8,
               columnspacing=1.5, borderaxespad=0)
    fig.subplots_adjust(left=0, right=1, top=0.91, bottom=0.09, wspace=0.03)
    save(fig, out_dir, "fig_aif_masks")


def span(ax, x0, x1, y, label=None, side="right", gap=0.1):
    """Double-headed measure between x0 and x1, with its name beside it."""
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0))
    if label:
        x, ha = (x1 + gap, "left") if side == "right" else (x0 - gap, "right")
        ax.text(x, y, label, ha=ha, va="center", color=INK)


def guide(ax, x, y0, y1):
    ax.plot([x, x], [y0, y1], color=GUIDE, lw=0.5, ls=(0, (3, 2)), zorder=1)


def fig_features(data, feat, out_dir):
    """One trusted curve: amplitude and timing features (a), shape features (b). Peak-normalised, baseline at 0."""
    r = feat[(feat.ab_pair_id == CURVE_EXAMPLE) & (feat.variant == "ica_top") & (feat.plane == "A")].iloc[0]
    d = np.load(data / "npz" / f"{r.run_id}.npz")
    t, c = d["time_s"], d["aif_ica_top"]
    assert np.isclose(c.max() - r.aif_baseline, r.aif_peak, rtol=1e-3), "npz curve differs from the feature table"
    y = (c - r.aif_baseline) / r.aif_peak
    half_left = np.interp(0.5, y[t <= r.aif_ttp_s], t[t <= r.aif_ttp_s])
    half_right = half_left + r.aif_fwhm_s
    y_arr = np.interp(r.aif_arrival_s, t, y)

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.16, 2.5), sharey=True)
    for ax in (a, b):
        ax.axhline(0, color=GUIDE, lw=0.5, zorder=1)
        ax.plot(t, y, color=INK, lw=1.3, zorder=3, solid_capstyle="round")
        ax.set_xlim(0, t[-1] + 0.15)
        ax.set_ylim(-0.04, 1.42)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel("Time from first frame (s)")
    a.set_ylabel("AIF (fraction of peak)")
    a.set_title("(a) Amplitude and timing")
    b.set_title("(b) Shape")

    # (a) first-pass AUC, peak height, arrival, time to peak, rise time, FWHM
    m = (t >= r.aif_arrival_10_s) & (t <= r.aif_first_pass_end_s)
    tf = np.r_[r.aif_arrival_10_s, t[m]]
    yf = np.interp(tf, t, y)
    a.fill_between(tf, 0, yf, color=FILL, lw=0, zorder=0)
    a.annotate("First-pass AUC", xy=(r.aif_ttp_s + 3, 0.04), xytext=(r.aif_ttp_s + 3, 0.36), ha="center", va="center",
               color=INK, arrowprops=dict(arrowstyle="-", color=GUIDE, lw=0.5, shrinkA=1, shrinkB=0))
    guide(a, r.aif_arrival_s, y_arr, 1.32)
    guide(a, r.aif_ttp_s, 1.0, 1.32)
    a.plot(r.aif_arrival_s, y_arr, "o", ms=2.8, color=INK, zorder=4)
    a.annotate("", xy=(r.aif_ttp_s, 1), xytext=(r.aif_ttp_s, 0),
               arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
    a.annotate("Peak height", xy=(r.aif_ttp_s, 0.8), xytext=(r.aif_ttp_s + 1.15, 0.8), va="center", color=INK,
               arrowprops=dict(arrowstyle="-", color=GUIDE, lw=0.5, shrinkA=1, shrinkB=0))
    span(a, 0, r.aif_arrival_s, 1.32, "Bolus arrival time", gap=r.aif_ttp_s - r.aif_arrival_s + 0.12)
    span(a, 0, r.aif_ttp_s, 1.2, "Time to peak", gap=0.12)
    span(a, r.aif_arrival_s, r.aif_ttp_s, 1.08, "Rise time", gap=0.12)
    span(a, half_left, half_right, 0.5, "FWHM", gap=0.12)

    # (b) bolus MTT, temporal SD, AUC / peak, tail fraction
    mu, sd = r.aif_centroid_s, r.aif_sd_s
    x0 = mu - r.aif_auc_norm_s / 2
    b.add_patch(Rectangle((x0, 0), r.aif_auc_norm_s, 1, fill=False, ec=GUIDE, lw=0.7, ls=(0, (3, 2)), zorder=2))
    guide(b, r.aif_arrival_s, y_arr, 1.2)
    guide(b, mu, 0, 1.2)
    b.plot(r.aif_arrival_s, y_arr, "o", ms=2.8, color=INK, zorder=4)
    span(b, r.aif_arrival_s, mu, 1.2, "Bolus MTT", gap=0.12)
    span(b, x0, x0 + r.aif_auc_norm_s, 1.08, "AUC / peak", gap=0.12)
    span(b, mu - sd, mu + sd, 0.62, "Temporal SD", gap=0.12)
    # skewness: the long right tail beyond mu + SD, value from the feature table
    x0_sk, x1_sk = mu + 0.6 * sd, t[-1] - 1.9
    y0_sk, y1_sk = np.interp(x0_sk, t, y) + 0.07, np.interp(x1_sk, t, y) + 0.06
    b.annotate("", xy=(x1_sk, y1_sk), xytext=(x0_sk, y0_sk),
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0,
                               connectionstyle="arc3,rad=-0.1"), zorder=4)
    b.text(x0_sk + 0.15, y0_sk + 0.05, "Skewness", ha="left", va="bottom",
           color=INK)
    b.annotate("", xy=(t[-1], y[-1]), xytext=(t[-1], 0),
               arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=0.6, mutation_scale=5, shrinkA=0, shrinkB=0), zorder=4)
    b.text(t[-1], y[-1] + 0.05, "Tail fraction", ha="right", va="bottom", color=INK)

    fig.tight_layout(w_pad=1.5)
    save(fig, out_dir, "fig_aif_features")
    print(r[["run_id", "aif_arrival_s", "aif_ttp_s", "aif_rise_s", "aif_fwhm_s", "aif_centroid_s", "aif_mtt_bolus_s",
             "aif_sd_s", "aif_auc_norm_s", "aif_tail_frac", "aif_first_pass_end_s"]].to_string())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    args = p.parse_args()
    style()
    out_dir = ROOT / "report" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(args.out / "features" / "features_aif.csv")
    fig_masks(args.out, feat, out_dir)
    fig_features(args.out, feat, out_dir)


if __name__ == "__main__":
    main()
