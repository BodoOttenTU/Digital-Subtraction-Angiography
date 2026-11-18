"""SQ4 slide figures.

    python scripts/sq4_slides.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_truncation import load_frames, perfusion, sample_aif  # noqa: E402
from sq4_aif_swap import PARAMS, parenchyma_roi, wide  # noqa: E402
from paths import DATA, SQ4

EXAMPLE = "PRAISE054_01_I01"
PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}
PLANE_NAME = {"A": "frontal (A)", "B": "lateral (B)"}
# slide 1: own ICA-top neutral, grown as in SQ1, other view in the lateral-plane colour
AIF_COL = {"ref": "#222222", "grown": "#00b3c7", "cross": "#eb6834"}
# only ica_top and ica_full_grown AIFs (the model-mask swap is left out of the slides)
SWAP_LABEL = {"grown": "ICA-full\nsame view", "cross": "ICA-top\nother view",
              "cross_peak": "ICA-top other view,\npeak matched"}
RNG = np.random.default_rng(0)

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "axes.labelsize": 14, "xtick.labelsize": 13, "ytick.labelsize": 13})


def save(fig, path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def slide1(out, acq, path):
    """Plane A tissue of one injection, deconvolved with its own AIF, the grown AIF and the lateral AIF."""
    g = acq[acq.ab_pair_id == EXAMPLE]
    run = {pl: g[g.plane == pl].index[0] for pl in ("A", "B")}
    frames = {pl: load_frames(acq.loc[run[pl], "source"]) for pl in ("A", "B")}
    n = min(f.shape[2] for f in frames.values())
    frames = {pl: f[:, :, :n] for pl, f in frames.items()}
    d = {pl: np.load(out / "npz" / f"{run[pl]}.npz") for pl in ("A", "B")}
    aifs = {"ref": sample_aif(frames["A"], d["A"]["mask_ica_top"]),
            "grown": sample_aif(frames["A"], d["A"]["mask_ica_full_grown"]),
            "cross": sample_aif(frames["B"], d["B"]["mask_ica_top"])}
    roi = parenchyma_roi(frames["A"])
    cbv = {k: perfusion(frames["A"], a)["CBV"] for k, a in aifs.items()}
    roi &= cbv["ref"] > 0
    med = {k: float(np.median(v[roi])) for k, v in cbv.items()}

    fig = plt.figure(figsize=(16, 6.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1, 1, 1], wspace=0.08)
    ax = fig.add_subplot(gs[0])
    t = np.arange(n) * 0.25
    names = {"ref": "own ICA-top (reference)", "grown": "own ICA-full", "cross": "ICA-top of the lateral view"}
    for k, a in aifs.items():
        ax.plot(t, a, color=AIF_COL[k], lw=3, ls="--" if k == "grown" else "-", label=names[k])
    ax.set_xlim(0, 8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("AIF (a.u.)")
    ax.set_yticks([])
    ax.legend(frameon=False, loc="upper right", fontsize=12)
    ax.set_title("Same tissue, three AIFs")

    ys, xs = np.nonzero(roi)
    crop = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    vmax = np.percentile(cbv["ref"][roi], 95)
    bg = d["A"]["minip"][crop]
    titles = {"ref": "own AIF", "grown": "ICA-full AIF", "cross": "lateral AIF"}
    for i, k in enumerate(aifs):
        ax = fig.add_subplot(gs[i + 1])
        ax.imshow(bg, cmap="gray", vmin=np.percentile(bg, 1), vmax=np.percentile(bg, 99.5))
        im = ax.imshow(np.ma.masked_where(~roi[crop], cbv[k][crop]), cmap="magma", vmin=0, vmax=vmax)
        ax.set_axis_off()
        ax.set_title(titles[k], color=AIF_COL[k], loc="center")
        sub = "reference" if k == "ref" else f"median CBV {100 * (med[k] / med['ref'] - 1):+.0f}%"
        ax.text(0.5, -0.04, sub, transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold",
                color=AIF_COL[k])
    cax = fig.add_axes([0.915, 0.25, 0.008, 0.5])
    fig.colorbar(im, cax=cax, ticks=[]).set_label("CBV (low → high)", fontsize=12)
    fig.suptitle("CBV map of one frontal run (same frames), only the AIF is changed", x=0.5, y=1.0,
                 fontsize=15, color="#555555")
    save(fig, path)
    return med


def slide2(w, path):
    """Per run: CBF and CBV with the swapped AIF ÷ with the own ICA-top AIF, per view."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    swaps = list(SWAP_LABEL)
    for ax, k in zip(axes, ("CBF", "CBV")):
        ax.axhspan(0.9, 1 / 0.9, color="#dddddd", zorder=0)
        for i, swap in enumerate(swaps):
            for j, plane in enumerate("AB"):
                v = np.exp(w.loc[(w.swap == swap) & (w.plane == plane), f"d_{k}"].dropna())
                x = i + (j - 0.5) * 0.36
                ax.scatter(x + RNG.uniform(-0.1, 0.1, len(v)), v, s=14, color=PLANE_COL[plane], alpha=0.5, lw=0)
                q1, m, q3 = np.percentile(v, [25, 50, 75])
                ax.plot([x, x], [q1, q3], color=PLANE_COL[plane], lw=5, solid_capstyle="butt")
                ax.plot([x - 0.13, x + 0.13], [m, m], color="k", lw=2.5)
        ax.axhline(1, color="k", lw=1)
        ax.set_yscale("log")
        ax.set_ylim(0.2, 6)
        ax.set_yticks([0.25, 0.5, 1, 2, 4], ["÷4", "÷2", "same", "×2", "×4"])
        ax.minorticks_off()
        ax.set_xticks(range(len(swaps)), [SWAP_LABEL[s] for s in swaps], fontsize=12)
        ax.set_title(k)
    axes[0].set_ylabel("value with swapped AIF\n÷ value with own AIF")
    axes[0].text(-0.45, 1.12, "±10 %", fontsize=11, color="#777777", va="center")
    handles = [Patch(color=PLANE_COL[p], label=f"{PLANE_NAME[p]} tissue") for p in "AB"]
    handles.append(plt.Line2D([], [], color="k", lw=2.5, label="median (bar = IQR, dot = run)"))
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), fontsize=13)
    fig.tight_layout(w_pad=2)
    save(fig, path)


def slide3(wb, path):
    """Within-patient change ÷ between-patient spread, worst of the two views, per swap and parameter."""
    swaps = list(SWAP_LABEL)
    r = wb.pivot_table(index=["swap", "param"], columns="plane", values="ratio")
    worst = np.array([[r.loc[(s, k)].max() for k in PARAMS] for s in swaps])
    cls = np.digitize(worst, [0.5, 1.0])
    cmap = ListedColormap(["#cfe8d4", "#fbe3b0", "#f4b6b0"])

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.imshow(cls, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i, s in enumerate(swaps):
        for j, k in enumerate(PARAMS):
            ax.text(j, i, f"{worst[i, j]:.2f}", ha="center", va="center", fontsize=17, fontweight="bold",
                    color="#222222")
    ax.set_xticks(range(len(PARAMS)), PARAMS, fontsize=15)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(swaps)), [SWAP_LABEL[s].replace("\n", " ") for s in swaps], fontsize=14)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(PARAMS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(swaps)), minor=True)
    ax.grid(which="minor", color="white", lw=4)
    ax.tick_params(which="minor", length=0)
    handles = [Patch(color=cmap(0), label="< 0.5  AIF choice ≪ patient differences"),
               Patch(color=cmap(1), label="0.5–1  comparable"),
               Patch(color=cmap(2), label="> 1  AIF choice > patient differences")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3, frameon=False, fontsize=12)
    ax.set_title("Change caused by the AIF ÷ spread between patients (worst view)", pad=40)
    fig.tight_layout()
    save(fig, path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    p.add_argument("--skip-example", action="store_true", help="Do not recompute slide 1 (needs the DICOMs).")
    args = p.parse_args()
    sq4 = SQ4
    dst = sq4 / "slides"
    dst.mkdir(exist_ok=True)
    acq = pd.read_csv(args.out / "acquisition_table.csv").set_index("run_id")
    if not args.skip_example:
        print("slide 1 example medians:", slide1(args.out, acq, dst / "sq4_slide1_design.png"))
    slide2(wide(pd.read_csv(sq4 / "swap_values.csv")), dst / "sq4_slide2_change.png")
    slide3(pd.read_csv(sq4 / "sq4_within_between.csv"), dst / "sq4_slide3_reliability.png")
    print(f"written to {dst}")


if __name__ == "__main__":
    main()
