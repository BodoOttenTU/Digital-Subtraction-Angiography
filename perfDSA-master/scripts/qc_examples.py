"""Two example AIFs that failed QC, for the presentation: truncated and no baseline.

    python scripts/qc_examples.py

Curves are the native AIFs from results/data/aif_curves.csv, normalised the same way as aif_qc.py
(baseline = median of the pre-arrival frames, peak = 1). Each example fails on one rule only.
Written to results/presentation/qc_examples/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_qc import TH, curve_features  # noqa: E402
from paths import DATA, RESULTS  # noqa: E402

OUT = RESULTS / "presentation" / "qc_examples"
EXAMPLES = {
    "truncated": ("PRAISE_PRAISE082_01_S12_B", "ica_top"),          # run ends before washout (acquisition)
    "no_baseline": ("PRAISE_PRAISE062_01_S18_A", "ica_full_grown"),  # bolus arrives at the start of the run
}
NAME = {"ica_top": "ICA-top", "ica_full_grown": "ICA-full"}
PLANE = {"A": "frontal", "B": "lateral"}
SCARLET, NAVY, GREY, GREEN = "#c81919", "#101073", "#8a8a9a", "#1a9e5a"

plt.rcParams.update({"font.size": 15, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 17, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def thresholds():
    th = dict(TH)
    cfg = DATA / "qc" / "qc_config.json"
    if cfg.exists():
        th.update({k: v for k, v in json.loads(cfg.read_text()).items() if k in th})
    return th


def curve(curves, rid, var, th):
    d = curves[(curves.run_id == rid) & (curves.variant == var)].sort_values("time_s")
    t, c = d.time_s.to_numpy(), d.aif.to_numpy()
    f, y = curve_features(t, c, th)
    return t, y, f


def base_axes(ax, t, y, title, sub):
    ax.axhline(0, color=GREY, lw=0.8)
    ax.plot(t, y, color=NAVY, lw=2.8, zorder=3)
    ax.plot(t, y, "o", color=NAVY, ms=4.5, zorder=4)
    ax.set_xlabel("time from first frame (s)")
    ax.set_ylabel("AIF (fraction of peak)")
    ax.set_ylim(-0.12, 1.25)
    ax.set_xlim(-0.3, max(t[-1] + 1.6, 9))
    ax.set_title(title, pad=26)
    ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=12, color=GREY, va="bottom")


def plot_truncated(ax, t, y, f, th, label):
    base_axes(ax, t, y, "Truncated: run ends before the washout", label)
    ax.axhline(th["TRUNC_LEVEL"], color=SCARLET, ls="--", lw=1.6)
    ax.text(t[-1] + 0.15, th["TRUNC_LEVEL"] - 0.04, f"QC limit\n{th['TRUNC_LEVEL']:.0%} of peak",
            color=SCARLET, fontsize=12, va="top")
    ax.axvspan(t[-1], ax.get_xlim()[1], color=SCARLET, alpha=0.07, lw=0)
    ax.axvline(t[-1], color=SCARLET, lw=1.6)
    ax.text(t[-1] + 0.15, 1.0, "run stops\nhere", color=SCARLET, fontsize=13, va="top", fontweight="bold")
    i_min = int(np.argmin(y[np.argmax(y):]) + np.argmax(y))
    ax.annotate(f"lowest point after peak\n= {f['post_min']:.0%} of peak", xy=(t[i_min], y[i_min]),
                xytext=(t[i_min] + 0.2, 0.62), fontsize=12, color=NAVY,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3))
    ax.text(3.4, 1.18, "The tail never returns to baseline,\nso the AIF area is cut off and\n"
            "the deconvolution is biased.", fontsize=11.5, color=GREY, va="top")


def plot_no_baseline(ax, t, y, f, th, label):
    base_axes(ax, t, y, "No baseline: bolus arrives at the start", label)
    nb = f["n_base"]
    need = th["MIN_BASE_FRAMES"]
    ax.axvspan(t[0] - 0.12, t[nb - 1] + 0.12, color=SCARLET, alpha=0.12, lw=0)
    ax.plot(t[:nb], y[:nb], "o", color=SCARLET, ms=9, zorder=5)
    ax.annotate(f"only {nb} frames before arrival\n(QC needs ≥ {need})", xy=(t[nb - 1] + 0.1, y[nb - 1] + 0.03),
                xytext=(3.4, 0.62), color=SCARLET, fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=SCARLET, lw=1.5))
    i_arr = nb
    ax.annotate("contrast arrives", xy=(t[i_arr], y[i_arr]), xytext=(t[i_arr] + 1.1, 0.12), fontsize=12,
                color=NAVY, arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.3))
    ax.text(3.4, 1.18, "Too few frames to estimate the baseline,\nso the zero level of the curve\n"
            "is uncertain.", fontsize=11.5, color=GREY, va="top")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    th = thresholds()
    curves = pd.read_csv(DATA / "aif_curves.csv")
    curves = curves[curves.sampling == "native"]
    lab = pd.read_csv(DATA / "qc" / "aif_qc_labels.csv").set_index(["run_id", "variant"])
    plotters = {"truncated": plot_truncated, "no_baseline": plot_no_baseline}

    data = {}
    for cat, (rid, var) in EXAMPLES.items():
        t, y, f = curve(curves, rid, var, th)
        row = lab.loc[(rid, var)]
        assert row.category == cat, (rid, var, row.category)
        pid = rid.split("_")[-4]
        label = f"{pid}, {PLANE[rid[-1]]} view, {NAME[var]} mask"
        data[cat] = (t, y, f, label)
        print(f"{cat}: {rid} {var}  n_base={f['n_base']}  post_min={f['post_min']:.3f}  "
              f"t_peak={f['t_peak']:.2f}  t_end={f['t_end']:.2f}  flags={row["flags"]}")

        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        plotters[cat](ax, t, y, f, th, label)
        fig.savefig(OUT / f"qc_{cat}.png", dpi=200, facecolor="white")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), constrained_layout=True, sharey=True)
    for ax, cat in zip(axes, EXAMPLES):
        t, y, f, label = data[cat]
        plotters[cat](ax, t, y, f, th, label)
    axes[1].set_ylabel("")
    fig.savefig(OUT / "qc_examples_both.png", dpi=200, facecolor="white")
    plt.close(fig)
    print("written to", OUT)


if __name__ == "__main__":
    main()
