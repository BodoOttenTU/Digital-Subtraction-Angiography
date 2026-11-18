"""SQ3 report figure: where the variance of each AIF feature comes from (needs sq3_variance.py).

    python scripts/sq3_report_figure.py
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paths import SQ3

# a selection of features, sorted by patient share; the full set is in table_sq3_components.tex
FEATURES = ["aif_fwhm_s", "aif_ttp_s", "aif_rise_s", "aif_arrival_s", "aif_auc_first_pass", "aif_tail_frac", "aif_peak"]
# full figure: every feature, AIF shape and perfusion parameters as two groups (sorted by patient share)
GROUPS = {"AIF shape": ["aif_arrival_s", "aif_ttp_s", "aif_mtt_bolus_s", "aif_peak", "aif_auc_first_pass", "aif_fwhm_s",
                        "aif_rise_s", "aif_auc_norm_s", "aif_sd_s", "aif_skewness", "aif_tail_frac"],
          "Perfusion (median)": ["CBV_median", "CBF_median", "MTT_median", "Tmax_median"]}
# name, fill, label colour
COMPONENTS = {"patient": ("Patient", "#2a78d6", "white"),
              "view": ("View", "#eb6834", "white"),
              "segment": ("Segment", "#1baf7a", "white"),
              "run": ("Between runs", "#eda100", "#0b0b0b"),
              "residual": ("Residual", "#d5d4cd", "#0b0b0b")}
INK, MUTED = "#0b0b0b", "#52514e"
MIN_LABEL = 0.10

plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.spines.left": False,
                     "axes.edgecolor": "#898781", "xtick.color": MUTED, "pdf.fonttype": 42})


def draw(v, rows, stem, height, headings=()):
    """rows: feature keys in plot order, None for an empty gap row; headings: (row, text) above a group."""
    fig, ax = plt.subplots(figsize=(3.5, height))
    keep = [i for i, f in enumerate(rows) if f is not None]
    feats = [rows[i] for i in keep]
    y = np.array(keep, dtype=float)
    left = np.zeros(len(feats))
    for comp, (name, fill, text) in COMPONENTS.items():
        w = v.loc[feats, comp].to_numpy()
        ax.barh(y, w, left=left, height=0.78, color=fill, edgecolor="white", linewidth=0.9, label=name)
        for yy, l, ww in zip(y, left, w):
            if ww >= MIN_LABEL:
                ax.text(l + ww / 2, yy, f"{ww * 100:.0f}", ha="center", va="center", fontsize=7.5, color=text)
        left += w
    labels = v.loc[feats, "label"]
    if headings:  # the group heading already says "(median)"
        labels = labels.str.replace(" (median)", "", regex=False)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, labelcolor=INK)
    for row, text in headings:
        ax.text(0, row, text, ha="left", va="center", fontsize=7.5, fontweight="bold", color=MUTED)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.01, 0.25))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("Share of total variance", color=MUTED)
    ax.legend(loc="lower center", bbox_to_anchor=(0.36, 1.0), ncol=3, frameon=False, handlelength=1.1,
              handleheight=1.1, columnspacing=1.2, handletextpad=0.45, fontsize=7.5)

    for ext in ("png", "pdf"):
        fig.savefig(SQ3 / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    report_fig = Path(__file__).resolve().parents[1] / "report" / "figures"
    if report_fig.is_dir():
        shutil.copy(SQ3 / f"{stem}.pdf", report_fig / f"{stem}.pdf")
    print(f"{stem}: largest deviation of the stacked bars from 100 %:", round(float(np.abs(left - 1).max()), 4))


def main():
    v = pd.read_csv(SQ3 / "sq3_variance.csv").set_index("feature")
    # the interaction belongs to the segment here: the effect of the segment differs per view
    v["segment"] = v.segment + v.interaction + v.shared
    draw(v, FEATURES, "fig_sq3_variance", 2.35)

    rows, headings = [], []
    for title, feats in GROUPS.items():
        headings.append((len(rows), title))
        rows.append(None)
        rows += sorted(feats, key=lambda f: -v.loc[f, "patient"])
    draw(v, rows, "fig_sq3_variance_full", 4.6, headings)


if __name__ == "__main__":
    main()
