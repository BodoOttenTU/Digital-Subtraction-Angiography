"""SQ3 slide figures (needs sq3_variance.py).

    python scripts/sq3_slide_figures.py
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paths import DATA, SQ3

VIEW_COL = {"A": "#1f3b5c", "B": "#eb6834"}
VIEW_NAME = {"A": "frontal (A)", "B": "lateral (B)"}
# only ica_top and ica_full_grown, as in the main model of sq3_variance.py
SEG_LS = {"ica_top": "-", "ica_full_grown": "--"}
SEG_NAME = {"ica_top": "supraclinoid ICA (top)", "ica_full_grown": "cervical-petrous ICA (full, grown)"}
COMP_COL = {"patient": "#2a78d6", "view": "#eb6834", "segment": "#2f9e44", "residual": "#b5b5b5"}
COMP_NAME = {"patient": "patient", "view": "projection view", "segment": "ICA segment",
             "residual": "other within-patient differences"}
# features shown on the slides, grouped; the full set is in table_sq3.tex
GROUPS = [("AIF timing", ["aif_ttp_s", "aif_mtt_bolus_s", "aif_fwhm_s"]),
          ("AIF amplitude", ["aif_peak", "aif_auc_first_pass"]),
          ("Perfusion parameters", ["CBF_median", "CBV_median", "MTT_median", "Tmax_median"])]
SHORT = {"CBF_median": "CBF", "CBV_median": "CBV", "MTT_median": "MTT", "Tmax_median": "Tmax"}
BANDS = [(0, 0.5, "poor"), (0.5, 0.75, "moderate"), (0.75, 0.9, "good"), (0.9, 1.0, "excellent")]

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def save(fig, out_dir, name, report_fig):
    fig.savefig(out_dir / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    if report_fig.is_dir():
        shutil.copy(out_dir / f"{name}.png", report_fig / f"{name}.png")


def pick_patients(feat):
    """Three patients with all four trusted curves: early / typical / late bolus, each with a typical within-spread."""
    tr = feat[feat.qc_trust == "trusted"]
    g = tr.groupby("patient_id").aif_ttp_s
    p = pd.DataFrame(dict(n=g.size(), mean=g.mean(), sd=g.std()))
    p = p[p.n == 2 * len(SEG_LS)]
    p = p[(p.sd - p.sd.median()).abs() <= p.sd.quantile(0.75) - p.sd.median()]
    p = p.sort_values("mean")
    return [p.index[int(q * (len(p) - 1))] for q in (0.1, 0.5, 0.9)]


def slide_question(feat, curves, out_dir, report_fig):
    pats = pick_patients(feat)
    tr = feat[(feat.qc_trust == "trusted") & feat.patient_id.isin(pats)]
    cv = curves[curves.sampling == "resampled_250ms"].set_index(["run_id", "variant"]).sort_index()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for k, (ax, pat) in enumerate(zip(axes, pats)):
        for _, r in tr[tr.patient_id == pat].iterrows():
            c = cv.loc[(r.run_id, r.variant)]
            ax.plot(c.time_s, c.aif, color=VIEW_COL[r.plane], ls=SEG_LS[r.variant], lw=2)
        ttp = tr[tr.patient_id == pat].aif_ttp_s
        ax.axvspan(ttp.min(), ttp.max(), color="#2a78d6", alpha=0.12, lw=0)
        ax.set_title(f"Patient {k + 1}", fontweight="bold")
        ax.set_xlim(0, 9)
        ax.set_xlabel("time (s)")
    axes[0].set_ylabel("AIF (contrast, a.u.)")
    handles = ([Line2D([], [], color=VIEW_COL[v], lw=2.5, label=VIEW_NAME[v]) for v in "AB"]
               + [Line2D([], [], color="k", ls=SEG_LS[s], lw=2, label=SEG_NAME[s]) for s in SEG_LS]
               + [Patch(color="#2a78d6", alpha=0.25, label="range of time-to-peak")])
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=10.5)
    fig.suptitle("One injection, four AIFs: are the differences between patients larger than within a patient?",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.03)
    save(fig, out_dir, "sq3_slide1_question", report_fig)
    return pats


def rows(var):
    v = var.set_index("feature")
    out = []
    for group, feats in GROUPS:
        for f in feats:
            r = v.loc[f]
            out.append(dict(group=group, label=SHORT.get(f, r.label), patient=r.patient, view=r.view,
                            segment=r.segment, residual=r.run + r.residual + r.interaction + r.shared,
                            icc=r.icc_agree, lo=r.icc_agree_lo, hi=r.icc_agree_hi, cv=r.cv_total))
    return pd.DataFrame(out)


def group_axis(ax, d):
    """y positions with a gap between groups, group names as bold labels."""
    y, ys, gy = 0, [], []
    for group, sub in d.groupby("group", sort=False):
        gy.append((group, y + 0.25))
        for _ in sub.index:
            y += 1
            ys.append(y)
        y += 0.9
    ys = np.array(ys, float)
    ax.set_yticks(ys, d.label)
    ax.set_ylim(y - 0.5, -0.6)
    for group, yy in gy:
        ax.text(-0.01, yy, group, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontweight="bold", fontsize=12, color="#333")
    ax.tick_params(axis="y", length=0)
    return ys


def slide_variance(d, n, out_dir, report_fig):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ys = group_axis(ax, d)
    left = np.zeros(len(d))
    for comp in ("patient", "view", "segment", "residual"):
        w = d[comp].to_numpy()
        ax.barh(ys, w, left=left, height=0.72, color=COMP_COL[comp], label=COMP_NAME[comp], edgecolor="white", lw=0.8)
        for yy, l, ww in zip(ys, left, w):
            if ww >= 0.07:
                ax.text(l + ww / 2, yy, f"{ww:.0%}", ha="center", va="center", fontsize=10.5,
                        color="white" if comp != "residual" else "#333", fontweight="bold")
        left += w
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_xlabel("share of total variance")
    ax.spines["left"].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False, fontsize=11)
    ax.set_title("Where does the variation come from?", pad=22)
    ax.text(0, 1.01, f"linear mixed model · {n[0]} AIFs from {n[1]} patients (2 views × 2 ICA segments each)",
            transform=ax.transAxes, fontsize=10.5, color="#555")
    save(fig, out_dir, "sq3_slide2_variance", report_fig)


def slide_icc(d, out_dir, report_fig):
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ys = group_axis(ax, d)
    shades = ["#f6d5d2", "#fbe9c9", "#dcefdc", "#c2e3c4"]
    for (a, b, name), s in zip(BANDS, shades):
        ax.axvspan(a, b, color=s, lw=0, zorder=0)
        ax.text((a + b) / 2, 1.01, name, ha="center", va="bottom", fontsize=10.5, color="#555",
                transform=ax.get_xaxis_transform())
    perf = (d.group == "Perfusion parameters").to_numpy()
    col = np.where(perf, "#c0392b", "#1f3b5c")
    ax.hlines(ys, d.lo, d.hi, color=col, lw=2.5, zorder=2)
    ax.scatter(d.icc, ys, color=col, s=70, zorder=3)
    for yy, r, c in zip(ys, d.itertuples(), col):
        ax.text(1.03, yy, f"±{r.cv:.0%}", ha="left", va="center", fontsize=11.5, color=c,
                fontweight="bold", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.text(1.03, 1.01, "spread within\none patient (CV)", ha="left", va="bottom", fontsize=10, color="#555",
            transform=ax.transAxes, clip_on=False)
    ax.set_xlim(0, 1)
    ax.set_xlabel("ICC of a single AIF (any view, any segment)   ·   95 % CI")
    ax.spines["left"].set_visible(False)
    ax.set_title("Reproducible AIF, unreliable perfusion maps", pad=26)
    save(fig, out_dir, "sq3_slide3_icc", report_fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    args = ap.parse_args()
    out_dir = SQ3 / "slides"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_fig = Path(__file__).resolve().parents[1] / "report" / "figures"
    feat = pd.read_csv(args.out / "features" / "features_aif.csv")
    feat = feat[feat.variant.isin(SEG_LS)]
    curves = pd.read_csv(args.out / "aif_curves.csv")
    var = pd.read_csv(SQ3 / "sq3_variance.csv")
    pats = slide_question(feat, curves, out_dir, report_fig)
    print("slide 1 patients:", pats)
    d = rows(var)
    tr = feat[feat.qc_trust == "trusted"]
    slide_variance(d, (len(tr), tr.patient_id.nunique()), out_dir, report_fig)
    slide_icc(d, out_dir, report_fig)
    print(d.round(2).to_string())


if __name__ == "__main__":
    main()
