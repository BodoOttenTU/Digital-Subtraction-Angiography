"""Simple curve figures for the method slides of SQ1–SQ4 in the final presentation.

    python scripts/presentation_sq_figures.py

Only AIF curves (no images), native sampling, baseline = median of the first two frames.
  sq1_method.png     one run: ICA-top and ICA-full AIF (PRAISE048, frontal; same run as segment_top_full.gif)
  sq2_method.png     one injection: frontal and lateral ICA-top AIF (PRAISE054)
  sq3_method.png     three patients, their four AIFs each, peak-normalised
  sq4_aif_own.png    minimal AIF thumbnails (frontal and lateral ICA-top, PRAISE054)
  sq4_aif_other.png
Written to results/presentation/sq/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, RESULTS  # noqa: E402

OUT = RESULTS / "presentation" / "sq"
RUN_A = "PRAISE_BATCH_2_PRAISE054_01_S11_A"
RUN_B = "PRAISE_BATCH_2_PRAISE054_01_S11_B"
RUN_SQ1 = "PRAISE_PRAISE048_01_S08_A"   # clean single-piece ICA-full mask
SQ3_PATIENTS = ["PRAISE063_01", "PRAISE054_01", "PRAISE058_01"]   # early, typical, late bolus
SCARLET, CYAN, NAVY, ORANGE, GREY = "#c81919", "#0092b5", "#101073", "#e86a00", "#8a8a9a"
PATIENT_COL = ["#0092b5", "#101073", "#c81919"]

plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})


def load():
    c = pd.read_csv(DATA / "aif_curves.csv")
    return c[c.sampling == "native"]


def curve(c, rid, var):
    d = c[(c.run_id == rid) & (c.variant == var)].sort_values("time_s")
    t, y = d.time_s.to_numpy(), d.aif.to_numpy()
    return t, y - np.median(y[:2])


def finish(ax, ylabel="AIF (a.u.)"):
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.set_yticks([])
    ax.set_xlim(0, 8)


def two_curves(c, specs, name, size=(5.2, 3.4)):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    peaks = []
    for rid, var, col, ls, lab in specs:
        t, y = curve(c, rid, var)
        ax.plot(t, y, color=col, lw=3, ls=ls, label=lab)
        peaks.append(y.max())
    finish(ax)
    ax.set_ylim(-0.05 * max(peaks), 1.2 * max(peaks))
    ax.legend(loc="upper right", fontsize=13)
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    return peaks


def thumb(c, rid, var, col, name):
    fig, ax = plt.subplots(figsize=(2.6, 2.1), constrained_layout=True)
    t, y = curve(c, rid, var)
    ax.plot(t, y, color=col, lw=2.5)
    finish(ax, "AIF")
    ax.set_ylim(-0.05, 1.0)   # shared scale so the lateral AIF looks higher
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)
    return y.max()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = load()

    p1 = two_curves(c, [(RUN_SQ1, "ica_top", SCARLET, "-", "ICA-top"),
                        (RUN_SQ1, "ica_full_grown", CYAN, "--", "ICA-full")], "sq1_method.png")
    p2 = two_curves(c, [(RUN_A, "ica_top", NAVY, "-", "frontal view"),
                        (RUN_B, "ica_top", ORANGE, "-", "lateral view")], "sq2_method.png")
    print(f"SQ1 example peak full/top = {p1[1] / p1[0]:.2f};  SQ2 example peak lateral/frontal = {p2[1] / p2[0]:.2f}")

    lab = pd.read_csv(DATA / "qc" / "aif_qc_labels.csv")
    lab = lab[lab.variant.isin(["ica_top", "ica_full_grown"]) & ~lab.is_duplicate & (lab.category == "usable")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for i, pid in enumerate(SQ3_PATIENTS):
        rows = lab[lab.patient_id == pid]
        for _, r in rows.iterrows():
            t, y = curve(c, r.run_id, r.variant)
            ax.plot(t, y / y.max(), color=PATIENT_COL[i], lw=2.2, alpha=0.85,
                    label=f"patient {i + 1}" if _ == rows.index[0] else None)
        print(f"SQ3 {pid}: {len(rows)} curves")
    finish(ax, "AIF (fraction of peak)")
    ax.set_ylim(-0.05, 1.25)
    ax.legend(loc="upper right", fontsize=13)
    fig.savefig(OUT / "sq3_method.png", dpi=300, facecolor="white")
    plt.close(fig)

    a = thumb(c, RUN_A, "ica_top", NAVY, "sq4_aif_own.png")
    b = thumb(c, RUN_B, "ica_top", ORANGE, "sq4_aif_other.png")
    print(f"SQ4 thumbnails peaks {a:.3f} / {b:.3f}")
    print("written to", OUT)


if __name__ == "__main__":
    main()
