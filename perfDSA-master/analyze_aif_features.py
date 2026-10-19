"""
AIF features of the usable AIF curves, and paired comparisons ICA full vs ICA top and plane A vs plane B.

Input : results folders with aif_curves.csv, acquisition_table.csv and aif_qc/aif_labels.csv
        (run classify_aif.py first). Default: results/praise and results/praise_batch_2, pooled.

Features (per curve, after subtracting the pre-bolus baseline = median of the first N_BASE frames):
    peak            maximum of the baseline-corrected curve                         [a.u.]
    ttp_s           time to peak, from the start of the acquisition                 [s]
    arrival_s       first time the curve exceeds ARRIVAL_FRAC of the peak (interp.) [s]
    rise_s          ttp_s - arrival_s; independent of when the acquisition started  [s]
    fwhm_s          full width at half maximum (interpolated crossings)            [s]
    auc             area under the first pass: from arrival until the curve falls
                    back below ARRIVAL_FRAC of the peak (or the end)               [a.u. s]
    auc_total       area under the whole baseline-corrected curve                   [a.u. s]
    width_s         auc / peak: effective bolus width, independent of the scale    [s]
  Note: curves are on a uniform 0.25 s grid, resampled from native 250-500 ms frames, so timing
  features have a resolution of roughly one native frame.

Output (in -o, default results/aif_analysis):
    aif_features.csv          every curve: run, patient, injection, plane, variant, category, features,
                              acquisition covariates (angles, frame interval, injection site)
    success_rate.csv          % usable AIF per dataset x variant (and per plane)
    paired_full_vs_top.csv    Wilcoxon signed-rank ICA full vs ICA top (runs where both are usable)
    paired_A_vs_B.csv         Wilcoxon signed-rank plane A vs plane B of the same injection, per variant
    paired_full_vs_top.png    paired plots of the features
    paired_A_vs_B.png

Example:
    python analyze_aif_features.py
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
N_BASE = 3
ARRIVAL_FRAC = 0.1
FEATURES = ["peak", "ttp_s", "arrival_s", "rise_s", "fwhm_s", "auc", "auc_total", "width_s"]
VARIANTS = ["ica_full", "ica_top"]
ACQ_COLS = ["patient_id", "series_number", "series_description", "plane", "primary_angle_deg",
            "secondary_angle_deg", "frame_interval_ms_median", "duration_s", "kvp", "magnification"]


def crossing(t, y, i0, i1, level):
    """Interpolated time where y crosses `level` between samples i0 and i1."""
    if y[i1] == y[i0]:
        return t[i0]
    return t[i0] + (level - y[i0]) * (t[i1] - t[i0]) / (y[i1] - y[i0])


def features(t, aif):
    y = aif - np.median(aif[:N_BASE])
    ipk = int(np.argmax(y))
    peak = y[ipk]

    lvl = ARRIVAL_FRAC * peak
    up = np.nonzero(y[:ipk + 1] < lvl)[0]
    i_arr = up[-1] if len(up) else 0
    arrival = crossing(t, y, i_arr, i_arr + 1, lvl) if len(up) and i_arr < ipk else t[0]
    down = np.nonzero(y[ipk:] < lvl)[0]
    i_end = ipk + down[0] if len(down) else len(y) - 1

    half = peak / 2
    lo = np.nonzero(y[:ipk + 1] < half)[0]
    hi = np.nonzero(y[ipk:] < half)[0]
    t_lo = crossing(t, y, lo[-1], lo[-1] + 1, half) if len(lo) else np.nan
    t_hi = crossing(t, y, ipk + hi[0] - 1, ipk + hi[0], half) if len(hi) else np.nan

    auc = float(np.trapezoid(y[i_arr:i_end + 1], t[i_arr:i_end + 1]))
    return {"peak": peak, "ttp_s": t[ipk], "arrival_s": arrival, "rise_s": t[ipk] - arrival,
            "fwhm_s": t_hi - t_lo, "auc": auc, "auc_total": float(np.trapezoid(y, t)), "width_s": auc / peak}


def injection_site(desc):
    d = str(desc).upper()
    return "CCA" if re.search(r"\bCCA\b", d) else "ICA" if re.search(r"\bICA\b", d) else "unknown"


def load(folder):
    folder = Path(folder)
    curves = pd.read_csv(folder / "aif_curves.csv")
    labels = pd.read_csv(folder / "aif_qc" / "aif_labels.csv", keep_default_na=False)
    acq = pd.read_csv(folder / "acquisition_table.csv")[["run_id", "dataset"] + ACQ_COLS]

    rows = []
    for (rid, var), d in curves.groupby(["run_id", "variant"]):
        d = d.sort_values("time_s")
        rows.append({"run_id": rid, "variant": var, **features(d.time_s.to_numpy(), d.aif.to_numpy())})
    df = pd.DataFrame(rows).merge(labels[["run_id", "variant", "category", "flags"]], on=["run_id", "variant"])
    df = df.merge(acq, on="run_id", how="left")

    # Re-exported duplicates (run_id ending in _2 with the same curve as the original) are dropped.
    dup = curves.groupby("run_id").aif.apply(tuple)
    same = [r for r in dup.index if re.search(r"_\d+$", r) and re.sub(r"_\d+$", "", r) in dup.index
            and dup[r] == dup[re.sub(r"_\d+$", "", r)]]
    return df[~df.run_id.isin(same)], same


def describe(x):
    q1, med, q3 = np.nanpercentile(x, [25, 50, 75])
    return f"{med:.3g} [{q1:.3g}-{q3:.3g}]"


def paired_tests(wide, a, b, label):
    """wide: one row per pair with columns (feature, a) and (feature, b)."""
    rows = []
    for f in FEATURES:
        x, y = wide[(f, a)].to_numpy(float), wide[(f, b)].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        diff = y - x
        nz = diff[diff != 0]
        if len(nz) >= 5:
            res = wilcoxon(x, y, zero_method="wilcox")
            ranks = pd.Series(np.abs(nz)).rank().to_numpy()
            r_rb = (ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum()  # rank-biserial, + means b > a
            p = res.pvalue
        else:
            r_rb = p = np.nan
        rel = 100 * diff / np.where(x != 0, np.abs(x), np.nan)
        rows.append({"comparison": label, "feature": f, "n_pairs": int(ok.sum()), a: describe(x), b: describe(y),
                     f"median_diff ({b}-{a})": np.median(diff), "median_rel_diff_%": np.nanmedian(rel),
                     "rank_biserial_r": r_rb, "p_wilcoxon": p})
    out = pd.DataFrame(rows)
    valid = out.p_wilcoxon.notna()
    out.loc[valid, "p_holm"] = multipletests(out.loc[valid, "p_wilcoxon"], method="holm")[1]
    return out


def paired_plot(wide, a, b, title, path):
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for ax, f in zip(axs.flat, FEATURES):
        x, y = wide[(f, a)].to_numpy(float), wide[(f, b)].to_numpy(float)
        for xi, yi in zip(x, y):
            ax.plot([0, 1], [xi, yi], color="0.6", lw=0.7, alpha=0.6)
        ax.scatter(np.zeros_like(x), x, s=10, color="tab:blue", zorder=3)
        ax.scatter(np.ones_like(y), y, s=10, color="tab:red", zorder=3)
        ax.boxplot([x[np.isfinite(x)], y[np.isfinite(y)]], positions=[-0.25, 1.25], widths=0.2,
                   showfliers=False)
        ax.set(xticks=[0, 1], xticklabels=[a, b], xlim=(-0.5, 1.5), title=f)
    fig.suptitle(f"{title} (n = {len(wide)} pairs)")
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="*", default=[str(HERE / "results" / "praise"),
                                                  str(HERE / "results" / "praise_batch_2")])
    p.add_argument("-o", dest="out_dir", default=str(HERE / "results" / "aif_analysis"))
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    parts, dropped = zip(*(load(f) for f in args.folders))
    df = pd.concat(parts, ignore_index=True)
    df["injection_id"] = df.patient_id + "_S" + df.series_number.astype(int).astype(str).str.zfill(2)
    df["injection_site"] = df.series_description.map(injection_site)
    df["usable"] = df.category == "usable"
    df = df[["dataset", "patient_id", "injection_id", "run_id", "plane", "variant", "category", "usable", "flags"]
            + FEATURES + [c for c in ACQ_COLS if c not in ("patient_id", "plane")] + ["injection_site"]]
    df.round(4).to_csv(out / "aif_features.csv", index=False)

    # Success rate
    rate = []
    for keys, g in [(("all", "all"), df)] + [((d, "all"), g) for d, g in df.groupby("dataset")] + \
                   [(("all", pl), g) for pl, g in df.groupby("plane")]:
        for var in VARIANTS:
            v = g[g.variant == var]
            rate.append({"dataset": keys[0], "plane": keys[1], "variant": var, "n_runs": len(v),
                         "n_usable": int(v.usable.sum()), "pct_usable": round(100 * v.usable.mean(), 1)})
    rate = pd.DataFrame(rate)
    rate.to_csv(out / "success_rate.csv", index=False)

    use = df[df.usable]

    # ICA full vs ICA top: same run, both usable
    w = use.pivot(index="run_id", columns="variant", values=FEATURES).dropna(subset=[(FEATURES[0], v) for v in VARIANTS])
    planes = df.drop_duplicates("run_id").set_index("run_id").plane
    fvt = pd.concat([paired_tests(w, "ica_full", "ica_top", "all planes")] +
                    [paired_tests(w[planes.reindex(w.index) == pl], "ica_full", "ica_top", f"plane {pl}")
                     for pl in ["A", "B"]])
    fvt.round(4).to_csv(out / "paired_full_vs_top.csv", index=False)
    paired_plot(w, "ica_full", "ica_top", "ICA full vs ICA top", out / "paired_full_vs_top.png")

    # Plane A vs plane B: same injection, same variant, both usable
    avb, wides = [], {}
    for var in VARIANTS:
        u = use[use.variant == var]
        wa = u.pivot_table(index="injection_id", columns="plane", values=FEATURES).dropna(
            subset=[(FEATURES[0], "A"), (FEATURES[0], "B")])
        wides[var] = wa
        avb.append(paired_tests(wa, "A", "B", var))
    avb = pd.concat(avb)
    avb.round(4).to_csv(out / "paired_A_vs_B.csv", index=False)
    paired_plot(wides["ica_full"], "A", "B", "Plane A (frontal) vs plane B (lateral), ICA full",
                out / "paired_A_vs_B_ica_full.png")
    paired_plot(wides["ica_top"], "A", "B", "Plane A (frontal) vs plane B (lateral), ICA top",
                out / "paired_A_vs_B_ica_top.png")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("dropped duplicate runs:", [r for d in dropped for r in d])
    print(f"\n{df.run_id.nunique()} runs, {df.patient_id.nunique()} patients, {df.injection_id.nunique()} injections")
    print("\nSUCCESS RATE\n", rate.to_string(index=False))
    print("\nICA FULL vs ICA TOP\n", fvt.round(3).to_string(index=False))
    print("\nPLANE A vs B\n", avb.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
