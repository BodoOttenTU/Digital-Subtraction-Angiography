"""Plane A vs B: whole-image contrast and vessel overlap per injection.

    python scripts/plane_difference.py
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr, wilcoxon

from cohort_common import parse_source, rd
from paths import DATA, PLANE

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
FRAME_INTERVAL_MS = 250
RING_PX = 15            # ring width around the mask (1024 px image)
VESSEL_LEVEL = 120      # MINIP grey level (background 170) below which a pixel counts as vessel
# feature -> (label, comparison): "ratio" = B/A on a log scale, "diff" = B - A
FEATURES = {
    "kvp": ("Tube voltage (kVp)", "diff"),
    "dap": ("Dose-area product", "ratio"),
    "magnification": ("Magnification", "ratio"),
    "global_peak": ("Whole-image contrast, peak", "ratio"),
    "global_auc": ("Whole-image contrast, AUC", "ratio"),
    "aif_peak": ("AIF peak", "ratio"),
    "aif_auc": ("AIF AUC", "ratio"),
    "aif_rel_peak": ("AIF peak / whole-image peak", "ratio"),
    "mask_px": ("Mask size", "ratio"),
    "venous_frac": ("Venous fraction of the mask", "diff"),
    "ring_vessel_frac": ("Vessel fraction around the mask", "diff"),
    "sat_frac": ("Saturated mask fraction", "diff"),
}


def global_signal(frames, px_area_mm2):
    """Whole-image contrast signal over time (first frame subtracted), in grey-level fraction x mm^2."""
    s = (1 - frames / 170).sum(axis=(0, 1))
    s = (s - s[0]) * px_area_mm2
    t = np.arange(len(s)) * FRAME_INTERVAL_MS / 1000
    return float(s.max()), float(np.trapezoid(np.clip(s, 0, None), t))


def ring_vessel_frac(minip, mask):
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * RING_PX + 1, 2 * RING_PX + 1))
    ring = cv.dilate(mask.astype(np.uint8), k).astype(bool) & ~mask
    return float((minip[ring] < VESSEL_LEVEL).mean()) if ring.any() else np.nan


def measure(out, acq, feat):
    rows = []
    runs = sorted(feat.run_id.unique())
    for i, run_id in enumerate(runs, 1):
        a = acq.loc[run_id]
        print(f"[{i}/{len(runs)}] {run_id}", flush=True)
        ds = rd.read_dicom(parse_source(a.source))
        frames = rd.load_uniform_sequence(ds, FRAME_INTERVAL_MS, 1024)
        # pixel size at the isocentre after resizing to 1024: detector spacing x native size / 1024 / magnification
        native = max(int(a.rows), int(a.columns))
        px_mm = float(a.imager_pixel_spacing_mm) * native / 1024 / float(a.magnification)
        g_peak, g_auc = global_signal(frames, px_mm ** 2)
        d = np.load(out / "npz" / f"{run_id}.npz")
        for v in VARIANTS:
            rows.append(dict(run_id=run_id, variant=v, global_peak=g_peak, global_auc=g_auc, px_mm=px_mm,
                             ring_vessel_frac=ring_vessel_frac(d["minip"], d[f"mask_{v}"])))
    return pd.DataFrame(rows)


def holm(p):
    """Holm-Bonferroni adjusted p-values (statsmodels is not in the BEP environment)."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def make_pairs(runs):
    cols = list(FEATURES) + ["projection_overlap", "trust"]
    a = runs[runs.plane == "A"].set_index(["ab_pair_id", "variant"])[cols + ["run_id", "injection_site"]]
    b = runs[runs.plane == "B"].set_index(["ab_pair_id", "variant"])[cols + ["run_id"]]
    p = a.join(b, lsuffix="_A", rsuffix="_B", how="inner").reset_index()
    for f, (_, kind) in FEATURES.items():
        p[f"{f}_cmp"] = (np.log(p[f"{f}_B"] / p[f"{f}_A"]) if kind == "ratio" else p[f"{f}_B"] - p[f"{f}_A"])
    return p


def tests(pairs):
    rows = []
    for v in VARIANTS:
        p = pairs[pairs.variant == v]
        for f, (label, kind) in FEATURES.items():
            x = p[f"{f}_cmp"].replace([np.inf, -np.inf], np.nan).dropna()
            stat = wilcoxon(x) if (x != 0).sum() > 5 else None
            q = x.quantile([0.25, 0.5, 0.75])
            fmt = (lambda y: f"{np.exp(y):.2f}") if kind == "ratio" else (lambda y: f"{y:+.3g}")
            rows.append(dict(variant=v, feature=f, label=label, kind="B/A" if kind == "ratio" else "B-A", n=len(x),
                             median_A=p[f"{f}_A"].median(), median_B=p[f"{f}_B"].median(),
                             effect=fmt(q[0.5]), effect_iqr=f"{fmt(q[0.25])} .. {fmt(q[0.75])}",
                             p=stat.pvalue if stat else np.nan))
    t = pd.DataFrame(rows)
    for v in VARIANTS:
        m = (t.variant == v) & t.p.notna()
        t.loc[m, "p_holm"] = holm(t.loc[m, "p"])
    return t


def correlations(pairs):
    """Which factor goes with the AIF difference between planes."""
    rows = []
    for v in VARIANTS:
        p = pairs[pairs.variant == v]
        y = p["aif_peak_cmp"]
        for f in ["global_peak", "kvp", "dap", "venous_frac", "ring_vessel_frac", "mask_px", "magnification"]:
            ok = y.notna() & p[f"{f}_cmp"].notna() & np.isfinite(p[f"{f}_cmp"])
            r = spearmanr(y[ok], p.loc[ok, f"{f}_cmp"])
            rows.append(dict(variant=v, outcome="log AIF peak B/A", factor=f"{f} ({FEATURES[f][1]})", n=int(ok.sum()),
                             rho=r.statistic, p=r.pvalue))
        # vessel-specific part (global scaling removed) against overlap
        y2 = p["aif_rel_peak_cmp"]
        for f in ["venous_frac", "ring_vessel_frac", "kvp"]:
            ok = y2.notna() & p[f"{f}_cmp"].notna() & np.isfinite(y2)
            r = spearmanr(y2[ok], p.loc[ok, f"{f}_cmp"])
            rows.append(dict(variant=v, outcome="log (AIF/global) B/A", factor=f"{f} ({FEATURES[f][1]})",
                             n=int(ok.sum()), rho=r.statistic, p=r.pvalue))
    return pd.DataFrame(rows)


def variance_share(pairs):
    """How much of the spread of log AIF peak B/A remains once the global scaling is removed."""
    rows = []
    for v in VARIANTS:
        p = pairs[pairs.variant == v]
        y, rel = p.aif_peak_cmp, p.aif_rel_peak_cmp
        rows.append(dict(variant=v, n=len(p), sd_log_aif=y.std(), sd_log_global=p.global_peak_cmp.std(),
                         sd_log_aif_rel=rel.std(), var_explained_by_global=1 - rel.var() / y.var()))
    return pd.DataFrame(rows)


PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}


def plot(pairs, runs, path):
    p = pairs[pairs.variant == "ica_top"]
    show = ["kvp", "global_peak", "aif_peak", "aif_rel_peak", "venous_frac", "ring_vessel_frac"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for ax, f in zip(axes.flat, show):
        a, b = p[f"{f}_A"], p[f"{f}_B"]
        for x0, x1 in zip(a, b):
            ax.plot([0, 1], [x0, x1], color="0.75", lw=0.6)
        ax.scatter(np.zeros(len(a)), a, color=PLANE_COL["A"], s=12, zorder=3)
        ax.scatter(np.ones(len(b)), b, color=PLANE_COL["B"], s=12, zorder=3)
        ax.set_xticks([0, 1], ["A (frontal)", "B (lateral)"])
        ax.set_title(FEATURES[f][0] + " (ica_top)", fontsize=10)
        if FEATURES[f][1] == "ratio":
            ax.set_yscale("log")
    ax = axes.flat[6]
    ax.scatter(p.global_peak_cmp, p.aif_peak_cmp, s=14, color="k")
    lim = [np.log(0.5), np.log(3)]
    ax.plot(lim, lim, color="0.6", ls="--", lw=0.8)
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    ax.set(xlim=lim, ylim=lim, xticks=np.log(ticks), yticks=np.log(ticks), xticklabels=ticks, yticklabels=ticks,
           xlabel="whole-image contrast peak B/A", ylabel="AIF peak B/A",
           title="AIF vs global contrast difference (ica_top)")
    ax = axes.flat[7]
    ax.scatter(p.ring_vessel_frac_cmp, p.aif_peak_cmp, s=14, color="k")
    ax.set(yticks=np.log(ticks), yticklabels=ticks, xlabel="vessel fraction around the mask, B - A",
           ylabel="AIF peak B/A", title="AIF vs overlap around the mask (ica_top)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fmt_p(x):
    return "" if pd.isna(x) else (f"{x:.1e}" if x < 0.001 else f"{x:.3f}")


def tex_table(t, n_pairs, path):
    tt = t[t.variant == "ica_top"]
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Plane B (lateral) versus plane A (frontal) of the same injection, ICA-top AIF "
             f"({n_pairs} pairs). Ratio B/A or difference B$-$A: median [IQR]; paired Wilcoxon, Holm-corrected.}}",
             "\\label{tab:plane_difference}", "\\begin{tabular}{llll}", "\\hline",
             "Feature & B vs A & median [IQR] & $p_\\mathrm{Holm}$ \\\\", "\\hline"]
    for r in tt.itertuples():
        lines.append(f"{r.label} & {r.kind.replace('-', '$-$')} & {r.effect} [{r.effect_iqr.replace('..', '--')}] & "
                     f"{fmt_p(r.p_holm)} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    args = p.parse_args()
    out = args.out
    out_dir = PLANE
    out_dir.mkdir(parents=True, exist_ok=True)

    acq = pd.read_csv(out / "acquisition_table.csv").set_index("run_id")
    feat = pd.read_csv(out / "features.csv")
    feat = feat[~feat.is_duplicate & feat.plane.isin(["A", "B"])]
    tax = pd.read_csv(out / "qc" / "taxonomy" / "taxonomy_labels.csv")[
        ["run_id", "variant", "taxonomy", "trust", "venous_frac"]]
    cache = out_dir / "plane_runs_measured.csv"
    if cache.exists():
        meas = pd.read_csv(cache)
    else:
        meas = measure(out, acq, feat)
        meas.to_csv(cache, index=False)

    runs = (feat[["run_id", "variant", "plane", "ab_pair_id", "injection_site", "aif_peak", "aif_auc", "mask_px",
                  "sat_frac", "kvp", "magnification"]]
            .merge(meas, on=["run_id", "variant"]).merge(tax, on=["run_id", "variant"])
            .merge(acq[["dap", "field_of_view_mm"]], left_on="run_id", right_index=True))
    runs["aif_rel_peak"] = runs.aif_peak / runs.global_peak * 1e4
    runs["projection_overlap"] = runs.taxonomy == "projection_overlap"
    runs.to_csv(out_dir / "plane_runs.csv", index=False)

    pairs = make_pairs(runs)
    pairs.to_csv(out_dir / "plane_pairs.csv", index=False)
    t = tests(pairs)
    t.to_csv(out_dir / "plane_tests.csv", index=False)
    corr = correlations(pairs)
    corr.to_csv(out_dir / "plane_correlations.csv", index=False)
    share = variance_share(pairs)
    plot(pairs, runs, out_dir / "plane_difference.png")
    tex_table(t, int((pairs.variant == "ica_top").sum()), out_dir / "table_plane_difference.tex")
    print(t[t.variant == "ica_top"][["label", "effect", "effect_iqr", "p_holm"]].to_string())
    print(share.to_string())


if __name__ == "__main__":
    main()
