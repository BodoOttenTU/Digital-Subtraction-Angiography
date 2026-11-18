"""SQ2: plane A (frontal) vs plane B (lateral) of the same injection.

    python scripts/sq2_plane.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sq1_full_vs_top import FEATURES, fmt_effect, fmt_p, holm  # noqa: E402
from paths import DATA, PLANE, SQ2

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
ALPHA = 0.05
SHAPE_ONLY = ["aif_fwhm_s", "aif_rise_s", "aif_auc_norm_s", "aif_sd_s", "aif_skewness", "aif_tail_frac"]
CURVE_WINDOW = (-1.0, 8.0)  # s after arrival
PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}
VAR_COL = {"ica_top": "#2a78d6", "ica_full_model": "#7a7a7a", "ica_full_grown": "#2f9e44"}


def make_pairs(feat, extra):
    """One row per injection x variant, both planes trusted; d_<f> = B - A (or log B/A)."""
    f = feat.merge(extra, on=["run_id", "variant"], how="left")
    f = f[f.qc_trust == "trusted"]
    rows = []
    cols = list(FEATURES) + ["sat_frac", "mask_px_geo", "venous_frac", "ring_vessel_frac", "global_peak", "snr"]
    for (pair, variant), g in f.groupby(["ab_pair_id", "variant"]):
        if set(g.plane) != {"A", "B"}:
            continue
        a, b = g[g.plane == "A"].iloc[0], g[g.plane == "B"].iloc[0]
        row = dict(ab_pair_id=pair, variant=variant, patient_id=a.patient_id, injection_site=a.injection_site,
                   run_A=a.run_id, run_B=b.run_id, kvp_A=a.kvp, kvp_B=b.kvp, r_ab=a.r_ab)
        for c in cols:
            row[f"{c}_A"], row[f"{c}_B"] = a[c], b[c]
        for c, (_, scale) in FEATURES.items():
            row[f"d_{c}"] = np.log(b[c] / a[c]) if scale == "log" else b[c] - a[c]
        row["d_rel_peak"] = np.log((b.aif_peak / b.global_peak) / (a.aif_peak / a.global_peak))
        row["d_global"] = np.log(b.global_peak / a.global_peak)
        row["d_kvp"] = b.kvp - a.kvp
        row["d_px"] = np.log(b.mask_px_geo / a.mask_px_geo)
        row["d_ring"] = b.ring_vessel_frac - a.ring_vessel_frac
        row["d_sat"] = b.sat_frac - a.sat_frac
        row["any_sat"] = bool(a.sat_frac > 0 or b.sat_frac > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def tests(pairs, label=""):
    rows = []
    for v in VARIANTS:
        p = pairs[pairs.variant == v]
        for c, (name, scale) in FEATURES.items():
            d = p[f"d_{c}"].replace([np.inf, -np.inf], np.nan).dropna()
            if len(d) < 8:
                continue
            sw = stats.shapiro(d) if d.nunique() > 2 else None
            t = stats.ttest_1samp(d, 0.0)
            w = stats.wilcoxon(d) if (d != 0).sum() > 5 else None
            normal = bool(sw and sw.pvalue >= ALPHA)
            sd = d.std(ddof=1)
            rows.append(dict(subset=label, variant=v, feature=c, label=name, scale=scale, n=len(d),
                             shapiro_p=sw.pvalue if sw else np.nan, normal=normal,
                             test="paired t" if normal else "Wilcoxon", p_t=t.pvalue,
                             p_wilcoxon=w.pvalue if w else np.nan,
                             p=t.pvalue if normal else (w.pvalue if w else np.nan),
                             mean=d.mean(), sd=sd, median=d.median(), q1=d.quantile(0.25), q3=d.quantile(0.75),
                             ba_bias=d.mean(), ba_loa_low=d.mean() - 1.96 * sd, ba_loa_high=d.mean() + 1.96 * sd,
                             np_loa_low=d.quantile(0.025), np_loa_high=d.quantile(0.975),
                             effect=fmt_effect(d.mean() if normal else d.median(), scale)))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for v, idx in t.groupby("variant").groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
    t["significant"] = t.p_holm < ALPHA
    return t


def mechanism(pairs):
    p = pairs[pairs.variant == "ica_top"]
    rows = []
    for outcome in ["d_aif_peak", "d_rel_peak"]:
        for factor, name in [("d_global", "log whole-image contrast B/A (exposure)"), ("d_kvp", "kVp B - A (exposure)"),
                             ("d_px", "log mask px B/A (projected length / path length)"),
                             ("d_ring", "ring vessel fraction B - A (overlap)"),
                             ("d_sat", "saturated fraction B - A (saturation)")]:
            ok = p[outcome].notna() & p[factor].notna()
            s = stats.spearmanr(p.loc[ok, outcome], p.loc[ok, factor])
            rows.append(dict(outcome="log AIF peak B/A" if outcome == "d_aif_peak" else "log (AIF / whole-image) B/A",
                             factor=name, n=int(ok.sum()), rho=s.statistic, p=s.pvalue))
    return pd.DataFrame(rows)


def saturation_check(pairs):
    p = pairs[pairs.variant == "ica_top"]
    rows = []
    for name, g in (("no saturated pixels", p[~p.any_sat]), ("saturated pixels in A or B", p[p.any_sat])):
        rows.append(dict(group=name, n=len(g), sat_in_A=int((g.sat_frac_A > 0).sum()), sat_in_B=int((g.sat_frac_B > 0).sum()),
                         peak_B_over_A=f"{np.exp(g.d_aif_peak.median()):.2f}",
                         rel_B_over_A=f"{np.exp(g.d_rel_peak.median()):.2f}"))
    a, b = p[~p.any_sat].d_aif_peak, p[p.any_sat].d_aif_peak
    mw = stats.mannwhitneyu(a, b) if len(b) >= 3 else None
    return pd.DataFrame(rows), (mw.pvalue if mw else np.nan)


def curve_rmse(pairs, curves, feat):
    """Per ica_top pair: RMSE between the A and B curve, arrival-aligned, in units of the mean peak."""
    f = feat.set_index(["run_id", "variant"])
    c = curves[(curves.variant == "ica_top") & curves.t_from_arrival_s.between(*CURVE_WINDOW)]
    c = c.set_index(["run_id", "t_from_arrival_s"])
    rows = []
    for r in pairs[pairs.variant == "ica_top"].itertuples():
        try:
            ca, cb = c.loc[r.run_A], c.loc[r.run_B]
        except KeyError:
            continue
        j = ca.join(cb, lsuffix="_A", rsuffix="_B", how="inner").dropna()
        pa, pb = f.loc[(r.run_A, "ica_top")], f.loc[(r.run_B, "ica_top")]
        ya, yb = j.c_peak_norm_A * pa.aif_peak, j.c_peak_norm_B * pb.aif_peak  # baseline-corrected curves
        m_peak = (pa.aif_peak + pb.aif_peak) / 2
        m_auc = (pa.aif_auc + pb.aif_auc) / 2
        rows.append(dict(ab_pair_id=r.ab_pair_id, n_points=len(j),
                         rmse_raw=float(np.sqrt(np.mean((ya - yb) ** 2)) / m_peak),
                         rmse_peak_norm=float(np.sqrt(np.mean((j.c_peak_norm_A - j.c_peak_norm_B) ** 2))),
                         rmse_auc_norm=float(np.sqrt(np.mean((j.c_auc_norm_A - j.c_auc_norm_B) ** 2)) * m_auc / m_peak)))
    return pd.DataFrame(rows)


def plot_forest(t, path):
    feats = list(FEATURES)
    y = np.arange(len(feats))[::-1]
    fig, ax = plt.subplots(figsize=(8, 7.5))
    offs = {"ica_top": 0.2, "ica_full_model": 0.0, "ica_full_grown": -0.2}
    for v in VARIANTS:
        r = t[t.variant == v].set_index("feature").reindex(feats)
        eff, lo, hi = r["median"] / r["sd"], r.q1 / r["sd"], r.q3 / r["sd"]
        yy = y + offs[v]
        ax.errorbar(eff, yy, xerr=[eff - lo, hi - eff], fmt="o", ms=4, lw=1, color=VAR_COL[v], label=v)
        sig = r.significant.fillna(False).astype(bool).to_numpy()
        ax.scatter(eff[sig], yy[sig], s=60, facecolors="none", edgecolors="k", lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, [FEATURES[f][0] for f in feats])
    ax.set_xlabel("median (B - A) / SD of the differences (IQR)")
    ax.set_title("Plane B (lateral) vs plane A (frontal), same injection")
    ax.legend(fontsize=8, title="ring = p_Holm < 0.05", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_ba(pairs, t, variant, path):
    feats = list(FEATURES)
    ncol = 5
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow), squeeze=False)
    p = pairs[pairs.variant == variant]
    for ax, c in zip(axes.flat, feats):
        name, scale = FEATURES[c]
        a, b = p[f"{c}_A"], p[f"{c}_B"]
        mean = np.sqrt(a * b) if scale == "log" else (a + b) / 2
        ax.scatter(mean, p[f"d_{c}"], s=9, color="k", alpha=0.7)
        r = t[(t.variant == variant) & (t.feature == c)]
        if len(r):
            r = r.iloc[0]
            ax.axhline(r.ba_bias, color="C3", lw=1)
            for val in (r.ba_loa_low, r.ba_loa_high):
                ax.axhline(val, color="C3", lw=0.8, ls="--")
        ax.axhline(0, color="0.5", lw=0.5)
        if scale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{np.expm1(v_):+.0%}"))
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    fig.suptitle(f"Bland-Altman, plane B - plane A, {variant} (red: bias, dashed: 95 % limits of agreement)")
    fig.supxlabel("mean of A and B (geometric mean for ratio features)")
    fig.supylabel("B - A")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_mechanism(pairs, path):
    p = pairs[pairs.variant == "ica_top"]
    ticks = [0.5, 0.75, 1, 1.5, 2, 3]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    col = np.where(p.any_sat, "C3", "k")
    spec = [("d_global", "whole-image contrast B/A", True), ("d_px", "mask size B/A (projected length)", True),
            ("d_ring", "vessel fraction around mask, B - A", False), ("d_kvp", "tube voltage B - A (kV)", False)]
    for ax, (x, xl, logx) in zip(axes, spec):
        ax.scatter(p[x], p.d_aif_peak, s=16, c=col)
        if x == "d_global":
            lim = [np.log(0.5), np.log(3)]
            ax.plot(lim, lim, color="0.6", ls="--", lw=0.8, label="AIF ratio = image ratio")
            ax.legend(fontsize=7)
        if logx:
            ax.set_xticks(np.log(ticks), ticks)
        ax.set_yticks(np.log(ticks), ticks)
        ax.axhline(0, color="0.6", lw=0.5)
        ax.set_xlabel(xl)
        ax.set_ylabel("AIF peak B/A")
    axes[0].set_title("exposure", fontsize=10)
    axes[1].set_title("path length (foreshortening)", fontsize=10)
    axes[2].set_title("overlap", fontsize=10)
    axes[3].set_title("tube voltage", fontsize=10)
    fig.suptitle("Why is the ica_top AIF peak higher in plane B? (red: saturated mask pixels in A or B)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_shape_curves(pairs, curves, feat, path):
    """Median arrival-aligned ica_top curve per plane: raw (AIF units) and peak-normalised."""
    c = curves[(curves.variant == "ica_top") & curves.t_from_arrival_s.between(*CURVE_WINDOW)]
    p = pairs[pairs.variant == "ica_top"]
    peak = feat[feat.variant == "ica_top"].set_index("run_id").aif_peak
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for plane in ("A", "B"):
        runs = p[f"run_{plane}"]
        sub = c[c.run_id.isin(runs)].copy()
        sub["raw"] = sub.c_peak_norm * sub.run_id.map(peak)
        for ax, col in zip(axes, ("raw", "c_peak_norm")):
            g = sub.groupby("t_from_arrival_s")[col]
            ax.plot(g.median().index, g.median(), color=PLANE_COL[plane], label=f"plane {plane}")
            ax.fill_between(g.median().index, g.quantile(0.25), g.quantile(0.75), color=PLANE_COL[plane], alpha=0.2)
    axes[0].set(title="baseline-corrected AIF (median, IQR)", xlabel="s after arrival", ylabel="AIF")
    axes[1].set(title="peak-normalised AIF (median, IQR)", xlabel="s after arrival", ylabel="AIF / peak")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


TABLE_FEATURES = ["aif_arrival_s", "aif_ttp_s", "aif_peak", "aif_auc_first_pass", "aif_fwhm_s", "aif_rise_s",
                  "aif_auc_norm_s", "aif_sd_s", "aif_tail_frac", "CBV_median", "CBF_median", "MTT_median", "Tmax_median"]


def tex_table(t, n, path):
    tt = t[t.variant == "ica_top"].set_index("feature")
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{Plane B (lateral) versus plane A (frontal) of the same injection, \\texttt{{ica\\_top}} AIF, "
             f"pairs with both curves trusted ($n$ = {n}). Effect: mean (paired $t$) or median ($^\\dagger$Wilcoxon) "
             "of B $-$ A; amplitudes and perfusion parameters as ratio B/A $-$ 1. LoA: Bland--Altman 95\\,\\% limits "
             "of agreement (bias $\\pm$ 1.96\\,SD; $^\\dagger$2.5th--97.5th percentile). $p$: Holm-corrected over 15 "
             "features. Shape-only features in italics.}",
             "\\label{tab:sq2_main}", "\\footnotesize", "\\begin{tabular}{lrcr}", "\\hline",
             "Feature & B vs A & 95\\,\\% LoA & $p$ \\\\", "\\hline"]
    for c in TABLE_FEATURES:
        if c not in tt.index:
            continue
        r = tt.loc[c]
        name = FEATURES[c][0] + (" (s)" if r.scale == "s" else "")
        if c in SHAPE_ONLY:
            name = f"\\emph{{{name}}}"
        eff = r.effect.replace("%", "\\%").replace(" s", "") + ("$^\\dagger$" if r.test == "Wilcoxon" else "")
        # non-normal differences: non-parametric limits (2.5th-97.5th percentile, Bland & Altman 1999)
        lo, hi = (r.ba_loa_low, r.ba_loa_high) if r.normal else (r.np_loa_low, r.np_loa_high)
        loa = (f"{np.expm1(lo):+.0%} to {np.expm1(hi):+.0%}" if r.scale == "log" else f"{lo:+.2f} to {hi:+.2f}")
        lines.append(f"{name} & {eff} & {loa.replace('%', chr(92) + '%')} & {fmt_p(r.p_holm)} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    args = p.parse_args()
    out = args.out
    out_dir = SQ2
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(out / "features" / "features_aif.csv")
    curves = pd.read_csv(out / "features" / "curves_shape.csv")
    tax = pd.read_csv(out / "qc" / "taxonomy" / "taxonomy_labels.csv")[
        ["run_id", "variant", "sat_frac", "venous_frac", "snr", "mask_px"]].rename(columns={"mask_px": "mask_px_geo"})
    pr = pd.read_csv(PLANE / "plane_runs.csv")[["run_id", "variant", "ring_vessel_frac", "global_peak"]]
    rab = pd.read_csv(out / "features.csv")[["run_id", "variant", "r_ab"]]
    extra = tax.merge(pr, on=["run_id", "variant"]).merge(rab, on=["run_id", "variant"])

    pairs = make_pairs(feat, extra)
    pairs.to_csv(out_dir / "sq2_pairs.csv", index=False)
    t = tests(pairs, "all pairs")
    t.to_csv(out_dir / "sq2_tests.csv", index=False)
    mech = mechanism(pairs)
    mech.to_csv(out_dir / "sq2_mechanism.csv", index=False)
    sat, sat_p = saturation_check(pairs)
    sat.to_csv(out_dir / "sq2_saturation.csv", index=False)
    rmse = curve_rmse(pairs, curves, feat)
    rmse.to_csv(out_dir / "sq2_curve_rmse.csv", index=False)

    # sensitivity: ICA injections only; pairs without saturated mask pixels
    key = ["aif_arrival_s", "aif_peak", "aif_auc_first_pass", "aif_fwhm_s", "CBV_median", "CBF_median", "MTT_median"]
    ica = pairs[pairs.injection_site.str.contains("ICA", na=False) & ~pairs.injection_site.str.contains("CCA", na=False)]
    sens_rows = []
    for name, sub in (("all pairs", pairs), ("ICA injections only", ica), ("no saturated pixels", pairs[~pairs.any_sat])):
        s = tests(sub, name)
        s = s[(s.variant == "ica_top") & s.feature.isin(key)]
        sens_rows += [dict(subset=name, n=int(r.n), feature=r.label, effect=r.effect, p_holm=fmt_p(r.p_holm))
                      for r in s.itertuples()]
    sens = pd.DataFrame(sens_rows).pivot_table(index="feature", columns="subset", values="effect", aggfunc="first")
    sens_n = pd.DataFrame(sens_rows).groupby("subset").n.max()
    sens.columns = [f"{c} (n={sens_n[c]})" for c in sens.columns]
    sens = sens.reindex([FEATURES[k][0] for k in key]).reset_index()
    sens.to_csv(out_dir / "sq2_sensitivity.csv", index=False)

    plot_forest(t, out_dir / "forest.png")
    plot_ba(pairs, t, "ica_top", out_dir / "ba_ica_top.png")
    plot_mechanism(pairs, out_dir / "mechanism.png")
    plot_shape_curves(pairs, curves, feat, out_dir / "shape_curves.png")
    tex_table(t, int((pairs.variant == "ica_top").sum()), out_dir / "table_sq2.tex")
    print(pairs.groupby("variant").size())
    print(t[t.variant == "ica_top"][["label", "test", "effect", "p_holm"]].to_string())
    print(mech.round(3).to_string())
    print(sat.to_string(), sat_p)
    print(rmse.describe().round(3).to_string())
    print(sens.to_string())


if __name__ == "__main__":
    main()
