"""SQ1: full-ICA AIF (model, grown) vs supraclinoid ICA top, paired per plane.

    python scripts/sq1_full_vs_top.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from paths import DATA, PLANE, SQ1

COMPARISONS = {"model": "ica_full_model", "grown": "ica_full_grown"}
PLANES = ["A", "B"]
ALPHA = 0.05
# feature -> (label, scale): "log" = log ratio full/top, "s" = difference in seconds, "abs" = difference
FEATURES = {
    "aif_arrival_s": ("Arrival time", "s"),
    "aif_ttp_s": ("Time to peak", "s"),
    "aif_mtt_bolus_s": ("Bolus MTT", "s"),
    "aif_peak": ("Peak height", "log"),
    "aif_auc_first_pass": ("First-pass AUC", "log"),
    "aif_fwhm_s": ("FWHM", "s"),
    "aif_rise_s": ("Rise time", "s"),
    "aif_auc_norm_s": ("AUC / peak", "s"),
    "aif_sd_s": ("Temporal SD", "s"),
    "aif_skewness": ("Skewness", "abs"),
    "aif_tail_frac": ("Tail fraction", "abs"),
    "CBV_median": ("CBV (median)", "log"),
    "CBF_median": ("CBF (median)", "log"),
    "MTT_median": ("MTT (median)", "log"),
    "Tmax_median": ("Tmax (median)", "s"),
}


def make_pairs(feat, r_top):
    top = feat[feat.variant == "ica_top"].set_index("run_id")
    rows = []
    for comp, variant in COMPARISONS.items():
        full = feat[feat.variant == variant].set_index("run_id")
        ok = top.index[(top.qc_trust == "trusted")].intersection(full.index[full.qc_trust == "trusted"])
        for run_id in ok:
            t, f = top.loc[run_id], full.loc[run_id]
            row = dict(comparison=comp, run_id=run_id, plane=t.plane, ab_pair_id=t.ab_pair_id,
                       qc_included=t.qc_included, temporal_r=r_top.get((run_id, variant), np.nan))
            for c, (_, scale) in FEATURES.items():
                row[f"{c}_top"], row[f"{c}_full"] = t[c], f[c]
                row[f"d_{c}"] = (np.log(f[c] / t[c]) if scale == "log" else f[c] - t[c])
            rows.append(row)
    return pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    q = p[ok]
    order = np.argsort(q)
    adj = np.maximum.accumulate((len(q) - np.arange(len(q))) * q[order])
    tmp = np.empty_like(q)
    tmp[order] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def fmt_effect(x, scale):
    return f"{np.expm1(x):+.1%}" if scale == "log" else (f"{x:+.3f} s" if scale == "s" else f"{x:+.3f}")


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation; zero differences are dropped, as in the Wilcoxon test."""
    nz = d[d != 0]
    if nz.empty:
        return np.nan
    r = stats.rankdata(nz.abs())
    return (r[nz > 0].sum() - r[nz < 0].sum()) / r.sum()


def tests(pairs):
    """Wilcoxon signed-rank for every feature; Shapiro-Wilk and the paired t-test are kept as sensitivity checks."""
    rows = []
    for comp in COMPARISONS:
        for plane in PLANES:
            p = pairs[(pairs.comparison == comp) & (pairs.plane == plane)]
            for c, (label, scale) in FEATURES.items():
                d = p[f"d_{c}"].replace([np.inf, -np.inf], np.nan).dropna()
                if len(d) < 8:
                    continue
                sw = stats.shapiro(d)
                t = stats.ttest_1samp(d, 0.0)
                w = stats.wilcoxon(d) if (d != 0).sum() > 5 else None
                sd = d.std(ddof=1)
                rows.append(dict(
                    comparison=comp, plane=plane, feature=c, label=label, scale=scale, n=len(d),
                    n_zero=int((d == 0).sum()), shapiro_W=sw.statistic, shapiro_p=sw.pvalue,
                    test="Wilcoxon", p_t=t.pvalue, p=w.pvalue if w else np.nan, rank_biserial=rank_biserial(d),
                    mean=d.mean(), sd=sd, median=d.median(), q1=d.quantile(0.25), q3=d.quantile(0.75),
                    ba_bias=d.mean(), ba_loa_low=d.mean() - 1.96 * sd, ba_loa_high=d.mean() + 1.96 * sd,
                    np_loa_low=d.quantile(0.025), np_loa_high=d.quantile(0.975),
                    effect=fmt_effect(d.median(), scale)))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    t["p_t_holm"] = np.nan
    for (comp, plane), idx in t.groupby(["comparison", "plane"]).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
        t.loc[idx, "p_t_holm"] = holm(t.loc[idx, "p_t"])
    t["significant"] = t.p_holm < ALPHA
    return t


def interaction(pairs):
    """Does the full-vs-top difference differ between plane A and B of the same injection?"""
    rows = []
    for comp in COMPARISONS:
        p = pairs[pairs.comparison == comp]
        for c, (label, scale) in FEATURES.items():
            w = p.pivot_table(index="ab_pair_id", columns="plane", values=f"d_{c}", aggfunc="first").dropna()
            if len(w) < 8 or not set(PLANES) <= set(w.columns):
                continue
            dd = w.B - w.A
            test = stats.wilcoxon(dd) if (dd != 0).sum() > 5 else None
            mA, mB = w.A.median(), w.B.median()
            rows.append(dict(comparison=comp, feature=c, label=label, n_injections=len(w),
                             median_A=fmt_effect(mA, scale), median_B=fmt_effect(mB, scale),
                             flips=bool(np.sign(mA) != np.sign(mB) and mA != 0 and mB != 0),
                             p=test.pvalue if test else np.nan))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for comp, idx in t.groupby("comparison").groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
    return t


def flip_correlations(pairs, runs, geo):
    """Spearman of the full-top difference with pixel count, overlap and foreshortening proxies."""
    r = runs.set_index(["run_id", "variant"])
    g = geo.set_index(["run_id", "variant"])
    rows = []
    for comp, variant in COMPARISONS.items():
        p = pairs[pairs.comparison == comp].copy()
        top_k = list(zip(p.run_id, ["ica_top"] * len(p)))
        full_k = list(zip(p.run_id, [variant] * len(p)))
        p["log_px_ratio"] = np.log(r.loc[full_k, "mask_px"].to_numpy() / r.loc[top_k, "mask_px"].to_numpy())
        p["d_ring_vessel"] = r.loc[full_k, "ring_vessel_frac"].to_numpy() - r.loc[top_k, "ring_vessel_frac"].to_numpy()
        p["d_venous"] = r.loc[full_k, "venous_frac"].to_numpy() - r.loc[top_k, "venous_frac"].to_numpy()
        p["top_px"] = r.loc[top_k, "mask_px"].to_numpy()
        p["top_spread"] = g.loc[top_k, "spread_px"].to_numpy()
        for plane in PLANES + ["both"]:
            q = p if plane == "both" else p[p.plane == plane]
            for c in ["aif_peak", "aif_arrival_s", "aif_fwhm_s", "CBV_median", "MTT_median"]:
                for factor in ["log_px_ratio", "d_ring_vessel", "d_venous", "top_px", "top_spread"]:
                    ok = q[f"d_{c}"].notna() & q[factor].notna()
                    s = stats.spearmanr(q.loc[ok, f"d_{c}"], q.loc[ok, factor])
                    rows.append(dict(comparison=comp, plane=plane, feature=c, factor=factor, n=int(ok.sum()),
                                     rho=s.statistic, p=s.pvalue))
    return pd.DataFrame(rows)


PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}


def plot_ba(pairs, t, comp, path):
    feats = list(FEATURES)
    ncol = 5
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.1 * nrow), squeeze=False)
    p = pairs[pairs.comparison == comp]
    for ax, c in zip(axes.flat, feats):
        label, scale = FEATURES[c]
        for plane in PLANES:
            q = p[p.plane == plane]
            a, b = q[f"{c}_top"], q[f"{c}_full"]
            mean = np.sqrt(a * b) if scale == "log" else (a + b) / 2
            ax.scatter(mean, q[f"d_{c}"], s=8, color=PLANE_COL[plane], alpha=0.8, label=plane)
            r = t[(t.comparison == comp) & (t.plane == plane) & (t.feature == c)]
            if len(r):
                r = r.iloc[0]
                ax.axhline(r.ba_bias, color=PLANE_COL[plane], lw=1)
                for v in (r.ba_loa_low, r.ba_loa_high):
                    ax.axhline(v, color=PLANE_COL[plane], lw=0.8, ls="--")
        ax.axhline(0, color="k", lw=0.5)
        if scale == "log":
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{np.expm1(v):+.0%}"))
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, title="plane", title_fontsize=7)
    fig.suptitle(f"Bland-Altman, {COMPARISONS[comp]} - ica_top (line: bias, dashed: bias +/- 1.96 SD; "
                 "log-ratio features in %)")
    fig.supxlabel("mean of full and top (geometric mean for ratio features)")
    fig.supylabel("full - top")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_forest(t, path):
    fig, axes = plt.subplots(1, len(COMPARISONS), figsize=(13, 7), sharey=True)
    feats = list(FEATURES)
    y = np.arange(len(feats))[::-1]
    for ax, comp in zip(axes, COMPARISONS):
        for k, plane in enumerate(PLANES):
            r = t[(t.comparison == comp) & (t.plane == plane)].set_index("feature").reindex(feats)
            # standardised effect: median / SD of the differences, so features share one axis
            eff = r["median"] / r["sd"]
            lo, hi = r.q1 / r["sd"], r.q3 / r["sd"]
            yy = y + (0.15 if plane == "A" else -0.15)
            ax.errorbar(eff, yy, xerr=[eff - lo, hi - eff], fmt="o", color=PLANE_COL[plane], ms=4, lw=1,
                        label=f"plane {plane}", mfc=PLANE_COL[plane])
            sig = r.significant.fillna(False).astype(bool)
            ax.scatter(eff[sig], yy[sig.to_numpy()], s=60, facecolors="none", edgecolors="k", lw=0.8)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(f"{COMPARISONS[comp]} - ica_top")
        ax.set_xlabel("median difference / SD (IQR)")
    axes[0].set_yticks(y, [FEATURES[f][0] for f in feats])
    axes[0].legend(fontsize=8, title="ring = p_Holm < 0.05", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def fmt_p(x):
    return "" if pd.isna(x) else (f"{x:.1e}" if x < 0.001 else f"{x:.3f}")


def tex_table(t, path):
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{SQ1: full ICA minus ICA top, per plane (pairs with both curves trusted). Effect: median of "
             "the difference; ratio features as \\%, times in s. $p$: Wilcoxon signed-rank, Holm-corrected within "
             "comparison and plane. $r$: rank-biserial correlation.}",
             "\\label{tab:sq1}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{l" + "rrr" * 4 + "}", "\\hline",
             " & \\multicolumn{6}{c}{ica\\_full\\_model $-$ top} & \\multicolumn{6}{c}{ica\\_full\\_grown $-$ top} \\\\",
             "Feature" + " & A & $p$ & $r$ & B & $p$ & $r$" * 2 + " \\\\", "\\hline"]
    for c, (label, _) in FEATURES.items():
        cells = []
        for comp in COMPARISONS:
            for plane in PLANES:
                r = t[(t.comparison == comp) & (t.plane == plane) & (t.feature == c)]
                if len(r):
                    r = r.iloc[0]
                    eff = r.effect.replace("%", "\\%").replace(" s", "")
                    cells += [eff, fmt_p(r.p_holm), f"{r.rank_biserial:+.2f}" if pd.notna(r.p) else ""]
                else:
                    cells += ["", "", ""]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    args = p.parse_args()
    out = args.out
    out_dir = SQ1
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(out / "features" / "features_aif.csv")
    old = pd.read_csv(out / "features.csv")
    r_top = old.set_index(["run_id", "variant"]).r_vs_top.to_dict()
    runs = pd.read_csv(PLANE / "plane_runs.csv")
    geo = pd.read_csv(out / "qc" / "taxonomy" / "taxonomy_labels.csv")[["run_id", "variant", "spread_px"]]

    pairs = make_pairs(feat, r_top)
    pairs.to_csv(out_dir / "sq1_pairs.csv", index=False)
    t = tests(pairs)
    t.to_csv(out_dir / "sq1_tests.csv", index=False)
    inter = interaction(pairs)
    inter.to_csv(out_dir / "sq1_interaction.csv", index=False)
    corr = flip_correlations(pairs, runs, geo)
    corr.to_csv(out_dir / "sq1_flip_correlations.csv", index=False)
    for comp in COMPARISONS:
        plot_ba(pairs, t, comp, out_dir / f"ba_{comp}.png")
    plot_forest(t, out_dir / "forest.png")
    tex_table(t, out_dir / "table_sq1.tex")
    print(pairs.groupby(["comparison", "plane"]).size().unstack())
    print(t[["comparison", "plane", "label", "test", "effect", "p_holm"]].to_string())
    print(inter.to_string())


if __name__ == "__main__":
    main()
