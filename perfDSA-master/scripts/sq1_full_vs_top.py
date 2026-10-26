"""
SQ1 — effect of the sampling segment: AIF from the full ICA (ica_full_model, ica_full_grown) versus the
supraclinoid ICA top (ica_top), per plane.

Pairs
    One pair per run and comparison (model vs top, grown vs top): both curves of the run must be trusted
    (aif_taxonomy.py trust flag), so a pair never contains a rejected curve. Per plane separately, because the
    projection changes which part of the ICA each mask covers.

Differences (full - top)
    timing features (s): plain difference; amplitude and perfusion features (strictly positive, error
    proportional to size): log ratio, reported as % (exp(d) - 1); dimensionless shape features: difference.
    `temporal r` (Pearson r of the full and top curve, features.csv r_vs_top) is summarised, not tested.

Normality and test choice
    Shapiro-Wilk on the paired differences of every feature x comparison x plane, plus Q-Q plots
    (qq_<comparison>.png). Pre-specified rule: paired t-test if Shapiro-Wilk p >= 0.05, else Wilcoxon
    signed-rank (the test does not assume normality, only a symmetric difference distribution). Both p-values
    are kept in the table; the reported one follows the rule.
    Holm correction over the features within one family = one comparison in one plane.

Agreement
    Bland-Altman per feature (ba_<comparison>.png): bias = mean difference, limits of agreement = bias +/-
    1.96 SD; when the differences are not normal also the non-parametric limits (2.5th-97.5th percentile).
    Log-ratio features are plotted on the log scale against the geometric mean.

Plane interaction ("does the direction flip?")
    For injections with a pair in both planes: Wilcoxon signed-rank on d_B - d_A (does the full-vs-top
    difference differ between the views), and the sign of the median difference per plane.
    Candidate explanations, Spearman correlation of d with (plane_difference/plane_runs.csv):
        pixel count     log(mask_px full / top)
        overlap         ring_vessel_frac full - top (vessels around the mask), venous_frac full - top
        foreshortening  ica_top mask_px (a foreshortened siphon projects onto fewer, denser pixels) and
                        mask_spread of the top mask

Input : <out>/aif_features/features_aif.csv, features.csv (r_vs_top), plane_difference/plane_runs.csv,
        aif_qc/taxonomy/taxonomy_labels.csv (spread_px)
Output: <out>/sq1_full_vs_top/
    sq1_pairs.csv, sq1_tests.csv, sq1_interaction.csv, sq1_flip_correlations.csv, table_sq1.tex,
    qq_<comparison>.png, ba_<comparison>.png, forest.png, sq1_report.md

Example:
    python scripts/sq1_full_vs_top.py results/combined_v1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

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


# ----------------------------------------------------------------------------
# Pairs
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
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


def tests(pairs):
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
                normal = sw.pvalue >= ALPHA
                sd = d.std(ddof=1)
                rows.append(dict(
                    comparison=comp, plane=plane, feature=c, label=label, scale=scale, n=len(d),
                    shapiro_W=sw.statistic, shapiro_p=sw.pvalue, normal=normal,
                    test="paired t" if normal else "Wilcoxon", p_t=t.pvalue, p_wilcoxon=w.pvalue if w else np.nan,
                    p=t.pvalue if normal else (w.pvalue if w else np.nan),
                    mean=d.mean(), sd=sd, median=d.median(), q1=d.quantile(0.25), q3=d.quantile(0.75),
                    ba_bias=d.mean(), ba_loa_low=d.mean() - 1.96 * sd, ba_loa_high=d.mean() + 1.96 * sd,
                    np_loa_low=d.quantile(0.025), np_loa_high=d.quantile(0.975),
                    effect=fmt_effect(d.median() if not normal else d.mean(), scale)))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for (comp, plane), idx in t.groupby(["comparison", "plane"]).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
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


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}


def plot_qq(pairs, comp, path):
    feats = list(FEATURES)
    ncol = 5
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3 * nrow), squeeze=False)
    p = pairs[pairs.comparison == comp]
    for ax, c in zip(axes.flat, feats):
        for plane in PLANES:
            d = p[p.plane == plane][f"d_{c}"].dropna()
            if len(d) < 3:
                continue
            (osm, osr), (slope, icpt, _) = stats.probplot(d, dist="norm")
            ax.scatter(osm, osr, s=8, color=PLANE_COL[plane], label=plane)
            ax.plot(osm, slope * osm + icpt, color=PLANE_COL[plane], lw=0.8)
        ax.set_title(FEATURES[c][0], fontsize=9)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(feats):]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7, title="plane", title_fontsize=7)
    fig.suptitle(f"Q-Q plots of the paired differences, {COMPARISONS[comp]} - ica_top")
    fig.supxlabel("theoretical normal quantile")
    fig.supylabel("observed difference")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


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


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def md_table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def fmt_p(x):
    return "" if pd.isna(x) else (f"{x:.1e}" if x < 0.001 else f"{x:.3f}")


def tex_table(t, path):
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{SQ1: full ICA minus ICA top, per plane (pairs with both curves trusted). Effect: mean "
             "(paired t) or median (Wilcoxon) of the difference; ratio features as \\%. $p$: Holm-corrected within "
             "comparison and plane; test chosen by Shapiro-Wilk ($p<0.05$: Wilcoxon).}",
             "\\label{tab:sq1}", "\\footnotesize", "\\begin{tabular}{l" + "rr" * 4 + "}", "\\hline",
             " & \\multicolumn{4}{c}{ica\\_full\\_model $-$ top} & \\multicolumn{4}{c}{ica\\_full\\_grown $-$ top} \\\\",
             "Feature & A & $p$ & B & $p$ & A & $p$ & B & $p$ \\\\", "\\hline"]
    for c, (label, _) in FEATURES.items():
        cells = []
        for comp in COMPARISONS:
            for plane in PLANES:
                r = t[(t.comparison == comp) & (t.plane == plane) & (t.feature == c)]
                if len(r):
                    r = r.iloc[0]
                    eff = r.effect.replace("%", "\\%").replace(" s", "")
                    mark = "$^\\dagger$" if r.test == "Wilcoxon" else ""
                    cells += [eff + mark, fmt_p(r.p_holm)]
                else:
                    cells += ["", ""]
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\multicolumn{9}{l}{$^\\dagger$ Wilcoxon signed-rank (non-normal differences); "
              "otherwise paired $t$-test. Times in s.} \\\\", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mask_context(pairs, runs):
    """Median mask size and overlap of each mask, per plane, over the pairs of each comparison."""
    r = runs.set_index(["run_id", "variant"])
    rows = []
    for comp, variant in COMPARISONS.items():
        for plane in PLANES:
            ids = pairs[(pairs.comparison == comp) & (pairs.plane == plane)].run_id
            for v in ("ica_top", variant):
                sub = r.loc[list(zip(ids, [v] * len(ids)))]
                rows.append(dict(comparison=comp, plane=plane, mask=v, mask_px=int(sub.mask_px.median()),
                                 ring_vessel_frac=round(sub.ring_vessel_frac.median(), 3),
                                 venous_frac=round(sub.venous_frac.median(), 3)))
    return pd.DataFrame(rows)


def write_report(out_dir, pairs, t, inter, corr, ctx, command):
    n = pairs.groupby(["comparison", "plane"]).size().unstack()
    r_t = pairs.groupby(["comparison", "plane"]).temporal_r.median().unstack().round(4)
    tt = t.assign(shapiro_p=t.shapiro_p.map(fmt_p), p_t=t.p_t.map(fmt_p), p_wilcoxon=t.p_wilcoxon.map(fmt_p),
                  p_holm=t.p_holm.map(fmt_p))
    nn = t.groupby(["comparison", "plane"]).normal.agg(["sum", "size"])
    ba = t.copy()
    for col in ["ba_bias", "ba_loa_low", "ba_loa_high", "np_loa_low", "np_loa_high"]:
        ba[col] = [fmt_effect(v, s) for v, s in zip(ba[col], ba.scale)]
    sig_corr = corr[(corr.p < 0.01) & (corr.plane != "both")].sort_values("p").round(3)
    dis = t[t.p_wilcoxon.notna() & ((t.p_t < ALPHA) != (t.p_wilcoxon < ALPHA))]
    disagree = len(dis)
    dis_list = [f"{r.comparison} {r.plane} {r.label}" for r in dis.itertuples()]
    ii = inter.assign(p=inter.p.map(fmt_p), p_holm=inter.p_holm.map(fmt_p))
    txt = f"""# SQ1 — full ICA vs ICA top ({out_dir.parent.name})

Generated by `scripts/sq1_full_vs_top.py`; do not edit by hand, re-run instead.

```bash
{command}
```

Pairs (both curves trusted), per comparison and plane:

{md_table(n.reset_index())}

Temporal agreement (median Pearson r of the full and the top curve):

{md_table(r_t.reset_index())}

## 1. Normality of the paired differences

Shapiro-Wilk per feature x comparison x plane (Q-Q plots: `qq_model.png`, `qq_grown.png`). Normal (p >= 0.05)
in this many of the features:

{md_table(nn.rename(columns={"sum": "normal", "size": "features"}).reset_index())}

Rule (pre-specified): paired t-test when normal, else Wilcoxon signed-rank. Both p-values are in `sq1_tests.csv`.
Robustness of the choice: the two tests disagree on p < 0.05 (uncorrected) for {disagree} of {len(t)} tests
{f"({', '.join(dis_list)})" if dis_list else ""}.

## 2. Paired tests per plane, Holm-corrected

Holm over the {len(FEATURES)} features of one comparison in one plane. LaTeX: `table_sq1.tex`; forest plot: `forest.png`.

{md_table(tt[["comparison", "plane", "label", "n", "shapiro_p", "test", "effect", "p_t", "p_wilcoxon", "p_holm"]])}

## 3. Bland-Altman

Bias and 95 % limits of agreement of full - top (ratio features as %); `ba_model.png`, `ba_grown.png`.
Non-parametric limits (2.5th-97.5th percentile) for the non-normal features.

{md_table(ba[["comparison", "plane", "label", "normal", "ba_bias", "ba_loa_low", "ba_loa_high", "np_loa_low", "np_loa_high"]])}

## 4. Does the direction flip between planes?

Injections with a pair in both planes: median difference per plane, `flips` = opposite sign, Wilcoxon on
d_B - d_A (Holm within comparison):

{md_table(ii)}

Candidate explanations (Spearman of the full - top difference with pixel count, overlap and foreshortening
proxies; `sq1_flip_correlations.csv`). Correlations with p < 0.01:

{md_table(sig_corr) if len(sig_corr) else "None."}

Factors: `log_px_ratio` = log(mask px full / top) (pixel count); `d_ring_vessel`, `d_venous` = full - top
vessel fraction around the mask / late-opacifying fraction (overlap); `top_px`, `top_spread` = size and spread
of the ica_top mask (a foreshortened siphon projects onto fewer pixels).

Mask context per plane (medians over the pairs):

{md_table(ctx)}

Reading: the vessel fraction around the ica_top mask is much higher in the lateral view (B) than in the frontal
view (A), whereas around the grown (cervical + petrous) mask it is similar in both views. Of the candidate
factors, only this overlap difference correlates consistently with the full - top difference (pixel count and
the foreshortening proxies do not). The correlation does not tell which of the two curves moves; the
plane-difference analysis (plane_difference/) does: normalised to the whole-image contrast, the ica_top AIF is
the same in both views (B/A 1.00) while the grown AIF is lower in B (0.84). The view dependence therefore sits
in the grown AIF, and a lower AIF raises CBV/CBF/MTT (the deconvolution divides tissue by AIF).

## Caveats

- Pairs require both curves to be trusted, so runs where one mask fails drop out; the comparison is about
  the effect of the segment when both extractions are valid.
- Exploratory correlations (section 4) are not corrected for multiple testing.
"""
    (out_dir / "sq1_report.md").write_text(txt, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="combined result folder, e.g. results/combined_v1")
    args = p.parse_args()
    out = args.out
    out_dir = out / "sq1_full_vs_top"
    out_dir.mkdir(parents=True, exist_ok=True)

    feat = pd.read_csv(out / "aif_features" / "features_aif.csv")
    old = pd.read_csv(out / "features.csv")
    r_top = old.set_index(["run_id", "variant"]).r_vs_top.to_dict()
    runs = pd.read_csv(out / "plane_difference" / "plane_runs.csv")
    geo = pd.read_csv(out / "aif_qc" / "taxonomy" / "taxonomy_labels.csv")[["run_id", "variant", "spread_px"]]

    pairs = make_pairs(feat, r_top)
    pairs.to_csv(out_dir / "sq1_pairs.csv", index=False)
    t = tests(pairs)
    t.to_csv(out_dir / "sq1_tests.csv", index=False)
    inter = interaction(pairs)
    inter.to_csv(out_dir / "sq1_interaction.csv", index=False)
    corr = flip_correlations(pairs, runs, geo)
    corr.to_csv(out_dir / "sq1_flip_correlations.csv", index=False)
    for comp in COMPARISONS:
        plot_qq(pairs, comp, out_dir / f"qq_{comp}.png")
        plot_ba(pairs, t, comp, out_dir / f"ba_{comp}.png")
    plot_forest(t, out_dir / "forest.png")
    tex_table(t, out_dir / "table_sq1.tex")
    write_report(out_dir, pairs, t, inter, corr, mask_context(pairs, runs), " ".join(["python"] + sys.argv))
    print(pairs.groupby(["comparison", "plane"]).size().unstack())
    print(t[["comparison", "plane", "label", "test", "effect", "p_holm"]].to_string())
    print(inter.to_string())


if __name__ == "__main__":
    main()
