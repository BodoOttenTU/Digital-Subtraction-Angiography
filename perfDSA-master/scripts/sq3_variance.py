"""SQ3: variance decomposition (patient, view, segment, run) and ICC per feature with a linear mixed model.

    python scripts/sq3_variance.py [--variants ica_top ica_full_model ica_full_grown] [--n-boot 1000]
"""

import argparse
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one process per feature, so no nested threading

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sq1_full_vs_top import FEATURES  # noqa: E402
from paths import DATA, SQ3

SEG = {"ica_top": "top", "ica_full_model": "model", "ica_full_grown": "grown"}
# ica_full_model is almost a copy of ica_top, so it is a sensitivity analysis and not part of the main model
PRIMARY = ["ica_top", "ica_full_grown"]
FULL = "y ~ C(view, Treatment('A')) * C(seg, Treatment('top'))"
NO_VIEW = "y ~ C(seg, Treatment('top'))"
NO_SEG = "y ~ C(view, Treatment('A'))"
NO_INT = "y ~ C(view, Treatment('A')) + C(seg, Treatment('top'))"
# one intercept per run (patient x view): AIFs of one run share its frames and its tissue signal
VC = {"run": "0 + C(view)"}
N_BOOT = 1000
SEED = 1
COLORS = {"patient": "#2a78d6", "view": "#eb6834", "segment": "#2f9e44", "interaction": "#b8860b",
          "shared": "#c9c9c9", "run": "#a9a9a9", "residual": "#7a7a7a"}


def fit(formula, d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, d, groups=d["patient"], re_formula="1", vc_formula=VC).fit(reml=True)


def components(res):
    """Variance of the fixed effects, patient, run and residual."""
    s2_p = float(res.cov_re.iloc[0, 0])
    s2_r = float(np.atleast_1d(res.vcomp)[0])
    s2_e = float(res.scale)
    s2_f = float(np.var(res.model.exog @ res.fe_params.to_numpy()))
    return s2_f, s2_p, s2_r, s2_e


def r2_marginal(res):
    s2_f, s2_p, s2_r, s2_e = components(res)
    return s2_f / (s2_f + s2_p + s2_r + s2_e)


def bootstrap_icc(d, rng, n_boot):
    """Cluster bootstrap of ICC_adj and ICC_agree: resample whole patients. Returns the CI and the converged refits."""
    patients = [g for _, g in d.groupby("patient")]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(len(patients), size=len(patients))
        b = pd.concat([patients[i].assign(patient=k) for k, i in enumerate(pick)], ignore_index=True)
        try:
            r = fit(FULL, b)
        except Exception:  # singular refit
            continue
        if not r.converged:
            continue
        f, p, run, e = components(r)
        out.append((p / (p + run + e), p / (f + p + run + e)))
    a = np.array(out)
    return (np.percentile(a, [2.5, 97.5], axis=0) if len(a) else np.full((2, 2), np.nan)), len(a)


def contrasts(res):
    """B vs A per segment and full vs top per view, with 95 % CI (linear combinations of the coefficients)."""
    names = list(res.fe_params.index)
    b = "C(view, Treatment('A'))[T.B]"

    def L(*terms):
        v = np.zeros(len(names))
        for t in terms:
            v[names.index(t)] = 1
        return v[None, :]  # one-row contrast matrix over the fixed effects

    rows = []
    specs = [("B vs A, top", [b]),
             ("B vs A, model", [b, f"{b}:C(seg, Treatment('top'))[T.model]"]),
             ("B vs A, grown", [b, f"{b}:C(seg, Treatment('top'))[T.grown]"]),
             ("model vs top, A", ["C(seg, Treatment('top'))[T.model]"]),
             ("model vs top, B", ["C(seg, Treatment('top'))[T.model]", f"{b}:C(seg, Treatment('top'))[T.model]"]),
             ("grown vs top, A", ["C(seg, Treatment('top'))[T.grown]"]),
             ("grown vs top, B", ["C(seg, Treatment('top'))[T.grown]", f"{b}:C(seg, Treatment('top'))[T.grown]"])]
    for name, terms in specs:
        if not all(t in names for t in terms):
            continue
        tt = res.t_test(L(*terms))
        est = float(np.squeeze(tt.effect))
        lo, hi = np.squeeze(tt.conf_int())
        rows.append(dict(contrast=name, estimate=est, ci_low=float(lo), ci_high=float(hi), p=float(np.squeeze(tt.pvalue))))
    # Wald test of the interaction (both interaction coefficients = 0)
    inter = [i for i, n in enumerate(names) if ":" in n]
    R = np.zeros((len(inter), len(res.params)))
    for k, i in enumerate(inter):
        R[k, i] = 1
    w = res.wald_test(R, scalar=True)
    return rows, float(w.pvalue)


def analyse_feature(c, label, scale, d, seed, n_boot):
    if scale == "log":
        d = d[d[c] > 0].copy()
        d["y"] = np.log(d[c])
    else:
        d = d.copy()
        d["y"] = d[c]
    if d.y.nunique() < 5:
        return None
    res = fit(FULL, d)
    s2_f, s2_p, s2_r, s2_e = components(res)
    total = s2_f + s2_p + s2_r + s2_e
    within = s2_r + s2_e
    r2m = s2_f / total
    # nested part R2: each main effect within the additive model, the interaction on top of it
    additive = r2_marginal(fit(NO_INT, d))
    parts = dict(view=max(additive - r2_marginal(fit(NO_VIEW, d)), 0.0),
                 segment=max(additive - r2_marginal(fit(NO_SEG, d)), 0.0),
                 interaction=max(r2m - additive, 0.0))
    shared = max(r2m - sum(parts.values()), 0.0)
    ci, n_ok = bootstrap_icc(d, np.random.default_rng(seed), n_boot)
    mean = d[c].mean()
    cv_w = np.sqrt(np.expm1(within)) if scale == "log" else np.sqrt(within) / abs(mean)
    cv_t = np.sqrt(np.expm1(within + s2_f)) if scale == "log" else np.sqrt(within + s2_f) / abs(mean)
    cons, p_int = contrasts(res)
    var_row = dict(feature=c, label=label, scale=scale, n_obs=len(d), n_patients=d.patient.nunique(),
                   patient=s2_p / total, view=parts["view"], segment=parts["segment"],
                   interaction=parts["interaction"], shared=shared, run=s2_r / total, residual=s2_e / total,
                   icc_adj=s2_p / (s2_p + within), icc_adj_lo=ci[0, 0], icc_adj_hi=ci[1, 0],
                   icc_agree=s2_p / total, icc_agree_lo=ci[0, 1], icc_agree_hi=ci[1, 1],
                   cv_within=cv_w, cv_total=cv_t, p_interaction=p_int, converged=bool(res.converged),
                   n_boot_converged=n_ok)
    fix_rows = [dict(feature=c, label=label, scale=scale, **r) for r in cons]
    print(f"{label}: ICC_adj {s2_p / (s2_p + within):.2f}, patient {s2_p / total:.0%}, run {s2_r / total:.0%}, "
          f"fixed {r2m:.0%}, bootstrap {n_ok}/{n_boot}", flush=True)
    return var_row, fix_rows


def analyse(feat, n_boot):
    d0 = feat[feat.qc_trust == "trusted"].copy()
    d0["view"] = d0.plane
    d0["seg"] = d0.variant.map(SEG)
    d0["patient"] = d0.patient_id
    with ProcessPoolExecutor(max_workers=min(len(FEATURES), os.cpu_count() or 1)) as pool:
        jobs = [pool.submit(analyse_feature, c, label, scale, d0[["patient", "view", "seg", c]].dropna(),
                            [SEED, k], n_boot)
                for k, (c, (label, scale)) in enumerate(FEATURES.items())]
        done = [j.result() for j in jobs]
    done = [x for x in done if x is not None]
    return pd.DataFrame([x[0] for x in done]), pd.DataFrame([r for x in done for r in x[1]])


def plot_fractions(v, path):
    v = v.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    left = np.zeros(len(v))
    for comp in ["patient", "view", "segment", "interaction", "shared", "run", "residual"]:
        ax.barh(v.label, v[comp], left=left, color=COLORS[comp], label=comp)
        left += v[comp].to_numpy()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of total variance")
    ax.set_title("Where does the variation of each feature come from?")
    ax.legend(ncol=3, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.09))
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def tex_variance(v, path):
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Variance decomposition per feature (linear mixed model, trusted curves of the included AIF variants and "
             "both views). Fractions of the total variance due to the patient, the systematic view and segment effects and "
             "their interaction (nested part $R^2$), differences between the two runs of a patient beyond the systematic view "
             "effect (between runs), and the residual. ICC$_\\mathrm{adj}$: agreement between repeated AIFs of one patient "
             "after removing the systematic view and segment effects; ICC$_\\mathrm{agree}$: without removing them (95\\,\\% "
             "cluster bootstrap CI). CV$_\\mathrm{w}$: within-patient coefficient of variation.}",
             "\\label{tab:sq3}", "\\footnotesize", "\\begin{tabular}{l rrrrrr cc r}", "\\hline",
             "Feature & Patient & View & Segment & View$\\times$seg. & Between runs & Residual & ICC$_\\mathrm{adj}$ & "
             "ICC$_\\mathrm{agree}$ & CV$_\\mathrm{w}$ \\\\", "\\hline"]
    for r in v.itertuples():
        pc = lambda x: f"{x:.0%}".replace("%", "\\%")  # noqa: E731
        cv = "" if r.scale == "abs" else pc(r.cv_within)
        lines.append(f"{r.label} & {pc(r.patient)} & {pc(r.view)} & {pc(r.segment)} & {pc(r.interaction)} & "
                     f"{pc(r.run)} & {pc(r.residual)} & {r.icc_adj:.2f} ({r.icc_adj_lo:.2f}--{r.icc_adj_hi:.2f}) & "
                     f"{r.icc_agree:.2f} ({r.icc_agree_lo:.2f}--{r.icc_agree_hi:.2f}) & {cv} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_fixed(f, path):
    want = [w for w in ["B vs A, top", "grown vs top, A", "grown vs top, B", "model vs top, A", "model vs top, B"]
            if w in set(f.contrast)]
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Fixed effects of the mixed model (estimate and 95\\,\\% CI): view effect for the supraclinoid AIF and "
             "segment effects per view. Amplitudes and perfusion parameters as \\%, times in s.}",
             "\\label{tab:sq3_fixed}", "\\footnotesize", "\\begin{tabular}{l" + "c" * len(want) + "}", "\\hline",
             "Feature & " + " & ".join(want) + " \\\\", "\\hline"]
    for feat, g in f.groupby("feature", sort=False):
        g = g.set_index("contrast")
        cells = []
        for w in want:
            r = g.loc[w]
            s = r.scale
            est, lo, hi = (np.expm1([r.estimate, r.ci_low, r.ci_high]) * 100 if s == "log"
                           else np.array([r.estimate, r.ci_low, r.ci_high]))
            unit = "\\%" if s == "log" else ""
            dig = 0 if s == "log" else 2
            cells.append(f"{est:+.{dig}f}{unit} ({lo:+.{dig}f}, {hi:+.{dig}f})")
        lines.append(f"{g.label.iloc[0]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    p.add_argument("--variants", nargs="+", choices=list(SEG), default=PRIMARY,
                   help="AIF variants to include; anything but the main model writes to sq3/<seg>_<seg>/")
    p.add_argument("--n-boot", type=int, default=N_BOOT, help="cluster bootstrap resamples")
    args = p.parse_args()
    suffix = "" if set(args.variants) == set(PRIMARY) else "_" + "_".join(SEG[v] for v in SEG if v in args.variants)
    out_dir = SQ3 / suffix.lstrip("_")
    out_dir.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(args.out / "features" / "features_aif.csv")
    feat = feat[feat.variant.isin(args.variants)]
    v, f = analyse(feat, args.n_boot)
    v.to_csv(out_dir / "sq3_variance.csv", index=False)
    if suffix:
        return
    f.to_csv(out_dir / "sq3_fixed.csv", index=False)
    plot_fractions(v, out_dir / "variance_fractions.png")
    tex_variance(v, out_dir / "table_sq3.tex")
    tex_fixed(f, out_dir / "table_sq3_fixed.tex")


if __name__ == "__main__":
    main()
