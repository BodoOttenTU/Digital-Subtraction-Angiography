"""SQ3: 95 % CI of every variance component, between- vs within-patient SD and the run-length check (needs sq3_variance.py).

Uses the same patient resamples as sq3_variance.py, so the ICC intervals must come out identical.

    python scripts/sq3_component_ci.py [--n-boot 1000]
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one process per feature, so no nested threading

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA, SQ3  # noqa: E402
from sq1_full_vs_top import FEATURES  # noqa: E402
from sq3_variance import FULL, NO_INT, NO_SEG, NO_VIEW, N_BOOT, PRIMARY, SEED, SEG, components, fit, r2_marginal  # noqa: E402

COMPS = ["patient", "view", "segment", "interaction", "run", "residual"]


def shares(res, d):
    """Fractions of the total variance, with the same nested part R2 as sq3_variance.py."""
    f, p, run, e = components(res)
    total = f + p + run + e
    additive = r2_marginal(fit(NO_INT, d))
    return dict(patient=p / total, run=run / total, residual=e / total,
                view=max(additive - r2_marginal(fit(NO_VIEW, d)), 0.0),
                segment=max(additive - r2_marginal(fit(NO_SEG, d)), 0.0),
                interaction=max(f / total - additive, 0.0),
                within=(run + e) / total, fixed=f / total, icc_adj=p / (p + run + e), icc_agree=p / total,
                sd_patient=np.sqrt(p), sd_within=np.sqrt(run + e), sd_ratio=np.sqrt(p / (run + e)))


def analyse_feature(c, label, scale, d0, seed, n_boot):
    d = d0[d0[c] > 0].copy() if scale == "log" else d0.copy()
    d["y"] = np.log(d[c]) if scale == "log" else d[c]
    rng = np.random.default_rng(seed)
    patients = [g for _, g in d.groupby("patient")]
    out = []
    for _ in range(n_boot):
        pick = rng.integers(len(patients), size=len(patients))
        b = pd.concat([patients[i].assign(patient=k) for k, i in enumerate(pick)], ignore_index=True)
        try:
            r = fit(FULL, b)
            if not r.converged:
                continue
            out.append(shares(r, b))
        except Exception:  # singular refit
            continue
    boot = pd.DataFrame(out)
    point = shares(fit(FULL, d), d)
    row = dict(feature=c, label=label, scale=scale, n_boot_used=len(boot), mean=d[c].mean(), **point)
    for k in boot.columns:
        row[f"{k}_lo"], row[f"{k}_hi"] = np.percentile(boot[k], [2.5, 97.5])
    # run length is the same in both views of an injection, so it can pass for a patient effect
    dur = d.dropna(subset=["duration_s"])
    _, p, run, e = components(fit(FULL + " + duration_s", dur))
    row["icc_adj_given_duration"] = p / (p + run + e)
    print(f"{label}: patient {point['patient']:.0%}, ICC_adj {point['icc_adj']:.2f}, "
          f"given run length {row['icc_adj_given_duration']:.2f}, bootstrap {len(boot)}/{n_boot}", flush=True)
    return row


def tex_table(t, path):
    pc = lambda r, k: (f"\\val{{{getattr(r, k) * 100:.0f}}} "  # noqa: E731
                       f"(\\val{{{getattr(r, k + '_lo') * 100:.0f}}}--\\val{{{getattr(r, k + '_hi') * 100:.0f}}})")
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Share of the total variance per feature, in percent, with 95\\,\\% cluster bootstrap CI "
             "(\\val{1000} resamples of patients). The patient share is also the agreement ICC. Between runs: differences "
             "between the two views of a patient beyond the average view effect. Residual: differences within a run beyond "
             "the average segment effect, including measurement noise. Rows may not add to exactly 100.}",
             "\\label{tab:sq3}", "\\footnotesize", "\\begin{tabular}{lrrrrrrc}", "\\hline",
             "Feature & Patient & View & Segment & View$\\times$segment & Between runs & Residual & ICC$_\\mathrm{adj}$ \\\\",
             "\\hline"]
    for r in t.itertuples():
        if r.feature == "CBV_median":
            lines.append("\\hline")
        cells = " & ".join(pc(r, k) for k in COMPS)
        lines.append(f"{r.label} & {cells} & \\val{{{r.icc_adj:.2f}}} (\\val{{{r.icc_adj_lo:.2f}}}--\\val{{{r.icc_adj_hi:.2f}}}) \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    p.add_argument("--n-boot", type=int, default=N_BOOT, help="cluster bootstrap resamples")
    args = p.parse_args()
    feat = pd.read_csv(args.out / "features" / "features_aif.csv")
    d0 = feat[feat.variant.isin(PRIMARY) & (feat.qc_trust == "trusted")].copy()
    d0["view"], d0["seg"], d0["patient"] = d0.plane, d0.variant.map(SEG), d0.patient_id
    with ProcessPoolExecutor(max_workers=min(len(FEATURES), os.cpu_count() or 1)) as pool:
        jobs = [pool.submit(analyse_feature, c, label, scale, d0[["patient", "view", "seg", "duration_s", c]].dropna(subset=[c]),
                            [SEED, k], args.n_boot)
                for k, (c, (label, scale)) in enumerate(FEATURES.items())]
        t = pd.DataFrame([j.result() for j in jobs])
    t.to_csv(SQ3 / "sq3_component_ci.csv", index=False)
    tex_table(t, SQ3 / "table_sq3_components.tex")
    v = pd.read_csv(SQ3 / "sq3_variance.csv").set_index("feature")
    dev = np.abs(t.set_index("feature").icc_adj_lo - v.icc_adj_lo).max()
    print("largest difference from the ICC intervals of sq3_variance.py:", round(float(dev), 6))


if __name__ == "__main__":
    main()
