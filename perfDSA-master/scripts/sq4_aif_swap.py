"""
SQ4 — does the AIF variability change the perfusion maps? AIF swap with the tissue curves held fixed.

For every run (tissue = its own DSA frames, unchanged) the perfDSA deconvolution is repeated with different AIFs:
    ref         own ica_top AIF (the perfDSA default; reference)
    model       own ica_full_model AIF                     } segment (SQ1)
    grown       own ica_full_grown AIF                     }
    cross       ica_top AIF of the other plane of the same injection (identical frame times)  } view (SQ2)
    cross_peak  the same, rescaled to the peak height of the own ica_top AIF: only its shape } differs
Because only the AIF changes, every difference in the maps is caused by the AIF. Maps are recomputed with the
unchanged pipeline (run_dataset.load_uniform_sequence, DSASequence), which reproduces the stored maps exactly.

ROI (fixed per run, the same for every AIF of that run)
    Enhancing parenchyma: pixels whose 3-frame-smoothed enhancement (max of 1 - frame/170, minus the median of the
    first two frames) is >= ROI_ENH (~5x the frame noise) and whose MINIP is >= VESSEL_LEVEL (no large vessels or
    black collimator/padding), cleaned with a binary opening. perfDSA sets CBV <= 0 to 0 in a large part of these
    pixels (reported as roi_cbv_pos_frac), so the analysis region is the ROI pixels with CBV > 0 in the reference
    map; this region is then kept fixed for every swap of the run, so differences cannot come from a changing pixel
    set. No vascular-territory atlas is available, so the ROI is the whole enhancing region of the view.

Per run x swap: median CBV, CBF, MTT and mean/median Tmax over the ROI, and the fraction of ROI pixels whose
Tmax differs from the reference. A swap is analysed only when both the reference and the donor AIF are trusted
(aif_taxonomy.py).

Analyses
    1. % change: log ratio swap / ref (Tmax: difference of the ROI mean, s); median, IQR, Wilcoxon, Holm over the
       four parameters of a swap.
    2. Bland-Altman per parameter per swap (log scale; parametric and non-parametric 95 % limits).
    3. Within-patient change vs between-patient spread: RMS of the log ratio swap/ref (includes bias) divided by
       the between-patient SD of the log reference value (per plane, one run per patient). Ratio < 1: the AIF
       choice moves a patient less than patients differ from each other.
    4. Rank stability: Spearman rho between the reference and the swapped values over the patients, per plane.
    5. Tmax: does the earlier arrival of the grown AIF (0.10-0.15 s) shift Tmax? Mean Tmax shift, fraction of
       pixels changed, and Spearman of the shift with the arrival difference top - grown.
    6. A vs B maps of the same injection: total log(B_own / A_own) split into an AIF part log(B_own / B with
       A-AIF) and a tissue part log(B with A-AIF / A_own).

Output: <out>/sq4_aif_swap/  swap_values.csv (raw, per run x swap), sq4_change.csv, sq4_within_between.csv,
        sq4_rank.csv, sq4_tmax.csv, sq4_decomposition.csv, table_sq4.tex, change.png, ba_<swap>.png,
        decomposition.png, roi_examples.png, sq4_report.md

Example:
    python scripts/sq4_aif_swap.py results/combined_v1            # computes (~10 min) and analyses
    python scripts/sq4_aif_swap.py results/combined_v1 --report-only
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.ndimage import binary_opening, uniform_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_truncation import load_frames, normalise, perfusion, sample_aif  # noqa: E402
from sq1_full_vs_top import fmt_p, holm, md_table  # noqa: E402

PARAMS = ["CBV", "CBF", "MTT", "Tmax"]
SWAPS = {"model": "segment: ica_full_model", "grown": "segment: ica_full_grown",
         "cross": "view: other plane's ica_top", "cross_peak": "view, shape only (peak-matched)"}
DONOR = {"ref": "ica_top", "model": "ica_full_model", "grown": "ica_full_grown", "cross": "ica_top", "cross_peak": "ica_top"}
ROI_ENH = 0.05        # enhancement threshold (fraction of the 0..1 contrast scale), ~5x frame noise 0.01-0.015
VESSEL_LEVEL = 120    # MINIP grey level below which a pixel is a large vessel / black region (background 170)
DT = 0.25
SWAP_COL = {"model": "#7a7a7a", "grown": "#2f9e44", "cross": "#eb6834", "cross_peak": "#f4a582"}


# ----------------------------------------------------------------------------
# Computation
# ----------------------------------------------------------------------------
def parenchyma_roi(frames):
    s = 1 - frames / 170
    enh = uniform_filter1d(s, 3, axis=2).max(axis=2) - np.median(s[..., :2], axis=2)
    roi = (frames.min(axis=2) >= VESSEL_LEVEL) & (enh >= ROI_ENH)
    return binary_opening(roi, iterations=2)


def summarise(maps, roi, ref_tmax=None):
    r = {f"{k}_median": float(np.nanmedian(maps[k][roi])) for k in ("CBV", "CBF", "MTT")}
    t = maps["Tmax"][roi]
    r.update(Tmax_median=float(np.median(t)), Tmax_mean=float(np.mean(t)))
    r["tmax_changed_frac"] = float(np.mean(t != ref_tmax)) if ref_tmax is not None else 0.0
    return r


def peak_matched(donor, target):
    """Donor shape, target amplitude and baseline."""
    yd, bd, _, _ = normalise(donor)
    _, bt, it, _ = normalise(target)
    return bt + yd * (target[it] - bt)


def compute(out, acq, trust, limit=0):
    pairs = acq[acq.plane.isin(["A", "B"]) & ~acq.is_duplicate.astype(bool)].groupby("ab_pair_id")
    rows = []
    pair_list = [(k, g) for k, g in pairs if set(g.plane) == {"A", "B"}]
    if limit:
        pair_list = pair_list[:limit]
    for i, (pair, g) in enumerate(pair_list, 1):
        print(f"[{i}/{len(pair_list)}] {pair}", flush=True)
        run = {pl: g[g.plane == pl].index[0] for pl in ("A", "B")}
        frames = {pl: load_frames(acq.loc[run[pl], "source"]) for pl in ("A", "B")}
        n = min(f.shape[2] for f in frames.values())
        frames = {pl: f[:, :, :n] for pl, f in frames.items()}
        aifs = {}
        for pl in ("A", "B"):
            d = np.load(out / "npz" / f"{run[pl]}.npz")
            aifs[pl] = {v: sample_aif(frames[pl], d[f"mask_{v}"]) for v in ("ica_top", "ica_full_model", "ica_full_grown")}
        for pl, other in (("A", "B"), ("B", "A")):
            roi = parenchyma_roi(frames[pl])
            own = aifs[pl]
            versions = {"ref": own["ica_top"], "model": own["ica_full_model"], "grown": own["ica_full_grown"],
                        "cross": aifs[other]["ica_top"],
                        "cross_peak": peak_matched(aifs[other]["ica_top"], own["ica_top"])}
            ref_maps = perfusion(frames[pl], versions["ref"])
            # perfDSA clips CBV <= 0 to 0 in many enhancing pixels; analyse the pixels that are perfused with the
            # reference AIF, the same set for every swap of this run
            roi_px, roi_pos_frac = int(roi.sum()), float((ref_maps["CBV"][roi] > 0).mean())
            roi = roi & (ref_maps["CBV"] > 0)
            for swap, aif in versions.items():
                maps = ref_maps if swap == "ref" else perfusion(frames[pl], aif)
                donor_run = run[other] if swap.startswith("cross") else run[pl]
                rows.append(dict(ab_pair_id=pair, run_id=run[pl], plane=pl, patient_id=acq.loc[run[pl], "patient_id"],
                                 injection_site=acq.loc[run[pl], "injection_site"], swap=swap,
                                 donor_run=donor_run, donor_variant=DONOR[swap], roi_px=roi_px,
                                 roi_cbv_pos_frac=roi_pos_frac, analysis_px=int(roi.sum()),
                                 cbv_zero_frac=float((maps["CBV"][roi] <= 0).mean()),
                                 ref_trusted=trust.get((run[pl], "ica_top")) == "trusted",
                                 donor_trusted=trust.get((donor_run, DONOR[swap])) == "trusted",
                                 aif_arrival_s=arrival(aif), **summarise(maps, roi, ref_maps["Tmax"][roi])))
        if i <= 2:
            save_roi_example(out, run, frames, i)
    return pd.DataFrame(rows)


def arrival(aif, frac=0.15):
    yn, _, i_pk, _ = normalise(aif)
    above = np.nonzero(yn[:i_pk + 1] >= frac)[0]
    if not above.size or above[0] == 0:
        return 0.0
    j = above[0]
    return float(np.interp(frac, yn[j - 1:j + 1], [(j - 1) * DT, j * DT]))


ROI_EXAMPLES = []


def save_roi_example(out, run, frames, i):
    ROI_EXAMPLES.append({pl: (frames[pl].min(axis=2), parenchyma_roi(frames[pl]), run[pl]) for pl in ("A", "B")})


def plot_roi_examples(path):
    if not ROI_EXAMPLES:
        return
    fig, axes = plt.subplots(len(ROI_EXAMPLES), 2, figsize=(9, 4.5 * len(ROI_EXAMPLES)), squeeze=False)
    for row, ex in zip(axes, ROI_EXAMPLES):
        for ax, pl in zip(row, ("A", "B")):
            minip, roi, r = ex[pl]
            ax.imshow(minip, cmap="gray")
            ax.imshow(np.ma.masked_where(~roi, roi), cmap="autumn", alpha=0.35)
            ax.set_title(f"{r.replace('PRAISE_BATCH_2_', 'B2_').replace('PRAISE_', '')}  ROI {roi.mean():.0%}", fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Analysis
# ----------------------------------------------------------------------------
def wide(vals):
    """Per run: reference and swapped values side by side, valid swaps only."""
    ref = vals[vals.swap == "ref"].set_index("run_id")
    rows = []
    for swap in SWAPS:
        s = vals[(vals.swap == swap) & vals.ref_trusted & vals.donor_trusted].set_index("run_id")
        for run_id, r in s.iterrows():
            b = ref.loc[run_id]
            # collapsed: the swapped map is zero in more than half of the region (median 0), log ratio undefined
            collapsed = bool(r.CBV_median <= 0 or r.MTT_median <= 0)
            row = dict(run_id=run_id, swap=swap, plane=r.plane, patient_id=r.patient_id, ab_pair_id=r.ab_pair_id,
                       injection_site=r.injection_site, tmax_changed_frac=r.tmax_changed_frac,
                       d_arrival=b.aif_arrival_s - r.aif_arrival_s, collapsed=collapsed)
            for k in ("CBV", "CBF", "MTT"):
                row[f"{k}_ref"], row[f"{k}_swap"] = b[f"{k}_median"], r[f"{k}_median"]
                row[f"d_{k}"] = np.nan if collapsed else np.log(r[f"{k}_median"] / b[f"{k}_median"])
            row["Tmax_ref"], row["Tmax_swap"] = b.Tmax_mean, r.Tmax_mean
            row["d_Tmax"] = r.Tmax_mean - b.Tmax_mean
            rows.append(row)
    return pd.DataFrame(rows)


def fmt(v, k):
    return f"{v:+.3f} s" if k == "Tmax" else f"{np.expm1(v):+.1%}"


def change(w):
    """Per swap and plane (the cross-plane swaps go in opposite directions in A and B tissue, so pooling would
    cancel them). Collapsed maps are excluded from the log statistics and counted."""
    rows = []
    for swap in SWAPS:
        for plane in ("A", "B"):
            s = w[(w.swap == swap) & (w.plane == plane)]
            for k in PARAMS:
                d = s[f"d_{k}"].dropna()
                if len(d) < 8:
                    continue
                test = stats.wilcoxon(d) if (d != 0).sum() > 5 else None
                sd = d.std(ddof=1)
                rows.append(dict(swap=swap, plane=plane, param=k, n=len(d), collapsed=int(s.collapsed.sum()),
                                 median=d.median(), q1=d.quantile(0.25), q3=d.quantile(0.75), mean=d.mean(), sd=sd,
                                 loa_low=d.mean() - 1.96 * sd, loa_high=d.mean() + 1.96 * sd,
                                 np_low=d.quantile(0.025), np_high=d.quantile(0.975),
                                 abs_median=d.abs().median(), p=test.pvalue if test else np.nan))
    t = pd.DataFrame(rows)
    t["p_holm"] = np.nan
    for _, idx in t.groupby(["swap", "plane"]).groups.items():
        t.loc[idx, "p_holm"] = holm(t.loc[idx, "p"])
    return t


def within_between(vals, w):
    """RMS of the swap change vs the between-patient SD of the reference value (per plane, log scale)."""
    ref = vals[(vals.swap == "ref") & vals.ref_trusted]
    rows = []
    for swap in SWAPS:
        for plane in ("A", "B"):
            s = w[(w.swap == swap) & (w.plane == plane) & ~w.collapsed]
            r = ref[ref.plane == plane]
            for k in PARAMS:
                if k == "Tmax":
                    between = r.Tmax_mean.std(ddof=1)
                    within = np.sqrt(np.mean(s.d_Tmax ** 2))
                else:
                    between = np.log(r[f"{k}_median"]).std(ddof=1)
                    within = np.sqrt(np.nanmean(s[f"d_{k}"] ** 2))
                rows.append(dict(swap=swap, plane=plane, param=k, n_swap=len(s), n_between=len(r),
                                 within_rms=within, between_sd=between, ratio=within / between))
    return pd.DataFrame(rows)


def rank_stability(w, vals):
    rows = []
    for swap in SWAPS:
        for plane in ("A", "B"):
            s = w[(w.swap == swap) & (w.plane == plane)]
            for k in PARAMS:
                r = stats.spearmanr(s[f"{k}_ref"], s[f"{k}_swap"])
                rows.append(dict(comparison=f"{swap} vs ref", plane=plane, param=k, n=len(s), rho=r.statistic, p=r.pvalue))
    # same injection, own maps of the two views (the clinical situation: which view was used)
    ref = vals[(vals.swap == "ref") & vals.ref_trusted]
    for k in PARAMS:
        col = "Tmax_mean" if k == "Tmax" else f"{k}_median"
        piv = ref.pivot_table(index="ab_pair_id", columns="plane", values=col).dropna()
        r = stats.spearmanr(piv.A, piv.B)
        rows.append(dict(comparison="plane A map vs plane B map", plane="A/B", param=k, n=len(piv), rho=r.statistic, p=r.pvalue))
    return pd.DataFrame(rows)


def tmax_shift(w):
    s = w[w.swap == "grown"]
    rows = []
    for plane in ("A", "B", "all"):
        q = s if plane == "all" else s[s.plane == plane]
        r = stats.spearmanr(q.d_arrival, q.d_Tmax)
        rows.append(dict(plane=plane, n=len(q), arrival_top_minus_grown_s=q.d_arrival.median(),
                         tmax_shift_mean_s=q.d_Tmax.median(), tmax_shift_iqr=f"{q.d_Tmax.quantile(0.25):+.3f} .. {q.d_Tmax.quantile(0.75):+.3f}",
                         pixels_changed=q.tmax_changed_frac.median(), rho_arrival_vs_shift=r.statistic, p=r.pvalue,
                         p_wilcoxon=stats.wilcoxon(q.d_Tmax).pvalue if (q.d_Tmax != 0).sum() > 5 else np.nan))
    return pd.DataFrame(rows)


def decomposition(vals):
    """B map vs A map of the same injection = AIF part x tissue part (log scale)."""
    v = vals.set_index(["run_id", "swap"])
    rows = []
    for pair, g in vals[vals.swap == "ref"].groupby("ab_pair_id"):
        if set(g.plane) != {"A", "B"} or not g.ref_trusted.all():
            continue
        ra, rb = g[g.plane == "A"].run_id.iloc[0], g[g.plane == "B"].run_id.iloc[0]
        row = dict(ab_pair_id=pair)
        for k in ("CBV", "CBF", "MTT"):
            a_own = v.loc[(ra, "ref"), f"{k}_median"]
            b_own = v.loc[(rb, "ref"), f"{k}_median"]
            b_with_a = v.loc[(rb, "cross"), f"{k}_median"]
            row[f"{k}_total"] = np.log(b_own / a_own)
            row[f"{k}_aif"] = np.log(b_own / b_with_a)
            row[f"{k}_tissue"] = np.log(b_with_a / a_own)
        rows.append(row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figures, table, report
# ----------------------------------------------------------------------------
def plot_change(t, wb, path):
    """Top: change per swap (median, IQR) for A tissue and B tissue. Bottom: within / between ratio."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    x = np.arange(len(PARAMS))
    width = 0.2
    conv = {k: (lambda v: np.expm1(v) * 100) for k in ("CBV", "CBF", "MTT")} | {"Tmax": lambda v: v * 100}
    for col, plane in enumerate(("A", "B")):
        ax = axes[0, col]
        for j, swap in enumerate(SWAPS):
            r = t[(t.swap == swap) & (t.plane == plane)].set_index("param").reindex(PARAMS)
            med = [conv[k](r.loc[k, "median"]) for k in PARAMS]
            lo = [conv[k](r.loc[k, "q1"]) for k in PARAMS]
            hi = [conv[k](r.loc[k, "q3"]) for k in PARAMS]
            xx = x + (j - 1.5) * width
            ax.bar(xx, med, width, color=SWAP_COL[swap], label=SWAPS[swap])
            ax.errorbar(xx, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor="k", lw=0.8)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(x, ["CBV (%)", "CBF (%)", "MTT (%)", "Tmax (0.01 s)"])
        ax.set_ylabel("change vs own ica_top AIF (median, IQR)")
        ax.set_title(f"Plane {plane} tissue: change with another AIF")
        ax = axes[1, col]
        for j, swap in enumerate(SWAPS):
            r = wb[(wb.swap == swap) & (wb.plane == plane)].set_index("param").ratio.reindex(PARAMS)
            ax.bar(x + (j - 1.5) * width, r, width, color=SWAP_COL[swap])
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xticks(x, PARAMS)
        ax.set_ylabel("within-patient change / between-patient SD")
        ax.set_title(f"Plane {plane}: AIF effect vs patient differences (< 1: smaller)")
    axes[0, 0].legend(fontsize=8, frameon=False)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_ba(w, t, swap, path):
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    s = w[(w.swap == swap) & ~w.collapsed]
    for ax, k in zip(axes, PARAMS):
        a, b = s[f"{k}_ref"], s[f"{k}_swap"]
        mean = (a + b) / 2 if k == "Tmax" else np.sqrt(a * b)
        for pl, c in (("A", "#2a78d6"), ("B", "#eb6834")):
            m = s.plane == pl
            ax.scatter(mean[m], s.loc[m, f"d_{k}"], s=10, color=c, label=f"plane {pl} tissue")
            r = t[(t.swap == swap) & (t.param == k) & (t.plane == pl)].iloc[0]
            ax.axhline(r["mean"], color=c, lw=1)
            for v in (r.loa_low, r.loa_high):
                ax.axhline(v, color=c, lw=0.8, ls="--")
        ax.axhline(0, color="0.6", lw=0.5)
        if k != "Tmax":
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{np.expm1(v):+.0%}"))
        ax.set_title(f"{k}{' (s)' if k == 'Tmax' else ''}", fontsize=10)
        ax.set_xlabel("mean of ref and swap")
    axes[0].set_ylabel("swap vs ref")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Bland-Altman, {SWAPS[swap]} vs own ica_top AIF (line: bias, dashed: 95 % LoA)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_decomposition(dec, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    params = ["CBV", "CBF", "MTT"]
    x = np.arange(len(params))
    parts = [("total", "B map / A map (own AIFs)", "#7b3fa0"), ("aif", "AIF part (B-AIF vs A-AIF on B tissue)", "#eb6834"),
             ("tissue", "tissue part (same A-AIF, B vs A tissue)", "#2a78d6")]
    for j, (part, label, col) in enumerate(parts):
        med = [np.exp(dec[f"{k}_{part}"].median()) for k in params]
        lo = [np.exp(dec[f"{k}_{part}"].quantile(0.25)) for k in params]
        hi = [np.exp(dec[f"{k}_{part}"].quantile(0.75)) for k in params]
        xx = x + (j - 1) * 0.26
        ax.bar(xx, np.subtract(med, 1), 0.26, bottom=1, color=col, label=label)
        ax.errorbar(xx, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="none", ecolor="k", lw=0.8)
    ax.axhline(1, color="k", lw=0.8)
    ax.set_xticks(x, params)
    ax.set_ylabel("ratio (median, IQR)")
    ax.set_title(f"Where does the A/B map difference come from? ({len(dec)} injections)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def tex_table(t, wb, rank, path):
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Effect of the AIF on the perfusion maps with the tissue curves held fixed, for plane A and plane B "
             "tissue. Change: median (IQR) of the swapped map relative to the map with the run's own \\texttt{ica\\_top} "
             "AIF over a fixed parenchymal region (Tmax: shift of the region mean, s); $^\\ast p_\\mathrm{Holm} < 0.05$ "
             "(Wilcoxon, four parameters per swap and plane). Ratio: within-patient change (RMS) / between-patient SD. "
             "$\\rho$: Spearman rank correlation of the patients between reference and swap.}",
             "\\label{tab:sq4}", "\\footnotesize", "\\begin{tabular}{ll rrr rrr}", "\\hline",
             " & & \\multicolumn{3}{c}{Plane A tissue} & \\multicolumn{3}{c}{Plane B tissue} \\\\",
             "AIF swap & Param. & Change (IQR) & Ratio & $\\rho$ & Change (IQR) & Ratio & $\\rho$ \\\\", "\\hline"]
    for swap in SWAPS:
        for j, k in enumerate(PARAMS):
            cells = []
            for plane in ("A", "B"):
                r = t[(t.swap == swap) & (t.param == k) & (t.plane == plane)].iloc[0]
                star = "$^\\ast$" if r.p_holm < 0.05 else ""
                if k == "Tmax":
                    eff = f"{r['median']:+.2f} ({r.q1:+.2f}, {r.q3:+.2f}){star}"
                else:
                    eff = (f"{np.expm1(r['median']):+.0%} ({np.expm1(r.q1):+.0%}, {np.expm1(r.q3):+.0%})"
                           .replace("%", "\\%") + star)
                ratio = wb[(wb.swap == swap) & (wb.param == k) & (wb.plane == plane)].ratio.iloc[0]
                rho = rank[(rank.comparison == f"{swap} vs ref") & (rank.param == k) & (rank.plane == plane)].rho.iloc[0]
                cells += [eff, f"{ratio:.2f}", f"{rho:.2f}"]
            name = SWAPS[swap] if j == 0 else ""
            lines.append(f"{name} & {k} & " + " & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(out_dir, vals, w, t, wb, rank, tm, dec, command):
    n_runs = vals.run_id.nunique()
    roi = vals[vals.swap == "ref"].roi_px / 1024 ** 2
    tt = t.copy()
    for c in ["median", "q1", "q3", "np_low", "np_high", "loa_low", "loa_high"]:
        tt[c] = [fmt(v, k) for v, k in zip(tt[c], tt.param)]
    tt["p_holm"] = tt.p_holm.map(fmt_p)
    wbs = wb.pivot_table(index=["swap", "param"], columns="plane", values="ratio").round(2).reset_index()
    rk = rank.pivot_table(index=["comparison", "param"], columns="plane", values="rho").round(3).reset_index()
    dd = pd.DataFrame([{"parameter": k, **{part: f"{np.exp(dec[f'{k}_{part}'].median()):.2f} "
                                              f"({np.exp(dec[f'{k}_{part}'].quantile(0.25)):.2f}-"
                                              f"{np.exp(dec[f'{k}_{part}'].quantile(0.75)):.2f})"
                                              for part in ("total", "aif", "tissue")}}
                       for k in ("CBV", "CBF", "MTT")])
    txt = f"""# SQ4 — AIF swap, tissue fixed ({out_dir.parent.name})

Generated by `scripts/sq4_aif_swap.py`; do not edit by hand, re-run instead.

```bash
{command}
```

{n_runs} runs ({vals.ab_pair_id.nunique()} A/B pairs). Fixed parenchymal ROI per run: median {roi.median():.0%} of the
image (IQR {roi.quantile(0.25):.0%}-{roi.quantile(0.75):.0%}); examples in `roi_examples.png`. A swap is used when both
the reference (own ica_top) and the donor AIF are trusted.

## 1-2. Change per swap and Bland-Altman

Log ratio swap / reference as %, Tmax as shift of the ROI mean (s). Wilcoxon, Holm per swap. `change.png`,
`ba_<swap>.png`.

{md_table(tt[["swap", "plane", "param", "n", "collapsed", "median", "q1", "q3", "loa_low", "loa_high", "np_low", "np_high", "p_holm"]])}

## 3. Within-patient change vs between-patient spread

RMS of the log change (incl. bias) / between-patient SD of the log reference value, per plane. < 1: the AIF choice
moves a patient less than patients differ.

{md_table(wbs)}

## 4. Rank stability (Spearman rho of the patients)

{md_table(rk)}

## 5. Does the earlier grown AIF shift Tmax?

{md_table(tm.round(3))}

## 6. A vs B maps of the same injection: AIF part and tissue part

Median ratio (IQR), {len(dec)} injections with both ica_top AIFs trusted. total = B map / A map (each with its own
AIF); aif = B map with B-AIF / B map with A-AIF; tissue = B map with A-AIF / A map. total = aif x tissue.
`decomposition.png`.

{md_table(dd)}
"""
    (out_dir / "sq4_report.md").write_text(txt, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--report-only", action="store_true", help="Analyse the saved swap_values.csv.")
    p.add_argument("--limit", type=int, default=0, help="Only the first N A/B pairs (testing).")
    args = p.parse_args()
    out = args.out
    out_dir = out / "sq4_aif_swap"
    out_dir.mkdir(parents=True, exist_ok=True)
    acq = pd.read_csv(out / "acquisition_table.csv").set_index("run_id")
    tax = pd.read_csv(out / "aif_qc" / "taxonomy" / "taxonomy_labels.csv")
    trust = tax.set_index(["run_id", "variant"]).trust.to_dict()

    if args.report_only:
        vals = pd.read_csv(out_dir / "swap_values.csv")
    else:
        vals = compute(out, acq, trust, args.limit)
        if args.limit:
            print(vals.round(3).to_string())
            return
        vals.to_csv(out_dir / "swap_values.csv", index=False)
        plot_roi_examples(out_dir / "roi_examples.png")

    w = wide(vals)
    t = change(w)
    t.to_csv(out_dir / "sq4_change.csv", index=False)
    wb = within_between(vals, w)
    wb.to_csv(out_dir / "sq4_within_between.csv", index=False)
    rank = rank_stability(w, vals)
    rank.to_csv(out_dir / "sq4_rank.csv", index=False)
    tm = tmax_shift(w)
    tm.to_csv(out_dir / "sq4_tmax.csv", index=False)
    dec = decomposition(vals)
    dec.to_csv(out_dir / "sq4_decomposition.csv", index=False)
    plot_change(t, wb, out_dir / "change.png")
    for swap in SWAPS:
        plot_ba(w, t, swap, out_dir / f"ba_{swap}.png")
    plot_decomposition(dec, out_dir / "decomposition.png")
    tex_table(t, wb, rank, out_dir / "table_sq4.tex")
    write_report(out_dir, vals, w, t, wb, rank, tm, dec, " ".join(["python"] + sys.argv))
    print(open(out_dir / "sq4_report.md", encoding="utf-8").read())


if __name__ == "__main__":
    main()
