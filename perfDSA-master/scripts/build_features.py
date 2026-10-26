"""
Build the per-run x per-AIF-variant feature table for the statistics (SQ1-SQ4).

Input  (in <out>, a postprocess_combined.py + select_runs.py result such as results/combined_v1):
    acquisition_table.csv, selection.csv, perfusion_summary.csv, aif_curves.csv, npz/<run_id>.npz
Output (in <out>):
    features.csv            long, one row per run_id x variant (ica_top, ica_full_model, ica_full_grown)
    features_paired.csv     wide, one row per patient x variant, <feature>_A / <feature>_B for every
                            numeric/boolean feature, only runs with in_paired_set (SQ2: plane A vs B)
    features_full_vs_top.csv  wide, one row per run, <feature>_top / _model / _grown for every
                            variant-dependent feature, run-level covariates once (SQ1: full vs top)
    features_summary.csv    median [min-max] and n per numeric feature per variant (all runs)

Identifiers / covariates (run level, identical for the three variants)
    from acquisition_table.csv: dataset, patient_id, plane (A = frontal, primary angle ~0 deg (up to
    ~35 deg); B = lateral, ~-90 deg), ab_pair_id, injection_site, frame_rate_regime,
    frame_interval_ms_median, duration_s, kvp, magnification, primary_angle_deg,
    secondary_angle_deg, is_duplicate
    from selection.csv: evt_phase, included (the one-plane-per-patient selection)
    in_paired_set   injection site is an ICA (R ICA / L ICA / ICA), the run is not a duplicate, and
                    both planes A and B of its ab_pair_id satisfy the same (74 runs / 37 patients)

Perfusion (from perfusion_summary.csv, per variant)
    mask_px, mask_components, perfused_px, CBV/CBF/MTT/Tmax _median and _mean

AIF features (per variant, from aif_curves.csv, sampling == resampled_250ms unless stated; the
resampled curves share a 250 ms grid starting at t = 0 = the first frame after the mask frame)
    c(t) is the AIF, i_pk = argmax c (global maximum), t the sample times.
    aif_baseline        b = median of the first 2 samples of c
    aif_peak            c(i_pk) - b  (baseline-corrected peak height; NaN-producing cases below)
    aif_ttp_s           t(i_pk), time to peak from t = 0 (on the 250 ms grid, so 0.25 s resolution)
    aif_arrival_s       bolus arrival: first time before/at the peak at which c rises above
                        b + f * peak, linear interpolation between the bracketing samples;
                        f = --arrival-frac (default 0.2). Identical algorithm to
                        postprocess_combined.arrival_time (used for aif_arrival_s_ica_top).
                        If the first sample is already above the level, t(0) is returned and
                        aif_arrival_at_start is set.
    aif_arrival_10_s    the same at f = 0.10 (always reported; aif_arrival_90_s at f = 0.90)
    aif_arrival_native_s / aif_arrival_native_10_s
                        the same on the native (non-resampled, variable frame interval) curve,
                        sampling == native
    aif_rise_s          aif_ttp_s - aif_arrival_s
    aif_rise_10_90_s    aif_arrival_90_s - aif_arrival_10_s
    aif_fwhm_s          full width at half maximum: right minus left crossing of b + peak/2,
                        both linearly interpolated, the left one searched before, the right one
                        after the peak. If c does not fall back below half maximum before the
                        end of the acquisition, aif_fwhm_s is NaN and aif_fwhm_censored is set
                        (the width is then > end - left crossing).
    aif_auc             trapezoidal integral of c - b over the whole curve (t = 0 to the last
                        sample), not clipped, in AIF units x s
    Shape-only (amplitude removed). Curve / peak: FWHM, rise times and arrival times are
    invariant to scaling, so their normalised versions equal the columns above and are not
    repeated. The remaining shape-only features:
    aif_auc_norm_s      AUC of (c - b) / peak = aif_auc / aif_peak (s; the "effective width")
    aif_peak_norm       peak of (c - b) / AUC = aif_peak / aif_auc (1/s)
    aif_centroid_s      first moment of the AUC-normalised curve, using max(c - b, 0)
    aif_sd_s            square root of its second central moment (temporal spread)
    aif_n_peaks         number of peaks of (c - b) / peak found by scipy.signal.find_peaks with
                        prominence >= --peak-prominence (default 0.1, i.e. 10 % of the peak);
                        the global maximum counts when it is an interior sample, plus 1 if the
                        global maximum is the last sample (find_peaks ignores edge samples)
    aif_tail_frac       (c(last) - b) / peak, the remaining enhancement at the end of the run
    aif_truncated       aif_tail_frac > --tail-threshold (default 0.3): the bolus has not washed
                        out within the acquisition (truncated curve, venous/recirculation tail)
    aif_late_rise       largest re-rise after the peak, max over k > i_pk of
                        (c(k) - min c(i_pk..k)) / peak, stored as aif_late_rise_amp; the flag
                        aif_late_rise is aif_late_rise_amp > --late-rise-threshold (default 0.15),
                        i.e. after falling the curve rises again by > 15 % of the peak
                        (second pass / venous contamination)
    aif_peak_at_end     the global maximum is the last sample (curve still rising at the end)
    NaN rules: all AIF features are NaN when the curve is all-NaN (no mask) or peak <= 0 (the
    curve never rises above its baseline). Arrival/rise columns are additionally NaN when the
    crossing does not exist before the peak, which cannot happen for f in (0, 1] once peak > 0.

Saturation (per variant, from npz/<run_id>.npz: minip and mask_<variant>, 0..255 scale where
low = more contrast)
    sat_frac            fraction of mask pixels with MINIP <= --sat-level (default 2)
    minip_min           minimum MINIP value inside the mask
    minip_median        median MINIP value inside the mask

Temporal agreement (per variant, resampled curves, both on the same 250 ms grid from t = 0;
truncated to the shorter length)
    r_vs_top            Pearson r with the ica_top curve of the same run (1 for ica_top itself)
    r_ab                Pearson r with the same variant of the A/B partner run: same ab_pair_id,
                        other plane, same is_duplicate value (duplicates pair with duplicates);
                        NaN when there is no partner

Example:
    python scripts/build_features.py results/combined_v1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
SHORT = {"ica_top": "top", "ica_full_model": "model", "ica_full_grown": "grown"}
ACQ_COLS = ["dataset", "patient_id", "plane", "ab_pair_id", "injection_site", "frame_rate_regime",
            "frame_interval_ms_median", "duration_s", "kvp", "magnification", "primary_angle_deg",
            "secondary_angle_deg", "is_duplicate"]
PERF_COLS = ["mask_px", "mask_components", "perfused_px"] + [f"{m}_{s}" for m in ("CBV", "CBF", "MTT", "Tmax")
                                                              for s in ("median", "mean")]


# ----------------------------------------------------------------------------
# Curve features
# ----------------------------------------------------------------------------
def crossing_up(t, c, level, i_end):
    """First upward crossing of `level` in c[:i_end + 1], linearly interpolated (t[0] if c[0] is above)."""
    above = np.nonzero(c[:i_end + 1] >= level)[0]
    if not above.size:
        return np.nan, False
    i = above[0]
    if i == 0:
        return float(t[0]), True
    return float(np.interp(level, [c[i - 1], c[i]], [t[i - 1], t[i]])), False


def crossing_down(t, c, level, i_start):
    """First downward crossing of `level` after index i_start, linearly interpolated (NaN if none)."""
    below = np.nonzero(c[i_start:] < level)[0]
    if not below.size:
        return np.nan
    i = i_start + below[0]
    return float(np.interp(level, [c[i], c[i - 1]], [t[i], t[i - 1]]))


def arrival(t, c, frac):
    """Arrival at `frac` of the rise: (time, arrival_at_start). Same algorithm as postprocess_combined."""
    t, c = np.asarray(t, float), np.asarray(c, float)
    if not np.isfinite(c).any():
        return np.nan, False
    base = np.nanmedian(c[:2])
    i_pk = int(np.nanargmax(c))
    if not c[i_pk] > base:
        return np.nan, False
    return crossing_up(t, c, base + frac * (c[i_pk] - base), i_pk)


def curve_features(t, c, args):
    t, c = np.asarray(t, float), np.asarray(c, float)
    keys = ["aif_baseline", "aif_peak", "aif_ttp_s", "aif_arrival_s", "aif_arrival_at_start", "aif_arrival_10_s",
            "aif_arrival_90_s", "aif_rise_s", "aif_rise_10_90_s", "aif_fwhm_s", "aif_fwhm_censored", "aif_auc",
            "aif_auc_norm_s", "aif_peak_norm", "aif_centroid_s", "aif_sd_s", "aif_n_peaks", "aif_tail_frac",
            "aif_truncated", "aif_late_rise_amp", "aif_late_rise", "aif_peak_at_end"]
    f = dict.fromkeys(keys, np.nan)
    if not c.size or not np.isfinite(c).all():
        return f
    base = float(np.median(c[:2]))
    i_pk = int(np.argmax(c))
    peak = float(c[i_pk] - base)
    f["aif_baseline"] = base
    if not peak > 0:
        return f
    y = c - base
    f.update(aif_peak=peak, aif_ttp_s=float(t[i_pk]))
    f["aif_arrival_s"], f["aif_arrival_at_start"] = crossing_up(t, y, args.arrival_frac * peak, i_pk)
    f["aif_arrival_10_s"], _ = crossing_up(t, y, 0.1 * peak, i_pk)
    f["aif_arrival_90_s"], _ = crossing_up(t, y, 0.9 * peak, i_pk)
    f["aif_rise_s"] = f["aif_ttp_s"] - f["aif_arrival_s"]
    f["aif_rise_10_90_s"] = f["aif_arrival_90_s"] - f["aif_arrival_10_s"]
    left, _ = crossing_up(t, y, 0.5 * peak, i_pk)
    right = crossing_down(t, y, 0.5 * peak, i_pk)
    f["aif_fwhm_s"] = right - left
    f["aif_fwhm_censored"] = bool(np.isnan(right))
    auc = float(np.trapezoid(y, t))
    f["aif_auc"] = auc
    f["aif_auc_norm_s"] = auc / peak
    f["aif_peak_norm"] = peak / auc if auc > 0 else np.nan
    w = np.clip(y, 0, None)
    area = np.trapezoid(w, t)
    if area > 0:
        mu = np.trapezoid(w * t, t) / area
        f["aif_centroid_s"] = float(mu)
        f["aif_sd_s"] = float(np.sqrt(np.trapezoid(w * (t - mu) ** 2, t) / area))
    yn = y / peak
    f["aif_n_peaks"] = len(find_peaks(yn, prominence=args.peak_prominence)[0]) + int(i_pk == len(c) - 1)
    f["aif_tail_frac"] = float(yn[-1])
    f["aif_truncated"] = bool(yn[-1] > args.tail_threshold)
    after = yn[i_pk:]
    f["aif_late_rise_amp"] = float(np.max(after - np.minimum.accumulate(after)))
    f["aif_late_rise"] = bool(f["aif_late_rise_amp"] > args.late_rise_threshold)
    f["aif_peak_at_end"] = bool(i_pk == len(c) - 1)
    return f


def pearson(a, b):
    if a is None or b is None:
        return np.nan
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n], float), np.asarray(b[:n], float)
    if n < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()) or a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def build(out, args):
    acq = pd.read_csv(out / "acquisition_table.csv")
    sel = pd.read_csv(out / "selection.csv")[["run_id", "evt_phase", "included"]]
    perf = pd.read_csv(out / "perfusion_summary.csv")[["run_id", "variant"] + PERF_COLS]
    curves = pd.read_csv(out / "aif_curves.csv")

    runs = acq[["run_id"] + ACQ_COLS].merge(sel, on="run_id", how="left")
    is_ica = runs.injection_site.astype(str).str.contains("ICA") & ~runs.is_duplicate.astype(bool)
    planes = runs[is_ica].groupby("ab_pair_id").plane.agg(lambda p: {"A", "B"} <= set(p))
    runs["in_paired_set"] = is_ica & runs.ab_pair_id.map(planes).fillna(False).astype(bool)

    curve = {k: (g.time_s.to_numpy(), g.aif.to_numpy())
             for k, g in curves.sort_values("time_s").groupby(["run_id", "variant", "sampling"])}
    partner = {}
    for _, g in runs.groupby(["ab_pair_id", "is_duplicate"]):
        for _, r in g.iterrows():
            other = g[g.plane != r.plane]
            if len(other) == 1:
                partner[r.run_id] = other.run_id.iloc[0]

    rows = []
    for _, r in runs.iterrows():
        npz = out / "npz" / f"{r.run_id}.npz"
        d = np.load(npz) if npz.exists() else None
        for v in VARIANTS:
            row = {"run_id": r.run_id, "variant": v}
            t, c = curve.get((r.run_id, v, "resampled_250ms"), (np.array([]), np.array([])))
            row.update(curve_features(t, c, args))
            tn, cn = curve.get((r.run_id, v, "native"), (np.array([]), np.array([])))
            row["aif_arrival_native_s"], _ = arrival(tn, cn, args.arrival_frac) if cn.size else (np.nan, False)
            row["aif_arrival_native_10_s"], _ = arrival(tn, cn, 0.1) if cn.size else (np.nan, False)
            row["r_vs_top"] = pearson(c if c.size else None, curve.get((r.run_id, "ica_top", "resampled_250ms"),
                                                                       (None, None))[1])
            p = partner.get(r.run_id)
            row["r_ab"] = pearson(c if c.size else None,
                                  curve.get((p, v, "resampled_250ms"), (None, None))[1] if p else None)
            if d is not None and f"mask_{v}" in d.files and d[f"mask_{v}"].any():
                vals = d["minip"][d[f"mask_{v}"]]
                row.update(sat_frac=float(np.mean(vals <= args.sat_level)), minip_min=float(vals.min()),
                           minip_median=float(np.median(vals)))
            else:
                row.update(sat_frac=np.nan, minip_min=np.nan, minip_median=np.nan)
            rows.append(row)
    feat = pd.DataFrame(rows)
    feat = runs.merge(feat, on="run_id").merge(perf, on=["run_id", "variant"], how="left")
    first = ["run_id", "variant"] + [c for c in runs.columns if c != "run_id"]
    return feat[first + [c for c in feat.columns if c not in first]], runs


def wide_tables(feat, runs):
    run_cols = [c for c in runs.columns if c != "run_id"]
    vcols = [c for c in feat.columns if c not in run_cols + ["run_id", "variant"]]
    num = [c for c in vcols + run_cols if pd.api.types.is_numeric_dtype(feat[c]) or feat[c].dtype == object
           and feat[c].dropna().isin([True, False]).all()]
    num = [c for c in num if c not in ("in_paired_set",)]

    # paired A vs B: patient x variant
    p = feat[feat.in_paired_set]
    keys = ["dataset", "patient_id", "ab_pair_id", "injection_site", "evt_phase", "variant"]
    w = p.set_index(keys + ["plane"])[[c for c in num if c not in keys]].unstack("plane")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    paired = w.reset_index()
    paired = paired[keys + sorted([c for c in paired.columns if c not in keys], key=lambda c: (
        num.index(c[:-2]) if c[:-2] in num else 999, c[-1]))]
    for pl in "BA":
        paired.insert(len(keys), f"run_id_{pl}", paired.ab_pair_id.map(
            p[(p.plane == pl)].drop_duplicates("ab_pair_id").set_index("ab_pair_id").run_id))

    # full vs top: run x (top / model / grown)
    w = feat.pivot(index="run_id", columns="variant", values=vcols)
    w.columns = [f"{a}_{SHORT[b]}" for a, b in w.columns]
    w = w[[f"{c}_{SHORT[v]}" for c in vcols for v in VARIANTS]]
    fvt = runs.merge(w.reset_index(), on="run_id")
    return paired, fvt


def summary(feat):
    rows = []
    for c in feat.columns:
        if c in ("run_id", "variant") or not pd.api.types.is_numeric_dtype(feat[c]) or feat[c].dtype == bool:
            continue
        for v in VARIANTS:
            s = feat.loc[feat.variant == v, c].astype(float)
            rows.append({"feature": c, "variant": v, "n": int(s.notna().sum()), "n_nan": int(s.isna().sum()),
                         "median": s.median(), "min": s.min(), "max": s.max()})
    for c in [c for c in feat.columns if feat[c].dtype == bool or feat[c].dtype == object and
              feat[c].dropna().isin([True, False]).all() and feat[c].notna().any()]:
        for v in VARIANTS:
            s = feat.loc[feat.variant == v, c].dropna().astype(bool)
            rows.append({"feature": c, "variant": v, "n": len(s), "n_nan": int(feat.loc[feat.variant == v, c].isna()
                                                                                  .sum()),
                         "median": np.nan, "min": np.nan, "max": np.nan, "n_true": int(s.sum())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="Combined result folder, e.g. results/combined_v1")
    p.add_argument("--arrival-frac", type=float, default=0.2, help="Fraction of the rise that defines arrival.")
    p.add_argument("--peak-prominence", type=float, default=0.1, help="find_peaks prominence, fraction of peak.")
    p.add_argument("--tail-threshold", type=float, default=0.3, help="Tail fraction above which a curve is truncated.")
    p.add_argument("--late-rise-threshold", type=float, default=0.15, help="Re-rise after the peak, fraction of peak.")
    p.add_argument("--sat-level", type=float, default=2.0, help="MINIP value (0..255) at or below which a pixel "
                                                                  "counts as saturated.")
    args = p.parse_args()

    feat, runs = build(args.out, args)
    paired, fvt = wide_tables(feat, runs)
    summ = summary(feat)
    feat.to_csv(args.out / "features.csv", index=False)
    paired.to_csv(args.out / "features_paired.csv", index=False)
    fvt.to_csv(args.out / "features_full_vs_top.csv", index=False)
    summ.to_csv(args.out / "features_summary.csv", index=False)

    print(f"features.csv: {len(feat)} rows ({feat.run_id.nunique()} runs x {feat.variant.nunique()} variants)")
    print(f"in_paired_set: {int(runs.in_paired_set.sum())} runs, {runs[runs.in_paired_set].patient_id.nunique()} "
          f"patients")
    print("features_paired.csv rows per variant:", paired.groupby("variant").size().to_dict())
    print(f"features_full_vs_top.csv: {len(fvt)} rows")
    for v in VARIANTS:
        f = feat[feat.variant == v]
        nan = f[f.aif_arrival_s.isna()].run_id.tolist()
        print(f"{v}: NaN aif_arrival_s in {len(nan)} runs {nan}; NaN native arrival in "
              f"{int(f.aif_arrival_native_s.isna().sum())}; arrival at start in {int(f.aif_arrival_at_start.sum())}")


if __name__ == "__main__":
    main()
