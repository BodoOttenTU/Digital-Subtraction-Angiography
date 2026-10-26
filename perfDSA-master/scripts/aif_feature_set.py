"""
Final AIF feature set of a combined-cohort result: every AIF feature per curve, with its QC labels, in
one folder with a data dictionary, ready for the statistics.

The curve features themselves come from build_features.curve_features (the same code that made
features.csv), now with the thresholds from scripts/aif_features_config.json instead of command-line
defaults, plus the features build_features does not have:
    aif_auc_first_pass      integral of c - b from the 10 % arrival to the first frame after the peak where
                            the curve has fallen below first_pass_end_frac of the peak (or the end): the
                            first-pass area without the venous/recirculation tail that aif_auc includes
    aif_first_pass_end_s    end of that window; aif_first_pass_censored when the curve never falls that far
    aif_mtt_bolus_s         centroid of the curve minus the arrival time: mean transit time of the bolus
                            through the ICA, independent of when the run started (aif_centroid_s is from t = 0)
    aif_skewness            third standardised moment of the AUC-normalised curve (shape: > 0 = long tail)
    aif_up_down_ratio       rise time (arrival -> peak) / fall time (peak -> right half-max crossing)

Arrival threshold
    arrival_frac (config) is the fraction of the peak at which the bolus "arrives". arrival_threshold.csv
    checks every candidate fraction two ways: precision (both planes of one injection see the same bolus
    at the same moment, so the spread of arrival_A - arrival_B over the trusted pairs is the measurement
    noise) and noise floor (trusted curves whose 3-sigma baseline noise, 3 / aif_qc SNR, reaches the
    level). The rule: the smallest fraction above the noise of every trusted curve; the script warns if the
    config differs.

Feature groups (column prefix in features_aif.csv; feature_dictionary.csv has unit and definition per column)
    id / acquisition    run_id, variant, patient, plane, pair, injection site, frame regime, kVp, ...
    qc                  qc_category (aif_qc.py), qc_taxonomy, qc_trust (aif_taxonomy.py), qc_included
    timing              arrival (at arrival_frac, 10 %, 90 %), TTP, rise times, FWHM, MTT of the bolus
    amplitude           peak, AUC, first-pass AUC, baseline
    shape_peak          peak-normalised (amplitude removed, c / peak): FWHM, rise times, AUC / peak,
                        up/down ratio, n_peaks, tail fraction, late rise
    shape_auc           AUC-normalised (c / AUC): peak / AUC, centroid, SD, skewness
    perfusion           mask size and CBV/CBF/MTT/Tmax medians (perfusion_summary)

Output: <out>/aif_features/
    features_aif.csv          long, one row per run x variant (all unique runs; filter on qc_trust)
    features_paired_AB.csv    wide <feature>_A / _B, pairs where both planes are trusted (SQ2)
    features_full_vs_top.csv  wide <feature>_top / _model / _grown, runs where all three are trusted (SQ1)
    curves_shape.csv          arrival-aligned curves on a common grid: c / peak and c / AUC (for shape plots)
    feature_dictionary.csv    column, group, unit, shape_only, definition
    arrival_threshold.csv     A/B arrival disagreement per candidate fraction
    config_used.json          the config this run used
    README.md                 what is where, how to load it

Example:
    python scripts/aif_feature_set.py results/combined_v1
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_features import crossing_down, crossing_up, curve_features  # noqa: E402

CONFIG = Path(__file__).resolve().parent / "aif_features_config.json"
VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
SHORT = {"ica_top": "top", "ica_full_model": "model", "ica_full_grown": "grown"}

# column -> (group, unit, shape_only, definition)
DICTIONARY = {
    "run_id": ("id", "", False, "run (one DSA series, one plane)"),
    "variant": ("id", "", False, "AIF mask: ica_top (supraclinoid), ica_full_model, ica_full_grown"),
    "patient_id": ("id", "", False, "patient"),
    "dataset": ("id", "", False, "PRAISE (batch 1) or PRAISE_BATCH_2"),
    "plane": ("id", "", False, "A = frontal, B = lateral"),
    "ab_pair_id": ("id", "", False, "injection; A and B of one injection share it"),
    "injection_site": ("acquisition", "", False, "catheter position"),
    "evt_phase": ("acquisition", "", False, "moment in the procedure"),
    "frame_rate_regime": ("acquisition", "", False, "frame-rate pattern, e.g. 4-2-1 fps"),
    "duration_s": ("acquisition", "s", False, "acquisition length"),
    "kvp": ("acquisition", "kV", False, "tube voltage"),
    "magnification": ("acquisition", "", False, "geometric magnification SID/SOD"),
    "qc_category": ("qc", "", False, "curve-shape QC (aif_qc.py): usable or the failure"),
    "qc_taxonomy": ("qc", "", False, "failure taxonomy (aif_taxonomy.py): ok, wrong_structure, too_small, "
                    "split_segments, projection_overlap, not_segmentation"),
    "qc_trust": ("qc", "", False, "trust flag (aif_taxonomy.py): trusted / review / reject"),
    "qc_included": ("qc", "", False, "the run is in the one-plane-per-patient selection"),
    "aif_baseline": ("amplitude", "AIF units", False, "median of the first 2 samples"),
    "aif_peak": ("amplitude", "AIF units", False, "baseline-corrected maximum"),
    "aif_auc": ("amplitude", "AIF units x s", False, "integral of c - b over the whole run (incl. tail)"),
    "aif_auc_first_pass": ("amplitude", "AIF units x s", False,
                           "integral of c - b from the 10 % arrival to the fall below first_pass_end_frac"),
    "aif_first_pass_end_s": ("timing", "s", False, "end of the first-pass window"),
    "aif_first_pass_censored": ("qc", "", False, "curve never falls below first_pass_end_frac after the peak"),
    "aif_arrival_s": ("timing", "s", True, "arrival: rise through arrival_frac of the peak (config)"),
    "aif_arrival_at_start": ("qc", "", False, "already above arrival_frac at the first sample"),
    "aif_arrival_10_s": ("timing", "s", True, "rise through 10 % of the peak"),
    "aif_arrival_90_s": ("timing", "s", True, "rise through 90 % of the peak"),
    "aif_ttp_s": ("timing", "s", True, "time to peak from the first frame"),
    "aif_rise_s": ("shape_peak", "s", True, "TTP - arrival"),
    "aif_rise_10_90_s": ("shape_peak", "s", True, "arrival_90 - arrival_10"),
    "aif_fwhm_s": ("shape_peak", "s", True, "full width at half maximum"),
    "aif_fwhm_censored": ("qc", "", False, "no fall below half maximum before the end"),
    "aif_mtt_bolus_s": ("timing", "s", True, "centroid - arrival: mean transit time of the bolus"),
    "aif_auc_norm_s": ("shape_peak", "s", True, "AUC / peak: effective width"),
    "aif_up_down_ratio": ("shape_peak", "", True, "rise time / fall time to half maximum"),
    "aif_n_peaks": ("shape_peak", "", True, "peaks with prominence >= peak_prominence"),
    "aif_tail_frac": ("shape_peak", "", True, "(c(last) - b) / peak"),
    "aif_truncated": ("qc", "", False, "tail_frac > tail_threshold"),
    "aif_late_rise_amp": ("shape_peak", "", True, "largest re-rise after the peak / peak"),
    "aif_late_rise": ("qc", "", False, "late_rise_amp > late_rise_threshold"),
    "aif_peak_at_end": ("qc", "", False, "maximum is the last sample"),
    "aif_peak_norm": ("shape_auc", "1/s", True, "peak / AUC: peak of the AUC-normalised curve"),
    "aif_centroid_s": ("shape_auc", "s", True, "first moment of the AUC-normalised curve, from t = 0"),
    "aif_sd_s": ("shape_auc", "s", True, "SD (second central moment) of the AUC-normalised curve"),
    "aif_skewness": ("shape_auc", "", True, "third standardised moment of the AUC-normalised curve"),
    "mask_px": ("perfusion", "px", False, "pixels in the AIF mask"),
    "CBV_median": ("perfusion", "a.u.", False, "median CBV over perfused pixels"),
    "CBF_median": ("perfusion", "a.u.", False, "median CBF over perfused pixels"),
    "MTT_median": ("perfusion", "s", False, "median MTT over perfused pixels"),
    "Tmax_median": ("perfusion", "s", False, "median Tmax over perfused pixels"),
}
ID_COLS = ["run_id", "variant", "patient_id", "dataset", "plane", "ab_pair_id", "injection_site", "evt_phase",
           "frame_rate_regime", "duration_s", "kvp", "magnification"]
QC_COLS = ["qc_category", "qc_taxonomy", "qc_trust", "qc_included"]
PERF_COLS = ["mask_px", "CBV_median", "CBF_median", "MTT_median", "Tmax_median"]


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


# ----------------------------------------------------------------------------
# Features
# ----------------------------------------------------------------------------
def extra_features(t, c, f, cfg):
    """Features build_features.curve_features does not compute (needs its output f)."""
    e = dict(aif_auc_first_pass=np.nan, aif_first_pass_end_s=np.nan, aif_first_pass_censored=np.nan,
             aif_mtt_bolus_s=np.nan, aif_skewness=np.nan, aif_up_down_ratio=np.nan)
    if not np.isfinite(f["aif_peak"]):
        return e
    y = c - f["aif_baseline"]
    i_pk = int(np.argmax(c))
    below = np.nonzero(y[i_pk:] < cfg["first_pass_end_frac"] * f["aif_peak"])[0]
    i_end = i_pk + int(below[0]) if below.size else len(y) - 1
    m = (t >= f["aif_arrival_10_s"]) & (np.arange(len(t)) <= i_end)
    e["aif_auc_first_pass"] = float(np.trapezoid(y[m], t[m])) if m.sum() > 1 else np.nan
    e["aif_first_pass_end_s"] = float(t[i_end])
    e["aif_first_pass_censored"] = not below.size
    e["aif_mtt_bolus_s"] = f["aif_centroid_s"] - f["aif_arrival_s"]
    w = np.clip(y, 0, None)
    area = np.trapezoid(w, t)
    if area > 0 and f["aif_sd_s"] > 0:
        e["aif_skewness"] = float(np.trapezoid(w * ((t - f["aif_centroid_s"]) / f["aif_sd_s"]) ** 3, t) / area)
    right = crossing_down(t, y, 0.5 * f["aif_peak"], i_pk)
    left, _ = crossing_up(t, y, 0.5 * f["aif_peak"], i_pk)
    fall = right - t[i_pk]
    if np.isfinite(fall) and fall > 0:
        e["aif_up_down_ratio"] = float((t[i_pk] - left) / fall)
    return e


def compute(curves, cfg, arrival_frac):
    args = SimpleNamespace(arrival_frac=arrival_frac, peak_prominence=cfg["peak_prominence"],
                           tail_threshold=cfg["tail_threshold"], late_rise_threshold=cfg["late_rise_threshold"])
    rows = []
    for (run_id, variant), g in curves.groupby(["run_id", "variant"], sort=False):
        t, c = g.time_s.to_numpy(float), g.aif.to_numpy(float)
        f = curve_features(t, c, args)
        rows.append(dict(run_id=run_id, variant=variant, **f, **extra_features(t, c, f, cfg)))
    return pd.DataFrame(rows)


def shape_curves(curves, feat, cfg):
    """Arrival-aligned curves on a common grid, peak- and AUC-normalised."""
    lo, hi, step = cfg["shape_grid_s"]
    grid = np.arange(lo, hi + step / 2, step)
    f = feat.set_index(["run_id", "variant"])
    rows = []
    for (run_id, variant), g in curves.groupby(["run_id", "variant"], sort=False):
        r = f.loc[(run_id, variant)]
        if not np.isfinite(r.aif_peak) or not r.aif_auc > 0:
            continue
        t = g.time_s.to_numpy(float) - r.aif_arrival_s
        y = g.aif.to_numpy(float) - r.aif_baseline
        yi = np.interp(grid, t, y, left=np.nan, right=np.nan)
        rows.append(pd.DataFrame(dict(run_id=run_id, variant=variant, t_from_arrival_s=grid,
                                      c_peak_norm=yi / r.aif_peak, c_auc_norm=yi / r.aif_auc)))
    return pd.concat(rows, ignore_index=True)


def arrival_threshold(curves, cfg, base, snr):
    """A/B arrival disagreement (ica_top, both planes trusted) and baseline noise per candidate fraction.
    snr: aif_qc frame SNR of the trusted curves (1 / noise in peak units)."""
    noise3 = 3 / snr
    top = base[(base.variant == "ica_top") & (base.qc_trust == "trusted")]
    pairs = top.groupby("ab_pair_id").filter(lambda g: set(g.plane) == {"A", "B"})
    sub = curves[(curves.variant == "ica_top") & curves.run_id.isin(pairs.run_id)]
    rows = []
    for frac in cfg["arrival_frac_candidates"]:
        f = compute(sub, cfg, frac)[["run_id", "aif_arrival_s", "aif_arrival_at_start"]].merge(
            pairs[["run_id", "plane", "ab_pair_id"]], on="run_id")
        w = f.pivot(index="ab_pair_id", columns="plane", values="aif_arrival_s").dropna()
        d = w.A - w.B
        rows.append(dict(arrival_frac=frac, pairs=len(d), sd_A_minus_B_s=d.std(), mad_A_minus_B_s=median_abs_deviation(d),
                         median_A_minus_B_s=d.median(), arrival_at_start=int(f.aif_arrival_at_start.sum()),
                         trusted_curves_noise_above=int((noise3 >= frac).sum())))
    return pd.DataFrame(rows)


def pick_arrival_frac(arr):
    """Smallest candidate above the 3-sigma baseline noise of every trusted curve."""
    ok = arr[arr.trusted_curves_noise_above == 0]
    return float(ok.arrival_frac.min()) if len(ok) else np.nan


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def md_table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_readme(out_dir, feat, arr, cfg, command, n_pairs, n_fvt):
    n = feat.groupby("variant").qc_trust.value_counts().unstack().reindex(VARIANTS).fillna(0).astype(int)
    best = pick_arrival_frac(arr)
    sd = arr.sd_A_minus_B_s
    groups = pd.Series({k: v[0] for k, v in DICTIONARY.items()}).value_counts()
    txt = f"""# AIF feature set — {out_dir.parent.name}

Generated by `scripts/aif_feature_set.py` with `scripts/aif_features_config.json`; do not edit by hand, re-run instead.

```bash
{command}
```

Config used (`config_used.json`): {", ".join(f"{k} {v}" for k, v in cfg.items() if k != "arrival_frac_candidates")}

## Files

| file | rows | use |
|---|---|---|
| `features_aif.csv` | one per run x variant ({len(feat)}) | everything; filter `qc_trust == "trusted"` |
| `features_paired_AB.csv` | one per injection x variant, both planes trusted ({n_pairs} for ica_top) | SQ2 plane A vs B |
| `features_full_vs_top.csv` | one per run, all three variants trusted ({n_fvt}) | SQ1 full vs top |
| `curves_shape.csv` | run x variant x grid point | arrival-aligned c/peak and c/AUC, for shape plots |
| `feature_dictionary.csv` | one per column | group, unit, shape-only flag, definition |
| `arrival_threshold.csv` | one per candidate fraction | choice of arrival_frac |

Columns are grouped (see `feature_dictionary.csv`): {", ".join(f"{g} ({k})" for g, k in groups.items())}.
`shape_only = True` marks features that do not change when the curve is scaled (peak- or AUC-normalised):
use these to compare curve *shape* between masks/planes independent of the contrast amount.

Trust per variant (all unique runs):

{md_table(n.reset_index())}

## Arrival threshold

Two objective checks per candidate fraction:
- **precision**: both planes see the same bolus at the same moment, so the A - B arrival difference measures
  arrival-time noise (ica_top, pairs trusted in both planes);
- **noise floor**: the level must lie above the baseline noise, otherwise noise triggers the "arrival";
  `trusted_curves_noise_above` counts trusted curves whose 3-sigma frame noise (3 / aif_qc SNR) reaches it.

{md_table(arr.round(3))}

The precision hardly depends on the fraction (SD {sd.min():.3f}-{sd.max():.3f} s, well below the 250 ms frame
interval), so the noise floor decides. **Rule: the smallest fraction above the 3-sigma noise of every trusted
curve = {best}** (config: {cfg["arrival_frac"]}). It is in the conventional 10-20 % range and the closest to the true
onset that is still noise-free. The 10 % and 90 % crossings are always reported as well.

## Load

```python
import pandas as pd
d = 'results/combined_v1/aif_features'
f = pd.read_csv(f'{{d}}/features_aif.csv')
ok = f[f.qc_trust == 'trusted']
shape_cols = pd.read_csv(f'{{d}}/feature_dictionary.csv').query('shape_only').column.tolist()
```
"""
    (out_dir / "README.md").write_text(txt, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="combined result folder, e.g. results/combined_v1")
    p.add_argument("--config", type=Path, default=CONFIG)
    args = p.parse_args()
    out, cfg = args.out, load_config(args.config)
    out_dir = out / "aif_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    curves = pd.read_csv(out / "aif_curves.csv")
    curves = curves[curves.sampling == "resampled_250ms"].sort_values(["run_id", "variant", "time_s"])
    old = pd.read_csv(out / "features.csv")
    tax = pd.read_csv(out / "aif_qc" / "taxonomy" / "taxonomy_labels.csv").rename(
        columns={"category": "qc_category", "taxonomy": "qc_taxonomy", "trust": "qc_trust", "included": "qc_included"})
    base = (old[~old.is_duplicate][[c for c in ID_COLS] + PERF_COLS]
            .merge(tax[["run_id", "variant"] + QC_COLS], on=["run_id", "variant"]))
    curves = curves[curves.run_id.isin(base.run_id)]

    arr = arrival_threshold(curves, cfg, base, tax.loc[tax.qc_trust == "trusted", "snr"].to_numpy())
    arr.to_csv(out_dir / "arrival_threshold.csv", index=False)
    best = pick_arrival_frac(arr)
    if not np.isclose(best, cfg["arrival_frac"]):
        warnings.warn(f"arrival_frac {cfg['arrival_frac']} in the config, the selection rule gives {best}")

    feat = base.merge(compute(curves, cfg, cfg["arrival_frac"]), on=["run_id", "variant"])
    feat_cols = [c for c in DICTIONARY if c in feat and c not in ID_COLS + QC_COLS + PERF_COLS]
    feat = feat[ID_COLS + QC_COLS + feat_cols + PERF_COLS]
    feat.to_csv(out_dir / "features_aif.csv", index=False)

    # SQ2: pairs where both planes are trusted; SQ1: runs where all three variants are trusted
    num = [c for c in feat_cols + PERF_COLS if pd.api.types.is_numeric_dtype(feat[c])]
    tr = feat[feat.qc_trust == "trusted"]
    ab = tr.groupby(["ab_pair_id", "variant"]).filter(lambda g: set(g.plane) == {"A", "B"})
    paired = ab.pivot(index=["ab_pair_id", "variant"], columns="plane", values=num)
    paired.columns = [f"{c}_{pl}" for c, pl in paired.columns]
    paired = paired.reset_index().merge(ab.drop_duplicates("ab_pair_id")[["ab_pair_id", "patient_id", "injection_site",
                                                                          "evt_phase"]], on="ab_pair_id")
    paired.to_csv(out_dir / "features_paired_AB.csv", index=False)
    all3 = tr.groupby("run_id").filter(lambda g: len(g) == 3)
    fvt = all3.pivot(index="run_id", columns="variant", values=num)
    fvt.columns = [f"{c}_{SHORT[v]}" for c, v in fvt.columns]
    fvt = fvt.reset_index().merge(all3.drop_duplicates("run_id")[[c for c in ID_COLS if c != "variant"]
                                                                 + ["qc_included"]], on="run_id")
    fvt.to_csv(out_dir / "features_full_vs_top.csv", index=False)

    shape_curves(curves, feat, cfg).to_csv(out_dir / "curves_shape.csv", index=False)
    pd.DataFrame([dict(column=k, group=g, unit=u, shape_only=s, definition=d) for k, (g, u, s, d) in DICTIONARY.items()
                  if k in feat]).to_csv(out_dir / "feature_dictionary.csv", index=False)
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    write_readme(out_dir, feat, arr, cfg, " ".join(["python"] + sys.argv),
                 int((paired.variant == "ica_top").sum()), len(fvt))
    print(arr.round(3).to_string(index=False))
    print(feat.groupby("variant").qc_trust.value_counts().unstack())


if __name__ == "__main__":
    main()
