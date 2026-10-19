"""
Sort the AIF curves of a results folder into quality categories, separately for ICA full and ICA top.

Every curve is baseline-corrected and normalised to its own peak (0 = pre-bolus baseline, 1 = peak), so the
thresholds below are fractions of the bolus amplitude and do not depend on the absolute DSA scale.

Categories (one per curve, first matching rule wins; all triggered rules are listed in `flags`):

    low_signal    no clear bolus: amplitude small compared with the frame-to-frame noise
    truncated     the acquisition ends before washout: the curve never drops below TRUNC_LEVEL of the peak
                  after the maximum
    multi_peak    the first pass is not one smooth peak: two or more peaks, or notches/steps on the up- or
                  downslope (measured as the back-and-forth movement before washout)
    late_rise     after washing out the curve rises again (second injection, recirculation, venous overlap)
    noisy         a single bolus, but the tail after washout is rough (large frame-to-frame jumps)
    baseline      no stable pre-bolus baseline (bolus already rising at the first frames, or baseline drift)
    usable        none of the above; borderline usable curves carry a minor_notch (BOLUS_EXCESS/2 .. BOLUS_EXCESS)
                  or minor_rebound (LATE_REBOUND/2 .. LATE_REBOUND) flag

Output (in each results folder, under aif_qc/):
    aif_labels.csv          run_id, variant, category, flags, features
    aif_summary.csv         counts per category x variant
    aif_agreement.csv       category of ica_full against ica_top for the same run
    <variant>_<category>.png grid of all curves of that variant in that category

Example:
    python classify_aif.py results/praise results/praise_batch_2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

CATEGORIES = ["usable", "truncated", "multi_peak", "late_rise", "noisy", "baseline", "low_signal"]
VARIANTS = ["ica_full", "ica_top"]

N_BASE = 3              # frames used for the pre-bolus baseline
MIN_SNR = 8.0           # amplitude / baseline noise below this is low_signal
TRUNC_LEVEL = 0.3       # never falls below this fraction of the peak after the maximum -> truncated
PEAK_PROM = 0.08        # prominence (fraction of peak) for a peak to be counted
BOLUS_EXCESS = 0.2      # back-and-forth movement before washout (fraction of peak) -> multi_peak
LATE_REBOUND = 0.2      # rise after washout (fraction of peak) that counts as late_rise
WASHED_OUT = 0.5        # the curve counts as washed out once it is below this fraction of the peak
ROUGH_MAX = 0.03        # mean |second difference| of the tail after washout, fraction of peak
BASE_DRIFT = 0.12       # spread of the baseline frames, fraction of peak


def curve_features(t, y):
    base = np.median(y[:N_BASE])
    amp = y.max() - base
    yn = (y - base) / amp
    ipk = int(np.argmax(yn))

    d1 = np.diff(y)
    noise = 1.4826 * np.median(np.abs(d1 - np.median(d1))) + 1e-9
    snr = amp / noise

    post = yn[ipk:]
    post_min = post.min()

    # Everything up to the first frame below WASHED_OUT after the maximum is the first pass of the bolus.
    below = np.nonzero(post < WASHED_OUT)[0]
    i_wash = ipk + below[0] if len(below) else len(yn)
    peaks, _ = find_peaks(np.r_[-1.0, yn, -1.0], prominence=PEAK_PROM)
    bolus_peaks = [p - 1 for p in peaks if p - 1 < i_wash and yn[p - 1] > 0.3]

    # A clean first pass goes up once and down once; any extra up-and-down movement is excess.
    seg = yn[:i_wash]
    excess = float(np.abs(np.diff(seg)).sum() - (1 - seg[0]) - (1 - seg[-1]))

    # Largest rise above the running minimum, after the curve has washed out.
    rebound = 0.0
    if i_wash < len(yn):
        tail = yn[i_wash:]
        rebound = float(np.max(tail - np.minimum.accumulate(tail)))

    # Roughness of the tail after washout, where the true curve is smooth (skip the bend right after washout).
    tail = yn[i_wash + 2:]
    rough = float(np.abs(np.diff(tail, 2)).mean()) if len(tail) >= 5 else 0.0

    base_spread = float(np.ptp(yn[:N_BASE]))
    rise_start = int(np.argmax(yn > 0.2))

    return dict(amp=amp, snr=snr, t_peak=t[ipk], duration=t[-1] - t[0], end_frac=yn[-1], post_min=post_min,
                n_bolus_peaks=len(bolus_peaks), excess=excess, rebound=rebound, rough=rough, base_spread=base_spread,
                rise_frame=rise_start)


def classify(f):
    flags = []
    if f["snr"] < MIN_SNR:
        flags.append("low_signal")
    if f["post_min"] > TRUNC_LEVEL:
        flags.append("truncated")
    if f["excess"] > BOLUS_EXCESS:
        flags.append("multi_peak")
    if f["rebound"] > LATE_REBOUND:
        flags.append("late_rise")
    if f["rough"] > ROUGH_MAX:
        flags.append("noisy")
    if f["base_spread"] > BASE_DRIFT or f["rise_frame"] < N_BASE:
        flags.append("baseline")
    if 0.5 * BOLUS_EXCESS < f["excess"] <= BOLUS_EXCESS:
        flags.append("minor_notch")
    if 0.5 * LATE_REBOUND < f["rebound"] <= LATE_REBOUND:
        flags.append("minor_rebound")
    order = ["low_signal", "truncated", "multi_peak", "late_rise", "noisy", "baseline"]
    cat = next((c for c in order if c in flags), "usable")
    return cat, flags


def plot_grid(df, curves, title, path):
    n = len(df)
    if n == 0:
        return
    cols = min(8, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(max(cols, 3) * 2.4, rows * 1.9 + 0.5), squeeze=False)
    for ax, (_, r) in zip(axs.flat, df.iterrows()):
        t, y = curves[(r.run_id, r.variant)]
        ax.plot(t, y, "C0" if r.variant == "ica_full" else "C3", lw=1.2)
        ax.set_title(r.run_id.replace("PRAISE_BATCH_2_", "").replace("PRAISE_", ""), fontsize=7)
        ax.text(0.98, 0.95, r["flags"].replace(";", ", "), transform=ax.transAxes,
                ha="right", va="top", fontsize=5, color="0.4")
        ax.tick_params(labelsize=5)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run(folder):
    folder = Path(folder)
    out = folder / "aif_qc"
    out.mkdir(exist_ok=True)
    aif = pd.read_csv(folder / "aif_curves.csv")

    rows, curves = [], {}
    for (rid, var), d in aif.groupby(["run_id", "variant"], sort=True):
        d = d.sort_values("time_s")
        t, y = d.time_s.to_numpy(), d.aif.to_numpy()
        curves[(rid, var)] = (t, y)
        f = curve_features(t, y)
        cat, flags = classify(f)
        rows.append(dict(run_id=rid, variant=var, category=cat, flags=";".join(flags), **f))
    lab = pd.DataFrame(rows)
    lab.round(4).to_csv(out / "aif_labels.csv", index=False)

    summ = (lab.groupby(["category", "variant"]).size().unstack(fill_value=0)
            .reindex(index=CATEGORIES, columns=VARIANTS, fill_value=0))
    summ.loc["total"] = summ.sum()
    summ.to_csv(out / "aif_summary.csv")

    wide = lab.pivot(index="run_id", columns="variant", values="category")
    agree = pd.crosstab(wide["ica_full"], wide["ica_top"]).reindex(index=CATEGORIES, columns=CATEGORIES,
                                                                   fill_value=0)
    agree.index.name, agree.columns.name = "ica_full \\ ica_top", None
    agree.to_csv(out / "aif_agreement.csv")

    for var in VARIANTS:
        for cat in CATEGORIES:
            sel = lab[(lab.variant == var) & (lab.category == cat)]
            plot_grid(sel, curves, f"{folder.name} | {var} | {cat} (n={len(sel)})", out / f"{var}_{cat}.png")

    print(f"\n== {folder.name}  ({lab.run_id.nunique()} runs)")
    print(summ.to_string())
    print(f"same category full vs top: {(wide.ica_full == wide.ica_top).mean():.0%}")
    return lab


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", help="results folders containing aif_curves.csv")
    for folder in p.parse_args().folders:
        run(folder)


if __name__ == "__main__":
    main()
