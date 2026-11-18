"""Shape QC of every AIF curve: one failure category (or usable) per curve, plus the list of usable AIFs.

    python scripts/aif_qc.py
"""

import argparse
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, medfilt
from paths import DATA

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
CATEGORIES = ["usable", "low_signal", "no_baseline", "truncated", "multi_peak", "late_rise", "noisy"]
ORDER = CATEGORIES[1:]  # priority of the failure rules
COLORS = {"ica_top": "C3", "ica_full_model": "C0", "ica_full_grown": "C2"}

TH = dict(
    MIN_SNR=10.0,           # peak / frame noise below this -> low_signal
    MIN_BASE_FRAMES=3,      # fewer pre-arrival frames -> no_baseline
    BASE_DRIFT=0.10,        # spread of the baseline frames -> no_baseline
    TRUNC_LEVEL=0.2,        # never below this after the peak -> truncated
    WASHED_OUT=0.5,         # below this after the peak = end of the first pass
    PEAK_PROM=0.08,         # prominence of a counted first-pass peak
    BOLUS_EXCESS=0.2,       # first-pass back-and-forth -> multi_peak
    LATE_REBOUND=0.15,      # rise after washout -> late_rise
    ROUGH_MAX=0.015,        # tail zig-zag (mean |y - running median|) -> noisy
    TAIL_ELEVATED=0.2,      # end_frac above this -> slow_washout flag
    BROAD_FWHM_S=4.0,       # FWHM above this -> broad_bolus flag
    VENOUS_HIGH=0.2,        # venous_frac (select_runs.py) above this counts as venous overlap in the mask
    CARDIAC_GAP_S=(0.6, 1.4),  # peak spacing compatible with one heartbeat (43-100 bpm)
    MASK_SMALL_PX=500,      # masks smaller than this average few pixels (noise)
    R_TWIN=0.99,            # r with another variant's curve at/above this: practically the same curve
    PLATEAU_SLOPE=0.03,     # |tail_slope| below this (peak fraction / s) = flat tail
)


def curve_features(t, c, th):
    t, c = np.asarray(t, float), np.asarray(c, float)
    i_pk = int(np.argmax(c))
    b0 = float(np.median(c[:2]))
    above = np.nonzero(c[:i_pk + 1] >= b0 + 0.1 * (c[i_pk] - b0))[0]
    n_base = int(above[0]) if above.size else 0
    base = float(np.median(c[:max(n_base, 1)]))
    amp = float(c[i_pk] - base)
    yn = (c - base) / amp if amp > 0 else np.full_like(c, np.nan)

    post = yn[i_pk:]
    below = np.nonzero(post < th["WASHED_OUT"])[0]
    i_wash = i_pk + int(below[0]) if below.size else len(yn)

    d2 = np.r_[np.diff(yn[:n_base], 2), np.diff(yn[i_wash:], 2)]
    d2 = d2 if d2.size >= 3 else np.diff(yn, 2)
    noise = 1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6) + 1e-9
    seg = yn[:i_wash]
    excess = float(np.abs(np.diff(seg)).sum() - (1 - seg[0]) - (1 - seg[-1])) if len(seg) > 1 else 0.0

    up, down = yn[:i_pk + 1], yn[i_pk:i_wash]
    gap_up, gap_down = np.maximum.accumulate(up) - up, down - np.minimum.accumulate(down)
    dip_up = float(gap_up.max()) if len(up) else 0.0
    dip_down = float(gap_down.max()) if len(down) else 0.0
    t_notch = float(t[int(np.argmax(gap_up))] if dip_up >= dip_down else t[i_pk + int(np.argmax(gap_down))])
    dt = np.diff(t)
    slow = np.nonzero(dt > 1.5 * dt[0])[0]
    t_switch = float(t[slow[0]]) if slow.size else np.nan

    pk, _ = find_peaks(np.r_[-1.0, yn, -1.0], prominence=th["PEAK_PROM"])
    pk = [p - 1 for p in pk if p - 1 < i_wash and yn[p - 1] > 0.3]
    top2 = sorted(pk, key=lambda i: -yn[i])[:2]
    peak_gap = float(abs(t[top2[0]] - t[top2[1]])) if len(top2) == 2 else np.nan

    rebound, t_reb = 0.0, np.nan
    if i_wash < len(yn):
        tail = yn[i_wash:]
        rise = tail - np.minimum.accumulate(tail)
        j = int(np.argmax(rise))
        rebound, t_reb = float(rise[j]), float(t[i_wash + j] - t[i_pk])
    tail = yn[i_wash:]
    rough = float(np.abs(tail - medfilt(tail, 3))[1:-1].mean()) if len(tail) >= 4 else 0.0
    tail_slope = float((yn[-1] - np.interp(t[-1] - 2.0, t, yn)) / 2.0)

    half = np.nonzero(yn[:i_pk + 1] >= 0.5)[0][0]
    left = np.interp(0.5, [yn[half - 1], yn[half]], [t[half - 1], t[half]]) if half > 0 else t[0]
    hb = np.nonzero(post < 0.5)[0]
    if hb.size:
        k = i_pk + int(hb[0])
        right = np.interp(0.5, [yn[k], yn[k - 1]], [t[k], t[k - 1]])
        fwhm = float(right - left)
    else:
        fwhm = np.nan

    return dict(n_frames=len(c), n_base=n_base, base_spread=float(np.ptp(yn[:max(n_base, 1)])), amp=amp,
                snr=float(1 / noise), t_peak=float(t[i_pk]), t_end=float(t[-1]), follow_s=float(t[-1] - t[i_pk]),
                post_min=float(post.min()), end_frac=float(yn[-1]), tail_slope=tail_slope, fwhm_s=fwhm, excess=excess,
                notch_side="upslope" if dip_up >= dip_down else "downslope", notch_depth=max(dip_up, dip_down), t_notch_s=t_notch, t_switch_s=t_switch,
                n_peaks=len(pk), peak_gap_s=peak_gap, rebound=rebound, t_rebound_s=t_reb, rough=rough), yn


def classify(f, th):
    flags = []
    if f["snr"] < th["MIN_SNR"]:
        flags.append("low_signal")
    if f["n_base"] < th["MIN_BASE_FRAMES"] or f["base_spread"] > th["BASE_DRIFT"]:
        flags.append("no_baseline")
    if f["post_min"] > th["TRUNC_LEVEL"]:
        flags.append("truncated")
    if f["excess"] > th["BOLUS_EXCESS"]:
        flags.append("multi_peak")
    if f["rebound"] > th["LATE_REBOUND"]:
        flags.append("late_rise")
    if f["rough"] > th["ROUGH_MAX"]:
        flags.append("noisy")
    cat = next((c for c in ORDER if c in flags), "usable")
    if 0.5 * th["BOLUS_EXCESS"] < f["excess"] <= th["BOLUS_EXCESS"]:
        flags.append("minor_notch")
    if 0.5 * th["LATE_REBOUND"] < f["rebound"] <= th["LATE_REBOUND"]:
        flags.append("minor_rebound")
    if 0.5 * th["ROUGH_MAX"] < f["rough"] <= th["ROUGH_MAX"]:
        flags.append("minor_noise")
    if f["end_frac"] > th["TAIL_ELEVATED"] and "truncated" not in flags:
        flags.append("slow_washout")
    if not f["fwhm_s"] <= th["BROAD_FWHM_S"]:
        flags.append("broad_bolus")
    return cat, flags


def describe(r, th):
    """What is wrong with the curve, in numbers (one sentence per triggered failure flag)."""
    s = []
    fl = r.flags.split(";")
    if "low_signal" in fl:
        s.append(f"bolus only {r.snr:.1f}x the frame noise (< {th['MIN_SNR']:g})")
    if "no_baseline" in fl:
        s.append(f"{r.n_base} pre-arrival frame(s), baseline spread {r.base_spread:.2f} of the peak"
                 if r.n_base < th["MIN_BASE_FRAMES"] else f"baseline drifts {r.base_spread:.2f} of the peak "
                                                            f"before arrival")
    if "truncated" in fl:
        s.append(f"never washes out: minimum after the peak {r.post_min:.2f} of the peak, still {r.end_frac:.2f} at "
                 f"the last frame, {r.follow_s:.1f} s recorded after the peak (run {r.duration_s:.2f} s)")
    if "multi_peak" in fl:
        two = (f"{r.n_peaks} peaks {r.peak_gap_s:.2f} s apart" if r.n_peaks >= 2
               else f"notch/step on the {r.notch_side}")
        s.append(f"first pass not one bolus: {two}, largest dip {r.notch_depth:.2f} on the {r.notch_side}, "
                 f"excess movement {r.excess:.2f} (> {th['BOLUS_EXCESS']:g})")
    if "late_rise" in fl:
        s.append(f"rises again by {r.rebound:.2f} of the peak after washout, {r.t_rebound_s:.1f} s after the peak")
    if "noisy" in fl:
        s.append(f"zig-zagging tail: mean deviation from the running median {r.rough:.3f} of the peak "
                 f"(> {th['ROUGH_MAX']:g}), frame-noise SNR {r.snr:.0f}")
    return "; ".join(s)


def causes(r, th, partner_cat, other_cats, r_pair):
    """Most likely origin of the defect of curve r: a verdict, then the evidence (rule based on run context).

    partner_cat: category of the same variant in the other plane (None without partner)
    other_cats:  {variant: category} of the other masks of the same run
    r_pair:      {variant: Pearson r between this native curve and that variant's (same frames)}
    """
    cat, fl = r.category, set(r.flags.split(";"))
    in_partner = partner_cat is not None and partner_cat == cat
    venous = pd.notna(r.venous_frac) and r.venous_frac > th["VENOUS_HIGH"]
    cca = "CCA" in str(r.injection_site)
    ev = []

    # same run, other mask passes: either a near-identical curve just over a threshold, or a different curve
    ok = [v for v, c in other_cats.items() if c == "usable"]
    twin = [v for v in ok if r_pair.get(v, 0) >= th["R_TWIN"]]
    mask_note = ""
    if twin:
        mask_note = (f"borderline: the {twin[0]} AIF of this run is nearly identical (r {r_pair[twin[0]]:.3f}) and "
                     f"passes, so this curve is only just over the threshold")
    elif ok:
        mask_note = (f"mask-specific: the {', '.join(ok)} AIF of this run passes and differs from this one "
                     f"(r {min(r_pair.get(v, np.nan) for v in ok):.2f}), so the defect comes from which pixels "
                     f"this mask averages")
        if ok == ["ica_full_grown"]:
            mask_note += " (the grown mask averages a longer vessel segment, which smooths short features)"
    plane_note = ("" if partner_cat is None else
                  "same defect in the other plane" if in_partner else f"other plane is {partner_cat}")

    if cat == "truncated":
        if abs(r.tail_slope) < th["PLATEAU_SLOPE"]:
            verdict = "persistent opacification in the mask (tail plateau), not a too-short run"
            ev.append(f"tail flat at ~{r.end_frac:.2f} over the last 2 s ({r.tail_slope:+.3f}/s) of "
                      f"{r.follow_s:.1f} s follow-up")
        elif r.tail_slope > 0:
            verdict = "tail rising again at the end: venous return / recirculation building up in the mask"
            ev.append(f"tail rising {r.tail_slope:+.3f}/s over the last 2 s")
        else:
            verdict = "acquisition too short for this bolus: still washing out when the run stops"
            ev.append(f"tail still falling {r.tail_slope:+.3f}/s at the end, {r.follow_s:.1f} s after the peak")
        if venous:
            ev.append(f"venous_frac {r.venous_frac:.0%}: late-opacifying (venous/parenchymal) pixels in the mask")
        if r.variant == "ica_full_grown":
            ev.append("grown mask reaches distal segments where contrast lingers")
        if "broad_bolus" in fl:
            ev.append("broad bolus (FWHM " + (f"{r.fwhm_s:.1f} s)" if pd.notna(r.fwhm_s) else "not reached)")
                      + ": long/slow injection")
        if not venous and abs(r.tail_slope) < th["PLATEAU_SLOPE"]:
            ev.append("no venous overlap measured: slow/stagnant ICA outflow is the alternative explanation")
    elif cat == "multi_peak":
        if in_partner:
            verdict = "irregular bolus (seen in both planes): injection pause/pulsation or pulsatile inflow"
            if pd.notna(r.peak_gap_s) and th["CARDIAC_GAP_S"][0] <= r.peak_gap_s <= th["CARDIAC_GAP_S"][1]:
                ev.append(f"peak spacing {r.peak_gap_s:.2f} s fits one heartbeat")
            if cca:
                ev.append(f"{r.injection_site} injection: the bolus is split at the carotid bifurcation")
        elif cca and r.notch_side == "upslope":
            verdict = "ECA branches overlapping the ICA mask in this projection (CCA injection)"
            ev.append("notch on the upslope = a second, earlier/later arrival front")
        elif ok:
            verdict = mask_note
        else:
            verdict = "projection-specific: vessel overlap or motion/misregistration in this view"
        if r.notch_side == "downslope" and pd.notna(r.t_switch_s) and abs(r.t_notch_s - r.t_switch_s) <= 0.25:
            ev.append(f"dip at {r.t_notch_s:.2f} s is at the 250->500 ms frame-interval switch ({r.t_switch_s:.2f} s): "
                      f"coarser sampling can make a smooth washout look stepped")
    elif cat == "late_rise":
        if venous:
            verdict = "venous phase projected over the ICA mask"
            ev.append(f"venous_frac {r.venous_frac:.0%}")
        elif in_partner:
            verdict = "true second bolus (second injection or recirculation), present in both planes"
        else:
            verdict = "a late-filling vessel (vein/sinus or ECA branch) overlaps the mask in this projection"
        if r.t_rebound_s >= 4:
            ev.append(f"re-rise {r.t_rebound_s:.1f} s after the peak: venous-return timing")
    elif cat == "noisy":
        verdict = ("low contrast-to-noise in this injection (both planes)" if in_partner
                   else "projection-specific noise: motion, overlap or a thin vessel segment in this view")
        if r.mask_px < th["MASK_SMALL_PX"]:
            ev.append(f"small mask ({int(r.mask_px)} px): few pixels averaged")
        if "1000" in str(r.frame_pattern):
            ev.append("tail sampled at 1 fps")
    elif cat == "no_baseline":
        if r.n_base < th["MIN_BASE_FRAMES"]:
            verdict = "bolus already arriving at the first frames: injection started before the acquisition"
        else:
            verdict = "baseline drift before arrival: patient motion / mask misregistration in the subtraction"
    else:  # low_signal
        verdict = "mask on a poorly opacified segment or very diluted contrast"
    if mask_note:
        verdict = mask_note if twin else verdict
        if verdict != mask_note:
            ev.insert(0, mask_note)
    if plane_note:
        ev.append(plane_note)
    return verdict + (" — evidence: " + "; ".join(ev) if ev else "")


CAUSE_CLASS = [("irregular bolus", "injection / bolus"), ("ECA branches", "ECA overlap (CCA injection)"),
               ("projection-specific", "projection (overlap / motion)"),
               ("venous phase", "venous / late overlap"), ("a late-filling", "venous / late overlap"),
               ("persistent opacification", "venous / late overlap"), ("tail rising", "venous / late overlap"),
               ("acquisition too short", "acquisition too short"), ("mask-specific", "mask content"),
               ("borderline", "borderline (near-identical curve passes)"), ("baseline drift", "baseline / motion"),
               ("bolus already", "baseline / motion"), ("low contrast", "contrast / noise"),
               ("mask on a poorly", "contrast / noise")]


def cause_class(cause):
    return next((c for k, c in CAUSE_CLASS if cause.startswith(k)), "") if cause else ""


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def plot_grid(lab, yn, title, path):
    n = len(lab)
    if not n:
        return
    cols = min(6, n)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.4 + 0.5), squeeze=False)
    for ax, r in zip(axs.flat, lab.itertuples()):
        t, y = yn[(r.run_id, r.variant)]
        ax.axhspan(-0.05, 0.05, color="0.92", lw=0)
        ax.axhline(0.3, color="0.8", lw=0.6, ls="--")
        ax.plot(t, y, color=COLORS[r.variant], lw=1.2, marker=".", ms=3)
        ax.set_ylim(-0.25, 1.15)
        name = r.run_id.replace("PRAISE_BATCH_2_", "B2_").replace("PRAISE_", "")
        ax.set_title(f"{name}{' *' if r.included else ''}  {r.injection_site}", fontsize=7)
        ax.text(0.98, 0.95, "\n".join(f for f in r.flags.split(";") if f), transform=ax.transAxes,
                ha="right", va="top", fontsize=6, color="0.35")
        ax.tick_params(labelsize=6)
    for ax in axs.flat[n:]:
        ax.axis("off")
    fig.suptitle(title + "   (* = included in selection; dashed = truncation level)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def run(out, th, command):
    qc = out / "qc"
    qc.mkdir(exist_ok=True)
    for f in qc.glob("grid_*.png"):
        f.unlink()
    inputs = {n: out / n for n in ("aif_curves.csv", "acquisition_table.csv", "selection.csv", "features.csv")}
    curves = pd.read_csv(inputs["aif_curves.csv"])
    curves = curves[curves.sampling == "native"].sort_values(["run_id", "variant", "time_s"])
    acq = pd.read_csv(inputs["acquisition_table.csv"]).set_index("run_id")
    sel = pd.read_csv(inputs["selection.csv"]).set_index("run_id")
    feat = pd.read_csv(inputs["features.csv"]).set_index(["run_id", "variant"])

    rows, yn = [], {}
    for (rid, var), d in curves.groupby(["run_id", "variant"], sort=True):
        t, c = d.time_s.to_numpy(), d.aif.to_numpy()
        f, y = curve_features(t, c, th)
        yn[(rid, var)] = (t, y)
        cat, flags = classify(f, th)
        a, s, fe = acq.loc[rid], sel.loc[rid], feat.loc[(rid, var)]
        rows.append(dict(run_id=rid, variant=var, category=cat, flags=";".join(flags),
                         included=bool(s.included), is_duplicate=bool(a.is_duplicate), dataset=a.dataset,
                         patient_id=a.patient_id, plane=a.plane, ab_pair_id=a.ab_pair_id,
                         injection_site=a.injection_site, evt_phase=s.evt_phase, duration_s=a.duration_s,
                         frame_pattern=a.frame_interval_pattern if "frame_interval_pattern" in a else a.frame_rate_regime,
                         venous_frac=s.get(f"venous_frac_{var}", np.nan), mask_px=fe.mask_px,
                         sat_frac=fe.sat_frac, r_vs_top=fe.r_vs_top, r_ab=fe.r_ab, **f))
    lab = pd.DataFrame(rows)

    # partner plane (same injection, same variant, same duplicate status) and other variants of the run
    key = lab.set_index(["ab_pair_id", "is_duplicate", "plane", "variant"]).category
    cat_of = lab.set_index(["run_id", "variant"]).category
    reasons, cause = [], []
    for r in lab.itertuples():
        other_plane = {"A": "B", "B": "A"}.get(r.plane)
        partner = key.get((r.ab_pair_id, r.is_duplicate, other_plane, r.variant))
        others = {v: cat_of[(r.run_id, v)] for v in VARIANTS if v != r.variant}
        r_pair = {v: float(np.corrcoef(yn[(r.run_id, r.variant)][1], yn[(r.run_id, v)][1])[0, 1]) for v in others}
        lab_r = SimpleNamespace(**lab.loc[r.Index].to_dict())
        if r.category == "usable":
            reasons.append("")
            cause.append("")
        else:
            reasons.append(describe(lab_r, th))
            cause.append(causes(lab_r, th, partner, others, r_pair))
    lab["partner_category"] = [key.get((r.ab_pair_id, r.is_duplicate, {"A": "B", "B": "A"}.get(r.plane), r.variant))
                               for r in lab.itertuples()]
    lab["reason"], lab["cause"] = reasons, cause
    lab["cause_class"] = lab.cause.map(cause_class)
    lead = ["run_id", "variant", "category", "flags", "reason", "cause_class", "cause", "included",
            "partner_category"]
    lab = lab[lead + [c for c in lab.columns if c not in lead]]
    lab.round(4).to_csv(qc / "aif_qc_labels.csv", index=False)

    summ = []
    for name, sub in (("all_unique_runs", lab[~lab.is_duplicate]), ("included_runs", lab[lab.included])):
        tab = sub.groupby(["category", "variant"]).size().unstack(fill_value=0).reindex(
            index=CATEGORIES, columns=VARIANTS, fill_value=0)
        tab.loc["total"] = tab.sum()
        tab.insert(0, "set", name)
        summ.append(tab)
    summ = pd.concat(summ)
    summ.to_csv(qc / "aif_qc_summary.csv")

    w = lab.pivot(index="run_id", columns="variant", values="category")[VARIANTS]
    use = lab.drop_duplicates("run_id").set_index("run_id")[
        ["dataset", "patient_id", "plane", "ab_pair_id", "injection_site", "evt_phase", "is_duplicate", "included"]]
    for v in VARIANTS:
        use[f"category_{v}"] = w[v]
        use[f"usable_{v}"] = w[v] == "usable"
    use["n_usable_variants"] = use[[f"usable_{v}" for v in VARIANTS]].sum(axis=1)
    use["use_primary"] = use.included & use.usable_ica_top
    # for included runs whose ica_top fails: is the other plane of the same injection usable?
    alt = {}
    for rid, r in use[use.included & ~use.usable_ica_top].iterrows():
        o = use[(use.ab_pair_id == r.ab_pair_id) & (use.plane != r.plane) & ~use.is_duplicate]
        alt[rid] = o.index[0] if len(o) and o.usable_ica_top.iloc[0] else ""
    use["alt_plane_run_if_failed"] = pd.Series(alt)
    use = use.reset_index().fillna({"alt_plane_run_if_failed": ""})
    use.to_csv(qc / "usable_aifs.csv", index=False)

    for v in VARIANTS:
        for c in CATEGORIES:
            sub = lab[(lab.variant == v) & (lab.category == c) & ~lab.is_duplicate]
            plot_grid(sub, yn, f"{v} | {c} (n={len(sub)})", qc / f"grid_{v}_{c}.png")

    cfg = dict(command=command, thresholds=th, inputs={n: md5(p) for n, p in inputs.items()},
               curve_sampling="native", n_curves=len(lab))
    (qc / "qc_config.json").write_text(json.dumps(cfg, indent=2))
    return lab, use, summ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    for k, v in TH.items():
        if not isinstance(v, tuple):
            p.add_argument(f"--{k.lower().replace('_', '-')}", type=type(v), default=v)
    args = p.parse_args()
    th = {k: getattr(args, k.lower(), v) for k, v in TH.items()}
    lab, use, summ = run(args.out, th, " ".join(["python"] + sys.argv))
    print(summ.to_string())
    print(f"\nincluded runs with usable ica_top: {int(use.use_primary.sum())} / {int(use.included.sum())}")


if __name__ == "__main__":
    main()
