"""
AIF truncation: how much does a curve that has not washed out bias the perfusion maps, which tail level
is acceptable, and what do the obviously truncated runs look like when fixed by hand?

Part A — simulation (data-driven truncation threshold)
    Runs with a trusted ica_top AIF that washes out completely (aif_qc post_min <= SIM_MAX_POST_MIN and at
    least SIM_MIN_FOLLOW_S recorded after the peak) are made "truncated" at tail level L (fraction of
    the peak) in the two ways a curve ends up truncated:
        short_run   the acquisition stops early: the uniform 250 ms sequence (AIF and tissue) ends at the
                    first frame after the peak where the baseline-corrected AIF falls to L
        aif_floor   the run is complete but contrast lingers in the AIF mask (veins over the ICA, slow
                    outflow): after the peak the AIF is max(AIF, L), the tissue curves are unchanged
    In both, L is the "tail fraction" the QC thresholds (aif_qc TRUNC_LEVEL = 0.3). The perfusion maps
    are recomputed with the unchanged perfDSA deconvolution and compared with those of the full run:
    bias = median over the perfused pixels of the full run (CBV > 0) of the modified map / same median of
    the full map - 1. Per scenario the threshold is the highest L at which the median |bias| of CBV, CBF
    and MTT over the runs stays below MAX_BIAS.

Part A2 — optimal scan duration (method of de la Rosa et al. 2023, PLoS One 18:e0283610)
    Frames are removed one at a time from the end of each simulation run (AIF and tissue). A shortened run
    is reliable when CBV, CBF and MTT all deviate < MAX_BIAS from the full run (their reliability
    criterion: lesion volumes within 10 %). The optimal scan duration (OSD) of a run is the shortest
    duration from which on every longer version is reliable. Reported per run: AIF coverage = OSD - AIF
    peak time (their most predictive truncation feature), post-bolus time = OSD - arrival (the quantity of
    Lyman et al. 2026, Transl Stroke Res, who need >= 7 s in DSA for a reliable MTT), and the AIF tail
    fraction at the OSD (the level the AIF has fallen to when the run becomes reliable). The P95 over the
    runs is the requirement that makes 95 % of runs reliable; the cohort is then checked against it.

Part B — the obviously truncated runs (meeting 19 Oct: PRAISE084 A/B, 082_B, 071_B)
    Each variant is recomputed three ways:
        original   as in the pipeline
        cut        the sequence ends at the minimum of the AIF after the peak (removes a re-rising tail;
                   a plateau stays)
        gamma      the AIF is replaced by a gamma-variate fit of its first pass (arrival to the frame
                   where it has fallen to GAMMA_FIT_TO of the peak), which extrapolates the washout to
                   baseline; the tissue curves are unchanged. This is the standard repair of a truncated
                   AIF.
    For each: AIF tail fraction, map medians and their change against the original; one figure per run
    with the three AIFs and the four maps.

Input : <out>/acquisition_table.csv (DICOM source), npz/<run_id>.npz (masks, stored maps),
        aif_qc/aif_qc_labels.csv, aif_qc/taxonomy/taxonomy_labels.csv (trust)
Output: <out>/aif_qc/truncation/
    truncation_sim.csv        run x level: map medians, bias
    truncation_sim.png        bias vs level (median, IQR) with the threshold
    truncation_cases.csv      case x variant x method: tail fraction, map medians, change vs original
    case_<run_id>.png         AIFs and maps per method
    truncation_report.md

Example:
    python scripts/aif_truncation.py results/combined_v1
"""

import argparse
import contextlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from cohort_common import MAP_KEYS, ROOT, parse_source, rd

sys.path.insert(0, str(ROOT))
from DSASequence import DSASequence  # noqa: E402

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
CASES = ["PRAISE_PRAISE084_01_S12_A", "PRAISE_PRAISE084_01_S12_B", "PRAISE_PRAISE082_01_S12_B",
         "PRAISE_PRAISE071_01_S11_B"]
METHODS = ["original", "cut", "gamma"]
LEVELS = [0.7, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]
LYMAN_POST_BOLUS_S = 7.0  # Lyman et al. 2026 (DSA): >= 7 s of post-bolus data for a reliable MTT
SCENARIOS = {"short_run": "run stops early (AIF + tissue cut)",
             "aif_floor": "AIF tail floor (run complete)"}
FRAME_INTERVAL_MS = 250  # run_dataset default, used for every stored map
ARRIVAL_FRAC = 0.15  # scripts/aif_features_config.json arrival_frac

TH = dict(
    SIM_MAX_POST_MIN=0.1,   # simulation runs: AIF falls below this after the peak ...
    SIM_MIN_FOLLOW_S=5.0,   # ... and is followed this long after the peak
    MAX_BIAS=0.10,          # acceptable median |bias| of CBV, CBF and MTT
    GAMMA_FIT_TO=0.4,       # gamma-variate fit: first-pass frames down to this fraction after the peak
)


# ----------------------------------------------------------------------------
# Perfusion
# ----------------------------------------------------------------------------
def load_frames(source):
    ds = rd.read_dicom(parse_source(source))
    return rd.load_uniform_sequence(ds, FRAME_INTERVAL_MS, 1024)


def perfusion(frames, aif):
    """perfDSA maps with a given AIF (same call as run_dataset.process_run)."""
    dsa = DSASequence(frames, frame_interval=FRAME_INTERVAL_MS / 1000)
    dsa._aif = np.asarray(aif, dtype=float)
    with contextlib.redirect_stdout(io.StringIO()):  # DSASequence prints timings
        cbv, cbf, mtt, tmax = dsa.get_perfusion_parameters()
    return {"CBV": cbv, "CBF": cbf, "MTT": mtt, "Tmax": tmax.astype(np.float32)}


def sample_aif(frames, mask):
    dsa = DSASequence(frames, frame_interval=FRAME_INTERVAL_MS / 1000)
    return np.asarray(np.ma.filled(dsa.sample_AIF_from_mask(mask), np.nan), dtype=float)


def medians(maps, region):
    return {k: float(np.nanmedian(maps[k][region])) for k in MAP_KEYS}


def normalise(aif):
    """Baseline (median of the frames before 10 % arrival) and peak-normalised curve, as aif_qc.py."""
    i_pk = int(np.argmax(aif))
    b0 = float(np.median(aif[:2]))
    above = np.nonzero(aif[:i_pk + 1] >= b0 + 0.1 * (aif[i_pk] - b0))[0]
    n_base = max(int(above[0]) if above.size else 0, 1)
    base = float(np.median(aif[:n_base]))
    return (aif - base) / (aif[i_pk] - base), base, i_pk, n_base


def tail_fraction(aif):
    yn, _, i_pk, _ = normalise(aif)
    return float(yn[i_pk:].min())


# ----------------------------------------------------------------------------
# Part A: simulation
# ----------------------------------------------------------------------------
def simulate(out, acq, lab, th):
    top = lab[lab.variant == "ica_top"]
    sel = top[(top.trust == "trusted") & (top.post_min <= th["SIM_MAX_POST_MIN"])
              & (top.follow_s >= th["SIM_MIN_FOLLOW_S"])].run_id.tolist()
    rows = []
    for i, run_id in enumerate(sel, 1):
        print(f"[sim {i}/{len(sel)}] {run_id}", flush=True)
        frames = load_frames(acq.loc[run_id, "source"])
        mask = np.load(out / "npz" / f"{run_id}.npz")["mask_ica_top"]
        aif = sample_aif(frames, mask)
        full = perfusion(frames, aif)
        region = full["CBV"] > 0
        ref = medians(full, region)
        yn, _, i_pk, _ = normalise(aif)
        _, base, _, _ = normalise(aif)
        amp = aif[i_pk] - base
        for s in SCENARIOS:
            rows.append(dict(scenario=s, run_id=run_id, level=0.0, n_frames=len(aif), **ref,
                             **{f"bias_{k}": 0.0 for k in MAP_KEYS}))
        for level in LEVELS:
            # short_run: the acquisition stops when the AIF has fallen to the level (AIF and tissue cut)
            below = np.nonzero(yn[i_pk:] <= level)[0]
            n = i_pk + int(below[0]) + 1 if below.size else len(aif)  # up to and including that frame
            if 6 <= n < len(aif):
                m = medians(perfusion(frames[:, :, :n], aif[:n]), region)
                rows.append(dict(scenario="short_run", run_id=run_id, level=level, n_frames=n, **m,
                                 **{f"bias_{k}": m[k] / ref[k] - 1 for k in MAP_KEYS}))
            # aif_floor: full run, but the AIF tail never falls below the level (contrast lingering in the mask)
            floored = aif.copy()
            floored[i_pk:] = np.maximum(aif[i_pk:], base + level * amp)
            m = medians(perfusion(frames, floored), region)
            rows.append(dict(scenario="aif_floor", run_id=run_id, level=level, n_frames=len(aif), **m,
                             **{f"bias_{k}": m[k] / ref[k] - 1 for k in MAP_KEYS}))
        # frame by frame (de la Rosa et al. 2023): drop the last frame until 2 frames after the peak
        dt = FRAME_INTERVAL_MS / 1000
        above = np.nonzero(yn[:i_pk + 1] >= ARRIVAL_FRAC)[0]
        t_arr = float(np.interp(ARRIVAL_FRAC, yn[above[0] - 1:above[0] + 1], [(above[0] - 1) * dt, above[0] * dt])
                      if above[0] > 0 else 0.0)
        for n in range(len(aif) - 1, i_pk + 2, -1):
            m = medians(perfusion(frames[:, :, :n], aif[:n]), region)
            rows.append(dict(scenario="frame_by_frame", run_id=run_id, level=float(yn[n - 1]), n_frames=n,
                             coverage_s=(n - 1 - i_pk) * dt, post_bolus_s=(n - 1) * dt - t_arr, **m,
                             **{f"bias_{k}": m[k] / ref[k] - 1 for k in MAP_KEYS}))
        rows.append(dict(scenario="frame_by_frame", run_id=run_id, level=float(yn[-1]), n_frames=len(aif),
                         coverage_s=(len(aif) - 1 - i_pk) * dt, post_bolus_s=(len(aif) - 1) * dt - t_arr, **ref,
                         **{f"bias_{k}": 0.0 for k in MAP_KEYS}))
    return pd.DataFrame(rows)


def optimal_duration(sim, th):
    """Per run: the shortest version from which on every longer one is reliable (|bias| < MAX_BIAS for
    CBV, CBF and MTT), as AIF coverage, post-bolus time and AIF tail fraction at that point."""
    fb = sim[sim.scenario == "frame_by_frame"].copy()
    fb["reliable"] = fb[[f"bias_{k}" for k in ("CBV", "CBF", "MTT")]].abs().max(axis=1) < th["MAX_BIAS"]
    rows = []
    for run_id, g in fb.sort_values("n_frames", ascending=False).groupby("run_id", sort=False):
        ok = g.reliable.cumprod().astype(bool)  # reliable from the full length down to here
        o = g[ok].iloc[-1]
        rows.append(dict(run_id=run_id, osd_frames=int(o.n_frames), coverage_s=o.coverage_s,
                         post_bolus_s=o.post_bolus_s, tail_frac_at_osd=o.level,
                         full_coverage_s=g.coverage_s.max(), full_post_bolus_s=g.post_bolus_s.max()))
    return pd.DataFrame(rows)


def cohort_check(lab, osd, out):
    """Apply the P95 requirements and Lyman's 7 s to every ica_top curve of the cohort (native frames)."""
    f = pd.read_csv(out / "features.csv")[["run_id", "variant", "aif_arrival_native_s"]]
    c = lab[lab.variant == "ica_top"].merge(f, on=["run_id", "variant"])
    c["coverage_s"] = c.t_end - c.t_peak
    c["post_bolus_s"] = c.t_end - c.aif_arrival_native_s
    req_cov, req_post, req_tail = (osd.coverage_s.quantile(0.95), osd.post_bolus_s.quantile(0.95),
                                   osd.tail_frac_at_osd.quantile(0.05))
    rows = []
    for name, sub in (("all unique runs", c), ("included runs", c[c.included.fillna(False).astype(bool)])):
        rows.append({"runs": name, "n": len(sub),
                     f"coverage >= P95 ({req_cov:.2f} s)": int((sub.coverage_s >= req_cov).sum()),
                     f"post-bolus >= P95 ({req_post:.2f} s)": int((sub.post_bolus_s >= req_post).sum()),
                     f"post-bolus >= {LYMAN_POST_BOLUS_S:.0f} s (Lyman)": int((sub.post_bolus_s >= LYMAN_POST_BOLUS_S).sum()),
                     f"tail fraction <= P5 ({req_tail:.2f})": int((sub.post_min <= req_tail).sum()),
                     "tail fraction <= 0.30 (current QC)": int((sub.post_min <= 0.3).sum())})
    return pd.DataFrame(rows), c, (req_cov, req_post, req_tail)


def sim_summary(sim, th):
    """Per level: median and IQR of the bias; threshold = highest level with median |bias| < MAX_BIAS for
    CBV, CBF and MTT."""
    g = sim[sim.level > 0].groupby("level")
    s = pd.DataFrame({"runs": g.size()})
    for k in MAP_KEYS:
        b = g[f"bias_{k}"]
        s[f"{k} median"] = b.median()
        s[f"{k} IQR"] = b.quantile(0.25).map("{:+.1%}".format) + " .. " + b.quantile(0.75).map("{:+.1%}".format)
    worst = s[[f"{k} median" for k in ("CBV", "CBF", "MTT")]].abs().max(axis=1)
    ok = worst[worst < th["MAX_BIAS"]]
    threshold = float(ok.index.max()) if len(ok) else np.nan
    return s.sort_index(ascending=False), threshold


def plot_sim(sim, thresholds, th, path):
    fig, axes = plt.subplots(len(SCENARIOS), 4, figsize=(16, 4 * len(SCENARIOS)), sharex=True)
    for row, scen in zip(axes, SCENARIOS):
        plot_sim_row(row, sim[(sim.scenario == scen) & (sim.level > 0)], thresholds[scen], th, SCENARIOS[scen])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Simulated truncation of {sim.run_id.nunique()} fully washed-out ica_top runs "
                 f"(green band: +/-{th['MAX_BIAS']:.0%}, dashed: threshold)")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_osd(osd, cohort, req, path):
    """Requirement (simulation runs, OSD) against what the cohort's ica_top curves have."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    spec = [("coverage_s", "AIF coverage at OSD vs in the cohort (s)", req[0], None),
            ("post_bolus_s", "post-bolus time at OSD vs in the cohort (s)", req[1], LYMAN_POST_BOLUS_S),
            ("tail", "AIF tail fraction at OSD vs in the cohort", req[2], 0.3)]
    for ax, (col, title, r, ref) in zip(axes, spec):
        a = osd.tail_frac_at_osd if col == "tail" else osd[col]
        b = cohort.post_min if col == "tail" else cohort[col]
        bins = np.histogram_bin_edges(pd.concat([a, b]).dropna(), 25)
        ax.hist(b.dropna(), bins=bins, color="0.7", label="cohort ica_top (available)")
        ax.hist(a.dropna(), bins=bins, histtype="step", color="C0", lw=2, label="simulation runs (needed, OSD)")
        ax.axvline(r, color="C0", ls="--", label="P95 requirement" if col != "tail" else "P5 at OSD")
        if ref is not None:
            ax.axvline(ref, color="C3", ls=":", label="Lyman 2026: 7 s" if col == "post_bolus_s" else "current QC 0.3")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_sim_row(axes, s, threshold, th, title):
    for ax, k in zip(axes, MAP_KEYS):
        g = s.groupby("level")[f"bias_{k}"]
        x = np.array(sorted(s.level.unique()))
        ax.fill_between(x, g.quantile(0.25).reindex(x), g.quantile(0.75).reindex(x), color="C0", alpha=0.25, label="IQR")
        ax.plot(x, g.median().reindex(x), "o-", color="C0", label="median")
        for _, r in s.groupby("run_id"):
            ax.plot(r.level, r[f"bias_{k}"], color="0.7", lw=0.5, zorder=0)
        ax.axhspan(-th["MAX_BIAS"], th["MAX_BIAS"], color="C2", alpha=0.1)
        if np.isfinite(threshold):
            ax.axvline(threshold, color="k", ls="--", lw=1)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{k} bias", fontsize=10)
        ax.set_xlabel("AIF tail fraction")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0%}"))
    axes[0].set_ylabel(f"{title}\nmedian map / full - 1")


# ----------------------------------------------------------------------------
# Part B: the obviously truncated runs
# ----------------------------------------------------------------------------
def gamma_variate(t, a, t0, alpha, beta):
    dt = np.clip(t - t0, 0, None)
    return a * dt ** alpha * np.exp(-dt / beta)


def gamma_aif(aif, t, th):
    """Gamma-variate fit of the first pass, extrapolated over the whole run (baseline added back)."""
    yn, base, i_pk, n_base = normalise(aif)
    amp = aif[i_pk] - base
    after = np.nonzero(yn[i_pk:] <= th["GAMMA_FIT_TO"])[0]
    i_end = i_pk + int(after[0]) if after.size else int(np.argmin(yn[i_pk:])) + i_pk
    i0 = max(n_base - 1, 0)
    sl = slice(i0, i_end + 1)
    t_pk = t[i_pk]
    p0 = (1.0, t[i0], 2.0, max((t_pk - t[i0]) / 2, 0.2))
    (a, t0, alpha, beta), _ = curve_fit(gamma_variate, t[sl], yn[sl], p0=p0, maxfev=20000,
                                        bounds=([0, t[0] - 2, 0.1, 0.05], [100, t_pk, 20, 10]))
    return base + amp * gamma_variate(t, a, t0, alpha, beta), i_end


def run_cases(out, acq, th, out_dir):
    rows = []
    for run_id in CASES:
        print(f"[case] {run_id}", flush=True)
        frames = load_frames(acq.loc[run_id, "source"])
        d = np.load(out / "npz" / f"{run_id}.npz")
        t = np.arange(frames.shape[2]) * FRAME_INTERVAL_MS / 1000
        res = {}
        for v in VARIANTS:
            aif = sample_aif(frames, d[f"mask_{v}"])
            yn, _, i_pk, _ = normalise(aif)
            i_cut = i_pk + int(np.argmin(yn[i_pk:])) + 1
            g, i_fit = gamma_aif(aif, t, th)
            versions = {"original": (frames, aif), "cut": (frames[:, :, :i_cut], aif[:i_cut]), "gamma": (frames, g)}
            maps = {m: perfusion(f, a) for m, (f, a) in versions.items()}
            region = maps["original"]["CBV"] > 0
            meds = {m: medians(mp, region) for m, mp in maps.items()}
            for m, (f, a) in versions.items():
                rows.append(dict(run_id=run_id, variant=v, method=m, n_frames=len(a), end_s=float(t[len(a) - 1]),
                                 tail_frac=tail_fraction(a) if m != "gamma" else float(normalise(a)[0][i_pk:].min()),
                                 **meds[m], **{f"change_{k}": meds[m][k] / meds["original"][k] - 1 for k in MAP_KEYS}))
            res[v] = dict(t=t, versions=versions, maps=maps, i_fit=i_fit)
        plot_case(run_id, res, out_dir / f"case_{run_id}.png")
    return pd.DataFrame(rows)


def plot_case(run_id, res, path):
    fig, axes = plt.subplots(len(VARIANTS), 1 + len(METHODS) * 2, figsize=(22, 3.4 * len(VARIANTS)))
    lim = {k: np.nanpercentile(res["ica_top"]["maps"]["original"][k][res["ica_top"]["maps"]["original"]["CBV"] > 0],
                               [2, 98]) for k in ("CBV", "MTT")}
    for row, v in zip(axes, VARIANTS):
        r = res[v]
        ax = row[0]
        for m, c in zip(METHODS, ["k", "C1", "C2"]):
            a = r["versions"][m][1]
            ax.plot(r["t"][:len(a)], a, color=c, lw=1.5 if m == "original" else 1, label=m)
        ax.axvline(r["t"][r["i_fit"]], color="C2", ls=":", lw=0.8)
        ax.set_title(f"{v}: AIF", fontsize=9)
        ax.set_xlabel("s")
        ax.legend(fontsize=7)
        for j, m in enumerate(METHODS):
            for k_i, k in enumerate(("CBV", "MTT")):
                a = row[1 + 2 * j + k_i]
                a.imshow(r["maps"][m][k], cmap="jet", vmin=lim[k][0], vmax=lim[k][1])
                a.set_title(f"{m} {k}", fontsize=9)
                a.axis("off")
    fig.suptitle(run_id)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def md_table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def per_run_reliability(sim, lab, th):
    """aif_floor: per level, runs whose CBV, CBF and MTT all stay within MAX_BIAS (the per-scan criterion of
    de la Rosa et al.), and how many trusted curves the level would reject."""
    f = sim[(sim.scenario == "aif_floor") & (sim.level > 0)]
    tr = lab[lab.trust == "trusted"]
    rows = []
    for level, g in f.groupby("level"):
        ok = (g[[f"bias_{k}" for k in ("CBV", "CBF", "MTT")]].abs().max(axis=1) < th["MAX_BIAS"]).sum()
        rows.append({"tail fraction": level, "runs reliable": f"{ok} / {len(g)} ({ok / len(g):.0%})",
                     "median CBV bias": f"{g.bias_CBV.median():+.1%}", "median MTT bias": f"{g.bias_MTT.median():+.1%}",
                     **{f"trusted {v} rejected": int((tr[tr.variant == v].post_min > level).sum()) for v in VARIANTS},
                     "included ica_top rejected": int(((tr.variant == "ica_top") & tr.included.fillna(False).astype(bool)
                                                       & (tr.post_min > level)).sum())})
    return pd.DataFrame(rows).sort_values("tail fraction", ascending=False)


def write_report(out_dir, summs, thresholds, cases, lab, th, command, n_sim, osd, check, req, rel):
    sim_md = []
    for scen, title in SCENARIOS.items():
        s = summs[scen].copy()
        for k in MAP_KEYS:
            s[f"{k} median"] = s[f"{k} median"].map("{:+.1%}".format)
        sim_md.append(f"**{scen}: {title}** — threshold **{thresholds[scen]}**\n\n{md_table(s.reset_index())}")
    sim_md = "\n\n".join(sim_md)
    c = cases.copy()
    for k in MAP_KEYS:
        c[f"change_{k}"] = c[f"change_{k}"].map("{:+.0%}".format)
        c[k] = c[k].round(3)
    c["tail_frac"] = c.tail_frac.round(2)
    c["run_id"] = c.run_id.str.replace("PRAISE_PRAISE", "PRAISE")
    qc = lab[lab.run_id.isin(CASES)][["run_id", "variant", "category", "trust", "post_min", "end_frac", "tail_slope",
                                      "follow_s", "venous_frac"]].round(3)
    qc["run_id"] = qc.run_id.str.replace("PRAISE_PRAISE", "PRAISE")
    o = cases[cases.method != "original"]
    txt = f"""# AIF truncation — {out_dir.parent.parent.name}

Generated by `scripts/aif_truncation.py`; do not edit by hand, re-run instead.

```bash
{command}
```

Thresholds: {", ".join(f"{k} {v}" for k, v in th.items())}

## A. Simulation: which tail fraction is acceptable?

{n_sim} runs with a trusted ica_top AIF that washes out completely (minimum after the peak <= {th["SIM_MAX_POST_MIN"]},
>= {th["SIM_MIN_FOLLOW_S"]} s after the peak) were made truncated at tail fraction L in the two ways it happens,
and the maps recomputed with the unchanged deconvolution. Bias = median of the modified map over the perfused
pixels / same median of the full run - 1 (median and IQR over the runs). Threshold = highest L with median
|bias| of CBV, CBF and MTT < {th["MAX_BIAS"]:.0%}. Figure: `truncation_sim.png`.

{sim_md}

Reading:
- **short_run** cuts the tissue curves too. The capillary/venous phase comes after the AIF peak, so a run that
  stops while the AIF is still at 30-70 % of its peak loses most of the tissue area: CBV and MTT collapse. This
  is an acquisition requirement (keep recording until the AIF is below the threshold), not an AIF-QC rule.
- **aif_floor** is what the QC's `truncated` category (and the four runs of part B) actually is: a complete run
  whose AIF does not wash out. Its threshold is the one for the AIF QC (aif_qc TRUNC_LEVEL = 0.3).

## A2. Optimal scan duration (method of de la Rosa et al. 2023)

Frames removed one at a time from the end; a version is reliable when CBV, CBF and MTT deviate < {th["MAX_BIAS"]:.0%}
from the full run; OSD = shortest duration from which on every longer version is reliable. Per run
(`truncation_osd.csv`, figure `truncation_osd.png`):

{md_table(osd[["coverage_s", "post_bolus_s", "tail_frac_at_osd"]].describe(percentiles=[0.05, 0.5, 0.95]).round(2).reset_index())}

Requirements that make 95 % of runs reliable: **AIF coverage >= {req[0]:.2f} s** after the AIF peak,
**post-bolus time >= {req[1]:.2f} s** after arrival ({ARRIVAL_FRAC:.0%} of the peak), equivalently the AIF must have
fallen to **<= {req[2]:.2f} of its peak** (P5 of the tail fraction at the OSD).

What the cohort's ica_top curves have (native frames; `truncation_cohort_coverage.csv`):

{md_table(check)}

## Literature basis

No AIF-truncation threshold exists for DSA perfusion with perfDSA: the perfDSA paper (Su et al. 2025, IJCARS,
doi:10.1007/s11548-025-03359-4) does not define one. The threshold here is built from published definitions:

- **Definition of truncation**: incomplete capture of the wash-in or wash-out of the first-pass bolus
  (Borst et al. 2015, PLoS One, doi:10.1371/journal.pone.0119409); the acute-stroke CTP recommendations they cite
  ask for complete wash-out of AIF and VOF. Borst found AIF truncation in 2/36 and VOF truncation in 7/36 short
  (48 s) CTP scans; truncation underestimates CBV because CBV is the ratio of tissue and AIF areas.
- **Reliability criterion (10 %)**: de la Rosa et al. 2023 (PLoS One, doi:10.1371/journal.pone.0283610) call a
  truncated scan reliable if its perfusion lesion volumes deviate < 10 % (or < 5 ml) from the untruncated scan;
  MAX_BIAS = 10 % uses the same bound on the map medians.
- **Method and predictor**: de la Rosa et al. remove the last frame one at a time to find the optimal scan
  duration; the best single predictor of unreliable truncation is the AIF coverage (scan end - AIF peak,
  ROC-AUC 0.98, better than scan duration). In CTP 13 s (mean) / 20 s (P95) of coverage was needed; part A2
  repeats this for DSA.
- **Magnitude**: Copen et al. 2014 (AJNR, doi:10.3174/ajnr.A4186) showed that shortening MR perfusion scans
  falsely lowers CBV by 48-64 % and raises MTT by 133-205 % of normal; the CBV/MTT sensitivity found here
  (CBF much less affected) matches their pattern.
- **DSA-specific**: Lyman et al. 2026 (Transl Stroke Res, doi:10.1007/s12975-026-01420-1) simulated truncation of
  DSA runs and needed >= 7 s of post-bolus data (with gamma-variate extrapolation) for a reliable MTT. Without
  extrapolation (perfDSA) the requirement can only be longer.

## Recommendation

Two questions, two answers:

1. **AIF QC (the curve does not wash out while the run continues; the `truncated` category).** Scenario
   aif_floor, judged with the 10 % reliability bound of de la Rosa et al. Per level, group-level (median) bias and
   per-run reliability, and what the level would cost in currently trusted curves:

{md_table(rel)}

   - **Tail fraction <= {thresholds["aif_floor"]}** keeps the median CBV/MTT bias within {th["MAX_BIAS"]:.0%}
     (the 0.30 used so far gives about -36 % CBV / -30 % MTT). This is the threshold for the AIF QC: the
     analyses compare groups of paired curves, so the group-level bias is the relevant quantity.
   - The per-scan criterion of de la Rosa et al. is stricter: at 0.15 not every run is within 10 %. Use 0.10 as a
     **sensitivity analysis** (repeat the main comparisons with it), not as the primary threshold, because it
     rejects about a third of the trusted curves.

2. **Acquisition length (the whole run stops early; tissue curves are cut as well).** Runs became reliable at a
   median of {osd.post_bolus_s.median():.1f} s after bolus arrival, in line with the >= 7 s Lyman et al. found
   for DSA; for 95 % of runs {req[1]:.1f} s was needed. Many PRAISE runs are shorter (table in A2). This cannot be
   repaired by AIF QC and affects the absolute CBV/MTT values; it largely cancels in the paired comparisons
   (full vs top: same run; plane A vs B: same injection and run length), so report it as a limitation and add
   the post-bolus time per run as a covariate / flag rather than excluding runs.

## B. The obviously truncated runs (meeting 19 Oct)

QC of the four runs (aif_qc + trust):

{md_table(qc)}

Each variant recomputed three ways (`case_<run_id>.png`: AIFs, CBV and MTT maps):
- **cut**: the run ends at the AIF minimum after the peak (removes a re-rising tail; a plateau stays).
- **gamma**: the AIF is replaced by a gamma-variate fit of its first pass (down to {th["GAMMA_FIT_TO"]} of the
  peak, dotted line), which extrapolates the washout to baseline; the tissue curves stay.

Change = median of the map over the perfused pixels of the original / original - 1.

{md_table(c[["run_id", "variant", "method", "n_frames", "end_s", "tail_frac"] + list(MAP_KEYS) + [f"change_{k}" for k in MAP_KEYS]])}

Range of the change over the four runs and three variants:

{md_table(o.groupby("method")[[f"change_{k}" for k in MAP_KEYS]].agg(["min", "median", "max"]).round(3).reset_index().set_axis(
    ["method"] + [f"{k} {s}" for k in MAP_KEYS for s in ("min", "median", "max")], axis=1))}

## Caveats

- aif_floor is a simple model of a lingering tail (a flat floor); a rising venous tail (PRAISE084_A grown) is
  worse than a floor at the same level.
- PRAISE082_B has median Tmax 0.25 s and CBV an order of magnitude above the other runs in every variant: its
  residue functions peak at the first frame, so its maps are unreliable regardless of truncation.
- The gamma fit assumes a single, smooth first pass; for multi-peak curves it is not meaningful.
"""
    (out_dir / "truncation_report.md").write_text(txt, encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="combined result folder, e.g. results/combined_v1")
    for k, v in TH.items():
        p.add_argument(f"--{k.lower().replace('_', '-')}", type=type(v), default=v)
    p.add_argument("--report-only", action="store_true",
                   help="Rebuild figures and report from the saved truncation_sim.csv / truncation_cases.csv.")
    args = p.parse_args()
    th = {k: getattr(args, k.lower()) for k in TH}
    out = args.out
    out_dir = out / "aif_qc" / "truncation"
    out_dir.mkdir(parents=True, exist_ok=True)

    acq = pd.read_csv(out / "acquisition_table.csv").set_index("run_id")
    lab = pd.read_csv(out / "aif_qc" / "aif_qc_labels.csv").merge(
        pd.read_csv(out / "aif_qc" / "taxonomy" / "taxonomy_labels.csv")[["run_id", "variant", "trust"]],
        on=["run_id", "variant"])

    if args.report_only:  # re-use the saved simulation (the full run takes about an hour)
        cases = pd.read_csv(out_dir / "truncation_cases.csv")
        sim = pd.read_csv(out_dir / "truncation_sim.csv")
    else:
        cases = run_cases(out, acq, th, out_dir)
        cases.to_csv(out_dir / "truncation_cases.csv", index=False)
        sim = simulate(out, acq, lab, th)
        sim.to_csv(out_dir / "truncation_sim.csv", index=False)
    summs, thresholds = {}, {}
    for scen in SCENARIOS:
        summs[scen], thresholds[scen] = sim_summary(sim[sim.scenario == scen], th)
        print(f"\n{scen}\n{summs[scen].to_string()}\nthreshold: {thresholds[scen]}")
    plot_sim(sim, thresholds, th, out_dir / "truncation_sim.png")
    osd = optimal_duration(sim, th)
    osd.to_csv(out_dir / "truncation_osd.csv", index=False)
    check, cohort, req = cohort_check(lab, osd, out)
    cohort[["run_id", "plane", "included", "category", "post_min", "coverage_s", "post_bolus_s"]].to_csv(
        out_dir / "truncation_cohort_coverage.csv", index=False)
    plot_osd(osd, cohort, req, out_dir / "truncation_osd.png")
    print(osd.describe().round(3).to_string())
    print(check.to_string())
    rel = per_run_reliability(sim, lab, th)
    print(rel.to_string(index=False))
    write_report(out_dir, summs, thresholds, cases, lab, th, " ".join(["python"] + sys.argv), sim.run_id.nunique(),
                 osd, check, req, rel)


if __name__ == "__main__":
    main()
