"""
Select one run per patient for the analysis, with the reason for every excluded run.

Criteria, in this order (every run gets all reasons that apply):
    1. not a duplicate copy of another run
    2. injection site ICA (CCA and unknown sites are excluded)
    3. plane A or B known, and an ica_top result exists (AIF + maps)
    4. per injection, the plane whose ICA-top AIF mask has the least venous overlap:
         venous_frac   fraction of mask pixels that are still opaque late, i.e. their maximum
                       concentration from `--venous-delay` s after the AIF peak on is more than
                       `--late-ratio` times their maximum up to 1 s after the peak (an artery has
                       washed out by then, a vein crossing the ICA is filling)
         aif_tail      the same late/early ratio of the mask-mean AIF, baseline-corrected
       Lowest venous_frac wins; within `--tie` of each other, the lower aif_tail wins.
       Both metrics are also given for the two ICA-full masks.

Every patient has one injection (one A/B pair), so this leaves at most one run per patient.

Output (in <out>):
    selection.csv       one row per run: included, reasons, evt_phase, venous metrics
    dataset_report.md   '## Selection' section added (or replaced)

Example:
    python scripts/select_runs.py results/combined_v1
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_common import parse_source, rd

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]


def evt_phase(desc):
    d = str(desc).upper()
    if "FINAL" in d or "FIN INJ" in d:
        return "final"
    if re.search(r"POST|PASS|ECR|RETRIEVAL|TREATMENT|STENT|INTERVENTION|VERAPAMIL", d):
        return "post-pass"
    if re.search(r"\bPRE\b|BASELINE|DIAGNOST", d):
        return "pre"
    return "unlabeled"


def venous_metrics(frames, mask, dt_s, venous_delay_s, late_ratio):
    if mask is None or not mask.any():
        return np.nan, np.nan
    conc = 1 - frames[mask] / 170  # (pixels, time), DSASequence's concentration scale
    aif = conc.mean(axis=0)
    i_peak = int(np.argmax(aif))
    i_early = min(len(aif), i_peak + 1 + int(round(1.0 / dt_s)))
    i_late = i_peak + int(round(venous_delay_s / dt_s))
    if i_late >= len(aif):
        return np.nan, np.nan  # run too short after the peak to judge
    base = np.median(aif[:2])
    early = conc[:, :i_early].max(axis=1) - base
    late = conc[:, i_late:].max(axis=1) - base
    frac = float(np.mean(late > late_ratio * np.maximum(early, 1e-6)))
    tail = float((aif[i_late:].max() - base) / (aif[i_peak] - base)) if aif[i_peak] > base else np.nan
    return frac, tail


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="postprocess_combined.py output folder")
    p.add_argument("--venous-delay", type=float, default=2.0, help="Seconds after the AIF peak where 'late' starts.")
    p.add_argument("--late-ratio", type=float, default=0.5, help="Late/early ratio above which a pixel counts as venous.")
    p.add_argument("--tie", type=float, default=0.02, help="venous_frac difference treated as a tie.")
    args = p.parse_args()
    out = Path(args.out)
    acq = pd.read_csv(out / "acquisition_table.csv")
    errors = pd.read_csv(out / "errors.csv").fillna("")
    cache = out / "_cache"

    rows = []
    for i, r in enumerate(acq.to_dict("records"), 1):
        row = {k: r[k] for k in ("run_id", "dataset", "patient_id", "series_number", "series_description", "plane",
                                 "injection_number", "injection_site", "injection_site_source", "ab_pair_id",
                                 "is_duplicate", "duplicate_of")}
        row["evt_phase"] = evt_phase(r["series_description"])
        npz = out / "npz" / f"{r['run_id']}.npz"
        f = cache / f"{r['run_id']}_venous.json"
        d = np.load(npz) if npz.exists() and not r["is_duplicate"] else None
        avail = [v for v in VARIANTS if d is not None and f"mask_{v}" in d]
        cached = json.loads(f.read_text()) if f.exists() else None
        if cached is not None and cached.pop("_variants", None) == avail:
            row.update(cached)
        elif d is not None:
            frames = rd.load_uniform_sequence(rd.read_dicom(parse_source(r["source"])), 250, 1024)
            m = {}
            for v in VARIANTS:
                mask = d[f"mask_{v}"] if f"mask_{v}" in d else None
                m[f"venous_frac_{v}"], m[f"aif_tail_{v}"] = venous_metrics(frames, mask, 0.25, args.venous_delay,
                                                                           args.late_ratio)
            f.write_text(json.dumps({**m, "_variants": avail}))
            row.update(m)
            print(f"[{i}/{len(acq)}] {r['run_id']}: venous_frac {m['venous_frac_ica_top']:.3f}, tail {m['aif_tail_ica_top']:.3f}")
        has_top = npz.exists() and not ((errors.run_id == r["run_id"]) & (errors.variant == "ica_top")).any()

        reasons = []
        if r["is_duplicate"]:
            reasons.append(f"duplicate of {r['duplicate_of']}")
        site = str(r["injection_site"])
        if not site.endswith("ICA"):
            reasons.append("injection site unknown" if site == "unknown" else f"injection site {site} (not ICA)")
        if r["plane"] not in ("A", "B"):
            reasons.append("plane unknown")
        if not has_top:
            reasons.append("no ica_top result (failed run)")
        row["eligible"] = not reasons
        row["reasons"] = reasons
        rows.append(row)

    sel = pd.DataFrame(rows)
    for c in [f"{m}_{v}" for v in VARIANTS for m in ("venous_frac", "aif_tail")]:
        if c not in sel:
            sel[c] = np.nan

    # one plane per patient injection among the eligible runs
    sel["chosen_plane"] = ""
    for (pid, n), g in sel[sel.eligible].groupby(["patient_id", "injection_number"]):
        g = g.assign(_vf=g.venous_frac_ica_top.fillna(np.inf), _tail=g.aif_tail_ica_top.fillna(np.inf))
        best = g.sort_values(["_vf", "_tail"]).iloc[0]
        tied = lambda vf: (vf == best._vf) | ((vf - best._vf).abs() <= args.tie)  # also inf == inf (no metric)
        close = g[tied(g._vf)]
        if len(close) > 1:
            best = close.sort_values("_tail").iloc[0]
        sel.loc[g.index, "chosen_plane"] = best.plane
        for idx in g.index:
            if idx == best.name:
                continue
            o = sel.loc[idx]
            how = (f"aif_tail {best._tail:.3f} vs {o.aif_tail_ica_top:.3f}" if tied(pd.Series([g.loc[idx, "_vf"]])).iloc[0]
                   else f"venous_frac {best._vf:.3f} vs {o.venous_frac_ica_top:.3f}")
            sel.at[idx, "reasons"] = o.reasons + [f"plane {best.plane} has less venous overlap ({how})"]

    sel["included"] = sel.reasons.map(len) == 0
    sel["reasons"] = sel.reasons.map("; ".join)
    sel = sel.drop(columns="eligible")
    sel.to_csv(out / "selection.csv", index=False)

    # report section
    inc = sel[sel.included]
    ex = sel[~sel.included]
    first_reason = ex.reasons.str.split("; ").str[0].str.replace(r" of .*| \(.*", "", regex=True)
    L = ["## Selection", "",
         "One run per patient: ICA injection, not a duplicate, plane known, ica_top result available; of the A/B "
         f"pair the plane with the least venous overlap in the ICA-top AIF mask (venous pixel: max concentration "
         f"≥ {args.venous_delay:g} s after the AIF peak > {args.late_ratio:g} × max up to 1 s after it; ties within "
         f"{args.tie:g} broken by the AIF tail ratio). See `selection.csv`.", "",
         f"**Included: {len(inc)} runs from {inc.patient_id.nunique()} patients** "
         f"(plane A: {int((inc.plane == 'A').sum())}, B: {int((inc.plane == 'B').sum())}; "
         + ", ".join(f"{k}: {v}" for k, v in inc.evt_phase.value_counts().items()) + ").", "",
         f"Excluded: {len(ex)} runs. First reason:", "",
         "| reason | runs |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in first_reason.value_counts().items()]
    L += ["", "EVT phase from SeriesDescription over all unique runs: "
          + ", ".join(f"{k}: {v}" for k, v in sel[~sel.is_duplicate].evt_phase.value_counts().items())
          + " (no pre-EVT runs in the data).", "",
          "Venous overlap of the included runs, median [min–max]: "
          + "; ".join(f"{v} {inc[f'venous_frac_{v}'].median():.3f} [{inc[f'venous_frac_{v}'].min():.3f}–"
                      f"{inc[f'venous_frac_{v}'].max():.3f}]" for v in VARIANTS if inc[f"venous_frac_{v}"].notna().any())
          + ".", "",
          "| run_id | included | reasons |", "|---|---|---|"]
    L += [f"| {r.run_id} | {'yes' if r.included else 'no'} | {r.reasons} |" for r in sel.sort_values("run_id").itertuples()]
    report = out / "dataset_report.md"
    text = report.read_text(encoding="utf-8") if report.exists() else ""
    text = re.sub(r"\n## Selection\n.*", "", text, flags=re.S).rstrip() + "\n\n" + "\n".join(L) + "\n"
    report.write_text(text, encoding="utf-8")
    print(f"Included {len(inc)} runs from {inc.patient_id.nunique()} patients; excluded {len(ex)}.")


if __name__ == "__main__":
    main()
