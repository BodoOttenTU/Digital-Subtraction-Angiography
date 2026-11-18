"""Cohort description for the report: counts, acquisition parameters, flowchart, cohort tables.

    python scripts/data_summary.py
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cohort_common import ROOT, parse_source, rd
from paths import DATA

BATCHES = {"PRAISE": "Batch 1", "PRAISE_BATCH_2": "Batch 2"}
HEADER_KEYS = ["PatientAge", "PatientSex", "PatientWeight", "PatientSize", "StudyDescription", "StationName",
               "PatientIdentityRemoved", "DeidentificationMethod"]


def med_iqr(s, fmt="{:.0f}"):
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return "–"
    q1, m, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{fmt.format(m)} [{fmt.format(q1)}–{fmt.format(q3)}]"


def n_pct(mask, total=None):
    mask = pd.Series(mask).astype(bool)
    total = len(mask) if total is None else total
    return f"{int(mask.sum())} ({100 * mask.sum() / total:.0f}%)" if total else "0"


def tex_escape(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("%", "\\%").replace("&", "\\&")
            .replace("_", "\\_").replace("#", "\\#").replace("–", "--").replace("×", "$\\times$")
            .replace("≥", "$\\geq$").replace("→", "$\\rightarrow$").replace("°", "$^\\circ$")
            .replace("−", "$-$"))


def tex_table(df, caption, label, colspec=None):
    colspec = colspec or "l" + "c" * (len(df.columns) - 1)
    L = ["\\begin{table}[!t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         "\\footnotesize", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
         " & ".join(f"\\textbf{{{tex_escape(c)}}}" for c in df.columns) + " \\\\", "\\midrule"]
    for r in df.itertuples(index=False):
        cells = [tex_escape(v) for v in r]
        if all(v == "" for v in cells[1:]):  # section header row
            L.append(f"\\multicolumn{{{len(cells)}}}{{l}}{{\\textit{{{cells[0]}}}}} \\\\")
        else:
            L.append(" & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(L)


def demographics(acq, cache, rescan):
    if cache.exists() and not rescan:
        return pd.read_csv(cache, dtype={"patient_id": str})
    rows = []
    for r in acq.drop_duplicates("patient_id").itertuples():
        ds = rd.read_dicom(parse_source(r.source), header_only=True)
        h = {k: str(ds.get(k, "") or "") for k in HEADER_KEYS}
        age = h["PatientAge"]
        rows.append(dict(patient_id=r.patient_id, dataset=r.dataset,
                         age_y=float(age[:3]) if age[:3].isdigit() and age.endswith("Y") else np.nan,
                         sex=h["PatientSex"] or "unknown",
                         weight_kg=pd.to_numeric(h["PatientWeight"], errors="coerce"),
                         height_m=pd.to_numeric(h["PatientSize"], errors="coerce"),
                         procedure=h["StudyDescription"], system=h["StationName"],
                         identity_removed=h["PatientIdentityRemoved"], deidentified_by=h["DeidentificationMethod"]))
    dem = pd.DataFrame(rows)
    dem.to_csv(cache, index=False)
    return dem


def procedure_class(s):
    s = str(s).lower()
    return "clot retrieval (EVT)" if "clot retrieval" in s else "cerebral DSA" if "cerebral" in s else "other"


def injection_level(acq, sel):
    """One row per injection (A/B pair), from the plane-A row (site, phase and timing are shared)."""
    inj = acq.sort_values("plane").drop_duplicates("ab_pair_id").merge(
        sel[["run_id", "evt_phase"]], on="run_id", how="left")
    inj["site_vessel"] = inj.injection_vessel.fillna("unknown").replace("", "unknown")
    inj["site_side"] = inj.injection_side.fillna("unknown").replace("", "unknown")
    return inj


def cohort_table(dem, acq, inj):
    groups = [(BATCHES[d], d) for d in BATCHES if d in set(acq.dataset)] + [("Total", None)]
    rows = []

    def add(label, fn):
        rows.append([label] + [fn(d) for _, d in groups])

    sub = lambda df, d: df if d is None else df[df.dataset == d]
    add("Patients", lambda d: str(sub(dem, d).patient_id.nunique()))
    add("Age, years", lambda d: med_iqr(sub(dem, d).age_y))
    add("Age range, years", lambda d: f"{sub(dem, d).age_y.min():.0f}–{sub(dem, d).age_y.max():.0f}")
    add("Male sex", lambda d: n_pct(sub(dem, d).sex == "M"))
    add("Weight, kg (recorded)", lambda d: f"{med_iqr(sub(dem, d).weight_kg)} (n={sub(dem, d).weight_kg.notna().sum()})")
    add("Procedure: clot retrieval", lambda d: n_pct(sub(dem, d).procedure.map(procedure_class) == "clot retrieval (EVT)"))
    add("Study period", lambda d: f"{pd.to_datetime(sub(acq, d).study_date.astype(str)).dt.year.min()}–"
                                  f"{pd.to_datetime(sub(acq, d).study_date.astype(str)).dt.year.max()}")
    rows.append(["Injections", "", "", ""][:len(groups) + 1])
    add("Injections (biplane A+B)", lambda d: str(len(sub(inj, d))))
    add("Per patient", lambda d: str(int(sub(inj, d).groupby("patient_id").size().max())))
    for v in ("ICA", "CCA", "unknown"):
        add(f"Site: {v}", lambda d, v=v: n_pct(sub(inj, d).site_vessel == v))
    add("Side: left / right / unknown", lambda d: " / ".join(
        str(int((sub(inj, d).site_side == s).sum())) for s in ("L", "R", "unknown")))
    for p in ("final", "post-pass", "unlabeled"):
        add(f"EVT phase: {p}", lambda d, p=p: n_pct(sub(inj, d).evt_phase == p))
    rows.append(["Runs", "", "", ""][:len(groups) + 1])
    add("Runs (plane A / B)", lambda d: f"{len(sub(acq, d))} ({(sub(acq, d).plane == 'A').sum()} / "
                                        f"{(sub(acq, d).plane == 'B').sum()})")
    add("Frames per run", lambda d: med_iqr(sub(acq, d).n_frames))
    add("Duration, s", lambda d: med_iqr(sub(acq, d).duration_s, "{:.2f}"))
    add("Frame-rate scheme 4–2–1 fps", lambda d: n_pct(sub(acq, d).frame_rate_regime == "4-2-1 fps"))
    add("Detector field of view 22 cm", lambda d: n_pct(sub(acq, d).field_of_view_mm.round() == 222))
    return pd.DataFrame(rows, columns=["Characteristic"] + [g for g, _ in groups])


def plane_table(acq):
    spec = [("Primary angle (LAO+/RAO−), °", "primary_angle_deg", "{:.0f}"),
            ("Secondary angle (CRA+/CAU−), °", "secondary_angle_deg", "{:.0f}"),
            ("Tube voltage, kVp", "kvp", "{:.1f}"), ("Tube current, mA", "tube_current_mA", "{:.0f}"),
            ("Pulse width, ms", "pulse_width_ms", "{:.1f}"), ("Source–image distance, mm", "sid_mm", "{:.0f}"),
            ("Geometric magnification", "magnification", "{:.2f}"),
            ("Dose–area product (DICOM units)", "dap", "{:.0f}")]
    rows = [[lab] + [med_iqr(acq.loc[acq.plane == p, col], f) for p in ("A", "B")] for lab, col, f in spec]
    rows.append(["Field of view, mm"] + [" / ".join(f"{k:.0f}: {v}" for k, v in acq.loc[acq.plane == p,
                                                    "field_of_view_mm"].value_counts().items()) for p in ("A", "B")])
    return pd.DataFrame(rows, columns=["Parameter", "Plane A (frontal)", "Plane B (lateral)"])


def flow_counts(praise, acq_all, acq, sel, use):
    n_dup = int(acq_all.is_duplicate.sum())
    ica = sel[~sel.is_duplicate & sel.injection_site.astype(str).str.endswith("ICA")]
    # (step, what remains, what was excluded on the way into this step)
    site = acq.sort_values("plane").drop_duplicates("ab_pair_id").injection_vessel.fillna("unknown")
    inc = sel[sel.included]
    steps = [("Patients listed in PRAISE.json (perfusion-evaluation run)", f"{len(praise)} patients", ""),
             ("Patients with data (batch 1 + batch 2)", f"{acq.patient_id.nunique()} patients, {len(acq_all)} DICOM runs",
              f"{len(praise) - acq.patient_id.nunique()} patients without imaging data"),
             ("Unique biplane injections (1 per patient)", f"{acq.ab_pair_id.nunique()} injections, {len(acq)} runs (A+B)",
              f"{n_dup} duplicate runs (identical pixel data)"),
             ("ICA injection", f"{ica.patient_id.nunique()} patients, {len(ica)} runs",
              f"injections: CCA {int((site == 'CCA').sum())}, site unknown {int(site.isin(['unknown', '']).sum())}"),
             ("One plane per patient (least venous overlap)",
              f"{len(inc)} runs (A {int((inc.plane == 'A').sum())}, B {int((inc.plane == 'B').sum())})",
              f"{len(ica) - len(inc)} runs of the other plane\n(kept for the A vs B analysis)")]
    if use is not None:
        u = use[use.included]
        steps.append(("Usable ICA-top AIF (aif_qc.py)", f"{int(u.use_primary.sum())} runs",
                      f"{int((~u.use_primary).sum())} runs failed AIF QC"))
    return steps


def paired_counts(use):
    """Patients with an ICA injection in which both planes have a usable AIF, per variant (SQ2)."""
    u = use[~use.is_duplicate & use.injection_site.astype(str).str.endswith("ICA")]
    out = {}
    for v in ("ica_top", "ica_full_model", "ica_full_grown"):
        ok = u.groupby("ab_pair_id")[f"usable_{v}"].agg(lambda s: len(s) == 2 and s.all())
        out[v] = int(ok.sum())
    out["all_three_both_planes"] = int(u.groupby("ab_pair_id").apply(
        lambda g: len(g) == 2 and (g.n_usable_variants == 3).all(), include_groups=False).sum())
    out["n_ica_injections"] = int(u.ab_pair_id.nunique())
    return out


def plot_flow(steps, path):
    fig, ax = plt.subplots(figsize=(7, 0.95 * len(steps)))
    ax.axis("off")
    n = len(steps)
    for i, (title, count, excl) in enumerate(steps):
        y = 1 - (i + 0.5) / n
        ax.text(0.30, y, f"{title}\n{count}", ha="center", va="center", fontsize=8.5, transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.45", fc="#eef2f7", ec="#4a5d73", lw=0.8))
        if i < n - 1:
            ax.annotate("", xy=(0.30, y - 0.68 / n), xytext=(0.30, y - 0.32 / n), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#4a5d73", lw=0.8))
        if excl:
            ax.annotate("", xy=(0.64, y + 0.5 / n), xytext=(0.30, y + 0.5 / n), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#9aa5b1", lw=0.6))
            ax.text(0.80, y + 0.5 / n, excl, ha="center", va="center", fontsize=7.5, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9aa5b1", lw=0.6))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    p.add_argument("--praise-json", type=Path, default=ROOT.parent / "PRAISE_BATCH_2" / "PRAISE.json")
    p.add_argument("--rescan", action="store_true", help="re-read the DICOM headers for the demographics")
    args = p.parse_args()

    dst = args.out / "cohort"
    dst.mkdir(exist_ok=True)
    acq_all = pd.read_csv(args.out / "acquisition_table.csv")
    acq = acq_all[~acq_all.is_duplicate].copy()
    sel = pd.read_csv(args.out / "selection.csv")
    qc = args.out / "qc" / "usable_aifs.csv"
    use = pd.read_csv(qc) if qc.exists() else None
    praise = json.loads(args.praise_json.read_text())

    dem = demographics(acq, dst / "demographics.csv", args.rescan)
    inj = injection_level(acq, sel)
    cohort = cohort_table(dem, acq, inj)
    planes = plane_table(acq)
    steps = flow_counts(praise, acq_all, acq, sel, use)
    paired = paired_counts(use) if use is not None else {}

    (dst / "table_cohort.tex").write_text(tex_table(
        cohort, "Cohort and acquisition characteristics. Median [IQR] or n (\\%).", "tab:cohort",
        "l" + "c" * (len(cohort.columns) - 1)), encoding="utf-8")
    (dst / "table_planes.tex").write_text(tex_table(
        planes, "Acquisition geometry and exposure per plane (all unique runs). Median [IQR].", "tab:planes"),
        encoding="utf-8")
    plot_flow(steps, dst / "flowchart")
    print(cohort.to_string(index=False))
    print()
    for s in steps:
        print(" -> ".join(x for x in s if x))
    print(paired)


if __name__ == "__main__":
    main()
