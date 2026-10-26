"""
Describe the dataset of a combined-cohort result (e.g. results/combined_v1) for the report: what kind of data
it is, how many patients / injections / runs / planes, how the runs were acquired, and how many survive each
selection and QC step. Everything is computed from the result tables plus one DICOM header per patient, so it
can be regenerated after any change.

Input (in <out>):
    acquisition_table.csv   postprocess_combined.py (one row per run, DICOM header fields)
    selection.csv           select_runs.py (included, evt_phase)
    aif_qc/usable_aifs.csv  aif_qc.py (AIF usable per variant)  -- optional, QC rows are skipped without it
    the DICOMs the acquisition table points to (one header per patient, for age/sex/weight/height/procedure;
    cached in data_summary/demographics.csv, delete it or pass --rescan to re-read)
    ../PRAISE_BATCH_2/PRAISE.json (patients listed by Melbourne with their perfusion-evaluation run)

Output (in <out>/data_summary/):
    data_summary.md         setup of the data runs, cohort description, tables, flowchart numbers
    table_cohort.tex        Table I: cohort and acquisition characteristics (booktabs, per batch + total)
    table_planes.tex        acquisition geometry/exposure per plane (A = frontal, B = lateral)
    flowchart.png / .pdf    patients -> runs -> included -> usable AIF
    demographics.csv        per patient: age, sex, weight, height, procedure, system (de-identified IDs)

Example:
    python scripts/data_summary.py results/combined_v1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cohort_common import ROOT, parse_source, rd

BATCHES = {"PRAISE": "Batch 1", "PRAISE_BATCH_2": "Batch 2"}
HEADER_KEYS = ["PatientAge", "PatientSex", "PatientWeight", "PatientSize", "StudyDescription", "StationName",
               "PatientIdentityRemoved", "DeidentificationMethod"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def med_iqr(s, fmt="{:.0f}"):
    s = pd.Series(s, dtype=float).dropna()
    if s.empty:
        return "–"
    q1, m, q3 = s.quantile([0.25, 0.5, 0.75])
    return f"{fmt.format(m)} [{fmt.format(q1)}–{fmt.format(q3)}]"


def med_range(s, fmt="{:.0f}"):
    s = pd.Series(s, dtype=float).dropna()
    return f"{fmt.format(s.median())} ({fmt.format(s.min())}–{fmt.format(s.max())})" if len(s) else "–"


def n_pct(mask, total=None):
    mask = pd.Series(mask).astype(bool)
    total = len(mask) if total is None else total
    return f"{int(mask.sum())} ({100 * mask.sum() / total:.0f}%)" if total else "0"


def md_table(df):
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    rows += ["| " + " | ".join(str(v).replace("\n", " ") for v in r) + " |" for r in df.itertuples(index=False)]
    return rows


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


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
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
        if excl:  # excluded on the arrow into this step
            ax.annotate("", xy=(0.64, y + 0.5 / n), xytext=(0.30, y + 0.5 / n), xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="#9aa5b1", lw=0.6))
            ax.text(0.80, y + 0.5 / n, excl, ha="center", va="center", fontsize=7.5, transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#9aa5b1", lw=0.6))
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------
def write_md(out, dst, dem, acq, inj, cohort, planes, steps, paired, praise):
    pat_planes = acq.groupby("ab_pair_id").plane.agg(lambda p: "".join(sorted(p)))
    proc = dem.procedure.map(procedure_class).value_counts()
    patt = acq.frame_interval_pattern.value_counts()
    L = [f"# Dataset summary — {out.name}", "",
         "Generated by `scripts/data_summary.py`; re-run instead of editing. Patient IDs are the de-identified "
         "PRAISE IDs.", "",
         "```bash", "conda activate BEP", "cd perfDSA-master", f"python scripts/data_summary.py results/{out.name}",
         "```", "",
         "## What kind of data", "",
         f"- **Clinical population: acute ischaemic stroke patients treated with endovascular thrombectomy (EVT).** "
         f"No healthy controls. The DICOM study description is *ANGIO Neck Clot Retrieval* for "
         f"{proc.get('clot retrieval (EVT)', 0)} patients and *ANGIO Cerebral DSA* for "
         f"{proc.get('cerebral DSA', 0)} (the generic code for a neuro-interventional session). The series "
         f"descriptions place {int((inj.evt_phase != 'unlabeled').sum())} of {len(inj)} injections during or "
         f"after the procedure (\"POST 1ST PASS\", \"POST 5TH PASS\", \"FINAL\", \"POST STENT/PLASTY\", "
         f"\"POST VERAPAMIL\"); the other {int((inj.evt_phase == 'unlabeled').sum())} only name the vessel "
         f"(e.g. \"LEFT ICA\"), so their moment in the procedure is not recorded.",
         "- **Source:** PRAISE study, de-identified by the University of Melbourne "
         f"(DICOM `PatientIdentityRemoved = YES`, {dem.deidentified_by.nunique()} de-identification source). "
         f"Received as two batches (batch 1: {acq[acq.dataset == 'PRAISE'].patient_id.nunique()} patients, "
         f"batch 2: {acq[acq.dataset == 'PRAISE_BATCH_2'].patient_id.nunique()}); `PRAISE.json` lists "
         f"{len(praise)} patients with the run chosen for perfusion evaluation (`DSA_perf_eval`).",
         f"- **One injection per patient**: exactly the perfusion-evaluation run, acquired **biplane**: every "
         f"injection has a plane A (frontal/AP, primary angle ~0°) and a plane B (lateral, ~−90°) run of the "
         f"same contrast bolus ({int((pat_planes == 'AB').sum())} of {len(pat_planes)} injections are complete "
         f"A+B pairs). No pre-EVT (baseline) runs: the timing is after one or more passes or the final run.",
         f"- **System:** Siemens AXIOM-Artis biplane, {dem.system.nunique()} rooms "
         + " / ".join(f"{k}: {v} patients" for k, v in dem.system.value_counts().items())
         + f"; studies {pd.to_datetime(acq.study_date.astype(str)).min():%b %Y}–"
           f"{pd.to_datetime(acq.study_date.astype(str)).max():%b %Y}.",
         f"- **Images:** vendor-subtracted DSA (stored as subtracted, display LUT applied), "
         f"{int(acq.rows.iloc[0])}×{int(acq["columns"].iloc[0])} px, {int(acq.bits_stored.iloc[0])} bit, iodine contrast. "
         f"Contrast volume and injection rate are **not** in the headers.",
         f"- **Variable frame rate in every run**: 4 fps for the first 11 intervals (2.75 s), then 2 fps, and in "
         f"{(acq.frame_rate_regime == '4-2-1 fps').sum()} of {len(acq)} runs 1 fps at the end "
         f"({patt.index[0]} is the most common pattern). Runs last {med_iqr(acq.duration_s, '{:.2f}')} s "
         f"(median [IQR]), {med_iqr(acq.n_frames)} frames.", "",
         "## How the data runs are set up", "",
         "Each step writes into `results/<name>/` and the next step reads its tables:", "",
         "1. `run_dataset.py -i ../PRAISE ../PRAISE_BATCH_2 -o results/combined_v1/raw_model --ica-full model "
         "--save-maps` and the same with `--ica-full grown` into `raw_grown`. Every multi-frame DICOM (also inside "
         "the .tar archives) is one **run**. Per run: read header → resample to a uniform 250 ms grid → ICA-top "
         "U-Net mask → AIF = mean of the mask over time → deconvolution → CBV/CBF/MTT/Tmax. The ICA-top AIF is "
         "computed in both raw runs (identical masks in all runs).",
         "2. `scripts/postprocess_combined.py results/combined_v1` merges both into **three AIF variants** "
         "(`ica_top`, `ica_full_model`, `ica_full_grown`), adds plane, injection site, A/B pairing, frame-rate "
         "pattern and duplicate detection (`acquisition_table.csv`), the native-timing AIFs (`aif_curves.csv`), "
         "per-run bundles (`npz/`) and `dataset_report.md`.",
         "3. `scripts/mask_overlap.py` adds the Dice/AIF correlation between the three masks.",
         "4. `scripts/select_runs.py` labels the EVT phase and picks one plane per patient for the patient-level "
         "analysis (ICA injection, not a duplicate; the plane with the least venous overlap in the ICA-top mask) "
         "→ `selection.csv`.",
         "5. `scripts/build_features.py` → `features*.csv` (AIF curve features per run × variant, paired A/B and "
         "full-vs-top tables).",
         "6. `scripts/aif_qc.py` → `aif_qc/` (AIF failure categories and `usable_aifs.csv`).",
         "7. `scripts/data_summary.py` (this summary) and `scripts/build_viewer.py` (interactive viewer).", "",
         "Units of analysis: **patient** (68) = **injection** (1 per patient) → **run** (plane A, plane B) → "
         "**AIF curve** (3 mask variants per run). A/B runs of one injection share the bolus, so A vs B is a "
         "paired comparison; the three variants of one run share the frames, so full vs top is paired as well.", "",
         "## Table I — cohort and acquisition", "", *md_table(cohort), "",
         "Age and sex from the de-identified DICOM header (`PatientAge`, `PatientSex`); weight only where the "
         "header has it. Injection site and EVT phase parsed from the series description; *unknown* = the "
         "description has no vessel (e.g. \"FINAL\", \"POST 1ST PASS\"). Values: median [IQR] or n (%).", "",
         "## Acquisition per plane", "", *md_table(planes), "",
         "## Flowchart (patients → analysed runs)", "", "![flowchart](flowchart.png)", "",
         *md_table(pd.DataFrame(steps, columns=["step", "remaining", "excluded"])), ""]
    if paired:
        L += ["Paired sets (ICA injections with a usable AIF in **both** planes, for A vs B):", "",
              *md_table(pd.DataFrame([(k, v) for k, v in paired.items()], columns=["set", "injections"])), ""]
    L += ["## Notes for the Methods section", "",
          "- Name the population precisely: AIS patients undergoing EVT, intra-/post-procedural control runs, "
          "not diagnostic or healthy-volunteer DSA. Perfusion may be partly restored, and the injected side is normally the "
          "treated side (not recorded explicitly).",
          "- Biplane gives two simultaneous projections of the **same** bolus; the header time resolution is 1 s, "
          "so the A/B offset is only known to < ~1 s (bolus arrival differs by a median 68 ms, see "
          "`dataset_report.md`).",
          "- CCA injections fill the ECA as well (overlap over the ICA); they are kept in the run-level tables "
          "but excluded from the patient-level selection.",
          "- Missing per-patient data: contrast volume/rate, occlusion site, NIHSS, eTICI and outcome are not in "
          "the DICOMs; ask the PRAISE team if the report needs them.",
          "- Table files for LaTeX: `table_cohort.tex`, `table_planes.tex` (need `booktabs`; both are already "
          "loaded in `report/main.tex`), figure `flowchart.pdf`.", ""]
    (dst / "data_summary.md").write_text("\n".join(L), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="combined result folder, e.g. results/combined_v1")
    p.add_argument("--praise-json", type=Path, default=ROOT.parent / "PRAISE_BATCH_2" / "PRAISE.json")
    p.add_argument("--rescan", action="store_true", help="re-read the DICOM headers for the demographics")
    args = p.parse_args()

    dst = args.out / "data_summary"
    dst.mkdir(exist_ok=True)
    acq_all = pd.read_csv(args.out / "acquisition_table.csv")
    acq = acq_all[~acq_all.is_duplicate].copy()
    sel = pd.read_csv(args.out / "selection.csv")
    qc = args.out / "aif_qc" / "usable_aifs.csv"
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
    write_md(args.out, dst, dem, acq, inj, cohort, planes, steps, paired, praise)
    print(cohort.to_string(index=False))
    print()
    for s in steps:
        print(" -> ".join(x for x in s if x))
    print(paired)


if __name__ == "__main__":
    main()
