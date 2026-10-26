"""
Segmentation failure taxonomy and objective QC criteria (trust flag) for the AIFs of a combined-cohort
result, on top of aif_qc.py (run that first).

aif_qc.py labels every curve by its *shape* (truncated, multi_peak, ...) and names the most likely cause.
This script asks the next question: is the failure due to the segmentation, and if so how?
Every curve (run x variant) gets one taxonomy class:

    ok                 the curve passes aif_qc.py
    wrong_structure    the mask is not on the ICA: it is scattered over the head (>= SCATTER_COMPONENTS
                       pieces with a spatial spread > SCATTER_SPREAD_PX, e.g. blobs on cortical veins or a
                       grown mask leaking into the sinus) or it lies at the image edge (border_frac >
                       BORDER_MAX, ICA masks: the supraclinoid ICA never lies at the edge of a head DSA)
    too_small          mask_px < MASK_MIN_PX: too few pixels averaged
    split_segments     ICA mask in separate pieces (largest connected component < LARGEST_MIN of the
                       mask), typically the supraclinoid plus a cervical/petrous ICA piece. Grown masks
                       are fragmented by design and are not tested. Does not override a bolus-wide defect
                       (aif_qc cause class injection / bolus).
    projection_overlap the mask is on the ICA but in this projection other vessels overlap it: most mask
                       pixels opacify late (venous_frac >= VENOUS_OVERLAP, veins/sinus over the siphon in
                       the lateral view), or aif_qc found a projection-specific defect (cause class
                       projection / venous-late overlap / ECA overlap)
    not_segmentation   the defect is in the bolus or the acquisition, not the mask (aif_qc cause class
                       injection / bolus, acquisition too short, contrast / noise, baseline / motion,
                       borderline, or mask content on a geometrically normal mask)

Order: wrong_structure > too_small > split_segments > projection_overlap > not_segmentation. The
geometric rules come first because a broken mask makes the cause analysis of aif_qc.py moot. The rules
were set on the MINIP overlays of every failed and flagged curve (grids below).
The same mask checks also run on the curves that pass ("silent" mask problems: the curve looks fine but
averages the wrong pixels); they are reported separately and feed the trust flag.

Objective QC criteria (trust flag)
    Every criterion is one measurable quantity with one threshold, and each threshold has a stated
    basis (physics, the curve-shape QC, a natural gap in the cohort distribution, or the visual check).
        criterion    quantity                                                        pass     level
        n_peaks      first-pass peaks (prominence >= 0.08 of the peak, aif_qc.py)   <= 1     reject
        tail         minimum after the peak / peak (aif_qc post_min)                 <= 0.30  reject
        sat          fraction of mask pixels saturated in the MINIP                  <= 0.10  reject
        mask_px      pixels in the mask                                              >= 500   reject
        scattered    >= 3 components and spread > 200 px                             no       reject
        border       mask pixels within 40 px of the image edge (ICA masks)          <= 0.05  reject
        components   largest component / mask (ICA masks)                            >= 0.8   review
        venous       late-opacifying fraction of the mask (select_runs.py)           <  0.5   review
    trust = "reject" if the shape QC or a reject criterion fails, else "review" if a review criterion
    fails (the curve passes but the mask is split or overprojected: check before use), else "trusted".
    The sensitivity table recounts every criterion at a stricter and a more lenient threshold.

Input : <out>/aif_qc/aif_qc_labels.csv, npz/<run_id>.npz (masks, MINIP)
Output: <out>/aif_qc/taxonomy/
    taxonomy_labels.csv      one row per curve: taxonomy class, mask geometry, every criterion, trust
    table_taxonomy.tex       Table II: class x (variant, plane), all unique runs
    table_criteria.tex       QC criteria, thresholds, basis, number of curves failing
    taxonomy_report.md       counts, criteria, sensitivity, silent mask problems, per-curve list
    grid_<class>.png         MINIP + mask contour of every failed curve of that class (grid_silent.png:
                             passing curves with a mask problem)
    criteria_hist.png        distribution of every criterion with its threshold

Example:
    python scripts/aif_taxonomy.py results/combined_v1
"""

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import spearmanr

VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
ICA_MASKS = ["ica_top", "ica_full_model"]
CLASSES = ["ok", "wrong_structure", "too_small", "split_segments", "projection_overlap", "not_segmentation"]
CLASS_LABEL = {"ok": "OK (passes QC)", "wrong_structure": "Wrong structure", "too_small": "Too small",
               "split_segments": "Split segments", "projection_overlap": "Projection overlap",
               "not_segmentation": "Not segmentation"}
PROJECTION_CAUSES = {"projection (overlap / motion)", "venous / late overlap", "ECA overlap (CCA injection)"}

TH = dict(
    VENOUS_OVERLAP=0.5,     # majority of the mask opacifies late: veins projected over it
    BORDER_PX=40,           # width of the image-border band (of 1024)
    BORDER_MAX=0.05,        # ICA masks with more than this fraction in the border band are off target
    SCATTER_COMPONENTS=3,   # this many pieces or more ...
    SCATTER_SPREAD_PX=200,  # ... spread (RMS distance from the centroid) over more than this: scattered
    MASK_MIN_PX=500,        # aif_qc MASK_SMALL_PX; ~1/3 of the median ICA-top mask
    LARGEST_MIN=0.8,        # ICA masks: largest component below this = mask in pieces
    TAIL_MAX=0.3,           # aif_qc TRUNC_LEVEL
    SAT_MAX=0.10,           # natural gap in sat_frac: 0.06 -> 0.82
    N_PEAKS_MAX=1,
)
# criterion -> (threshold it uses, stricter value, more lenient value) for the sensitivity table
SENSITIVITY = {"fail_n_peaks": ("N_PEAKS_MAX", 1, 2), "fail_tail": ("TAIL_MAX", 0.2, 0.4),
               "fail_sat": ("SAT_MAX", 0.05, 0.5), "fail_mask_px": ("MASK_MIN_PX", 750, 250),
               "fail_scattered": ("SCATTER_SPREAD_PX", 150, 250), "fail_border": ("BORDER_MAX", 0.02, 0.10),
               "fail_components": ("LARGEST_MIN", 0.9, 0.6), "fail_venous": ("VENOUS_OVERLAP", 0.4, 0.6)}
REJECT = ["fail_n_peaks", "fail_tail", "fail_sat", "fail_mask_px", "fail_scattered", "fail_border"]
REVIEW = ["fail_components", "fail_venous"]


# ----------------------------------------------------------------------------
# Mask geometry
# ----------------------------------------------------------------------------
def mask_geometry(mask, border_px):
    n, _, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))
    areas = stats[1:, cv.CC_STAT_AREA]
    band = np.zeros_like(mask)
    band[:border_px] = band[-border_px:] = True
    band[:, :border_px] = band[:, -border_px:] = True
    ys, xs = np.nonzero(mask)
    px = int(mask.sum())
    return dict(geo_px=px, n_components=n - 1,
                largest=float(areas.max() / areas.sum()) if areas.size else np.nan,
                border_frac=float((mask & band).sum() / px) if px else np.nan,
                spread_px=float(np.hypot(ys.std(), xs.std())) if px else np.nan)


def geometry_table(out, run_ids, th):
    rows = []
    for r in run_ids:
        d = np.load(out / "npz" / f"{r}.npz")
        for v in VARIANTS:
            rows.append(dict(run_id=r, variant=v, **mask_geometry(d[f"mask_{v}"], th["BORDER_PX"])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Criteria, taxonomy, trust
# ----------------------------------------------------------------------------
def criteria(df, th):
    """Boolean fail column per criterion."""
    ica = df.variant.isin(ICA_MASKS)
    return pd.DataFrame({
        "fail_n_peaks": df.n_peaks > th["N_PEAKS_MAX"],
        "fail_tail": df.post_min > th["TAIL_MAX"],
        "fail_sat": df.sat_frac > th["SAT_MAX"],
        "fail_mask_px": df.mask_px < th["MASK_MIN_PX"],
        "fail_scattered": (df.n_components >= th["SCATTER_COMPONENTS"]) & (df.spread_px > th["SCATTER_SPREAD_PX"]),
        "fail_border": ica & (df.border_frac > th["BORDER_MAX"]),
        "fail_components": ica & (df.largest < th["LARGEST_MIN"]),
        "fail_venous": df.venous_frac >= th["VENOUS_OVERLAP"],
    }, index=df.index)


def mask_problem(r):
    """Geometric mask class (independent of the curve), or '' for a normal mask."""
    if r.fail_scattered or r.fail_border:
        return "wrong_structure"
    if r.fail_mask_px:
        return "too_small"
    if r.fail_components:
        return "split_segments"
    if r.fail_venous:
        return "projection_overlap"
    return ""


def taxonomy_class(r):
    if r.category == "usable":
        return "ok"
    if r.mask_problem in ("wrong_structure", "too_small", "projection_overlap"):
        return r.mask_problem
    if r.mask_problem == "split_segments" and r.cause_class != "injection / bolus":
        return r.mask_problem
    return "projection_overlap" if r.cause_class in PROJECTION_CAUSES else "not_segmentation"


def taxonomy_reason(r, th):
    if r.taxonomy == "ok":
        return ""
    if r.taxonomy == "wrong_structure":
        why = (f"{r.border_frac:.0%} of the mask in the {th['BORDER_PX']} px image border" if r.fail_border
               else f"scattered: {r.n_components} pieces, spread {r.spread_px:.0f} px")
    elif r.taxonomy == "too_small":
        why = f"{r.mask_px:.0f} px (< {th['MASK_MIN_PX']})"
    elif r.taxonomy == "split_segments":
        why = f"{r.n_components} components, largest {r.largest:.0%} of the mask"
    elif r.taxonomy == "projection_overlap":
        why = (f"venous_frac {r.venous_frac:.2f}: veins projected over the mask" if r.fail_venous
               else f"mask on the ICA, defect from this projection ({r.cause_class})")
    else:
        why = f"mask normal or not the cause, defect from the bolus / acquisition ({r.cause_class})"
    return f"{r.category}: {why}"


def trust(r):
    if r.category != "usable" or any(r[c] for c in REJECT):
        return "reject"
    return "review" if any(r[c] for c in REVIEW) else "trusted"


def sensitivity_table(lab, th):
    rows = []
    for col, (key, strict, lenient) in SENSITIVITY.items():
        n = {name: int(criteria(lab, {**th, key: val})[col].sum())
             for name, val in (("strict", strict), ("default", th[key]), ("lenient", lenient))}
        rows.append(dict(criterion=col.removeprefix("fail_"), threshold=key, strict=f"{strict}: {n['strict']}",
                         default=f"{th[key]}: {n['default']}", lenient=f"{lenient}: {n['lenient']}"))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------
def count_table(lab):
    """Table II: class x (variant, plane)."""
    t = pd.crosstab(lab.taxonomy, [lab.variant, lab.plane]).reindex(
        index=CLASSES, columns=pd.MultiIndex.from_product([VARIANTS, ["A", "B"]]), fill_value=0)
    t.loc["total"] = t.sum()
    return t


def tex_taxonomy(t, n_runs, path):
    head = " & ".join(f"\\multicolumn{{2}}{{c}}{{{v.replace('_', chr(92) + '_')}}}" for v in VARIANTS)
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{AIF failure taxonomy per mask variant and plane ({n_runs} unique runs). Each curve gets "
             "one class; failed curves are attributed to the segmentation (wrong structure, too small, split "
             "segments), to the projection, or to the bolus/acquisition (not segmentation).}",
             "\\label{tab:taxonomy}", "\\begin{tabular}{l" + "rr" * len(VARIANTS) + "}", "\\hline",
             f"Class & {head} \\\\", " & " + " & ".join(["A", "B"] * len(VARIANTS)) + " \\\\", "\\hline"]
    for k, row in t.iterrows():
        if k == "total":
            lines.append("\\hline")
        lines.append(f"{CLASS_LABEL.get(k, 'Total')} & " + " & ".join(str(int(x)) for x in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


CRITERIA_TEXT = [  # column, name, pass (LaTeX), basis, level
    ("fail_n_peaks", "First-pass peaks", "$\\leq$ 1", "one injection = one bolus", "reject"),
    ("fail_tail", "Tail fraction (min after peak / peak)", "$\\leq$ 0.30", "shape QC; truncation analysis", "reject"),
    ("fail_sat", "Saturated mask pixels", "$\\leq$ 10\\%", "natural gap (6\\% $\\rightarrow$ 82\\%)", "reject"),
    ("fail_mask_px", "Mask size", "$\\geq$ 500 px", "noise $\\propto 1/\\sqrt{N}$; 1/3 of median", "reject"),
    ("fail_scattered", "Scattered mask", "not $\\geq$3 parts over $>$200 px", "visual check", "reject"),
    ("fail_border", "Mask in image border (ICA masks)", "$\\leq$ 5\\%", "ICA never at the image edge", "reject"),
    ("fail_components", "Largest component (ICA masks)", "$\\geq$ 80\\%", "supraclinoid ICA is one segment", "review"),
    ("fail_venous", "Late-opacifying mask fraction", "$<$ 50\\%", "veins over the mask", "review"),
]


def tex_criteria(lab, path):
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Objective AIF QC criteria (trust flag). Reject: the AIF is not used; review: the curve "
             "passes but the mask is split or overprojected. Failing: number of curves failing the criterion, "
             f"of {len(lab)} unique curves.}}",
             "\\label{tab:qc_criteria}", "\\begin{tabular}{lllcr}", "\\hline",
             "Criterion & Pass & Basis & Level & Failing \\\\", "\\hline"]
    for col, name, thr, basis, level in CRITERIA_TEXT:
        lines.append(f"{name} & {thr} & {basis} & {level} & {int(lab[col].sum())} \\\\")
    n = lab.trust.value_counts()
    lines += ["\\hline", f"\\multicolumn{{4}}{{l}}{{Trusted / review / reject (incl.\\ shape QC)}} & "
              f"{n.get('trusted', 0)} / {n.get('review', 0)} / {n.get('reject', 0)} \\\\",
              "\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def short(run_id):
    return run_id.replace("PRAISE_BATCH_2_", "B2_").replace("PRAISE_", "")


def plot_grid(out, rows, title, path, ncol=5):
    if rows.empty:
        return
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        d = np.load(out / "npz" / f"{r.run_id}.npz")
        ax.imshow(d["minip"], cmap="gray")
        ax.contour(d[f"mask_{r.variant}"], [0.5], colors="r", linewidths=0.7)
        ax.set_title(f"{short(r.run_id)}\n{r.variant} | {r.category}", fontsize=7)
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)


def plot_criteria(lab, th, path):
    spec = [("n_peaks", th["N_PEAKS_MAX"], "first-pass peaks", False), ("post_min", th["TAIL_MAX"], "tail fraction", False),
            ("sat_frac", th["SAT_MAX"], "saturated fraction", False), ("mask_px", th["MASK_MIN_PX"], "mask px", True),
            ("spread_px", th["SCATTER_SPREAD_PX"], "spread px", False), ("border_frac", th["BORDER_MAX"], "border_frac", True),
            ("largest", th["LARGEST_MIN"], "largest component", True), ("venous_frac", th["VENOUS_OVERLAP"], "venous_frac", False)]
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    ica = lab[lab.variant.isin(ICA_MASKS)]
    for ax, (col, thr, name, ica_only) in zip(axes.flat, spec):
        src = ica if ica_only else lab
        ok = src[src.category == "usable"][col].dropna()
        bad = src[src.category != "usable"][col].dropna()
        bins = np.histogram_bin_edges(src[col].dropna(), 30)
        ax.hist([ok, bad], bins=bins, stacked=True, color=["0.6", "C3"], label=["shape QC usable", "shape QC failed"])
        ax.axvline(thr, color="k", ls="--")
        ax.set_title(name + (" (ICA masks)" if ica_only else " (all masks)"))
        ax.set_yscale("log")
    axes.flat[0].legend(fontsize=8)
    ax = axes.flat[-1]  # mask size vs frame noise: the physical basis of MASK_MIN_PX
    ax.scatter(ica.mask_px, ica.snr, s=8, c=np.where(ica.category == "usable", "0.5", "C3"))
    ax.axvline(th["MASK_MIN_PX"], color="k", ls="--")
    ax.set_xlabel("mask px")
    ax.set_ylabel("frame SNR")
    ax.set_yscale("log")
    ax.set_title("mask size vs AIF SNR (ICA masks)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def md_table(df):
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join("" if pd.isna(x) else str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def md_counts(t):
    t = t.copy()
    t.columns = [f"{v} {p}" for v, p in t.columns]
    return md_table(t.rename_axis("class").reset_index())


def trust_by_variant(lab):
    t = pd.crosstab(lab.variant, lab.trust).reindex(index=VARIANTS, columns=["trusted", "review", "reject"], fill_value=0)
    t.insert(0, "shape usable", lab.groupby("variant").category.apply(lambda s: int((s == "usable").sum())))
    return md_table(t.reset_index())


def write_report(out_dir, lab, t_all, t_inc, sens, rho, th, command):
    failed = lab[lab.taxonomy != "ok"].sort_values(["taxonomy", "run_id", "variant"])
    silent = lab[(lab.taxonomy == "ok") & (lab.mask_problem != "")].sort_values(["mask_problem", "run_id"])
    crit = pd.DataFrame([dict(criterion=name, level=level,
                              threshold=thr.replace("$\\leq$", "<=").replace("$\\geq$", ">=").replace("$<$", "<")
                              .replace("$>$", ">").replace("\\%", "%"),
                              failing=int(lab[col].sum()),
                              **{"failing, shape QC usable": int(lab.loc[lab.category == "usable", col].sum())})
                         for col, name, thr, _, level in CRITERIA_TEXT])
    seg = int(t_all.loc[["wrong_structure", "too_small", "split_segments"]].values.sum())
    txt = f"""# AIF failure taxonomy and objective QC criteria — {out_dir.parent.parent.name}

Generated by `scripts/aif_taxonomy.py` (on top of `scripts/aif_qc.py`); do not edit by hand, re-run instead.
{lab.run_id.nunique()} unique runs x 3 variants = {len(lab)} curves (duplicates left out).

```bash
{command}
```

Thresholds: {", ".join(f"{k} {v}" for k, v in th.items())}

## Taxonomy (Table II)

Each failed curve gets exactly one class, in this order: wrong structure > too small > split segments >
projection overlap > not segmentation. A mask that is off target, too small or broken wins, because then
the cause analysis of the curve is moot; a split mask does not override a defect that aif_qc traced to the
bolus itself (same defect in both planes and the other masks). LaTeX: `table_taxonomy.tex`.

**All unique runs**

{md_counts(t_all)}

**Included runs (one plane per patient)**

{md_counts(t_inc)}

The segmentation itself (wrong structure + too small + split) explains {seg} of the
{int(t_all.loc[CLASSES[1:]].values.sum())} failed curves, projection overlap {int(t_all.loc["projection_overlap"].sum())},
the bolus/acquisition (not segmentation) {int(t_all.loc["not_segmentation"].sum())}.
Mask overlays per class: `grid_<class>.png`.

## Silent mask problems (curve passes, mask is off)

The shape QC cannot see these: the curve looks like a bolus but is averaged over other pixels than the
supraclinoid ICA alone. {len(silent)} curves (`grid_silent.png`); wrong structure / too small are rejected,
split / overprojected masks go to review.

{md_table(silent[["run_id", "variant", "plane", "included", "mask_problem", "trust", "mask_px", "n_components", "largest", "spread_px", "venous_frac"]].round(2)) if len(silent) else "None."}

## Objective QC criteria (trust flag)

LaTeX: `table_criteria.tex`; distributions with thresholds: `criteria_hist.png`.

{md_table(crit)}

Basis per threshold:
- **first-pass peaks <= 1**: one injection gives one first-pass bolus; a second peak with prominence >= 8 % of
  the peak is a notch or a second front (aif_qc PEAK_PROM). The aif_qc `multi_peak` rule (excess movement)
  is stricter and also catches steps without a separate peak.
- **tail fraction <= 0.30**: the aif_qc truncation level; the truncation analysis (`aif_qc/truncation/`) shows
  its effect on the perfusion maps.
- **saturation <= 10 %**: sat_frac has a natural gap in this cohort, from 6 % to 82 %; above it the ICA is
  black in the MINIP and the AIF peak is clipped. The run above it passes the shape QC, so only this criterion
  catches it.
- **mask size >= 500 px**: the frame noise of the AIF falls with the number of averaged pixels; Spearman
  rho(mask_px, SNR) = {rho[0]:.2f} (p = {rho[1]:.2g}) over the ICA masks. 500 px is about 1/3 of the median
  ICA-top mask ({int(lab[lab.variant == "ica_top"].mask_px.median())} px).
- **scattered**: >= 3 pieces with an RMS distance from the centroid > 200 px (a fifth of the image). On the
  overlays every such mask has pieces on cortical vessels, sinus or skull edge; two-piece ICA masks (supraclinoid
  plus cervical) stay below 200 px or have < 3 pieces.
- **border <= 5 %** (ICA masks): the supraclinoid ICA never touches the edge of a head DSA.
- **largest component >= 80 %** (ICA masks, review): the supraclinoid ICA is one segment; below 80 % the mask
  also averages a second piece, mostly the cervical/petrous ICA, which Su et al. avoid because of
  overprojection and contrast stasis. Grown masks are fragmented by design (median
  {lab[lab.variant == "ica_full_grown"].n_components.median():.0f} components) and are not tested.
- **venous_frac < 50 %** (review): if most mask pixels are still opaque in the venous phase, veins/sinus are
  projected over the mask (on the overlays: the siphon in the lateral view).

**Trust flag**: reject if the shape QC fails or a reject criterion fails; review if only a review criterion
fails; trusted otherwise.

All unique curves:

{trust_by_variant(lab)}

Included runs:

{trust_by_variant(lab[lab.included])}

### Sensitivity

Curves failing each criterion at a stricter and a more lenient threshold (threshold: count). A count that
barely moves means the threshold sits in a gap of the distribution.

{md_table(sens)}

## Per-curve list (failed curves)

{md_table(failed[["run_id", "variant", "plane", "included", "taxonomy", "taxonomy_reason"]])}

## Caveats

- The taxonomy is rule-based; the rules were set and checked on the overlays, but each class is the most
  likely explanation, not a verified diagnosis.
- `projection_overlap` partly relies on the aif_qc cause (defect in one plane only, ECA/venous overlap).
"""
    (out_dir / "taxonomy_report.md").write_text(txt, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def run(out, th, command):
    out_dir = out / "aif_qc" / "taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(out / "aif_qc" / "aif_qc_labels.csv")
    lab = lab[~lab.is_duplicate.astype(bool)].copy()
    lab["included"] = lab.included.fillna(False).astype(bool)
    lab["venous_frac"] = lab.venous_frac.fillna(0.0)
    lab["sat_frac"] = lab.sat_frac.fillna(0.0)
    lab = lab.merge(geometry_table(out, sorted(lab.run_id.unique()), th), on=["run_id", "variant"])
    assert (lab.geo_px == lab.mask_px).all(), "npz masks do not match aif_qc_labels.csv"

    lab = lab.join(criteria(lab, th))
    lab["mask_problem"] = lab.apply(mask_problem, axis=1)
    lab["taxonomy"] = lab.apply(taxonomy_class, axis=1)
    lab["taxonomy_reason"] = lab.apply(taxonomy_reason, axis=1, th=th)
    lab["trust"] = lab.apply(trust, axis=1)

    t_all = count_table(lab)
    t_inc = count_table(lab[lab.included])
    ica = lab[lab.variant.isin(ICA_MASKS)]
    rho = spearmanr(ica.mask_px, ica.snr)

    fail_cols = [c for c in lab if c.startswith("fail_")]
    cols = ["run_id", "variant", "plane", "included", "category", "cause_class", "taxonomy", "taxonomy_reason",
            "mask_problem", "trust", "mask_px", "n_components", "largest", "border_frac", "spread_px",
            "venous_frac", "sat_frac", "n_peaks", "post_min", "end_frac", "snr"] + fail_cols
    lab[cols].to_csv(out_dir / "taxonomy_labels.csv", index=False)
    tex_taxonomy(t_all, lab.run_id.nunique(), out_dir / "table_taxonomy.tex")
    tex_criteria(lab, out_dir / "table_criteria.tex")
    for c in CLASSES[1:]:
        plot_grid(out, lab[lab.taxonomy == c], CLASS_LABEL[c], out_dir / f"grid_{c}.png")
    plot_grid(out, lab[(lab.taxonomy == "ok") & (lab.mask_problem != "")], "Silent mask problems (curve passes)",
              out_dir / "grid_silent.png")
    plot_criteria(lab, th, out_dir / "criteria_hist.png")
    write_report(out_dir, lab, t_all, t_inc, sensitivity_table(lab, th), rho, th, command)
    return lab, t_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, help="combined result folder, e.g. results/combined_v1")
    for k, v in TH.items():
        p.add_argument(f"--{k.lower().replace('_', '-')}", type=type(v), default=v)
    args = p.parse_args()
    th = {k: getattr(args, k.lower()) for k in TH}
    lab, t_all = run(args.out, th, " ".join(["python"] + sys.argv))
    print(t_all.to_string())
    print("\n" + lab.trust.value_counts().to_string())


if __name__ == "__main__":
    main()
