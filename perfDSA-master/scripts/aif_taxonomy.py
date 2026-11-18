"""Segmentation failure taxonomy, objective QC criteria and trust flag per curve (run aif_qc.py first).

    python scripts/aif_taxonomy.py
"""

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from paths import DATA

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
    TAIL_MAX=0.2,           # aif_qc TRUNC_LEVEL
    SAT_MAX=0.10,           # natural gap in sat_frac: 0.06 -> 0.82
    N_PEAKS_MAX=1,
)


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
    """The shape QC alone decides; the mask criteria only describe the curve."""
    return "trusted" if r.category == "usable" else "reject"


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
    ("fail_tail", "Tail fraction (min after peak / peak)", "$\\leq$ 0.20", "shape QC; truncation analysis", "reject"),
    ("fail_sat", "Saturated mask pixels", "$\\leq$ 10\\%", "natural gap (6\\% $\\rightarrow$ 82\\%)", "reject"),
    ("fail_mask_px", "Mask size", "$\\geq$ 500 px", "noise $\\propto 1/\\sqrt{N}$; 1/3 of median", "reject"),
    ("fail_scattered", "Scattered mask", "not $\\geq$3 parts over $>$200 px", "visual check", "reject"),
    ("fail_border", "Mask in image border (ICA masks)", "$\\leq$ 5\\%", "ICA never at the image edge", "reject"),
    ("fail_components", "Largest component (ICA masks)", "$\\geq$ 80\\%", "supraclinoid ICA is one segment", "review"),
    ("fail_venous", "Late-opacifying mask fraction", "$<$ 50\\%", "veins over the mask", "review"),
]


def tex_criteria(lab, path):
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Descriptive mask and curve criteria; only the shape QC sets the trust flag. "
             "Failing: number of curves failing the criterion, "
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
    ax = axes.flat[-1]
    ax.scatter(ica.mask_px, ica.snr, s=8, c=np.where(ica.category == "usable", "0.5", "C3"))
    ax.axvline(th["MASK_MIN_PX"], color="k", ls="--")
    ax.set_xlabel("mask px")
    ax.set_ylabel("frame SNR")
    ax.set_yscale("log")
    ax.set_title("mask size vs AIF SNR (ICA masks)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def run(out, th, command):
    out_dir = out / "qc" / "taxonomy"
    out_dir.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(out / "qc" / "aif_qc_labels.csv")
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
    return lab, t_all


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    for k, v in TH.items():
        p.add_argument(f"--{k.lower().replace('_', '-')}", type=type(v), default=v)
    args = p.parse_args()
    th = {k: getattr(args, k.lower()) for k in TH}
    lab, t_all = run(args.out, th, " ".join(["python"] + sys.argv))
    print(t_all.to_string())
    print("\n" + lab.trust.value_counts().to_string())


if __name__ == "__main__":
    main()
