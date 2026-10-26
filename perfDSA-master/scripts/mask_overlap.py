"""
Overlap of the ICA-top and ICA-full masks of a run_dataset.py output (needs --save-maps).

Output (in <out>/mask_overlap/):
    mask_overlap.csv     one row per run: pixel counts, intersection, Dice, Jaccard, centroid
                         distance, and how similar the two AIFs are (Pearson r, peak ratio)
    <run_id>.png         MINIP with both masks (full view + zoom) and both AIFs

Example:
    python scripts/mask_overlap.py results/combined_v1
"""

import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cohort_common import dice, load_variant_npz, parse_source, rd

COLORS = {"ica_top": (230, 60, 30), "ica_full": (30, 140, 255)}  # RGB


def overlay(minip, masks):
    img = cv.cvtColor(np.clip(minip, 0, 255).astype(np.uint8), cv.COLOR_GRAY2RGB)
    for variant, m in masks.items():
        contours, _ = cv.findContours(m.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(img, contours, -1, COLORS[variant], thickness=2)
    return img


def figure(run_id, minip, masks, curves, stats, path):
    img = overlay(minip, masks)
    union = masks["ica_top"] | masks["ica_full"]
    ys, xs = np.nonzero(union)
    pad = 60
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6), tight_layout=True)
    ax[0].imshow(img)
    ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1, ls="--"))
    ax[0].set(title="MINIP", xticks=[], yticks=[])
    ax[1].imshow(img[y0:y1, x0:x1], interpolation="nearest")
    ax[1].set(title="Zoom on masks", xticks=[], yticks=[])
    for variant, (t, a) in curves.items():
        ax[2].plot(t, a, color=np.array(COLORS[variant]) / 255, lw=2,
                   label=f"{variant} ({int(masks[variant].sum())} px)")
    ax[2].set(title="AIF", xlabel="Time (s)", ylabel="Dye concentration [a.u.]")
    ax[2].legend()
    ax[2].grid(alpha=.3)
    fig.suptitle(f"{run_id}   Dice {stats['dice']:.3f}   top {stats['top_px']} px, full {stats['full_px']} px, "
                 f"overlap {stats['intersection_px']} px   AIF r = {stats['aif_pearson_r']:.3f}")
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="run_dataset.py output folder")
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()
    out = Path(args.out)
    dest = out / "mask_overlap"
    dest.mkdir(exist_ok=True)
    acq = pd.read_csv(out / "acquisition_table.csv").set_index("run_id")

    rows = []
    for f in sorted((out / "runs").glob("*.json")):
        run_id = f.stem
        if "error" in json.loads(f.read_text()):
            continue
        npz = {v: load_variant_npz(out, v, run_id) for v in ("ica_top", "ica_full")}
        if any(x is None for x in npz.values()):
            continue
        masks = {v: x["mask"].astype(bool) for v, x in npz.items()}
        top, full = masks["ica_top"], masks["ica_full"]
        ct = np.argwhere(top).mean(axis=0) if top.any() else np.full(2, np.nan)
        cf = np.argwhere(full).mean(axis=0) if full.any() else np.full(2, np.nan)
        a_top, a_full = npz["ica_top"]["aif"], npz["ica_full"]["aif"]
        stats = {"run_id": run_id, "top_px": int(top.sum()), "full_px": int(full.sum()),
                 "intersection_px": int((top & full).sum()), "union_px": int((top | full).sum()),
                 "top_only_px": int((top & ~full).sum()), "full_only_px": int((full & ~top).sum()),
                 "dice": dice(top, full), "jaccard": float((top & full).sum() / max((top | full).sum(), 1)),
                 "full_over_top_px": full.sum() / top.sum() if top.any() else np.nan,
                 "centroid_dist_px": float(np.linalg.norm(ct - cf)),
                 "aif_pearson_r": float(np.corrcoef(a_top, a_full)[0, 1]),
                 "aif_peak_ratio_full_over_top": float(np.nanmax(a_full) / np.nanmax(a_top))}
        rows.append(stats)
        if not args.no_figures:
            ds = rd.read_dicom(parse_source(acq.loc[run_id, "source"]))
            minip = rd.load_uniform_sequence(ds, 250, 1024).min(axis=2)
            curves = {v: (x["time_s"], x["aif"]) for v, x in npz.items()}
            figure(run_id, minip, masks, curves, stats, dest / f"{run_id}.png")
        print(f"{run_id}: Dice {stats['dice']:.3f}  top {stats['top_px']}  full {stats['full_px']}  "
              f"overlap {stats['intersection_px']}  AIF r {stats['aif_pearson_r']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(dest / "mask_overlap.csv", index=False)
    if len(df):
        print(f"\n{len(df)} runs: Dice median {df.dice.median():.3f} [{df.dice.min():.3f}-{df.dice.max():.3f}], "
              f"full/top px median {df.full_over_top_px.median():.2f}")


if __name__ == "__main__":
    main()
