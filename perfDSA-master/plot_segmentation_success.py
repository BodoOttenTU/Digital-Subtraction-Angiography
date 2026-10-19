"""
Quality control of the perfDSA ICA-top segmentation on a dataset without reference masks.

There are no manual ICA annotations for PRAISE, so a Dice score cannot be computed. Instead
every run is checked against four criteria a correct ICA-top mask has to meet:

    found     : the model returns a mask of at least MIN_MASK_PX pixels
    compact   : one vessel segment, the largest connected component holds >= MIN_LARGEST_FRACTION
    on vessel : the mask is as opaque in the MINIP as the densest vessels, i.e. more opaque
                than MIN_OPACITY_PERCENTILE % of the (non-collimated) image
    bolus AIF : the AIF sampled from the mask rises to a peak >= MIN_AIF_PEAK and washes out
                again by >= MIN_WASHOUT of that peak before the run ends

These are necessary, not sufficient: a mask on another dense artery passes them as well, so
the overlay sheets are written for visual confirmation.

Output (in -o, default ./results/praise):
    segmentation_qc.csv               one row per run: metrics and pass/fail per criterion
    segmentation_success.png          pass rates, per-run result matrix, AIF curves
    segmentation_overlays.png         every MINIP with its mask
    segmentation_overlays_zoom.png    256 px crop around every mask
    seg_qc/<run_id>.npz               per-run cache; delete (or --redo) to recompute

Example:
    python plot_segmentation_success.py -i ../PRAISE -o results/praise -d cpu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

from run_dataset import (HERE, ICA_TOP_MODEL, discover_runs, load_model, load_uniform_sequence,
                         pick_device, read_dicom)

MIN_MASK_PX = 100
MIN_LARGEST_FRACTION = 0.9
MIN_OPACITY_PERCENTILE = 95
MIN_AIF_PEAK = 0.2
MIN_WASHOUT = 0.5
CRITERIA = ["found", "compact", "on_vessel", "bolus_aif"]
CRITERIA_LABELS = ["Mask found", "Single segment", "On dense vessel", "Bolus-shaped AIF"]

PASS, FAIL = "#2a78d6", "#e34948"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CROP = 256


def analyse_run(run, net, device, args):
    ds = read_dicom(run["source"])
    frames = load_uniform_sequence(ds, args.frame_interval, args.size)
    minip = frames.min(axis=2)
    with torch.no_grad():
        prob = torch.sigmoid(net(torch.from_numpy(minip / 255).float()[None, None].to(device)))
    prob = prob.squeeze().cpu().numpy()
    mask = prob > args.threshold

    m = {"mask_px": int(mask.sum()), "prob_max": float(prob.max())}
    aif = np.full(frames.shape[2], np.nan)
    cy, cx = minip.shape[0] // 2, minip.shape[1] // 2
    if mask.any():
        _, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))
        areas = stats[1:, cv.CC_STAT_AREA]
        ys, xs = np.nonzero(labels == 1 + int(np.argmax(areas)))
        cy, cx = int(ys.mean()), int(xs.mean())
        opacity = 1 - minip / 170  # DSASequence's concentration scale
        valid = minip > 5  # black collimator/shutter areas are not vessels
        aif = (1 - frames[mask] / 170).mean(axis=0)
        i = int(np.argmax(aif))
        m.update({
            "n_components": int(len(areas)),
            "largest_fraction": float(areas.max() / areas.sum()),
            "prob_mean_in_mask": float(prob[mask].mean()),
            "opacity_in_mask": float(opacity[mask].mean()),
            "opacity_percentile": float(100 * (opacity[valid] < opacity[mask].mean()).mean()),
            "centroid_x": cx / minip.shape[1], "centroid_y": cy / minip.shape[0],
            "aif_peak": float(aif[i]), "aif_ttp_s": i * args.frame_interval / 1000,
            "aif_peak_at_edge": bool(i == 0 or i == len(aif) - 1),
            "aif_washout": float(1 - aif[i:].min() / aif[i]) if aif[i] > 0 else 0.0,
        })

    y0 = int(np.clip(cy - CROP // 2, 0, minip.shape[0] - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, minip.shape[1] - CROP))
    return m, {"thumb": cv.resize(minip, (512, 512), interpolation=cv.INTER_AREA).astype(np.uint8),
               "mask": mask, "crop": minip[y0:y0 + CROP, x0:x0 + CROP].astype(np.uint8),
               "crop_mask": mask[y0:y0 + CROP, x0:x0 + CROP], "aif": aif}


def judge(df):
    df["found"] = df["mask_px"] >= MIN_MASK_PX
    df["compact"] = df["found"] & (df["largest_fraction"] >= MIN_LARGEST_FRACTION)
    df["on_vessel"] = df["found"] & (df["opacity_percentile"] >= MIN_OPACITY_PERCENTILE)
    df["bolus_aif"] = (df["found"] & (df["aif_peak"] >= MIN_AIF_PEAK) & (df["aif_washout"] >= MIN_WASHOUT)
                       & ~df["aif_peak_at_edge"].fillna(True).astype(bool))
    df["success"] = df[CRITERIA].all(axis=1)
    return df


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=9)


def plot_success(df, cache, dt_s, path):
    n = len(df)
    fig = plt.figure(figsize=(17, 8.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.25], height_ratios=[1, 1],
                          left=0.09, right=0.98, top=0.8, bottom=0.13, wspace=0.32, hspace=0.42)
    n_ok = int(df["success"].sum())
    fig.suptitle(f"perfDSA ICA-top segmentation on PRAISE: {n_ok} of {n} runs pass all checks "
                 f"({100 * n_ok / n:.0f}%)", x=0.09, ha="left", fontsize=16, fontweight="bold", color=INK)
    fig.text(0.09, 0.915, f"{df['patient_id'].nunique()} patients, frontal (A) and lateral (B) plane. "
             "No reference masks exist, so success is judged by four plausibility checks, not Dice.",
             fontsize=10.5, color=INK2)

    # pass rate per criterion
    ax = fig.add_subplot(gs[:, 0])
    style(ax)
    names = CRITERIA_LABELS + ["All four"]
    counts = [int(df[c].sum()) for c in CRITERIA] + [n_ok]
    y = np.arange(len(names))[::-1]
    ax.barh(y, [n] * len(y), height=0.5, color=FAIL, alpha=0.9)
    ax.barh(y, counts, height=0.5, color=PASS)
    for yi, c in zip(y, counts):
        ax.text(n * 1.02, yi, f"{c}/{n}", va="center", fontsize=11, color=INK, fontweight="bold")
        if n - c:
            ax.text(n - 0.5, yi, f"{n - c} fail", va="center", ha="right", fontsize=8.5, color="white")
    ax.set_yticks(y, names, fontsize=11, color=INK)
    ax.set_xlim(0, n * 1.16)
    ax.set_xticks(np.linspace(0, n, 4).astype(int))
    ax.set_xlabel("Runs", color=INK2, fontsize=10)
    ax.spines["left"].set_visible(False)
    ax.set_title("Runs passing each check", loc="left", fontsize=12, color=INK, pad=10)
    ax.legend(handles=[Rectangle((0, 0), 1, 1, color=PASS), Rectangle((0, 0), 1, 1, color=FAIL)],
              labels=["pass", "fail"], loc="lower left", bbox_to_anchor=(0, -0.14), ncol=2, frameon=False,
              fontsize=9.5, labelcolor=INK2)

    # per-run result matrix: patients x (plane A | plane B) x criterion
    ax = fig.add_subplot(gs[:, 1])
    ax.set_facecolor(SURFACE)
    patients = sorted(df["patient_id"].unique())
    for r, pid in enumerate(patients):
        for p, plane in enumerate("AB"):
            row = df[(df["patient_id"] == pid) & (df["plane"] == plane)]
            for c, crit in enumerate(CRITERIA):
                x = p * (len(CRITERIA) + 0.6) + c
                if row.empty:
                    continue
                ok = bool(row[crit].iloc[0])
                ax.add_patch(Rectangle((x + 0.06, r + 0.08), 0.88, 0.84, color=PASS if ok else FAIL))
                if not ok:
                    ax.text(x + 0.5, r + 0.5, "x", ha="center", va="center", color="white", fontsize=8,
                            fontweight="bold")
    width = 2 * len(CRITERIA) + 0.6
    ax.set_xlim(0, width)
    ax.set_ylim(len(patients), 0)
    ax.set_yticks(np.arange(len(patients)) + 0.5, [p.replace("PRAISE", "").replace("_01", "") for p in patients],
                  fontsize=7.5, color=INK2)
    xt = [p * (len(CRITERIA) + 0.6) + c + 0.5 for p in range(2) for c in range(len(CRITERIA))]
    ax.set_xticks(xt, ["found", "single", "vessel", "AIF"] * 2, fontsize=8, color=INK2, rotation=45, ha="right")
    for p, plane in enumerate("AB"):
        ax.text(p * (len(CRITERIA) + 0.6) + len(CRITERIA) / 2, -0.5, f"Plane {plane}", ha="center",
                fontsize=10, color=INK)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel("PRAISE patient", color=INK2, fontsize=10)
    ax.set_title("Result per run", loc="left", fontsize=12, color=INK, pad=24)

    # AIF curves
    ax = fig.add_subplot(gs[0, 2])
    style(ax)
    for ok, colour, z in ((True, PASS, 2), (False, FAIL, 3)):
        for rid in df.loc[df["success"] == ok, "run_id"]:
            aif = cache[rid]["aif"]
            if np.isfinite(aif).any():
                ax.plot(np.arange(len(aif)) * dt_s, aif, color=colour, lw=1.4 if not ok else 0.9,
                        alpha=0.95 if not ok else 0.45, zorder=z)
    ax.axhline(MIN_AIF_PEAK, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.text(ax.get_xlim()[1], MIN_AIF_PEAK, " min. peak", va="bottom", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Time (s)", color=INK2, fontsize=10)
    ax.set_ylabel("Dye concentration in mask [a.u.]", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_title("AIF sampled from each mask", loc="left", fontsize=12, color=INK, pad=10)
    ax.legend(handles=[plt.Line2D([], [], color=PASS, lw=2), plt.Line2D([], [], color=FAIL, lw=2)],
              labels=["passes all checks", "fails a check"], frameon=False, fontsize=9, labelcolor=INK2)

    # mask size against vessel opacity
    ax = fig.add_subplot(gs[1, 2])
    style(ax)
    found = df[df["mask_px"] > 0]
    for ok, colour, marker, label in ((True, PASS, "o", "passes all checks"), (False, FAIL, "X", "fails a check")):
        g = found[found["success"] == ok]
        ax.scatter(g["mask_px"], g["opacity_percentile"], s=46, color=colour, marker=marker,
                   edgecolor=SURFACE, linewidth=1, label=label, zorder=3)
    ax.axhline(MIN_OPACITY_PERCENTILE, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.axvline(MIN_MASK_PX, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_xscale("log")
    ax.set_xlabel("Mask size (pixels, 1024 x 1024 image)", color=INK2, fontsize=10)
    ax.set_ylabel("Mask opacity, percentile of image", color=INK2, fontsize=10)
    ax.grid(color=GRID, lw=0.6)
    ax.set_title("Mask size and how dense the vessel under it is", loc="left", fontsize=12, color=INK, pad=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")

    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_overlays(df, cache, path, zoom):
    df = df.sort_values(["patient_id", "plane"])
    cols = 10
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.45 * rows), facecolor="black")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        c = cache[r["run_id"]]
        img, mask = (c["crop"], c["crop_mask"]) if zoom else (c["thumb"], c["mask"])
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        if not zoom:
            mask = cv.resize(mask.astype(np.uint8), img.shape[::-1], interpolation=cv.INTER_NEAREST)
        contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        cv.drawContours(rgb, contours, -1, (255, 220, 0), thickness=1 if zoom else 2)
        ax.imshow(rgb)
        failed = [lab for crit, lab in zip(CRITERIA, ["found", "single", "vessel", "AIF"]) if not r[crit]]
        ax.set_title(f"{r['patient_id'].replace('PRAISE', '').replace('_01', '')} {r['plane']}"
                     + (f"  FAIL: {', '.join(failed)}" if failed else "  pass"),
                     fontsize=8, color="#ff8a80" if failed else "#9ec5f4")
    fig.suptitle("ICA-top mask (yellow) on the MINIP" + (f", {CROP} px crop around the mask" if zoom else ""),
                 color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


def get_args():
    p = argparse.ArgumentParser(description="QC of the perfDSA ICA-top segmentation on a dataset.")
    p.add_argument("-i", dest="inputs", nargs="+", default=[str(HERE.parent / "PRAISE")], help="Dataset root(s).")
    p.add_argument("-o", dest="out_dir", default=str(HERE / "results" / "praise"), help="Output folder.")
    p.add_argument("-d", dest="device", default="cuda", help='"cuda" or "cpu".')
    p.add_argument("-f", dest="frame_interval", default=250, type=int, help="Frame interval in ms.")
    p.add_argument("--size", type=int, default=1024, help="Resize frames to size x size.")
    p.add_argument("--min-frames", type=int, default=5, help="Skip DICOMs with fewer frames.")
    p.add_argument("--threshold", type=float, default=0.5, help="Probability threshold of the segmentation.")
    p.add_argument("--redo", action="store_true", help="Recompute runs that are cached already.")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = get_args()
    out = Path(args.out_dir)
    (out / "seg_qc").mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.inputs, args.min_frames)
    device = pick_device(args.device)
    net = None
    rows, cache = [], {}
    for i, run in enumerate(runs, 1):
        f = out / "seg_qc" / f"{run['run_id']}.npz"
        if not f.exists() or args.redo:
            logging.info("[%d/%d] %s", i, len(runs), run["run_id"])
            net = net or load_model(ICA_TOP_MODEL, device)
            metrics, arrays = analyse_run(run, net, device, args)
            np.savez_compressed(f, metrics=np.array(json.dumps(metrics)), **arrays)
        with np.load(f) as z:
            cache[run["run_id"]] = {k: z[k] for k in ("thumb", "mask", "crop", "crop_mask", "aif")}
            metrics = json.loads(str(z["metrics"]))
        acq = run["acq"]
        rows.append({"patient_id": acq["patient_id"], "run_id": run["run_id"], "plane": acq["plane"],
                     "series_description": acq["series_description"], **metrics})

    df = judge(pd.DataFrame(rows))
    df.to_csv(out / "segmentation_qc.csv", index=False)
    plot_success(df, cache, args.frame_interval / 1000, out / "segmentation_success.png")
    plot_overlays(df, cache, out / "segmentation_overlays.png", zoom=False)
    plot_overlays(df, cache, out / "segmentation_overlays_zoom.png", zoom=True)
    logging.info("%d of %d runs pass all checks -> %s", int(df["success"].sum()), len(df),
                 out / "segmentation_success.png")


if __name__ == "__main__":
    main()
