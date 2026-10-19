"""
Compare three ICA masks on a dataset, and the AIF curves sampled from them:

    ICA top    : models/best_model_ica_top.pt
    ICA full   : models/best_model_ica.pt
    ICA grown  : the ICA-top mask grown down the vessel (run_dataset.arterial_full_ica)

Output (in -o, default ./results/praise):
    ica_full_overlays.png        every MINIP with the three masks
    ica_full_overlays_zoom.png   256 px crop around the model masks
    aif_full_vs_top_curves.png   the three AIF curves of every run
    aif_full_vs_top_summary.png  AIF features of the full and grown masks against the top mask
    aif_full_vs_top.csv          one row per run: mask sizes, overlap, AIF features of the three masks
    ica_full_qc/<run_id>.npz     per-run cache; delete (or --redo) to recompute

Example:
    python plot_ica_full_comparison.py -i ../PRAISE -o results/praise -d cpu
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from run_dataset import (HERE, ICA_MODEL, ICA_TOP_MODEL, aif_features, arterial_full_ica, discover_runs,
                         load_model, load_uniform_sequence, pick_device, read_dicom, segment)

# name: (label, plot colour, overlay colour RGB, line width); curves are drawn widest first
MASKS = {"top": ("ICA top (best_model_ica_top.pt)", "#2a78d6", (60, 150, 255), 3.0),
         "grown": ("ICA grown from the top mask", "#1baf7a", (40, 225, 150), 1.6),
         "full": ("ICA full (best_model_ica.pt)", "#eb6834", (255, 140, 40), 1.0)}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
CROP = 256


def short(patient_id):
    return patient_id.replace("PRAISE", "").replace("_01", "")


def analyse_run(run, load_nets, device, args, prev=None):
    """prev: an earlier cache entry of this run; its model masks are reused instead of segmenting again."""
    ds = read_dicom(run["source"])
    frames = load_uniform_sequence(ds, args.frame_interval, args.size)
    minip = frames.min(axis=2)
    if prev is not None:
        top, full = prev["top"], prev["full"]
    else:
        nets = load_nets()
        top, _ = segment(nets["top"], minip, device, threshold=args.threshold)
        full, _ = segment(nets["full"], minip, device, threshold=args.threshold)
    grown = arterial_full_ica(frames, top, args.frame_interval / 1000)

    def aif(mask):  # as DSASequence.sample_AIF_from_mask
        return (1 - frames[mask] / 170).mean(axis=0) if mask.any() else np.full(frames.shape[2], np.nan)

    both = top | full  # the crop stays on the model masks; the grown mask runs out of it
    cy, cx = [int(v.mean()) for v in np.nonzero(both)] if both.any() else [s // 2 for s in minip.shape]
    y0 = int(np.clip(cy - CROP // 2, 0, minip.shape[0] - CROP))
    x0 = int(np.clip(cx - CROP // 2, 0, minip.shape[1] - CROP))
    box = np.s_[y0:y0 + CROP, x0:x0 + CROP]
    out = {"thumb": cv.resize(minip, (512, 512), interpolation=cv.INTER_AREA).astype(np.uint8),
           "crop": minip[box].astype(np.uint8)}
    for name, mask in (("top", top), ("full", full), ("grown", grown)):
        out.update({name: mask, f"crop_{name}": mask[box], f"aif_{name}": aif(mask)})
    return out


def summarise(run, c, dt_s):
    acq = run["acq"]
    top, full = c["top"], c["full"]
    t = np.arange(len(c["aif_top"])) * dt_s
    row = {"patient_id": acq["patient_id"], "run_id": run["run_id"], "plane": acq["plane"],
           "top_px": int(top.sum()), "full_px": int(full.sum()), "grown_px": int(c["grown"].sum()),
           "dice": float(2 * (top & full).sum() / max(top.sum() + full.sum(), 1))}
    for name in MASKS:
        aif = c[f"aif_{name}"]
        if np.isfinite(aif).all():
            row.update({f"{k}_{name}": v for k, v in aif_features(aif, t).items()})
    for name in ("full", "grown"):  # each against the ICA-top curve
        if np.isfinite(c["aif_top"]).all() and np.isfinite(c[f"aif_{name}"]).all():
            row[f"curve_corr_{name}"] = float(np.corrcoef(c["aif_top"], c[f"aif_{name}"])[0, 1])
            row[f"curve_rmse_rel_{name}"] = float(np.sqrt(np.mean((c[f"aif_{name}"] - c["aif_top"]) ** 2))
                                                  / c["aif_top"].max())
    return row


def plot_overlays(df, cache, path, zoom):
    df = df.sort_values(["patient_id", "plane"])
    cols = 10
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.45 * rows), facecolor="black")
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        c = cache[r["run_id"]]
        img = c["crop"] if zoom else c["thumb"]
        rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
        for key in ("grown", "full", "top"):
            mask = c[f"crop_{key}"] if zoom else cv.resize(c[key].astype(np.uint8), img.shape[::-1],
                                                           interpolation=cv.INTER_NEAREST)
            contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            cv.drawContours(rgb, contours, -1, MASKS[key][2], thickness=1 if zoom else 2)
        ax.imshow(rgb)
        ax.set_title(f"{short(r['patient_id'])} {r['plane']}   top {r['top_px']}, full {r['full_px']}, "
                     f"grown {r['grown_px']} px", fontsize=6.8, color="white")
    fig.suptitle("ICA-top model mask (blue), ICA-full model mask (orange) and grown ICA mask (green) on the MINIP"
                 + (f", {CROP} px crop around the model masks" if zoom else ""), color="white", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=110, facecolor="black")
    plt.close(fig)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, length=0, labelsize=8)


def plot_curves(df, cache, dt_s, path):
    df = df.sort_values(["patient_id", "plane"])
    cols = 10
    rows = int(np.ceil(len(df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 1.75 * rows + 1.1), facecolor=SURFACE,
                             sharex=True, sharey=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        ax.axis("on")
        style(ax)
        c = cache[r["run_id"]]
        t = np.arange(len(c["aif_top"])) * dt_s
        for name, (_, colour, _, lw) in MASKS.items():
            ax.plot(t, c[f"aif_{name}"], color=colour, lw=lw)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_title(f"{short(r['patient_id'])} {r['plane']}", fontsize=8.5, color=INK, loc="left", pad=3)
    fig.suptitle("AIF from the three ICA masks, per run", x=0.035, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.legend(handles=[plt.Line2D([], [], color=m[1], lw=m[3]) for m in MASKS.values()],
               labels=[m[0] for m in MASKS.values()], loc="upper right",
               bbox_to_anchor=(0.99, 0.995), ncol=3, frameon=False, fontsize=10, labelcolor=INK2)
    fig.supxlabel("Time (s)", color=INK2, fontsize=10)
    fig.supylabel("Dye concentration in mask [a.u.]", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.955))
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def plot_summary(df, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.8), facecolor=SURFACE)
    d = df.dropna(subset=["aif_peak_top", "aif_peak_full", "aif_peak_grown"])
    series = (("full", "ICA full model", "o"), ("grown", "ICA grown", "D"))
    fig.suptitle(f"AIF of the full-ICA masks against the ICA-top model mask, {len(d)} PRAISE runs",
                 x=0.045, ha="left", fontsize=15, fontweight="bold", color=INK)
    parts = []
    for name, label, _ in series:
        rel = 100 * (d[f"aif_peak_{name}"] / d["aif_peak_top"] - 1)
        parts.append(f"{label}: AIF peak {rel.median():+.1f}% (range {rel.min():+.0f}% to {rel.max():+.0f}%), "
                     f"curve correlation {d[f'curve_corr_{name}'].median():.3f}")
    fig.text(0.045, 0.885, "Median difference to the ICA-top AIF.  " + ".  ".join(parts) + ".",
             fontsize=10.5, color=INK2)
    fig.text(0.045, 0.85, "Each marker is one run; on the diagonal a mask gives the same value as the ICA-top mask.",
             fontsize=10.5, color=INK2)

    for ax, key, label in ((axes[0], "aif_peak", "AIF peak [a.u.]"),
                           (axes[1], "aif_auc", "AIF area under curve [a.u. s]"),
                           (axes[2], "aif_fwhm_s", "AIF width at half maximum (s)")):
        style(ax)
        cols = [d[f"{key}_{n}"] for n in MASKS]
        lo, hi = min(c.min() for c in cols), max(c.max() for c in cols)
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        for name, lab, marker in series:
            ax.scatter(d[f"{key}_top"], d[f"{key}_{name}"], s=40, color=MASKS[name][1], marker=marker,
                       edgecolor=SURFACE, linewidth=1, zorder=3, label=lab)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("ICA top mask", color=INK2, fontsize=10)
        ax.set_ylabel("Full-ICA mask", color=INK2, fontsize=10)
        ax.set_title(label, loc="left", fontsize=12, color=INK, pad=8)
    axes[0].legend(frameon=False, fontsize=9.5, labelcolor=INK2, loc="upper left")

    ax = axes[3]
    style(ax)
    counts = pd.DataFrame({name: (d[f"aif_ttp_s_{name}"] - d["aif_ttp_s_top"]).round(2).value_counts()
                           for name, _, _ in series}).fillna(0).astype(int).sort_index()
    x = np.arange(len(counts))
    for k, (name, lab, _) in enumerate(series):
        xs = x + (k - 0.5) * 0.4
        ax.bar(xs, counts[name], width=0.36, color=MASKS[name][1], label=lab)
        for xi, n in zip(xs, counts[name]):
            if n:
                ax.text(xi, n + 0.6, str(n), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x, [f"{v:+.2f}" if v else "0" for v in counts.index], rotation=45, ha="right")
    ax.set_xlabel("Time to peak, mask minus ICA top (s)", color=INK2, fontsize=10)
    ax.set_ylabel("Runs", color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title("Shift of the AIF peak in time", loc="left", fontsize=12, color=INK, pad=8)

    fig.tight_layout(rect=(0.03, 0, 1, 0.83), w_pad=2.5)
    fig.savefig(path, dpi=140, facecolor=SURFACE)
    plt.close(fig)


def get_args():
    p = argparse.ArgumentParser(description="ICA top, ICA full and grown ICA: masks and AIF curves.")
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
    (out / "ica_full_qc").mkdir(parents=True, exist_ok=True)
    dt_s = args.frame_interval / 1000

    runs = discover_runs(args.inputs, args.min_frames)
    device = pick_device(args.device)
    nets = {}

    def load_nets():
        if not nets:
            nets.update(top=load_model(ICA_TOP_MODEL, device), full=load_model(ICA_MODEL, device))
        return nets

    rows, cache = [], {}
    for i, run in enumerate(runs, 1):
        f = out / "ica_full_qc" / f"{run['run_id']}.npz"
        prev = None
        if f.exists() and not args.redo:
            with np.load(f) as z:
                prev = {k: z[k] for k in z.files}
        if prev is None or "grown" not in prev:  # caches written before the grown mask was added lack it
            logging.info("[%d/%d] %s", i, len(runs), run["run_id"])
            prev = analyse_run(run, load_nets, device, args, prev)
            np.savez_compressed(f, **prev)
        cache[run["run_id"]] = prev
        rows.append(summarise(run, prev, dt_s))

    df = pd.DataFrame(rows)
    df.to_csv(out / "aif_full_vs_top.csv", index=False)
    plot_overlays(df, cache, out / "ica_full_overlays.png", zoom=False)
    plot_overlays(df, cache, out / "ica_full_overlays_zoom.png", zoom=True)
    plot_curves(df, cache, dt_s, out / "aif_full_vs_top_curves.png")
    plot_summary(df, out / "aif_full_vs_top_summary.png")
    logging.info("Wrote overlays and AIF comparison for %d runs to %s", len(df), out)


if __name__ == "__main__":
    main()
