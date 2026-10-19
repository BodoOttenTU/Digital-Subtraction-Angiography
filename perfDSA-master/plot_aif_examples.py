"""
Side-by-side ICA full vs ICA top figures for a few example runs: usable AIFs and one run per failure type.

Each figure has two rows (ICA full, ICA top), each row:  MINIP + mask | AIF | CBV | CBF | MTT | Tmax.
The two rows of a map share one colour scale, so the effect of the AIF on the maps is directly visible.

The runs are reprocessed with the same steps and defaults as run_dataset.py (ICA top from the model;
ICA full grown from it, or with --ica-full model from models/best_model_ica.pt), because the maps of the
dataset run were only stored as rendered figures. Each --ica-full mode has its own list of examples.
Maps are cached in <out>/maps/<run_id>.npz; delete them (or --redo) to recompute.

Output (in -o, default ./results/aif_examples, or ./results/ica_model/aif_examples for --ica-full model):
    <nn>_<label>_<run_id>.png   one figure per example
    overview_usable.png         all usable examples, stacked
    overview_failures.png       all failure examples, stacked

Example:
    python plot_aif_examples.py -d cpu
    python plot_aif_examples.py -d cpu --ica-full model
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

from DSASequence import DSASequence
from run_dataset import (HERE, ICA_MODEL, ICA_TOP_MODEL, arterial_full_ica, discover_runs, load_model, load_uniform_sequence,
                         pick_device, read_dicom, segment)

# Per --ica-full mode: (run_id, label, short explanation shown in the figure title)
EXAMPLES = {"grown": [
    ("PRAISE_BATCH_2_PRAISE001_01_S02_B", "usable", "Usable: mask on the ICA, single smooth bolus"),
    ("PRAISE_PRAISE008_01_S08_A", "usable", "Usable: mask on the ICA, single smooth bolus"),
    ("PRAISE_PRAISE064_01_S11_A", "usable", "Usable: mask on the ICA, single smooth bolus"),
    ("PRAISE_BATCH_2_PRAISE041_01_S02_A", "wrong_structure",
     "Failure - mask on the wrong structure (no contrast-filled vessel under the mask)"),
    ("PRAISE_BATCH_2_PRAISE020_01_S25_A", "vessel_overlap",
     "Failure - multi-peak: ICA overlaps other arteries in projection"),
    ("PRAISE_PRAISE073_01_S08_A", "venous_overlap",
     "Failure - late rise: ICA-top mask picks up late (venous) filling; grown ICA-full mask is fine"),
    ("PRAISE_PRAISE084_01_S12_A", "elevated_tail",
     "Failure - 'truncated': tail stays at ~30% of peak, no return to baseline"),
    ("PRAISE_BATCH_2_PRAISE059_01_S16_A", "small_mask",
     "Failure - ICA-top mask too small on a thin segment: jagged AIF (note shutter-edge artefact in maps)"),
    ("PRAISE_PRAISE062_01_S18_B", "injection",
     "Failure - not segmentation: correct mask, early contrast spike before the bolus"),
], "model": [
    ("PRAISE_BATCH_2_PRAISE001_01_S02_B", "usable", "Usable: both masks on the ICA, single smooth bolus"),
    ("PRAISE_PRAISE008_01_S08_A", "usable", "Usable: both masks on the ICA, single smooth bolus"),
    ("PRAISE_PRAISE064_01_S11_A", "usable", "Usable: both masks on the ICA, single smooth bolus"),
    ("PRAISE_BATCH_2_PRAISE041_01_S02_A", "wrong_structure",
     "Failure - both masks on the wrong structure (no contrast-filled vessel under the mask)"),
    ("PRAISE_BATCH_2_PRAISE002_01_S03_A", "split_mask",
     "Failure - multi-peak: ICA-top mask split over 3 segments; single model mask still double-peaked"),
    ("PRAISE_BATCH_2_PRAISE020_01_S25_A", "vessel_overlap",
     "Failure - multi-peak: ICA overlaps other arteries in projection"),
    ("PRAISE_PRAISE073_01_S08_A", "venous_overlap",
     "Failure - late rise: both masks pick up late (venous) filling distal to the siphon"),
    ("PRAISE_PRAISE084_01_S12_A", "elevated_tail",
     "Failure - 'truncated': tail stays at ~30% of peak, no return to baseline"),
    ("PRAISE_BATCH_2_PRAISE059_01_S16_A", "small_mask",
     "Failure - both masks small (~370 px) on a thin segment: jagged AIF (note shutter-edge artefact)"),
    ("PRAISE_PRAISE062_01_S18_B", "injection",
     "Failure - not segmentation: correct masks, early contrast spike before the bolus"),
]}
VARIANTS = [("ica_full", "ICA full", "tab:blue"), ("ica_top", "ICA top", "tab:red")]
MAPS = ["CBV", "CBF", "MTT", "Tmax"]

# Same defaults as run_dataset.py
PIPELINE = SimpleNamespace(frame_interval=250, size=1024, native_subtraction="log", native_vessel_level=20,
                           threshold=0.5, tta=False, largest_component=False)


def compute(run, nets, device, cache):
    if cache.exists():
        return dict(np.load(cache))
    p = PIPELINE
    frames = load_uniform_sequence(run["ds"], p.frame_interval, p.size, p.native_subtraction,
                                   p.native_vessel_level)
    dt_s = p.frame_interval / 1000
    minip = frames.min(axis=2)
    seg = dict(threshold=p.threshold, tta=p.tta, largest_component=p.largest_component)
    top, _ = segment(nets["ica_top"], minip, device, **seg)
    full = segment(nets["ica_full"], minip, device, **seg)[0] if "ica_full" in nets else arterial_full_ica(
        frames, top, dt_s)
    masks = {"ica_full": full, "ica_top": top}

    out = {"minip": minip.astype(np.float32)}
    for variant, mask in masks.items():
        dsa = DSASequence(frames, frame_interval=dt_s)
        aif = np.asarray(np.ma.filled(dsa.sample_AIF_from_mask(mask), np.nan), dtype=float)
        dsa._aif = aif
        cbv, cbf, mtt, tmax = dsa.get_perfusion_parameters()
        out.update({f"{variant}_mask": mask, f"{variant}_aif": aif, "time_s": dsa.Time(),
                    **{f"{variant}_{k}": v.astype(np.float32) for k, v in zip(MAPS, (cbv, cbf, mtt, tmax))}})
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **out)
    return out


def crop_box(masks, shape, margin=60, min_size=320):
    ys, xs = np.nonzero(np.any(masks, axis=0))
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(min_size, np.ptp(ys) + 2 * margin, np.ptp(xs) + 2 * margin) / 2
    y0 = int(np.clip(cy - half, 0, max(0, shape[0] - 2 * half)))
    x0 = int(np.clip(cx - half, 0, max(0, shape[1] - 2 * half)))
    return slice(y0, int(y0 + 2 * half)), slice(x0, int(x0 + 2 * half))


def draw(axs, d, full_name):
    """axs: 2 x 6 axes. Draws one example."""
    masks = [d[f"{v}_mask"] > 0 for v, _, _ in VARIANTS]
    sy, sx = crop_box(masks, d["minip"].shape)
    t = d["time_s"]
    aif_lim = (min(np.nanmin(d[f"{v}_aif"]) for v, _, _ in VARIANTS),
               max(np.nanmax(d[f"{v}_aif"]) for v, _, _ in VARIANTS) * 1.05)

    for row, (variant, name, col), mask in zip(axs, VARIANTS, masks):
        a = row[0]
        a.imshow(d["minip"][sy, sx], cmap="gray")
        a.imshow(np.ma.masked_where(~mask[sy, sx], mask[sy, sx]), cmap=ListedColormap([col]), alpha=0.45)
        a.contour(mask[sy, sx], levels=[0.5], colors=[col], linewidths=1.2)
        a.set(xticks=[], yticks=[])
        a.set_ylabel(full_name if variant == "ica_full" else name, fontsize=11, color=col, fontweight="bold")
        a.set_title(f"MINIP + mask ({int(mask.sum())} px)", fontsize=9)

        a = row[1]
        a.plot(t, d[f"{variant}_aif"], color=col, lw=1.5)
        a.set(ylim=aif_lim, xlabel="Time (s)")
        a.set_title("AIF", fontsize=9)
        a.tick_params(labelsize=7)
        a.xaxis.label.set_size(8)

    for j, key in enumerate(MAPS, start=2):
        vals = np.concatenate([d[f"{v}_{key}"][d["ica_full_CBV"] > 0].ravel() for v, _, _ in VARIANTS])
        vals = vals[np.isfinite(vals)]
        vmin, vmax = np.percentile(vals, [1, 99]) if vals.size else (0, 1)
        for row, (variant, _, _) in zip(axs, VARIANTS):
            m = np.ma.masked_invalid(d[f"{variant}_{key}"])
            im = row[j].imshow(m, cmap="jet", vmin=vmin, vmax=vmax)
            row[j].set(xticks=[], yticks=[])
            row[j].set_title(f"{key}  (median {np.nanmedian(d[f'{variant}_{key}'][d[f'{variant}_CBV'] > 0]):.2f})",
                             fontsize=9)
        plt.colorbar(im, ax=[axs[0][j], axs[1][j]], fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)


def figure(examples, path, full_name):
    n = len(examples)
    fig = plt.figure(figsize=(21, 7.2 * n), layout="constrained")
    subs = fig.subfigures(n, 1, squeeze=False)[:, 0]
    for sub, (d, title) in zip(subs, examples):
        axs = sub.subplots(2, 6, gridspec_kw={"width_ratios": [1, 1.3, 1, 1, 1, 1]})
        draw(axs, d, full_name)
        sub.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", dest="inputs", nargs="+", default=[str(HERE.parent / "PRAISE"),
                                                            str(HERE.parent / "PRAISE_BATCH_2")])
    p.add_argument("-o", dest="out_dir", default=None)
    p.add_argument("-d", dest="device", default="cuda")
    p.add_argument("--ica-full", choices=["grown", "model"], default="grown",
                   help="ICA full mask: grown from ICA top (default, as run_dataset.py) or the ICA-full model.")
    p.add_argument("--redo", action="store_true", help="Recompute the cached maps.")
    args = p.parse_args()
    out = Path(args.out_dir or HERE / "results" / ("ica_model" if args.ica_full == "model" else "") / "aif_examples")
    out.mkdir(parents=True, exist_ok=True)
    examples = EXAMPLES[args.ica_full]
    full_name = f"ICA full ({args.ica_full})"

    wanted = {rid for rid, _, _ in examples}
    runs = {r["run_id"]: r for r in discover_runs(args.inputs, 5) if r["run_id"] in wanted}
    missing = wanted - runs.keys()
    if missing:
        logging.warning("Not found: %s", ", ".join(sorted(missing)))

    device = pick_device(args.device)
    nets = {"ica_top": load_model(ICA_TOP_MODEL, device)}
    if args.ica_full == "model":
        nets["ica_full"] = load_model(ICA_MODEL, device)

    done = {"usable": [], "failure": []}
    for i, (rid, label, title) in enumerate(examples, 1):
        if rid not in runs:
            continue
        cache = out / "maps" / f"{rid}.npz"
        if args.redo:
            cache.unlink(missing_ok=True)
        logging.info("[%d/%d] %s", i, len(examples), rid)
        run = dict(runs[rid])
        run["ds"] = None if cache.exists() else read_dicom(run["source"])
        d = compute(run, nets, device, cache)
        full_title = f"{rid.replace('PRAISE_BATCH_2_', 'BATCH_2 ').replace('PRAISE_PRAISE', 'PRAISE')}  -  {title}"
        figure([(d, full_title)], out / f"{i:02d}_{label}_{rid}.png", full_name)
        done["usable" if label == "usable" else "failure"].append((d, full_title))

    figure(done["usable"], out / "overview_usable.png", full_name)
    figure(done["failure"], out / "overview_failures.png", full_name)
    logging.info("Figures in %s", out)


if __name__ == "__main__":
    main()
