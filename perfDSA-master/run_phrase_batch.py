"""
run_phrase_batch.py
Run the perfDSA perfusion pipeline over the whole PHRASE dataset in one go,
with two AIF variants per run:
    ica_full : AIF sampled from the full ICA segmentation mask
    ica_top  : AIF sampled from only the top (most cranial) part of that mask

Input : folder with the multi-frame DICOMs produced by stack_ima.py
Output: per variant -> perfusion maps (.npz), mask preview (.png)
        summary.csv  (one row per run x variant: AIF features + median perfusion values)
        aif_curves.csv (long format: run, variant, frame, time_s, intensity)
        errors.csv   (runs that failed, with the error message)

Already-processed runs are skipped, so you can stop and restart safely.
Keep everything on local/approved storage: PHRASE is confidential.

There are 3 places marked  >>> ADAPT <<<  that depend on code I haven't seen
(the ICA model's input format and perfDSA's deconvolution function).
"""

import argparse
import csv
import traceback
from pathlib import Path

import numpy as np
import pydicom
from scipy import ndimage

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **k: x  # noqa: E731

VARIANTS = ["ica_full", "ica_top"]
PARAMS = ["CBF", "CBV", "MTT", "Tmax"]


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_dsa(path):
    """Return frames (T,H,W) float32 and frame times (T,) in seconds."""
    ds = pydicom.dcmread(path)
    frames = ds.pixel_array.astype(np.float32)
    if frames.ndim == 2:
        raise ValueError("single-frame file, expected multi-frame DICOM")
    n = frames.shape[0]

    if "FrameTimeVector" in ds:          # ms between frames, first entry usually 0
        times = np.cumsum(np.asarray(ds.FrameTimeVector, dtype=float)) / 1000.0
    elif "FrameTime" in ds:              # constant ms per frame
        times = np.arange(n) * float(ds.FrameTime) / 1000.0
    else:
        raise ValueError("no FrameTimeVector/FrameTime in header")
    return frames, times[:n]


# ----------------------------------------------------------------------------
# ICA segmentation
# ----------------------------------------------------------------------------
def load_ica_model(model_path, device):
    # >>> ADAPT 1 <<<  load the ICA segmentation model the way its repo does it
    import torch
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval()
    return model


def segment_ica(model, frames, device, threshold=0.5):
    """Return a boolean ICA mask (H,W)."""
    # >>> ADAPT 2 <<<  match the model's expected input (MinIP? specific frames?
    # size? normalisation?). Assumed here: min-intensity projection, scaled 0-1.
    import torch
    minip = frames.min(axis=0)
    img = (minip - minip.min()) / (np.ptp(minip) + 1e-8)
    x = torch.from_numpy(img)[None, None].float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob > threshold


def largest_component(mask):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (np.argmax(sizes) + 1)


def top_of_mask(mask, fraction):
    """Keep only the top `fraction` of the mask's vertical extent (row 0 = top)."""
    mask = largest_component(mask)
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return mask
    cutoff = rows.min() + max(1, int(round(fraction * (rows.max() - rows.min() + 1))))
    out = mask.copy()
    out[cutoff:, :] = False
    return out


# ----------------------------------------------------------------------------
# AIF + perfusion
# ----------------------------------------------------------------------------
def extract_aif(frames, mask, n_baseline=2):
    """Mean intensity in mask per frame, baseline-subtracted."""
    curve = frames[:, mask].mean(axis=1)
    return curve - curve[:n_baseline].mean()


def aif_features(aif, times):
    i_peak = int(np.argmax(np.abs(aif)))
    return {
        "aif_peak": float(aif[i_peak]),
        "aif_ttp_s": float(times[i_peak]),
        "aif_auc": float(np.trapezoid(aif, times)),
    }


def run_perfusion(frames, times, aif, aif_mask):
    """Return dict {param_name: 2D map}."""
    # >>> ADAPT 3 <<<  call perfDSA's own interpolation + deconvolution here,
    # passing YOUR aif/mask instead of the one from its built-in segmentation.
    # e.g.:
    #   from perfDSA.perfusion import compute_maps
    #   return compute_maps(frames, times, aif=aif)
    raise NotImplementedError("hook up perfDSA's deconvolution in run_perfusion()")


# ----------------------------------------------------------------------------
# Batch loop
# ----------------------------------------------------------------------------
def save_mask_png(frames, mask, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(frames.min(axis=0), cmap="gray")
    ax.contour(mask, colors="r", linewidths=0.6)
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def append_rows(path, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def process_run(dcm, model, args, out):
    run_id = dcm.stem
    todo = [v for v in VARIANTS if not (out / v / f"{run_id}.npz").exists()]
    if not todo:
        return "skipped"

    frames, times = load_dsa(dcm)
    full_mask = largest_component(segment_ica(model, frames, args.device))
    if full_mask.sum() < args.min_pixels:
        raise ValueError(f"ICA mask too small ({int(full_mask.sum())} px)")
    masks = {"ica_full": full_mask, "ica_top": top_of_mask(full_mask, args.top_fraction)}

    for v in todo:
        mask = masks[v]
        aif = extract_aif(frames, mask)
        maps = run_perfusion(frames, times, aif, mask)

        np.savez_compressed(out / v / f"{run_id}.npz", mask=mask, aif=aif, times=times, **maps)
        save_mask_png(frames, mask, out / v / f"{run_id}_mask.png")

        row = {"run_id": run_id, "variant": v, "mask_px": int(mask.sum()), **aif_features(aif, times)}
        for p in PARAMS:
            m = maps.get(p)
            row[f"{p}_median"] = float(np.nanmedian(m[m > 0])) if m is not None and (m > 0).any() else np.nan
        append_rows(out / "summary.csv", [row])
        append_rows(out / "aif_curves.csv",
                    [{"run_id": run_id, "variant": v, "frame": i, "time_s": t, "intensity": a}
                     for i, (t, a) in enumerate(zip(times, aif))])
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="folder with stacked multi-frame DICOMs")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-m", "--model", required=True, help="ICA segmentation model weights")
    ap.add_argument("--top-fraction", type=float, default=0.25, help="part of ICA kept for ica_top")
    ap.add_argument("--min-pixels", type=int, default=20)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    out = Path(args.output)
    for v in VARIANTS:
        (out / v).mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in Path(args.input).rglob("*") if p.suffix.lower() in {".dcm", ".ima"})
    print(f"{len(files)} runs found")
    model = load_ica_model(args.model, args.device)

    counts = {"done": 0, "skipped": 0, "failed": 0}
    for dcm in tqdm(files, desc="PHRASE"):
        try:
            counts[process_run(dcm, model, args, out)] += 1
        except Exception as e:
            counts["failed"] += 1
            append_rows(out / "errors.csv", [{"run_id": dcm.stem, "error": repr(e),
                                               "trace": traceback.format_exc(limit=3)}])
    print(counts)


if __name__ == "__main__":
    main()
