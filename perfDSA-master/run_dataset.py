"""
Run perfDSA over the whole dataset with two AIF variants and build an acquisition table.

    ica_top  : AIF sampled from the mask of models/best_model_ica_top.pt  (ICA top)
    ica_full : AIF sampled from the full ICA, grown down the vessel from the ICA-top mask
               on the MINIP, avoiding veins found by their timing (see arterial_full_ica). With
               --ica-full model it comes from models/best_model_ica.pt instead; note that model
               was trained on ICA-top labels as well and gives nearly the same mask.

Input : one or more dataset roots (default ../PRAISE and ../EX_AGNES). Every multi-frame
        DICOM (.dcm / .IMA, also inside .tar archives) is one run; single-frame screenshots
        and reference images are skipped. Runs stored un-subtracted (ImageType ORIGINAL with
        a MaskSubtractionSequence, e.g. EX_AGNES) are subtracted here with their own mask frame.

Output (in -o, default ./results/dataset_run):
    acquisition_table.csv    one row per run: acquisition parameters from the DICOM header
    acquisition_summary.csv  per dataset: median [min-max] of the numeric parameters
    perfusion_summary.csv    one row per run x variant: mask size, AIF features, map medians
    aif_curves.csv           long format: run_id, variant, time_s, aif
    errors.csv               runs that failed, with the error message
    figures/<variant>/<run_id>.png   AIF, MINIP + mask, CBV, CBF, MTT, Tmax
    maps/<variant>/<run_id>.npz      (only with --save-maps)
    runs/<run_id>.json       per-run cache; finished runs are skipped on restart

Examples:
    python run_dataset.py
    python run_dataset.py --table-only
    python run_dataset.py -i ../PRAISE -o results/praise --save-maps
"""

import os

# pip-installed torch inside a conda env ships a second libiomp5md.dll, which aborts the process
# on Windows ("OMP: Error #15"). Tolerate it; the proper fix is removing one copy from the env.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import logging
import re
import sys
import tarfile
import traceback
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pydicom
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from skimage.graph import route_through_array

from DSASequence import DSASequence
from unet import UNet

HERE = Path(__file__).resolve().parent
ICA_TOP_MODEL = HERE / "models" / "best_model_ica_top.pt"
ICA_MODEL = HERE / "models" / "best_model_ica.pt"  # only used with --ica-full model
DICOM_SUFFIXES = (".dcm", ".ima")
DSA_BASE_LEVEL = 170 / 255  # background level DSASequence assumes for subtracted images


# ----------------------------------------------------------------------------
# File access (Windows long paths, files inside .tar archives)
# ----------------------------------------------------------------------------
def long_path(p):
    """PRAISE paths exceed the 260 char Windows limit; the \\\\?\\ prefix lifts it."""
    s = str(Path(p).resolve())
    if os.name == "nt" and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def read_dicom(source, header_only=False):
    """source is a file path, or (tar_path, member_name) for a file inside an archive."""
    if isinstance(source, tuple):
        with tarfile.open(long_path(source[0])) as tar:
            return pydicom.dcmread(tar.extractfile(source[1]), stop_before_pixels=header_only, force=True)
    return pydicom.dcmread(long_path(source), stop_before_pixels=header_only, force=True)


def source_str(source):
    return f"{source[0]}::{source[1]}" if isinstance(source, tuple) else str(source)


def iter_dicom_sources(root):
    for dirpath, _, filenames in os.walk(long_path(root)):
        for fn in sorted(filenames):
            path = Path(dirpath.replace("\\\\?\\", "")) / fn
            if fn.lower().endswith(DICOM_SUFFIXES):
                yield path
            elif fn.lower().endswith(".tar"):
                with tarfile.open(long_path(path)) as tar:
                    for m in tar.getmembers():
                        if m.isfile() and m.name.lower().endswith(DICOM_SUFFIXES):
                            yield (path, m.name)


# ----------------------------------------------------------------------------
# Run discovery + acquisition table
# ----------------------------------------------------------------------------
def is_dsa_run(ds, min_frames):
    image_type = [str(t).upper() for t in ds.get("ImageType", [])]
    return (ds.get("Modality") == "XA" and int(ds.get("NumberOfFrames", 1)) >= min_frames
            and "SECONDARY" not in image_type)


def plane_of(ds):
    last = str(ds.get("ImageType", [""])[-1]).upper()
    m = re.search(r"(?:BIPLANE|SINGLE)\s+([AB])$", last)
    return m.group(1) if m else "A"


def frame_intervals_ms(ds):
    n = int(ds.NumberOfFrames)
    if "FrameTimeVector" in ds and ds.FrameTimeVector is not None:
        return np.asarray(ds.FrameTimeVector, dtype=float)[1:n]
    if "FrameTime" in ds:
        return np.full(n - 1, float(ds.FrameTime))
    return np.array([])


def needs_subtraction(ds):
    image_type = [str(t).upper() for t in ds.get("ImageType", [])]
    return "ORIGINAL" in image_type and "MaskSubtractionSequence" in ds


def mask_frame_numbers(ds):
    """1-based mask frame numbers from the MaskSubtractionSequence."""
    frames = ds.MaskSubtractionSequence[0].get("MaskFrameNumbers", 1)
    return [int(f) for f in (frames if isinstance(frames, (list, pydicom.multival.MultiValue)) else [frames])]


def num(ds, key):
    v = ds.get(key)
    try:
        return float(v[0] if isinstance(v, (list, pydicom.multival.MultiValue)) else v)
    except (TypeError, ValueError, IndexError):
        return np.nan


def acquisition_row(dataset, run_id, ds, source):
    dt = frame_intervals_ms(ds)
    sid, sod = num(ds, "DistanceSourceToDetector"), num(ds, "DistanceSourceToPatient")
    spacing = num(ds, "ImagerPixelSpacing")
    subtract = needs_subtraction(ds)
    return {
        "dataset": dataset,
        "patient_id": str(ds.get("PatientID", "")),
        "run_id": run_id,
        "study_date": str(ds.get("StudyDate", "")),
        "series_number": int(ds.get("SeriesNumber", 0) or 0),
        "series_description": str(ds.get("SeriesDescription", "")),
        "plane": plane_of(ds),
        "manufacturer": str(ds.get("Manufacturer", "")),
        "model": str(ds.get("ManufacturerModelName", "")),
        "detector": str(ds.get("DetectorDescription", "")),
        "n_frames": int(ds.NumberOfFrames),
        "frame_interval_ms_median": float(np.median(dt)) if dt.size else np.nan,
        "frame_interval_ms_min": float(dt.min()) if dt.size else np.nan,
        "frame_interval_ms_max": float(dt.max()) if dt.size else np.nan,
        "frame_rate_fps": 1000 / float(np.median(dt)) if dt.size else np.nan,
        "duration_s": float(dt.sum()) / 1000 if dt.size else np.nan,
        "rows": int(ds.Rows),
        "columns": int(ds.Columns),
        "bits_stored": int(ds.BitsStored),
        "imager_pixel_spacing_mm": spacing,
        "field_of_view_mm": spacing * int(ds.Rows),
        "intensifier_size_mm": num(ds, "IntensifierSize"),
        "kvp": num(ds, "KVP"),
        "tube_current_mA": num(ds, "XRayTubeCurrent"),
        "exposure_time_ms": num(ds, "ExposureTime"),
        "pulse_width_ms": num(ds, "AveragePulseWidth"),
        "focal_spot_mm": num(ds, "FocalSpots"),
        "dap": num(ds, "ImageAndFluoroscopyAreaDoseProduct"),
        "sid_mm": sid,
        "sod_mm": sod,
        "magnification": sid / sod if sod else np.nan,
        "primary_angle_deg": num(ds, "PositionerPrimaryAngle"),
        "secondary_angle_deg": num(ds, "PositionerSecondaryAngle"),
        "contrast_agent": str(ds.get("ContrastBolusAgent", "")),
        "pixel_intensity_relationship": str(ds.get("PixelIntensityRelationship", "")),
        "stored_as": "native (subtracted here)" if subtract else "subtracted",
        "mask_frames": ";".join(map(str, mask_frame_numbers(ds))) if subtract else "",
        "source": source_str(source),
    }


def discover_runs(roots, min_frames):
    runs, seen_uids, used_ids = [], set(), set()
    for root in roots:
        dataset = Path(root).resolve().name
        for source in iter_dicom_sources(root):
            try:
                ds = read_dicom(source, header_only=True)
            except Exception as e:
                logging.warning("Unreadable DICOM %s: %r", source_str(source), e)
                continue
            uid = str(ds.get("SOPInstanceUID", source_str(source)))
            if uid in seen_uids or not is_dsa_run(ds, min_frames):
                continue  # duplicates occur when a .tar was also extracted next to itself
            seen_uids.add(uid)

            base = f"{ds.get('PatientID', 'unknown')}_S{int(ds.get('SeriesNumber', 0) or 0):02d}_{plane_of(ds)}"
            base = re.sub(r"[^\w\-]+", "_", f"{dataset}_{base}")
            run_id, k = base, 2
            while run_id in used_ids:
                run_id, k = f"{base}_{k}", k + 1
            used_ids.add(run_id)
            runs.append({"run_id": run_id, "source": source,
                         "acq": acquisition_row(dataset, run_id, ds, source)})
    return runs


def summarise_acquisitions(acq):
    """Per dataset: n patients/runs and median [min-max] of every numeric column."""
    skip = {"series_number", "rows", "columns", "bits_stored"}
    numeric = [c for c in acq.select_dtypes("number").columns if c not in skip]
    rows = []
    for dataset, g in list(acq.groupby("dataset")) + [("all", acq)]:
        row = {"dataset": dataset, "n_patients": g["patient_id"].nunique(), "n_runs": len(g),
               "matrix": ", ".join(sorted({f"{r}x{c}" for r, c in zip(g["rows"], g["columns"])})),
               "manufacturer_model": ", ".join(sorted(set(g["manufacturer"] + " " + g["model"]))),
               "stored_as": ", ".join(sorted(set(g["stored_as"])))}
        for c in numeric:
            v = g[c].dropna()
            row[c] = f"{v.median():.4g} [{v.min():.4g}, {v.max():.4g}]" if len(v) else ""
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset").T  # parameters as rows, datasets as columns


# ----------------------------------------------------------------------------
# Loading + preprocessing (follows perfDSA.load_and_preprocess_dicom)
# ----------------------------------------------------------------------------
def load_uniform_sequence(ds, frame_interval_ms, size, native_subtraction="log", native_vessel_level=20):
    """Return a uniformly-timed (H,W,N) DSA sequence scaled to 0..255, like perfDSA.py.

    Native runs are subtracted here. "log" mimics the scanner's DSA: subtraction in the log
    domain, then a per-run gain so the densest vessels end up at `native_vessel_level` (0..255),
    as dark as the ICA in the vendor-subtracted images the models were trained on. "linear" is a
    plain frame - mask, which leaves the vessels only slightly darker than the background.
    """
    full_scale = 2 ** ds.BitsStored - 1
    seq = ds.pixel_array.astype(np.float32)
    first = 1  # perfDSA drops the first frame (non-contrast or un-subtracted)

    if needs_subtraction(ds):
        mask_idx = [f - 1 for f in mask_frame_numbers(ds)]
        first = max(mask_idx)  # frames before the mask are not part of the subtracted run
        base = DSA_BASE_LEVEL * full_scale
        mask = seq[mask_idx].mean(axis=0)
        if native_subtraction == "log":
            # contrast signal, > 0 where contrast is; pixels stored as LOG are in that domain already
            if str(ds.get("PixelIntensityRelationship", "")).upper() == "LOG":
                d = mask - seq
            else:
                d = np.log(mask + 1) - np.log(seq + 1)
            peak = float(np.percentile(d[first:].max(axis=0), 99.9))
            gain = (base - native_vessel_level / 255 * full_scale) / peak if peak > 0 else 0.0
            seq = base - gain * d
        else:
            seq = seq - mask + base
        seq = np.clip(seq, 0, full_scale)

    h, w = seq.shape[1:]
    if h != w:  # pad non-square detectors with background so resizing keeps the aspect ratio
        n = max(h, w)
        pad = ((0, 0), ((n - h) // 2, n - h - (n - h) // 2), ((n - w) // 2, n - w - (n - w) // 2))
        seq = np.pad(seq, pad, constant_values=DSA_BASE_LEVEL * full_scale if needs_subtraction(ds)
                     else float(np.median(seq)))
    if size and seq.shape[1:] != (size, size):
        seq = np.stack([cv.resize(f, (size, size), interpolation=cv.INTER_AREA) for f in seq])

    dt = frame_intervals_ms(ds)
    if dt.size != seq.shape[0] - 1:
        raise ValueError("missing frame time information")
    times = np.concatenate([[0], np.cumsum(dt)])
    keep = ~pd.Series(times).duplicated().to_numpy()
    keep[:first] = False
    times, seq = times[keep], seq[keep]

    interp = interp1d(times, seq, axis=0)
    seq = interp(np.arange(times[0], times[-1], frame_interval_ms))
    return np.transpose(255 * seq.astype(np.float32) / full_scale, (1, 2, 0))


# ----------------------------------------------------------------------------
# Segmentation + perfusion
# ----------------------------------------------------------------------------
def pick_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        logging.info("CUDA requested but not available, using CPU instead.")
        return torch.device("cpu")
    return torch.device(device)


def load_model(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict):  # best_model_ica.pt is a state_dict, best_model_ica_top.pt a full model
        net = UNet(n_channels=1, n_classes=1, bilinear=True)
        net.load_state_dict(obj)
        obj = net
    return obj.to(device).eval()


def segment(net, minip, device, threshold=0.5, tta=False, largest_component=False):
    """MINIP (0..255) -> (boolean mask, max probability); the models were trained on MINIPs scaled to 0..1."""
    x = torch.from_numpy(minip / 255).float()[None, None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(net(x))
        if tta:  # training augmented with horizontal flips, so the flipped MINIP is in distribution
            prob = (prob + torch.sigmoid(net(x.flip(-1))).flip(-1)) / 2
    prob = prob.squeeze().cpu().numpy()
    mask = prob > threshold
    if largest_component and mask.any():
        _, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))
        mask = labels == 1 + int(np.argmax(stats[1:, cv.CC_STAT_AREA]))
    return mask, float(prob.max())


def grow_full_ica(minip, top_mask, venous=None, contrast_fraction=0.35, min_width_px=5, border_px=20):
    """Extend the ICA-top mask down the ICA by region growing on the MINIP.

    1. Vessels: pixels at least `contrast_fraction` as opaque as the ICA top (the ICA is the
       densest vessel in an ICA injection), at least `min_width_px` wide, away from the image
       border and from black collimator/shutter areas.
    2. Growth is limited to rows below the top of the ICA-top mask, because the ICA top
       continues distally into MCA/ACA, which run level with or above it.
    3. In the vessel region connected to the ICA top, the centreline path from the ICA top to
       the lowest non-venous vessel pixel is the ICA; vessel pixels within one vessel width
       of that path are kept (width taken from the ICA top), which prunes side branches.
       Pixels in `venous` are avoided by the path and left out of the mask.
    """
    if not top_mask.any():
        return top_mask
    venous = np.zeros_like(top_mask) if venous is None else venous & ~top_mask
    smooth = cv.GaussianBlur(minip.astype(np.float32), (0, 0), 2)
    contrast = np.median(smooth) - smooth  # vessels are dark in the MINIP
    vessel = (contrast > contrast_fraction * np.median(contrast[top_mask])) & (minip > 5)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (min_width_px, min_width_px))
    vessel = cv.morphologyEx(vessel.astype(np.uint8), cv.MORPH_OPEN, kernel).astype(bool)
    vessel[:np.nonzero(top_mask.any(axis=1))[0].min()] = False
    vessel[:border_px] = vessel[-border_px:] = False
    vessel[:, :border_px] = vessel[:, -border_px:] = False
    vessel |= top_mask

    n, labels = cv.connectedComponents(vessel.astype(np.uint8))
    region = np.isin(labels, np.unique(labels[top_mask]))
    ys, xs = np.nonzero(region & ~venous)
    end = (ys.max(), xs[ys == ys.max()][len(xs[ys == ys.max()]) // 2])
    tys, txs = np.nonzero(top_mask)
    start = (tys[len(tys) // 2], txs[len(txs) // 2])
    if end[0] <= tys.max():
        return top_mask  # nothing found below the ICA top

    dist = cv.distanceTransform(region.astype(np.uint8), cv.DIST_L2, 5)
    # cheapest along the vessel centre; crossing a vein (overlap) is allowed but expensive
    cost = np.where(region, (1 + 20 * venous) / (dist + 1), np.inf)
    path, _ = route_through_array(cost, start, end, fully_connected=True, geometric=True)
    path_mask = np.zeros_like(region)
    path_mask[tuple(np.array(path).T)] = True
    # vessel radius from the ICA top: the noisy region around a faint ICA would overestimate it
    radius = int(np.ceil(1.5 * cv.distanceTransform(top_mask.astype(np.uint8), cv.DIST_L2, 5).max())) + 1
    tube = cv.dilate(path_mask.astype(np.uint8),
                     cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1,) * 2)).astype(bool)
    return top_mask | (region & tube & ~venous)


def arterial_full_ica(frames, top_mask, dt_s, arterial_window_s=1.0, venous_delay_s=2.0, extra_late_ratio=0.3):
    """Full-ICA mask without veins.

    In projection the jugular vein and venous sinuses overlap the cervical ICA, and the MINIP
    cannot tell them apart. Venous pixels are the ones that stay opaque clearly longer than the
    ICA top itself: their late/early peak ratio (late = from `venous_delay_s` after the ICA-top
    peak, early = up to `arterial_window_s` after it) exceeds the ICA top's own ratio by more
    than `extra_late_ratio`. Being relative to the ICA top keeps slow or long injections, where
    the ICA itself stays opaque, from being marked venous. The ICA is grown on the full MINIP
    (best connectivity) while avoiding and excluding the venous pixels.
    """
    if not top_mask.any():
        return top_mask
    conc = 1 - frames / 170  # DSASequence's concentration scale, contrast arrival > 0
    top_curve = conc[top_mask].mean(axis=0)
    i_peak = int(np.argmax(top_curve))
    i_art = min(conc.shape[2], i_peak + 1 + int(round(arterial_window_s / dt_s)))
    i_late = i_peak + int(round(venous_delay_s / dt_s))

    venous = None
    if i_late < conc.shape[2]:
        early = conc[..., :i_art].max(axis=2)
        late = conc[..., i_late:].max(axis=2)
        top_ratio = top_curve[i_late:].max() / top_curve[:i_art].max()
        venous = late > (top_ratio + extra_late_ratio) * np.maximum(early, 0)
    return grow_full_ica(frames.min(axis=2), top_mask, venous)


def aif_features(aif, t):
    aif = np.asarray(aif, dtype=float)
    i = int(np.argmax(aif))
    half = aif[i] / 2
    above = np.where(aif >= half)[0]
    return {"aif_peak": float(aif[i]), "aif_ttp_s": float(t[i]),
            "aif_fwhm_s": float(t[above.max()] - t[above.min()]) if above.size else np.nan,
            "aif_auc": float(np.trapezoid(aif, t))}


def map_stats(maps):
    """Median/mean over pixels with CBV > 0 (perfused pixels; background has CBV = 0)."""
    valid = (maps["CBV"] > 0) & np.all([np.isfinite(m) for m in maps.values()], axis=0)
    out = {"perfused_px": int(valid.sum())}
    for k, m in maps.items():
        out[f"{k}_median"] = float(np.median(m[valid])) if valid.any() else np.nan
        out[f"{k}_mean"] = float(np.mean(m[valid])) if valid.any() else np.nan
    return out


def save_figure(dsa, mask, maps, title, path, dpi):
    minip = cv.cvtColor(dsa.MINIP().astype(np.uint8), cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    cv.drawContours(minip, contours, -1, (255, 0, 0), thickness=2)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10), tight_layout=True)
    ax[0, 0].plot(dsa.Time(), dsa.AIF())
    ax[0, 0].set(title="AIF", ylabel="Dye concentration [a.u.]", xlabel="Time (s)")
    ax[1, 0].imshow(minip)
    ax[1, 0].set(title="MINIP + AIF mask", xticks=[], yticks=[])
    for a, key in zip([ax[0, 1], ax[1, 1], ax[0, 2], ax[1, 2]], ["CBV", "CBF", "MTT", "Tmax"]):
        im = a.imshow(maps[key], cmap="jet")
        a.set(title=key, xticks=[], yticks=[])
        fig.colorbar(im, ax=a, orientation="vertical", fraction=0.046, pad=0.04)
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def process_run(run, nets, device, args, out):
    ds = read_dicom(run["source"])
    frames = load_uniform_sequence(ds, args.frame_interval, args.size,
                                   args.native_subtraction, args.native_vessel_level)
    dt_s = args.frame_interval / 1000
    minip = frames.min(axis=2)

    seg_args = dict(threshold=args.threshold, tta=args.tta, largest_component=args.largest_component)
    top, top_prob_max = segment(nets["ica_top"], minip, device, **seg_args)
    if args.ica_full == "model":
        full, _ = segment(nets["ica_full"], minip, device, **seg_args)
    else:
        full = arterial_full_ica(frames, top, dt_s)
    masks = {"ica_full": full, "ica_top": top}

    result = {"run_id": run["run_id"], "n_frames_uniform": frames.shape[2],
              "top_prob_max": top_prob_max, "variants": {}}
    for variant, mask in masks.items():
        if not mask.any():  # keep the other variant; report this one in errors.csv
            result["variants"][variant] = {"error": "empty ICA mask"}
            continue
        n_components = cv.connectedComponents(mask.astype(np.uint8))[0] - 1

        dsa = DSASequence(frames, frame_interval=dt_s)
        aif = np.asarray(np.ma.filled(dsa.sample_AIF_from_mask(mask), np.nan), dtype=float)
        dsa._aif = aif
        cbv, cbf, mtt, tmax = dsa.get_perfusion_parameters()
        maps = {"CBV": cbv, "CBF": cbf, "MTT": mtt, "Tmax": tmax.astype(np.float32)}

        save_figure(dsa, mask, maps, f"{run['run_id']} - {variant}",
                    out / "figures" / variant / f"{run['run_id']}.png", args.dpi)
        if args.save_maps:
            (out / "maps" / variant).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "maps" / variant / f"{run['run_id']}.npz", mask=mask, aif=aif,
                                time_s=dsa.Time(), **{k: v.astype(np.float32) for k, v in maps.items()})

        result["variants"][variant] = {
            "mask_px": int(mask.sum()), "mask_components": int(n_components),
            **aif_features(aif, dsa.Time()), **map_stats(maps),
            "time_s": dsa.Time().tolist(), "aif": aif.tolist()}
    return result


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description="Run perfDSA (ICA full + ICA top) over a whole dataset.")
    p.add_argument("-i", dest="inputs", nargs="+",
                   default=[str(HERE.parent / "PRAISE"), str(HERE.parent / "EX_AGNES")],
                   help="Dataset root folder(s). Default: ../PRAISE ../EX_AGNES")
    p.add_argument("-o", dest="out_dir", default=str(HERE / "results" / "dataset_run"), help="Output folder.")
    p.add_argument("-d", dest="device", default="cuda", help='"cuda" or "cpu".')
    p.add_argument("-f", dest="frame_interval", default=250, type=int,
                   help="Frame interval in ms of the uniformly-timed sequence.")
    p.add_argument("--size", type=int, default=1024,
                   help="Resize frames to size x size (models were trained at 1024). 0 = keep native size.")
    p.add_argument("--min-frames", type=int, default=5, help="Skip DICOMs with fewer frames.")
    p.add_argument("--save-maps", action="store_true", help="Also store the perfusion maps as .npz.")
    p.add_argument("--dpi", type=int, default=100, help="Resolution of the figures.")
    p.add_argument("--table-only", action="store_true", help="Only write the acquisition table.")
    p.add_argument("--redo", action="store_true", help="Reprocess runs that already have results.")
    p.add_argument("--sample", type=int, default=0, help="Only process N runs spread over the dataset.")
    p.add_argument("--ica-full", choices=["grown", "model"], default="grown",
                   help="ica_full mask: grown from the ICA top (default), or from models/best_model_ica.pt.")
    p.add_argument("--native-subtraction", choices=["log", "linear"], default="log",
                   help="Subtraction of un-subtracted (native) runs: log domain with contrast gain "
                        "(default), or plain linear frame - mask.")
    p.add_argument("--native-vessel-level", type=float, default=20,
                   help="Grey level (0..255) the densest vessels get with --native-subtraction log; "
                        "background is 170. Lower = more contrast.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Probability threshold of the segmentation. Lower = larger masks.")
    p.add_argument("--tta", action="store_true",
                   help="Average the segmentation with that of the horizontally flipped MINIP.")
    p.add_argument("--largest-component", action="store_true",
                   help="Keep only the largest connected component of the segmentation.")
    return p.parse_args()


def write_outputs(out, runs):
    rows, curves, errors = [], [], []
    for run in runs:
        f = out / "runs" / f"{run['run_id']}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        if "error" in res:
            errors.append({"run_id": run["run_id"], "variant": "all", "error": res["error"], "trace": res["trace"]})
            continue
        acq = run["acq"]
        for variant, v in res["variants"].items():
            if "error" in v:
                errors.append({"run_id": run["run_id"], "variant": variant, "error": v["error"], "trace": ""})
                continue
            rows.append({"dataset": acq["dataset"], "patient_id": acq["patient_id"], "run_id": run["run_id"],
                         "plane": acq["plane"], "variant": variant, "top_prob_max": res.get("top_prob_max"),
                         **{k: x for k, x in v.items() if k not in ("time_s", "aif")}})
            curves += [{"run_id": run["run_id"], "variant": variant, "time_s": t, "aif": a}
                       for t, a in zip(v["time_s"], v["aif"])]
    pd.DataFrame(rows).to_csv(out / "perfusion_summary.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "aif_curves.csv", index=False)
    pd.DataFrame(errors, columns=["run_id", "variant", "error", "trace"]).to_csv(out / "errors.csv", index=False)
    return len(rows), len(errors)


def main():
    logging.basicConfig(level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S",
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = get_args()
    out = Path(args.out_dir)
    (out / "runs").mkdir(parents=True, exist_ok=True)

    logging.info("Scanning %s", ", ".join(args.inputs))
    runs = discover_runs(args.inputs, args.min_frames)
    acq = pd.DataFrame([r["acq"] for r in runs])
    acq.to_csv(out / "acquisition_table.csv", index=False)
    summarise_acquisitions(acq).to_csv(out / "acquisition_summary.csv")
    logging.info("Found %d runs from %d patients -> %s", len(runs), acq["patient_id"].nunique(),
                 out / "acquisition_table.csv")
    if args.table_only:
        return

    device = pick_device(args.device)
    nets = {"ica_top": load_model(ICA_TOP_MODEL, device)}
    if args.ica_full == "model":
        nets["ica_full"] = load_model(ICA_MODEL, device)
    if args.sample:  # evenly spread over the run list, so both datasets are covered
        runs = [runs[i] for i in np.unique(np.linspace(0, len(runs) - 1, args.sample).astype(int))]
        logging.info("Processing a sample of %d runs", len(runs))

    for i, run in enumerate(runs, 1):
        f = out / "runs" / f"{run['run_id']}.json"
        if f.exists() and not args.redo and "error" not in json.loads(f.read_text()):
            logging.info("[%d/%d] %s: done already, skipping", i, len(runs), run["run_id"])
            continue
        logging.info("[%d/%d] %s", i, len(runs), run["run_id"])
        try:
            res = process_run(run, nets, device, args, out)
        except Exception as e:
            logging.error("%s failed: %r", run["run_id"], e)
            res = {"run_id": run["run_id"], "error": repr(e), "trace": traceback.format_exc(limit=4)}
        f.write_text(json.dumps(res))

    n_rows, n_err = write_outputs(out, runs)
    logging.info("Wrote %d run x variant rows (%d failures) to %s", n_rows, n_err, out)


if __name__ == "__main__":
    main()
