"""
Shared helpers for the combined-cohort post-processing (mask_overlap.py, postprocess_combined.py,
build_viewer.py). Reuses run_dataset.py as a module, so the DICOM loading and preprocessing are
exactly the ones the perfusion run used.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]  # perfDSA-master
sys.path.insert(0, str(ROOT))

import cv2 as cv  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_dataset as rd  # noqa: E402

VARIANTS = ("ica_top", "ica_full")
MAP_KEYS = ("CBV", "CBF", "MTT", "Tmax")


def parse_source(s):
    """Inverse of run_dataset.source_str: 'tar::member' -> (Path(tar), member), else a Path."""
    if "::" in s:
        tar, member = s.split("::", 1)
        return Path(tar), member
    return Path(s)


def load_native_sequence(ds, size=1024, native_subtraction="log", native_vessel_level=20):
    """Same preprocessing as run_dataset.load_uniform_sequence, but without the resampling to a
    uniform frame interval. Returns (frames (H,W,N) on the 0..255 scale, times_s (N,)), with time 0
    at the first kept frame, which is also time 0 of the resampled sequence."""
    full_scale = 2 ** ds.BitsStored - 1
    seq = ds.pixel_array.astype(np.float32)
    first = 1
    if rd.needs_subtraction(ds):
        mask_idx = [f - 1 for f in rd.mask_frame_numbers(ds)]
        first = max(mask_idx)
        base = rd.DSA_BASE_LEVEL * full_scale
        mask = seq[mask_idx].mean(axis=0)
        if native_subtraction == "log":
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
    if h != w:
        n = max(h, w)
        pad = ((0, 0), ((n - h) // 2, n - h - (n - h) // 2), ((n - w) // 2, n - w - (n - w) // 2))
        seq = np.pad(seq, pad, constant_values=rd.DSA_BASE_LEVEL * full_scale if rd.needs_subtraction(ds)
                     else float(np.median(seq)))
    if size and seq.shape[1:] != (size, size):
        seq = np.stack([cv.resize(f, (size, size), interpolation=cv.INTER_AREA) for f in seq])

    dt = rd.frame_intervals_ms(ds)
    if dt.size != seq.shape[0] - 1:
        raise ValueError("missing frame time information")
    times = np.concatenate([[0], np.cumsum(dt)])
    keep = ~pd.Series(times).duplicated().to_numpy()
    keep[:first] = False
    times, seq = times[keep], seq[keep]
    frames = np.transpose(255 * seq.astype(np.float32) / full_scale, (1, 2, 0))
    return frames, (times - times[0]) / 1000


def aif_from_frames(frames, mask):
    """AIF as DSASequence.sample_AIF_from_mask computes it: mean of 1 - frame/170 inside the mask."""
    return (1 - frames[mask] / 170).mean(axis=0) if mask.any() else np.full(frames.shape[2], np.nan)


def load_variant_npz(out, variant, run_id):
    f = Path(out) / "maps" / variant / f"{run_id}.npz"
    return dict(np.load(f)) if f.exists() else None


def dice(a, b):
    s = a.sum() + b.sum()
    return float(2 * (a & b).sum() / s) if s else np.nan
