"""
Build the local HTML viewer for a postprocess_combined.py output. It opens straight from disk
(file://): no server, no network requests, nothing leaves the machine.

Output (in <out>/viewer/):
    index.html          the viewer (copied from scripts/viewer_template.html)
    index_data.js       run metadata, perfusion summary, mask overlap, selection and all AIF curves
    data/<run_id>.js    per run: MINIP (JPEG), the three masks (one RGB PNG) and the maps as
                        8-bit PNGs, quantized linearly between 0 and a per-run upper value
                        (99.5th percentile over the variants; Tmax: its maximum), so the browser
                        can recolor, rescale and subtract them

Example:
    python scripts/build_viewer.py results/combined_v1
"""

import argparse
import base64
import json
import math
import shutil
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VARIANTS = ["ica_top", "ica_full_model", "ica_full_grown"]
MAP_KEYS = ["CBV", "CBF", "MTT", "Tmax"]
META_COLS = ["dataset", "patient_id", "run_id", "series_number", "series_description", "plane", "injection_number",
             "n_injections_patient", "injection_site", "injection_site_source", "praise_json_perf_eval", "ab_pair_id",
             "ab_offset_ms_header", "ab_offset_ms_aif_arrival", "acquisition_start", "n_frames", "duration_s",
             "frame_rate_regime", "frame_interval_pattern", "variable_frame_rate", "is_duplicate", "duplicate_of",
             "duplicate_type", "duplicate_scope", "study_date", "model", "primary_angle_deg", "secondary_angle_deg"]
PERF_COLS = ["mask_px", "mask_components", "aif_peak", "aif_ttp_s", "aif_fwhm_s", "aif_auc", "perfused_px",
             "CBV_median", "CBF_median", "MTT_median", "Tmax_median"]


def data_url(img, ext, **params):
    flags = [cv.IMWRITE_JPEG_QUALITY, params["quality"]] if ext == ".jpg" else [cv.IMWRITE_PNG_COMPRESSION, 9]
    ok, buf = cv.imencode(ext, img, flags)
    mime = "image/jpeg" if ext == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buf.tobytes()).decode()}"


def clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(v) else round(float(v), 5)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def run_payload(npz, map_size):
    d = np.load(npz)
    variants = [v for v in VARIANTS if f"mask_{v}" in d]
    minip = np.clip(d["minip"], 0, 255).astype(np.uint8)
    out = {"variants": variants, "minip": data_url(minip, ".jpg", quality=92), "size": int(minip.shape[0])}
    if not variants:
        return out
    masks = np.zeros(minip.shape + (3,), np.uint8)  # B, G, R order for cv.imencode -> R=top, G=model, B=grown
    for i, v in enumerate(VARIANTS):
        if f"mask_{v}" in d:
            masks[..., 2 - i] = d[f"mask_{v}"].astype(np.uint8) * 255
    out["masks"] = data_url(masks, ".png")
    out["maps"], out["map_size"] = {}, map_size
    for k in MAP_KEYS:
        stack = [d[f"{k}_{v}"] for v in variants]
        vals = np.concatenate([s[np.isfinite(s) & (s > 0)] for s in stack])
        hi = float(np.nanmax(vals)) if k == "Tmax" and vals.size else (
            float(np.percentile(vals, 99.5)) if vals.size else 1.0)
        hi = hi if hi > 0 else 1.0
        out["maps"][k] = {"hi": hi}
        for v, s in zip(variants, stack):
            s = np.nan_to_num(s, nan=0.0, posinf=hi, neginf=0.0)
            if s.shape[0] != map_size:
                s = cv.resize(s, (map_size, map_size), interpolation=cv.INTER_AREA)
            q = np.round(np.clip(s / hi, 0, 1) * 255).astype(np.uint8)
            out["maps"][k][v] = data_url(q, ".png")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", help="postprocess_combined.py output folder")
    p.add_argument("--map-size", type=int, default=512, help="Resolution of the maps in the viewer.")
    args = p.parse_args()
    out = Path(args.out)
    dest = out / "viewer"
    (dest / "data").mkdir(parents=True, exist_ok=True)

    acq = pd.read_csv(out / "acquisition_table.csv")
    perf = pd.read_csv(out / "perfusion_summary.csv")
    errors = pd.read_csv(out / "errors.csv").fillna("")
    overlap = pd.read_csv(out / "mask_overlap.csv").set_index("run_id")
    curves = pd.read_csv(out / "aif_curves.csv")
    sel_f = out / "selection.csv"  # from select_runs.py, optional
    selection = pd.read_csv(sel_f).fillna({"reasons": ""}).set_index("run_id") if sel_f.exists() else None
    curves_by_run = {r: g for r, g in curves.groupby("run_id")}

    runs = []
    for _, r in acq.iterrows():
        run_id = r["run_id"]
        meta = {c: clean(r[c]) for c in META_COLS if c in r}
        pr = perf[perf.run_id == run_id]
        meta["perf"] = {row["variant"]: {c: clean(row[c]) for c in PERF_COLS if c in row} for _, row in pr.iterrows()}
        meta["errors"] = [{"variant": e["variant"], "error": str(e["error"])} for _, e in errors[errors.run_id == run_id].iterrows()]
        if selection is not None and run_id in selection.index:
            sr = selection.loc[run_id]
            meta["selection"] = {"included": bool(sr["included"]), "reasons": str(sr["reasons"]),
                                 "evt_phase": sr["evt_phase"],
                                 **{c: clean(sr[c]) for c in sr.index if c.startswith(("venous_frac_", "aif_tail_"))}}
        meta["overlap"] = {k: clean(v) for k, v in overlap.loc[run_id].items()} if run_id in overlap.index else {}
        cv_ = {}
        for (v, s), g in curves_by_run.get(run_id, pd.DataFrame(columns=["variant", "sampling"])).groupby(["variant", "sampling"]):
            cv_.setdefault(v, {})["t" if s == "resampled_250ms" else "tn"] = [round(float(x), 4) for x in g.time_s]
            cv_[v]["a" if s == "resampled_250ms" else "an"] = [clean(float(x)) for x in g.aif]
        meta["curves"] = cv_
        npz = out / "npz" / f"{run_id}.npz"
        meta["has_data"] = npz.exists()
        if npz.exists():
            payload = run_payload(npz, args.map_size)
            (dest / "data" / f"{run_id}.js").write_text(
                f"window.RUN_DATA=window.RUN_DATA||{{}};window.RUN_DATA[{json.dumps(run_id)}]={json.dumps(payload)};")
        runs.append(meta)
        print(run_id)

    (dest / "index_data.js").write_text("window.COHORT=" + json.dumps({"runs": runs, "variants": VARIANTS,
                                                                       "maps": MAP_KEYS}) + ";")
    shutil.copy(HERE / "viewer_template.html", dest / "index.html")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"Viewer with {len(runs)} runs ({size:.0f} MB): {(dest / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
