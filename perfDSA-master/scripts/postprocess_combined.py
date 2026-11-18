"""Merge raw_model/ and raw_grown/ into one result with three AIF variants (npz bundles, tables, dataset report).

    python scripts/postprocess_combined.py
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cohort_common import (MAP_KEYS, ROOT, aif_from_frames, dice, load_native_sequence, load_variant_npz,
                           parse_source, rd)
from paths import DATA

# variant name -> (raw run folder, variant inside that run)
SOURCES = {"ica_top": ("raw_model", "ica_top"),
           "ica_full_model": ("raw_model", "ica_full"),
           "ica_full_grown": ("raw_grown", "ica_full")}
VARIANTS = list(SOURCES)
HEADER_TIME_RESOLUTION_MS = 1000  # AcquisitionTime is stored to the second in these exports


def plane_or_unknown(ds):
    last = str(ds.get("ImageType", [""])[-1]).upper()
    m = re.search(r"(?:BIPLANE|SINGLE)\s+([AB])$", last)
    return m.group(1) if m else "unknown"


def acquisition_start(ds):
    d, t = str(ds.get("AcquisitionDate", "") or ds.get("SeriesDate", "")), str(ds.get("AcquisitionTime", "") or "")
    if len(d) != 8 or len(t) < 6:
        return ""
    frac = t[7:13] if "." in t else ""
    return f"{d[:4]}-{d[4:6]}-{d[6:]}T{t[:2]}:{t[2:4]}:{t[4:6]}" + (f".{frac[:3]}" if frac.strip("0") else "")


def frame_pattern(intervals_ms):
    """Run-length pattern of the frame intervals, e.g. '11x250, 9x500, 2x1000 ms' and '4-2-1 fps'."""
    if not len(intervals_ms):
        return "", ""
    segs = []
    for v in intervals_ms:
        if segs and segs[-1][0] == v:
            segs[-1][1] += 1
        else:
            segs.append([v, 1])
    pattern = ", ".join(f"{n}x{v:g}" for v, n in segs) + " ms"
    fps = []
    for v, _ in segs:
        f = f"{1000 / v:g}" if v else "inf"
        if not fps or fps[-1] != f:
            fps.append(f)
    return pattern, "-".join(fps) + " fps"


SIDE = {"LEFT": "L", "LT": "L", "L": "L", "RIGHT": "R", "RT": "R", "R": "R"}
VESSEL = {"ICA": "ICA", "CCA": "CCA", "ECA": "ECA", "VA": "VA", "VERT": "VA", "VERTEBRAL": "VA"}


def parse_site(text):
    """Injection site ('L ICA', 'R CCA', 'ICA', ...) from free text such as 'Final- Left CCA' or a
    folder name such as 'FINALLTICAIC'. Returns '' when no vessel is named."""
    t = re.sub(r"[^A-Z]", " ", str(text).upper())
    squashed = t.replace(" ", "")
    # side directly followed by a vessel also matches concatenated folder names (LTICA, RIGHTCCA)
    m = re.search(r"(LEFT|RIGHT|LT|RT|L|R)(ICA|CCA|ECA|VERTEBRAL|VERT)", squashed)
    if m:
        return f"{SIDE[m.group(1)]} {VESSEL[m.group(2)]}"
    m = re.search(r"\b(ICA|CCA|ECA|VA|VERT|VERTEBRAL)\b", t)
    if not m:
        m = re.search(r"(ICA|CCA|ECA)", squashed)
    if not m:
        return ""
    side = re.search(r"\b(LEFT|RIGHT|LT|RT|L|R)\b", t)
    return f"{SIDE[side.group(1)]} {VESSEL[m.group(1)]}" if side else VESSEL[m.group(1)]


def series_folder(source):
    """'.../XA/20190306T103820/LEFTICA_1.3.12...' -> ('20190306T103820', 'LEFTICA')."""
    member = source.split("::", 1)[-1].replace("\\", "/")
    parts = member.split("/")
    folder = re.sub(r"_\d+(\.\d+){4,}$", "", parts[-2]) if len(parts) >= 2 else ""
    folder = re.sub(r"^\d+_", "", folder)  # '2_LT_CCA_IC' -> 'LT_CCA_IC'
    return (parts[-3] if len(parts) >= 3 else ""), folder


def load_praise_json():
    f = ROOT.parent / "PRAISE_BATCH_2" / "PRAISE.json"
    if not f.exists():
        return {}
    out = {}
    for pid, v in json.loads(f.read_text()).items():
        path = v.get("DSA_perf_eval")
        if isinstance(path, str) and path:
            parts = path.replace("\\", "/").split("/")
            out[pid] = (parts[-2], re.sub(r"^\d+_", "", re.sub(r"_\d+(\.\d+){4,}$", "", parts[-1])))
        else:
            out[pid] = None
    return out


def process_run(out, row, masks_available):
    run_id = row["run_id"]
    ds = rd.read_dicom(parse_source(row["source"]))
    pixels = ds.pixel_array
    intervals = rd.frame_intervals_ms(ds)
    pattern, regime = frame_pattern(intervals)
    info = {
        "run_id": run_id,
        "pixel_md5": hashlib.md5(np.ascontiguousarray(pixels).tobytes()).hexdigest(),
        "plane": plane_or_unknown(ds),
        "image_type": "\\".join(map(str, ds.get("ImageType", []))),
        "acquisition_start": acquisition_start(ds),
        "content_time": str(ds.get("ContentTime", "")),
        "study_instance_uid": str(ds.get("StudyInstanceUID", "")),
        "series_instance_uid": str(ds.get("SeriesInstanceUID", "")),
        "sop_instance_uid": str(ds.get("SOPInstanceUID", "")),
        "acquisition_number": int(ds.get("AcquisitionNumber", 0) or 0),
        "variable_frame_rate": bool(len(set(np.round(intervals, 3))) > 1),
        "frame_interval_pattern": pattern,
        "frame_rate_regime": regime,
    }

    try:
        frames, t_native = load_native_sequence(ds)
        uniform = rd.load_uniform_sequence(ds, 250, 1024)
    except Exception as e:
        info.update(variants=list(masks_available), native={}, bundle_error=repr(e))
        return info
    minip = uniform.min(axis=2)
    bundle = {"minip": minip.astype(np.float32), "time_native_s": t_native}
    for variant in masks_available:
        raw, v = SOURCES[variant]
        npz = load_variant_npz(out / raw, v, run_id)
        mask = npz["mask"].astype(bool)
        bundle[f"mask_{variant}"] = mask
        bundle[f"aif_{variant}"] = npz["aif"]
        bundle["time_s"] = npz["time_s"]
        bundle[f"aif_native_{variant}"] = aif_from_frames(frames, mask)
        for k in MAP_KEYS:
            bundle[f"{k}_{variant}"] = npz[k].astype(np.float32)
    (out / "npz").mkdir(exist_ok=True)
    np.savez_compressed(out / "npz" / f"{run_id}.npz", **bundle)

    info["variants"] = list(masks_available)
    info["native"] = {"time_s": t_native.tolist(),
                      **{v: bundle[f"aif_native_{v}"].tolist() for v in masks_available}}
    return info


def available_variants(out, run_id):
    ok = []
    for variant, (raw, v) in SOURCES.items():
        f = out / raw / "runs" / f"{run_id}.json"
        if not f.exists():
            continue
        res = json.loads(f.read_text())
        if "error" not in res and "error" not in res["variants"].get(v, {"error": 1}) \
                and (out / raw / "maps" / v / f"{run_id}.npz").exists():
            ok.append(variant)
    return ok


def arrival_time(t, aif, frac=0.2):
    """Bolus arrival: first time the AIF rises above baseline + frac * (peak - baseline)."""
    t, aif = np.asarray(t, float), np.asarray(aif, float)
    if not np.isfinite(aif).any():
        return np.nan
    base = np.nanmedian(aif[:2])
    i_peak = int(np.nanargmax(aif))
    level = base + frac * (aif[i_peak] - base)
    above = np.nonzero(aif[:i_peak + 1] >= level)[0]
    if not above.size or above[0] == 0:
        return float(t[0]) if above.size else np.nan
    i = above[0]
    return float(np.interp(level, [aif[i - 1], aif[i]], [t[i - 1], t[i]]))


def build_acquisition_table(acq, infos, praise_json):
    info = pd.DataFrame([{k: v for k, v in i.items() if k not in ("native", "variants")} for i in infos])
    acq = acq.drop(columns=["plane"]).merge(info, on="run_id", how="left")
    acq["patient_id"] = acq["patient_id"].astype(str)

    # injection site: SeriesDescription, else the series folder name, else PRAISE.json
    sites, sources, in_json = [], [], []
    for _, r in acq.iterrows():
        date_dir, folder = series_folder(r["source"])
        entry = praise_json.get(r["patient_id"].split("_")[0], "missing")
        is_eval = entry not in ("missing", None) and entry == (date_dir, folder)
        in_json.append("not listed" if entry == "missing" else ("no perf-eval run" if entry is None
                                                                 else ("yes" if is_eval else "no")))
        for src, text in (("SeriesDescription", r["series_description"]), ("folder name", folder),
                          ("PRAISE.json", entry[1] if is_eval else "")):
            site = parse_site(text)
            if site:
                sites.append(site)
                sources.append(src)
                break
        else:
            sites.append("unknown")
            sources.append("none")
    acq["injection_site"] = sites
    acq["injection_side"] = [s.split()[0] if len(s.split()) == 2 else "unknown" for s in sites]
    acq["injection_vessel"] = [s.split()[-1] if s != "unknown" else "unknown" for s in sites]
    acq["injection_site_source"] = sources
    acq["praise_json_perf_eval"] = in_json

    # duplicates: identical pixel data, or the same acquisition (patient, start, series, plane, frames)
    acq["acq_key"] = (acq["patient_id"] + "|" + acq["acquisition_start"].astype(str) + "|"
                      + acq["series_number"].astype(str) + "|" + acq["plane"] + "|" + acq["n_frames"].astype(str))
    order = {"PRAISE": 0}
    acq["_ord"] = acq["dataset"].map(lambda d: order.get(d, 1))
    acq = acq.sort_values(["_ord", "patient_id", "acquisition_start", "series_number", "plane", "run_id"])
    acq["duplicate_of"], acq["duplicate_type"], acq["duplicate_scope"] = "", "", ""
    for key, how in (("pixel_md5", "identical pixel data"), ("acq_key", "same acquisition, different pixel data")):
        for _, g in acq[acq["duplicate_of"] == ""].groupby(key):
            if len(g) < 2:
                continue
            first = g.iloc[0]
            for idx in g.index[1:]:
                acq.loc[idx, "duplicate_of"] = first["run_id"]
                acq.loc[idx, "duplicate_type"] = how
                acq.loc[idx, "duplicate_scope"] = ("between batches" if acq.loc[idx, "dataset"] != first["dataset"]
                                                   else f"within {first['dataset']}")
    acq["is_duplicate"] = acq["duplicate_of"] != ""

    # injection number per patient: distinct series (per acquisition date) in time order. A and B
    # planes of one injection and duplicate copies share the number; their AcquisitionTime can
    # differ by a second (PRAISE026), so the time itself is not part of the key
    acq["injection_key"] = acq["acquisition_start"].astype(str).str[:10] + "|" + acq["series_number"].astype(str)
    inj = {}
    for pid, g in acq.groupby("patient_id"):
        first = g.groupby("injection_key")["acquisition_start"].min().sort_values()
        for n, key in enumerate(first.index, 1):
            inj[(pid, key)] = n
    acq["injection_number"] = [inj[(p, k)] for p, k in zip(acq["patient_id"], acq["injection_key"])]
    acq["n_injections_patient"] = acq.groupby("patient_id")["injection_number"].transform("max")

    # A/B pairs among the non-duplicate runs of one injection
    acq["ab_pair_id"], acq["ab_offset_ms_header"] = "", np.nan
    for (pid, n), g in acq[~acq["is_duplicate"]].groupby(["patient_id", "injection_number"]):
        planes = set(g["plane"])
        if {"A", "B"} <= planes:
            pair_id = f"{pid}_I{n:02d}"
            a, b = g[g.plane == "A"].iloc[0], g[g.plane == "B"].iloc[0]
            ta, tb = pd.to_datetime(a["acquisition_start"]), pd.to_datetime(b["acquisition_start"])
            offset = (tb - ta).total_seconds() * 1000
            rows = acq.index[(acq.patient_id == pid) & (acq.injection_number == n)]
            acq.loc[rows, "ab_pair_id"] = pair_id
            acq.loc[rows, "ab_offset_ms_header"] = offset
    acq["ab_offset_header_resolution_ms"] = np.where(acq["ab_pair_id"] != "", HEADER_TIME_RESOLUTION_MS, np.nan)
    acq["ab_role"] = np.where(acq["ab_pair_id"] != "",
                              np.where(acq["is_duplicate"], "duplicate", acq["plane"]), "")
    return acq.drop(columns=["_ord", "injection_key"])


def add_aif_offsets(acq, infos):
    """Data-derived A/B timing: bolus arrival (20 % rise) of the native-time ICA-top AIF, B minus A."""
    native = {i["run_id"]: i["native"] for i in infos}
    acq["aif_arrival_s_ica_top"] = [arrival_time(native[r]["time_s"], native[r]["ica_top"])
                                    if r in native and "ica_top" in native[r] else np.nan for r in acq.run_id]
    acq["ab_offset_ms_aif_arrival"] = np.nan
    for pair_id, g in acq[(acq.ab_pair_id != "") & ~acq.is_duplicate].groupby("ab_pair_id"):
        a, b = g[g.plane == "A"].aif_arrival_s_ica_top, g[g.plane == "B"].aif_arrival_s_ica_top
        if len(a) and len(b):
            acq.loc[acq.ab_pair_id == pair_id, "ab_offset_ms_aif_arrival"] = 1000 * (b.iloc[0] - a.iloc[0])
    return acq


def build_perfusion_and_errors(out):
    perf, errs = [], []
    for variant, (raw, v) in SOURCES.items():
        if not (out / raw / "perfusion_summary.csv").exists():
            continue
        p = pd.read_csv(out / raw / "perfusion_summary.csv")
        p = p[p.variant == v].assign(variant=variant)
        perf.append(p)
        e = pd.read_csv(out / raw / "errors.csv").fillna("")
        e = e[(e.variant == v) | (e.variant == "all")].assign(variant=variant)
        errs.append(e)
    return pd.concat(perf, ignore_index=True), pd.concat(errs, ignore_index=True)


def build_aif_curves(out, run_ids, infos, arrival):
    """arrival: run_id -> bolus arrival (s) of that run's ICA-top AIF, used for every variant of the run."""
    native = {i["run_id"]: i["native"] for i in infos}
    rows = []
    for run_id in run_ids:
        f = out / "npz" / f"{run_id}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        for variant in VARIANTS:
            if f"aif_{variant}" not in d:
                continue
            rows += [(run_id, variant, "resampled_250ms", t, a) for t, a in zip(d["time_s"], d[f"aif_{variant}"])]
            nat = native.get(run_id, {})
            rows += [(run_id, variant, "native", t, a) for t, a in zip(nat.get("time_s", []), nat.get(variant, []))]
    df = pd.DataFrame(rows, columns=["run_id", "variant", "sampling", "time_s", "aif"])
    df["time_from_arrival_s"] = df["time_s"] - df["run_id"].map(arrival)
    return df


def build_mask_overlap(out, run_ids):
    rows = []
    for run_id in run_ids:
        f = out / "npz" / f"{run_id}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        m = {v: d[f"mask_{v}"] for v in VARIANTS if f"mask_{v}" in d}
        row = {"run_id": run_id, **{f"{v}_px": int(x.sum()) for v, x in m.items()}}
        for a, b in (("ica_top", "ica_full_model"), ("ica_top", "ica_full_grown"), ("ica_full_model", "ica_full_grown")):
            if a in m and b in m:
                row[f"dice_{a}_vs_{b}"] = dice(m[a], m[b])
                row[f"overlap_px_{a}_vs_{b}"] = int((m[a] & m[b]).sum())
                ra, rb = d[f"aif_{a}"], d[f"aif_{b}"]
                row[f"aif_r_{a}_vs_{b}"] = float(np.corrcoef(ra, rb)[0, 1]) if np.isfinite(ra).all() and np.isfinite(rb).all() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path, nargs="?", default=DATA, help="data folder")
    p.add_argument("--redo", action="store_true", help="Recompute the per-run cache and npz bundles.")
    p.add_argument("--available-only", action="store_true", help="Only runs that already have results (testing).")
    args = p.parse_args()
    out = Path(args.out)
    cache = out / "_cache"
    cache.mkdir(exist_ok=True)

    acq = pd.read_csv(out / "raw_model" / "acquisition_table.csv")
    if args.available_only:
        acq = acq[[bool(available_variants(out, r)) for r in acq.run_id]]
    infos = []
    for i, row in enumerate(acq.to_dict("records"), 1):
        run_id = row["run_id"]
        avail = available_variants(out, run_id)
        f = cache / f"{run_id}.json"
        if f.exists() and not args.redo:
            info = json.loads(f.read_text())
            if info.get("variants") == avail and (out / "npz" / f"{run_id}.npz").exists():
                infos.append(info)
                continue
        logging.info("[%d/%d] %s (%s)", i, len(acq), run_id, ", ".join(avail) or "no variants")
        info = process_run(out, row, avail)
        if "bundle_error" in info:
            logging.error("%s: no npz bundle: %s", run_id, info["bundle_error"])
        f.write_text(json.dumps(info))
        infos.append(info)

    praise_json = load_praise_json()
    table = add_aif_offsets(build_acquisition_table(acq, infos, praise_json), infos)
    table = table.sort_values(["dataset", "patient_id", "injection_number", "plane", "run_id"])
    table.to_csv(out / "acquisition_table.csv", index=False)
    table[table.is_duplicate][["run_id", "dataset", "patient_id", "series_number", "plane", "acquisition_start",
                               "duplicate_of", "duplicate_type", "duplicate_scope", "source"]] \
        .to_csv(out / "duplicates.csv", index=False)

    perf, errors = build_perfusion_and_errors(out)
    perf.to_csv(out / "perfusion_summary.csv", index=False)
    errors.to_csv(out / "errors.csv", index=False)
    run_ids = table.run_id.tolist()
    build_aif_curves(out, run_ids, infos, dict(zip(table.run_id, table.aif_arrival_s_ica_top))).to_csv(out / "aif_curves.csv", index=False)
    overlap = build_mask_overlap(out, run_ids)
    overlap.to_csv(out / "mask_overlap.csv", index=False)
    logging.info("Wrote tables and %d npz bundles to %s", len(list((out / "npz").glob("*.npz"))), out)


if __name__ == "__main__":
    main()
