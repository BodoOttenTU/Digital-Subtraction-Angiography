"""Three patients side by side, frontal DSA with the ICA-top mask, on one shared clock, for the patient slide.

    python scripts/presentation_patients_gif.py

Same three patients and colours as sq3_method.png (patient 1 = PRAISE063, early bolus; 2 = PRAISE054;
3 = PRAISE058, late bolus). Frame k of every panel is k x 250 ms after the first frame of that run, so
the different arrival times are visible. A run shorter than MAX_S holds its last frame. Each panel is a
square zoom around its ICA-top mask (tinted and outlined in the patient colour), with its own fixed grey
window (from the whole frame) and a thin white rim around the outline. Canvas style as views_ap_lateral.gif and segment_top_full.gif (slides 11 and 13).
Written to results/presentation/patients_ica_top.gif (real time, looping).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_truncation import FRAME_INTERVAL_MS, load_frames  # noqa: E402
from paths import DATA, RESULTS  # noqa: E402
from presentation_views_gif import GAP, HEAD, PANEL, font  # noqa: E402

PATIENTS = [("PRAISE_PRAISE063_01_S08_A", "patient 1", (0, 146, 181)),
            ("PRAISE_BATCH_2_PRAISE054_01_S11_A", "patient 2", (16, 16, 115)),
            ("PRAISE_BATCH_2_PRAISE058_01_S08_A", "patient 3", (200, 25, 25))]
SIDE = 440      # px (full resolution) of the square zoom around ICA-top
TINT = 0.35
MAX_S = 8.0     # x-axis of sq3_method.png
MIN_RANGE = 90  # minimum grey-window width (raw units); the other two runs span about 90-115
OUT = RESULTS / "presentation" / "patients_ica_top.gif"


def crop_around(mask, shape):
    ys, xs = np.nonzero(mask)
    cy, cx = int(ys.mean()), int(xs.mean())
    y0 = int(np.clip(cy - SIDE // 2, 0, shape[0] - SIDE))
    x0 = int(np.clip(cx - SIDE // 2, 0, shape[1] - SIDE))
    return slice(y0, y0 + SIDE), slice(x0, x0 + SIDE)


def panel_view(run, acq):
    frames = load_frames(acq.loc[run, "source"]).astype(np.float32)
    mask = np.load(DATA / "npz" / f"{run}.npz")["mask_ica_top"].astype(bool)
    cy, cx = crop_around(mask, frames.shape[:2])
    stack = frames[cy, cx, :]
    fin = frames[np.isfinite(frames) & (frames > 5)]  # window from the whole frame, not the zoom
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99.5)
    lo = min(lo, hi - MIN_RANGE)   # a low-contrast run would otherwise get a narrow, noisy window
    m = np.array(Image.fromarray(mask[cy, cx].astype(np.uint8) * 255).resize((PANEL, PANEL), Image.NEAREST)) > 127
    edge = ndi.binary_dilation(m ^ ndi.binary_erosion(m), iterations=1)
    halo = ndi.binary_dilation(edge, iterations=1) & ~edge   # white rim so a dark outline shows on a dark vessel
    return dict(stack=stack, lo=lo, hi=hi, fill=m, edge=edge, halo=halo, crop=(cy.start, cx.start), mask_px=int(mask.sum()))


def main():
    acq = pd.read_csv(DATA / "acquisition_table.csv").set_index("run_id")
    views = [panel_view(r, acq) for r, _, _ in PATIENTS]
    n = int(MAX_S * 1000 / FRAME_INTERVAL_MS) + 1
    W, H = 3 * PANEL + 2 * GAP, HEAD + PANEL
    f_lab, f_t = font(40), font(30)   # larger than slides 11/13: this GIF is shown smaller per panel

    imgs = []
    for k in range(n):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for j, ((run, label, col), v) in enumerate(zip(PATIENTS, views)):
            kk = min(k, v["stack"].shape[2] - 1)          # hold the last frame of a shorter run
            f = np.nan_to_num(v["stack"][:, :, kk], nan=v["hi"])
            u8 = (np.clip((f - v["lo"]) / (v["hi"] - v["lo"]), 0, 1) * 255).astype(np.uint8)
            rgb = np.repeat(np.array(Image.fromarray(u8, "L").resize((PANEL, PANEL), Image.LANCZOS))[..., None],
                            3, 2).astype(np.float32)
            rgb[v["fill"]] = (1 - TINT) * rgb[v["fill"]] + TINT * np.array(col)
            rgb[v["halo"]] = 255
            rgb[v["edge"]] = col
            x0 = j * (PANEL + GAP)
            canvas.paste(Image.fromarray(rgb.astype(np.uint8)), (x0, HEAD))
            dr.text((x0 + 4, HEAD // 2), label, fill=col, font=f_lab, anchor="lm")
        dr.text((W - 10, H - 10), f"{k * FRAME_INTERVAL_MS / 1000:.2f} s", fill=(255, 255, 255), font=f_t,
                anchor="rb", stroke_width=2, stroke_fill=(0, 0, 0))
        imgs.append(canvas.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=FRAME_INTERVAL_MS, loop=0, optimize=True)

    for (run, label, _), v in zip(PATIENTS, views):
        print(f"{label}: {run} frames {v['stack'].shape[2]} (held after {(v['stack'].shape[2] - 1) * FRAME_INTERVAL_MS / 1000:.2f} s)"
              f" crop at {v['crop']} window {v['lo']:.1f}-{v['hi']:.1f} ICA-top {v['mask_px']} px,"
              f" {v['fill'].mean():.1%} of panel; arrival {acq.loc[run, 'aif_arrival_s_ica_top']:.2f} s")
    with Image.open(OUT) as chk:
        print("gif", chk.size, "n_frames", chk.n_frames, "ms/frame", chk.info.get("duration"),
              f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
