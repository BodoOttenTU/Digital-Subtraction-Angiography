"""Frontal (AP) and lateral view of one injection, side by side and in sync, for the projection-view slide.

    python scripts/presentation_views_gif.py

PRAISE054 (same injection as the pipeline slide). Both planes are resampled to the same 250 ms grid and
start together (AIF arrival differs 23 ms), so frame k of A and B is the same moment of the same bolus.
The ICA-top mask is outlined in scarlet in both views. Each view has one fixed grey window for the
whole run (its own 1-99.5 percentile), so the bolus passage is visible in both.
Written to results/presentation/views_ap_lateral.gif (played in real time, looping).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_truncation import FRAME_INTERVAL_MS, load_frames  # noqa: E402
from paths import DATA, RESULTS  # noqa: E402

EXAMPLE = "PRAISE054_01_I01"
PANEL = 360          # px per view (square); 3.35 in wide on the slide
GAP, HEAD = 12, 56   # px between panels, label row on top
SCARLET = (200, 25, 25)
LABEL = {"A": "A  frontal (AP)", "B": "B  lateral"}
OUT = RESULTS / "presentation" / "views_ap_lateral.gif"


def font(size):
    try:
        return ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        return ImageFont.load_default()


def view(run):
    frames = load_frames(ACQ.loc[run, "source"]).astype(np.float32)
    mask = np.load(DATA / "npz" / f"{run}.npz")["mask_ica_top"].astype(bool)
    fin = frames[np.isfinite(frames) & (frames > 5)]  # leave out the black border outside the field of view
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99.5)
    m = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((PANEL, PANEL), Image.NEAREST)) > 127
    edge = ndi.binary_dilation(m ^ ndi.binary_erosion(m), iterations=1)
    return frames, lo, hi, edge, int(mask.sum())


def main():
    global ACQ
    ACQ = pd.read_csv(DATA / "acquisition_table.csv").set_index("run_id")
    g = ACQ[ACQ.ab_pair_id == EXAMPLE]
    runs = {p: g[g.plane == p].index[0] for p in ("A", "B")}
    views = {p: view(r) for p, r in runs.items()}
    n = min(v[0].shape[2] for v in views.values())
    W, H = 2 * PANEL + GAP, HEAD + PANEL
    f_lab, f_t = font(26), font(22)

    imgs = []
    for k in range(n):
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(canvas)
        for j, p in enumerate(("A", "B")):
            frames, lo, hi, edge, _ = views[p]
            f = np.nan_to_num(frames[:, :, k], nan=hi)
            u8 = (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
            rgb = np.repeat(np.array(Image.fromarray(u8, "L").resize((PANEL, PANEL), Image.LANCZOS))[..., None], 3, 2)
            rgb[edge] = SCARLET
            x0 = j * (PANEL + GAP)
            canvas.paste(Image.fromarray(rgb), (x0, HEAD))
            d.text((x0 + 4, HEAD // 2), LABEL[p], fill=(16, 16, 115), font=f_lab, anchor="lm")
        d.text((W - 10, H - 10), f"{k * FRAME_INTERVAL_MS / 1000:.2f} s", fill=(255, 255, 255), font=f_t,
               anchor="rb", stroke_width=2, stroke_fill=(0, 0, 0))
        imgs.append(canvas.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=FRAME_INTERVAL_MS, loop=0, optimize=True)

    # numeric checks instead of viewing the patient images
    for p, r in runs.items():
        frames, lo, hi, edge, npx = views[p]
        print(p, r, "frames", frames.shape, f"window {lo:.1f}-{hi:.1f}", "mask px", npx,
              "outline px on panel", int(edge.sum()), "AIF arrival", f"{ACQ.loc[r, 'aif_arrival_s_ica_top']:.3f} s")
    with Image.open(OUT) as chk:
        print("gif", chk.size, "n_frames", chk.n_frames, "ms/frame", chk.info.get("duration"),
              f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
