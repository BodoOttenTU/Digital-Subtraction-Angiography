"""ICA-top and ICA-full mask on the same frontal run, side by side and in sync, for the sampling-segment slide.

    python scripts/presentation_segment_gif.py

PRAISE048 plane A (same run as sq1_method.png). Chosen from the 46 frontal runs with both AIFs usable:
ICA-full in one piece (PRAISE054's was in 32), and a peak and arrival difference close to the frontal median. Both panels show exactly the same
frames; only the outlined mask differs: ICA-top (scarlet) left, ICA-full = grown mask (cyan) right, the
colours of the two curves in sq1_method.png. Each mask is tinted and outlined. Both panels are zoomed on
the same square around the two masks, so the sampled segment is visible. One fixed grey window for the run.
Same canvas layout as views_ap_lateral.gif (slide 13), so the two slides match.
Written to results/presentation/segment_top_full.gif (real time, looping).
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

RUN = "PRAISE_PRAISE048_01_S08_A"  # one clean ICA-full piece; peak +11 %, arrival -0.19 s (frontal median +7 %, -0.14 s)
PAD = 50        # px (full resolution) around the two masks
TINT = 0.35     # mask fill opacity
MAX_S = 8.0     # same time span as the x-axis of sq1_method.png
MASKS = [("mask_ica_top", "ICA-top", (200, 25, 25)), ("mask_ica_full_grown", "ICA-full", (0, 146, 181))]
OUT = RESULTS / "presentation" / "segment_top_full.gif"


def square_crop(union, shape):
    ys, xs = np.nonzero(union)
    side = max(ys.max() - ys.min(), xs.max() - xs.min()) + 2 * PAD
    side = min(side, *shape)
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    y0 = int(np.clip(cy - side // 2, 0, shape[0] - side))
    x0 = int(np.clip(cx - side // 2, 0, shape[1] - side))
    return slice(y0, y0 + side), slice(x0, x0 + side)


def to_panel(mask):
    m = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize((PANEL, PANEL), Image.NEAREST)) > 127
    return m, ndi.binary_dilation(m ^ ndi.binary_erosion(m), iterations=1)


def main():
    acq = pd.read_csv(DATA / "acquisition_table.csv").set_index("run_id")
    frames = load_frames(acq.loc[RUN, "source"]).astype(np.float32)
    d = np.load(DATA / "npz" / f"{RUN}.npz")
    masks = {k: d[k].astype(bool) for k, _, _ in MASKS}
    cy, cx = square_crop(masks["mask_ica_top"] | masks["mask_ica_full_grown"], frames.shape[:2])
    n_keep = int(MAX_S * 1000 / FRAME_INTERVAL_MS) + 1
    stack = frames[cy, cx, :n_keep]
    fin = stack[np.isfinite(stack) & (stack > 5)]
    lo, hi = np.percentile(fin, 1), np.percentile(fin, 99.5)
    panels = {k: to_panel(m[cy, cx]) for k, m in masks.items()}

    W, H = 2 * PANEL + GAP, HEAD + PANEL
    f_lab, f_t = font(26), font(22)
    imgs = []
    for k in range(stack.shape[2]):
        f = np.nan_to_num(stack[:, :, k], nan=hi)
        u8 = (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        grey = np.repeat(np.array(Image.fromarray(u8, "L").resize((PANEL, PANEL), Image.LANCZOS))[..., None], 3, 2)
        canvas = Image.new("RGB", (W, H), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for j, (key, label, col) in enumerate(MASKS):
            fill, edge = panels[key]
            rgb = grey.astype(np.float32)
            rgb[fill] = (1 - TINT) * rgb[fill] + TINT * np.array(col)
            rgb[edge] = col
            x0 = j * (PANEL + GAP)
            canvas.paste(Image.fromarray(rgb.astype(np.uint8)), (x0, HEAD))
            dr.text((x0 + 4, HEAD // 2), label, fill=col, font=f_lab, anchor="lm")
        dr.text((W - 10, H - 10), f"{k * FRAME_INTERVAL_MS / 1000:.2f} s", fill=(255, 255, 255), font=f_t,
                anchor="rb", stroke_width=2, stroke_fill=(0, 0, 0))
        imgs.append(canvas.quantize(colors=255, method=Image.MEDIANCUT, dither=Image.NONE))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=FRAME_INTERVAL_MS, loop=0, optimize=True)

    # numeric checks instead of viewing the patient images
    print("run", RUN, "crop rows", (cy.start, cy.stop), "cols", (cx.start, cx.stop), f"window {lo:.1f}-{hi:.1f}")
    for key, label, _ in MASKS:
        full = masks[key]
        fill, edge = panels[key]
        print(f"{label}: {int(full.sum())} px, inside crop {int(full[cy, cx].sum())} px, on panel {int(fill.sum())} px "
              f"({fill.mean():.1%} of panel), outline {int(edge.sum())} px")
    top, grown = masks["mask_ica_top"], masks["mask_ica_full_grown"]
    print(f"overlap: {int((top & grown).sum())} px of ICA-top inside ICA-full ({(top & grown).sum() / top.sum():.0%})")
    with Image.open(OUT) as chk:
        print("gif", chk.size, "n_frames", chk.n_frames, "ms/frame", chk.info.get("duration"),
              f"{OUT.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
