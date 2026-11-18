"""Animated DSA series for step 1 ("DSA series") of the pipeline slide.

    python scripts/pipeline_series_gif.py

Same run and crop as 1_frame.png in pipeline_thumbnails.py (PRAISE054 plane A), but every
subtracted frame in order, played in real time (250 ms per frame, the uniform resampling
interval), looping. One fixed grey window for the whole run, so contrast arrival and washout
are visible instead of being normalised away per frame. A small time stamp sits in the corner.
Written to results/presentation/pipeline/1_series.gif at 780 x 630 px (the 1.3 x 1.05 in box at 2x, 300 dpi).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_truncation import FRAME_INTERVAL_MS, load_frames  # noqa: E402
from paths import DATA  # noqa: E402
from pipeline_thumbnails import EXAMPLE, OUT, PLANE, SIZE, full_crop  # noqa: E402

PX = (int(SIZE[0] * 300), int(SIZE[1] * 300))  # 780 x 630, same pixels as the png thumbnails


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acq = pd.read_csv(DATA / "acquisition_table.csv").set_index("run_id")
    g = acq[acq.ab_pair_id == EXAMPLE]
    run = g[g.plane == PLANE].index[0]
    frames = load_frames(acq.loc[run, "source"]).astype(np.float32)
    crop = full_crop(frames.shape[:2], SIZE[0] / SIZE[1])
    stack = frames[crop[0], crop[1], :]

    finite = stack[np.isfinite(stack)]
    lo, hi = np.percentile(finite, 1), np.percentile(finite, 99.5)  # one window for all frames
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except OSError:
        font = ImageFont.load_default()

    imgs = []
    for k in range(stack.shape[2]):
        f = np.nan_to_num(stack[:, :, k], nan=hi)
        u8 = (np.clip((f - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(u8, "L").resize(PX, Image.LANCZOS)
        ImageDraw.Draw(im).text((PX[0] - 16, PX[1] - 14), f"{k * FRAME_INTERVAL_MS / 1000:4.1f} s",
                                fill=255, font=font, anchor="rb", stroke_width=2, stroke_fill=0)
        imgs.append(im)  # stays "L": 256-grey palette, no web-palette dithering

    path = OUT / "1_series.gif"
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=FRAME_INTERVAL_MS, loop=0,
                 optimize=True)

    # numeric checks instead of viewing the patient images
    means = stack.reshape(-1, stack.shape[2]).mean(axis=0)
    print("run", run, "frames", stack.shape, f"window {lo:.1f}-{hi:.1f}",
          f"darkest frame {int(np.argmin(means))} ({np.argmin(means) * FRAME_INTERVAL_MS / 1000:.2f} s)")
    with Image.open(path) as chk:
        print("gif", chk.size, "n_frames", chk.n_frames, "ms/frame", chk.info.get("duration"),
              "loop", chk.info.get("loop"), f"{path.stat().st_size / 1e6:.2f} MB")
    print("written to", path)


if __name__ == "__main__":
    main()
