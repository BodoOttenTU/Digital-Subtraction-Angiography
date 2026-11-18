"""Five thumbnails for the perfDSA pipeline slide of the final presentation.

    python scripts/pipeline_thumbnails.py

One frontal run (the SQ4 example, PRAISE054 plane A) is used for every step:
  1_frame.png    one subtracted DSA frame at the AIF peak
  2_mask.png     the ICA-top mask on the minIP, zoomed on the ICA
  3_aif.png      the ICA-top AIF over time
  4_tissue.png   mean parenchyma curve against the AIF (the two curves deconvolution relates)
  5_cbv.png      the CBV map from perfDSA with this AIF
Written to results/presentation/pipeline/ at the aspect ratio of the slide boxes (1.3 x 1.05 in).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aif_truncation import FRAME_INTERVAL_MS, load_frames, perfusion, sample_aif  # noqa: E402
from paths import DATA, RESULTS  # noqa: E402
from sq4_aif_swap import parenchyma_roi  # noqa: E402

EXAMPLE = "PRAISE054_01_I01"
PLANE = "A"
OUT = RESULTS / "presentation" / "pipeline"
SIZE = (2.6, 2.1)          # inches, 1.3 x 1.05 box at 2x
SCARLET, NAVY, CYAN = "#c81919", "#101073", "#0092b5"
DT = FRAME_INTERVAL_MS / 1000
SVD_THRESHOLD = 0.2        # relative singular value cut-off for the illustrative residue

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)


def image_panel(img, name, overlay=None, overlay_cmap=None, vmax=None, crop=None):
    fig = plt.figure(figsize=SIZE)
    ax = fig.add_axes([0, 0, 1, 1])
    if crop is not None:
        img = img[crop]
        overlay = None if overlay is None else overlay[crop]
    ax.imshow(img, cmap="gray", vmin=np.percentile(img, 1), vmax=np.percentile(img, 99.5),
              aspect="equal")
    if overlay is not None:
        ax.imshow(overlay, cmap=overlay_cmap, vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    fig.savefig(OUT / name, dpi=300, facecolor="black")
    plt.close(fig)


def box_crop(mask, aspect, pad):
    """Crop around a mask with the slide box aspect (w/h)."""
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    h = max(ys.max() - ys.min(), (xs.max() - xs.min()) / aspect) + 2 * pad
    w = h * aspect
    H, W = mask.shape
    y0, x0 = int(max(0, cy - h / 2)), int(max(0, cx - w / 2))
    return slice(y0, int(min(H, y0 + h))), slice(x0, int(min(W, x0 + w)))


def full_crop(shape, aspect):
    H, W = shape
    w = int(H * aspect)
    if w <= W:
        x0 = (W - w) // 2
        return slice(0, H), slice(x0, x0 + w)
    h = int(W / aspect)
    y0 = (H - h) // 2
    return slice(y0, y0 + h), slice(0, W)


def tsvd_residue(aif, tissue, threshold=SVD_THRESHOLD):
    """k(t) = CBF·R(t) from tissue = dt · A k, with A the lower-triangular AIF matrix (truncated SVD)."""
    n = len(aif)
    A = np.array([[aif[i - j] if i >= j else 0.0 for j in range(n)] for i in range(n)]) * DT
    U, S, Vt = np.linalg.svd(A)
    inv = np.where(S >= threshold * S.max(), 1 / S, 0.0)
    return Vt.T @ (inv * (U.T @ tissue))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    acq = pd.read_csv(DATA / "acquisition_table.csv").set_index("run_id")
    g = acq[acq.ab_pair_id == EXAMPLE]
    run = g[g.plane == PLANE].index[0]
    frames = load_frames(acq.loc[run, "source"])
    d = np.load(DATA / "npz" / f"{run}.npz")
    mask = d["mask_ica_top"].astype(bool)
    aif = sample_aif(frames, mask)
    t = np.arange(len(aif)) * DT
    aspect = SIZE[0] / SIZE[1]

    # 1. frame at the AIF peak
    k_pk = int(np.nanargmax(aif))
    image_panel(frames[:, :, k_pk], "1_frame.png", crop=full_crop(frames.shape[:2], aspect))

    # 2. ICA-top mask on the minIP, zoomed on the ICA
    minip = d["minip"]
    ov = np.ma.masked_where(~mask, np.ones(mask.shape))
    image_panel(minip, "2_mask.png", overlay=ov, overlay_cmap=plt.matplotlib.colors.ListedColormap([SCARLET]),
                vmax=1, crop=box_crop(mask, aspect, pad=60))

    # 3. AIF curve
    base = np.nanmedian(aif[:2])
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    ax.plot(t, aif - base, color=SCARLET, lw=2.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("AIF")
    ax.set_yticks([])
    ax.set_xlim(0, t[-1])
    save(fig, "3_aif.png")

    # 4. mean parenchyma curve and its residue (illustration of the deconvolution)
    roi = parenchyma_roi(frames)
    sig = 1 - frames / 170
    tissue = np.nanmean(sig[roi], axis=0)
    tissue = tissue - np.median(tissue[:2])
    a = aif - base
    k = tsvd_residue(a, tissue)
    # shown: AIF and mean tissue curve, peak-normalised; the noisy residue k is only checked numerically
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    ax.plot(t, a / np.nanmax(a), color=SCARLET, lw=1.8, label="AIF")
    ax.plot(t, tissue / np.nanmax(tissue), color=NAVY, lw=2.5, label="tissue")
    ax.set_xlabel("time (s)")
    ax.set_yticks([])
    ax.set_xlim(0, t[-1])
    ax.set_ylim(-0.05, 1.35)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=2, handlelength=1.2)
    save(fig, "4_tissue.png")

    # 5. CBV map with this AIF
    cbv = perfusion(frames, aif)["CBV"]
    roi_map = roi & (cbv > 0)
    vmax = np.percentile(cbv[roi_map], 95)
    crop = box_crop(roi_map, aspect, pad=10)
    image_panel(minip, "5_cbv.png", overlay=np.ma.masked_where(~roi_map, cbv), overlay_cmap="magma",
                vmax=vmax, crop=crop)

    # numeric checks instead of viewing the patient images
    print("run", run, "frames", frames.shape, "AIF peak frame", k_pk, f"({k_pk * DT:.2f} s)")
    print("mask pixels", int(mask.sum()), "parenchyma roi pixels", int(roi.sum()),
          "roi with CBV>0", int(roi_map.sum()))
    print("tissue peak at", f"{t[int(np.nanargmax(tissue))]:.2f} s", "residue max at",
          f"{t[int(np.nanargmax(k))]:.2f} s", "residue finite", bool(np.isfinite(k).all()))
    print("written to", OUT)


if __name__ == "__main__":
    main()
