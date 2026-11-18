"""Synthetic (fake) AIF curves that illustrate QC failures, in the style of the pipeline slide AIF.

    python scripts/qc_synthetic_examples.py

Not patient data: a gamma-variate bolus with recirculation, sampled at 4 frames/s.
  synthetic_truncated.png    run stops while the curve is still high
  synthetic_no_baseline.png  run starts when the bolus is already arriving
  synthetic_multi_peak.png   two peaks in the first pass
Written to results/presentation/qc_examples/.
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

OUT = Path(__file__).resolve().parents[1] / "results" / "presentation" / "qc_examples"
SIZE = (2.6, 2.1)
SCARLET = "#c81919"
DT = 0.25

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def bolus(t, t0, alpha=3.0, beta=0.35):
    """Gamma variate first pass plus a small, delayed recirculation, peak about 1."""
    def gv(t, t0, a, b):
        s = np.clip(t - t0, 0, None)
        return s ** a * np.exp(-s / b)
    first = gv(t, t0, alpha, beta)
    first /= first.max()
    recirc = 0.12 * gv(t, t0 + 2.2, alpha, 2 * beta)
    recirc /= max(gv(t, t0 + 2.2, alpha, 2 * beta).max(), 1e-9) / 0.12
    return first + recirc


def panel(t, y, name):
    fig, ax = plt.subplots(figsize=SIZE, constrained_layout=True)
    ax.plot(t, y, color=SCARLET, lw=2.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("AIF")
    ax.set_yticks([])
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.05, 1.12)
    fig.savefig(OUT / name, dpi=300, facecolor="white")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)

    # early truncation: normal start, but the run stops at 3.0 s, still at about half of the peak
    t = np.arange(0, 3.0 + DT / 2, DT)
    y = bolus(t, t0=1.0) + rng.normal(0, 0.008, t.size)
    panel(t, y, "synthetic_truncated.png")
    print(f"truncated: last frame {t[-1]:.2f} s at {y[-1]:.2f} of peak")

    # no baseline: the bolus is already rising in the first frame
    t = np.arange(0, 8 + DT / 2, DT)
    y = bolus(t + 1.4, t0=1.0) + rng.normal(0, 0.008, t.size)
    panel(t, y, "synthetic_no_baseline.png")
    print(f"no baseline: first frame at {y[0]:.2f} of peak, second at {y[1]:.2f}")

    # multi-peak: two boluses about 1.5 s apart in the first pass (e.g. an injection pause)
    t = np.arange(0, 8 + DT / 2, DT)
    y = bolus(t, t0=0.9, beta=0.22) + 0.8 * bolus(t, t0=2.4, beta=0.25) + rng.normal(0, 0.008, t.size)
    y /= y.max()
    panel(t, y, "synthetic_multi_peak.png")
    print("multi-peak: written")
    print("written to", OUT)


if __name__ == "__main__":
    main()
