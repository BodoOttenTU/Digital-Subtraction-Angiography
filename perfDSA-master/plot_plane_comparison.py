"""
One figure comparing plane A (frontal) and plane B (lateral) of the same injection: AIF success rate,
AIF characteristics, and what the projection does to the deconvolution-based perfusion parameters.

    (a) % usable AIF per plane, ICA full and ICA top
    (b) AIF features: paired difference B vs A (median and IQR over injections), % of plane A
    (c) perfusion map medians (CBV, CBF, MTT): paired difference B vs A, %; Tmax in seconds
    (d) AIF AUC ratio B/A against CBF ratio B/A per injection (log scale)
    (e) same against CBV ratio
    (f) agreement between planes: ICC(A,1) of each perfusion parameter (log values; Tmax linear)

Only injections where the AIF is usable in both planes (same variant) are paired.
Colour = plane (a); filled / open marker = ICA full / ICA top (b-f).

Input : results/aif_analysis/aif_features.csv (analyze_aif_features.py) and the perfusion_summary.csv
        of results/praise and results/praise_batch_2.
Output: results/aif_analysis/plane_comparison.png, plane_comparison.csv

Example:
    python plot_plane_comparison.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "aif_analysis"

# Reference data-viz palette: categorical slots 1-2, neutral inks, light chart surface
PLANE_COL = {"A": "#2a78d6", "B": "#eb6834"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
VARIANTS = {"ica_full": ("ICA full", "o", True), "ica_top": ("ICA top", "o", False)}  # label, marker, filled

AIF_FEATURES = [("peak", "Peak height"), ("auc", "AUC (first pass)"), ("fwhm_s", "FWHM"),
                ("width_s", "Width (AUC / peak)"), ("rise_s", "Rise time"), ("arrival_s", "Arrival time"),
                ("ttp_s", "Time to peak")]
PERF = [("CBV", "CBV"), ("CBF", "CBF"), ("MTT", "MTT")]


def load():
    f = pd.read_csv(OUT / "aif_features.csv")
    p = pd.concat([pd.read_csv(RES / d / "perfusion_summary.csv") for d in ["praise", "praise_batch_2"]])
    cols = [f"{k}_median" for k in ["CBV", "CBF", "MTT", "Tmax"]]
    p = p[["run_id", "variant"] + cols].rename(columns=lambda c: c.replace("_median", ""))
    return f.merge(p, on=["run_id", "variant"], how="left")


def pairs(df, variant, cols):
    u = df[df.usable & (df.variant == variant)]
    w = u.pivot_table(index="injection_id", columns="plane", values=cols)
    return w.dropna(subset=[(cols[0], "A"), (cols[0], "B")])


def summary(a, b, relative=True):
    """Paired B vs A: per-pair differences (% of A, or absolute), median, IQR, Wilcoxon p."""
    d = 100 * (b - a) / a if relative else b - a
    d = d[np.isfinite(d)]
    p = wilcoxon(a, b).pvalue if np.count_nonzero(b - a) >= 5 else np.nan
    return {"n": len(d), "median": np.median(d), "q1": np.percentile(d, 25), "q3": np.percentile(d, 75), "p": p}


def icc_a1(a, b):
    """ICC(A,1): two-way random effects, absolute agreement, single measurement (McGraw & Wong)."""
    x = np.column_stack([a, b])
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(axis=1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(axis=0) - gm) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def stars(p):
    return "" if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8.5)
    ax.title.set_color(INK)


def marker_kw(variant, color=INK2, size=7):
    _, m, filled = VARIANTS[variant]
    return dict(marker=m, ms=size, mec=color, mew=1.6, mfc=color if filled else SURFACE, ls="none")


def forest(ax, rows, xlabel, zero=0.0):
    """rows: list of (label, {variant: summary})."""
    offs = {"ica_full": -0.17, "ica_top": 0.17}
    for i, (label, res) in enumerate(rows):
        for v, s in res.items():
            y = i + offs[v]
            ax.plot([s["q1"], s["q3"]], [y, y], color=INK2, lw=2, solid_capstyle="round")
            ax.plot(s["median"], y, **marker_kw(v))
            if stars(s["p_holm"]):
                ax.annotate(stars(s["p_holm"]), (s["q3"], y), xytext=(4, -3.5), textcoords="offset points",
                            fontsize=8, color=INK2)
    ax.axvline(zero, color=MUTED, lw=1, zorder=0)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlabel(xlabel, fontsize=9, color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def holm(results):
    flat = [s for _, res in results for s in res.values()]
    ps = np.array([s["p"] for s in flat])
    ok = np.isfinite(ps)
    adj = np.full_like(ps, np.nan)
    adj[ok] = multipletests(ps[ok], method="holm")[1]
    for s, q in zip(flat, adj):
        s["p_holm"] = q


def main():
    df = load()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11, "axes.titleweight": "bold",
                         "axes.titlelocation": "left"})
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5), layout="constrained", facecolor=SURFACE)
    for ax in axs.flat:
        style(ax)
    table = []

    # (a) success rate
    ax = axs[0, 0]
    for j, v in enumerate(VARIANTS):
        for k, pl in enumerate("AB"):
            g = df[(df.variant == v) & (df.plane == pl)]
            pct = 100 * g.usable.mean()
            x = j + (k - 0.5) * 0.36
            ax.bar(x, pct, width=0.34, color=PLANE_COL[pl], edgecolor=SURFACE, linewidth=2)
            ax.text(x, pct + 1.2, f"{pct:.0f}%\n{g.usable.sum()}/{len(g)}", ha="center", va="bottom",
                    fontsize=8.5, color=INK2)
            table.append({"panel": "a", "variant": v, "plane": pl, "metric": "pct_usable", "value": pct,
                          "n": len(g)})
    ax.set_xticks([0, 1], [VARIANTS[v][0] for v in VARIANTS])
    ax.set(ylim=(0, 112), ylabel="Usable AIF (%)")
    ax.yaxis.label.set_color(INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=PLANE_COL[p]) for p in "AB"],
              labels=["Plane A (frontal)", "Plane B (lateral)"], frameon=False, fontsize=8.5,
              loc="upper center", ncol=2, labelcolor=INK2)
    ax.set_title("(a) AIF success rate per plane")

    # (b) AIF features
    wa = {v: pairs(df, v, [f for f, _ in AIF_FEATURES]) for v in VARIANTS}
    rows_b = [(lab, {v: summary(wa[v][(f, "A")].values, wa[v][(f, "B")].values) for v in VARIANTS})
              for f, lab in AIF_FEATURES]
    holm(rows_b)
    forest(axs[0, 1], rows_b, "Plane B vs plane A, paired difference (% of A)")
    n_b = {v: len(wa[v]) for v in VARIANTS}
    axs[0, 1].set_title(f"(b) AIF characteristics, B vs A  (n = {n_b['ica_full']} / {n_b['ica_top']} injections)")

    # (c) perfusion parameters: % for CBV/CBF/MTT, seconds for Tmax (inset below)
    wp = {v: pairs(df, v, ["CBV", "CBF", "MTT", "Tmax", "auc"]) for v in VARIANTS}
    rows_c = [(lab, {v: summary(wp[v][(k, "A")].values, wp[v][(k, "B")].values) for v in VARIANTS})
              for k, lab in PERF]
    rows_t = [("Tmax", {v: summary(wp[v][("Tmax", "A")].values, wp[v][("Tmax", "B")].values, relative=False)
                        for v in VARIANTS})]
    holm(rows_c + rows_t)
    ax = axs[0, 2]
    forest(ax, rows_c, "Plane B vs plane A, paired difference (% of A)")
    ax.set_title("(c) Perfusion map medians, B vs A")
    tax = ax.inset_axes([0.0, -0.36, 1.0, 0.14])
    style(tax)
    forest(tax, rows_t, "Tmax: plane B - plane A (s)")
    for panel, rows in [("b", rows_b), ("c", rows_c + rows_t)]:
        for lab, res in rows:
            for v, s in res.items():
                table.append({"panel": panel, "variant": v, "metric": lab,
                              **{k: s[k] for k in ["n", "median", "q1", "q3", "p", "p_holm"]}})

    # (d, e) does the AIF difference propagate into the maps?
    for ax, (key, lab) in zip([axs[1, 0], axs[1, 1]], [("CBF", "CBF"), ("CBV", "CBV")]):
        lim = 0
        for v in VARIANTS:
            w = wp[v]
            x = np.log2(w[("auc", "B")] / w[("auc", "A")])
            y = np.log2(w[(key, "B")] / w[(key, "A")])
            ax.plot(x, y, **marker_kw(v, size=6), alpha=0.85, label=VARIANTS[v][0])
            lim = max(lim, np.nanmax(np.abs(np.r_[x, y])))
            rho = pd.Series(x).corr(pd.Series(y), method="spearman")
            table.append({"panel": "d" if key == "CBF" else "e", "variant": v, "metric": f"spearman AIF-AUC vs {key}",
                          "value": rho, "n": len(w)})
        lim = np.ceil(lim) + 0.3
        g = np.array([-lim, lim])
        ax.plot(g, -g, color=MUTED, lw=1.2, ls="--")
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(0, color=MUTED, lw=1)
        ax.text(lim * 0.9, -lim * 0.9 + 0.15, "map scales with 1 / AIF", rotation=-45, rotation_mode="anchor",
                ha="right", va="bottom", fontsize=8, color=INK2)
        ax.text(-lim * 0.95, 0.12, "no plane effect", fontsize=8, color=INK2, va="bottom")
        ticks = np.arange(-np.floor(lim), np.floor(lim) + 1, 2 if lim > 4 else 1)
        labels = [f"{2 ** t:g}" if t >= 0 else f"1/{2 ** -t:g}" for t in ticks]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.set(xlim=g, ylim=g, aspect="equal")
        ax.set_xlabel("AIF AUC ratio, plane B / plane A", fontsize=9, color=INK2)
        ax.set_ylabel(f"{lab} median ratio, plane B / plane A", fontsize=9, color=INK2)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK2)
        ax.set_title(f"({'d' if key == 'CBF' else 'e'}) Does the AIF difference reach {lab}?")

    # (f) agreement between planes
    ax = axs[1, 2]
    params = ["CBV", "CBF", "MTT", "Tmax"]
    for v in VARIANTS:
        for i, k in enumerate(params):
            a, b = wp[v][(k, "A")].values, wp[v][(k, "B")].values
            if k != "Tmax":
                a, b = np.log(a), np.log(b)
            icc = icc_a1(a, b)
            y = i + (-0.17 if v == "ica_full" else 0.17)
            ax.plot([0, icc], [y, y], color=GRID, lw=2, solid_capstyle="round")
            ax.plot(icc, y, **marker_kw(v))
            ax.annotate(f"{icc:.2f}", (icc, y), xytext=(7 if icc >= 0 else -7, -3.5), textcoords="offset points",
                        fontsize=8, color=INK2, ha="left" if icc >= 0 else "right")
            table.append({"panel": "f", "variant": v, "metric": f"ICC(A,1) {k}", "value": icc, "n": len(a)})
    for x0 in [0.5, 0.75, 0.9]:
        ax.axvline(x0, color=MUTED, lw=0.8, ls=":")
    for xc, txt in [(0.2, "poor"), (0.625, "moderate"), (0.825, "good"), (0.95, "exc.")]:
        ax.text(xc, len(params) - 0.55, txt, fontsize=7.5, color=MUTED, ha="center", va="bottom")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(params)), params)
    ax.set(xlim=(-0.2, 1.0), ylim=(len(params) - 0.5, -0.5))
    ax.set_xlabel("ICC(A,1) between plane A and plane B (CBV, CBF, MTT on log scale)", fontsize=9, color=INK2)
    ax.legend(handles=[plt.Line2D([], [], **marker_kw(v)) for v in VARIANTS],
              labels=[VARIANTS[v][0] for v in VARIANTS], frameon=False, fontsize=8.5, loc="upper right",
              labelcolor=INK2)
    ax.set_title("(f) Agreement of perfusion parameters between planes")

    # shared marker legend for (b), (c)
    axs[0, 1].legend(handles=[plt.Line2D([], [], **marker_kw(v)) for v in VARIANTS],
                     labels=[f"{VARIANTS[v][0]}  (median, IQR)" for v in VARIANTS], frameon=False, fontsize=8.5,
                     loc="lower right", labelcolor=INK2)
    fig.suptitle("Projection dependence of the AIF and the perfusion parameters: plane A (frontal) vs "
                 "plane B (lateral) of the same injection", fontsize=13, fontweight="bold", color=INK, x=0.01,
                 ha="left")
    fig.text(0.01, -0.01, "Only injections with a usable AIF in both planes are paired.  * / ** / ***: Wilcoxon "
             "signed-rank, Holm-corrected p < 0.05 / 0.01 / 0.001 within each panel.  ICC thresholds after "
             "Koo & Li (2016).\nExtreme map ratios in (d, e) come from runs whose maps are dominated by the "
             "shutter-edge artefact; medians and ICCs are reported without removing them.", fontsize=8.5, color=INK2)
    fig.get_layout_engine().set(h_pad=0.25, hspace=0.12)

    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "plane_comparison.png", dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(table).round(4).to_csv(OUT / "plane_comparison.csv", index=False)
    print(pd.DataFrame(table).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
