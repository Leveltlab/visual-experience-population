"""Compact 150 × 100 pt contrasts and full-space D-radius barplots."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from analysis_config import OUT, AREAS, COLORS
from plotting.panels_bcd import MARKERS
from plotting.statistics import horizontal_bracket
from plotting.style import use_myriad_pro


def draw(fig, kind, groups, tests, offset=0):
    w, h = fig.get_size_inches() * 72

    def text(x, y, s, size=5, **kw):
        return fig.text((offset + x) / w, y / h, s, fontsize=size, **kw)

    handles = [
        Line2D([], [], color=c, marker=m, ms=2.7, lw=0.6, label=l)
        for c, m, l in zip(COLORS, MARKERS, ["DR base", "NR base", "DR chronic", "NR chronic"])
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=((offset + 76.5) / w, 98 / h),
        ncol=4,
        frameon=False,
        fontsize=4,
        handlelength=1.1,
        columnspacing=0.7,
        handletextpad=0.3,
        borderaxespad=0,
    )
    text(1, 81, "D" if kind == "radius_bars" else "B", 7)
    text(
        78,
        87,
        {
            "cosine_contrast": "Cosine: C(180°) − C(90°)",
            "euclidean_contrast": "Distance: d(90°) − d(180°)",
            "radius_bars": "Population radius · D insets",
        }[kind],
        4,
        ha="center",
    )
    scale = 1000 if kind == "radius_bars" else 1
    lo = min(0, groups.ci_low.min() * scale)
    hi = groups.ci_high.max() * scale
    ylim = (lo - 0.045 * (hi - lo) if kind != "radius_bars" else 0, hi + 0.20 * (hi - lo))
    for j, area in enumerate(AREAS):
        ax = fig.add_axes([(offset + 18 + j * 33) / w, 22 / h, 27 / w, 52 / h])
        for fi, color in enumerate(COLORS):
            g = groups[(groups.area == area) & (groups.fi == fi)].iloc[0]
            y = g["mean"] * scale
            err = np.maximum(0, [[y - g.ci_low * scale], [g.ci_high * scale - y]])
            if kind == "radius_bars":
                ax.bar(fi, y, width=0.65, color=color, linewidth=0, zorder=2)
                ax.errorbar(fi, y, yerr=err, fmt="none", ecolor=".2", capsize=0.8, lw=0.5, zorder=3)
            else:
                ax.errorbar(
                    fi,
                    y,
                    yerr=err,
                    fmt=MARKERS[fi],
                    color=color,
                    ms=2.85,
                    mew=0.4,
                    capsize=0.9,
                    lw=0.55,
                    zorder=3,
                )
        if kind != "radius_bars":
            ax.axhline(0, color=".65", lw=0.45, ls=":", zorder=0)
        ax.set(xlim=(-0.6, 3.6), ylim=ylim, xticks=[0.5, 2.5], xticklabels=["Base", "Chronic"])
        ax.set_title(area, fontsize=6, pad=3)
        ax.tick_params(labelsize=4, length=1.6, width=0.45, pad=1, labelleft=j == 0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.45)
        for r in tests[(tests.area == area) & (tests.p_holm < 0.05)].itertuples():
            fi = 0 if r.timepoint == "baseline" else 2
            ceiling = (
                groups[(groups.area == area) & groups.fi.isin([fi, fi + 1])].ci_high.max() * scale
            )
            fraction = (ceiling - ylim[0]) / (ylim[1] - ylim[0]) + 0.035
            horizontal_bracket(ax, fi, fi + 1, fraction, r.stars, height=0.025, size=4)
        if j == 0:
            ax.set_ylabel(
                {
                    "cosine_contrast": "Cosine difference",
                    "euclidean_contrast": "Distance difference",
                    "radius_bars": r"Population r² ($\times10^{-3}$)",
                }[kind],
                fontsize=5,
                labelpad=2,
            )
    text(81, 10, "Recording timepoint", 5, ha="center")
    return list(ylim)


def area_radius(area, radius, rtests):
    fig = plt.figure(figsize=(55 / 72, 100 / 72))
    ax = fig.add_axes([18 / 55, 0.22, 30 / 55, 0.52])
    g = radius[radius.area == area].sort_values("fi")
    top = g.ci_high.max() * 1000 * 1.25
    for r in g.itertuples():
        ax.bar(r.fi, r.mean * 1000, width=0.65, color=COLORS[r.fi], linewidth=0, zorder=2)
        ax.errorbar(
            r.fi,
            r.mean * 1000,
            yerr=np.maximum(0, [[1000 * (r.mean - r.ci_low)], [1000 * (r.ci_high - r.mean)]]),
            fmt="none",
            ecolor=".2",
            capsize=0.8,
            lw=0.5,
            zorder=3,
        )
    ax.set(xlim=(-0.6, 3.6), ylim=(0, top), xticks=range(4), xticklabels=["DR", "NR", "DR", "NR"])
    ax.set_title(area, fontsize=6, pad=3)
    ax.set_ylabel(r"Population r² ($\times10^{-3}$)", fontsize=5, labelpad=2)
    ax.tick_params(labelsize=4, length=1.6, width=0.45, pad=1)
    for r in rtests[(rtests.area == area) & (rtests.p_holm < 0.05)].itertuples():
        fi = 0 if r.timepoint == "baseline" else 2
        ceiling = g[g.fi.isin([fi, fi + 1])].ci_high.max() * 1000
        horizontal_bracket(ax, fi, fi + 1, ceiling / top + 0.035, r.stars, height=0.025, size=4)
    for x, label in [(0.5, "Base"), (2.5, "Chronic")]:
        ax.text(
            x, -0.22, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=4
        )
    fig.text(33 / 55, 0.87, "D-inset radius", ha="center", fontsize=4)
    return fig


def render(panel):
    use_myriad_pro()
    plt.rcParams.update(
        {
            "font.size": 4,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.linewidth": 0.45,
            "svg.fonttype": "none",
        }
    )
    if panel.startswith("radius"):
        groups = pd.read_csv(OUT / "direction_cosine_top2/supporting/radius_groups.csv")
        tests = (
            pd.read_csv(OUT / "manuscript_statistics/posthoc_results.csv")
            .query("panel=='D'")
            .rename(columns={"p_adjusted": "p_holm"})
        )
        if panel != "radius_bars":
            area = next(a for a in AREAS if panel == "radius_" + a.replace("-", "_"))
            return area_radius(area, groups, tests)
    else:
        metric = panel.removesuffix("_contrast")
        groups = pd.read_csv(OUT / "orientation_contrasts/endpoint_groups.csv")
        tests = pd.read_csv(OUT / "orientation_contrasts/nr_dr_tests.csv")
        groups = groups[(groups.metric == metric) & (groups.endpoint == "contrast")]
        tests = tests[(tests.metric == metric) & (tests.endpoint == "contrast")]
    fig = plt.figure(figsize=(150 / 72, 100 / 72))
    draw(fig, panel, groups, tests)
    return fig
