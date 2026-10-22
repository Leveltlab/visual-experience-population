import numpy as np
from analysis_config import OUT, AREAS, COLORS
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MARKERS = ["o", "s", "o", "s"]
OFFSETS = np.array([-1.5, -0.5, 0.5, 1.5])
from plotting.statistics import vertical_bracket
from plotting.style import use_myriad_pro

DEST = OUT / "euclidean_panel_B"


def render(groups, tests, factorial):
    use_myriad_pro()
    plt.rcParams.update(
        {"font.size": 4, "axes.titlesize": 6, "axes.labelsize": 5, "axes.linewidth": 0.45}
    )
    fig = plt.figure(figsize=(150 / 72, 100 / 72))
    fig.text(1 / 150, 81 / 100, "B", fontsize=7)
    row = factorial.set_index("factor").loc["level"]
    fig.text(
        78 / 150, 87 / 100, "Direction-difference effect " + row.stars, ha="center", fontsize=4
    )
    handles = [
        Line2D([], [], color=c, marker=m, ms=2.7, lw=0.6, label=l)
        for c, m, l in zip(COLORS, MARKERS, ["DR base", "NR base", "DR chronic", "NR chronic"])
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.98),
        ncol=4,
        frameon=False,
        fontsize=4,
        handlelength=1.3,
        columnspacing=1,
        handletextpad=0.4,
        borderpad=0,
        borderaxespad=0,
        labelspacing=0.25,
    )
    ymax = groups.ci_high.max() * 1.08
    for j, area in enumerate(AREAS):
        ax = fig.add_axes([(18 + j * 33) / 150, 22 / 100, 27 / 150, 52 / 100])
        for fi, color in enumerate(COLORS):
            g = (
                groups[(groups.area == area) & (groups.fi == fi)]
                .set_index("difference_degrees")
                .loc[[180, 90]]
            )
            mean = g["mean"].to_numpy()
            err = np.maximum(0, [mean - g.ci_low, g.ci_high - mean])
            ax.errorbar(
                np.array([0, 1]) + OFFSETS[fi] * 0.15,
                mean,
                yerr=err,
                color=color,
                marker=MARKERS[fi],
                ms=6.2 * 0.46,
                capsize=0.92,
                lw=1.3 * 0.45,
                elinewidth=0.5,
                mew=0.4,
            )
        ax.set(xlim=(-0.3, 1.3), ylim=(0, ymax), xticks=[0, 1], xticklabels=["180°", "90°"])
        ax.set_title(area, fontsize=6, pad=3)
        ax.tick_params(labelsize=4, length=1.6, width=0.45, pad=1)
        for r in tests[(tests.area == area) & (tests.p_holm < 0.05)].itertuples():
            fi = 0 if r.timepoint == "baseline" else 2
            g = groups[(groups.area == area) & (groups.difference_degrees == 180)].set_index("fi")
            vertical_bracket(
                ax, 0.08 if fi == 0 else 0.31, g.loc[fi, "mean"], g.loc[fi + 1, "mean"], r.stars
            )
        if j == 0:
            ax.set_ylabel("Euclidean distance", fontsize=5, labelpad=2)
        else:
            ax.tick_params(labelleft=False)
    fig.text(81 / 150, 10 / 100, "Direction difference (∆ / °)", ha="center", fontsize=5)
    return fig
