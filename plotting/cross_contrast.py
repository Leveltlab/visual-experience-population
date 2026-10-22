import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis_config import AREAS, COLORS, LABELS
from plotting.panels_bcd import MARKERS, OFFSETS
from plotting.style import use_myriad_pro


def render(groups, validation):
    use_myriad_pro()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.7), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.77, bottom=0.24, wspace=0.20)
    for ax, area in zip(axes, AREAS):
        for fi, color in enumerate(COLORS):
            sub = groups[(groups.area == area) & (groups.fi == fi)].sort_values("pair")
            mean = sub["mean"].to_numpy()
            err = np.maximum(0, np.array([mean - sub.ci_low, sub.ci_high - mean]))
            ax.errorbar(
                np.arange(4) + OFFSETS[fi] * 0.08,
                mean,
                yerr=err,
                color=color,
                marker=MARKERS[fi],
                ms=5.8,
                lw=1.2,
                capsize=2,
            )
        ax.set(
            xticks=np.arange(4),
            xticklabels=["1 +\n1.93", "3.72 +\n7.19", "13.89 +\n26.82", "51.79 + 100\n(held out)"],
            xlim=(-0.25, 3.3),
            title=area,
            xlabel="Target contrast pool (%)",
            ylim=(-0.2, 1.02),
            yticks=[0, 0.25, 0.5, 0.75, 1],
        )
        ax.tick_params(labelsize=8)
        ax.axhline(0, color=".65", ls=":", lw=0.7)
    axes[0].set_ylabel("Same-direction cosine")
    handles = [
        Line2D([], [], color=c, marker=m, ms=5.8, lw=1, label=l)
        for c, l, m in zip(COLORS, LABELS, MARKERS)
    ]
    fig.legend(
        handles=handles,
        ncol=4,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.54, 0.99),
        fontsize=9,
    )
    fig.text(
        0.54,
        0.87,
        "Reference: random 15 trials from 51.79% + 100% · N = 10 · 1,000 draws",
        ha="center",
        fontsize=9,
        color=".35",
    )
    fig.text(
        0.54,
        0.025,
        "Target: 30 trials at lower contrasts; 15 disjoint trials at highest contrasts. Error bars: 95% mouse-bootstrap intervals.",
        ha="center",
        fontsize=8,
        color=".35",
    )
    return fig
