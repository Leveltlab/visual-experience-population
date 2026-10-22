import numpy as np
from analysis_config import COLORS
import matplotlib

matplotlib.use("Agg")
MARKERS = ["o", "s", "o", "s"]
OFFSETS = np.array([-1.5, -0.5, 0.5, 1.5])


def draw_panel(ax, area, mouse, groups, full_circle=False, title=True, individual=True, dodge=0.10):
    if full_circle:
        angle = np.linspace(45, 315, 541)
        for harmonic, color in [(2, ".55"), (1, "black")]:
            ideal = np.cos(harmonic * np.deg2rad(angle))
            ax.fill_between(
                angle,
                np.maximum(-1, ideal - 0.035),
                np.minimum(1, ideal + 0.035),
                color=color,
                alpha=0.3,
                linewidth=0,
                zorder=0,
            )
            ax.plot(angle, ideal, color=color, alpha=0.5, lw=0.7, zorder=0)
    else:
        # Separate guides at categorical endpoints; no intermediate angles implied.
        for harmonic, color in [(2, ".55"), (1, "black")]:
            for x, angle in enumerate([180, 90]):
                ideal = np.cos(harmonic * np.deg2rad(angle))
                ax.fill_between(
                    [x - 0.22, x + 0.22],
                    max(-1, ideal - 0.035),
                    min(1, ideal + 0.035),
                    color=color,
                    alpha=0.3,
                    linewidth=0,
                    zorder=0,
                )
                ax.plot(
                    [x - 0.22, x + 0.22], [ideal, ideal], color=color, alpha=0.5, lw=0.7, zorder=0
                )
    for fi, color in enumerate(COLORS):
        sub = mouse[(mouse.area == area) & (mouse.fi == fi)]
        values = sub.pivot(index="mouse", columns="difference_degrees", values="cosine")
        g = groups[(groups.area == area) & (groups.fi == fi)].set_index("difference_degrees")
        if full_circle:
            order = ANGLES[1:]
            x = order + OFFSETS[fi] * 6.0  # Symmetric -9, -3, +3, +9 degree offsets.
            for row in values[order].to_numpy():
                ax.plot(
                    x,
                    row,
                    linestyle="none",
                    color=color,
                    alpha=0.14,
                    marker=MARKERS[fi],
                    ms=2.3,
                    zorder=1,
                )
            mean = g.loc[order, "mean"].to_numpy()
            err = np.maximum(
                0, np.array([mean - g.loc[order, "ci_low"], g.loc[order, "ci_high"] - mean])
            )
            ax.errorbar(
                x,
                mean,
                yerr=err,
                color=color,
                fmt=MARKERS[fi] + "-",
                ms=4.3,
                lw=1.25,
                capsize=2,
                elinewidth=1.2,
                zorder=3,
            )
        else:
            x, order = np.array([0, 1]) + OFFSETS[fi] * dodge, [180, 90]
            for row in values[order].to_numpy() if individual else []:
                ax.plot(x, row, lw=0.55, color=color, alpha=0.17, marker=MARKERS[fi], ms=2.5)
            mean = g.loc[order, "mean"].to_numpy()
            err = np.maximum(
                0, np.array([mean - g.loc[order, "ci_low"], g.loc[order, "ci_high"] - mean])
            )
            ax.errorbar(
                x, mean, yerr=err, color=color, marker=MARKERS[fi], ms=6.2, capsize=2, lw=1.3
            )
    ax.axhline(0, color=".65", lw=0.7, ls=":")
    ax.set(ylim=(-1.05, 1.05), yticks=[-1, -0.5, 0, 0.5, 1], title=area if title else "")
    if full_circle:
        ax.set(xlim=(30, 330), xticks=ANGLES[1:], xlabel="Direction difference Δ (°)")
    else:
        ax.set(xlim=(-0.3, 1.3), xticks=[0, 1], xticklabels=["Opposite (180°)", "Orthogonal (90°)"])
    ax.tick_params(labelsize=8)


def decoding_panel(ax, area, groups, small=False):
    for fi, color in enumerate(COLORS):
        for ti, task in enumerate(["orientation", "direction"]):
            g = groups[(groups.area == area) & (groups.fi == fi) & (groups.task == task)].iloc[0]
            x = fi + (-0.19 if ti == 0 else 0.19)
            ax.bar(
                x,
                g["mean"] * 100,
                width=0.34,
                color=color if ti == 0 else lighter(color),
                edgecolor=color,
                hatch="" if ti == 0 else "///",
                linewidth=0.7,
                zorder=2,
            )
            ax.errorbar(
                x,
                g["mean"] * 100,
                yerr=np.maximum(
                    0, [[100 * (g["mean"] - g.ci_low)], [100 * (g.ci_high - g["mean"])]]
                ),
                fmt="none",
                ecolor=".18",
                capsize=2,
                lw=0.8,
                zorder=3,
            )
    ax.axhline(25, color=".25", ls="--", lw=0.9, zorder=4)
    ax.axhline(12.5, color=".5", ls=":", lw=1, zorder=4)
    ax.set(
        ylim=(0, 80),
        yticks=[0, 25, 50, 75],
        xticks=np.arange(4),
        xticklabels=["DR\nbase", "NR\nbase", "DR\nchron", "NR\nchron"],
        title=area,
    )
    ax.tick_params(labelsize=7 if small else 8)
    ax.spines[["top", "right"]].set_visible(False)


ANGLES = np.arange(8) * 45


def lighter(c):
    from matplotlib.colors import to_rgb

    return tuple(0.5 + 0.5 * np.array(to_rgb(c)))
