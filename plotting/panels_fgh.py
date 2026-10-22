import numpy as np
import pandas as pd
from analysis_config import OUT, AREAS, LABELS, COLORS
import matplotlib

matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb
from matplotlib.ticker import NullLocator

MARKERS = ["o", "s", "o", "s"]
OFFSETS = np.array([-1.5, -0.5, 0.5, 1.5])
from plotting.statistics import main_label, comparisons, horizontal_bracket

DEST = OUT / "contrast_scaling_pca"
PCA_AREAS = ["AM-PM", "A-AL-RL"]
TRANSFER_YLIM = (20, 65)
CAMERA = None


def canvas(fig):
    w, h = fig.get_size_inches() * 72

    def ax(x, y, ww, hh):
        return fig.add_axes([x / w, y / h, ww / w, hh / h])

    def text(x, y, s, size=5, **kw):
        return fig.text(x / w, y / h, s, fontsize=size, **kw)

    return ax, text, w, h


def quantitative(fig, offset=0):
    axes, text, w, h = canvas(fig)
    peak = pd.read_csv(DEST / "group_peak_radius.csv")
    transfer = pd.read_csv(DEST / "group_reverse_transfer.csv")
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker=m, ms=2.4, lw=0.6, label=l)
            for c, m, l in zip(COLORS, MARKERS, LABELS)
        ],
        loc="center",
        bbox_to_anchor=((offset + 80) / w, 177 / h),
        ncol=2,
        frameon=False,
        fontsize=4.5,
        handlelength=1.15,
        columnspacing=1.0,
        handletextpad=0.4,
        borderpad=0,
    )
    text(offset + 1, 161, "F", 7)
    text(offset + 1, 81, "G", 7)
    text(offset + 82, 166, main_label("F"), 4.5, ha="center")
    text(offset + 82, 81, main_label("G"), 4.5, ha="center")
    low = float(peak.ci_low.min()) * 1000 * 0.65
    high = float(peak.ci_high.max()) * 1000 * 1.5
    assert low > 0
    powers = np.arange(np.ceil(np.log10(low)), np.floor(np.log10(high)) + 1)
    if len(powers) > 4:
        powers = powers[::2]
    audit = []
    for row, data in enumerate([peak, transfer]):
        for j, area in enumerate(AREAS):
            ax = axes(
                offset + (20 if row == 0 else 19) + 32 * j,
                111 if row == 0 else 26,
                27 if row == 0 else 29,
                45 if row == 0 else 43,
            )
            for fi, (color, marker) in enumerate(zip(COLORS, MARKERS)):
                g = data[(data.area == area) & (data.fi == fi)].sort_values("step")
                x = np.arange(len(g)) + (fi - 1.5) * (0.13 if row == 0 else 0.14)
                factor = 1000 if row == 0 else 1
                y, lo, hi = [g[k].to_numpy() * factor for k in ["mean", "ci_low", "ci_high"]]
                ax.errorbar(
                    x,
                    y,
                    yerr=np.maximum(0, [y - lo, hi - y]),
                    color=color,
                    marker=marker,
                    ms=2.0,
                    mew=0.25,
                    lw=0.55,
                    capsize=0.7,
                    zorder=3,
                )
            if row == 0:
                ax.set_yscale("log")
                ax.set(
                    ylim=(low, high),
                    yticks=10.0**powers,
                    xlim=(-0.35, 1.35),
                    xticks=[0, 1],
                    xticklabels=["Low", "High"],
                )
                ax.yaxis.set_minor_locator(NullLocator())
                ax.set_title(area, fontsize=5.3, pad=3)
            else:
                ax.set(
                    ylim=TRANSFER_YLIM,
                    yticks=[25, 40, 55, 65],
                    xlim=(-0.4, 2.4),
                    xticks=[0, 1, 2],
                    xticklabels=[r"$4\!\to\!3$", r"$3\!\to\!2$", r"$2\!\to\!1$"],
                )
                ax.axhline(25, color=".55", ls="--", lw=0.4)
                for tick in ax.get_xticklabels():
                    tick.set_rotation(35)
                    tick.set_ha("right")
                    tick.set_rotation_mode("anchor")
            ax.tick_params(labelsize=4.5, length=1.5, width=0.4, pad=1, labelleft=j == 0)
            ax.spines[["top", "right"]].set_visible(False)
            for spine in ax.spines.values():
                spine.set_linewidth(0.45)
            if j == 0:
                ax.set_ylabel(
                    r"Peak r² ($\times10^{-3}$)" if row == 0 else "Transfer accuracy (%)",
                    fontsize=5,
                    labelpad=2,
                )
            for test in comparisons("F" if row == 0 else "G", area).itertuples():
                fi = 0 if test.timepoint == "baseline" else 2
                step = 2 if row == 0 else int(test.step)
                ymax = data[
                    (data.area == area) & (data.fi.isin([fi, fi + 1])) & (data.step == step)
                ].ci_high.max()
                yy = (
                    (np.log(ymax * 1000) - np.log(low)) / (np.log(high) - np.log(low))
                    if row == 0
                    else (ymax - TRANSFER_YLIM[0]) / (TRANSFER_YLIM[1] - TRANSFER_YLIM[0])
                )
                dodge = 0.13 if row == 0 else 0.14
                horizontal_bracket(
                    ax,
                    step - 1 + (fi - 1.5) * dodge,
                    step - 1 + (fi - 0.5) * dodge,
                    yy + 0.025,
                    test.stars,
                    size=4.5,
                )
            audit.append(
                dict(
                    area=area,
                    panel="F" if row == 0 else "G",
                    x_order=["bottom two", "top two"] if row == 0 else ["4→3", "3→2", "2→1"],
                    ylim=list(ax.get_ylim()),
                    individual_points=0,
                )
            )
    text(offset + 82, 97, "Low: 1–2%   High: 52–100%", 4.7, ha="center")
    text(offset + 82, 9, r"Train bin $\to$ test bin", 5, ha="center")
    return audit


def pca(fig, offset=0):
    axes, text, w, h = canvas(fig)
    with np.load(DEST / "pca_geometry.npz") as z:
        coords = z["coordinates"]
    elev, azim = np.deg2rad([CAMERA["elevation"], CAMERA["azimuth"]])
    basis = np.array(
        [
            [-np.sin(azim), np.cos(azim), 0],
            [-np.sin(elev) * np.cos(azim), -np.sin(elev) * np.sin(azim), np.cos(elev)],
        ]
    )
    xy = coords @ basis.T
    limit = float(abs(xy).max()) * 1.14 * 62 / (57 * 1.30)
    text(offset + 1, 178, "H", 7)
    text(offset + 41, 158, "Baseline", 5.5, ha="center")
    text(offset + 105, 158, "Chronic", 5.5, ha="center")
    audit = []
    for ai, area in enumerate(PCA_AREAS):
        y = 100 if ai == 0 else 44
        text(offset + 5, y + 31, area, 5.5, ha="center", va="center", rotation=90)
        for col in range(2):
            ax = axes(offset + 10 + 64 * col, y, 62, 62)
            ax.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect="equal")
            ax.set_axis_off()
            points = xy[ai, 2 * col : 2 * col + 2].reshape(-1, 2)
            center = (points.min(0) + points.max(0)) / 2
            for fi in [2 * col + 1, 2 * col]:
                low, high = xy[ai, fi] - center
                color = COLORS[fi]
                q = high[[0, 1, 2, 3, 0]]
                ax.plot(q[:, 0], q[:, 1], color=color, alpha=0.62, lw=0.6, zorder=2, clip_on=False)
                for lo, hi in zip(low, high):
                    ax.annotate(
                        "",
                        xy=hi,
                        xytext=lo,
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=color,
                            lw=0.58,
                            alpha=0.85,
                            mutation_scale=7,
                            shrinkA=1.22,
                            shrinkB=2.05,
                        ),
                        zorder=3,
                    )
                ax.plot(
                    high[:, 0],
                    high[:, 1],
                    ls="none",
                    marker=MARKERS[fi],
                    ms=3.8,
                    mfc="white",
                    mec=color,
                    mew=0.8,
                    zorder=4,
                    clip_on=False,
                )
            # Opaque pastel fills remain visible above all high-contrast markers.
            for fi in [2 * col + 1, 2 * col]:
                low = xy[ai, fi, 0] - center
                pastel = 1 - 0.30 * (1 - np.array(to_rgb(COLORS[fi])))
                ax.plot(
                    low[:, 0],
                    low[:, 1],
                    ls="none",
                    marker=MARKERS[fi],
                    ms=2.2,
                    color=pastel,
                    mew=0,
                    zorder=5,
                    clip_on=False,
                )
            audit.append(
                dict(
                    area=area,
                    timepoint=["baseline", "chronic"][col],
                    conditions=[2 * col, 2 * col + 1],
                    pools=["bottom four", "top four"],
                    limits=[-limit, limit],
                    width_pt=62,
                    height_pt=62,
                    points=16,
                    points_per_raw_unit=62 / (2 * limit),
                    center_translation=center.tolist(),
                    expansion_vectors=8,
                )
            )
    # One common PC triad and one scale bar for every panel, no corner labels.
    triad = axes(offset + 15, 2, 28, 19)
    triad.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
    triad.set_axis_off()
    for k, d in enumerate(basis.T):
        triad.plot([0, d[0]], [0, d[1]], color=".55", lw=0.45)
        triad.text(*(d * 1.25), f"PC{k + 1}", fontsize=4.5, color=".45", ha="center", va="center")
    length = 0.06  # Keep the accepted shared raw-coordinate scale bar.
    length_pt = length * 62 / (2 * limit)
    center = offset + 106
    fig.add_artist(
        Line2D(
            [(center - length_pt / 2) / w, (center + length_pt / 2) / w],
            [12 / h, 12 / h],
            transform=fig.transFigure,
            color=".4",
            lw=0.8,
        )
    )
    text(center, 4, f"{length:g}", 4.5, ha="center")
    return dict(
        axes=audit,
        camera=CAMERA,
        projection_basis=basis.tolist(),
        shared_scale_bar=dict(raw_units=float(length), length_pt=float(length_pt)),
        labels=["PC1", "PC2", "PC3"],
        common_limits=[-limit, limit],
    )
