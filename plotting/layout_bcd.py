"""Fixed-size Illustrator B/C/D assembly; presentation only, saved data unchanged."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib.text import Text
import numpy as np
from plotting.statistics import main_label, comparisons, horizontal_bracket, vertical_bracket


def dashed_marker(ax, x, y, marker, size, color):
    """A real dashed vector outline in physical points (Line2D markers cannot dash)."""
    style = MarkerStyle(marker)
    path = style.get_path().transformed(style.get_transform())
    # Update the display-space translation at draw time via a point offset transform.
    from matplotlib.transforms import ScaledTranslation

    transform = (
        Affine2D().scale(size / 72)
        + ax.figure.dpi_scale_trans
        + ScaledTranslation(x, y, ax.transData)
    )
    patch = PathPatch(
        path,
        transform=transform,
        facecolor="white",
        edgecolor=color,
        linewidth=0.55,
        linestyle=(0, (1.1, 0.7)),
        alpha=0.98,
        zorder=4,
    )
    ax.add_patch(patch)


def render_compact(report, groups, mouse, angular):
    R = report
    plt.rcParams.update(
        {
            "font.size": 4,
            "font.weight": "normal",
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlesize": 6,
            "axes.labelsize": 5,
            "xtick.labelsize": 4,
            "ytick.labelsize": 4,
            "axes.linewidth": 0.45,
        }
    )
    fig = plt.figure(figsize=(300 / 72, 200 / 72))

    def axes(x, y, w, h):
        return fig.add_axes([x / 300, y / 200, w / 300, h / 200])

    def text(x, y, s, size=5, **kw):
        return fig.text(x / 300, y / 200, s, fontsize=size, **kw)

    for x, letter in [(1, "B"), (151, "C")]:
        text(x, 181, letter, 7)
    text(1, 93, "D", 7)
    text(78, 187, main_label("B"), 4, ha="center")
    text(228, 187, main_label("C"), 4, ha="center")
    handles = [
        Line2D([], [], color=c, marker=m, ms=2.7, lw=0.6, label=l)
        for c, m, l in zip(
            R.COLORS, R.MARKERS, ["DR baseline", "NR baseline", "DR chronic", "NR chronic"]
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.997),
        ncol=4,
        frameon=False,
        fontsize=5,
        handlelength=1.3,
        columnspacing=1.2,
        handletextpad=0.4,
        borderpad=0,
    )
    for j, area in enumerate(R.AREAS):
        ax = axes(18 + j * 33, 122, 27, 52)
        R.draw_panel(ax, area, mouse, angular, individual=False, dodge=0.15)
        ax.set_title(area, fontsize=6, pad=3)
        ax.set_xticklabels(["180°", "90°"], fontsize=4)
        ax.set_yticks([-1, 0, 1])
        ax.tick_params(labelsize=4, length=1.6, width=0.45, pad=1)
        for line in ax.lines:
            line.set_linewidth(line.get_linewidth() * 0.45)
            line.set_markersize(line.get_markersize() * 0.46)
            line.set_markeredgewidth(0.4)
        for collection in ax.collections:
            collection.set_linewidth(0.5)
        for row in comparisons("B", area).itertuples():
            fi = 0 if row.timepoint == "baseline" else 2
            table = angular[(angular.area == area) & (angular.difference_degrees == 180)].set_index(
                "fi"
            )
            vertical_bracket(
                ax,
                0.08 if fi == 0 else 0.31,
                table.loc[fi, "mean"],
                table.loc[fi + 1, "mean"],
                row.stars,
            )
        if j == 0:
            ax.set_ylabel("Cosine similarity", labelpad=2, fontsize=5)
        else:
            ax.tick_params(labelleft=False)
        ax = axes(169 + j * 33, 122, 27, 52)
        R.decoding_panel(ax, area, groups, small=True)
        ax.set_title(area, fontsize=6, pad=3)
        ax.set_xticklabels(["DR", "NR", "DR", "NR"], fontsize=4)
        ax.tick_params(labelsize=4, length=1.6, width=0.45, pad=1)
        for line in ax.lines:
            line.set_linewidth(0.45)
            line.set_markersize(1.6)
            line.set_markeredgewidth(0.45)
        for collection in ax.collections:
            collection.set_linewidth(0.45)
        for patch in ax.patches:
            patch.set_linewidth(0.35)
        for row in comparisons("C", area).itertuples():
            fi = 0 if row.timepoint == "baseline" else 2
            ceiling = (
                groups[(groups.area == area) & (groups.fi.isin([fi, fi + 1]))].ci_high.max() * 100
            )
            horizontal_bracket(ax, fi - 0.19, fi + 0.81, ceiling / 80 + 0.035, row.stars)
        if j == 0:
            ax.set_ylabel("Accuracy (%)", labelpad=2, fontsize=5)
        else:
            ax.tick_params(labelleft=False)
    text(81, 110, "Direction difference (∆ / °)", ha="center")
    fig.legend(
        handles=[
            Line2D([], [], color=".55", lw=2, alpha=0.5, label="Ideal orientation"),
            Line2D([], [], color="black", lw=2, alpha=0.5, label="Ideal direction"),
        ],
        loc="center",
        bbox_to_anchor=(0.27, 0.516),
        frameon=False,
        ncol=2,
        fontsize=4,
        handlelength=1.2,
        columnspacing=1,
        handletextpad=0.5,
    )
    fig.legend(
        handles=[
            Patch(facecolor=".4", label="Orientation"),
            Patch(facecolor=".85", edgecolor=".4", hatch="///", label="Direction"),
            Line2D([], [], color=".25", ls="--", lw=0.6, label="Chance: 25%"),
            Line2D([], [], color=".5", ls=":", lw=0.6, label="12.5%"),
        ],
        loc="center",
        bbox_to_anchor=(0.76, 0.54),
        frameon=False,
        ncol=4,
        fontsize=4,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    # Direction key beneath the leading area heading, within the first D panel.
    key = axes(5, 80, 6, 6)
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.set_axis_off()
    key.plot(0.5, 0.5, "o", color=".3", ms=2.3)
    text(12, 81, "0°–135°", 4)
    key = axes(35, 80, 6, 6)
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.set_axis_off()
    dashed_marker(key, 0.5, 0.5, "o", 3.8, ".3")
    text(42, 81, "180°–315°", 4)
    e, a = np.deg2rad([R.CAMERA["elevation"], R.CAMERA["azimuth"]])
    basis = np.array(
        [[-np.sin(a), np.cos(a), 0], [-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)]]
    )
    all_xy = R.SHARED_COORDS @ basis.T
    layout = []
    for ai, area in enumerate(R.AREAS):
        x = ai * 75
        text(x + 36, 93, area, 6, ha="center")
        ax = axes(x + 8, 15, 59, 59)
        xy = all_xy[ai]
        extent = np.max(abs(xy))
        lim = extent * 1.33
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        ax.set_axis_off()
        for start, ls in [(4, "--"), (0, "-")]:
            for fi, color in enumerate(R.COLORS):
                q = xy[fi]
                inds = np.array([0, 1, 2, 3, 0]) + start
                ax.plot(
                    q[inds, 0],
                    q[inds, 1],
                    color=color,
                    ls=ls,
                    lw=0.55,
                    alpha=0.48 if start else 0.65,
                )
                for point in q[start : start + 4]:
                    if start:
                        dashed_marker(ax, *point, R.MARKERS[fi], 3.8, color)
                    else:
                        ax.plot(
                            *point,
                            marker=R.MARKERS[fi],
                            ms=2.3,
                            mew=0.3,
                            color=color,
                            alpha=1,
                            zorder=6,
                        )
        for fi, color in enumerate(R.COLORS):
            for d in range(4):
                q = xy[fi, [d, d + 4]]
                ax.plot(q[:, 0], q[:, 1], color=color, ls=":", lw=0.4, alpha=0.5, zorder=2)
        local = (xy[:, :4] + xy[:, 4:]).mean(0) / 2
        # Fixed label slots outside the geometry cloud; every corner has a leader.
        label_slots = [(-0.94, 0.53), (-0.50, -0.87), (0.91, -0.69), (-0.30, 1.06)]
        for d, slot in enumerate(label_slots):
            ax.annotate(
                f"{d * 45}°/{(d + 4) * 45}°",
                xy=local[d],
                xytext=np.array(slot) * lim,
                ha="center",
                va="center",
                fontsize=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.5, alpha=0.95),
                arrowprops=dict(arrowstyle="-", color=".6", lw=0.3, shrinkA=2, shrinkB=3),
                zorder=8,
            )
        ax.plot(0, 0, marker="+", color=".6", ms=2, mew=0.4)
        unit = 10 ** np.floor(np.log10(extent * 0.5))
        length = np.floor(extent * 0.5 / unit) * unit
        bar = axes(x + 39, 5, 20, 5)
        bar.set(xlim=(0, 20 * 2 * lim / 59), ylim=(0, 1))
        bar.set_axis_off()
        bar.plot([0, length], [0.8, 0.8], color=".3", lw=0.7)
        bar.text(length / 2, 0.25, f"{length:g}", fontsize=4, ha="center", va="top")
        # The triad is projected with exactly the same orthographic camera.
        triad = axes(x + 6, 3, 23, 11)
        triad.set(xlim=(-1.4, 1.4), ylim=(-1.2, 1.5))
        triad.set_axis_off()
        for k, delta in enumerate(basis.T):
            triad.annotate(
                "",
                xy=delta,
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-", color=".5", lw=0.4, shrinkA=0, shrinkB=0),
            )
            end = delta * 1.35
            triad.text(*end, f"PC{k + 1}", fontsize=4, color=".4", ha="center", va="center")
        inset = axes(x + 48, 65, 22, 12)
        table = R.RADIUS_GROUPS[R.RADIUS_GROUPS.area == area]
        for fi, color in enumerate(R.COLORS):
            g = table[table.fi == fi].iloc[0]
            inset.errorbar(
                g["mean"],
                3 - fi,
                xerr=np.maximum(0, [[g["mean"] - g.ci_low], [g.ci_high - g["mean"]]]),
                fmt=R.MARKERS[fi],
                color=color,
                ms=1.9,
                capsize=0.8,
                lw=0.45,
                mew=0.35,
            )
        top = table.ci_high.max() * 1.08
        inset.set(xlim=(0, top), ylim=(-0.65, 3.65), yticks=[], xticks=[0, top * 0.8])
        inset.set_xticklabels(["0", f"{top * 0.8:.2g}"])
        inset.set_title("r²", fontsize=4, pad=1)
        inset.tick_params(labelsize=4, length=1, pad=0.5)
        inset.spines[["top", "right", "left"]].set_visible(False)
        for row in comparisons("D", area).itertuples():
            fi = 0 if row.timepoint == "baseline" else 2
            vertical_bracket(inset, top * 0.94, 3 - fi, 2 - fi, row.stars, width=top * 0.05)
        layout.append(
            dict(
                area=area,
                zoom=float(np.max(abs(all_xy)) / extent),
                retained_3d=float(
                    np.sum(R.SHARED_COORDS[ai] ** 2) / np.sum(R.SHARED_ALIGNED[ai] ** 2)
                ),
            )
        )
    from plotting.vector_hatching import expand_hatches

    assert expand_hatches(fig) == 17  # Sixteen direction bars and their legend key.
    fig.canvas.draw()
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 4
    assert all(t.get_fontweight() in ("normal", 400) for t in texts)
    assert all(t.get_fontfamily() == ["Myriad Pro"] for t in texts)
    return fig
