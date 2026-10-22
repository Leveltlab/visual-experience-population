"""README-only layouts; saved figure coordinates and main panel exports are unchanged.

Run from the repository root: uv run python -m plotting.previews.render
"""

from pathlib import Path
import os
import json

ROOT = Path(__file__).resolve().parents[2]
os.environ["VEP_RESULTS_DIR"] = str(ROOT / "plotting/source_data")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from analysis_config import AREAS, COLORS, LABELS, OUT
from plotting.style import use_myriad_pro
from plotting.layout_bcd import dashed_marker
from plotting.statistics import comparisons, vertical_bracket
from plotting import panels_fgh

DEST = Path(__file__).resolve().parent
MARKERS = ["o", "s", "o", "s"]


def legend(fig, y):
    fig.legend(
        handles=[
            Line2D([], [], color=c, marker=m, ms=2.8, lw=0.6, label=l)
            for c, m, l in zip(COLORS, MARKERS, LABELS)
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, y),
        ncol=2,
        frameon=False,
        fontsize=4.5,
        handlelength=1.2,
        columnspacing=1.1,
        borderaxespad=0,
    )


def direction_preview(camera):
    with np.load(OUT / "direction_cosine_top2/supporting/shared_pca.npz") as data:
        coords = data["coordinates"]
    radius = pd.read_csv(OUT / "direction_cosine_top2/supporting/radius_groups.csv")
    fig = plt.figure(figsize=(164 / 72, 216 / 72))
    row_offset = 0

    def axes(x, y, w, h):
        return fig.add_axes([x / 164, (y + row_offset) / 216, w / 164, h / 216])

    def text(x, y, label, size=5, **kwargs):
        return fig.text(x / 164, (y + row_offset) / 216, label, fontsize=size, **kwargs)

    text(1, 207, "D", 7)
    legend(fig, 0.97)
    key = axes(33, 192, 6, 6)
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.set_axis_off()
    key.plot(0.5, 0.5, "o", color=".3", ms=2.3)
    text(40, 193, "0°–135°", 4)
    key = axes(81, 192, 6, 6)
    key.set(xlim=(0, 1), ylim=(0, 1))
    key.set_axis_off()
    dashed_marker(key, 0.5, 0.5, "o", 3.8, ".3")
    text(88, 193, "180°–315°", 4)
    e, a = np.deg2rad([camera["elevation"], camera["azimuth"]])
    basis = np.array(
        [[-np.sin(a), np.cos(a), 0], [-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)]]
    )
    all_xy = coords @ basis.T
    for ai, area in enumerate(AREAS):
        x = (ai % 2) * 82
        row_offset = (1 - ai // 2) * 95
        text(x + 36, 93, area, 6, ha="center")
        ax = axes(x + 8, 15, 59, 59)
        xy = all_xy[ai]
        extent = np.max(abs(xy))
        lim = extent * 1.33
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal")
        ax.set_axis_off()
        for start, ls in [(4, "--"), (0, "-")]:
            for fi, color in enumerate(COLORS):
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
                        dashed_marker(ax, *point, MARKERS[fi], 3.8, color)
                    else:
                        ax.plot(
                            *point,
                            marker=MARKERS[fi],
                            ms=2.3,
                            mew=0.3,
                            color=color,
                            alpha=1,
                            zorder=6,
                        )
        for fi, color in enumerate(COLORS):
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
        table = radius[radius.area == area]
        for fi, color in enumerate(COLORS):
            g = table[table.fi == fi].iloc[0]
            inset.errorbar(
                g["mean"],
                3 - fi,
                xerr=np.maximum(0, [[g["mean"] - g.ci_low], [g.ci_high - g["mean"]]]),
                fmt=MARKERS[fi],
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
    return fig


def contrast_preview(camera):
    # Reposition the existing vector artists, retaining the common coordinate scale.
    panels_fgh.CAMERA = camera
    fig = plt.figure(figsize=(140 / 72, 185 / 72))
    panels_fgh.pca(fig)
    boxes = [(ax, ax.get_position().bounds) for ax in fig.axes]
    texts = [(t, t.get_position()) for t in fig.texts]
    fig.set_size_inches(140 / 72, 148 / 72)
    for ax, (x, y, w, h) in boxes:
        y = y * 185 - (35 if y * 185 > 90 else 23 if y * 185 > 30 else 0)
        ax.set_position([x, y / 148, w, h * 185 / 148])
    for t, (x, y) in texts:
        t.set_position((x, (y * 185 - (35 if y * 185 > 100 else 23 if y * 185 > 30 else 0)) / 148))
    for label in fig.texts:
        if label.get_text() == "H":
            label.set_position((1 / 140, 139 / 148))
    for line in fig.artists:
        if isinstance(line, Line2D):
            line.set_ydata(np.asarray(line.get_ydata()) * 185 / 148)
    legend(fig, 0.98)
    return fig


def main():
    use_myriad_pro()
    plt.rcParams.update({"font.size": 4, "axes.linewidth": 0.45})
    camera = json.loads((OUT / "camera.json").read_text())
    for name, render in [("D", direction_preview), ("H", contrast_preview)]:
        with plt.rc_context():
            fig = render(camera)
            for ext in ["png", "svg"]:
                fig.savefig(DEST / f"panel_{name}.{ext}", dpi=300, facecolor="white")
            plt.close(fig)


if __name__ == "__main__":
    main()
