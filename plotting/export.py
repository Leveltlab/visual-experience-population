"""Export individual figure panels from source_data or computed results."""

from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
from analysis_config import OUT, AREAS, LABELS, COLORS
from plotting.style import use_myriad_pro
from plotting import panels_bcd as bcd, panels_fgh as fgh
from plotting.layout_bcd import render_compact


ORIENTATION_PANELS = [
    "B_angular_distance",
    "orientation_contrasts",
    "B_cosine_endpoints",
    "B_cosine_contrast",
    "B_euclidean_endpoints",
    "B_euclidean_contrast",
]
COMPACT_PANELS = [
    "cosine_contrast",
    "euclidean_contrast",
    "radius_bars",
    *["radius_" + area.replace("-", "_") for area in AREAS],
]
PANELS = [
    "B",
    "C",
    "D",
    "F",
    "G",
    "H",
    "B_angular",
    "B_euclidean",
    "B_cross_contrast",
    *ORIENTATION_PANELS,
    *COMPACT_PANELS,
]


def bcd_figure():
    dest = OUT / "direction_cosine_top2/supporting"
    with np.load(dest / "shared_pca.npz") as z:
        coords = z["coordinates"]
    report = SimpleNamespace(
        AREAS=AREAS,
        LABELS=LABELS,
        COLORS=COLORS,
        MARKERS=bcd.MARKERS,
        CAMERA=json.loads((OUT / "camera.json").read_text()),
        SHARED_COORDS=coords,
        RADIUS_GROUPS=pd.read_csv(dest / "radius_groups.csv"),
        draw_panel=bcd.draw_panel,
        decoding_panel=bcd.decoding_panel,
    )
    return render_compact(
        report,
        pd.read_csv(dest / "decoding_groups.csv"),
        pd.read_csv(OUT / "direction_cosine_top2/mouse_profiles.csv"),
        pd.read_csv(OUT / "direction_cosine_top2/group_profiles.csv"),
    )


def isolate(fig, bounds, legend=True):
    """Hide other panels before cropping; keep the original physical panel scale."""
    x, y, w, h = bounds
    fw, fh = fig.get_size_inches() * 72
    for ax in fig.axes:
        center = ax.get_position().get_points().mean(0) * [fw, fh]
        ax.set_visible(x <= center[0] <= x + w and y <= center[1] <= y + h)
    for text in fig.texts:
        tx, ty = np.array(text.get_position()) * [fw, fh]
        text.set_visible(x <= tx < x + w and y <= ty <= y + h)
    # Preserve panel-specific keys; the shared condition key is replaced below.
    for item in fig.legends:
        labels = [t.get_text() for t in item.get_texts()]
        if any(label in LABELS for label in labels):
            item.set_visible(False)
        else:
            box = item.get_bbox_to_anchor().transformed(fig.transFigure.inverted())
            cx, cy = np.array([box.x0, box.y0]) * [fw, fh]
            item.set_visible(x <= cx <= x + w and y <= cy <= y + h)
    for artist in fig.artists:
        if isinstance(artist, Line2D):
            cx, cy = np.array([np.mean(artist.get_xdata()), np.mean(artist.get_ydata())]) * [fw, fh]
            artist.set_visible(x <= cx <= x + w and y <= cy <= y + h)
    extra = 18 if legend else 0
    if legend:
        handles = [
            Line2D([], [], color=c, marker=m, ms=2.5, lw=0.6, label=l)
            for c, m, l in zip(COLORS, bcd.MARKERS, LABELS)
        ]
        fig.legend(
            handles=handles,
            loc="center",
            bbox_to_anchor=((x + w / 2) / fw, (y + h + 9) / fh),
            ncol=2 if w < 200 else 4,
            frameon=False,
            fontsize=4.5,
            handlelength=1.2,
            columnspacing=1,
            borderpad=0,
        )
    return Bbox.from_bounds(x / 72, y / 72, w / 72, (h + extra) / 72)


def export_panel(panel, destination):
    with plt.rc_context():
        return _export_panel(panel, destination)


def _export_panel(panel, destination):
    use_myriad_pro()
    if panel not in ORIENTATION_PANELS + COMPACT_PANELS:
        # Explicit compact-panel defaults keep legacy supplements independent of export order.
        plt.rcParams.update(
            {
                "axes.titleweight": "normal",
                "axes.labelweight": "normal",
                "axes.titlesize": 6,
                "axes.labelsize": 5,
                "xtick.labelsize": 4,
                "ytick.labelsize": 4,
                "axes.linewidth": 0.45,
            }
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if panel in ["B", "C", "D"]:
        fig = bcd_figure()
        bounds = {"B": (0, 100, 150, 100), "C": (150, 100, 150, 100), "D": (0, 0, 300, 100)}[panel]
    elif panel in ["F", "G", "H"]:
        fgh.CAMERA = json.loads((OUT / "camera.json").read_text())
        if panel == "H":
            fig = plt.figure(figsize=(140 / 72, 185 / 72))
            fgh.pca(fig)
            bounds = (0, 0, 140, 185)
        else:
            fig = plt.figure(figsize=(150 / 72, 185 / 72))
            fgh.quantitative(fig)
            bounds = (0, 90, 150, 95) if panel == "F" else (0, 0, 150, 90)
    elif panel == "B_euclidean":
        from plotting.supplement import render, DEST

        fig = render(
            pd.read_csv(DEST / "group_estimates.csv"),
            pd.read_csv(DEST / "posthoc_results.csv"),
            pd.read_csv(DEST / "factorial_euclidean.csv"),
        )
        bounds = (0, 0, 150, 100)
    elif panel == "B_cross_contrast":
        from plotting.cross_contrast import render

        fig = render(
            pd.read_csv(OUT / "direction_cosine_top2/cross_contrast/group_metrics.csv"), {}
        )
        bounds = (0, 0, 828, 266.4)
    elif panel == "B_angular":
        fig, axes = plt.subplots(1, 4, figsize=(7, 2), sharey=True)
        for ax, area in zip(axes, AREAS):
            bcd.draw_panel(
                ax,
                area,
                pd.read_csv(OUT / "direction_cosine_top2/mouse_profiles.csv"),
                pd.read_csv(OUT / "direction_cosine_top2/group_profiles.csv"),
                full_circle=True,
                individual=False,
            )
        fig.tight_layout()
        bounds = (0, 0, 504, 144)
    elif panel in ORIENTATION_PANELS:
        from plotting.orientation_contrasts import render

        fig, tight = render(panel)
    elif panel in COMPACT_PANELS:
        from plotting.compact_contrasts import render

        fig = render(panel)
        tight = False
    else:
        raise ValueError(f"Unknown panel: {panel}")
    if panel in ORIENTATION_PANELS + COMPACT_PANELS:
        box = "tight" if tight else None
    else:
        box = isolate(fig, bounds)
    paths = []
    for ext in ["png", "svg", "pdf"]:
        fig.canvas.draw()
        path = destination / f"panel_{panel}.{ext}"
        fig.savefig(path, bbox_inches=box, dpi=300, transparent=ext != "png", facecolor="white")
        paths.append(path)
    with plt.rc_context({"svg.fonttype": "path"}):
        fig.savefig(destination / f"panel_{panel}_outlined.svg", bbox_inches=box, transparent=True)
    plt.close(fig)
    return paths


def export_all(destination):
    return {panel: export_panel(panel, destination) for panel in PANELS}
