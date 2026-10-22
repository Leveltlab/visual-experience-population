"""Expand rectangular Matplotlib hatches to plain vector fills and line segments."""

import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


def expand_hatches(fig):
    """Keep shaded rectangles editable without SVG pattern-paint dependencies.

    Endpoints are clipped numerically to each rectangle. Physical coordinates
    preserve stripe angle and spacing across PNG, PDF and SVG export DPI.
    """
    fig.canvas.draw()
    patches = [p for p in fig.findobj(Patch) if p.get_hatch()]
    for patch in patches:
        assert patch.get_hatch() == "///", "Only the direction-bar slash hatch is supported"
        bounds = patch.get_window_extent().extents / fig.dpi
        x0, y0, x1, y1 = bounds
        patch.set_hatch(None)  # Retains the explicit pale face color.
        spacing = 4 / 72
        for offset in np.arange(np.floor((x0 - y1) / spacing) * spacing, x1 - y0, spacing):
            lo = max(x0, y0 + offset)
            hi = min(x1, y1 + offset)
            if hi <= lo:
                continue
            width, height = fig.get_size_inches()
            line = Line2D(
                np.array([lo, hi]) / width,
                np.array([lo - offset, hi - offset]) / height,
                transform=fig.transFigure,
                color=patch.get_edgecolor(),
                linewidth=0.55,
                solid_capstyle="butt",
                clip_on=False,
                zorder=patch.get_zorder() + 0.01,
            )
            if patch.axes is not None:
                patch.axes.add_line(line)
            else:
                line.set_zorder(6)  # Legend swatches are figure-level artists.
                fig.add_artist(line)
    return len(patches)
