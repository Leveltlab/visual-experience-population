import pandas as pd
from analysis_config import OUT
import xml.etree.ElementTree as ET

DEST = OUT / "manuscript_statistics"


def editable_regular_svg(path):
    """Explicit Illustrator font mapping without outlining or PDF round-trips."""
    ns = "http://www.w3.org/2000/svg"
    ET.register_namespace("", ns)
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    tree = ET.parse(path)
    texts = list(tree.getroot().iter("{" + ns + "}text"))
    assert texts, "Editable SVG must contain text nodes"
    for text in texts:
        for node in text.iter():
            # Keep explicit symbol fallback only for unsupported mathematical arrows.
            if "".join(node.itertext()).strip() in ["→"]:
                continue
            node.set(
                "style",
                node.get("style", "")
                + ";font-family:'MyriadPro-Regular','Myriad Pro';font-weight:400;font-style:normal",
            )
    tree.write(path, encoding="utf-8", xml_declaration=True)


def main_label(panel):
    table = pd.read_csv(DEST / "factorial_main_effects.csv")
    row = table[(table.panel == panel) & (table.factor == "level")].iloc[0]
    names = {
        "B": "Direction-difference effect",
        "C": "Decoder-task effect",
        "F": "Contrast effect",
        "G": "Contrast-transition effect",
    }
    return names[panel] + " " + row.stars


def comparisons(panel, area, step=None):
    data = pd.read_csv(DEST / "posthoc_results.csv")
    selected = data[(data.panel == panel) & (data.area == area) & (data.p_adjusted < 0.05)]
    if step is not None:
        selected = selected[selected.step == step]
    return selected


def horizontal_bracket(ax, x0, x1, y, stars, height=0.025, size=4):
    """Data x coordinates and axes-fraction y; independent of log/linear scale."""
    transform = ax.get_xaxis_transform()
    ax.plot(
        [x0, x0, x1, x1],
        [y, y + height, y + height, y],
        transform=transform,
        color=".12",
        lw=0.35,
        clip_on=False,
    )
    ax.text(
        (x0 + x1) / 2,
        y + height + 0.015,
        stars,
        transform=transform,
        ha="center",
        va="bottom",
        fontsize=size,
        weight="normal",
    )


def vertical_bracket(ax, x, y0, y1, stars, width=0.04, size=4):
    ax.plot([x - width, x, x, x - width], [y0, y0, y1, y1], color=".12", lw=0.35, clip_on=False)
    ax.text(
        x + width * 0.4,
        (y0 + y1) / 2,
        stars,
        ha="left",
        va="center",
        fontsize=size,
        weight="normal",
    )
