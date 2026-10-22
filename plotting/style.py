"""Consistent figure typography with a portable Matplotlib font fallback."""

from pathlib import Path
import os
import warnings
import matplotlib.pyplot as plt
from matplotlib import font_manager


def use_myriad_pro(font_directory=None):
    """Use Myriad Pro when installed, otherwise DejaVu Sans; VEP_FONT overrides either."""
    directory = font_directory or os.environ.get("VEP_FONT_DIR")
    directories = ([Path(directory)] if directory else []) + [
        Path(
            "/Library/Application Support/Adobe/Acrobat/DC/WebResources/Resource0/OWP/default/fonts"
        ),
        Path.home() / "Library/Fonts",
    ]
    for folder in directories:
        for path in sorted(folder.glob("MyriadPro-*.otf")):
            font_manager.fontManager.addfont(path)
    family = os.environ.get("VEP_FONT", "Myriad Pro")
    try:
        path = font_manager.findfont(family, fallback_to_default=False)
    except ValueError:
        if "VEP_FONT" in os.environ:
            raise
        family = "DejaVu Sans"
        path = font_manager.findfont(family)
        warnings.warn(
            "Myriad Pro is unavailable; using DejaVu Sans. Data and panel scale are unchanged.",
            stacklevel=2,
        )
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "font.weight": "normal",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": family,
            "mathtext.it": f"{family}:italic",
            "mathtext.bf": f"{family}:bold",
            "mathtext.sf": family,
            "mathtext.tt": family,
            "mathtext.cal": family,
            "mathtext.fallback": "stixsans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return path
