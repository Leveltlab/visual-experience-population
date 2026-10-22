from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_FAMILY = "Myriad Pro"
FONT_DIRECTORIES = [
    Path("/Library/Application Support/Adobe/Acrobat/DC/WebResources/Resource0/OWP/default/fonts"),
    Path.home() / "Library/Fonts",
]


def use_myriad_pro(font_directory=None):
    """Register local Adobe fonts and fail explicitly if Myriad Pro is unavailable."""
    directories = ([Path(font_directory)] if font_directory else []) + FONT_DIRECTORIES
    for directory in directories:
        for path in sorted(directory.glob("MyriadPro-*.otf")):
            font_manager.fontManager.addfont(path)
    regular = font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "mathtext.fontset": "custom",
            "mathtext.rm": FONT_FAMILY,
            "mathtext.it": f"{FONT_FAMILY}:italic",
            "mathtext.bf": f"{FONT_FAMILY}:bold",
            "mathtext.sf": FONT_FAMILY,
            "mathtext.tt": FONT_FAMILY,
            "mathtext.cal": FONT_FAMILY,
            "mathtext.fallback": "stixsans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return regular
