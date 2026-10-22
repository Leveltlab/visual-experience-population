"""Every bundled data file must be used by the figure exporter."""

from pathlib import Path
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
import pytest

ROOT = Path(__file__).resolve().parents[1]
from plotting.export import PANELS, COMPACT_PANELS


@pytest.mark.integration
def test_all_panels_and_only_necessary_source_data(tmp_path):
    data = ROOT / "plotting/source_data"
    out = tmp_path / "panels"
    script = """
import sys
from pathlib import Path
seen=set();data=Path(sys.argv[1]).resolve()
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        p=Path(args[0]).resolve()
        if data in p.parents:seen.add(p)
sys.addaudithook(audit)
from plotting.export import export_all
export_all(sys.argv[2])
expected={p.resolve() for p in data.rglob('*') if p.is_file()}
assert seen==expected, f'Unused inputs: {expected-seen}; unexpected inputs: {seen-expected}'
"""
    env = {
        **os.environ,
        "VEP_RESULTS_DIR": str(data),
        "VEP_FONT": "DejaVu Sans",
        "MPLBACKEND": "Agg",
    }
    subprocess.run(
        [sys.executable, "-c", script, str(data), str(out)], cwd=ROOT, env=env, check=True
    )
    assert len(list(out.iterdir())) == 4 * len(PANELS)
    for panel in PANELS:
        for ext in ["png", "pdf", "svg"]:
            assert (out / f"panel_{panel}.{ext}").stat().st_size > 100
        for suffix in [".svg", "_outlined.svg"]:
            tree = ET.parse(out / f"panel_{panel}{suffix}")
            assert not tree.findall(".//{http://www.w3.org/2000/svg}image")
            if panel in COMPACT_PANELS:
                expected_width = (
                    "55pt" if panel.startswith("radius_") and panel != "radius_bars" else "150pt"
                )
                assert tree.getroot().attrib["width"] == expected_width
                assert tree.getroot().attrib["height"] == "100pt"
            if "outlined" in suffix:
                assert not tree.findall(".//{http://www.w3.org/2000/svg}text")


def test_font_falls_back_when_myriad_is_unavailable(monkeypatch):
    from plotting import style
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    find = font_manager.findfont

    def missing_myriad(name, *args, **kwargs):
        if name == "Myriad Pro":
            raise ValueError("not installed")
        return find(name, *args, **kwargs)

    monkeypatch.delenv("VEP_FONT", raising=False)
    monkeypatch.setattr(font_manager, "findfont", missing_myriad)
    with pytest.warns(UserWarning, match="DejaVu Sans"):
        style.use_myriad_pro()
    assert plt.rcParams["font.family"] == ["DejaVu Sans"]
