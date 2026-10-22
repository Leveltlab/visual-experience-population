"""End-to-end check with newly generated MATLAB files and no saved analysis inputs."""

import os
from pathlib import Path
import subprocess
import sys
import pytest
from tests.test_preprocess import write_recording
from plotting.export import PANELS
from preprocess.matlab2py import CONDITIONS


@pytest.mark.integration
def test_raw_recordings_through_all_panels(tmp_path):
    raw = tmp_path / "raw"
    for fi, (folder, group, time) in enumerate(CONDITIONS):
        for m in range(2):
            write_recording(raw, folder, f"{group}{m}", 100 + fi * 2 + m)
    env = {
        **os.environ,
        "VEP_DATA_DIR": str(tmp_path / "prepared"),
        "VEP_RESULTS_DIR": str(tmp_path / "results"),
        "VEP_FONT": "DejaVu Sans",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
    }
    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [sys.executable, str(root / "run.py"), "--input", str(raw)],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=900,
    )
    assert len(list((tmp_path / "results/panels").glob("*.png"))) == len(PANELS)
