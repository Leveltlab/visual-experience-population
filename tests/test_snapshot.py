"""Portable regression checks against the included accepted figure tables."""

from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from geometry import inference

SNAPSHOT = Path(__file__).resolve().parents[1] / "plotting/source_data"


def test_snapshot_hashes():
    spec = json.loads((SNAPSHOT / "manifest.json").read_text())
    for name, expected in spec["sha256"].items():
        assert hashlib.sha256((SNAPSHOT / name).read_bytes()).hexdigest() == expected


def test_accepted_inference_inputs_and_adjusted_pvalues(monkeypatch):
    monkeypatch.setattr(inference, "OUT", SNAPSHOT)
    tables = inference.inputs()
    keys = ["mouse", "area", "time", "level"]
    for panel, table in tables.items():
        old = pd.read_csv(SNAPSHOT / "manuscript_statistics" / f"input_{panel}.csv")
        np.testing.assert_allclose(
            table.sort_values(keys).response,
            old.sort_values(keys).response,
            atol=1e-12,
            equal_nan=True,
        )
    new = inference.posthocs(tables)
    old = pd.read_csv(SNAPSHOT / "manuscript_statistics/posthoc_results.csv")
    keys = ["panel", "area", "timepoint", "step"]
    np.testing.assert_allclose(
        new.sort_values(keys).p_adjusted, old.sort_values(keys).p_adjusted, atol=1e-12
    )
