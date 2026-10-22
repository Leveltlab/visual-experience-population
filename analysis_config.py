"""Common paths, experimental conditions and mouse-level aggregation."""

from pathlib import Path
import hashlib, json
import numpy as np

import os

ROOT = Path(__file__).resolve().parent
SIGNAL = os.environ.get("VEP_SIGNAL", "spikes")
if SIGNAL not in ("spikes", "calcium"):
    raise ValueError("VEP_SIGNAL must be spikes or calcium")
DATA = Path(os.environ.get("VEP_DATA_DIR", ROOT / "data" / SIGNAL)).resolve()
OUT = Path(os.environ.get("VEP_RESULTS_DIR", ROOT / "results" / SIGNAL)).resolve()
AREAS = ["V1", "AM-PM", "A-AL-RL", "LM"]
LABELS = ["DR baseline", "NR baseline", "DR chronic", "NR chronic"]
COLORS = ["#a64428", "#276889", "#de963e", "#269d9a"]
N = 10


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol():
    spec = json.loads((ROOT / "protocol.json").read_text())
    spec["signal"] = SIGNAL
    spec["signal_description"] = (
        "Raw deconvolved activity" if SIGNAL == "spikes" else "Raw calcium activity"
    )
    return spec


def manifest():
    return json.loads((DATA / "manifest.json").read_text())


def ids_for(p, draws=None):
    count = protocol()["draws"] if draws is None else draws
    return np.load(DATA / "selections" / f"p{p['pid']:03d}.npy")[:count]


def mouse_average(pops, values):
    groups = []
    meta = []
    for fi in range(4):
        each = []
        for mouse in sorted({p["mouse"] for p in pops if p["fi"] == fi}):
            selected = [p for p in pops if p["fi"] == fi and p["mouse"] == mouse]
            avg = np.mean([values[p["pid"]] for p in selected], axis=0)
            each.append(avg)
            meta.append(dict(fi=fi, mouse=mouse, sessions=len(selected), values=avg))
        groups.append(np.mean(each, axis=0))
    return np.array(groups), meta


def bootstrap_weights(count, fi):
    return (
        np.random.default_rng([20261018, fi, count]).multinomial(
            count, np.ones(count) / count, size=5000
        )
        / count
    )
