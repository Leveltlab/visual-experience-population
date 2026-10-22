"""Read the neural fields needed by the population analysis from MATLAB v7.3 files."""

from dataclasses import dataclass
from pathlib import Path
import h5py
import numpy as np

# This order defines session IDs and therefore the deterministic sampling seeds.
CONDITIONS = (
    ("darkReared_baseline", "dark", "baseline"),
    ("normReared_baseline", "normal", "baseline"),
    ("darkReared_chronic", "dark", "chronic"),
    ("normReared_chronic", "normal", "chronic"),
)


@dataclass(frozen=True)
class Session:
    path: Path
    relative_path: str
    fi: int
    group: str
    timepoint: str
    mouse: str


def discover_sessions(root):
    """Condition order above, then lexicographic filename order within condition."""
    root = Path(root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Recording directory does not exist: {root}")
    sessions = []
    for fi, (folder, group, timepoint) in enumerate(CONDITIONS):
        for path in sorted((root / folder).glob("*.mat")):
            if path.name.startswith("."):
                continue
            sessions.append(
                Session(
                    path,
                    path.relative_to(root).as_posix(),
                    fi,
                    group,
                    timepoint,
                    path.stem.split("_", 1)[0],
                )
            )
    if not sessions:
        raise FileNotFoundError(
            f"No .mat recordings found under the four condition folders in {root}"
        )
    return sessions


def load_session(session, signal="spikes"):
    """Return time × trial × neuron raw signal, time, trial labels and atlas regions.

    MATLAB dimensions are reversed by h5py. Signals/time/labels are read as float32;
    subsequent averaging is float64. Region IDs 1–7 become 0–6; 0 means unassigned.
    """
    fields = {"spikes": "Res/CaDeconCorrected", "calcium": "Res/CaSigCorrected"}
    if signal not in fields:
        raise ValueError("signal must be spikes or calcium")
    if not h5py.is_hdf5(session.path):
        raise ValueError(f"Expected MATLAB v7.3/HDF5: {session.path}")
    with h5py.File(session.path, "r") as file:

        def read(name):
            if name not in file:
                raise ValueError(f"{session.relative_path}: missing {name}")
            return np.ascontiguousarray(file[name][()].T, dtype=np.float32)

        values = read(fields[signal])
        time = read("Res/ax").reshape(-1)
        stim = read("info/Stim/log")
        if values.ndim != 3 or values.shape[:2] != (len(time), len(stim)) or stim.shape[1:] != (2,):
            raise ValueError(
                f"{session.relative_path}: expected time × trial × neuron signal and trial × 2 labels"
            )
        if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
            raise ValueError(f"{session.relative_path}: time must be finite and increasing")
        if not np.isfinite(stim).all():
            raise ValueError(f"{session.relative_path}: stimulus labels must be finite")
        if "ABAroi/region" in file:
            regions = np.asarray(file["ABAroi/region"][()]).reshape(-1)
            if (
                len(regions) != values.shape[2]
                or not np.isfinite(regions).all()
                or np.any((regions < 0) | (regions > 7) | (regions % 1 != 0))
            ):
                raise ValueError(f"{session.relative_path}: invalid atlas region IDs")
            regions = regions.astype(np.int16) - 1
        else:
            regions = np.full(values.shape[2], -1, dtype=np.int16)
    return values, time, stim, regions
