"""Materialize the requested MATLAB fields as lean NumPy/JAX-ready files."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import sys
import uuid

import h5py
import numpy as np


DEFAULT_INPUT = Path("raw")
DEFAULT_OUTPUT = Path(__file__).with_name("pydata")
FACTORS = {
    "darkReared_baseline": ("dark", "baseline"),
    "darkReared_chronic": ("dark", "chronic"),
    "normReared_baseline": ("normal", "baseline"),
    "normReared_chronic": ("normal", "chronic"),
}


@dataclass(frozen=True)
class SourceSession:
    path: Path
    relative_path: str
    group: str
    timepoint: str
    mouse: str


def discover_sessions(root: Path) -> list[SourceSession]:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    sessions = []
    for folder, (group, timepoint) in FACTORS.items():
        for path in sorted((root / folder).glob("*.mat")):
            if path.name.startswith("._"):
                continue
            sessions.append(
                SourceSession(
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    group=group,
                    timepoint=timepoint,
                    mouse=path.stem.split("_", 1)[0],
                )
            )
    if not sessions:
        raise FileNotFoundError(f"No MATLAB sessions found in {root}")
    return sessions


def _read(file: h5py.File, path: str, dtype=None) -> np.ndarray:
    if path not in file:
        raise KeyError(path)
    dataset = file[path]
    if dataset.dtype.kind == "O":
        values = [np.asarray(file[ref][()]).item() for ref in dataset[()].reshape(-1)]
        return np.asarray(values, dtype=dtype)
    values = dataset.astype(dtype)[()] if dtype else dataset[()]
    return np.ascontiguousarray(np.asarray(values).T, dtype=dtype)


def _scalar(array: np.ndarray, name: str, cast):
    if array.size != 1:
        raise ValueError(f"{name} must be scalar, got {array.shape}")
    return cast(array.reshape(-1)[0])


def load_session(source: SourceSession) -> tuple[dict[str, np.ndarray], dict]:
    if not h5py.is_hdf5(source.path):
        raise ValueError(f"Expected a MATLAB v7.3/HDF5 file: {source.path}")

    with h5py.File(source.path, "r") as file:
        calcium = _read(file, "Res/CaSigCorrected", np.float32)
        spikes = _read(file, "Res/CaDeconCorrected", np.float32)
        whisker = _read(file, "Res/facemap/motion", np.float32)
        running = _read(file, "Res/speed", np.float32)
        time = _read(file, "Res/ax", np.float32).reshape(-1)
        stim = _read(file, "info/Stim/log", np.float32)
        n_timesteps, n_trials, n_neurons = calcium.shape

        if spikes.shape != calcium.shape:
            raise ValueError(f"Signal shapes differ in {source.relative_path}")
        if whisker.shape != (n_timesteps, n_trials) or running.shape != whisker.shape:
            raise ValueError(f"Behavior shapes differ in {source.relative_path}")
        if time.size != n_timesteps or stim.shape != (n_trials, 2):
            raise ValueError(f"Time/stimulus shapes differ in {source.relative_path}")

        if "ABAroi/region" in file:
            matlab_region = _read(file, "ABAroi/region").reshape(-1)
            if matlab_region.size != n_neurons or np.any(
                (matlab_region < 0) | (matlab_region > 7) | (matlab_region % 1 != 0)
            ):
                raise ValueError(f"Invalid regions in {source.relative_path}")
            region = matlab_region.astype(np.int16) - 1
            region_missing = False
        else:
            region = np.full(n_neurons, -1, dtype=np.int16)
            region_missing = True

        has_x, has_y = "ABAroi/X" in file, "ABAroi/Y" in file
        if has_x != has_y:
            raise ValueError(f"Only one ABA coordinate exists in {source.relative_path}")
        if has_x:
            neuron_x = _read(file, "ABAroi/X", np.float32).reshape(-1)
            neuron_y = _read(file, "ABAroi/Y", np.float32).reshape(-1)
            coordinate_missing = False
        else:
            neuron_x = np.full(n_neurons, np.nan, dtype=np.float32)
            neuron_y = np.full(n_neurons, np.nan, dtype=np.float32)
            coordinate_missing = True
        if neuron_x.size != n_neurons or neuron_y.size != n_neurons:
            raise ValueError(f"ABA coordinate shapes differ in {source.relative_path}")

        parameter = "info/Stim/Parameters"
        metadata = {
            "source_file": source.relative_path,
            "mouse": source.mouse,
            "n_timesteps": n_timesteps,
            "n_trials": n_trials,
            "n_neurons": n_neurons,
            "time": time.tolist(),
            "stim_frames": [
                _scalar(_read(file, "info/framesbeforstim"), "framesbeforstim", int),
                _scalar(_read(file, "info/framesafterstim"), "framesafterstim", int),
            ],
            "sampling_freq": _scalar(_read(file, "info/Freq"), "Freq", float),
            "cpd": _scalar(_read(file, f"{parameter}/cpd"), "cpd", float),
            "cps": _scalar(_read(file, f"{parameter}/cps"), "cps", float),
            "contrast": _read(file, f"{parameter}/contrast", np.float32).reshape(-1).tolist(),
            "direction": np.unique(stim[:, 0]).tolist(),
            "stim": stim.tolist(),
            "region": region.tolist(),
            "region_missing": region_missing,
            "coordinate_missing": coordinate_missing,
            "neuron_x": neuron_x.tolist(),
            "neuron_y": neuron_y.tolist(),
        }

    return {
        "calcium": calcium,
        "spikes": spikes,
        "whisker": whisker,
        "running": running,
    }, metadata


def _safe(name: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")
    if not result:
        raise ValueError(f"Invalid mouse name: {name!r}")
    return result


def _shared(metadata: list[dict], key: str):
    values = [np.asarray(session[key]) for session in metadata]
    if not all(np.array_equal(values[0], value, equal_nan=True) for value in values[1:]):
        raise ValueError(f"{key} differs across pooled sessions")
    return values[0].tolist()


def _aggregate(metadata: list[dict]) -> tuple[list[dict], dict]:
    sessions, neuron_start = [], 0
    for index, source in enumerate(metadata):
        neuron_stop = neuron_start + source["n_neurons"]
        sessions.append(
            {
                "session_key": f"s{index:04d}",
                "source_file": source["source_file"],
                "mouse": source["mouse"],
                "behavior_index": index,
                "neuron_start": neuron_start,
                "neuron_stop": neuron_stop,
            }
        )
        neuron_start = neuron_stop

    numeric = {
        key: _shared(metadata, key)
        for key in (
            "n_timesteps",
            "n_trials",
            "time",
            "stim_frames",
            "sampling_freq",
            "cpd",
            "cps",
            "contrast",
            "direction",
        )
    }
    numeric.update(
        n_sessions=len(metadata),
        n_neurons=neuron_start,
        n_neurons_by_session=[session["n_neurons"] for session in metadata],
        region_missing_by_session=[session["region_missing"] for session in metadata],
        coordinate_missing_by_session=[session["coordinate_missing"] for session in metadata],
        stim=np.stack([session["stim"] for session in metadata]).tolist(),
        region=np.concatenate([session["region"] for session in metadata]).tolist(),
        neuron_x=np.concatenate([session["neuron_x"] for session in metadata]).tolist(),
        neuron_y=np.concatenate([session["neuron_y"] for session in metadata]).tolist(),
        neuron_session_index=np.repeat(
            np.arange(len(metadata)), [session["n_neurons"] for session in metadata]
        ).tolist(),
    )
    return sessions, numeric


def _write_bundle(
    folder: Path,
    suffix: str,
    records: list[tuple[dict, dict]],
    group: str,
    timepoint: str,
    mouse: str | None = None,
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    signals = [record[0] for record in records]
    sessions, numeric = _aggregate([record[1] for record in records])

    np.save(
        folder / f"calcium_{suffix}.npy",
        np.dstack([x["calcium"] for x in signals]),
        allow_pickle=False,
    )
    np.save(
        folder / f"spikes_{suffix}.npy",
        np.dstack([x["spikes"] for x in signals]),
        allow_pickle=False,
    )
    np.save(
        folder / f"whisker_{suffix}.npy",
        np.stack([x["whisker"] for x in signals]),
        allow_pickle=False,
    )
    np.save(
        folder / f"running_{suffix}.npy",
        np.stack([x["running"] for x in signals]),
        allow_pickle=False,
    )
    document = {
        "format_version": 2,
        "group": group,
        "timepoint": timepoint,
        "region_index_base": 0,
        "region_unassigned": -1,
        "sessions": sessions,
        "numeric": numeric,
    }
    if mouse is not None:
        document["mouse"] = mouse
    (folder / f"metadata_{suffix}.json").write_text(json.dumps(document, separators=(",", ":")))


def convert_sessions(sessions: list[SourceSession], output: Path, verbose: bool = False) -> int:
    factors = defaultdict(list)
    for source in sessions:
        factors[(source.group, source.timepoint)].append(source)

    for (group, timepoint), sources in factors.items():
        suffix = f"{group}_{timepoint}"
        if verbose:
            print(f"Loading {suffix}: {len(sources)} session(s)", flush=True)
        records = []
        for index, source in enumerate(sources, 1):
            signals, metadata = load_session(source)
            records.append((signals, metadata))
            if verbose:
                print(f"  {index}/{len(sources)} {source.relative_path}", flush=True)

        _write_bundle(output / "pooled", suffix, records, group, timepoint)
        by_mouse = defaultdict(list)
        for source, record in zip(sources, records):
            by_mouse[source.mouse].append(record)
        for mouse, mouse_records in by_mouse.items():
            safe_mouse = _safe(mouse)
            _write_bundle(
                output / "by_mouse" / group / timepoint / safe_mouse,
                f"{suffix}_{safe_mouse}",
                mouse_records,
                group,
                timepoint,
                mouse,
            )
    return len(sessions)


def preprocess(root: Path, output: Path = DEFAULT_OUTPUT, overwrite: bool = False) -> int:
    output = output.resolve()
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} already exists")
        shutil.rmtree(output)

    staging = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    try:
        count = convert_sessions(discover_sessions(root), staging, verbose=True)
        staging.replace(output)
        return count
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.dry_run:
            for source in discover_sessions(args.input):
                print(source.group, source.timepoint, source.mouse, source.relative_path)
        else:
            count = preprocess(args.input, args.output, args.overwrite)
            print(f"Wrote {count} session(s) to {args.output}")
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
