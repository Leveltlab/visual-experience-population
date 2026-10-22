"""Load the materialized JAX neuroscience data."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


PYDATA = Path(os.environ.get("VEP_PYDATA_DIR", Path(__file__).with_name("pydata")))


@dataclass
class CalciumData:
    calcium: jax.Array
    metadata: dict
    ord_stim: jax.Array | None = None


@dataclass
class SpikeData:
    spikes: jax.Array
    metadata: dict
    ord_stim: jax.Array | None = None


@dataclass
class BehaviorData:
    whisker: jax.Array
    running: jax.Array
    metadata: dict


Data = CalciumData | SpikeData | BehaviorData
NeuralData = CalciumData | SpikeData


@dataclass
class ProcessedData:
    """One independently processed neural signal and its aligned metadata."""

    signal_type: str
    signal: jax.Array
    time: jax.Array
    direction: jax.Array
    orientation: jax.Array
    contrast: jax.Array
    relative_phase: jax.Array
    cpd: jax.Array
    cps: jax.Array
    mouse: jax.Array
    mouse_names: tuple[str, ...]
    region: jax.Array
    ord_stim: jax.Array
    trial_permutation: jax.Array


def get_data(
    kind: Literal["pooled", "mouse"],
    signal: Literal["calcium", "spike", "behavior"],
    group: Literal["dark", "normal"],
    timepoint: Literal["baseline", "chronic"],
    mouse: str | None = None,
) -> Data:
    """Load only the requested signal and its matching metadata."""
    suffix = f"{group}_{timepoint}"
    if kind == "pooled":
        folder = PYDATA / "pooled"
    elif kind == "mouse" and mouse:
        folder = PYDATA / "by_mouse" / group / timepoint / mouse
        suffix = f"{suffix}_{mouse}"
    elif kind == "mouse":
        raise ValueError("mouse is required when kind='mouse'")
    else:
        raise ValueError("kind must be 'pooled' or 'mouse'")

    metadata_path = folder / f"metadata_{suffix}.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    metadata = json.loads(metadata_path.read_text())
    metadata["numeric"] = {name: jnp.asarray(value) for name, value in metadata["numeric"].items()}

    def load(name: str) -> jax.Array:
        path = folder / f"{name}_{suffix}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        return jnp.asarray(np.load(path, mmap_mode="r"))

    if signal == "calcium":
        return CalciumData(load("calcium"), metadata)
    if signal == "spike":
        return SpikeData(load("spikes"), metadata)
    if signal == "behavior":
        return BehaviorData(load("whisker"), load("running"), metadata)
    raise ValueError("signal must be 'calcium', 'spike', or 'behavior'")


def _trial_order(metadata: dict) -> tuple[jax.Array, jax.Array, jax.Array]:
    stim = np.asarray(metadata["numeric"]["stim"])
    if stim.ndim != 3 or stim.shape[2] != 2:
        raise ValueError(f"stim must have shape (sessions, trials, 2), got {stim.shape}")

    permutations = np.stack([np.lexsort((session[:, 1], session[:, 0])) for session in stim])
    ordered = np.take_along_axis(stim, permutations[:, :, None], axis=1)
    if not all(np.array_equal(ordered[0], session, equal_nan=True) for session in ordered[1:]):
        raise ValueError("Sessions do not contain the same number of the same stimulus trials")
    return jnp.asarray(permutations), jnp.asarray(ordered), jnp.asarray(ordered[0])


def sort_trials(data: NeuralData) -> NeuralData:
    """Canonically order each session's trials before restacking its neurons."""
    if not isinstance(data, (CalciumData, SpikeData)):
        raise TypeError("sort_trials currently supports CalciumData and SpikeData only")
    permutations, ordered_stim, ord_stim = _trial_order(data.metadata)
    sessions = data.metadata["sessions"]
    if len(sessions) != permutations.shape[0]:
        raise ValueError("Session metadata does not match the stimulus session axis")

    metadata = dict(data.metadata)
    metadata["numeric"] = dict(metadata["numeric"])
    metadata["numeric"].update(
        stim=ordered_stim,
        trial_permutation=permutations,
        ord_stim=ord_stim,
    )
    metadata["trial_order"] = "direction, then contrast; canonical index -> original index"

    values = data.calcium if isinstance(data, CalciumData) else data.spikes
    blocks = [
        jnp.take(
            values[:, :, session["neuron_start"] : session["neuron_stop"]],
            permutation,
            axis=1,
        )
        for session, permutation in zip(sessions, permutations)
    ]
    ordered = jnp.concatenate(blocks, axis=2)
    if isinstance(data, CalciumData):
        return CalciumData(ordered, metadata, ord_stim)
    return SpikeData(ordered, metadata, ord_stim)


def final_preprocess(data: NeuralData) -> ProcessedData:
    """Sort and package one calcium or spike matrix without touching behavior."""
    ordered = sort_trials(data)
    numeric = ordered.metadata["numeric"]
    ord_stim = ordered.ord_stim
    assert ord_stim is not None

    direction = jnp.deg2rad(ord_stim[:, 0])
    orientation = jnp.deg2rad(jnp.mod(ord_stim[:, 0], 180))
    time = numeric["time"]
    phase = jnp.mod(2 * jnp.pi * numeric["cps"] * jnp.maximum(time, 0), 2 * jnp.pi)
    relative_phase = jnp.broadcast_to(phase[:, None], (time.size, ord_stim.shape[0]))

    mouse_names = tuple(dict.fromkeys(session["mouse"] for session in ordered.metadata["sessions"]))
    mouse_lookup = {name: index for index, name in enumerate(mouse_names)}
    mouse_by_session = jnp.asarray(
        [mouse_lookup[session["mouse"]] for session in ordered.metadata["sessions"]]
    )
    mouse = mouse_by_session[numeric["neuron_session_index"]]

    is_calcium = isinstance(ordered, CalciumData)

    return ProcessedData(
        signal_type="calcium" if is_calcium else "spike",
        signal=ordered.calcium if is_calcium else ordered.spikes,
        time=time,
        direction=direction,
        orientation=orientation,
        contrast=ord_stim[:, 1],
        relative_phase=relative_phase,
        cpd=numeric["cpd"],
        cps=numeric["cps"],
        mouse=mouse,
        mouse_names=mouse_names,
        region=numeric["region"],
        ord_stim=ord_stim,
        trial_permutation=numeric["trial_permutation"],
    )
