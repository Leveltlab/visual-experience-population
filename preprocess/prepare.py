"""Prepare raw MATLAB recordings for all figure analyses, one session at a time."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from analysis_config import DATA, AREAS, SIGNAL, protocol
from preprocess.matlab2py import discover_sessions, load_session
from preprocess.folds import make_folds

REGIONS = {"V1": [6], "AM-PM": [0, 1], "A-AL-RL": [2, 3, 4], "LM": [5]}


def prepare(input_dir, output_dir=DATA, signal=SIGNAL, overwrite=False):
    """Discover recordings and save responses, native temporal frames, folds and draws."""
    sessions = discover_sessions(input_dir)
    destination = Path(output_dir).resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} already exists; use --overwrite to rebuild it")
    raw_root = Path(input_dir).resolve()
    if (
        destination == raw_root
        or destination in raw_root.parents
        or raw_root in destination.parents
    ):
        raise ValueError("Prepared output must be separate from the raw recording tree")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
    spec = protocol()
    sources = []
    populations = []
    common_time = None
    contrasts = None
    try:
        for name in ["responses", "temporal", "selections", "decoding_folds"]:
            (staging / name).mkdir()
        for sid, session in enumerate(sessions):
            raw, time, stim, regions = load_session(session, signal)
            folds = make_folds(stim, sid)
            if common_time is None:
                common_time = time
                contrasts = np.unique(stim[:, 1])
            elif not np.array_equal(time, common_time) or not np.array_equal(
                np.unique(stim[:, 1]), contrasts
            ):
                raise ValueError(
                    f"{session.relative_path}: time grid and contrast levels must match across sessions"
                )
            window = (time >= spec["averaged_window"][0]) & (time < spec["averaged_window"][1])
            temporal = (time >= spec["temporal_window"][0]) & (time <= spec["temporal_window"][1])
            if not window.any() or not temporal.any():
                raise ValueError(f"{session.relative_path}: no samples in the analysis window")
            values = raw.astype(np.float64)
            good = np.isfinite(values).all(axis=(0, 1))
            np.savez_compressed(
                staging / "responses" / f"s{sid:03d}.npz",
                X=values[window].mean(0),
                stim=stim,
                folds=folds,
            )
            np.save(staging / "temporal" / f"s{sid:03d}.npy", raw[temporal])
            np.save(staging / "decoding_folds" / f"s{sid:03d}.npy", folds)
            source = dict(
                sid=sid,
                fi=session.fi,
                group=session.group,
                timepoint=session.timepoint,
                mouse=session.mouse,
                file=session.relative_path,
            )
            sources.append(source)
            for ai, area in enumerate(AREAS):
                eligible = np.flatnonzero(good & np.isin(regions, REGIONS[area]))
                if len(eligible) < spec["n"]:
                    continue
                pid = len(populations)
                populations.append(
                    dict(**source, pid=pid, area=area, n=spec["n"], eligible=len(eligible))
                )
                rng = np.random.default_rng([spec["selection_seed"], sid, ai, spec["n"]])
                ids = np.array(
                    [rng.choice(eligible, spec["n"], replace=False) for _ in range(spec["draws"])]
                )
                np.save(staging / "selections" / f"p{pid:03d}.npy", ids)
            print(f"{signal} session {sid}: {session.relative_path}", flush=True)
        if not populations:
            raise ValueError("No session–area has ten finite, atlas-assigned neurons")
        np.save(staging / "temporal/time.npy", common_time[temporal])
        (staging / "manifest.json").write_text(
            json.dumps(
                dict(
                    signal=signal,
                    sources=sources,
                    populations=populations,
                    contrasts=contrasts.tolist(),
                    draws=spec["draws"],
                ),
                indent=2,
            )
            + "\n"
        )
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    print(f"Prepared {len(sources)} sessions and {len(populations)} populations in {destination}")
    return dict(sessions=len(sources), populations=len(populations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--signal", choices=["spikes", "calcium"], default=SIGNAL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or (DATA if args.signal == SIGNAL else DATA.parent / args.signal)
    prepare(args.input, output, args.signal, args.overwrite)


if __name__ == "__main__":
    main()
