"""MATLAB ingestion is tested using fresh files containing only required fields."""

from pathlib import Path
import h5py
import numpy as np
import pytest
from preprocess.matlab2py import CONDITIONS, discover_sessions, load_session
from preprocess.prepare import prepare
from preprocess.folds import make_folds


def write_recording(root, folder, mouse, seed=7, regions=None):
    rng = np.random.default_rng(seed)
    time = np.linspace(-1, 1.5, 41, dtype=np.float32)
    stim = np.array(
        [(d * 45, c) for d in range(8) for c in np.geomspace(0.01, 1, 8) for _ in range(15)],
        dtype=np.float32,
    )
    regions = np.repeat([7, 1, 3, 6], 12) if regions is None else np.asarray(regions)
    direction = np.deg2rad(stim[:, 0])
    preferred = rng.uniform(0, np.pi, len(regions))
    tuning = (1 + np.cos(2 * (direction[:, None] - preferred))) * stim[:, 1, None]
    gain = rng.uniform(0.003, 0.012, len(regions))
    response = (time[:, None, None] >= 0) * (time[:, None, None] < 0.6) * tuning[None] * gain
    x = np.maximum(
        0, rng.normal(0.03, 0.03, size=(len(time), 960, len(regions))) + response
    ).astype(np.float32)
    path = Path(root) / folder / f"{mouse}_session.mat"
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        for name, a in [
            ("Res/CaDeconCorrected", x),
            ("Res/CaSigCorrected", x * 2),
            ("Res/ax", time),
            ("info/Stim/log", stim),
            ("ABAroi/region", regions),
        ]:
            f.create_dataset(name, data=a.T)
    return path, x, time, stim


def test_discovery_order_and_minimal_raw_fields(tmp_path):
    for fi, (folder, _, _) in reversed(list(enumerate(CONDITIONS))):
        for mouse in ["Z", "A"]:
            write_recording(tmp_path, folder, mouse, fi)
    sessions = discover_sessions(tmp_path)
    assert [(s.fi, s.mouse) for s in sessions] == [(fi, m) for fi in range(4) for m in ["A", "Z"]]
    x, time, stim, region = load_session(sessions[0])
    assert x.shape == (41, 960, 48) and stim.shape == (960, 2)
    np.testing.assert_array_equal(load_session(sessions[0], "calcium")[0], x * 2)
    assert set(region) == {0, 2, 5, 6}


def test_prepare_from_recordings_preserves_values_and_sampling(tmp_path):
    raw = tmp_path / "raw"
    folder = CONDITIONS[0][0]
    _, x, time, stim = write_recording(raw, folder, "Mouse")
    output = tmp_path / "prepared"
    result = prepare(raw, output)
    assert result == dict(sessions=1, populations=4)
    with np.load(output / "responses/s000.npz") as z:
        np.testing.assert_array_equal(z["X"], x.astype(float)[(time >= 0) & (time < 0.5)].mean(0))
        np.testing.assert_array_equal(z["stim"], stim)
    np.testing.assert_array_equal(np.load(output / "decoding_folds/s000.npy"), make_folds(stim, 0))
    ids = np.load(output / "selections/p000.npy")
    expected_rng = np.random.default_rng([20261018, 0, 0, 10])
    np.testing.assert_array_equal(
        ids, np.array([expected_rng.choice(np.arange(12), 10, replace=False) for _ in range(1000)])
    )
    with pytest.raises(FileExistsError):
        prepare(raw, output)
    prepare(raw, tmp_path / "calcium", signal="calcium")
    with (
        np.load(tmp_path / "calcium/responses/s000.npz") as c,
        np.load(output / "responses/s000.npz") as s,
    ):
        np.testing.assert_allclose(c["X"], 2 * s["X"], rtol=0, atol=0)


def test_bad_stimulus_and_missing_input_are_actionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="No .mat"):
        discover_sessions(tmp_path)
    with pytest.raises(ValueError, match="960 trials"):
        make_folds(np.zeros((12, 2)), 0)
    path, _, _, _ = write_recording(tmp_path, CONDITIONS[0][0], "M")
    with h5py.File(path, "a") as f:
        del f["Res/CaDeconCorrected"]
    with pytest.raises(ValueError, match="CaDeconCorrected"):
        load_session(discover_sessions(tmp_path)[0])


def test_failed_overwrite_keeps_prepared_data(tmp_path):
    raw = tmp_path / "raw"
    write_recording(raw, CONDITIONS[0][0], "M")
    dest = tmp_path / "prepared"
    prepare(raw, dest)
    expected = (dest / "manifest.json").read_bytes()
    path = next(raw.rglob("*.mat"))
    with h5py.File(path, "a") as f:
        del f["Res/ax"]
    with pytest.raises(ValueError):
        prepare(raw, dest, overwrite=True)
    assert (dest / "manifest.json").read_bytes() == expected
