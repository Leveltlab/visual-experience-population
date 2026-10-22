"""Five seeded folds stratified by all eight directions and eight contrasts."""

import numpy as np
from analysis_config import DATA, protocol


def make_folds(stim, sid):
    if (
        stim.shape != (960, 2)
        or not np.array_equal(np.unique(stim[:, 0]), np.arange(8) * 45)
        or len(np.unique(stim[:, 1])) != 8
    ):
        raise ValueError(
            "Expected 960 trials: eight directions (0,45,...,315°), eight contrasts, 15 repeats per cell"
        )
    _, cells = np.unique(stim, axis=0, return_inverse=True)
    if not np.array_equal(np.bincount(cells), np.full(64, 15)):
        raise ValueError("Every direction × contrast cell must have exactly 15 trials")
    folds = np.full(len(stim), -1, dtype=np.int8)
    for cell in range(64):
        trials = np.flatnonzero(cells == cell)
        rng = np.random.default_rng([protocol()["decoding_fold_seed"], int(sid), cell])
        folds[rng.permutation(trials)] = np.repeat(np.arange(5), 3)
    return folds


def load_folds(sid):
    return np.load(DATA / "decoding_folds" / f"s{sid:03d}.npy")
