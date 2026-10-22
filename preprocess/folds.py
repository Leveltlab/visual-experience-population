"""Five fixed, shuffled folds stratified by direction × contrast within session."""

import json
import numpy as np
from analysis_config import DATA, manifest, protocol, sha


def make_folds(stim, sid):
    _, cells = np.unique(stim, axis=0, return_inverse=True)
    folds = np.full(len(stim), -1, dtype=np.int8)
    for cell in range(64):
        trials = np.flatnonzero(cells == cell)
        assert len(trials) == 15
        rng = np.random.default_rng([protocol()["decoding_fold_seed"], int(sid), cell])
        folds[rng.permutation(trials)] = np.repeat(np.arange(5), 3)
    assert np.all(folds >= 0)
    return folds


def load_folds(sid):
    return np.load(DATA / "decoding_folds" / f"s{sid:03d}.npy")


def main():
    dest = DATA / "decoding_folds"
    dest.mkdir(exist_ok=True)
    hashes = {}
    for source in manifest()["sources"]:
        sid = source["sid"]
        with np.load(DATA / "responses" / f"s{sid:03d}.npz") as z:
            folds = make_folds(z["stim"], sid)
        path = dest / f"s{sid:03d}.npy"
        np.save(path, folds)
        hashes[path.name] = sha(path)
    (dest / "manifest.json").write_text(
        json.dumps(
            dict(
                folds=5,
                train_fraction=0.8,
                test_fraction=0.2,
                seed=protocol()["decoding_fold_seed"],
                trials_per_direction_contrast=dict(train=12, test=3),
                hashes=hashes,
            ),
            indent=2,
        )
    )
    print(f"Prepared five decoding folds for {len(hashes)} sessions; geometry folds unchanged")


if __name__ == "__main__":
    main()
