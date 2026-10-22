import numpy as np
from analysis_config import DATA, OUT, AREAS, ids_for, manifest, mouse_average
from collections import defaultdict
from dimensionality_reduction.pooled import orthogonal_map

DEST = OUT / "contrast_scaling_pca"
DISPLAY_AREAS = ["AM-PM", "A-AL-RL"]


def direction_centers(x, stim, folds, fold):
    ci = np.searchsorted(np.unique(stim[:, 1]), stim[:, 1])
    di = (stim[:, 0] / 45).astype(int)
    result = []
    for train in [True, False]:
        mask = folds != fold if train else folds == fold
        c = np.array(
            [
                [x[mask & (ci // 4 == pool) & (di == d)].mean(0) for d in range(8)]
                for pool in range(2)
            ]
        )
        assert all(
            (mask & (ci // 4 == pool) & (di == d)).sum() == (48 if train else 12)
            for pool in range(2)
            for d in range(8)
        )
        c -= c.mean(-2, keepdims=True)
        result.append(c)
    return tuple(result)


def gram_target(group_grams):
    """D's empirical training Gram target extended to 16 labels in N10 space."""
    gram = np.mean(group_grams, 0)
    eig, vec = np.linalg.eigh(gram)
    # Best rank-10 embedding of the 16-label mean Gram. Only the target is
    # truncated; rotations preserve all original N10 centroid distances.
    target = vec[:, -10:] * np.sqrt(np.maximum(eig[-10:], 0))
    anchor = np.random.default_rng(20261020).normal(size=(16, 10))
    anchor = anchor.reshape(2, 8, 10)
    anchor -= anchor.mean(1, keepdims=True)
    anchor = anchor.reshape(16, 10)
    target = target @ orthogonal_map(target, anchor)
    return target, float(np.maximum(eig[-10:], 0).sum() / np.maximum(eig, 0).sum())


def register_all_areas(configurations):
    """D's rigid registration and labeled PC sign rule, with a pool axis."""
    a = np.asarray(configurations)
    assert a.shape[:2] == (4, 4) and a.shape[-2:] == (8, 10)
    flat = a.reshape(4, -1, 10)
    target = flat[0].copy()
    for iteration in range(200):
        rotations = orthogonal_map(flat, target)
        aligned = flat @ rotations
        new = aligned.mean(0)
        if np.linalg.norm(new - target) <= 1e-12 * max(np.linalg.norm(target), 1):
            break
        target = new
    rotations = orthogonal_map(flat, target)
    aligned = (flat @ rotations).reshape(a.shape)
    _, s, vt = np.linalg.svd(aligned.reshape(-1, 10), full_matrices=False)
    axes = vt[:3].T
    coords = aligned @ axes
    corners = ((coords[..., :4, :] + coords[..., 4:, :]) / 2).mean((0, 1, 2))
    comparisons = [
        corners[0, 0] - corners[2, 0],
        corners[1, 1] - corners[3, 1],
        corners[0, 2] - corners[1, 2],
    ]
    for k, value in enumerate(comparisons):
        if value < 0:
            axes[:, k] *= -1
    return aligned, aligned @ axes, rotations, axes, s * s / (s * s).sum(), iteration + 1


def run():
    DEST.mkdir(parents=True, exist_ok=True)
    pops = manifest()["populations"]
    by_session = defaultdict(list)
    centers = {}
    selections = {}

    for p in pops:
        by_session[p["sid"]].append(p)
    for sid, ps in by_session.items():
        path = DATA / "responses" / f"s{sid:03d}.npz"
        with np.load(path) as z:
            x = z["X"]
            stim = z["stim"]
        path = DATA / "decoding_folds" / f"s{sid:03d}.npy"
        folds = np.load(path)
        centers[sid] = [direction_centers(x, stim, folds, f) for f in range(5)]
        for p in ps:
            selections[p["pid"]] = ids_for(p)
    means = []
    orth_error = 0.0
    gram_error = 0.0
    for area in AREAS:
        ps = [p for p in pops if p["area"] == area]
        sums = {p["pid"]: np.zeros((16, 10)) for p in ps}
        gramsums = {p["pid"]: np.zeros((16, 16)) for p in ps}
        for f in range(5):
            arrays = {}
            for p in ps:
                c = centers[p["sid"]][f]
                ids = selections[p["pid"]]
                arrays[p["pid"]] = tuple(
                    np.transpose(v[:, :, ids], (2, 0, 1, 3)).reshape(1000, 16, 10) for v in c
                )
            grams = {pid: np.mean(a @ a.swapaxes(-1, -2), 0) for pid, (a, b) in arrays.items()}
            groups, _ = mouse_average(ps, grams)
            target, retained = gram_target(groups)
            for p in ps:
                pid = p["pid"]
                a, b = arrays[pid]
                r = orthogonal_map(a, target)
                rb = b @ r
                orth_error = max(orth_error, float(abs(r.swapaxes(-1, -2) @ r - np.eye(10)).max()))
                before = b @ b.swapaxes(-1, -2)
                after = rb @ rb.swapaxes(-1, -2)
                gram_error = max(gram_error, float(abs(after - before).max()))
                np.testing.assert_allclose(after, before, atol=1e-11, rtol=1e-11)
                sums[pid] += rb.mean(0) / 5
                gramsums[pid] += before.mean(0) / 5
            print(f"{area}: D-style fold {f + 1}/5 complete", flush=True)
        avg, mice = mouse_average(ps, sums)
        means.append(avg.reshape(4, 2, 8, 10))
    before = np.array(means)
    aligned, directions, rotations, axes, explained, iters = register_all_areas(before)
    flat = before.reshape(4, -1, 10)
    after = aligned.reshape(4, -1, 10)
    np.testing.assert_allclose(
        flat @ flat.swapaxes(-1, -2), after @ after.swapaxes(-1, -2), atol=1e-12
    )
    # Only now collapse antipodal directions for display.
    orientations = (directions[..., :4, :] + directions[..., 4:, :]) / 2
    displayed = [AREAS.index(a) for a in DISPLAY_AREAS]
    np.savez_compressed(DEST / "pca_geometry.npz", coordinates=orientations[displayed])
    print("Shared PCA retained variance:", float(explained[:3].sum()), flush=True)
