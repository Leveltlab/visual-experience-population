import numpy as np
from analysis_config import OUT, LABELS, ids_for, mouse_average
from decoding.pooled import source

DEST = OUT / "direction_cosine_top2/supporting"


def centroids(p, f):
    x, di, folds = source(p)
    tr = np.array([x[(di == d) & (folds != f)].mean(0) for d in range(8)])
    te = np.array([x[(di == d) & (folds == f)].mean(0) for d in range(8)])
    # Common translation removed separately from both configurations for the shape view.
    tr -= tr.mean(0)
    te -= te.mean(0)
    ids = ids_for(p)
    return np.moveaxis(tr[:, ids], 1, 0), np.moveaxis(te[:, ids], 1, 0)


def orthogonal_map(a, target):
    u, _, vt = np.linalg.svd(a.swapaxes(-1, -2) @ target, full_matrices=False)
    return u @ vt


def embed(area, pops):
    # The target uses equal conditions and equal mice, and is fitted on training centroids only.
    sums = {p["pid"]: np.zeros((8, 10)) for p in pops}
    gram_sums = {p["pid"]: np.zeros((8, 8)) for p in pops}
    rotated_train = []
    rotation_error = 0.0
    anchor = np.random.default_rng(20261020).normal(size=(8, 10))
    anchor -= anchor.mean(0)
    for f in range(5):
        arrays = {p["pid"]: centroids(p, f) for p in pops}
        grams = {pid: np.mean(a @ a.swapaxes(-1, -2), 0) for pid, (a, b) in arrays.items()}
        group, _ = mouse_average(pops, grams)
        eig, vec = np.linalg.eigh(group.mean(0))
        target = np.pad(vec * np.sqrt(np.maximum(eig, 0)), ((0, 0), (0, 2)))
        target = target @ orthogonal_map(target, anchor)
        training = {}
        for p in pops:
            pid = p["pid"]
            a, b = arrays[pid]
            r = orthogonal_map(a, target)
            err = np.max(abs(r.swapaxes(-1, -2) @ r - np.eye(10)))
            rotation_error = max(rotation_error, float(err))
            assert err < 1e-10
            rb = b @ r
            np.testing.assert_allclose(
                rb @ rb.swapaxes(-1, -2), b @ b.swapaxes(-1, -2), atol=1e-10, rtol=1e-10
            )
            sums[pid] += rb.mean(0) / 5
            gram_sums[pid] += (b @ b.swapaxes(-1, -2)).mean(0) / 5
            training[pid] = (a @ r).mean(0)
        avg, _ = mouse_average(pops, training)
        rotated_train.append(avg)
        print(f"PCA {area} fold {f + 1}/5", flush=True)
    avg, mice = mouse_average(pops, sums)
    # Descriptive common PCA, not a decoding feature transform. Shared across all conditions.
    _, singular, vt = np.linalg.svd(avg.reshape(-1, 10), full_matrices=False)
    axes = vt[:3].T
    # Deterministic signs for display only.
    for col in range(3):
        if (avg[1] @ axes)[np.argmax(abs(avg[1] @ axes[:, col])), col] < 0:
            axes[:, col] *= -1
    coords = avg @ axes
    groupgrams, _ = mouse_average(pops, gram_sums)
    audits = []
    for fi in range(4):
        full = avg[fi] @ avg[fi].T
        projected = coords[fi, :, :2] @ coords[fi, :, :2].T

        # Projection retention of opposite separations, not just total variance.
        def distances(g):
            return np.diag(g)[:, None] + np.diag(g)[None, :] - 2 * g

        d = distances(full)
        d2 = distances(projected)
        dg = distances(groupgrams[fi])
        opposite = np.mean([d[i, (i + 4) % 8] for i in range(8)])
        orthogonal = np.mean([d[i, (i + 2) % 8] for i in range(8)])
        audits.append(
            dict(
                area=area,
                fi=fi,
                condition=LABELS[fi],
                retained_2d=float(np.sum(coords[fi, :, :2] ** 2) / np.sum(avg[fi] ** 2)),
                retained_3d=float(np.sum(coords[fi] ** 2) / np.sum(avg[fi] ** 2)),
                opposite_squared_full=float(opposite),
                orthogonal_squared_full=float(orthogonal),
                opposite_squared_2d=float(np.mean([d2[i, (i + 4) % 8] for i in range(8)])),
                unaveraged_opposite_squared=float(np.mean([dg[i, (i + 4) % 8] for i in range(8)])),
                unaveraged_orthogonal_squared=float(
                    np.mean([dg[i, (i + 2) % 8] for i in range(8)])
                ),
            )
        )
    np.savez_compressed(
        DEST / f"pca_{area}.npz",
        coordinates=coords,
        aligned_means=avg,
        axes=axes,
        explained=singular**2 / np.sum(singular**2),
        mouse_coordinates=np.array([m["values"] @ axes for m in mice]),
        mouse_ids=np.array([m["mouse"] for m in mice]),
        mouse_conditions=np.array([m["fi"] for m in mice]),
    )
    return audits, rotation_error
