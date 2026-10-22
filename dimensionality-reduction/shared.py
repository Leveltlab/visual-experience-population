import json
import numpy as np
import pandas as pd
from analysis_config import OUT, AREAS, LABELS, sha
from dimensionality_reduction.pooled import orthogonal_map

DEST = OUT / "direction_cosine_top2/supporting"


def register(configurations):
    """Generalized orthogonal Procrustes, no translation/scale/label collapse."""
    a = np.asarray(configurations)
    assert a.shape == (4, 4, 8, 10)
    flat = a.reshape(4, 32, 10)
    target = flat[0].copy()  # Initial gauge only; all areas enter every target update.
    for iteration in range(200):
        rotations = orthogonal_map(flat, target)
        aligned = flat @ rotations
        new = aligned.mean(0)
        if np.linalg.norm(new - target) <= 1e-12 * max(np.linalg.norm(target), 1):
            break
        target = new
    rotations = orthogonal_map(flat, target)
    aligned = (flat @ rotations).reshape(a.shape)
    _, singular, vt = np.linalg.svd(aligned.reshape(-1, 10), full_matrices=False)
    axes = vt[:3].T
    coordinates = aligned @ axes
    # Stable signs, tied to labeled orientation corners, not arbitrary eigenvector signs.
    corners = (coordinates[:, :, :4] + coordinates[:, :, 4:]).mean((0, 1)) / 2
    comparisons = [
        corners[0, 0] - corners[2, 0],
        corners[1, 1] - corners[3, 1],
        corners[0, 2] - corners[1, 2],
    ]
    for k, value in enumerate(comparisons):
        if value < 0:
            axes[:, k] *= -1
    return (
        aligned,
        aligned @ axes,
        rotations,
        axes,
        singular**2 / np.sum(singular**2),
        iteration + 1,
    )


def choose_camera(coordinates):
    """One orthographic view chosen to separate the four labeled corners in every area."""
    corners = (coordinates[:, :, :4] + coordinates[:, :, 4:]).mean(1) / 2
    differences = np.stack([corners[:, i] - corners[:, j] for i in range(4) for j in range(i)], 1)
    rms = np.sqrt(np.mean(np.sum(corners**2, -1), -1))
    best = (-np.inf, 25, 35)
    for elev in range(0, 91, 5):
        for azim in range(-180, 180, 5):
            e, a = np.deg2rad([elev, azim])
            basis = np.array(
                [
                    [-np.sin(a), np.cos(a), 0],
                    [-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)],
                ]
            )
            distances = np.linalg.norm(differences @ basis.T, axis=-1) / rms[:, None]
            score = distances.min()
            if score > best[0]:
                best = (score, elev, azim)
    return dict(
        elevation=best[1], azimuth=best[2], minimum_standardized_corner_separation=float(best[0])
    )


def build():
    configurations = []
    hashes = {}
    for area in AREAS:
        path = DEST / f"pca_{area}.npz"
        hashes[area] = sha(path)
        with np.load(path) as z:
            configurations.append(z["aligned_means"])
    configurations = np.array(configurations)
    aligned, coords, rotations, axes, explained, iterations = register(configurations)
    before = configurations.reshape(4, 32, 10)
    after = aligned.reshape(4, 32, 10)
    error = float(np.max(abs(before @ before.swapaxes(-1, -2) - after @ after.swapaxes(-1, -2))))
    np.testing.assert_allclose(
        before @ before.swapaxes(-1, -2), after @ after.swapaxes(-1, -2), atol=1e-12, rtol=1e-12
    )
    np.testing.assert_allclose(
        rotations.swapaxes(-1, -2) @ rotations,
        np.broadcast_to(np.eye(10), rotations.shape),
        atol=1e-12,
    )
    np.testing.assert_allclose(coords.mean(2), 0, atol=1e-12)
    camera = choose_camera(coords)
    rows = []
    coordinate_rows = []
    for ai, area in enumerate(AREAS):
        for fi, label in enumerate(LABELS):
            v = aligned[ai, fi]
            p = coords[ai, fi]

            def separation(x, offset):
                return float(np.mean(np.sum((x - np.roll(x, -offset, axis=0)) ** 2, axis=-1)))

            rows.append(
                dict(
                    area=area,
                    fi=fi,
                    condition=label,
                    retained_2d=np.sum(p[:, :2] ** 2) / np.sum(v**2),
                    retained_3d=np.sum(p**2) / np.sum(v**2),
                    opposite_squared_full=separation(v, 4),
                    opposite_squared_2d=separation(p[:, :2], 4),
                    opposite_squared_3d=separation(p, 4),
                    orthogonal_squared_full=separation(v, 2),
                )
            )
            for d in range(8):
                coordinate_rows.append(
                    dict(
                        area=area,
                        fi=fi,
                        condition=label,
                        direction=d * 45,
                        **{f"PC{k + 1}": float(p[d, k]) for k in range(3)},
                    )
                )
    pd.DataFrame(rows).to_csv(DEST / "shared_pca_audit.csv", index=False)
    pd.DataFrame(coordinate_rows).to_csv(DEST / "pca_group_coordinates.csv", index=False)
    np.savez_compressed(
        DEST / "shared_pca.npz",
        coordinates=coords,
        aligned_means=aligned,
        rotations=rotations,
        axes=axes,
        explained=explained,
    )
    validation = dict(
        validated=True,
        source=sha(__file__),
        inputs=hashes,
        iterations=iterations,
        gram_max_error=error,
        common_pca_retained_2d=float(explained[:2].sum()),
        common_pca_retained_3d=float(explained[:3].sum()),
        camera=camera,
        registration="One orthogonal map per area applied jointly to all 32 condition-direction centroids. Iterative target is an equal-area mean. No scale fitting or relabeling. All within-area pair distances, including between conditions, are preserved before projection.",
        pca="One SVD across all 128 registered centroids, with equal rows/areas/conditions and no variance normalization. These are shared descriptive axes, not anatomically matched neuron axes.",
        display="Common axes and orthographic camera in all panels. Fill-panel views use area-specific zoom stated in each panel; equal-scale companion uses identical limits and zoom.",
        caution="Registration fitted to condition means is descriptive. Alignment and averaging can hide individual shape differences; identical axes do not provide an independent statistical test of conserved shape.",
    )
    (DEST / "shared_pca_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return coords, explained, camera, validation
