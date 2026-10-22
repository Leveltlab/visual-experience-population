import json
import numpy as np
from analysis_config import OUT, AREAS
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
    for area in AREAS:
        path = DEST / f"pca_{area}.npz"
        with np.load(path) as z:
            configurations.append(z["aligned_means"])
    configurations = np.array(configurations)
    aligned, coords, rotations, axes, explained, iterations = register(configurations)
    before = configurations.reshape(4, 32, 10)
    after = aligned.reshape(4, 32, 10)
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
    np.savez_compressed(DEST / "shared_pca.npz", coordinates=coords)
    (OUT / "camera.json").write_text(
        json.dumps({k: camera[k] for k in ["elevation", "azimuth"]}, indent=2) + "\n"
    )
    return coords, explained, camera
