import numpy as np
from dimensionality_reduction.shared import register, choose_camera


def test_shared_registration_recovers_rotated_shapes_without_removing_scale():
    rng = np.random.default_rng(19)
    base = rng.normal(size=(4, 8, 10))
    base -= base.mean(1, keepdims=True)
    gains = np.array([1.0, 0.7, 1.3, 2.0])
    data = np.array([base @ np.linalg.qr(rng.normal(size=(10, 10)))[0] * gain for gain in gains])
    original = data.copy()
    aligned, coords, rotations, axes, explained, _ = register(data)
    np.testing.assert_array_equal(data, original)
    for i in range(4):
        np.testing.assert_allclose(aligned[i] / gains[i], aligned[0], atol=1e-10)
        # Includes distances between conditions, not just within each manifold.
        x = data[i].reshape(32, 10)
        y = aligned[i].reshape(32, 10)
        np.testing.assert_allclose(x @ x.T, y @ y.T, atol=1e-10)
    np.testing.assert_allclose(coords, aligned @ axes, atol=1e-12)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(explained.sum(), 1)


def test_one_camera_is_deterministic_and_distinguishes_four_corners():
    tetra = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1.0]])
    config = np.broadcast_to(np.concatenate([tetra, tetra]), (4, 4, 8, 3)).copy()
    a = choose_camera(config)
    assert a == choose_camera(config)
    assert a["minimum_standardized_corner_separation"] > 1
