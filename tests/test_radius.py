import numpy as np
from geometry.radius import geometry


def test_radius_is_full_population_norm_and_rotation_invariant():
    rng = np.random.default_rng(30)
    x = rng.normal(size=(8, 10))
    ids = np.arange(10)[None]
    radius, fraction = geometry(x, ids)
    expected = np.mean(np.sum((x - x.mean(0)) ** 2, axis=1))
    np.testing.assert_allclose(radius, [expected])
    q, _ = np.linalg.qr(rng.normal(size=(10, 10)))
    r2, f2 = geometry(x @ q + 17, ids)
    np.testing.assert_allclose(r2, radius)
    np.testing.assert_allclose(f2, fraction)
    r3, _ = geometry(2 * x, ids)
    np.testing.assert_allclose(r3, 4 * radius)


def test_four_repeated_orientation_corners_have_rank_at_most_three():
    rng = np.random.default_rng(8)
    corners = rng.normal(size=(4, 10))
    _, fraction = geometry(np.vstack([corners, corners]), np.arange(10)[None])
    np.testing.assert_allclose(fraction, 1, atol=1e-12)
