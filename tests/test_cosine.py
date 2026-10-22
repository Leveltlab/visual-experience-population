"""Tests for the changed scientific estimand: pooled centroids and angular bins."""

import numpy as np
from scipy.spatial.distance import cosine

from geometry.cosine import ANGLES, angular_profile, centered_cosines, pooled_centroids


def test_direction_and_orientation_ring_profiles():
    q, _ = np.linalg.qr(np.random.default_rng(9).normal(size=(10, 2)))
    for harmonic in [1, 2]:
        phase = harmonic * np.deg2rad(ANGLES)
        v = np.stack([np.cos(phase), np.sin(phase)], -1) @ q.T
        matrix, valid = centered_cosines(v[None])
        assert valid.all()
        np.testing.assert_allclose(angular_profile(matrix)[0], np.cos(phase), atol=1e-14)


def test_all_pairs_and_mathematical_reflection():
    rng = np.random.default_rng(35)
    vectors = rng.normal(size=(4, 8, 10))
    matrix, valid = centered_cosines(vectors)
    centered = vectors - vectors.mean(1, keepdims=True)
    profile = angular_profile(matrix)
    counts = np.zeros((8, 8), int)
    for k in range(8):
        pairs = [(i, (i + k) % 8) for i in range(8)]
        for i, j in pairs:
            counts[i, j] += 1
        expected = [np.mean([1 - cosine(v[i], v[j]) for i, j in pairs]) for v in centered]
        np.testing.assert_allclose(profile[:, k], expected, atol=1e-14)
    np.testing.assert_array_equal(counts, 1)
    assert valid.all()
    for k in [1, 2, 3]:
        np.testing.assert_allclose(profile[:, k], profile[:, 8 - k], atol=1e-14)
    # Random responses satisfy reflection without having orientation periodicity.
    assert np.max(abs(profile[:, 0] - profile[:, 4])) > 0.5


def test_pool_trials_before_cosine():
    rng = np.random.default_rng(52)
    stim = np.array([(d, c) for d in ANGLES for c in [0.1, 0.5179, 1.0] for _ in range(15)])
    x = rng.normal(size=(len(stim), 10))
    x[stim[:, 1] == 0.5179] *= 10
    pooled, separate = pooled_centroids(x, stim)
    for i, d in enumerate(ANGLES):
        expected = x[(stim[:, 0] == d) & (stim[:, 1] >= 0.5)].mean(0)
        np.testing.assert_allclose(pooled[i], expected, atol=1e-14)
    correct = centered_cosines(pooled[None])[0]
    wrong = np.mean([centered_cosines(a[None])[0] for a in separate], axis=0)
    assert np.max(abs(correct - wrong)) > 0.1


def test_zero_vectors_exclude_whole_draw_and_common_origin_is_removed():
    v = np.random.default_rng(2).normal(size=(3, 8, 10))
    v[0] = 0
    a, valid = centered_cosines(v)
    assert not valid[0] and valid[1:].all()
    assert np.isnan(a[0]).all()
    b, valid2 = centered_cosines(v + 100)
    np.testing.assert_array_equal(valid, valid2)
    np.testing.assert_allclose(a, b, atol=1e-12, equal_nan=True)
