import numpy as np
from geometry.cross_contrast import compare_vectors, pool_vectors, split_high_vectors, ANGLES


def test_gain_and_offset_preserve_cosine():
    base = np.random.default_rng(37).normal(size=(2, 8, 10))
    gains = np.array([0.1, 0.3, 0.7, 1.0])
    vectors = base[:, None] * gains[None, :, None, None] + np.arange(4)[None, :, None, None] * 100
    c = compare_vectors(vectors, base + 77)
    np.testing.assert_allclose(c, 1, atol=1e-12)


def test_zero_vector_cosine_undefined():
    x = np.random.default_rng(3).normal(size=(1, 4, 8, 10))
    x[:, 0] = 0
    c = compare_vectors(x, x[:, 3])
    assert np.isnan(c[:, 0]).all()


def test_disjoint_pool_membership_and_saved_neuron_order():
    contrasts = np.geomspace(0.01, 1, 8)
    stim = np.array([(d, c) for d in ANGLES for c in contrasts for _ in range(15)])
    x = np.random.default_rng(4).normal(size=(960, 12))
    ids = np.broadcast_to(np.arange(10)[::-1], (1000, 10))
    v = pool_vectors(x, stim, ids)
    for k in range(4):
        for i, angle in enumerate(ANGLES):
            mask = (stim[:, 0] == angle) & np.isin(stim[:, 1], contrasts[2 * k : 2 * k + 2])
            assert mask.sum() == 30
            np.testing.assert_allclose(v[17, k, i], x[mask].mean(0)[ids[17]], atol=1e-14)


def test_split_is_disjoint_reproducible_and_varies_across_draws():
    contrasts = np.geomspace(0.01, 1, 8)
    stim = np.array([(d, c) for d in ANGLES for c in contrasts for _ in range(15)])
    x = np.random.default_rng(4).normal(size=(960, 12))
    ids = np.broadcast_to(np.arange(10)[::-1], (1000, 10))
    a, b, ar, br = split_high_vectors(x, stim, ids, 7)
    for actual, repeat in zip((a, b, ar, br), split_high_vectors(x, stim, ids, 7)):
        np.testing.assert_array_equal(actual, repeat)
    assert ar.shape == br.shape == (1000, 8, 15)
    assert not np.array_equal(np.sort(ar[0], axis=-1), np.sort(ar[1], axis=-1))
    for d, angle in enumerate(ANGLES):
        expected = np.flatnonzero((stim[:, 0] == angle) & np.isin(stim[:, 1], contrasts[-2:]))
        np.testing.assert_array_equal(
            np.sort(np.concatenate([ar[:, d], br[:, d]], -1)), np.broadcast_to(expected, (1000, 30))
        )
    np.testing.assert_allclose((a + b) / 2, pool_vectors(x, stim, ids)[:, 3], atol=1e-14)
    np.testing.assert_allclose(a[13, 4], x[ar[13, 4]][:, ids[13]].mean(0))
    c = compare_vectors(b[:, None], a)
    assert c.shape == (1000, 1, 8)
    assert np.isfinite(c).all() and c.mean() < 0.2
