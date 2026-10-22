"""Checks for D-compatible all-direction registration and PCA-only scope."""

import numpy as np
from dimensionality_reduction.contrast import direction_centers, register_all_areas, gram_target


def test_centering_and_pooling_preserve_eight_direction_labels():
    rng = np.random.default_rng(38)
    stim = np.array([(d * 45, c) for d in range(8) for c in range(8) for trial in range(15)])
    folds = np.tile(np.repeat(np.arange(5), 3), 64)
    x = rng.normal(size=(960, 10))
    tr, te = direction_centers(x, stim, folds, 0)
    assert tr.shape == te.shape == (2, 8, 10)
    for train, c in [(True, tr), (False, te)]:
        mask = folds != 0 if train else folds == 0
        expected = np.array(
            [
                [
                    x[mask & (stim[:, 1] // 4 == p) & (stim[:, 0] == d * 45)].mean(0)
                    for d in range(8)
                ]
                for p in range(2)
            ]
        )
        expected -= expected.mean(1, keepdims=True)
        np.testing.assert_allclose(c, expected)
        np.testing.assert_allclose(c.mean(1), 0, atol=1e-15)
        assert np.max(abs(c[:, :4] - c[:, 4:])) > 0.01
    changed = x.copy()
    changed[folds == 0] += rng.normal(size=(192, 10)) * 5
    tr2, te2 = direction_centers(changed, stim, folds, 0)
    np.testing.assert_array_equal(tr, tr2)
    assert np.max(abs(te - te2)) > 0.1


def test_generalized_registration_reproduces_D_for_one_pool():
    original = np.random.default_rng(71).normal(size=(4, 4, 8, 10))
    original -= original.mean(2, keepdims=True)
    a, c, r, axes, e, it = register_all_areas(original[:, :, None])
    from dimensionality_reduction.shared import register

    da, dc, dr, daxes, de, _ = register(original)
    np.testing.assert_allclose(a[:, :, 0], da, atol=1e-12)
    np.testing.assert_allclose(c[:, :, 0], dc, atol=1e-12)
    np.testing.assert_allclose(e, de, atol=1e-12)


def test_joint_training_target_and_rigid_registration():
    rng = np.random.default_rng(39)
    a = rng.normal(size=(4, 4, 2, 8, 10))
    a -= a.mean(-2, keepdims=True)
    flat = a.reshape(4, 4, 16, 10)
    grams = np.mean(flat @ flat.swapaxes(-1, -2), 0)
    target, ret = gram_target(grams)
    eig = np.linalg.eigvalsh(grams.mean(0))
    np.testing.assert_allclose(ret, eig[-10:].sum() / eig.sum())
    assert target.shape == (16, 10)
    aligned, c, r, axes, e, it = register_all_areas(a)
    for before, after in zip(a, aligned):
        b = before.reshape(-1, 10)
        v = after.reshape(-1, 10)
        np.testing.assert_allclose(b @ b.T, v @ v.T, atol=1e-11)
    np.testing.assert_allclose(c, aligned @ axes)
    np.testing.assert_allclose(axes.T @ axes, np.eye(3), atol=1e-12)
