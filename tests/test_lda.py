import numpy as np
from sklearn.covariance import ledoit_wolf
from decoding.lda import fit_diagnostics, predict
from dimensionality_reduction import pooled


def test_lda_scores_and_shrinkage_match_independent_sklearn_reference():
    rng = np.random.default_rng(32)
    y = np.repeat(np.arange(4), 12)
    x = rng.normal(size=(48, 10)) + np.eye(10)[y]
    w, rank, alpha = fit_diagnostics(x[None], y[None], n_classes=4)
    means = np.array([x[y == k].mean(0) for k in range(4)])
    cov, expected_alpha = ledoit_wolf(x - means[y], assume_centered=True)
    expected = means @ np.linalg.inv(cov)
    np.testing.assert_allclose(w[0, :, :-1], expected, atol=1e-12)
    np.testing.assert_allclose(
        w[0, :, -1], -0.5 * (expected * means).sum(1) + np.log(0.25), atol=1e-12
    )
    np.testing.assert_allclose(alpha, [expected_alpha], atol=1e-12)
    assert int(rank[0]) == 10


def test_silent_data_ties_choose_smallest_class():
    y = np.repeat(np.arange(4), 12)
    w, rank, alpha = fit_diagnostics(np.zeros((1, 48, 10)), y[None], n_classes=4)
    assert np.isfinite(w).all() and int(rank[0]) == 0
    np.testing.assert_array_equal(predict(w[0, :, -1]), 0)


def test_training_centroids_do_not_use_heldout_trials(monkeypatch):
    rng = np.random.default_rng(3)
    di = np.repeat(np.arange(8), 30)
    folds = np.tile(np.repeat(np.arange(5), 6), 8)
    x = rng.normal(size=(240, 12))
    ids = np.broadcast_to(np.arange(10), (1000, 10))
    monkeypatch.setattr(pooled, "source", lambda p: (x, di, folds))
    monkeypatch.setattr(pooled, "ids_for", lambda p: ids)
    a, b = pooled.centroids({}, 0)
    x[folds == 0] += rng.normal(size=x[folds == 0].shape) * 100
    aa, bb = pooled.centroids({}, 0)
    np.testing.assert_array_equal(a, aa)
    assert not np.allclose(b, bb)
