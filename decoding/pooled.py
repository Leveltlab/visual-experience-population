import numpy as np
from analysis_config import DATA, OUT, ids_for
import jax.numpy as jnp
from sklearn.covariance import ledoit_wolf
from decoding.lda import fit_diagnostics, pooled_scores, predict
from preprocess.folds import load_folds

DEST = OUT / "direction_cosine_top2/supporting"
TASKS = ["orientation", "direction"]


def source(p):
    with np.load(DATA / "responses" / f"s{p['sid']:03d}.npz") as z:
        keep = z["stim"][:, 1] >= np.unique(z["stim"][:, 1])[-2]
        x, stim = z["X"][keep], z["stim"][keep]
    folds = load_folds(p["sid"])[keep]
    labels = (stim[:, 0] / 45).astype(int)
    assert len(x) == 240
    for f in range(5):
        assert (folds == f).sum() == 48
        np.testing.assert_array_equal(np.bincount(labels[folds == f], minlength=8), 6)
        _, counts = np.unique(stim[folds == f], axis=0, return_counts=True)
        np.testing.assert_array_equal(counts, 3)
    return x, labels, folds


def decode(p):
    x, di, folds = source(p)
    ids = ids_for(p)
    correct = np.empty((1000, 2, 5))
    confusion = np.zeros((2, 8, 8))
    reference_error = 0.0
    for start in range(0, 1000, 100):
        xx = np.moveaxis(x[:, ids[start : start + 100]], 1, 0).astype("float64")
        for f in range(5):
            tr, te = folds != f, folds == f
            for ti, k in enumerate([4, 8]):
                y = di % k
                w, rank, alpha = fit_diagnostics(
                    jnp.asarray(xx[:, tr]),
                    jnp.asarray(np.broadcast_to(y[tr], (len(xx), tr.sum()))),
                    n_classes=k,
                )
                values, acc = pooled_scores(w, jnp.asarray(xx[:, te]), jnp.asarray(y[te]))
                correct[start : start + len(xx), ti, f] = np.asarray(acc)
                preds = np.asarray(predict(values))
                for a in range(k):
                    for b in range(k):
                        confusion[ti, a, b] += (preds[:, y[te] == a] == b).sum()
                # Independent sklearn covariance and direct score reference for every population/task.
                if start == 0 and f == 0:
                    train = xx[0, tr]
                    means = np.array([train[y[tr] == c].mean(0) for c in range(k)])
                    residual = train - means[y[tr]]
                    cov, alpha_ref = ledoit_wolf(residual, assume_centered=True)
                    eig, vec = np.linalg.eigh(cov)
                    cutoff = max(eig[-1], 0) * 100 * np.finfo(float).eps
                    inv = np.divide(1.0, eig, out=np.zeros_like(eig), where=eig > cutoff)
                    weights = means @ ((vec * inv) @ vec.T)
                    expected = (
                        xx[0, te] @ weights.T - 0.5 * (weights * means).sum(-1) + np.log(1 / k)
                    )
                    actual = np.asarray(values)[0]
                    np.testing.assert_allclose(alpha[0], alpha_ref, atol=1e-12)
                    np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
                    reference_error = max(reference_error, float(np.max(abs(actual - expected))))
    for ti, k in enumerate([4, 8]):
        confusion[ti, :k, :k] /= 1000 * (240 / k)
        np.testing.assert_allclose(confusion[ti, :k, :k].sum(-1), 1)
        np.testing.assert_allclose(
            np.trace(confusion[ti, :k, :k]) / k, correct[:, ti].mean(), atol=1e-7
        )
    return correct, confusion, reference_error
