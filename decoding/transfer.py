import numpy as np
from analysis_config import DATA, OUT, ids_for, sha
from functools import lru_cache
from preprocess.folds import load_folds, make_folds
from decoding.lda import fit_diagnostics, pooled_scores
import jax.numpy as jnp
from sklearn.covariance import ledoit_wolf

DEST = OUT / "pooled_radius_transfer"


@lru_cache(maxsize=8)
def source(sid):
    with np.load(DATA / "responses" / f"s{sid:03d}.npz") as z:
        x, stim = z["X"].astype("float64"), z["stim"]
    folds = load_folds(sid)
    np.testing.assert_array_equal(folds, make_folds(stim, sid))
    assert x.shape[0] == 960 and np.isfinite(x).all()
    contrasts = np.unique(stim[:, 1])
    assert len(contrasts) == 8
    bins = np.searchsorted(contrasts, stim[:, 1]) // 2
    labels = (stim[:, 0] / 45).astype(int) % 4
    for b in range(4):
        assert (bins == b).sum() == 240
        for f in range(5):
            keep = (bins == b) & (folds == f)
            np.testing.assert_array_equal(np.bincount(labels[keep]), np.full(4, 12))
            _, counts = np.unique(stim[keep], axis=0, return_counts=True)
            np.testing.assert_array_equal(counts, np.full(16, 3))
    return x, stim, folds, bins, labels


def input_signature(p):
    return dict(
        response=sha(DATA / "responses" / f"s{p['sid']:03d}.npz"),
        selection=sha(DATA / "selections" / f"p{p['pid']:03d}.npy"),
        folds=sha(DATA / "decoding_folds" / f"s{p['sid']:03d}.npy"),
    )


def compute(p):
    x, stim, folds, bins, y = source(p["sid"])
    ids = ids_for(p)
    transfer = np.empty((1000, 4, 4, 5))
    score_error = 0.0
    for f in range(5):
        for s in range(4):
            tr = (bins == s) & (folds != f)
            assert tr.sum() == 192
            for start in range(0, 1000, 250):
                subset = ids[start : start + 250]
                xx = np.moveaxis(x[:, subset], 1, 0)
                w, rank, alpha = fit_diagnostics(
                    jnp.asarray(xx[:, tr]),
                    jnp.asarray(np.broadcast_to(y[tr], (len(xx), tr.sum()))),
                    n_classes=4,
                )
                for target in range(4):
                    te = (bins == target) & (folds == f)
                    assert te.sum() == 48 and not np.any(tr & te)
                    scores, acc = pooled_scores(w, jnp.asarray(xx[:, te]), jnp.asarray(y[te]))
                    transfer[start : start + len(xx), s, target, f] = np.asarray(acc)
                    if start == 0:
                        train_x = xx[0, tr]
                        means = np.array([train_x[y[tr] == c].mean(0) for c in range(4)])
                        cov, refalpha = ledoit_wolf(train_x - means[y[tr]], assume_centered=True)
                        ev, vec = np.linalg.eigh(cov)
                        inverse = np.divide(
                            1.0,
                            ev,
                            out=np.zeros_like(ev),
                            where=ev > max(ev[-1], 0) * 100 * np.finfo(float).eps,
                        )
                        weights = means @ ((vec * inverse) @ vec.T)
                        expected = (
                            xx[0, te] @ weights.T - 0.5 * (weights * means).sum(-1) + np.log(0.25)
                        )
                        actual = np.asarray(scores)[0]
                        score_error = max(score_error, float(abs(expected - actual).max()))
                        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
                        np.testing.assert_allclose(alpha[0], refalpha, atol=1e-12)
    return dict(transfer=transfer, score_error=score_error)
