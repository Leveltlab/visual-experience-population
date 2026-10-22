from itertools import product
import numpy as np
from analysis_config import bootstrap_weights


def finite_mean(x, axis):
    """NaN only when no valid estimates; do not substitute zero cosines."""
    count = np.isfinite(x).sum(axis=axis)
    return np.divide(
        np.nansum(x, axis=axis),
        count,
        out=np.full(np.shape(count), np.nan, dtype=float),
        where=count > 0,
    )


def unit_vectors(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return np.divide(x, norm, out=np.full_like(x, np.nan), where=norm > 0)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(np.where(np.isfinite(p), p, np.inf))
    q = np.empty_like(p)
    q[order] = np.minimum(
        1,
        np.maximum.accumulate(
            np.where(np.isfinite(p[order]), p[order], 1) * np.arange(len(p), 0, -1)
        ),
    )
    return np.where(np.isfinite(p), q, np.nan)


def signflip_p(x):
    signs = np.array(list(product([-1, 1], repeat=len(x))))
    return float(np.mean(abs(signs @ x / len(x)) >= abs(x.mean()) - 1e-14))


def interval(x, fi):
    if len(x) < 2 or not np.isfinite(x).all():
        return np.nan, np.nan
    return np.quantile(bootstrap_weights(len(x), fi) @ x, [0.025, 0.975])
