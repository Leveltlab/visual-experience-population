"""LDA with analytic Ledoit-Wolf shrinkage in raw deconvolved coordinates.

Pool training residuals after subtracting their training class means. Estimate
Ledoit-Wolf shrinkage toward trace(S)/p * I separately for every fitted model.
This matches sklearn.covariance.ledoit_wolf(residual, assume_centered=True),
not sklearn LDA's classwise, internally standardized shrinkage='auto'.
"""

from functools import partial
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.jit
def shrink_covariance(covariance, squared_residual_moment, n_samples):
    """LW from S=E[rr'] and Q=E[(r*r)(r*r)']; arbitrary batch dimensions.

    Select a neuron's S and Q submatrices BEFORE computing shrinkage. A
    full-population shrinkage coefficient is not valid for a selected subset.
    """
    p = covariance.shape[-1]
    mu = jnp.trace(covariance, axis1=-2, axis2=-1) / p
    norm2 = jnp.sum(covariance * covariance, axis=(-2, -1))
    delta = jnp.maximum(norm2 - p * mu * mu, 0)
    beta = jnp.maximum((jnp.sum(squared_residual_moment, axis=(-2, -1)) - norm2) / n_samples, 0)
    alpha = jnp.where(
        delta > 0,
        jnp.minimum(beta, delta) / jnp.maximum(delta, jnp.finfo(covariance.dtype).tiny),
        0,
    )
    shrunk = (1 - alpha[..., None, None]) * covariance + (alpha * mu)[..., None, None] * jnp.eye(
        p, dtype=covariance.dtype
    )
    return shrunk, alpha


@jax.jit
def weights_from_covariance(means, covariance, priors):
    # Numerical fallback remains necessary for completely silent training data.
    eigen, vectors = jnp.linalg.eigh(covariance)
    cutoff = jnp.maximum(eigen[..., -1:], 0) * (
        10 * covariance.shape[-1] * jnp.finfo(covariance.dtype).eps
    )
    inverse = jnp.where(eigen > cutoff, 1 / jnp.maximum(eigen, jnp.finfo(covariance.dtype).tiny), 0)
    precision = (vectors * inverse[..., None, :]) @ vectors.swapaxes(-1, -2)
    weights = means @ precision
    bias = -0.5 * jnp.sum(weights * means, -1) + jnp.log(priors)
    rank = (eigen > cutoff).sum(-1)
    return jnp.concatenate([weights, bias[..., None]], -1), rank


@partial(jax.jit, static_argnames=("n_classes",))
def fit_diagnostics(train, labels, n_classes=4):
    onehot = jax.nn.one_hot(labels, n_classes, dtype=train.dtype)
    counts = onehot.sum(1)
    means = jnp.einsum("bsk,bsn->bkn", onehot, train) / counts[..., None]
    residual = train - jnp.einsum("bsk,bkn->bsn", onehot, means)
    covariance = jnp.einsum("bsn,bsm->bnm", residual, residual) / train.shape[1]
    square = residual * residual
    fourth = jnp.einsum("bsn,bsm->bnm", square, square) / train.shape[1]
    shrunk, alpha = shrink_covariance(covariance, fourth, train.shape[1])
    weights, rank = weights_from_covariance(means, shrunk, counts / train.shape[1])
    return weights, rank, alpha


@partial(jax.jit, static_argnames=("n_classes",))
def fit(train, labels, n_classes=4):
    weights, rank, _ = fit_diagnostics(train, labels, n_classes)
    return weights, rank


@jax.jit
def predict(scores):
    """Smallest class index among scores tied within float64 numerical tolerance.

    Sparse/rank-deficient LDA can yield mathematically identical class scores.
    Reduction-order roundoff must not determine the assigned class.
    """
    tolerance = 1e-10 * jnp.maximum(jnp.max(jnp.abs(scores), axis=-1, keepdims=True), 1.0)
    return jnp.argmax(scores >= scores.max(axis=-1, keepdims=True) - tolerance, axis=-1)


@jax.jit
def pooled_scores(w, test, labels):
    values = jnp.einsum("dkn,dqn->dqk", w[..., :-1], test) + w[..., -1][:, None, :]
    return values, (predict(values) == labels[None]).mean(-1)
