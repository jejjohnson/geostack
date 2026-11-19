"""Algorithms over a fitted model."""

from __future__ import annotations

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float

from krigax._model import GaussianProcess


def predict(
    model: GaussianProcess, test_inputs: Float[Array, "test feature"]
) -> Float[Array, " test"]:
    """Predictive mean μ_* = K_*X α."""
    gram_test_train = model.kernel(test_inputs, model.train_inputs)
    return einx.dot("test sample, sample -> test", gram_test_train, model.weights)


def log_marginal_likelihood(model: GaussianProcess) -> Float[Array, ""]:
    """log p(y) = -½ yᵀ α - ½ log det(K_XX + σ² I) - (N/2) log 2π."""
    num = model.train_inputs.shape[0]
    gram = model.kernel(model.train_inputs, model.train_inputs)
    noisy = gram + model.noise * jnp.eye(num, dtype=gram.dtype)
    _, logdet = jnp.linalg.slogdet(noisy)
    fit_term = einx.dot("sample, sample ->", model.train_targets, model.weights)
    return -0.5 * fit_term - 0.5 * logdet - 0.5 * num * jnp.log(2.0 * jnp.pi)
