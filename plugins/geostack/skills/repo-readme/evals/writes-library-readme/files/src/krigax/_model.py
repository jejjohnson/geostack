"""The model: a fitted Gaussian process and the function that fits it."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float

from krigax._kernels import RBF, Matern32
from krigax._solvers import CGSolver, CholeskySolver


class GaussianProcess(eqx.Module):
    """A GP conditioned on training data.

    Holds the kernel, the training inputs and the weights
    α = (K_XX + σ² I)⁻¹ y, so prediction is one kernel matrix and one product.
    """

    kernel: RBF | Matern32
    train_inputs: Float[Array, "sample feature"]
    train_targets: Float[Array, " sample"]
    weights: Float[Array, " sample"]
    noise: float


def fit(
    kernel: RBF | Matern32,
    train_inputs: Float[Array, "sample feature"],
    train_targets: Float[Array, " sample"],
    *,
    noise: float = 1e-2,
    solver: CholeskySolver | CGSolver | None = None,
) -> GaussianProcess:
    """Condition a GP on data: solve (K_XX + σ² I) α = y once.

    Rasmussen and Williams (2006), Algorithm 2.1.
    """
    solver = CholeskySolver() if solver is None else solver
    gram = kernel(train_inputs, train_inputs)
    noisy = gram + noise * jnp.eye(train_inputs.shape[0], dtype=gram.dtype)
    operator = lx.MatrixLinearOperator(noisy, lx.positive_semidefinite_tag)
    weights = lx.linear_solve(operator, train_targets, solver.as_lineax()).value
    return GaussianProcess(kernel, train_inputs, train_targets, weights, noise)
