"""Kernels: structures that know how to build a Gram matrix."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float


def _sqdist(
    x: Float[Array, "n feature"], z: Float[Array, "m feature"]
) -> Float[Array, "n m"]:
    diff = x[:, None, :] - z[None, :, :]
    return jnp.sum(diff**2, axis=-1)


class RBF(eqx.Module):
    """Squared-exponential kernel k(x, z) = σ² exp(-‖x - z‖² / (2 ℓ²))."""

    variance: float = 1.0
    lengthscale: float = 1.0

    def __call__(
        self, x: Float[Array, "n feature"], z: Float[Array, "m feature"]
    ) -> Float[Array, "n m"]:
        return self.variance * jnp.exp(-0.5 * _sqdist(x, z) / self.lengthscale**2)


class Matern32(eqx.Module):
    """Matérn-3/2 kernel k(x, z) = σ² (1 + √3 r / ℓ) exp(-√3 r / ℓ)."""

    variance: float = 1.0
    lengthscale: float = 1.0

    def __call__(
        self, x: Float[Array, "n feature"], z: Float[Array, "m feature"]
    ) -> Float[Array, "n m"]:
        scaled = jnp.sqrt(3.0 * _sqdist(x, z) + 1e-12) / self.lengthscale
        return self.variance * (1.0 + scaled) * jnp.exp(-scaled)
