"""Beer-Lambert transmittance of a layered atmosphere."""

import einx
import jax.numpy as jnp
from jaxtyping import Array, Float


def transmittance(
    absorption: Float[Array, "level channel"],
    path: Float[Array, "level"],
) -> Float[Array, "channel"]:
    """Transmittance through all levels, per channel.

    T = exp(−τ), with the optical depth τ = Σ_level k u.

    Args:
        absorption: Absorption coefficient k of each level and channel.
        path: Absorber amount u along the path in each level.

    Returns:
        The transmittance T of each channel.
    """
    # τ = Σ_level k u
    optical_depth = einx.dot("level channel, level -> channel", absorption, path)
    # T = exp(−τ)
    return jnp.exp(-optical_depth)
