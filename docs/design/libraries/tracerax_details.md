---
title: "tracerax: design details"
short_title: "tracerax details"
description: "The implementation reference behind the tracerax design: the core types, what callers pass in, the algorithms with their pitfalls, JAX behaviour, tests, dependencies, and a survey of the code that moves in."
date: 2026-11-05
keywords:
  - tracerax
  - design
  - details
  - GeoStack
---

The implementation reference behind the [tracerax design](tracerax.md): the core types, what callers pass in, the algorithms with their pitfalls, JAX behaviour, tests, dependencies, and a survey of the code that moves in.

## Core types

The state is a pytree of arrays; the closure is an abstract module; results are small modules that carry their units. Every name is proposed.

```python
# Shapes: N particles; nx × ny × nz bins; nx × ny surface cells

import abc

import equinox as eqx
from jaxtyping import Array, Bool, Float


# proposed (tracerax)
class ParticleState(eqx.Module):
    """An ensemble of N particles; every field is a pytree leaf."""

    # xₚ = (x, y, z) in metres, in a local frame around the source
    position: Float[Array, "particle 3"]
    # u'ₚ, the turbulent velocity fluctuation (zeros for pure advection)
    velocity: Float[Array, "particle 3"]
    # mₚ in kg; a release of rate Q over T gives mₚ = Q T / N
    mass: Float[Array, "particle"]
    # seconds since release; negative before a staggered release
    age: Float[Array, "particle"]
    # False before release and after leaving the domain
    active: Bool[Array, "particle"]


# proposed (tracerax)
class TurbulenceModel(eqx.Module):
    """σ(z) and τ_L(z), the closure a Langevin step reads."""

    @abc.abstractmethod
    def at(
        self, height: Float[Array, "particle"]
    ) -> tuple[Float[Array, "particle 3"], Float[Array, "particle 3"]]:
        """Return (σ, τ_L) per particle, one column per velocity component."""


# proposed (tracerax)
class Reflect(eqx.Module):
    """Reflecting walls in z: the ground and, optionally, the PBL top."""

    ground: float = 0.0
    top: float | None = None


# proposed (tracerax)
class Residence(eqx.Module):
    """Particle-seconds per bin: Rⱼ = ∑ₚ ∑ₖ Δtₖ 1[xₚ(tₖ) ∈ j]."""

    seconds: Float[Array, "x y z"]
    n_particles: int = eqx.field(static=True)


# proposed (tracerax)
class Footprint(eqx.Module):
    """Fⱼ in s m² kg⁻¹: sensitivity of one receptor to a surface flux in j."""

    sensitivity: Float[Array, "x y"]
    surface_layer_depth: float
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `ParticleState` | positions, velocity fluctuations, masses, ages, active flags | every leaf has the same N; inactive particles keep finite values and are masked, never NaN; mass ≥ 0 |
| `TurbulenceModel` | a closure's σ(z) and τ\_L(z) | σ > 0 and τ\_L > 0; `at` is differentiable in z, because the drift needs ∂σ\_w²/∂z, and finite at z = 0 |
| `Reflect` | the wall heights | ground < top |
| `Residence` | particle-seconds per bin and N | ∑ⱼ Rⱼ ≤ N T, with equality when no particle leaves the bins |
| `Footprint` | a sensitivity map and the surface-layer depth h\_s | a receptor value is y = ∑ⱼ Fⱼ qⱼ for a surface flux q in kg m⁻² s⁻¹ |

The bins keep plumax's (x, y, z) axis order so results compare directly with plumax's; xrops relabels them when it wraps the output.

## Types to adhere to

Callers pass three things in: a wind, a turbulence closure, and bin edges.

- **The wind** is a function of positions and time, `wind(position, time) -> velocity`, with shapes (N, 3), () → (N, 3). A gridded wind is a functax interpolation plan closed over the field. plumax's wind today is a function of time only, t → (u, v, w), uniform in space; `trx.advect.uniform(wind_of_time)` adapts it.
- **The closure** subclasses `TurbulenceModel`. plumax's `HomogeneousTurbulence` and `HannaTurbulence` (the profiles of @hanna1982) already satisfy the shape of `at` as frozen dataclasses; under tracerax they become `eqx.Module` subclasses, or are wrapped, so that `jax.grad` of σ\_w² traces through them.
- **The bins** are edge arrays per axis. They come from a gridax analysis geometry, from a finitevolX model grid for a model run, or are written by hand.

```python
# Shapes: N particles

from plumax.lagrangian import HannaTurbulence


# user
# u(x, t) at every particle, from a functax plan over the gridded wind
def wind_at(position, time): ...  # (N, 3), () → (N, 3)


# today (plumax)
# σ(z), τ_L(z) from Hanna (1982) boundary-layer scaling
turbulence = HannaTurbulence(u_star=0.4, pbl_height=1000.0, obukhov_length=-50.0)
```

## Algorithms

### Deterministic advection

Each particle follows the resolved wind:

```{math}
\frac{d x_p}{dt} = u(x_p, t)
```

1. Wrap `wind_at` as a diffrax `ODETerm` over the (N, 3) position array, so the ensemble is one state.
2. Step it with an integrix stepper: a fixed-step SSP Runge–Kutta for gridded winds, an adaptive pair for analytic ones.
3. Save at the requested times, or only the final state.
4. Deactivate particles that leave the domain: they keep their last position and `active = False`.

Cost: K steps × s stages × one wind evaluation of N points. Pitfalls: a linearly interpolated wind is only continuous, not smooth, so an adaptive error estimate stalls at cell faces; use fixed steps or a cubic plan there. A particle outside the wind's grid must be clamped before interpolation, or it reads garbage.

### The Langevin step, well mixed, with reflection

For Gaussian turbulence with σ(z) and τ\_L(z) per component, the velocity fluctuation is an Ornstein–Uhlenbeck process plus, for the vertical component, Thomson's (1987) well-mixed drift. The linear part is integrated with its exact transition over a step Δt, with ξ a standard normal draw:

```{math}
u'_{n+1} = e^{-\Delta t / \tau_L} u'_n + \sigma \sqrt{1 - e^{-2 \Delta t / \tau_L}} \xi
```

The vertical drift is added with an Euler step, and the position follows the mean wind plus the fluctuation:

```{math}
u'_{w,n+1} \leftarrow u'_{w,n+1} + \frac{1}{2} \left( 1 + \frac{u'^2_{w,n+1}}{\sigma_w^2} \right) \frac{\partial \sigma_w^2}{\partial z} \Delta t
```

```{math}
x_{n+1} = x_n + (u(x_n, t_n) + u'_{n+1}) \Delta t
```

1. Read σ, τ\_L from the closure at each particle's height; floor σ at a small positive value.
2. Draw ξ from this step's key and apply the exact OU transition to all three components.
3. Compute ∂σ\_w²/∂z by `jax.vmap(jax.grad(...))` of the closure, and add the drift to u'\_w.
4. Move the particles.
5. Reflect: below the ground, z → −z and u'\_w → −u'\_w; above the top h, z → 2h − z and u'\_w → −u'\_w.
6. Shorten the last step so the run ends exactly at the horizon (plumax's `step_durations`).

Cost: O(N) per step, plus one gradient of σ\_w² per particle. Pitfalls:

- The exact OU transition makes homogeneous runs well mixed for any Δt, but the drift term is first order, so Δt must stay small against τ\_L near the ground, where τ\_L is shortest.
- Convective profiles carry a (z/h)^(2/3) factor whose derivative is singular at the ground; plumax floors z at 0.1 m before evaluating the closure.
- Perfect reflection keeps the well-mixed state only when σ\_w is uniform near the wall [@wilson1993boundaries]; with strong gradients at the wall it biases the near-ground density, which is exactly where a footprint is counted.

### Binning by scatter-add

With edges e₀ < … < eₙ per axis, a particle's cell index is i = searchsorted(e, x) − 1, kept when 0 ≤ i < n. The three indices flatten to one, and a scatter-add accumulates the weight:

```{math}
R_j \leftarrow R_j + \sum_p \Delta t_k \mathbb{1}[x_p(t_k) \in j]
```

For a continuous source of rate Q released as N particles at t = 0 in a steady flow, the steady concentration is cⱼ = Q Rⱼ / (N Vⱼ), with Vⱼ the cell volume. Cost: O(N) per step with a single `.at[].add`. Pitfalls: a histogram is piecewise constant in the positions, so its gradient with respect to them is zero almost everywhere. Gradients with respect to the wind or the closure's parameters need a smooth deposit instead, such as cloud-in-cell (linear) weights, which spread each particle over the 2ᵈ nearest cells.

### Backward footprints

Particles released at a receptor and integrated backward in time give the receptor's sensitivity to a surface flux. With h\_s = f h the surface-layer depth and ρ the air density:

```{math}
F_j = \frac{1}{N \rho h_s} \sum_p \sum_k \Delta t_k \mathbb{1}[x_p(t_k) \in j, z_p(t_k) < h_s]
```

1. Reverse the sign of the mean wind.
2. Reverse the order of the forward step durations, so the partial step comes first, and sample the wind at each interval's start on the receptor's clock.
3. Run the Langevin step; accumulate residence of the particles below h\_s per surface cell.
4. Divide by N ρ h\_s. There is no division by cell area: F pairs with a flux in kg m⁻² s⁻¹.

Pitfall: reusing the forward integrator with the wind reversed is exact for the OU part when the turbulence is stationary, because that process is time-reversible. In inhomogeneous turbulence the backward-time drift is not the forward one with the wind reversed [@flesch1995backward]; this needs a test before the footprints are trusted near the ground.

### Finite-time Lyapunov exponents

With Φ the flow map of the deterministic advection from t₀ to t₁ and ∇Φ its Jacobian at a seed x₀:

```{math}
C(x_0) = \nabla\Phi(x_0)^\top \nabla\Phi(x_0)
```

```{math}
\Lambda(x_0) = \frac{1}{|t_1 - t_0|} \ln \sqrt{\lambda_{\max}(C(x_0))}
```

1. Write the flow map for one seed as a function x₀ ↦ x(t₁).
2. Take `jax.jacfwd` of it and `jax.vmap` over the seeds.
3. Form C with einx (`einx.dot("... a b, ... a c -> ... b c", J, J)`) and take the largest eigenvalue with `jnp.linalg.eigvalsh`.

Cost: about d times one integration for d dimensions, against 2d extra trajectories per seed for central differences. Pitfalls: ∇Φ grows exponentially with |t₁ − t₀|, so long windows overflow in float32; split the window and multiply the Jacobians, accumulating the logarithm. FTLE is defined for the deterministic flow only: no turbulence draw.

### Residence time and connectivity

With regions labelled on the bins, residence in region A is τ\_A = ∑ₖ Δtₖ 1\[xₚ(tₖ) ∈ A\] per particle, and the connectivity from release region i to region j is the fraction of i's particles found in j at the end (or ever, for a visiting matrix):

```{math}
C_{ij} = \frac{1}{N_i} \sum_{p \in i} \mathbb{1}[x_p(t_1) \in j]
```

Both reuse the binning: a label lookup per particle, then a one-hot contraction over particles with einx, O(N × regions) per step.

## JAX compatibility

| Function | jit | vmap | grad | Notes |
| --- | --- | --- | --- | --- |
| `advect` (deterministic) | yes | yes, over ensembles or seeds | yes, through diffrax (wind parameters, initial positions) | Adaptive steps need a smooth wind; fixed steps for gridded winds |
| `langevin_step` | yes | yes | yes, with respect to the closure's parameters (pathwise, fixed key) | Reflection is a `jnp.where`, piecewise linear; the gradient at a reflection is one-sided |
| `integrate` (scan of steps) | yes, step count static | yes, over keys for independent ensembles | yes, reverse mode through the scan; checkpoint long runs | `n_steps` comes from the horizon and `dt` on the host, as plumax's `n_steps_for_horizon` does |
| `bin_residence` | yes, edges as arrays | yes | zero with respect to positions (histogram); smooth with cloud-in-cell weights | Out-of-range particles masked, not dropped, so shapes stay static |
| `footprint` | yes | yes, over receptors | as `bin_residence` | Returns arrays; the xarray wrapper lives in the adapter, not under jit |
| `ftle` | yes | yes, over seeds | not needed (it is itself a derivative) | `jacfwd` of the flow map; second derivatives cost another factor d |
| `residence_time`, `connectivity` | yes | yes | zero with respect to positions | Region labels are integer arrays |

Keys: one key per run, split up front into one key per step (`jr.split(key, n_steps)`), so a scan step never reuses a key and the draws do not depend on how the run is chunked. The step function takes its key as an argument; nothing reads `numpy.random`.

Scan carry: `(ParticleState, accumulator)`, where the accumulator is the residence array of the bins. Saving trajectories turns the per-step position into a scan output instead.

Memory: O(N + bins) for an accumulating run; O(K N) when the trajectory is saved, which is the usual way a long run runs out of memory. Reverse-mode gradients through a long scan store every step's carry; use `jax.checkpoint` on the step, or forward mode when the parameters are few.

Precision: positions in metres far from the origin lose resolution in float32 (a 1 cm step at 10⁶ m is below float32's resolution). Run in a local frame with the origin near the source (xrops' `LocalFrame`), or in float64.

## Testing

- **plumax oracle.** For homogeneous and Hanna turbulence with a uniform wind and the same keys, the moved Langevin step, binning and footprint match plumax 28f257e to round-off. This test is deleted when plumax re-exports tracerax.
- **Well-mixed stays well-mixed.** Particles started uniform in \[0, h\] with Hanna profiles stay uniform after many τ\_L: a Kolmogorov–Smirnov test on the heights, with the threshold from the KS statistic's sampling distribution at the chosen N, not a tuned tolerance.
- **@taylor1921.** In homogeneous turbulence, the variance of the displacement matches the closed form, within a few standard errors of the sample variance:

```{math}
\mathrm{Var}[z(t)] = 2 \sigma^2 \tau_L \left( t - \tau_L (1 - e^{-t / \tau_L}) \right)
```

- **Mass conservation.** With reflecting walls and a domain larger than the plume, total residence equals N times the horizon; with an open boundary it equals the sum over active particles.
- **Forward and backward duality.** For a uniform wind and homogeneous turbulence, the forward concentration at a receptor from a unit surface source equals the backward footprint summed over that source's cells, within the Monte Carlo error.
- **Deterministic advection.** Solid-body rotation returns particles to their start after one period; the double gyre's FTLE ridges match @shadden2005lcs, and the `jacfwd` Jacobian matches central finite differences.
- **Transforms.** Every public function gives the same result eagerly, under `jit` and under `vmap` over a batch of one; gradients with respect to closure parameters match finite differences at a fixed key.

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| JAX | arrays, `lax.scan`, PRNG, `vmap`, `grad`, `jacfwd` | required |
| Equinox | `ParticleState`, `TurbulenceModel` and the result types as modules | required |
| einx | contractions (Cauchy–Green tensor, one-hot connectivity) | required |
| integrix (and diffrax beneath it) | the Runge–Kutta steppers for deterministic advection | required |
| functax | sampling a gridded wind at particle positions | required for gridded winds |
| gridax | bin edges from a grid, cell volumes | optional (edges can be passed by hand) |
| geophysix | air density, PBL height and other inputs to the closures | optional |
| xarray | the Dataset wrappers of `simulate` and `footprint` results | optional, adapter only |
| plumax | the oracle tests | tests only |

## Where it stands

Nothing named tracerax exists yet. The code that moves in is plumax's `lagrangian` subpackage at **plumax 28f257e**:

| File | Lines | What it holds | Moves to |
| --- | --- | --- | --- |
| `lagrangian/particles.py` | 265 | `ParticleState` (a frozen dataclass registered as a pytree; position and velocity only), the `TurbulenceModel` protocol, `langevin_step` (exact OU on all three components, Euler drift on w, ground and PBL-top reflection), `n_steps_for_horizon`, `step_durations`, `integrate_particles` (a `lax.scan`, keys split up front), `uniform_wind`, `wind_from_speed_direction` | tracerax `step` and `integrate`; `ParticleState` gains mass, age and active |
| `lagrangian/concentration.py` | 239 | `_cell_indices` (searchsorted), `bin_positions`, `_accumulate_residence`, `simulate_lagrangian` (all particles released at t = 0; returns an xarray Dataset with `concentration` and `column_concentration`) | tracerax `bin`; the Dataset wrapper to the adapter |
| `lagrangian/footprint.py` | 173 | `compute_footprint`: reversed wind, reversed step order, the wind sampled on the receptor clock, F = R / (N ρ f h); returns NumPy arrays | tracerax `footprint`, returning JAX arrays |
| `lagrangian/turbulence.py` | — | `HomogeneousTurbulence`, `hanna_profiles`, `HannaTurbulence` (with `z_floor = 0.1`) | stays in plumax; subclasses tracerax's `TurbulenceModel` |
| `lagrangian/inversion.py` | — | `GaussianPosterior`, `matern32_covariance`, `observation_covariance` | stays in plumax |

What the move adds rather than carries over: a wind that varies in space (plumax's wind is a function of time only, `t → (u, v, w)`), deterministic advection through integrix, FTLE, residence time and connectivity, staggered releases, and the active mask. None of these exist anywhere in the stack today.

Neighbours checked and left alone: finitevolX 82090a3 has `semi_lagrangian_step` (field in, field out by `map_coordinates` at departure points), which is Eulerian transport and stays there; xrtoolz b3855d7 has `xrops.metrics._src.lagrangian`, a six-line stub reserved for a later view, and nothing to move.
