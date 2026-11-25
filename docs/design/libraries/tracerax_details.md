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

The state is a pytree of arrays; the closure is a protocol; a step moves a state; results are small modules. The names below are tracerax's, at v0.0.4.

```python
# Shapes: N particles in D = 2 or 3 dimensions; nx × ny surface cells

from typing import Protocol

import equinox as eqx
from jaxtyping import Array, Bool, Float


# today (tracerax.particles)
class ParticleState(eqx.Module):
    """An ensemble of N particles; every field is a pytree leaf."""

    # xₚ, in metres in a local frame, or (longitude, latitude) in degrees
    position: Float[Array, "particle axis"]
    # the step's memory: u'ₚ of a Langevin step, the object's velocity of an inertial one
    velocity: Float[Array, "particle axis"]
    # mₚ in kg; a release of rate Q over T gives mₚ = Q T / N
    mass: Float[Array, " particle"]
    # when it is released, in seconds from the run's epoch
    release_time: Float[Array, " particle"]
    # False after leaving the domain
    active: Bool[Array, " particle"]


# today (tracerax.particles)
class ParticleStep(Protocol):
    """What moves a state: each particle over its own duration δ ≥ 0."""

    def noise_shape(self, state): ...
    def __call__(self, state, start, duration, direction, noise): ...


# today (tracerax.advect)
class TurbulenceModel(Protocol):
    """σ(z) and τ_L(z), the closure a Langevin step reads."""

    def at(
        self, height: Float[Array, " particle"]
    ) -> tuple[Float[Array, "particle 3"], Float[Array, "particle 3"]]: ...


# today (tracerax.advect)
class Walls(eqx.Module):
    """Reflecting walls in z: the ground and, optionally, a top."""

    bottom: float = 0.0
    top: float | None = None


# today (tracerax.footprint)
class Footprint(eqx.Module):
    """Fⱼ in s m² kg⁻¹: sensitivity of one receptor to a surface flux in j."""

    sensitivity: Float[Array, "x y"]
    residence: Float[Array, "x y"]
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `ParticleState` | positions, memories, masses, release times, active flags | every leaf has the same N; inactive particles keep finite values and are masked, never dropped; mass ≥ 0 |
| `ParticleStep` | a velocity, a closure, parameters | a particle with duration 0 is unchanged bit for bit; releases, masks, domains and noise are the loop's |
| `TurbulenceModel` | a closure's σ(z) and τ\_L(z) | σ > 0 and τ\_L > 0; `at` is differentiable in z, because the drift needs ∂σ\_w²/∂z, and finite at z = 0 |
| `Walls` | the wall heights | bottom < top |
| `Cells` | increasing edges per axis, in the order of the position columns | a cell is [e\_i, e\_{i+1}); the last edge is outside |
| `Residence`, `RegionTime` | accumulators: seconds per cell, and per particle and region | ∑ⱼ Rⱼ ≤ N T, with equality when no particle leaves the cells |
| `Footprint` | a sensitivity map and the particle-seconds under it | a receptor value is y = ∑ⱼ Fⱼ qⱼ for a surface flux q in kg m⁻² s⁻¹ |

The cells keep plumax's (x, y, z) axis order so results compare directly with plumax's; xrops relabels them when it wraps the output.

## Types to adhere to

Callers pass three things in: a wind, a turbulence closure, and cell edges.

- **The wind** is a function of time and positions, `velocity(t, position)`, with one velocity per particle, at one time or one time per particle. A gridded wind is `GriddedVelocity`, one functax `Interpolant` per component on its own staggered nodes, which equals plumax's `GriddedWind` to 10⁻¹⁵. A wind that is uniform in space is `uniform_in_space(f)`, and a current plus a share of the wind is `WeightedSum`.
- **The closure** is any object with `at(height)`. plumax's `HomogeneousTurbulence` and `HannaTurbulence` (the profiles of @hanna1982) have it; as `eqx.Module`s, `jax.grad` of σ\_w² reaches their parameters too.
- **The cells** are edge arrays per axis, `Cells((x_edges, y_edges))`. They may be stretched, and their volumes are functax's `quad.cell_area`.

```python
# Shapes: N particles

from plumax.lagrangian import HannaTurbulence
from tracerax.velocity import GriddedVelocity, cgrid_grids

# today (tracerax)
# u at (xᵢ + Δx/2, yⱼ), v at (xᵢ, yⱼ + Δy/2), w at the T points
wind = GriddedVelocity.fit(cgrid_grids(times, x, y, z), (u, v, w))
wind(t, position)  # (N, 3) → (N, 3)

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

1. `deterministic` wraps the velocity as a diffrax term over the (N, D) position array, so the ensemble is one state, and solves it in equal steps with an integrix or diffrax solver. Each particle has its own clock, so a release on a step time is exact.
2. `Deterministic` is one explicit Runge–Kutta step, unrolled from the solver's tableau, for the loop. There a release inside a step is exact too, and forward and reverse mode both work.
3. Save at the requested times, or only the final state.
4. Deactivate particles that leave the domain: they keep their last position and `active = False`.

Cost: K steps × s stages × one wind evaluation of N points. Pitfalls: a linearly interpolated wind is only continuous, not smooth, so use fixed steps or a cubic scheme there. A particle outside the wind's grid is clamped by default. A diffrax right-hand side that switches on at a release is read by the last stage of the step before, so the release is a control with an increment per particle.

### The sphere

Positions are (longitude, latitude) in degrees and velocities eastward and northward. With r the unit vector of a position and R the radius:

```{math}
\frac{d r}{dt} = \frac{u e_\lambda + v e_\varphi}{R}
```

and r is divided by its norm after each step. Solid-body rotation about an axis tilted by 0.4 rad, one turn in 100 RK4 steps, returns to 1.9 m on a path up to 55.8° and to 0.003 m on one up to 88.8°, where steps in longitude and latitude were off by 1.3 km. Pitfalls: a longitude comes back in [−180°, 180°), so distances along a path are great-circle ones; within √ε of a pole the velocity is read at that distance from it, because a derivative through a longitude is the difference of two numbers of size 1/ε; in float32 a step's rounding may have the same sign at every step, so a displacement per step of a few metres is lost.

### Inertial particles

The Maxey–Riley equation for the surface ocean (Beron-Vera, Olascoaga and Miron, 2019) is linear in the object's velocity once the flow is frozen over a step, so the step is exact in the drag and the rotation, with the flow evaluated at the middle. In a cellular flow of 50 km and 0.5 m s⁻¹ the error after a day is below 1.6 m at steps of 900 s, for response times from a minute to an hour, and falls about four times per halving. The slow manifold is a velocity without memory, off by order τ².

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
3. Compute ∂σ\_w²/∂z by forward mode of the closure, and add the drift to u'\_w.
4. Move the particles.
5. Reflect by a triangle-wave fold of period 2h, with u'\_w changing sign when the number of folds is odd. One reflection per wall is not enough: with a 50 m layer, σ = 1.5 m s⁻¹ and τ = 100 s, plumax's step leaves 6.8 % of the particles outside the layer at Δt = 30 s and 39 % at 60 s ([plumax#392](https://github.com/jejjohnson/plumax/issues/392)); the fold leaves none.
6. Shorten the last step so the run ends exactly at the horizon (plumax's `step_durations`).

Cost: O(N) per step, plus one gradient of σ\_w² per particle. Pitfalls:

- The exact OU transition makes homogeneous runs well mixed for any Δt, but the drift term is first order, so Δt must stay small against τ\_L near the ground, where τ\_L is shortest.
- Convective profiles carry a (z/h)^(2/3) factor whose derivative is singular at the ground; plumax floors z at 0.1 m before evaluating the closure.
- Perfect reflection keeps the well-mixed state only when σ\_w is uniform near the wall [@wilson1993boundaries]; with strong gradients at the wall it biases the near-ground density, which is exactly where a footprint is counted.

### Binning by scatter-add

With edges e₀ < … < eₙ per axis, a particle's cell index is i = max{j : eⱼ ≤ x}, kept when 0 ≤ i < n. The indices flatten to one, and the transpose of a functax stencil accumulates the weight:

```{math}
R_j \leftarrow R_j + \sum_p \Delta t_k \mathbb{1}[x_p(t_k) \in j]
```

For a continuous source of rate Q released as N particles at t = 0 in a steady flow, the steady concentration is cⱼ = Q Rⱼ / (N Vⱼ), with Vⱼ the cell volume. Cost: O(N) per step with a single `.at[].add`. Pitfalls: a histogram is piecewise constant in the positions, so its gradient with respect to them is zero almost everywhere. Gradients with respect to the wind or the closure's parameters need `kind="linear"`, cloud-in-cell weights, which spread each particle over the 2ᵈ nearest cells. functax's `Binning` alone does not do either: the nearest one ignores an axis's explicit edges, so stretched cells get the wrong samples ([functax#168](https://github.com/jejjohnson/functax/issues/168)), and the linear one drops what falls between an end centre and its wall, which is the ground of a footprint ([functax#169](https://github.com/jejjohnson/functax/issues/169)). tracerax locates the cells itself, and folds ghost cells back, until they close; a particle inside the edges then deposits its whole weight.

### Backward footprints

Particles released at a receptor and integrated backward in time give the receptor's sensitivity to a surface flux. With h\_s = f h the surface-layer depth and ρ the air density:

```{math}
F_j = \frac{1}{N \rho h_s} \sum_p \sum_k \Delta t_k \mathbb{1}[x_p(t_k) \in j, z_p(t_k) < h_s]
```

1. Run the loop with a negative duration from the receptor time; the Langevin step takes −u.
2. The loop reverses the order of the step durations, so the partial step comes first, and the wind is read at the physical start of each forward interval.
3. Run the Langevin step; accumulate residence of the particles below h\_s per surface cell.
4. Divide by N ρ h\_s. There is no division by cell area: F pairs with a flux in kg m⁻² s⁻¹.

Pitfall: reusing the forward integrator with the wind reversed is exact for the OU part when the turbulence is stationary, because that process is time-reversible. In inhomogeneous turbulence the backward-time drift is not the forward one with the wind reversed [@flesch1995backward]; so a footprint in inhomogeneous turbulence is approximate, as plumax's is. In homogeneous turbulence the reciprocity test of Wave 2 agrees to its sampling error.

### Finite-time Lyapunov exponents

With Φ the flow map of the deterministic advection from t₀ to t₁ and ∇Φ its Jacobian at a seed x₀:

```{math}
C(x_0) = \nabla\Phi(x_0)^\top \nabla\Phi(x_0)
```

```{math}
\Lambda(x_0) = \frac{1}{|t_1 - t_0|} \ln \sqrt{\lambda_{\max}(C(x_0))}
```

1. Solve every seed's path once, with one tangent per position column pushed through the solve: a seed's velocity depends on its own position only, so the seeds share them. The solve needs `diffrax.ForwardMode()`, because diffrax's default adjoint has no forward mode (`integrix.march.run` does too).
2. Split the duration in windows, multiply their Jacobians, and divide by the largest entry after each, accumulating the logarithm, so that nothing overflows.
3. Form C with einx and take the largest eigenvalue with `jnp.linalg.eigvalsh`.

On the sphere the Jacobian is taken in the local east–north frames at both ends. Cost: one solve with d tangents, against 2d extra trajectories per seed for central differences. Pitfalls: a linear read's gradient jumps at cell faces, so the ridges depend on the scheme; FTLE is defined for the deterministic flow only: no turbulence draw.

### Residence time and connectivity

With regions labelled on the bins, residence in region A is τ\_A = ∑ₖ Δtₖ 1\[xₚ(tₖ) ∈ A\] per particle, and the connectivity from release region i to region j is the fraction of i's particles found in j at the end (or ever, for a visiting matrix):

```{math}
C_{ij} = \frac{1}{N_i} \sum_{p \in i} \mathbb{1}[x_p(t_1) \in j]
```

Regions are an integer label per cell of a `Cells`, with −1 for none. `RegionTime` sums per particle and region inside the loop, right to one step per crossing, and `connectivity` is a one-hot contraction over particles at the end; `visited_connectivity` counts every region a particle has spent time in.

### Dispersion and skill

Absolute and pair dispersion are mean squared distances, along great circles on the sphere. The normalised cumulative separation of a trajectory (Liu and Weisberg, 2011) is the sum of its separations from an observed one over the sum of the observed path's lengths; its gradient is finite where the two coincide, so it is a loss. Missing observations are NaN, left out of both sums, and the length bridges the gap.

## JAX compatibility

| Function | jit | vmap | grad | Notes |
| --- | --- | --- | --- | --- |
| `deterministic`, `deterministic_sphere` | yes | yes, over ensembles or seeds | yes, through diffrax (wind parameters, initial positions) | Forward mode needs `diffrax.ForwardMode()` |
| `Deterministic`, `DeterministicSphere`, `Inertial` | yes | yes | yes, both modes | Unrolled stages, with a duration per particle |
| `Langevin`, `RandomDisplacement` | yes | yes | yes, with respect to the closure's parameters (pathwise, fixed noise) | Walls are a fold, piecewise linear |
| `integrate` | yes, step count static | yes, over keys for independent ensembles | yes, reverse mode through the scan, checkpointed or truncated | The number of steps comes from the duration and the step on the host |
| `deposit`, `Residence` | yes, edges as arrays | yes | zero for `"nearest"`; smooth for `"linear"` between cell centres | Out-of-range particles masked, not dropped, so shapes stay static |
| `surface` | yes | yes, over receptors, times and keys | in the wind, with `kind="linear"` | Nothing flows through the step of the surface layer, so the gradient in the receptor's height or the closure is not the footprint's |
| `ftle` | yes | yes, over seeds | with `adjoint=diffrax.DirectAdjoint()` | One solve carries the tangents |
| `connectivity` | yes | yes | zero with respect to positions | Region labels are integer arrays |
| dispersion, separation, skill | yes | yes | yes, finite where two points coincide | Missing positions are NaN |

Keys: one key per run, split up front into one key per step, or the noise itself, one array per step, for antithetic pairs or common random numbers. A step takes its noise as an argument; nothing reads `numpy.random`.

Scan carry: `(ParticleState, accumulator)`, where the accumulator is the residence array of the cells, or any `Accumulator`. Saving trajectories adds the position every few steps as a scan output.

Memory: O(N + bins) for an accumulating run; O(K N) when the trajectory is saved, which is the usual way a long run runs out of memory. Reverse-mode gradients through a long scan store every step's carry; use `integrix.adjoint.Checkpointed(every=k)`, which stores O(K/k + k) of them, or forward mode when the parameters are few.

Precision: positions in metres far from the origin lose resolution in float32 (a 1 cm step at 10⁶ m is below float32's resolution). Run in a local frame with the origin near the source (xrops' `LocalFrame`), or in float64. Times have the same limit: the loop raises when a concrete time is too coarse for the step.

## Testing

- **plumax oracle.** plumax will import tracerax, so its values are stored, written at `96449d2`: the Langevin step, the gridded wind, the binning, the residence and the footprint match them to 10⁻¹² or better. The stored values are deleted when plumax switches.
- **Well-mixed stays well-mixed.** Particles started uniform in \[0, h\] with Hanna profiles stay uniform after many τ\_L: a Kolmogorov–Smirnov test on the heights, with the threshold from the KS statistic's sampling distribution at the chosen N, not a tuned tolerance.
- **@taylor1921.** In homogeneous turbulence, the variance of the displacement matches the closed form, within a few standard errors of the sample variance:

```{math}
\mathrm{Var}[z(t)] = 2 \sigma^2 \tau_L \left( t - \tau_L (1 - e^{-t / \tau_L}) \right)
```

- **Mass conservation.** Each particle inside the edges deposits its whole weight to 10⁻¹², at the walls and on stretched cells. Total residence equals N times the horizon when no particle leaves.
- **Forward and backward duality.** For a uniform wind and homogeneous turbulence, the mean time from one 50 m box to another 500 m away is 3.052 ± 0.020 s forward and 3.008 ± 0.013 s backward (8 runs of 10,000 particles).
- **Deterministic advection.** Solid-body rotation returns particles to their start after one period; the double gyre of @shadden2005lcs has exponents from 0.009 to 0.463 s⁻¹, which match central differences of the flow map; a path through both poles of the sphere returns to its start.
- **Transforms.** Every public function gives the same result eagerly, under `jit` and under `vmap` over a batch of one; gradients with respect to closure parameters match finite differences at a fixed key.

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| JAX | arrays, `lax.scan`, PRNG, `vmap`, `grad`, `jvp` | required |
| Equinox | the state, the steps, the velocities and the results as modules | required |
| einx, jaxtyping | contractions, broadcasts and reductions; annotations | required |
| functax (`functax[jax]`) | reading a gridded wind at particles, the cells' axes and volumes, stencil transposes, unit vectors and east–north frames | required |
| integrix, and diffrax | Runge–Kutta solvers, checkpointed and truncated scans | required |
| geophysix | the sphere's radius and great-circle distances; it brings pipekit-array | required, since Wave 3 |
| gridax | staggered coordinates and masks, for an adapter | optional (`[gridax]`) |
| plumax | the oracle's stored values | none: never imported, not even by a test |

functax and integrix are pinned to tags and move together, as integrix fixes functax's tag; geophysix and gridax too, as gridax fixes geophysix's.

## Where it stands

tracerax v0.0.4 holds Waves 0 to 3. The code that moves in is plumax's `lagrangian` subpackage at **plumax `96449d2`**, under `lagrangian/_src/` since [plumax#306](https://github.com/jejjohnson/plumax/pull/306), and plumax still has its own copy until Wave 4:

| File | What it holds | Moves to |
| --- | --- | --- |
| `particles.py` | `ParticleState` (position and velocity only), the `TurbulenceModel` protocol, `langevin_step`, `n_steps_for_horizon`, `step_durations`, `integrate_particles`, `uniform_wind`, `wind_from_speed_direction` | `tracerax.particles`, `tracerax.advect`, `tracerax.velocity`; `wind_from_speed_direction` stays |
| `wind_interp.py` | `GriddedWind` and its helpers; the functions that build one from meteorology | `tracerax.velocity.GriddedVelocity`; the builders stay |
| `concentration.py` | `bin_positions`, `_accumulate_residence`, `simulate_lagrangian` (returns an xarray Dataset) | `tracerax.bin`; `simulate_lagrangian` stays, on the loop and `Residence` |
| `footprint.py` | `compute_footprint` and its helpers; returns NumPy arrays | `tracerax.footprint.surface`, behind a thin call |
| `emulator.py`, `met_ensemble.py` | `FootprintSimulator`, ensembles of meteorology | stay; the simulator runs `surface` under `vmap` |
| `turbulence.py` | `HomogeneousTurbulence`, `hanna_profiles`, `HannaTurbulence` (with `z_floor = 0.1`) | stays, as `eqx.Module`s |
| `inversion.py` | `GaussianPosterior`, `matern32_covariance`, `observation_covariance` | stays |

What tracerax added rather than carried over: releases over time and inside a step, walls that fold any number of times, deposits that keep the mass at walls, random displacement, the sphere, inertial particles, FTLE, residence times per region, connectivity, dispersion and trajectory skill.

Neighbours checked and left alone: `integrix.march.semi_lagrangian_step` (field in, field out at departure points) is Eulerian transport and stays there; xrops has `metrics/_src/lagrangian.py`, a stub reserved for a later view, and nothing to move.
