---
title: "tracerax: Lagrangian particles and trajectories for JAX"
short_title: "tracerax"
description: "tracerax carries positions forward in time and analyses where they went: particle seeding, deterministic and stochastic advection, binning to grids, footprints, finite-time Lyapunov exponents, residence times and connectivity. It sits at GeoNumerics tier 2, on functax's interpolation, integrix's stepping and geophysix's sphere, and takes over the particle machinery now inside plumax. Status: built, with the consumers still to switch."
date: 2026-11-05
keywords:
  - tracerax
  - design
  - GeoStack
---

tracerax carries positions forward in time and analyses where they went: particle seeding, deterministic and stochastic advection, binning to grids, footprints, finite-time Lyapunov exponents, residence times and connectivity. It sits at GeoNumerics tier 2, on functax's interpolation, integrix's stepping and geophysix's sphere, and takes over the particle machinery now inside plumax. Status: built, with the consumers still to switch.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoNumerics, tier 2 |
| Imports | functax, integrix and geophysix; diffrax, JAX, Equinox, einx; gridax as the `gridax` extra |
| Imported by | plumax, sesmax-ocean and xrops, all planned |
| Takes over | plumax `lagrangian`: `ParticleState`, `langevin_step`, `integrate_particles`, `GriddedWind`, `bin_positions`, `compute_footprint` and the generic loop of `simulate_lagrangian` |
| Does not own | operators on fields, including semi-Lagrangian advection, whose departure points are discarded (integrix); turbulence closures, plume physics and source inversions (plumax); interpolation and binning (functax); steppers and adjoints (integrix) |
| Status | built: [jejjohnson/tracerax](https://github.com/jejjohnson/tracerax), Waves 0 to 3 of the backlog of [geostack#304](https://github.com/jejjohnson/geostack/issues/304); Wave 4, the consumers, is open |
| Tracked in | the [layer map](../stack_layer_map.md); the companion [integrix](integrix.md) and [gridax](gridax.md) designs |

```{figure} ../../images/libraries/tracerax.svg
:alt: where tracerax sits · GeoNumerics tier 2
:width: 100%

Where tracerax sits · geonumerics tier 2.
```

The line with integrix: a scheme that discards its departure points after each step is integrix's (`integrix.march.semi_lagrangian_step`, [integrix#92](https://github.com/jejjohnson/integrix/issues/92)); anything whose state is the positions themselves is tracerax's. This page is the design; core types, algorithms and the code survey are in [tracerax: design details](tracerax_details.md).

## The problem

Two questions about a methane plume are Lagrangian. Forward: where does the gas from this source go, and what concentration does it leave in each pixel? Backward: which surface cells could have produced the enhancement this satellite pixel saw? Both release particles, move them with the wind plus turbulence, and count where they end up:

| Step | What it computes | Where plumax codes it (`96449d2`, under `lagrangian/_src/`) |
| --- | --- | --- |
| Release and carry particles | a particle state (positions and velocities, all released at t = 0), a Markov-1 Langevin step and its loop | `particles.py`: `ParticleState`, `langevin_step`, `integrate_particles` |
| Wind at each particle | a wind read from a grid in space and time, since [plumax#180](https://github.com/jejjohnson/plumax/issues/180) | `wind_interp.py`: `GriddedWind` |
| Concentration on a grid | particles binned to cells | `concentration.py`: `bin_positions`, `simulate_lagrangian` |
| Sensitivity of a receptor to surface fluxes | backward integration and residence near the ground | `footprint.py`: `compute_footprint` |
| Turbulence for the step | σ\_w(z) and τ\_L(z) | `turbulence.py`: `HannaTurbulence`, `HomogeneousTurbulence`; stays |
| Meteorological ensembles, a footprint emulator | many footprints, and a model of them | `met_ensemble.py`, `emulator.py`; stay |
| Transport barriers, retention, connectivity | FTLE, residence time, connectivity matrices | nowhere; `tracerax.analysis` has them since Wave 3 |

The particle machinery is generic, but it sits inside a plume model, so the ocean side (sesmax's drifters, larval connectivity, retention in eddies) and xrops' planned Lagrangian views would have had to import plumax to use it, or write it again.

## Positions carried forward

The state of a Lagrangian calculation is a set of N particles: positions xₚ, masses mₚ, a release time per particle, an active flag, and the step's memory (the velocity fluctuation u'ₚ of a stochastic model, or an object's own velocity). Everything in tracerax either moves that state or reads it.

**Deterministic advection.** Each particle follows the resolved wind, dxₚ/dt = u(xₚ, t), with u at the particle's position read from a grid by a functax interpolant and the step taken by an integrix or diffrax solver. On the sphere the step works on unit vectors, so a path crosses the date line and the poles.

**Stochastic advection.** Unresolved turbulence enters as a Markov-1 Langevin model for the velocity fluctuations: an Ornstein–Uhlenbeck update on all three components, as plumax does it, with the well-mixed drift of @thomson1987 acting in the vertical, with σ\_w(z) the velocity standard deviation and τ\_L(z) the Lagrangian time scale supplied by a turbulence closure:

```{math}
du'_w = -\frac{u'_w}{\tau_L} dt + \frac{1}{2} \frac{\partial \sigma_w^2}{\partial z} \left( 1 + \frac{u_w'^2}{\sigma_w^2} \right) dt + \sqrt{\frac{2 \sigma_w^2}{\tau_L}} dW
```

Particles reflect at the ground and, when it is given, at the top of the boundary layer, by a fold that leaves none outside however far a step goes.

**Diffusion and inertia.** A surface drifter has a diffusivity in place of a closure, and follows a random displacement with the drift ∂K/∂x that keeps a uniform cloud uniform. A buoyant object of finite size follows the Maxey–Riley equation for the surface ocean, or its slow manifold.

**Binning.** The concentration in cell j of volume Vⱼ is the particle mass inside it:

```{math}
c_j = \frac{1}{V_j} \sum_p m_p \mathbb{1}[x_p \in j]
```

**Footprints.** Released backward in time from a receptor, particles that spend time near the ground say how sensitive the receptor is to a surface flux in each cell. With h\_s the depth of the surface layer and ρ the air density:

```{math}
F_j = \frac{1}{N \rho h_s} \sum_p \int \mathbb{1}[x_p(t) \in j, z_p(t) < h_s] dt
```

**Trajectory analysis.** With Φ the flow map that carries a position from t₀ to t₁, the finite-time Lyapunov exponent measures how fast neighbours separate; its ridges are transport barriers:

```{math}
\Lambda(x_0) = \frac{1}{|t_1 - t_0|} \ln \sqrt{\lambda_{\max}\left( \nabla\Phi(x_0)^\top \nabla\Phi(x_0) \right)}
```

In JAX the gradient of the flow map is the forward-mode derivative of the integration itself, so no finite-difference grid of seeds is needed. Residence time counts how long particles stay in a region; a connectivity matrix Cᵢⱼ counts the fraction released in region i that reaches region j.

## Modules and boundaries

Six modules. The particle state is an Equinox module of arrays, so it is a pytree that `jax.lax.scan`, `jit` and `vmap` carry as plumax's already is.

| Module | Owns | Comes from |
| --- | --- | --- |
| `tracerax.particles` | the particle state, with a release time per particle; the step protocol `ParticleStep`; seeding at points, along lines, in boxes and over time | plumax `ParticleState`; releases over time are new |
| `tracerax.velocity` | velocities at particles, `velocity(t, position)`: uniform, gridded through a functax `Interpolant` on staggered nodes, and weighted sums (a current and a share of the wind) | plumax `GriddedWind`, `uniform_wind` |
| `tracerax.advect` | one loop that runs any step; deterministic steps, on the plane and on the sphere, and the Langevin, random-displacement and inertial steps; the `TurbulenceModel` and `Diffusivity` protocols; walls and the domain | plumax `langevin_step`, `integrate_particles`, `n_steps_for_horizon`, `step_durations`; the rest is new |
| `tracerax.bin` | cells from edges, deposits that keep the mass at walls, accumulated residence, concentrations | plumax `bin_positions` and the accumulation in `simulate_lagrangian` |
| `tracerax.footprint` | backward runs from receptors and surface footprints | plumax `compute_footprint` |
| `tracerax.analysis` | FTLE, residence time and connectivity of labelled regions, absolute and pair dispersion, the skill of a trajectory | new; the stub in xrops `metrics/_src/lagrangian.py` wraps it |

### What tracerax owns, and what it hands off

The test for every name: is its state a set of positions carried forward in time, or does it analyse such positions? Then tracerax owns it.

| Neighbour | tracerax owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| integrix | the right-hand side, the loop and the boundaries | solvers, adjoints through time, semi-Lagrangian schemes | departure points that are discarded are integrix's; positions kept from step to step are tracerax's |
| gridax, spectraldiffx | particles that persist | operators on fields | a field's state stays on its grid |
| plumax | the particle machinery | turbulence closures (`HannaTurbulence`, σ\_w(z) and τ\_L(z) profiles), plume physics, source inversions | particles take the closure as an argument |
| functax | where the particles go, the cells | reading a grid at points and its transpose, axes with edges, cell areas (`quad.cell_area`) | tracerax calls functax interpolants and plans |
| geophysix | paths on the sphere | the sphere's radius and the great-circle distance | passed as an `Ellipsoid` record, `earth=` |
| gridax | — | staggered coordinates and masks of a model's grid | an adapter behind the `gridax` extra, when a consumer needs it |
| diagnostix | Lagrangian analysis and trajectory skill | Eulerian diagnostics of fields | FTLE is tracerax's; vorticity is diagnostix's |
| xrops | the maths | DataArray views of trajectories ([geotoolz#696](https://github.com/jejjohnson/geotoolz/issues/696)) | xrops wraps tracerax |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| plumax `lagrangian/_src/particles.py` | `ParticleState`, `langevin_step`, `integrate_particles`, `n_steps_for_horizon`, `step_durations`, the `TurbulenceModel` protocol | `tracerax.particles`, `tracerax.advect`; the closures stay and satisfy tracerax's protocol |
| plumax `lagrangian/_src/wind_interp.py` | `GriddedWind` and its helpers | `tracerax.velocity.GriddedVelocity`; the functions that build one from meteorology stay |
| plumax `lagrangian/_src/concentration.py` | `bin_positions`, the generic loop of `simulate_lagrangian` | `tracerax.bin`; plumax keeps a plume-specific `simulate_lagrangian` built on it |
| plumax `lagrangian/_src/footprint.py` | `compute_footprint` | `tracerax.footprint.surface`, behind a thin call |
| xrops | the empty Lagrangian metrics stub | `tracerax.analysis`, wrapped by xrops |

## In use

The plume both ways: release particles at the source and bin them into a concentration field; release them backward from a satellite pixel and read its surface footprint.

```python
# Shapes: N particles; Nx × Ny × Nz cells, Nx × Ny of them at the surface

import jax
import jax.numpy as jnp
from plumax.lagrangian import HannaTurbulence
from tracerax.advect import Langevin, Walls, integrate
from tracerax.bin import Cells, concentration
from tracerax.footprint import surface
from tracerax.particles import at_point, release_times, with_turbulent_velocity
from tracerax.velocity import GriddedVelocity

start_key, forward_key, backward_key = jax.random.split(jax.random.key(0), 3)

# today (tracerax)
# u_d(t, x) = ∑ₛ wₛ c_{d, iₛ}, each component on its own staggered grid
wind = GriddedVelocity.fit(grids, (u, v, w))
# today* (plumax)
# σ(z), τ_L(z) from boundary-layer scaling, as an eqx.Module
turbulence = HannaTurbulence(...)
walls = Walls(bottom=0.0, top=boundary_layer_height)
cells = Cells((x_edges, y_edges, z_edges))

# --- forward: where the gas goes ---
# today (tracerax)
# one release time per particle
state = at_point(  # N particles
    source, N, mass=rate * duration, release_time=release_times(0.0, duration, N)
)
# today (tracerax)
# u' ~ 𝒩(0, σ(z)²)
state = with_turbulent_velocity(state, turbulence, key=start_key)
# today (tracerax)
# du' = −u'/τ_L dt + ½ ∂σ_w²/∂z (1 + u'²/σ_w²) dt + (2σ_w²/τ_L)^½ dW
result = integrate(
    state,
    Langevin(wind, turbulence, walls),
    t0=0.0,
    duration=duration,
    step_size=2.0,
    key=forward_key,
)
# today (tracerax)
# c_j = ∑ₚ mₚ 1[xₚ ∈ j] / V_j
field = concentration(result.state, cells, t=duration)  # → (Nx, Ny, Nz)

# --- backward: what the pixel could have seen ---
# today (tracerax)
# F_j = ∑ₚ ∫ 1[xₚ(t) ∈ j, zₚ(t) < h_s] dt / (N ρ h_s)
footprint = surface(
    receptor,
    wind,
    turbulence,
    Cells((x_edges, y_edges)),
    n_particles=N,
    receptor_time=duration,
    duration=600.0,
    step_size=2.0,
    surface_depth=100.0,
    air_density=1.2,
    key=backward_key,
    walls=walls,
)
sensitivity = footprint.sensitivity  # → (Nx, Ny)
```

The footprint times a prior flux map is the pixel's predicted enhancement, which is what an inversion in plumax fits. Because the run is a JAX function, the same footprint is differentiable in the wind and the closure's parameters.

## Integration across the stack

| Consumer | Uses today | With tracerax |
| --- | --- | --- |
| plumax | its own particle state, wind, Langevin step, binning and footprint | tracerax for all five; keeps the turbulence closures, the plume-specific simulation, the ensembles, the emulator and the inversions |
| sesmax-ocean (planned) | — | drifters, inertial particles, retention in eddies and connectivity on its own velocity fields |
| xrops | an empty Lagrangian stub | DataArray views of trajectories and of `tracerax.analysis` |

For plumax the change is an import, and the argument order of a wind, which becomes `velocity(t, position)`:

```python
# Today:
# today (plumax)
from plumax.lagrangian.particles import ParticleState, langevin_step

# After:
# today (tracerax)
from tracerax.particles import ParticleState

# today (tracerax)
from tracerax.advect import langevin_step
```

## Decisions and open questions

All decided but decision 7. The owner took decisions 1 to 6 and 8 to 12 on 23 Nov 2026, after a scan of 25 stack repositories; the measurements are in [tracerax#14](https://github.com/jejjohnson/tracerax/issues/14). Each entry gives the position first, then the trade-off.

### 1. Layer and dependencies

**Decided: GeoNumerics tier 2, on functax, integrix and diffrax; geophysix for the sphere; gridax as an extra; nothing else from the stack.** The cells and their volumes come from functax (`Axis` edges, `quad.cell_area`), so a grid library is not needed to bin particles. GeoStack may import GeoNumerics ([geostack#306](https://github.com/jejjohnson/geostack/issues/306)), so xrops may wrap tracerax.

### 2. The line with the Eulerian libraries

**Decided.** A scheme that traces back to departure points and then discards them is integrix's (`integrix.march.semi_lagrangian_step`). tracerax owns everything whose state is the positions.

### 3. Turbulence closures stay in plumax

**Decided: tracerax defines the protocol.** Particles take σ(z) and τ\_L(z) through `TurbulenceModel`, any object with `at(height)`, and plumax's closures satisfy it. So sesmax-ocean can write its own closures without plumax.

### 4. Stepping through integrix or a loop of its own

**Decided: one loop, which runs any step.** Deterministic steps use integrix's and diffrax's solvers. The Langevin step reflects particles at walls, a discrete event that diffrax's SDE solvers handle awkwardly, so it is tracerax's. Every step is a `ParticleStep`, and the loop scans it with `integrix.adjoint.checkpointed_scan` or `truncated_scan`.

### 5. Trajectories stored or accumulated

**Decided: accumulate by default.** A footprint or a concentration needs only running sums, an `Accumulator` inside the scan; storing K × N × 3 positions is optional, every few steps.

### 6. Randomness and time

**Decided.** One key per run, split up front into one key per step, or the noise itself passed in. A run has a start that may be traced and a static signed duration, negative for a backward run; times inside it are offsets, because float32 resolves only 128 s at Unix-epoch seconds.

### 7. Coastlines and other boundaries

**Open, for Wave 4.** The ground and the boundary-layer top fold a particle any number of times. A coastline waits for an ocean consumer: on unstaggered grids the velocity is extended onto land with a slip coefficient. Putting a particle back on the nearest wet cell was measured and made coasts worse, so it is not planned.

### 8. Diffusion

**Decided.** A random displacement with the drift ∂K/∂x, without which particles gather where K is small at every step size. The closures that give K stay upstream.

### 9. Inertial particles

**Decided.** The Maxey–Riley equation is a step with the object's velocity as its memory, forward only, and its slow manifold is a velocity, which runs backward and on the sphere.

### 10. Trajectory skill

**Decided: tracerax's, so that it can be a loss.** Scores of an ensemble are not particle calculations, and are not here.

### 11. Gaps found in functax

**Decided: issues there, worked around here.** Nearest binning ignores an axis's edges ([functax#168](https://github.com/jejjohnson/functax/issues/168)) and linear binning drops the outer half cells ([functax#169](https://github.com/jejjohnson/functax/issues/169)).

### 12. JAX only

**Decided.** functax's interpolation and integrix both require JAX. Nothing inside the library calls `jax.jit`; the caller does, once, at the top.

## Roadmap

Five waves, each a GitHub issue epic in the [tracerax repository](https://github.com/jejjohnson/tracerax). The umbrella is [geostack#304](https://github.com/jejjohnson/geostack/issues/304).

| Wave | Delivers | Ends | State |
| --- | --- | --- | --- |
| [0. Scaffold and decisions](https://github.com/jejjohnson/tracerax/issues/8) | the repository from the library template; CI, capability index; reference tests against plumax's stored outputs | — | done, v0.0.1 |
| [1. Particles and advection](https://github.com/jejjohnson/tracerax/issues/9) | `tracerax.particles`, `tracerax.velocity`, `tracerax.advect` | plumax's particle state, wind and Langevin step | done, v0.0.2 |
| [2. Deposits and footprints](https://github.com/jejjohnson/tracerax/issues/10) | `tracerax.bin`, `tracerax.footprint` | plumax's binning and footprint | done, v0.0.3 |
| [3. The sphere and trajectory analysis](https://github.com/jejjohnson/tracerax/issues/11) | particles on the sphere and inertial ones; FTLE, residence time, connectivity, dispersion, trajectory skill | the xrops stub | done, v0.0.4 |
| [4. Consumers switch](https://github.com/jejjohnson/tracerax/issues/12) | plumax onto tracerax; xrops' Lagrangian epic; sesmax-ocean from the start; coastlines | the copies | open |

## Sources

- [The GeoStack layer map](../stack_layer_map.md): tracerax's layer, tier, imports and moves.
- @thomson1987: the well-mixed condition.
- @lin2003stilt: backward footprints.
- @haller2015lcs: FTLE and transport barriers.
- [tracerax#14](https://github.com/jejjohnson/tracerax/issues/14): the decisions, and the measurements behind them.
- Code read on 23 Nov 2026: plumax `96449d2` (`lagrangian/_src/`), xrops `metrics/_src/lagrangian.py`, integrix's semi-Lagrangian step; on 25 Nov 2026, tracerax v0.0.4.
