---
title: "From somax to an Earth-system workspace: plan and names"
subtitle: "Design · esmax: a core, one package per component, and a coupler"
short_title: esmax workspace
description: "Plans somax's growth into an Earth-system workspace for data assimilation, not climate simulation (core, ocean, atmosphere, land, ice, coupler), the names checked on PyPI, where each somax module goes, and phases with gates."
date: 2026-10-22
keywords:
  - somax
  - esmax
  - ocean
  - atmosphere
  - land
  - sea ice
  - coupler
  - workspace
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Motivation: models for data assimilation, not climate simulation

esmax is for data assimilation.
Its models are the forward models inside an assimilation window, a forecast step or an emulator's training loop, run over a fixed horizon of hours to weeks (a season at most).
They are not climate models: nothing here has to run stably for 100 years, so the requirements that come with long simulations do not apply.

That narrows the scope:

|  | In scope | Out of scope |
| --- | --- | --- |
| Horizon | a window of hours to weeks, and the forecast after it | multi-decade and centennial runs, spin-up to equilibrium |
| Conservation | budgets closed within the window, to the tolerance the cost function needs | drift-free energy and water over centuries |
| Physics | simple dynamics that are good enough over the window: shallow water, QG, slab and bucket models | full parameterisation suites, CMIP-style coupling |
| Learned parts | learned closures, corrections and forcings, each with a prior | — |
| Speed that matters | gradients through the window (4D-Var), ensembles (EnKF, `vmap`), batched training | throughput of one long trajectory |

In a 4D-Var window the controls are the initial state x₀, the learned parameters θ and the forcing f, each with a prior (background x_b, θ_b, f_b; covariances B, Σ_θ, Σ_f).
With observations yₖ, observation operators Hₖ, observation-error covariances Rₖ, and xₖ the model state at step k of K:

```{math}
:label: esmax-4dvar-cost
J(x_0, \theta, f) = \tfrac12 \lVert x_0 - x_b \rVert^2_{B^{-1}}
  + \tfrac12 \lVert \theta - \theta_b \rVert^2_{\Sigma_\theta^{-1}}
  + \tfrac12 \lVert f - f_b \rVert^2_{\Sigma_f^{-1}}
  + \tfrac12 \sum_{k=0}^{K} \lVert y_k - H_k(x_k) \rVert^2_{R_k^{-1}}
```

What follows for the package:

- **Differentiability and batching come first.**
  Every L1 step is a pure function of state, parameters and forcing that `jax.grad` and `vmap` go through over the whole window.
- **Learned components and learned forcings are first-class.**
  In data assimilation the model is often part physics, part learned: a learned closure for unresolved terms, or a forcing (wind stress, heat flux) estimated with a prior instead of read from a file.
  `esmax-core` gives both a typed slot that carries its prior, so either can be a control variable in vardax or an augmented state in filterax.
- **Stability is judged over the window.**
  A model needs to be stable and accurate over the horizon it is assimilated over, plus a margin.
  Cycled assimilation still chains many windows, so the bar is “stable across a cycle of windows”, not “stable for 100 years”.
  Gates use window-length runs, not years.
- **Short horizons still need closed budgets.**
  Spurious drift inside a window biases the analysis, so conservation stays a test; it is measured over the window.

## Decision

somax (“simple ocean model in JAX”) becomes a workspace of simple Earth-system components: a shared core, one package per component (ocean, atmosphere, land, ice), and a coupler on top.
Today's somax becomes the ocean component plus the core; atmosphere comes next; land and ice are added only when they have real code.

Why now:

- **The name stops fitting.** “Simple ocean model” does not describe an atmosphere, land or ice component.
- **The name is not available on PyPI.**
  `somax` is registered by IRCAM's Somax2, a music-improvisation library (version 0.1.2 on 22 Oct 2026).
  Publishing this stack under that name is not possible.
- **One package would not stay healthy.**
  Coupling is a package's worth of work on its own, and land (column physics) and sea ice (rheology plus thermodynamics) share the core but little else with the ocean.
  Separate packages can ship and be tested alone, as the pyrox workspace does.

The workspace keeps the levels rule from the [levels doc](./07_stack_levels.md): every component has L1 dynamics on arrays, L2 operators on carriers behind a `[geo]` extra, and an L3 `simulate` that takes readable types.

## Names

Recommended: **esmax** (“Earth-system models in JAX”), as a workspace named like pyrox: one distribution per package, `esmax-<part>`, imported as `esmax_<part>`.

| Option | Umbrella | Fits land and ice? | On PyPI (22 Oct 2026) | Notes |
| --- | --- | --- | --- | --- |
| **A. esmax** (recommended) | `esmax` | yes | free | reads as ESM + JAX; matches the stack's `-ax` names (plumax, radtrax, vardax) |
| B. fluidx | `fluidx` | no: land is not a fluid | free | your first idea; names three of four components |
| C. keep somax | `somax` | no | **taken** (IRCAM Somax2) | could stay a GitHub name only; never installable as `somax` |
| D. one name per component | `oceax`, `atmax`, `landax`, `icax` | yes | all free | four unrelated names, no umbrella; the coupler has nowhere obvious to live |

Also checked and taken: `terrax` (Google's NeuralGCM group, “AI-first Earth System Modeling”) and `climax` (an argparse wrapper).
Only PyPI was checked, not GitHub or conda-forge.

The names under option A:

| Distribution | Import | Holds | Tier in the workspace |
| --- | --- | --- | --- |
| `esmax-core` | `esmax_core` | `State`, `Params`, term algebra, domains, steppers, DA adapters, slots for learned closures and forcings with their priors, monitors, xarray I/O | 0 |
| `esmax-ocean` | `esmax_ocean` | today's somax ocean models: shallow water, QG, Burgers toy | 1 |
| `esmax-atmos` | `esmax_atmos` | simple atmosphere models | 1 |
| `esmax-land` | `esmax_land` | bucket hydrology, surface energy balance (when there is code) | 1 |
| `esmax-ice` | `esmax_ice` | thermodynamic slab sea ice, then rheology (when there is code) | 1 |
| `esmax-couple` | `esmax_couple` | flux exchange, regridding between components, coupled `simulate` | 2 |
| `esmax` | `esmax` | code-free umbrella that installs the set, as `geotoolz` does | — |

The GitHub repository `jejjohnson/somax` is renamed to `esmax` (GitHub redirects the old URL), and `esmax-ocean` 0.1.0 is the continuation of somax 0.0.15.

## The workspace

Three tiers inside one workspace: the core at the bottom, one package per component above it, and the coupler on top; arrows point to what a package imports.

```text
esmax-couple         tier 2: flux exchange, regridding,
     │                       sub-cycled steps, coupled simulate
     ▼
esmax-ocean  esmax-atmos  esmax-land  esmax-ice      tier 1
(today's     (next, P2)   (later, P4) (later, P4)
 somax)
     │
     ▼
esmax-core           tier 0: State, term algebra, steppers,
                             DA adapters, monitors,
                             Step [geo], xarray I/O [xarray]

imports below: finitevolX, spectraldiffx, gaussx (GeoNumerics),
vardax, filterax (GeoML), geoarray only behind [geo]
```

Dashed components start only when they have real code.
The whole workspace sits in GeoModels, so it may import GeoML, GeoNumerics and, behind `[geo]`, GeoStack's carriers.

## Where today's somax goes

Every somax 0.0.15 module has one destination.
Model-agnostic machinery goes to the core; anything that knows it is an ocean goes to the ocean component; duplicates of GeoML go back to GeoML.

| somax module | Holds today | Goes to | Change |
| --- | --- | --- | --- |
| `core` | `State`, `Params`, term algebra, forcings, Helmholtz caches | `esmax-core` | none |
| `core` spatial bases | `spatial_from_eof/fourier/rbf/wavelet` | vardax | removed; import `vardax.*_basis` (duplicate) |
| `domain` | `Domain`, `TimeDomain` | `esmax-core` | none |
| `solvers` | IMEX KenCarp3, PID step control | `esmax-core` | none |
| `flows` | flowjax bijection, learned prior | `esmax-core.da` | grouped with the DA adapters |
| `da` | adapters to filterax and vardax, `make_ensemble` | `esmax-core.da` | `SubsampleObs` removed; use vardax's observation operators |
| `eval` | KE, enstrophy, divergence, balance residuals | split | generic residual machinery to `esmax-core.eval`; ocean diagnostics to `esmax-ocean.eval`, renamed off finitevolX's names |
| `operators` | `SomaxModelOp`, `Burgers2DOp` | `esmax-core.geo`; Burgers to `esmax-ocean.toys` | the generic model operator becomes `Step`, behind `esmax-core[geo]` |
| `io` | `State` ↔ xarray `Dataset`, Zarr | `esmax-core` behind `[xarray]` | none |
| `monitor` | run-loop monitors | `esmax-core` | none |
| `models` | the model classes (not in the capability index) | `esmax-ocean`; any atmosphere models to `esmax-atmos` | split by component; the split is inferred until the module is read |

Nothing moves into GeoNumerics: finitevolX and spectraldiffx stay the stencils and solvers every component imports.

## Each component at L1, L2 and L3

Every component has the same three-level shape, so a user who knows one knows them all.

| Component | L1 (arrays) | L2 (`[geo]`, carriers) | L3 (readable types) |
| --- | --- | --- | --- |
| ocean | shallow water, QG, tracer advection on a C-grid | `Step` on `GeoArray["y x"]` states | `simulate(GeoMap) -> GeoImageSeries` |
| atmosphere | 2-layer QG, shallow water on the sphere | same | same |
| land | bucket hydrology, surface energy balance per column | `Step` on `GeoArray["y x"]` soil-moisture maps | same |
| ice | zero-layer thermodynamic slab ice, then EVP rheology | `Step` on ice-thickness maps | same |
| coupler | bulk flux formulae, regridding weights | — | coupled `simulate` over several components |

The model lists for atmosphere, land and ice are starting points, not commitments.

**Demo API: the ocean component.**
Sketches of the target API.

```python
# L1, esmax_ocean: a model on arrays and a grid bundle
grid: CGrid = finitevolx.grids.cartesian(nx=256, ny=256, dx=4_000.0, dy=4_000.0)
model: esmax_ocean.ShallowWater = esmax_ocean.ShallowWater(
    depth=4_000.0, dt=600.0, grid=grid
)
state: esmax_core.State = model.init(eta0)  # eta0: Float[Array, "Ny Nx"]
# ∂η/∂t = −H∇·𝐮 ; ∂𝐮/∂t + f𝐳×𝐮 = −g∇η, one IMEX step
state_next: esmax_core.State = model.step(state)

# L2, esmax_core[geo]: one step on a carrier; the grid comes from its transform
step: Operator = esmax_core.geo.Step(model)
eta_next: GeoArray["y x"] = step(eta_map)  # eta_map: GeoArray["y x"]

# L3: a run, readable types in and out; model.l1 is the raw loop for runners and vmap
series: GeoArray["time y x"] = esmax_ocean.simulate(eta_map, times, model)

# xarray I/O lives in the core, behind [xarray]
ds: Dataset["time y x"] = esmax_core.io.to_dataset(series)
```

**Demo API: land, once it exists.**
Same shape, different physics.

```python
# L1, esmax_land: a bucket model per column, vectorised over the grid
bucket: esmax_land.Bucket = esmax_land.Bucket(capacity=0.15, dt=3_600.0)
# ∂W/∂t = P − E − R, R = max(0, W − W_max) / Δt
w_next: Float[Array, "Ny Nx"] = bucket.step(w, precip, evap)

# L3: driven by ERA5 forcing, read by value first
forcing: Dataset["time lat lon"] = xrproducts.cds.open(
    "era5_single_levels", vars=["tp", "e"], region=iberia, year=2025
)
soil: GeoArray["time y x"] = esmax_land.simulate(w0_map, forcing, bucket)
```

## The coupler

The coupler is where the work is: components exchange fluxes at their interface, on different grids and at different time steps.
It sits one tier above the components, so no component imports another.

| Job | L1 primitive | Notes |
| --- | --- | --- |
| momentum and heat fluxes | bulk formulae on arrays | wind stress τ = ρₐ C_D \|U\| U; sensible heat Q = ρₐ cₚ C_H \|U\| (T_s − Tₐ) |
| regridding between components | sparse weights applied as `einx.dot` | weights computed once at set-up from the two grids (derived geometry); conservative for fluxes, bilinear for states |
| time steps | sub-cycling inside one `lax.scan` step | the slower component steps once per k fast steps; k fixed at set-up so one trace serves the run |
| conservation | budget residuals across the interface | a coupled run must conserve the exchanged energy and water to round-off |

```python
# L1, esmax_couple: wind stress on the atmosphere grid, then onto the ocean grid
# τ = ρₐ C_D |U| U, each component (Na,)
speed: Float[Array, "Na"] = jnp.hypot(u_atm, v_atm)
tau_x_atm: Float[Array, "Na"] = RHO_AIR * C_D * speed * u_atm
# conservative remap with precomputed weights W, (No, Na)
tau_x_ocn: Float[Array, "No"] = einx.dot("o a, a -> o", w_atm_to_ocn, tau_x_atm)

# L3, esmax_couple: a coupled run, readable types in and out
coupled: esmax_couple.Coupled = esmax_couple.Coupled(
    ocean=ocean_model, atmos=atmos_model, substeps=6
)
result: CoupledSeries = esmax_couple.simulate(
    coupled, ocean_state=sst_map, atmos_state=wind_map, times=times
)
```

Start with the smallest useful coupling: an atmosphere over a slab ocean, before any dynamic ocean–atmosphere pair.

## Phases and gates

Each phase ships on its own and merges only when its gate passes.

| Phase | Scope | Gate |
| --- | --- | --- |
| P0 Rename and split | repo `somax` → `esmax`; workspace with `esmax-core` and `esmax-ocean`; the module moves above; vardax bases and observation operators imported, not copied | every somax 0.0.15 test passes under `esmax-ocean` with only import changes; `esmax-core` imports no component |
| P1 Levels | `_src/array.py` per component; `Step` and `simulate` behind `[geo]`; I/O behind `[xarray]` | the import check passes (primitive modules import no carrier); `simulate` exposes `model.l1` |
| P2 Atmosphere | `esmax-atmos` from somax's atmosphere models, or a 2-layer QG to start | runs and tests alone, with only `esmax-core` installed beside it |
| P3 First coupling | `esmax-couple`: atmosphere over a slab ocean | exchanged energy is conserved to round-off over a 30-day assimilation window, and the coupled model stays stable over a cycle of windows |
| P4 Land, then ice | when there is real code for them | each component runs and tests alone |

P0 and P1 change no science; P2 onwards adds it. geostack's `scripts/dependency_graph.py` gets the new packages and their tiers in P0.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| N1 | Umbrella name | `esmax` | `fluidx` (free, but land is not a fluid) |
| N2 | One distribution per component, or one distribution with subpackages | one per component, like pyrox | one `esmax` distribution with `esmax.ocean` etc.: simpler releases, but every user installs every component |
| N3 | When land and ice start | when there is real code (P4) | stubs now, which xrtoolz's empty `ice` module shows go stale |
| N4 | plumax's atmospheric transport (`les_fvm`) | stays in plumax; it may later run on `esmax-atmos` winds | moves into `esmax-atmos` as its tracer component |
| N5 | Synthetic climates (`xtremax.simulations`) | not in esmax: a statistical weather generator is not a fluid model (see the levels doc) | a stochastic component in esmax |

Evidence: somax's modules come from the GeoModels capability index (somax 0.0.15); the name checks are PyPI lookups on 22 Oct 2026.
