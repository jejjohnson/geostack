---
title: "Levels across the stack: GeoNumerics, GeoML, GeoModels"
subtitle: "Design · where L1 primitives, L2 operators and L3 algorithms sit in every package above GeoStack"
short_title: Levels across the stack
description: "Classifies every GeoNumerics, GeoML and GeoModels package by level, lists the modules that mix levels and the duplicates to remove, and settles where xrtoolz's ocean and atmosphere diagnostics belong."
date: 2026-10-22
keywords:
  - levels
  - primitives
  - somax
  - xtremax
  - plumax
  - radtrax
  - finitevolX
  - ocean diagnostics
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

How [the three levels](#design-levels) land in every package above GeoStack: which levels each package holds today, which it should hold, and what moves.

## At a glance

Most of the stack above GeoStack is L1, as it should be; carriers (L2) and runs (L3) appear only where a package may import them, which is mostly GeoModels.

| Level | GeoNumerics | GeoML | GeoModels |
| --- | --- | --- | --- |
| L3 algorithms and pipelines | none | vardax: 4DVarNet training, cycles; xtremax.simulations | plumax: inference, scenario runs; somax: monitor, run loop |
| L2 operators on carriers | none (may not import carriers) | xtremax.extraction (mixed); filterax.pipekit (inferred) | plumax: operators, met, matched filter (mixed); somax: operators, io |
| L1 primitives and machinery | gaussx, finitevolX, spectraldiffx | geonnax, kernellib, pyrox, gauss_flows, optax_bayes, vardax costs and solvers, filterax, xtremax distributions | radtrax (all of it); somax core, solvers, eval, da; plumax transport, radtran, DA |

“Mixed” marks a module that holds two levels at once today; those are the modules that move.
GeoNumerics can only ever be L1, because it may not import GeoStack's carriers.

## GeoNumerics: all L1, and it should stay that way

All three packages are primitives and the machinery around them, on raw arrays and array-only bundles; none touches a carrier.
GeoNumerics may import neither GeoStack nor anything above it, so it can never hold L2 or L3.

| Package | Holds | Level | Notes |
| --- | --- | --- | --- |
| gaussx 0.6.5 | structured lineax operators, solvers and log-dets, Gaussian distributions, Kalman/EnKF filtering, VI and EP | L1 + machinery | the linear algebra everything above uses |
| finitevolX 0.0.47 | C-grid grids and masks (array bundles), stencils, operators, diagnostics (KE, enstrophy, Okubo–Weiss, PV), time steppers, elliptic solvers | L1 + machinery | `solve_ocean_pde` borders L3 (a whole diffrax integration) |
| spectraldiffx 0.1.1 | Fourier, Chebyshev and spherical grids, derivatives, filters, Helmholtz/Poisson solvers | L1 + machinery | shares the capacitance solver and `SphericalGrid2D` with finitevolX |

What changes: nothing in level.
The grids already are the array-only bundles the primitives rule asks for (`CGrid`-style).
Their carrier-facing constructors (`cgrid_from_transform`) cannot live here; they go in the package above that uses them (somax\[geo\]).

**Demo API.**
Sketches of the target API; names follow the capability index where it has them.

```python
# L1, finitevolX: a C-grid bundle (arrays only) and two diagnostics
grid: CGrid = finitevolx.grids.cartesian(nx=256, ny=256, dx=2_000.0, dy=2_000.0)
u: Float[Array, "Ny Nx"] = ...  # zonal velocity on u-points, m s⁻¹
v: Float[Array, "Ny Nx"] = ...  # meridional velocity on v-points, m s⁻¹

# ζ = ∂v/∂x − ∂u/∂y on f-points
zeta: Float[Array, "Ny Nx"] = finitevolx.relative_vorticity_cgrid(u, v, grid)
# KE = ½(u² + v²) averaged onto T-points
ke: Float[Array, "Ny Nx"] = finitevolx.kinetic_energy(u, v, grid)

# L1, gaussx: a structured solve, no dense matrix
op: gaussx.Kronecker = gaussx.Kronecker(k_time, k_space)  # K = Kₜ ⊗ Kₛ
alpha: Float[Array, "TN"] = gaussx.solve(op, y)  # α = K⁻¹y

# never here: carrier constructors. They live in the package above that needs them
cgrid: CGrid = somax.geo.cgrid_from_carrier(u_map)  # u_map: GeoArray["y x"]
```

Evidence: the GeoNumerics capability index.
Its packages export flat, so the families in this table are grouped by name, not by module.

## GeoML: L1 with a thin L3 edge

Your intuition holds for geonnax and most of GeoML: networks, kernels, flows and inference engines are L1.
Two packages reach L3 (vardax, xtremax) and one reaches L2 by accident (xtremax).

| Package | L1 (primitives, machinery) | L2 (carriers) | L3 (algorithms, pipelines) |
| --- | --- | --- | --- |
| geonnax 0.0.9 | FNO, SFNO, WNO, UNet, SIREN, spherical-harmonic encoders, heads | — | — |
| kernellib 0.0.21 | kernels (incl. great-circle, sphere, graph), feature maps, KRR/Falkon estimators, HSIC/MMD | — | — |
| pyrox family 0.1.x | GP priors and inference, latent Gaussian models, INLA, Bayesian layers | — | — |
| gauss_flows 0.1.9 | bijections, RBIG, information measures, latent-space Kalman filters | — | — |
| optax_bayes 0.0.3 | Bayesian learning rule as optax transforms (not indexed) | — | — |
| filterax 0.0.5 | ensemble Kalman filters and smoothers (not indexed) | `filterax.pipekit` adapters (inferred) | — |
| vardax 0.1.9 | costs, observation operators, 3D/4D-Var and OI solvers, reduced bases, Lorenz toys | — | 4DVarNet training loops; `VarDACycle`, `VarSmootherCycle` |
| xtremax 0.0.4 | GEV/GPD/Gumbel, point processes | `extraction`: thresholds and block maxima **on xarray** | `simulations`: synthetic Iberia climate built into a Dataset |

What changes:

- **xtremax extraction gets primitives (decided 22 Oct 2026).**
  The missing piece was L1: extraction methods existed only as functions on xarray.
  Each becomes a primitive on arrays with time last, and the xarray versions become thin L2 operators behind `xtremax[xarray]`:
  - `block_maxima(x, block)`: `(…, T)` → `(…, T // block)`, Mₖ = max over block k;
  - `peaks_over_threshold(x, u)`: exceedances x − u and their indices where x > u;
  - `decluster_runs(idx, run)`: keeps one peak per cluster, clusters separated by more than `run` steps;
  - quantile thresholds: a fitted operator (fit u(s) by quantile regression once, then apply as a kernel), like the matched filter.
- **`xtremax.simulations` waits on its purpose** (SL-O6 below): a test fixture for the EVT methods goes to GeoBench or `xtremax.testing`; a stochastic weather generator for scenarios becomes a GeoModels package.
- **vardax keeps its L3** (cycles, 4DVarNet training) but they take readable types at their boundary (`GeoMap`, `GeoTracks`), as `vardax.geo.assimilate` does in the case studies.
  The Lorenz plotting moves out of the main namespace.
- **filterax and optax_bayes** are missing from the index (their import probably fails during generation); fixing that comes before classifying them properly.

**Demo API.**

```python
# L1, geonnax: a network is a function of arrays
net: geonnax.UNet = geonnax.UNet(in_channels=1, out_channels=1, key=key)
x: Float[Array, "1 H W"] = alpha_patch[None]
# logits = f_θ(x)
logits: Float[Array, "H W"] = net(x)

# L1, kernellib: a Matérn-3/2 on unit vectors, so distances are chordal on the sphere
kernel: kernellib.Kernel = kernellib.GreatCircleMatern(nu=1.5, length_scale=0.05)
# Kᵢⱼ = k(xᵢ, xⱼ), x ∈ 𝕊² ⊂ ℝ³
K: Float[Array, "N N"] = kernel(xo, xo)  # xo: Float[Array, "N 3"]

# L1 → L2, xtremax after the split: the same block maxima on arrays and on a cube
# Mᵧ(y, x) = max_{t ∈ year y} T(t, y, x)
annual_max: Float[Array, "Y H W"] = xtremax.block_maxima(tmax_daily, block=365)
to_annual_max: Operator = xtremax.xarray.BlockMaxima(
    dim="time", freq="YS"
)  # behind xtremax[xarray]
annual_max_cube: DataArray["year lat lon"] = to_annual_max(tmax_cube)

# L3, vardax: readable types at the boundary, raw arrays inside;
# the forward model M: GeoMap × times → GeoMapSeries comes from the caller (GeoML never imports GeoModels)
series: GeoArray["time y x"] = vardax.geo.assimilate(
    background, obs, times, forward, sigma_b=0.05, sigma_o=0.03
)
```

## GeoModels: all three levels, and where they mix

GeoModels is the only layer that legitimately holds L1, L2 and L3: it may import GeoStack carriers, GeoML and GeoNumerics.
The problem is not that it spans levels but that some modules mix them.

**somax 0.0.15** is mostly L1 by module count, with thin L2 and L3 on top; your guess of “L2, L3, maybe L1” is the other way round today.
It is also about to become a workspace of Earth-system components (ocean, atmosphere, land, ice, coupler) under a new name, because `somax` is taken on PyPI: see [From somax to an Earth-system workspace](./esmax.md).
The table below is today's somax; the [esmax page](./esmax.md) says where each module goes.

| Module | Holds | Level today | Target |
| --- | --- | --- | --- |
| `core`, `domain` | `State`/`Params`, term algebra, forcings, Helmholtz caches, spatial bases, `Domain`, `TimeDomain` | L1 | L1 |
| `solvers`, `flows` | IMEX KenCarp3, PID step control, flowjax priors | L1 machinery | L1 |
| `eval` | KE, enstrophy, divergence, balance residuals on `(model, state)` | L1 | L1, renamed where it clashes with finitevolX (`kinetic_energy`, `total_enstrophy`) |
| `da` | adapters to filterax/vardax, `SubsampleObs`, `make_ensemble` | L1 adapters (flat vectors) | L1; `SubsampleObs` folds into vardax's obs operators |
| `operators` | `SomaxModelOp`, `Burgers2DOp` | L2 (pipekit Operators) | L2 under `somax[geo]`: model steps on `GeoArray` states |
| `io` | `State` ↔ xarray `Dataset`, Zarr | L2 | L2 under `somax[xarray]` |
| `monitor`, `models` | run-loop monitors; model run loops | L3 | L3: `simulate(GeoMap) -> GeoMapSeries`, exposing its L1 loop (case study CS-F16) |

**plumax 0.1.2** mixes levels inside modules: `matched_filter` has array scores next to `apply_image_xarray` and file reading; `hapi_lut` has cross-section kernels next to xarray writers and a LUT-building pipeline; `gauss_plume`, `gauss_puff`, `les_fvm` and `population` pair forward models with NUTS or 4D-Var runs.
Each splits into `_src/array.py` (L1), an operator module (L2, behind `plumax[geo]`) and a runs module (L3).

**radtrax 0.0.0** is pure L1 today: atmosphere, spectroscopy, instrument, slab forwards, all ported from `plumax.radtran`.
It gains the L2 operators that the plume pipeline needs (`radtrax.geo.MatchedFilter`, `model_xch4`) behind `radtrax[geo]`.

The pattern across all three: L1 in `_src/array.py`, L2 behind a `[geo]` or `[xarray]` extra, L3 in a runs or pipelines module that takes readable types.

**Demo API: somax at all three levels.**

```python
# L1: the model on arrays; a step is a pure function of the state
model: somax.ShallowWater = somax.ShallowWater(depth=4_000.0, dt=600.0, grid=grid)
state: somax.State = model.init(eta0)  # eta0: Float[Array, "Ny Nx"]
# ∂η/∂t = −H∇·𝐮 ; ∂𝐮/∂t = −g∇η, one IMEX step
state_next: somax.State = model.step(state)
# ∫ ½|𝐮|² dA over the domain
total_ke: Float[Array, ""] = somax.eval.total_kinetic_energy(model, state_next)

# L2, somax[geo]: one step on a carrier; the grid is derived from its transform
step: Operator = somax.geo.Step(model)
eta_map_next: GeoArray["y x"] = step(eta_map)

# L2, somax[xarray]: a trajectory as a Dataset
trajectory: somax.Trajectory = model.run(state, n_steps=1_440)
ds: Dataset["time y x"] = somax.io.to_dataset(trajectory)

# L3, somax[geo]: a run, readable types in and out, exposing its L1 loop (model.l1) for runners
series: GeoArray["time y x"] = somax.geo.simulate(eta_map, times, model)
```

**Demo API: plumax and radtrax after the split.**

```python
# before: one call mixes array maths, xarray and file reading
alpha_da: DataArray["y x"] = plumax.matched_filter.apply_image_xarray(
    scene_path, target
)

# after, L1 in radtrax: αᵣ = x̃ᵣᵀΣ⁻¹t / tᵀΣ⁻¹t
alpha: Float[Array, "R"] = radtrax.matched_filter_score(x, mu, sigma, t)

# after, L2 in radtrax[geo]: fitted per column, on a swath
mf: Operator = radtrax.geo.MatchedFilter(target=t, dim="band", fit_per="col")
mf_fitted: Operator = mf.fit(swath)  # swath: GeoSwath["band row col"]
alpha_swath: GeoSwath["row col"] = mf_fitted(swath)

# after, L3 in plumax.runs: emission rate by NUTS, readable types in
posterior: Posterior = plumax.runs.infer_emission_rate(
    plume_map, winds, num_samples=2_000
)
```

## The ocean question: move xrtoolz's ocean physics to somax?

Recommended: no, not wholesale.
Almost every ocean and atmosphere function in xrtoolz is a diagnostic computable from any collocated gridded state (GLORYS, ERA5, an L4 satellite map), and somax has an equivalent for none of them.
Move only what depends on a model's discretisation or on one product.

**What xrtoolz holds today** (main at 2f6318d, 17 Oct 2026; not in the capability index):

| Group | Functions | Needs a model's discretisation? | Equivalent elsewhere |
| --- | --- | --- | --- |
| ocean kinematics | Coriolis, streamfunction, geostrophic and ageostrophic velocity, KE and EKE, vorticities, divergence, strain, Okubo–Weiss, advection, frontogenesis, barotropic PV | no: collocated lon/lat grids (staggered grids explicitly unsupported) | finitevolX has C-grid versions; somax.eval has domain integrals only |
| ocean thermodynamics | TEOS-10 density, N², mixed-layer depth | no | none (finitevolX has a linear density only) |
| atmosphere | wind speed and direction, column integral, hypsometric height, PBL height | no | none |
| budgets | control-volume integrals, boundary fluxes, heat/salt/volume/KE residuals | no in code; yes for exact closure | none |
| CH₄ columns | averaging kernel, dry-air column, mixing ratio ↔ column | no | duplicated in `plumax.coupled` |
| products | SSH composition from altimetry, SSH/velocity validation, WRF reader and destaggering | product- or model-specific | `plumax.met.load_wrf` overlaps the WRF reader |

**Options:**

|  | What moves | Cost |
| --- | --- | --- |
| A. Keep in xrops (OP-O1 as decided) | nothing; restyle into L1 primitives (`xrops/ocn/_src/array.py`) and L2 xarray operators | none new |
| B. Move to somax | L1 into somax, L2 under `somax[xarray]` | computing MLD on GLORYS needs JAX, equinox and diffrax; GeoStack cannot import somax, so all ocean analysis leaves GeoStack and somax becomes an analysis library |
| C. Move L1 to finitevolX | collocated kernels next to its C-grid diagnostics, xarray wrappers in somax | same user cost as B; GeoNumerics gains collocated-grid code |
| **D. Split by discretisation** (recommended) | A, plus: model-consistent budget closure → somax.eval (via finitevolX C-grid stencils); WRF reader and SSH product composition → xrproducts; CH₄ column physics → radtrax | small; each piece goes to the package that owns its reason to change |

The rule behind D: **same physics on a different discretisation is not a duplicate; same physics on the same discretisation is.**
Relative vorticity on a collocated reanalysis grid (xrops) and on a model's C-grid (finitevolX) are two primitives, each correct for its grid.
Two TEOS-10 densities on the same grid would be a duplicate.

What somax should own instead: diagnostics of its own states on its own grid (today's `somax.eval`), the run loop as an L3 `simulate`, and L2 operators that step a model on a `GeoArray` state.

**Demo API under option D.**
Where each ocean call lives once the split is done.

```python
# xrops L1 (xrops/ocn/_src/array.py): MLD from one profile, any collocated grid
# MLD = z at the first level where σ₀(z) − σ₀(z_ref) > 0.03 kg m⁻³
mld_profile: Float[Array, ""] = mixed_layer_depth(sigma0_profile, depth, threshold=0.03)

# xrops L2: the same kernel on GLORYS, lazy; no JAX, no somax
to_mld: Operator = xrops.ocn.MixedLayerDepth(threshold=0.03, dim="depth")
mld: DataArray["time lat lon"] = to_mld(sigma0_cube)

# relative vorticity on a reanalysis grid: xrops, collocated
to_zeta: Operator = xrops.ocn.RelativeVorticity()
zeta_glorys: DataArray["time lat lon"] = to_zeta(uv_cube)

# relative vorticity of a somax state: finitevolX on the C-grid, reached through somax.eval
zeta_model: Float[Array, "Ny Nx"] = somax.eval.relative_vorticity(model, state)

# exact heat-budget closure on the model grid: somax.eval, using the model's own fluxes
# R = ∂θ/∂t + ∇·(𝐮θ) − Q, zero to round-off
residual: Float[Array, "Ny Nx"] = somax.eval.heat_budget_residual(
    model, state, state_next
)

# WRF: one reader in xrproducts; plumax and xrops both consume it
wrf: Dataset["time level y x"] = xrproducts.wrf.open("wrfout_d01_2026-07-14_00:00:00")

# CH₄ columns: L1 in radtrax (SL-O3 decides where the xarray version lives)
delta_vmr: Float[Array, "H W"] = radtrax.column_mass_to_delta_vmr(
    column_mass, dry_air_column
)
```

## Overlaps and the moves they imply

Each row is a duplicate found in the capability indexes (or, for xrtoolz, its source); the move keeps one copy in the lowest package that owns the concept.

| Overlap | Copies today | Keep | Move or remove |
| --- | --- | --- | --- |
| matched filter | `plumax.radtran.matched_filter_*`, `plumax.matched_filter`, `geoops.MatchedFilter*` | radtrax: L1 `matched_filter_score`, L2 `radtrax.geo.MatchedFilter` | retire both plumax copies (plumax#131) and the geoops one (RTX-30) |
| radiative transfer | `plumax.radtran` + `plumax.hapi_lut` vs radtrax (about 25 names) | radtrax | plumax re-exports nothing; it imports radtrax (plumax#133) |
| CH₄ column physics | `plumax.coupled.column_mass_to_delta_vmr`, `xrtoolz.atm.gas.ch4.*` | radtrax L1 | open: xrops cannot import radtrax, so the xarray versions need a home (SL-O3 below) |
| WRF reading | `plumax.met.load_wrf`, `xrtoolz.atm.open_wrfout` | xrproducts | both callers read through xrproducts |
| 3D-/4D-Var | `plumax.assimilation`, `plumax.les_fvm.solve_4dvar` vs vardax (inferred) | vardax | plumax supplies the forward model and cost terms only |
| linear Gaussian inversion | `plumax.lagrangian.linear_gaussian_inversion` vs `vardax.OptimalInterpolation` (inferred) | vardax | — |
| reduced bases | `vardax.*_basis` vs `somax.core.spatial_from_*` | vardax | somax imports them |
| observation subsampling | `somax.da.SubsampleObs` vs vardax `LinearObs`/`MaskedIdentity` | vardax | — |
| GP kernels and likelihoods | kernellib and pyrox_gp kernels; gaussx and pyrox_gp likelihoods; a fourth Matérn in plumax | kernellib (kernels), gaussx (likelihoods) | pyrox_gp imports them |
| neural layers | geonnax vs pyrox_nn (rank-1, ensemble, MC layers, RFF GP) | geonnax | pyrox_nn adds only the Bayesian parts |
| spectral pieces | capacitance solver, `SphericalGrid2D` in finitevolX and spectraldiffx; `SphericalHarmonicTransform` in geonnax and spectraldiffx | spectraldiffx | finitevolX and geonnax import it |
| KE and enstrophy names | finitevolX, somax.eval, xrtoolz.ocn | all three, renamed | different discretisations or integrals (the rule above); names must differ so the capability index stops flagging them |

None of these moves crosses the layer rule: every “keep” is at or below every package that uses it.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| SL-O1 | Ocean and atmosphere diagnostics | **Decided 29 Oct 2026:** D: stay in xrops as L1 + L2; model-consistent closure to somax; products to xrproducts | B: move all to somax, at the cost of a JAX install for every reanalysis user |
| SL-O2 | Where somax's carrier code lives | `somax[geo]` (GeoArray steps, `simulate`) and `somax[xarray]` (`io`) extras; in the workspace plan, `esmax-core[geo]` and `[xarray]` | in the base install |
| SL-O3 | CH₄ column physics on xarray | L1 in radtrax; xarray operators behind `radtrax[xarray]` | keep an xrops copy for analysts, accepting one duplicate across layers |
| SL-O4 | L3 runs in plumax (NUTS, 4D-Var, scenario grids) | a `plumax.runs` module per model, taking readable types | leave them next to the forward models |
| SL-O5 | filterax and optax_bayes missing from the index | fix the import failure in `scripts/capabilities.py` before classifying them | classify from source |
| SL-O6 | What `xtremax.simulations` is for | if it tests EVT methods with known return levels: GeoBench or `xtremax.testing`; if it generates scenarios: a GeoModels package, once it is more than one module | leave it in xtremax as is |
| SL-O7 | somax's future | the Earth-system workspace on the [esmax page](./esmax.md) (`esmax`, one package per component, a coupler on top) | keep one ocean-only package under a new name |

Evidence: the three capability indexes, `KNOWN_CLASHES` in `scripts/capabilities.py`, and a read of xrtoolz main for its ocean and atmosphere modules.
The library sources under `references/` were not checked out, so module internals marked “inferred” were not read.
