---
title: "diagnostix: diagnostics of geophysical fields"
short_title: "diagnostix"
description: "diagnostix computes what a geophysical field says, physically and statistically: kinematics (velocity gradients, advection, frontogenesis, the geostrophic residual), column quantities (N², mixed-layer depth, boundary-layer height, column integrals), budgets, climatologies and anomalies, spectra, area-weighted EOFs and skill scores. It sits at GeoNumerics tier 2, above gridax, whose differences and named diagnostics it builds on, and holds the maths that xrops wraps for xarray. It absorbs what was planned as skillax. Status: scaffold built; the public modules are planned."
date: 2026-11-05
keywords:
  - diagnostix
  - design
  - GeoStack
---

diagnostix computes what a geophysical field says, physically and statistically: kinematics (velocity gradients, advection, frontogenesis, the geostrophic residual), column quantities (N², mixed-layer depth, boundary-layer height, column integrals), budgets, climatologies and anomalies, spectra, area-weighted EOFs and skill scores. It sits at GeoNumerics tier 2, above gridax, whose differences and named diagnostics it builds on, and holds the maths that xrops wraps for xarray. It absorbs what was planned as skillax. Status: scaffold built; the public modules are planned.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoNumerics, tier 2 |
| Imports | gridax, geophysix, pipekit-array, NumPy; JAX and xarray optional |
| Imported by | xrops; sesmax-ocean (planned) |
| Takes over | the maths of xrops `ocn`, `atm` (vertical and wind), `budgets`, `anomaly`, the spectra in `transforms` and `wavelets`, and the scores in `metrics`; somax's `eval/metrics.py` |
| Does not own | the DataArray API (xrops); differences, metrics and the diagnostics gridax already names (gridax); facts at a point (geophysix); bases and generic signal processing (functax); benchmark contracts and leaderboards (pipekit-evaluate); extremes modelling (xtremax) |
| Status | the scaffold, the test harness and the shared reductions are built in [diagnostix](https://github.com/jejjohnson/diagnostix); the public modules are planned |
| Tracked in | [geostack#309](https://github.com/jejjohnson/geostack/issues/309); the [layer map](../stack_layer_map.md); the companion [gridax](gridax.md) and [functax](functax.md) designs |

```{figure} ../../images/libraries/diagnostix.svg
:alt: where diagnostix sits · GeoNumerics tier 2
:width: 100%

Where diagnostix sits · geonumerics tier 2.
```

diagnostix sits one tier above gridax, whose differences and metrics every diagnostic is built from. This page is the design; core types, algorithms and the code survey are in [diagnostix: design details](diagnostix_details.md).

## The problem

A plume's emission rate is checked with diagnostics before anyone trusts it, and every one of them lives in xrops, on xarray, in NumPy and SciPy (as read on 5 Nov 2026):

| Step | The diagnostic | Where it is coded today |
| --- | --- | --- |
| How deep the plume mixes | boundary-layer height from the bulk Richardson number | xrops `pbl_height_bulk_richardson` |
| The enhancement over background | a climatology and the anomaly from it | xrops `calculate_climatology`, `calculate_anomaly` and their smoothed forms |
| The rate by mass balance | the flux through the faces of a box around the source | xrops `boundary_flux` (which sums over every face of the field, not a chosen box), `control_volume_integral`, `budget_residual` |
| Which scales a reconstruction resolves | power spectra and the PSD score | xrops `power_spectrum`, `isotropic_power_spectrum`, `psd_score`, `resolved_scale` |
| How good the transport model is | RMSE, bias, correlation, spread–skill | xrops `metrics` (`rmse`, `bias`, `correlation`, `spread_skill_ratio`, `rank_histogram`) |
| The ocean side, for sesmax | vorticity, strain, Okubo–Weiss, geostrophy, kinetic energy | xrops `ocn/kinematics.py`; somax `eval/metrics.py`; gridax `operators/diagnostics.py` |

Because they sit on xarray, none of them traces under JAX or differentiates, so a model that wants a diagnostic inside its loop or as a loss writes it again: kinetic energy, enstrophy and geostrophic imbalance each exist in xrops, in somax and in gridax. And xrops' diagnostics run on geoops' difference kernels, so the xarray toolkit depends on the GeoArray toolkit for maths that belongs to neither.

## What a field says

A diagnostic is a function of one or more fields on a grid that returns a smaller or more telling field: a derivative combination, an integral, a residual, a spectrum, a decomposition or a score. Each is built from gridax's differences and metrics and geophysix's constants. Vorticity, the strains, Okubo–Weiss and the geostrophic velocity below are gridax's functions, on every grid it supports (decision 4); they are written here because what diagnostix owns is built from them.

**Kinematics.** With u, v the horizontal velocity and f the Coriolis parameter, vorticity and the two strains are

```{math}
\zeta = \partial_x v - \partial_y u
```

```{math}
\sigma_n = \partial_x u - \partial_y v
```

```{math}
\sigma_s = \partial_x v + \partial_y u
```

The Okubo–Weiss parameter [@okubo1970; @weiss1991] W = σₙ² + σₛ² − ζ² separates strain-dominated from rotation-dominated flow, and the geostrophic velocity from sea-surface height η is u\_g = −(g / f) ∂ᵧη, v\_g = (g / f) ∂ₓη.

**Columns.** The buoyancy frequency is N² = −(g / ρ₀) ∂ρ/∂z with ρ from geophysix's equation of state. The boundary-layer height is the lowest z where the bulk Richardson number reaches a critical value Ri\_c, with θ\_v the virtual potential temperature:

```{math}
\mathrm{Ri}_b(z) = \frac{g}{\theta_{v,0}} \frac{(\theta_v(z) - \theta_{v,0}) z}{u(z)^2 + v(z)^2}
```

**Budgets.** For a quantity q carried by the velocity u, with source S, over a control volume V with surface ∂V and outward normal n,

```{math}
\frac{d}{dt} \int_V q dV + \oint_{\partial V} q \mathbf{u} \cdot \mathbf{n} dS = \int_V S dV
```

and the residual of that balance is the diagnostic. For a plume in steady state, the flux through the faces of a box around the source is the emission rate, Q = ∮ Ω' (u · n) dl, with Ω' the column enhancement.

**Climate.** With q̄(d) the climatology at calendar day d (on a leap-year calendar, so each date keeps one slot in every year), the anomaly is q'(t) = q(t) − q̄(d(t)); trends, detrending and temporal filters act on q'.

**Spectra.** With q̂(k) the Fourier transform of q, the power spectrum is P(k) = |q̂(k)|², the isotropic spectrum averages it over rings of |k|, and the spectral flux Π(k) sums the nonlinear transfer below k. Cross, coherence and rotary spectra and wavelets follow the same pattern.

**EOFs.** With X the anomaly matrix (time × space) and W the diagonal of cell areas from gridax, the area-weighted EOFs come from the SVD of the weighted anomalies:

```{math}
W^{1/2} X^\top = U \Sigma V^\top
```

The EOFs are W^(−1/2) U, the principal components VΣ, and the i-th explains a fraction σᵢ² / ∑ⱼ σⱼ² of the variance.

**Skill.** With ŷ a prediction and y the truth over valid cells with weights w, RMSE = (∑ w (ŷ − y)² / ∑ w)^½, the bias, the correlation, and for ensembles the spread–skill ratio and the rank histogram.

## Modules and boundaries

Seven modules, each created by its first feature issue. Functions over a grid or an axis take plain arrays, and a DataArray there raises an error; elementwise functions keep a DataArray's labels. xrops keeps the names, attributes and units.

| Module | Owns | Comes from |
| --- | --- | --- |
| `diagnostix.kinematics` | the velocity gradient, shear and curvature vorticity, advection, frontogenesis, the ageostrophic velocity, the geostrophic residual | xrops `ocn/kinematics.py`, `metrics/_src/physical.py` |
| `diagnostix.columns` | N², lapse rate, first crossings, mixed-layer depth, boundary-layer height, column integrals | xrops `ocn/kinematics.py` (N², mixed layer), `atm/_src/vertical.py` |
| `diagnostix.budgets` | control volumes, boundary fluxes, budgets and their residuals | xrops `budgets/`; the flux through a region's own boundary is new |
| `diagnostix.climate` | climatologies, anomalies, trends, running means | xrops `anomaly/`; trends are new |
| `diagnostix.spectra` | power, isotropic, cross, coherence, rotary and wavelet spectra, spectral fluxes, slopes | xrops `transforms/_src/fourier.py`, `wavelets/_src/` |
| `diagnostix.eofs` | area-weighted EOFs and their reading: explained variance, principal components, truncation | xrops `transforms/_src/decompose.py` (`eof`), which is unweighted scikit-learn PCA; the area weighting is new |
| `diagnostix.skill` | pointwise, ensemble, spectral and physical scores: RMSE, bias, correlation, CRPS, spread–skill, rank histogram, PSD score, resolved scale, geostrophic imbalance | xrops `metrics/`; somax `eval/metrics.py` |

### What diagnostix owns, and what it hands off

The test for every name: does it take a field and say something about it, physically or statistically, and does no library below already name it? Then diagnostix owns it. A one-line combination of functions the stack has (absolute vorticity, eddy kinetic energy) is written by the caller.

| Neighbour | diagnostix owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| xrops | the maths | the DataArray API: names, attributes, units, dask chunking | xrops calls diagnostix kernels |
| gridax | the diagnostics gridax does not name | grids, masks, derivatives, area means, and the diagnostics it names: curl, divergence, kinetic energy, strain, Okubo–Weiss, geostrophic velocity, shallow-water potential vorticity | diagnostix builds on them and defines no second name (decision 4) |
| geophysix | whatever differentiates, integrates or averages a field | facts at a point: density, θ, wind speed, Coriolis, constants | a fact at a point is geophysix's |
| functax | the EOFs, their area weights and their reading | bases and generic signal processing | functax's `eof_basis` is the oracle (decision 6) |
| tracerax | Eulerian diagnostics of fields | Lagrangian analysis and trajectory skill | vorticity is Eulerian, FTLE is tracerax's |
| pipekit-evaluate | the scores | benchmark contracts, evaluation protocols, leaderboards | a score is a function here; a benchmark that runs it is pipekit's |
| xtremax | anomalies and climatologies | extremes modelling: GEV, GPD, return levels | xtremax fits extremes to what diagnostix extracts |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| xrops | the maths of `ocn`, `atm/_src/vertical.py`, `budgets`, `anomaly`, the spectra in `transforms` and `wavelets`, and `metrics` | diagnostix and gridax underneath; xrops keeps every DataArray function and Operator, and `metrics/leaderboard.py` goes to pipekit-evaluate |
| somax | `eval/metrics.py`: kinetic energy, enstrophy, RMS divergence, geostrophic imbalance, QG balance residual | gridax, `diagnostix.kinematics`, `diagnostix.skill`; sesmax-core inherits nothing to move |
| the skillax plan | skill scores | `diagnostix.skill` (decision 2) |

## In use

Checking a plume's emission rate: how deep it mixes, its enhancement over background, its rate by mass balance, and how well the transport model reproduces it. Every diagnostix name is proposed, by the issues of Waves 1, 2 and 4; jitting is the caller's.

```python
# Shapes: Z levels; T times; Ny × Nx lon/lat cells

import gridax
from diagnostix import budgets, climate, columns, skill

# today (gridax)
grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon)
cell_area = gridax.area_weights(grid)  # → (Ny, Nx)

# --- how deep the plume mixes ---
# proposed (diagnostix#22)
# lowest z where Ri_b(z) = g (θ_v − θ_vs) (z − z_s) / (θ_vs |u − u_s|²) reaches Ri_c
mixing_height = columns.boundary_layer_height(  # (Z, Ny, Nx) → (Ny, Nx)
    virtual_potential_temperature, eastward_wind, northward_wind, height, axis=0
)

# --- enhancement over background ---
# proposed (diagnostix#25)
# d, the calendar day on a leap-year calendar, 1 … 366
day = climate.calendar_day(month, day_of_month)  # → (T,)
# Ω̄(d), the mean of each calendar day
daily = climate.climatology(column, day, axis=0)  # (T, Ny, Nx) → (366, Ny, Nx)
# proposed (diagnostix#26)
# Ω' = Ω − Ω̄(d(t))
enhancement = climate.anomaly(column, daily, day, axis=0)  # → (T, Ny, Nx)

# --- the rate by mass balance ---
# proposed (diagnostix#23)
# a boolean region, made into signed boundary faces once, on the host
box = budgets.ControlVolume.from_region(source_region, grid)  # (Ny, Nx) → volume
# Q = ∮ Ω' (u · n) dl over the faces of the box
rate = budgets.boundary_flux(  # (T, Ny, Nx) → (T,)
    enhancement, eastward_wind_pbl, northward_wind_pbl, grid, box
)

# --- skill of the transport model ---
# proposed (diagnostix#12)
# RMSE = (∑ w (ŷ − y)² / ∑ w)^½, with w the cell areas
error = skill.rmse(  # (T, Ny, Nx) → (T,)
    simulated, enhancement, weights=cell_area, axis=(-2, -1)
)
```

Each call is a pure function of the array API standard, so it runs on NumPy for xrops, and the same `rmse` and `boundary_flux` trace and differentiate under JAX inside a training loss for an emulator of the transport model, where today they cannot.

## Integration across the stack

Two libraries would import diagnostix, and one model's metrics retire into it.

| Consumer | Uses today | With diagnostix |
| --- | --- | --- |
| xrops | every diagnostic written on xarray in NumPy and SciPy, its differences through geoops | the same DataArray functions and Operators, each a thin wrapper over a diagnostix or gridax function; differences through gridax, so xrops' calculus no longer needs geoops |
| sesmax-ocean (planned) | — | kinematics, spectra and skill as in-loop diagnostics and losses |
| somax | `eval/metrics.py` | retires with somax; sesmax-ocean imports diagnostix instead |

For an xrops user nothing changes; inside xrops the kernel moves:

```python
# Today: xrops computes N² on xarray with geoops' difference kernels
# today (xrops)
from xrops.ocn import brunt_vaisala_frequency

# After: the same function, a wrapper over the array function
# proposed (diagnostix#20)
from diagnostix.columns import buoyancy_frequency_squared
```

## Decisions and open questions

Six decided, one leaning. Each entry gives the position first, then the trade-off. The decisions of 24 Nov 2026 and the measurements behind them are in [diagnostix#15](https://github.com/jejjohnson/diagnostix/issues/15).

### 1. Layer: GeoNumerics tier 2

**Decided (5 Nov 2026, on the layer map).** diagnostix imports gridax (tier 1), geophysix (tier 0) and pipekit-array (Core). xrops and sesmax-ocean sit above it. **Dependencies (24 Nov 2026):** NumPy, gridax, geophysix and `pipekit-array[numpy,compat]`, all required. functax, integrix and gaussx need JAX for what diagnostix would use, so functax is a test reference and the others are not imported.

### 2. One library for physical and statistical diagnostics

**Decided (5 Nov 2026).** The skill scores planned for a separate skillax live here. A PSD score and a resolved scale need the same spectra as an energy-flux diagnostic, and a geostrophic-imbalance score needs the same kinematics; two libraries would import each other.

### 3. The DataArray API stays in xrops

**Decided (5 Nov 2026, on the layer map).** diagnostix takes arrays and a gridax grid; its elementwise functions also keep a DataArray's labels, and the others refuse one, so labels are never lost silently (24 Nov 2026). Names, attributes, units, CF conventions and dask chunking are xrops', which wraps each kernel.

### 4. One home per diagnostic, on every grid

**Decided (24 Nov 2026, [geostack#287](https://github.com/jejjohnson/geostack/issues/287)).** gridax owns a diagnostic it names, on every grid it supports, and diagnostix owns everything else. gridax's functions dispatch on the grid type, so it registers its collocated grids for the strains, Okubo–Weiss, the geostrophic velocity and the shallow-water potential vorticity ([gridax#565](https://github.com/jejjohnson/gridax/issues/565)), and Okubo–Weiss of a reanalysis and of a model's output is one call. diagnostix holding every collocated kernel would give Okubo–Weiss one name on a C-grid and another on an A-grid.

### 5. Array-API code, JAX optional

**Decided (24 Nov 2026).** Every function is written once in the Python array API standard, through `pipekit_array.functions`, so xrops can compute a diagnostic without JAX and a model can trace the same function. Nothing inside the library calls `jax.jit`; the caller jits once, at the top. Wavelets port too, because xrops' Morlet, Paul and derivative-of-Gaussian transforms are FFTs; only their significance test needs SciPy and stays on the host, and the PyWavelets transforms stay in xrops. JAX and Equinox alone would be simpler code, but xrops would need JAX.

### 6. EOFs without functax

**Decided (24 Nov 2026; it replaces the decision of 5 Nov, which took the basis from functax).** `diagnostix.eofs` takes the SVD of the area-weighted anomalies itself. functax's `eof_basis` requires JAX and takes no weights, so it is the oracle instead. diagnostix supplies the area weights from gridax, the truncation and the reading of the modes.

### 7. Scores here, benchmarks in pipekit-evaluate

**Leaning.** A score is a function of a prediction and a truth; a benchmark that fixes the data, the splits and the leaderboard is pipekit-evaluate's. xrops' `metrics/leaderboard.py` moves there, not here.

## Roadmap

Six waves, each a GitHub issue epic in the diagnostix repository.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| [0. Scaffold](https://github.com/jejjohnson/diagnostix/issues/8) | the repository from the library template, with its rules; the test harness (closed forms, stored xrops values, library oracles); grouped sums and masked reductions | — | — |
| [1. Kinematics and columns](https://github.com/jejjohnson/diagnostix/issues/9) | `diagnostix.kinematics`, `diagnostix.columns`; gridax's collocated registrations | the maths of xrops `ocn` and `atm/vertical` | 0 |
| [2. Budgets and climate](https://github.com/jejjohnson/diagnostix/issues/10) | `diagnostix.budgets`, `diagnostix.climate` | the maths of xrops `budgets` and `anomaly` | 0 |
| [3. Spectra and EOFs](https://github.com/jejjohnson/diagnostix/issues/11) | `diagnostix.spectra`, `diagnostix.eofs` | the maths of xrops `transforms/_src/fourier.py`, its `eof`, and `wavelets` | 0 |
| [4. Skill](https://github.com/jejjohnson/diagnostix/issues/12) | `diagnostix.skill` | the maths of xrops `metrics`; somax `eval/metrics.py` | 1, 3 |
| [5. Consumers switch](https://github.com/jejjohnson/diagnostix/issues/13) | xrops onto diagnostix; sesmax-ocean from the start | the copies | 1–4 |

Waves 1, 2 and 3 can run side by side.

## Sources

- [The GeoStack layer map](../stack_layer_map.md): diagnostix's layer, tier, imports and moves.
- @okubo1970 and @weiss1991: the Okubo–Weiss parameter.
- @vogelezang1996: the bulk Richardson method.
- [diagnostix#15](https://github.com/jejjohnson/diagnostix/issues/15): the decisions of 24 Nov 2026, with their measurements.
- Code read on 5 Nov 2026 at xrtoolz `b3855d7` and geostack's submodule pins: xrops `ocn/_src/kinematics.py`, `atm/_src/vertical.py`, `budgets/_src/`, `anomaly/_src/detrend.py`, `transforms/_src/` (`fourier.py`, `wavelet.py`, `decompose.py`), `wavelets/_src/`, `metrics/_src/`; somax `_src/eval/metrics.py`; gridax (then named differently) `_src/operators/diagnostics.py`.
