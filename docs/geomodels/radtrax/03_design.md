---
title: "radtrax: design and roadmap"
subtitle: radtrax · 3 · A differentiable radiative transfer package in JAX and Equinox, formerly rtmx — what to build, in what order, and what the stack already has
short_title: radtrax · 3 · Design
description: The design of radtrax (formerly rtmx), a differentiable radiative transfer package for methane and multi-gas remote sensing, checked against the stack it would be built from. A corrected v0 slice reproduces plumax's forward model and Jacobian to 10⁻¹⁵ and shows why layers matter; the source note's Equinox sketch, its solver choices and its stack integrations are corrected where the code disagrees.
keywords:
  - radtrax
  - rtmx
  - radiative transfer
  - Equinox
  - JAX
  - design
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

radtrax is the radiative transfer package the stack does not have yet: a
forward model $y = F(x)$ from an atmospheric state to what a sensor
measures, written in JAX and Equinox so that every Jacobian a retrieval
needs comes from automatic differentiation rather than hand-derived code.
It was called rtmx in the earlier design; the design itself is unchanged
in intent. This page sets out the architecture (a typed pipeline from
atmosphere to instrument, with the per-layer optical properties as the
canonical differentiable interface), a staged roadmap from Beer–Lambert to
multiple scattering, and the package's place in the stack, and checks each
against code that exists. Three findings shape it:

- **v0 is a port, not a start.** plumax already has cross-section tables,
  a Beer–Lambert forward model with Jacobians, band integration, matched
  filters and a 3D-Var retrieval. They are NumPy or frozen dataclasses on
  one slab. A layered Equinox v0 reproduces plumax's slab radiance and
  hand-written Jacobian to $10^{-15}$, and shows the slab's cost: one slab
  at the surface's pressure misstates a band's optical depth by up to 18 %.
- **The source note's Equinox sketch needs three fixes**: no JAX arrays in
  static fields (Equinox warns), no `AbstractVar` on a concrete module, and
  optimistix and optax in place of jaxopt and optax-contrib.
- **The stack has half of the v4 linear algebra.** gaussx's
  block-tridiagonal operator solves the boundary-value problem's structure
  to $10^{-16}$, but only for a symmetric band; discrete ordinates needs a
  general one.

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Proposal. radtrax does not exist; nothing here changes code. The [package boundaries](../package_boundaries.md) page decides what moves into it from plumax and geotoolz |
| **Formerly** | rtmx. The name changes to radtrax; the scope (differentiable SWIR radiative transfer for multi-gas retrievals) does not |
| **Reading time** | about 35 minutes |
| **Audience** | whoever builds radtrax, and anyone deciding whether a retrieval should wait for it |
| **Companions** | [radtrax · 1](01_column_retrieval.md) (the single-pass column retrieval v0 must reproduce); [radtrax · 2](02_mbmp.md) (the Sentinel-2 multi-pass family); [package boundaries](../package_boundaries.md) (the moves and the dependency rules); [plumax · 1](../plumax/01_emission_rate.md) (the emission rate downstream of the column) |
| **Code checked against** | plumax 0.1.2 at `28f257e` and gaussx 0.6.5 at `7fa1a38` (clones), geotoolz 0.10.2 at `088475e` and georeader-spaceml 2.4.1 from `references/`, finitevolX at `32ffb45`, spectraldiffx at `b5f3dc4`, pyrox at `94eef13`, somax at `aea3a03`; equinox 0.13.8, lineax 0.1.1, optimistix 0.1.0, optax 0.2.8. Read on 14 October 2026 |
| **Running example** | every number is from one run of `notebooks/geomodels/radtrax_03_design.py`, which asserts each one, on a synthetic table of 150 Voigt lines near 4300 cm⁻¹ (2.33 μm) |
| **Sources** | the design note supplied by the author [@geostack_radtrax_design_note]; Korkin et al. [@korkin2022]; ExoJAX [@kawahara2022]; VLIDORT [@spurr2006]; DISORT [@stamnes1988]; Rodgers [@rodgers2000] |
:::

:::{important} What changed from the source note
The note's **architecture stands**: a pipeline of typed modules,
atmosphere → spectroscopy → optical properties → solver → instrument, with
$\{\tau, \omega, \text{phase moments}, \text{surface}\}$ per layer as the
interface every solver consumes and automatic differentiation produces the
K matrix through. So does the **staged roadmap**, v0 Beer–Lambert to v4
multiple scattering, one fidelity axis per release. What **does not**:

- **v0 is not greenfield.** plumax's `hapi_lut`, `radtran`, `assimilation`
  and `matched_filter` already cover most of v0 and part of v1, on a single
  slab; the roadmap below ports them rather than rewriting them.
- **The Equinox sketch.** Marking arrays `eqx.field(static=True)` makes
  Equinox warn "A JAX array is being set as static!"; arrays stay leaves,
  and the derivative is steered with filters. `eqx.AbstractVar` on a
  concrete `ForwardModel` behaves as an ordinary required field; it belongs
  on an abstract base. The `bilinear` placeholder is filled in below.
- **The solvers it names.** JAXopt is no longer maintained (its README says
  so); fixed points, Gauss–Newton and Levenberg–Marquardt are in
  optimistix, which plumax already uses, and L-BFGS is in optax itself,
  not optax-contrib. Lineax's tridiagonal operator is scalar, not block;
  gaussx's block-tridiagonal operator is symmetric only.
- **The stack integrations.** finitevolx (C-grid finite volumes for ocean
  layers) and spectraldiffx (spectral derivatives) do not fit radtrax's
  layer grid or its band integration, which are a few lines of `jnp`;
  somax is a model zoo of simple dynamics, not atmospheric correction; the
  emulator library the note calls gpyroX is pyrox. The plume coupling
  $\partial y / \partial e$ lives in plumax, which imports radtrax, never
  the reverse.
- **Single precision.** The note puts float32's line-centre error at
  ~$10^{-7}$ cm⁻¹; that is float32's relative precision. In absolute terms
  float32 resolves 4300 cm⁻¹ to $4.9\times10^{-4}$ cm⁻¹, 7 % of methane's
  Doppler half-width there.
- **Sentinel-2.** "Implement Varon et al. exactly" carries the corrections
  of [radtrax · 2](02_mbmp.md): scale factors on background pixels, the
  exact two-band table, and band 11's absorption.

Two citations are marked rather than corrected: PyMieDiff is arXiv
2512.08614 (DOI 10.1063/5.0316814), whose journal volume we did not check
against the note's; and Ray-trax (arXiv 2511.09389) is a ray tracer for
turbulent astrophysical flows, a pattern reference only. A **Differs** box
marks each place.
:::

(rx3-purpose)=
## What radtrax is for

**TL;DR.** One forward model, every derivative of it, and four retrievals
that need them.

**Problem.** A retrieval inverts $y = F(x) + \varepsilon$ for the state $x$
(trace-gas profiles, temperature, surface, instrument), and every serious
inversion needs $K = \partial F / \partial x$: optimal estimation iterates
on it [@rodgers2000], a matched filter's target spectrum is one column of
it, and an emission-rate fit chains it with a plume model. The heritage
codes either ship $K$ by hand-derived linearisation (VLIDORT [@spurr2006])
or not at all (MODTRAN tables). radtrax's one bet is to make
$\{\tau_l, \omega_l, \chi_{l,k}, \text{surface}\}$, the per-layer optical
depth, single-scattering albedo, phase-function moments and surface
kernels, the interface between spectroscopy and solvers, and to get $K$
through it by `jax.jacrev`. Its uses, in the order the roadmap reaches
them:

1. **Matched-filter targets** for EMIT, PRISMA, EnMAP and AVIRIS-NG: the
   unit absorption spectrum $t = \partial \ln y / \partial \Delta X$ for a
   scene's geometry and water vapour, replacing a MODTRAN table.
2. **Optimal-estimation retrievals** of $X_{\text{CH}_4}$, $X_{\text{H}_2\text{O}}$,
   surface and spectral shift, with $K$ from autodiff.
3. **Multispectral retrievals** (Sentinel-2, Landsat): the band tables and
   the multi-pass family of [radtrax · 2](02_mbmp.md).
4. **Emission rates by maximum likelihood**: plumax maps an emission $e$ to
   a column, radtrax maps the column to radiance, and one gradient gives
   $\partial \text{loss} / \partial e$; the IME and the cross-sectional flux
   of [plumax · 1](../plumax/01_emission_rate.md) become its starting point.

(rx3-today)=
## What exists today

**TL;DR.** Most of v0 is in plumax already, in NumPy, on one slab.

| Piece | Where it is today | What it is | What radtrax changes |
|---|---|---|---|
| HITRAN cross-section tables $\sigma(\nu; p, T)$ | plumax `hapi_lut` (`fetch_hitran_data`, `build_lut_dataset`, `create_multi_gas_luts`) | HAPI, offline, to an `xarray.Dataset` on a (ν, T, p) grid | kept as the offline table builder behind `radtrax[hitran]` |
| Beer–Lambert forward model | plumax `radtran.forward` (`forward_nonlinear`, `forward_taylor`, `forward_maclaurin`, normalised variants) | NumPy, one slab at one (p, T), Jacobian derived by hand | Equinox, layered, Jacobian by autodiff |
| Band integration, band tables | plumax `radtran.srf` (`SpectralResponseFunction`), `radtran.nb_lut`; georeader `read_srf` | NumPy matrix; `nB` without the background column ([radtrax · 2](02_mbmp.md)) | one JAX `Instrument`; tables with the background |
| PSF, ground sampling | plumax `radtran.instrument` | NumPy with hand-written adjoints for tests | JAX, adjoint by autodiff, the hand adjoints kept as tests |
| Matched filters | plumax `matched_filter`, `radtran.matched_filter`, `radtran.gaussx_solve`; geotoolz `matched_filter` | three implementations | one JAX core ([package boundaries](../package_boundaries.md)) |
| Observation operator, 3D-Var | plumax `assimilation` (`RadianceObservationModel`, `run_gauss_newton`, `run_lbfgs`) | JAX, frozen dataclass, linear in $\Delta X$, one slab; optimistix and lineax | the v1 retrieval drivers, on the nonlinear layered model |
| Structured linear algebra | gaussx (`LowRankUpdate`, `BlockTriDiag`, `solve`) | lineax operators | used as is |

The pieces radtrax needs are present; what is missing is the layer
dimension, a single Equinox forward model they share, and everything past
Beer–Lambert.

(rx3-architecture)=
## The architecture

**TL;DR.** Five stages, typed pytrees between them, and protocols where
the alternatives plug in.

```text
AtmosphericState      p, T, z per layer; VMR per gas         (L,), dict
      │  OpacityProvider: σ(ν; p_l, T_l) per gas, Σ n x σ Δz
      ▼
OpticalProperties     τ (L, G), ω (L, G), χ (L, G, K)        ← K flows here
      │  + Surface: Lambertian | RossLi | RPV | CoxMunk
      │  RTSolver: BeerLambert | SingleScatter | TwoStream
      │            | DiscreteOrdinates(N) | DoublingAdding
      ▼
Radiance              L(ν) at the top of the atmosphere       (G,)
      │  Instrument: SRF, spectral shift, FWHM, PSF, GSD
      ▼
Observation           y_b per band; NoiseModel → Σ_y          (B,)
      │  Retrieval: OE (optimistix LM / GN) | MAP (optax)
      │             | Bayesian (numpyro) | matched filter
      ▼
State estimate        x̂, Ŝ, averaging kernel
```

Four rules come out of the note and the checks below:

1. **Modules carry state; functions carry transforms.** A table, a line
   list, an SRF or a noise covariance is an `eqx.Module`; a Voigt profile,
   a Chapman factor or a delta-M scaling is a function.
2. **Arrays are leaves.** A JAX array in a static field makes Equinox warn
   and breaks hashing under `filter_jit`. Grids, tables and SRFs stay
   leaves; what is not differentiated is chosen with `eqx.partition` or
   `jax.lax.stop_gradient`.
3. **Protocols at the seams.** `OpacityProvider`, `RTSolver`, `Surface`,
   `Instrument` and `NoiseModel` are abstract Equinox modules
   (`eqx.AbstractVar` and `eqx.abstractmethod` on the base, concrete
   fields on the subclass), so `filter_jit` compiles each combination.
4. **`vmap` outside `jacrev`.** A swath is a `filter_vmap` over pixels of a
   per-pixel Jacobian, never a Jacobian of a batch.

The proposed layout:

```text
radtrax/
├── atmosphere/   AtmosphericState, AFGL profiles, hydrostatics
├── spectroscopy/ tables (v0), line-by-line (v2): partition functions,
│                 line strengths, Voigt, continua, CIA
├── optics/       OpticalProperties, Rayleigh, HG and Mie, delta-M
├── surface/      Lambertian, kernel BRDFs, Cox–Munk
├── solvers/      BeerLambert, SingleScatter, TwoStream, ...
├── instrument/   SRF, shift and FWHM, PSF, GSD, noise
├── retrieval/    optimal estimation, matched filter, multi-pass
└── geo/          GeoTensor wrappers, sensor SRF tables  [geo] extra
```

(rx3-v0)=
## The v0 slice, run

**TL;DR.** A layered Equinox forward model reproduces plumax's slab
exactly, and shows what the slab gets wrong.

**Problem.** The smallest radtrax that is worth having is the note's v0:
an (ln p, T) table lookup per layer, Beer–Lambert along the two-way path,
and a band integration with a spectral shift. The notebook writes it with
every array a leaf:

```{code-block} python
:caption: The corrected v0 forward model and its Jacobian, from a real run.
:linenos:

# Shapes: L = 20 layers, G = 10,000 wavenumbers, B = 7 bands
# σ_l = bilinear(σ; ln p_l, T_l);  τ_l = σ_l n_l x_l Δz_l       (L, G)
tau = TableOpacity(log_p, t_grid, sigma)(atm)    # proposed (radtrax)
# L = (μ₀/π) A exp(−m Σ_l τ_l),  m = 1/μ₀ + 1/μ                  (G,)
rad = BeerLambert()(tau, albedo, mu0, mu)        # proposed (radtrax)
# y_b = Σ_g f_bg L(ν_g + δ)                                      (B,)
y = Instrument(srf, nu, shift)(rad)              # proposed (radtrax)
# one slab against plumax: L and ∂L/∂x                           (G,)
ref = forward_nonlinear(ds, nu, T_K=T0, p_atm=P0, vmr=X0, ...)  # today
# → radiance to 9 × 10⁻¹⁶, Jacobian to 4 × 10⁻¹⁶, relative      user
# K = ∂y/∂(ΔX₀, A, δ) per pixel, 256 pixels        (256, 3) → (256, 7, 3)
K = eqx.filter_vmap(jax.jacrev(y_of))(thetas)    # user
# → finite differences to 2.4 × 10⁻⁸; vmap equals a loop exactly
```

The layered model reduces to plumax's slab, radiance and Jacobian alike,
to rounding; the Jacobian that plumax derives by hand comes out of
`jax.jacrev` unchanged. The layers are not decoration. The same 1.9 ppm
column ($4.1\times10^{19}$ molecules/cm²) through 20 one-kilometre layers
and through one slab at the surface's pressure differ by up to 18 % in a
band's optical depth, because the slab broadens every line as if all the
methane sat at 1 atm; a slab at the column's pressure-weighted mean
differs by up to 4 %. plumax's forward model and its observation
operator are both single-slab, which is the first thing v0 changes.

:::{warning} Differs from the plan
The note's sketch marks `nu_hi`, `nu_sensor`, `srf_kernel`, `p_grid`,
`T_grid` and the solar spectrum `eqx.field(static=True)`; Equinox 0.13.8
warns "A JAX array is being set as static!" on each. Its `ForwardModel`
declares `opacity: eqx.AbstractVar[OpacityProvider]`: on a concrete class
that is an ordinary required field (the class builds with it and fails
without it like any missing argument), so the abstract base the note
means is a separate `AbstractOpacity` with `AbstractVar` fields of its
own. Its `LookupTableOpacity` loops over a fixed tuple of gases in
Python; a `dict` of tables keyed by gas, mapped with `jax.tree.map`,
keeps the gas list in the data. Its `K_xch4_shift_albedo` differentiates
three Python floats; that works for one pixel, and the swath version takes
the state as one array, as above.
:::

(rx3-roadmap)=
## The roadmap

**TL;DR.** Six releases, each a working retrieval, each adding one
fidelity axis, never a refactor and a feature together.

:::{table} The releases, what each ports and adds, and the check that closes it. Durations are the note's estimates.
:label: tbl-rx3-roadmap

| Release | Ports from the stack | Adds | Delivers | Closed when |
|---|---|---|---|---|
| **v0** Beer–Lambert, 4–6 weeks | plumax `hapi_lut` (tables), `radtran.forward`, `srf`, `nb_lut`, `target` | layers; one Equinox forward model; band tables with the background; tables for CH₄, H₂O, CO₂, N₂O, CO | matched-filter targets $t(\theta_s, \theta_v, h, \text{H}_2\text{O})$ with their gradients | slab equals plumax to $10^{-12}$ (done here); transmittance within 0.1 % of HAPI on AFGL US Standard; $K$ against finite differences |
| **v1** geometry, surface, instrument, noise, 4–6 weeks | plumax `assimilation`, `radtran.instrument`; geotoolz `plume.SBMP` and the MBSP of [radtrax · 1](01_column_retrieval.md) | Chapman air mass above 70°; kernel BRDFs; SRF shift and FWHM; noise covariance; the multi-pass family of [radtrax · 2](02_mbmp.md) | Rodgers OE for $X_{\text{CH}_4}$, $X_{\text{H}_2\text{O}}$, surface, shift | EMIT plume scenes against the operational product |
| **v2** line-by-line, 8–12 weeks | none | Voigt (direct), DIT/FFT and precomputed line-basis opacity; partition functions; continua (MT_CKD), CIA, line mixing | tables built in JAX, differentiable in the line parameters | 0.1 % against HAPI and LBLRTM on three AFGL atmospheres; single lines to $10^{-6}$ |
| **v3** single scattering, 8–10 weeks | none | Rayleigh with King factor; HG and double HG; Mie on a precomputed grid; analytic single scatter with a TMS correction | aerosol-aware SWIR forward model | Coulson tables, miepython, 6SV |
| **v4** multiple scattering, 12–16 weeks | gaussx operators | two-stream and Toon adding; doubling–adding; discrete ordinates with implicit differentiation | TROPOMI-class SWIR + O₂ A-band forward model | DISORT and VLIDORT to 0.5 % in radiance and 2 % in $K$ |
| **v5+** | pyrox | polarisation; Cox–Munk; 3-D; emulators behind the `RTSolver` protocol | operational throughput | I3RC; an emulator's error budget |
:::

Each release ships the regression set of the one before; the gradient
check, autodiff against central finite differences on a reduced state, is
the one test every release runs.

(rx3-diff)=
## Differentiability, component by component

**TL;DR.** Most of radtrax differentiates as written; four places do not,
and the stack has some of the fixes.

**Problem.** Reverse mode through `jnp` code is exact where the code is
smooth. The places it is not are known from the heritage codes, and the
notebook reproduces each one:

```{code-block} python
:caption: The four places autodiff needs help, from a real run.
:linenos:

# (a) float32 on a wavenumber grid, near 4300 cm⁻¹
np.spacing(np.float32(4300.0))   # → 4.9e-4 cm⁻¹ (Doppler HWHM 6.6e-3)
# (b) M(μ) = (e^{−τ/μ₀} − e^{−τ/μ}) / (1/μ − 1/μ₀) at μ = μ₀  () → ()
multiplier_naive(mu0)            # → nan, and its gradient nan   user
multiplier_safe(mu0)             # Taylor branch, double-where   user
# → τ e^{−τ/μ₀} exactly; slope = finite differences to 10⁻⁴
# (c) ∂/∂A of eigenvectors, repeated eigenvalue        (3, 3) → (3, 3)
jax.grad(first_eigvec_sum)(diag(1, 1, 2))      # → not finite    user
jax.grad(first_eigvec_sum)(diag(1, 1.001, 2))  # → 500, as 1/gap user
# (d) block-tridiagonal solve, 30 blocks of 8          (240,) → (240,)
gx.solve(gx.BlockTriDiag(D, A), b)   # → dense solve to 5e-16   today
```

1. **Precision.** Line-by-line opacity runs in float64. In float32 the
   wavenumber grid itself is coarse: $4.9\times10^{-4}$ cm⁻¹ at 4300 cm⁻¹,
   7 % of the Doppler half-width, so line centres move by a visible
   fraction of a line.
2. **Small denominators.** The discrete-ordinates streaming multipliers are
   0/0 at $\mu = \mu_0$, in value and in gradient. VLIDORT switches to a
   Taylor series within $\varepsilon \approx 10^{-3}$ [@spurr2006]; in JAX
   that branch needs the double-where (a safe denominator in the branch
   not taken) or `jnp.where` still propagates the NaN into the gradient.
3. **Repeated eigenvalues.** `jnp.linalg.eigh` gradients of eigenvectors
   scale as one over the eigenvalue gap and are not finite at a repeated
   one. Discrete ordinates' reduced eigenproblem [@stamnes1988] needs a
   `custom_jvp` that differentiates the invariant subspace, not the
   individual vectors.
4. **The boundary-value solve.** Discrete ordinates ends in one
   block-tridiagonal system per Fourier component; implicit
   differentiation through it costs one more solve. gaussx's
   `BlockTriDiag` solves that structure in $O(N d^3)$ and matches a dense
   solve to $5\times10^{-16}$, but its band is symmetric ($A_k$ below,
   $A_k^\top$ above), and with `symmetric=False` it falls back to a dense
   LU. The boundary-value matrix is not symmetric, so v4 needs a general
   block-tridiagonal solve, which belongs in gaussx.

Fixed points (successive orders, SHDOM-like iterations) and the
Gauss–Newton and Levenberg–Marquardt loops of optimal estimation use
optimistix (`FixedPointIteration`, `GaussNewton`, `LevenbergMarquardt`,
with `ImplicitAdjoint`), which plumax's `assimilation` already depends on;
L-BFGS for MAP retrievals is `optax.lbfgs`.

:::{warning} Differs from the plan
The note gives the float32 line-centre error as ~$10^{-7}$ cm⁻¹, float32's
relative precision rather than its absolute spacing at 4300 cm⁻¹, which is
$4.9\times10^{-4}$ cm⁻¹. It routes the discrete-ordinates solve through
"`lineax.linear_solve` with a tridiagonal operator"; lineax's
`TridiagonalLinearOperator` has scalar bands, and the system has blocks.
It names `jaxopt.FixedPointIteration` and `optimistix.FixedPointSolver`;
JAXopt is no longer maintained, and optimistix's class is
`FixedPointIteration` (`AbstractFixedPointSolver` is its base).
:::

(rx3-deps)=
## Dependencies and boundaries

**TL;DR.** radtrax depends on Numerics and nothing above it; geo carriers
arrive through an extra; plumax imports radtrax.

| Depends on | Why | How |
|---|---|---|
| jax, equinox, jaxtyping | the forward model and its pytrees | core |
| lineax, optimistix | linear solves, fixed points, OE drivers, implicit adjoints | core |
| gaussx | structured covariances (matched filter, OE), block-tridiagonal solves | core |
| hitran-api (HAPI), radis | offline line lists and tables | `radtrax[hitran]` |
| georeader, geotoolz | GeoTensor wrappers, sensor SRF tables | `radtrax[geo]` |
| numpyro, optax | Bayesian and MAP retrievals | `radtrax[bayes]` |
| pyrox | GP emulators behind the `RTSolver` protocol | `radtrax[emulate]` |

:::{warning} Differs from the plan
The note wires radtrax to finitevolx for "layer discretisation", to
spectraldiffx for "spectral convolution kernels", and to somax for
"atmospheric correction". finitevolx is Arakawa C-grid finite volumes, and
its vertical module is multilayer ocean dynamics; spectraldiffx is
Chebyshev, Fourier and spherical-harmonic derivatives; somax is a zoo of
simple ocean and atmosphere dynamics. A radiative transfer layer grid is
three arrays and a band integration is a matrix product, so none of the
three edges is needed, and each would pull a dynamics package under
radtrax. The emulator library the note calls gpyroX is pyrox (Gaussian
processes and Bayesian networks in Equinox and NumPyro). And the coupling
to plumax runs the other way: the [package boundaries](../package_boundaries.md)
rule P3 says radtrax never imports plumax, so the end-to-end
$\partial y / \partial e$ lives in plumax, as the composition of its
plume model with radtrax's forward model.
:::

(rx3-landscape)=
## What to inherit from the landscape

The note surveys the field; the table keeps what each code contributes to
radtrax's design. Entries marked † were checked for this page (the class
names against ExoJAX's documentation, the status against JAXopt's
README, the papers against their publication records); the rest are the note's.

| Code | Contributes | Where in radtrax |
|---|---|---|
| ExoJAX† [@kawahara2022] | the closest JAX design: `OpaDirect`, `OpaModit`, `OpaPremodit` opacities; `ArtEmisPure`, `ArtEmisScat`, `ArtTransPure`, `ArtReflectPure` solvers; layer-wise `scan` for memory | v2 spectroscopy; v4 two-stream; plain JAX, which radtrax re-frames in Equinox |
| VLIDORT [@spurr2006] | analytic K matrices in $\tau$, $\omega$, phase moments and surface; Taylor branches for small denominators | the K-matrix interface; v4 tests |
| DISORT [@stamnes1988] | the reduced eigenproblem; exponential scaling; delta-M and TMS | v4 discrete ordinates |
| ARTS | workspace and agendas: typed state, rewireable pipelines | the module DAG |
| libRadtran | one optical setup, many solvers | the `RTSolver` protocol |
| py4CAtS | typed stages: cross section → absorption → optical depth → radiance | the pytree chain |
| RTTOV | forward, tangent linear, adjoint and K as one model | `jvp`, `vjp`, `jacrev` of one function |
| Korkin et al.† [@korkin2022] (`gsit`) | make it right, then fast; Fourier azimuth modes as a `vmap` axis; analytic single scatter subtracted and added back | v3 and v4 structure |
| PyMieDiff† (arXiv 2512.08614) | autograd-safe Bessel recurrences, in PyTorch | v3 Mie |
| sRTMnet, ISOFIT | emulated radiative transfer in operations | v5 emulators |

(rx3-open-questions)=
## Open questions

1. **Where does the 3D-Var retrieval live?** plumax's `assimilation` is
   radtrax's v1 driver in all but name; the [package boundaries](../package_boundaries.md)
   page asks whether it belongs in radtrax or in vardax, and v1 needs the
   answer.
2. **Tables or line-by-line as the default?** v0's tables are fast and
   differentiable in the state; v2's line-by-line is differentiable in the
   line parameters too, which is what HITRAN-uncertainty propagation needs.
   Whether retrievals default to v2 once it exists is a cost question.
3. **How general must the block-tridiagonal solve be?** A non-symmetric
   `BlockTriDiag` in gaussx serves discrete ordinates and any other
   two-point boundary-value problem; it is the one Numerics addition v4
   needs.
4. **Which sensor first for v1?** EMIT has operational products to compare
   against; Sentinel-2 has the multi-pass family already written down.

(rx3-todos)=
## Upcoming work

:::{table} What radtrax needs, in order.
:label: tbl-rx3-todos

| Step | Where | Why |
|---|---|---|
| Create the radtrax repository from the package template, with the layout above | new repo | the package does not exist |
| Port `hapi_lut`, `radtran` and `assimilation`'s observation operator into one layered Equinox forward model | radtrax v0, from plumax | stage 2–3: plumax's slab, its hand Jacobian and an 18 % slab error |
| Build band tables against the background column | radtrax v0 | [radtrax · 2](02_mbmp.md): 1.1 % on band 12 |
| Add a non-symmetric `BlockTriDiag` | gaussx | stage 5: discrete ordinates' boundary-value matrix is not symmetric |
| Add the gradient check (autodiff against finite differences) to every public forward function's tests | radtrax | the one test every release runs |
| Update the [package boundaries](../package_boundaries.md) graph with radtrax's extras | geostack docs | `[hitran]`, `[bayes]` and `[emulate]` are new edges |
:::

(rx3-references)=
## References

- The design note, "A differentiable JAX/Equinox RTM for methane and
  multi-gas remote sensing — design and staged roadmap"
  [@geostack_radtrax_design_note].
- Korkin et al., the practical guide [@korkin2022]; ExoJAX
  [@kawahara2022]; VLIDORT [@spurr2006]; DISORT [@stamnes1988]; Rodgers,
  inverse methods [@rodgers2000].
- PyMieDiff, arXiv 2512.08614; Ray-trax, arXiv 2511.09389; JAXopt's README
  (no longer maintained).
- plumax, gaussx, geotoolz, georeader, finitevolX, spectraldiffx, pyrox and
  somax at the revisions in the header.
