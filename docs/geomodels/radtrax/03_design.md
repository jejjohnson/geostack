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

radtrax is the radiative transfer package that the stack does not have yet. It is a forward model $\vect{y} = h(\vect{x})$ from an atmospheric state $\vect{x} \in \mathbb{R}^{D_x}$ to what a sensor measures, $\vect{y} \in \mathbb{R}^{D_y}$, and we write it in JAX and Equinox so that every Jacobian a retrieval needs comes from automatic differentiation rather than from hand-derived code. It was called rtmx in the earlier design, and the design itself is unchanged in intent. In this page, we set out 1) the architecture (a typed pipeline from atmosphere to instrument, with the per-layer optical properties as the canonical differentiable interface), 2) a staged roadmap from Beer–Lambert to multiple scattering, and 3) the package's place in the stack, and we check each against code that exists. Three findings shape it:

- **v0 is a port, not a start.** plumax already has most of v0 on one slab. A layered Equinox v0 reproduces its radiance and its hand-written Jacobian to $10^{-15}$, and one slab at the surface's pressure misstates a band's optical depth by up to 19 %.
- **The source note's Equinox sketch needs three fixes**: no JAX arrays in static fields (Equinox warns), no `AbstractVar` on a concrete module, and optimistix and optax in place of jaxopt and optax-contrib.
- **The stack has half of the v4 linear algebra.** gaussx's block-tridiagonal operator solves the structure of the boundary-value problem to $10^{-16}$, but only for a symmetric band, whereas discrete ordinates needs a general one.

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
| **Code checked against** | plumax 0.1.2 at `28f257e`, gaussx 0.6.5 at `7fa1a38`, geotoolz 0.10.2 at `088475e`, georeader-spaceml 2.4.1, finitevolX at `32ffb45` and spectraldiffx at `b5f3dc4` from `references/`; pyrox at `94eef13`, somax at `aea3a03`; equinox 0.13.8, lineax 0.1.1, optimistix 0.1.0, optax 0.2.8. Read on 14 October 2026 |
| **Running example** | every number is from one run of `notebooks/geomodels/radtrax_03_design.py`, which asserts each one, on a synthetic table of 150 Voigt lines near 4300 cm⁻¹ (2.33 μm); code comments carry the [status markers](../../overview_examples.md#ex-markers) |
| **Sources** | the design note supplied by the author [@geostack_radtrax_design_note]; Korkin et al. [@korkin2022]; ExoJAX [@kawahara2022]; VLIDORT [@spurr2006]; DISORT [@stamnes1988]; Rodgers [@rodgers2000] |
:::

:::{important} What changed from the source note
The note's **architecture stands**: a pipeline of typed modules (from the atmosphere to the spectroscopy, the optical properties, the solver and the instrument), with $\{\tau, \omega, \text{phase moments}, \text{surface}\}$ per layer as the interface which every solver consumes and through which automatic differentiation produces the Jacobian $\mat{J}_h$ (Rodgers' K matrix). So does the **staged roadmap**, from v0 Beer–Lambert to v4 multiple scattering, with one fidelity axis per release. What **does not** stand is argued in a [Differs box](../../overview.md#sec-conventions) or in a section of its own:

- **v0 is not greenfield**: plumax covers most of it ([today](#rx3-today)).
- **The Equinox sketch**: static arrays, `AbstractVar` ([v0 slice](#rx3-v0)).
- **The solvers and the precision it names** ([differentiability](#rx3-diff)).
- **The stack integrations and the plumax coupling** ([dependencies](#rx3-deps)).
- **Sentinel-2** carries the corrections of [radtrax · 2](02_mbmp.md).

We mark two citations rather than correct them. PyMieDiff is arXiv 2512.08614 (DOI 10.1063/5.0316814), and we did not check its journal volume against the note's. Ray-trax (arXiv 2511.09389) is a ray tracer for turbulent astrophysical flows, so it is a pattern reference only.
:::

(rx3-purpose)=
## What radtrax is for

**TL;DR.** One forward model, every derivative of it, and four retrievals that need them.

**Problem.** A retrieval inverts $\vect{y} = h(\vect{x}) + \gvect{\varepsilon}$ for the state $\vect{x} \in \mathbb{R}^{D_x}$ (trace-gas profiles, temperature, surface, instrument), where $\vect{y} \in \mathbb{R}^{D_y}$ is the measurement and $\gvect{\varepsilon} \in \mathbb{R}^{D_y}$ is its noise. Every serious inversion needs the Jacobian $\mat{J}_h = \partial h / \partial \vect{x} \in \mathbb{R}^{D_y \times D_x}$: optimal estimation iterates on it [@rodgers2000], a matched filter's target spectrum is one column of it, and an emission-rate fit chains it with a plume model. The heritage codes either ship $\mat{J}_h$ by hand-derived linearisation (VLIDORT [@spurr2006]) or not at all (MODTRAN tables). radtrax's one bet is to make $\{\tau_l, \omega_l, \chi_{l,k}, \text{surface}\}$ the interface between spectroscopy and solvers (the optical depth, single-scattering albedo and phase-function moments $k$ of layer $l$, all scalars, and the surface kernels), and to get $\mat{J}_h$ through it with `jax.jacrev`. Its uses, in the order the roadmap reaches them, are:

1. **Matched-filter targets** for EMIT, PRISMA, EnMAP and AVIRIS-NG: the unit absorption spectrum $\partial \ln \vect{y} / \partial \Delta\mathrm{VMR} \in \mathbb{R}^{D_y}$ for a scene's geometry and water vapour, in place of a MODTRAN table. It is the target $\vect{J}_h$ (the one column of $\mat{J}_h$ for the methane enhancement) of the matched filter in [radtrax · 1](01_column_retrieval.md#rx1-stage-5) and [Report 16](../../reports/16_geo_data_stack.md#r16-stage-2). [CHECK: radtrax · 1 defines $\vect{J}_h$ as the derivative of the reflectance with respect to $\Delta\Omega$, which is this log derivative times $\gvect{\mu}_y$ and a conversion from $\Delta\mathrm{VMR}$ to $\Delta\Omega$, not the unit absorption spectrum itself.]
2. **Optimal-estimation retrievals** of $\mathrm{XCH_4}$, $\mathrm{XH_2O}$, surface and spectral shift, with $\mat{J}_h$ from autodiff.
3. **Multispectral retrievals** (Sentinel-2, Landsat): the band tables and the multi-pass family of [radtrax · 2](02_mbmp.md).
4. **Emission rates by maximum likelihood**: plumax maps an emission rate $Q$ to a column, radtrax maps the column to radiance, and one gradient gives $\partial \mathcal{L} / \partial Q$ of the loss $\mathcal{L}$. The IME and the cross-sectional flux of [plumax · 1](../plumax/01_emission_rate.md) become its starting point.

(rx3-today)=
## What exists today

**TL;DR.** Most of v0 is in plumax already, in NumPy, on one slab.

The [package boundaries inventory](../package_boundaries.md#pb-today) lists every module and its size, and the table below keeps what radtrax changes in each.

| Piece | Today | What radtrax changes |
|---|---|---|
| HITRAN cross-section tables $\sigma(\nu; p, T)$ | plumax `hapi_lut`: HAPI, offline, on a (ν, T, p) grid | kept as the offline table builder behind `radtrax[hitran]` |
| Beer–Lambert forward model | plumax `radtran.forward`: NumPy, one slab at one (p, T), Jacobian by hand | Equinox, layered, Jacobian by autodiff |
| Band integration, band tables | plumax `radtran.srf`, `radtran.nb_lut`; georeader `read_srf`: `nB` without the background column ([radtrax · 2](02_mbmp.md#rx2-band)) | one JAX `Instrument`; tables with the background |
| PSF, ground sampling | plumax `radtran.instrument`: NumPy, hand-written adjoints | JAX, adjoint by autodiff, the hand adjoints kept as tests |
| Matched filters | [three implementations](../package_boundaries.md#pb-stage-3) | one JAX core |
| Observation operator, 3D-Var | plumax `assimilation`: JAX, frozen dataclass, linear in $\Delta\mathrm{VMR}$, one slab | the v1 retrieval drivers, on the nonlinear layered model |
| Structured linear algebra | gaussx (`LowRankUpdate`, `BlockTriDiag`, `solve`) | used as is |

We have the pieces radtrax needs. What is missing is the layer dimension, a single Equinox forward model which they share, and everything past Beer–Lambert.

(rx3-architecture)=
## The architecture

**TL;DR.** Five stages, typed pytrees between them, and protocols where the alternatives plug in.

```text
AtmosphericState      p, T, z per layer; VMR per gas         (N_L,), dict
      │  OpacityProvider: σ(ν; p_l, T_l) per gas, Σ n VMR σ Δz
      ▼
OpticalProperties     τ, ω (N_L, N_ν), χ (N_L, N_ν, N_χ)     ← J_h flows here
      │  + Surface: Lambertian | RossLi | RPV | CoxMunk
      │  RTSolver: BeerLambert | SingleScatter | TwoStream
      │            | DiscreteOrdinates(N) | DoublingAdding
      ▼
Radiance              L(ν) at the top of the atmosphere       (N_ν,)
      │  Instrument: SRF, spectral shift, FWHM, PSF, GSD
      ▼
Observation           y_b per band; NoiseModel → Σ_ε          (N_B,)
      │  Retrieval: OE (optimistix LM / GN) | MAP (optax)
      │             | Bayesian (numpyro) | matched filter
      ▼
State estimate        x̂, Σ_x|y, averaging kernel
```

We take four rules from the note and from the checks below:

1. **Modules carry state; functions carry transforms.** A table, a line list, an SRF or a noise covariance is an `eqx.Module`, whereas a Voigt profile, a Chapman factor or a delta-M scaling is a function.
2. **Arrays are leaves.** A JAX array in a static field makes Equinox warn and breaks hashing under `filter_jit`. So grids, tables and SRFs stay leaves, and we choose what is not differentiated with `eqx.partition` or `jax.lax.stop_gradient`.
3. **Protocols at the seams.** `OpacityProvider`, `RTSolver`, `Surface`, `Instrument` and `NoiseModel` are abstract Equinox modules (`eqx.AbstractVar` and `eqx.abstractmethod` on the base, concrete fields on the subclass), so `filter_jit` compiles each combination.
4. **`vmap` outside `jacrev`.** A swath is a `filter_vmap` over pixels of a per-pixel Jacobian, and never a Jacobian of a batch.

The layout we propose:

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

**TL;DR.** A layered Equinox forward model reproduces plumax's slab exactly, and shows what the slab gets wrong.

**Problem.** The smallest radtrax that is worth having is the note's v0: an (ln p, T) table lookup per layer, Beer–Lambert along the two-way path, and a band integration with a spectral shift. We write it in the notebook with every array as a leaf:

```{code-block} python
:caption: The corrected v0 forward model and its Jacobian, from a real run.
:linenos:

# Shapes: N_L = 20 layers, N_ν = 14,000 wavenumbers, N_B = 7 bands (8.5 nm)
# σ_l = bilinear(σ; ln p_l, T_l);  τ_l = σ_l n_l VMR_l Δz_l     (N_L, N_ν)
tau = TableOpacity(log_p, t_grid, sigma)(atm)    # proposed (radtrax)
# L = (μ₀/π) A exp(−AMF Σ_l τ_l),  AMF = 1/μ₀ + 1/μ             (N_ν,)
rad = BeerLambert()(tau, albedo, mu0, mu)        # proposed (radtrax)
# y_b = Σ_g f_bg L(ν_g + δ)                                     (N_B,)
y = Instrument(srf, nu, shift)(rad)              # proposed (radtrax)
# one slab against plumax: L and ∂L/∂x                          (N_ν,)
ref = forward_nonlinear(ds, nu, T_K=T0, p_atm=P0, vmr=X0, ...)  # today
# → radiance to 1.3 × 10⁻¹⁵, Jacobian to 6 × 10⁻¹⁶, relative  user
# J_h = ∂y/∂(ΔVMR₀, A, δ) per pixel, 256 pixels    (256, 3) → (256, 7, 3)
j_h = eqx.filter_vmap(jax.jacrev(y_of))(thetas)  # user
# → finite differences to 2.3 × 10⁻⁸; vmap equals a loop exactly
```

The layered model reduces to plumax's slab to rounding, for the radiance and the Jacobian alike, and the Jacobian that plumax derives by hand comes out of `jax.jacrev` unchanged. However, the layers are not decoration. The same 1.9 ppm column ($4.1 \times 10^{19}$ molecules/cm²) through 20 one-kilometre layers and through one slab at the surface's pressure differ by up to 19 % in a band's optical depth, because the slab broadens every line as if all the methane sat at 1 atm, whereas a slab at the column's pressure-weighted mean differs by up to 3 %. plumax's forward model and its observation operator are both single-slab, which is the first thing v0 changes.

:::{warning} Differs from the plan
The note's sketch marks `nu_hi`, `nu_sensor`, `srf_kernel`, `p_grid`, `T_grid` and the solar spectrum as `eqx.field(static=True)`, and Equinox 0.13.8 warns "A JAX array is being set as static!" on each. Its `ForwardModel` declares `opacity: eqx.AbstractVar[OpacityProvider]`. On a concrete class that is an ordinary required field (the class builds with it and fails without it like any missing argument), so the abstract base the note means is a separate `AbstractOpacity` with `AbstractVar` fields of its own. Its `LookupTableOpacity` loops over a fixed tuple of gases in Python, whereas a `dict` of tables keyed by gas, mapped with `jax.tree.map`, keeps the gas list in the data. Its `K_xch4_shift_albedo` differentiates three Python floats. That works for one pixel, and the swath version takes the state as one array, as above.
:::

(rx3-roadmap)=
## The roadmap

**TL;DR.** Six releases, each a working retrieval, each adding one fidelity axis, never a refactor and a feature together.

:::{table} The releases, what each ports and adds, and the check that closes it. Durations are the note's estimates.
:label: tbl-rx3-roadmap

| Release | Ports from the stack | Adds | Delivers | Closed when |
|---|---|---|---|---|
| **v0** Beer–Lambert, 4–6 weeks | plumax `hapi_lut` (tables), `radtran.forward`, `srf`, `nb_lut`, `target` | layers; one Equinox forward model; band tables with the background; tables for CH₄, H₂O, CO₂, N₂O, CO | matched-filter targets $\vect{J}_h(\mathrm{SZA}, \mathrm{VZA}, z, \mathrm{H_2O})$ with their gradients | slab equals plumax to $10^{-12}$ (done here); transmittance within 0.1 % of HAPI on AFGL US Standard; $\mat{J}_h$ against finite differences |
| **v1** geometry, surface, instrument, noise, 4–6 weeks | plumax `assimilation`, `radtran.instrument`; geotoolz `plume.SBMP` (an [MBMP](02_mbmp.md#rx2-stack)) and the MBSP of [radtrax · 1](01_column_retrieval.md) | Chapman air mass above 70°; kernel BRDFs; SRF shift and FWHM; noise covariance; the multi-pass family of [radtrax · 2](02_mbmp.md) | Rodgers OE for $\mathrm{XCH_4}$, $\mathrm{XH_2O}$, surface, shift | EMIT plume scenes against the operational product |
| **v2** line-by-line, 8–12 weeks | none | Voigt (direct), DIT/FFT and precomputed line-basis opacity; partition functions; continua (MT_CKD), CIA, line mixing | tables built in JAX, differentiable in the line parameters | 0.1 % against HAPI and LBLRTM on three AFGL atmospheres; single lines to $10^{-6}$ |
| **v3** single scattering, 8–10 weeks | none | Rayleigh with King factor; HG and double HG; Mie on a precomputed grid; analytic single scatter with a TMS correction | aerosol-aware SWIR forward model | Coulson tables, miepython, 6SV |
| **v4** multiple scattering, 12–16 weeks | gaussx operators | two-stream and Toon adding; doubling–adding; discrete ordinates with implicit differentiation | TROPOMI-class SWIR + O₂ A-band forward model | DISORT and VLIDORT to 0.5 % in radiance and 2 % in $\mat{J}_h$ |
| **v5+** | pyrox | polarisation; Cox–Munk; 3-D; emulators behind the `RTSolver` protocol | operational throughput | I3RC; an emulator's error budget |
:::

Each release in [](#tbl-rx3-roadmap) ships the regression set of the one before, and the gradient check (autodiff against central finite differences on a reduced state) is the one test every release runs.

(rx3-diff)=
## Differentiability, component by component

**TL;DR.** Most of radtrax differentiates as written; four places do not, and the stack has some of the fixes.

**Problem.** Reverse mode through `jnp` code is exact where the code is smooth. The places where it is not are known from the heritage codes, and the notebook reproduces each one:

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

1. **Precision.** Line-by-line opacity runs in float64. In float32 the wavenumber grid itself is coarse, $4.9 \times 10^{-4}$ cm⁻¹ at 4300 cm⁻¹ (7 % of the Doppler half-width), so line centres move by a visible fraction of a line.
2. **Small denominators.** The discrete-ordinates streaming multipliers are 0/0 at $\mu = \mu_0$, in value and in gradient. VLIDORT switches to a Taylor series once $\vert \mu - \mu_0 \vert$ is within about $10^{-3}$ [@spurr2006]. In JAX that branch needs the double-where (a safe denominator in the branch not taken), or else `jnp.where` still propagates the NaN into the gradient.
3. **Repeated eigenvalues.** The gradients of eigenvectors from `jnp.linalg.eigh` scale as one over the eigenvalue gap, and they are not finite at a repeated eigenvalue. So the reduced eigenproblem of discrete ordinates [@stamnes1988] needs a `custom_jvp` that differentiates the invariant subspace and not the individual vectors.
4. **The boundary-value solve.** Discrete ordinates ends in one block-tridiagonal system per Fourier component, and implicit differentiation through it costs one more solve. gaussx's `BlockTriDiag` solves that structure in $O(N d^3)$ (for $N$ blocks of size $d$) and matches a dense solve to $5 \times 10^{-16}$. However, its band is symmetric ($\mat{A}_k$ below the diagonal, $\mat{A}_k^\top$ above), and with `symmetric=False` it falls back to a dense LU. The boundary-value matrix is not symmetric, so v4 needs a general block-tridiagonal solve, which belongs in gaussx.

For the fixed points (successive orders, SHDOM-like iterations) and the Gauss–Newton and Levenberg–Marquardt loops of optimal estimation we use optimistix (`FixedPointIteration`, `GaussNewton`, `LevenbergMarquardt`, with `ImplicitAdjoint`), which plumax's `assimilation` already depends on, and L-BFGS for MAP retrievals is `optax.lbfgs`.

:::{warning} Differs from the plan
The note gives the float32 line-centre error as ~$10^{-7}$ cm⁻¹, which is float32's relative precision rather than its absolute spacing at 4300 cm⁻¹ ($4.9 \times 10^{-4}$ cm⁻¹). It routes the discrete-ordinates solve through "`lineax.linear_solve` with a tridiagonal operator". However, lineax's `TridiagonalLinearOperator` has scalar bands, and the system has blocks. It also names `jaxopt.FixedPointIteration` and `optimistix.FixedPointSolver`, but JAXopt is no longer maintained, and optimistix's class is `FixedPointIteration` (`AbstractFixedPointSolver` is its base).
:::

(rx3-deps)=
## Dependencies and boundaries

**TL;DR.** radtrax depends on GeoNumerics and nothing above it; geo carriers arrive through an extra; plumax imports radtrax.

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
The note wires radtrax to finitevolx for "layer discretisation", to spectraldiffx for "spectral convolution kernels", and to somax for "atmospheric correction". However, finitevolx is Arakawa C-grid finite volumes (its vertical module is multilayer ocean dynamics), spectraldiffx is Chebyshev, Fourier and spherical-harmonic derivatives, and somax is a zoo of simple ocean and atmosphere dynamics. A radiative transfer layer grid is three arrays and a band integration is a matrix product, so we need none of the three edges, and each one would pull a dynamics package under radtrax. The emulator library the note calls gpyroX is pyrox (Gaussian processes and Bayesian networks in Equinox and NumPyro). Lastly, the coupling to plumax runs the other way: [rule 2](../package_boundaries.md#pb-rules) of the package boundaries says that radtrax never imports plumax, so the end-to-end $\partial \vect{y} / \partial Q$ lives in plumax, as the composition of its plume model with radtrax's forward model.
:::

(rx3-landscape)=
## What to inherit from the landscape

The note surveys the field, and the table keeps what each code contributes to radtrax's design. We checked the entries marked † for this page (the class names against ExoJAX's documentation, the status against JAXopt's README, the papers against their publication records), and the rest are the note's.

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

1. **Where does the 3D-Var retrieval live?** plumax's `assimilation` is radtrax's v1 driver in all but name. The [package boundaries](../package_boundaries.md) page asks whether it belongs in radtrax or in vardax, and v1 needs the answer.
2. **Tables or line-by-line as the default?** v0's tables are fast and differentiable in the state, whereas v2's line-by-line is also differentiable in the line parameters, which is what HITRAN-uncertainty propagation needs. Whether retrievals default to v2 once it exists is a cost question.
3. **How general must the block-tridiagonal solve be?** A non-symmetric `BlockTriDiag` in gaussx serves discrete ordinates and any other two-point boundary-value problem, and it is the one GeoNumerics addition that v4 needs.
4. **Which sensor first for v1?** EMIT has operational products to compare against, whereas Sentinel-2 has the multi-pass family already written down.

(rx3-todos)=
## Upcoming work

[](#tbl-rx3-todos) lists the steps, in order.

:::{table} What radtrax needs, in order.
:label: tbl-rx3-todos

| Step | Where | Why |
|---|---|---|
| Create the radtrax repository from the package template, with the layout above | new repo | the package does not exist |
| Port `hapi_lut`, `radtran` and `assimilation`'s observation operator into one layered Equinox forward model | radtrax v0, from plumax | [v0 slice](#rx3-v0): plumax's slab, its hand Jacobian and a 19 % slab error |
| Build band tables against the background column | radtrax v0 | [radtrax · 2](02_mbmp.md#rx2-band): 1.1 % on band 12 |
| Add a non-symmetric `BlockTriDiag` | gaussx | [differentiability](#rx3-diff), item 4: discrete ordinates' boundary-value matrix is not symmetric |
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
