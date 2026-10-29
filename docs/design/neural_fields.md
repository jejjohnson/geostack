---
title: "Neural fields across the stack: what exists, what is missing, and the plan"
subtitle: "Design · geonnax, pyrox-nn and vardax as one neural-field toolkit"
short_title: Neural fields
description: "Maps the 2025–2026 neural-field literature (report 18) onto geonnax, pyrox-nn and vardax, from a survey of their code on 23 October 2026. The stack already holds most of a geoscience neural-field toolkit; the missing parts are plumbing between libraries: a point-observation path, coordinate-derivative operators, observation operators inside likelihoods and a network as the 4D-Var control. Work is tracked in Wave 6."
date: 2026-10-23
keywords:
  - neural fields
  - implicit neural representations
  - geonnax
  - pyrox
  - vardax
  - data assimilation
  - observation operators
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Motivation

[Report 18](../reports/18_neural_fields.md) reviews neural fields in the geosciences for 2025 and 2026. Its conclusion for GeoStack is that neural fields are worth building for 1) **gridless reconstruction** (fitting tracks, swaths and pixels directly), 2) **uncertainty-aware gap-filling**, and 3) **differentiable state parameterizations in variational assimilation**. However, they are not worth building as an archival **compression** format, where SZ3 still wins.

We surveyed geonnax (v0.0.9), pyrox (pyrox-nn, pyrox-gp, pyrox-lgm) and vardax (v0.1.9) on 23 October 2026 and found that the stack already has most of the parts, and what it lacks is the plumbing between them. So this page records what exists, what is missing, where each missing piece belongs under the dependency rules, and the order in which we build it. The work is [Wave 6](https://github.com/jejjohnson/geostack/issues/82).

## Decision

1. **Deterministic neural-field parts go in geonnax**: encoders, architectures and coordinate-derivative operators. geonnax keeps its rules (no numpyro, no kernels, no losses, one example per call).
2. **Probabilistic parts go in pyrox-nn**, as wrappers which swap named geonnax fields for NumPyro sites, plus an estimator whose likelihood takes per-observation noise and an observation operator.
3. **Assimilation parts go in vardax**: a point-observation batch, a nonlinear `StateParameterization`, an optax minimiser, a PDE residual cost and a space-time neural-field 4D-Var model.
4. **Acceptance tests are existing case studies**: case 9 (SSH mapping, with DUACS/OI as the oracle), case 11 (SST gap-filling, with the GP as the oracle) and overview stage 3 (methane L2 → L3, with kriging as the oracle).
5. **Compression is a research bet**, which we score against SZ3/ZFP at matched maximum error before anything is built for it.

## Where a neural field flows

```text
  observations  y ∈ ℝ^{N_obs}  (tracks, swaths, pixels)
  geostack: GeoPoints / GeoTracks, to_jax(pad_to=N) + valid mask   (CS-F3)
        │
        ▼
  geonnax · encode         γ(s, t), s = (λ, φ)
    SphericalHarmonicEncoder, SlepianEncoder, Cartesian3DEncoder,
    CyclicEncoder, GeoContextEncoder
    ✗ Gaussian Fourier features, HEALPix grid / feature grid
        │
        ▼
  geonnax · represent      x(s, t; θ) = g_net(γ(s, t); θ)
    SIREN, MultiScaleSIREN, FourierNet, GaborNet,
    ConditionedINR + FiLM / HyperLinear, HyperSIREN
    ✗ separable, latent table + shift modulation, LIIF, BNF core
        │
        ├── ✗ geonnax · differentiate   ∇x, ∇²x; on S²: ∇_S² x, ∇²_S² x
        │
        ▼
  pyrox-nn · make Bayesian  θ ~ p_prior(θ)
    BayesianSIREN, BayesianFourierNet, BayesianNeuralField,
    EnsembleMAP, EnsembleVI, BNFEstimator
        │
        ▼
  vardax · observe          ŷ = ℋ[x(·; θ)]
    AveragingKernel, MultiInstrumentFusion
    ✗ point evaluation at observation coordinates
        │
        ▼
  likelihood / loss         p_lik(y | ŷ; Σ_ε)  or  ℒ(θ)
    ✗ pyrox-nn: per-observation σ, ℋ inside the likelihood
    ✗ vardax: θ as the 4D-Var control, PDE residual term
        │
        ▼
  geostack · score          case 9, case 11, methane stage 3
```

## What exists, and the gap

| Research theme | Already in the stack | Gap | Home |
| --- | --- | --- | --- |
| Sphere encodings | geonnax `SphericalHarmonicEncoder` (tested to L = 150), `SlepianEncoder`, `HybridSphericalSlepianEncoder`, `needlet_basis`, `SphericalWaveletTransform`, `fibonacci_sphere`, `icosphere` | HEALPix grid; sphere-native multiresolution feature grid; stratified (coast / island) evaluation | geonnax |
| Spectral bias and multiscale | `MultiScaleSIREN`, `log_spaced_omegas`, `GaborNet`, `basis.fourier_features` (dyadic) | Gaussian Fourier-feature encoder | geonnax |
| Compression | `SIREN`, `ConditionedINR`; `CPTensor`, `TuckerTensor`, `TTTensor` | A fair benchmark first | geostack |
| Gridless reconstruction | encoders + INRs; `GeoFNO(x_out=…)`; `GeoPoints` with padding | A point-observation operator; vardax batches assume observation shape = state shape | vardax |
| Bayesian / UQ fields | pyrox-nn `BayesianNeuralField` + `BNFEstimator` (BayesNF port), `BayesianSIREN`, `BayesianFourierNet`, `EnsembleMAP`, `EnsembleVI`; SNGP | BNF noise fixed at σ = 0.1; Gaussian only; BNF-only, pandas-only estimator; no sphere encoder in BNF | pyrox-nn |
| Amortized fields | FiLM `AffineModulation`, `HyperLinear`, `HyperSIREN`, `BayesianAffineModulation` | Latent table, shift modulation, MFN conditioning | geonnax |
| Physics constraints | vardax `DynIncrements`, `DynTrajectory` (ODE residuals); geonnax `divfree_basis`, `vector_spherical_harmonics` | Coordinate-derivative operators; PDE residual at collocation points | geonnax, vardax |
| Neural-field 4D-Var | vardax strong / weak / incremental 4D-Var; linear `ReducedBasis` (only in `Incremental4DVar`, only at t = 0) | Nonlinear parameterization; first-order minimiser; space-time field model | vardax |
| Trace gases | vardax `AveragingKernel`; geonnax README XCH₄ demo | Averaging kernel inside a pyrox likelihood | pyrox-nn |
| Foundation-model decoders | `GeoFNO` point queries, `SpectralConv.to_points` | LIIF-style local decoder | geonnax |

## The plan by library

### geonnax ([theme 1](https://github.com/jejjohnson/geonnax/issues/130), [theme 2](https://github.com/jejjohnson/geonnax/issues/131))

| Issue | What | Stage |
| --- | --- | --- |
| [geonnax#132](https://github.com/jejjohnson/geonnax/issues/132) | `geonnax.diffops`: gradient, divergence, Laplacian, surface gradient, Laplace–Beltrami | A |
| [geonnax#133](https://github.com/jejjohnson/geonnax/issues/133) | `GaussianFourierEncoder` | B |
| [geonnax#134](https://github.com/jejjohnson/geonnax/issues/134) | `NeuralFieldMLP`: the BayesNF architecture as a deterministic core | B |
| [geonnax#135](https://github.com/jejjohnson/geonnax/issues/135) | Notebook: neural fields on the Western Mediterranean SSH tracks | B |
| [geonnax#136](https://github.com/jejjohnson/geonnax/issues/136) | `SeparableINR` (SPINN) | C |
| [geonnax#137](https://github.com/jejjohnson/geonnax/issues/137) | `FourierNet` / `GaborNet` inside `ConditionedINR` | D |
| [geonnax#138](https://github.com/jejjohnson/geonnax/issues/138) | `LatentTable` and `ShiftModulation` | D |
| [geonnax#139](https://github.com/jejjohnson/geonnax/issues/139) | `LocalImplicitDecoder` (LIIF) | D |
| [geonnax#140](https://github.com/jejjohnson/geonnax/issues/140) | `healpix_grid` | D |
| [geonnax#141](https://github.com/jejjohnson/geonnax/issues/141) | Research: `HEALPixFeatureGrid` | D |

The derivative operators are operators and not losses, so they keep geonnax's "no losses" rule, and the residual costs live in vardax. The BayesNF core moves down from pyrox-nn for two reasons: 1) so that pyrox-nn's BNF becomes a wrapped core like every other Bayesian network there, and 2) so that it can take spherical-harmonic features.

### pyrox-nn ([theme](https://github.com/jejjohnson/pyrox/issues/315))

| Issue | What | Stage |
| --- | --- | --- |
| [pyrox#316](https://github.com/jejjohnson/pyrox/issues/316) | Learned and per-observation noise in the BNF likelihood | A |
| [pyrox#317](https://github.com/jejjohnson/pyrox/issues/317) | `NeuralFieldEstimator` for any `PyroxModule`, on arrays or xarray | B |
| [pyrox#318](https://github.com/jejjohnson/pyrox/issues/318) | Observation operator inside the likelihood (averaging kernels, footprints) | B |
| [pyrox#319](https://github.com/jejjohnson/pyrox/issues/319) | `BayesianNeuralField` wraps `geonnax.NeuralFieldMLP` | B |
| [pyrox#320](https://github.com/jejjohnson/pyrox/issues/320) | Regression heteroscedastic head | B |
| [pyrox#321](https://github.com/jejjohnson/pyrox/issues/321) | `SphericalHarmonicFeatures`: a weight-space GP on S² (the OI-equivalent baseline) | B |
| [pyrox#322](https://github.com/jejjohnson/pyrox/issues/322) | Last-layer GGN Laplace for any `PyroxModule` | C |
| [pyrox#323](https://github.com/jejjohnson/pyrox/issues/323) | [Design] neural-field mean plus GP residual | C |

pyrox (GeoML tier 2) may not import vardax (tier 3), so the likelihood takes the observation operator as a plain callable, which vardax's `AveragingKernel` satisfies.

### vardax ([theme](https://github.com/jejjohnson/vardax/issues/82))

| Issue | What | Stage |
| --- | --- | --- |
| [vardax#83](https://github.com/jejjohnson/vardax/issues/83) | `PointObsBatch` and `FieldPointObs`; observation shape no longer tied to state shape | A |
| [vardax#84](https://github.com/jejjohnson/vardax/issues/84) | `StateParameterization`, generalizing `ReducedBasis` | C |
| [vardax#85](https://github.com/jejjohnson/vardax/issues/85) | `OptaxMinimiser` | C |
| [vardax#86](https://github.com/jejjohnson/vardax/issues/86) | `ResidualPenalty`: PDE residual at collocation points | C |
| [vardax#87](https://github.com/jejjohnson/vardax/issues/87) | `SpaceTimeFieldVar`: neural-field 4D-Var (Oh 2025) | C |
| [vardax#88](https://github.com/jejjohnson/vardax/issues/88) | Notebook: neural-field 4D-Var on 2-D Kolmogorov flow | C |

The point-observation batch also fixes the classical models on along-track data, so it pays off without any neural field. somax stays a forbidden import, so the PDE residuals enter as callables, as the forward models do today.

The space-time neural-field 4D-Var loss is written over the network weights $\gvect{\theta} \in \mathbb{R}^{D_\theta}$, with the field $x(\vect{s}, t; \gvect{\theta}) \in \mathbb{R}$ at the coordinates $\vect{s} = (\lambda, \phi)$ and time $t$. Each observation $y_n \in \mathbb{R}$, $n = 1, \ldots, N_\text{obs}$, sits at $(\vect{s}_n, t_n)$ with error standard deviation $\sigma_n \in \mathbb{R}_{>0}$. The PDE residual operator $\mathcal{R}[x]$ vanishes on the true flow and is evaluated at the collocation points $(\vect{s}_j, t_j)$, $j = 1, \ldots, N_c$, and $\beta_\text{PDE}, \beta_\theta \in \mathbb{R}_{\ge 0}$ are the weights of the two penalties:

$$
\mathcal{L}(\gvect{\theta}) = \frac{1}{2} \sum_{n=1}^{N_\text{obs}} \frac{(y_n - x(\vect{s}_n, t_n; \gvect{\theta}))^2}{\sigma_n^2} + \frac{\beta_\text{PDE}}{2 N_c} \sum_{j=1}^{N_c} \Vert \mathcal{R}[x(\cdot; \gvect{\theta})](\vect{s}_j, t_j) \Vert^2 + \frac{\beta_\theta}{2} \Vert \gvect{\theta} \Vert^2
$$ (nf-eq-loss)

There is no background term (the $\gmat{\Sigma}_x$ term of classical 4D-Var), as the network's spectral bias plays its role. We write it as a loss $\mathcal{L}$ rather than a cost $\mathcal{J}$ because the control is the parameters $\gvect{\theta}$ and not a state.

### geostack ([theme](https://github.com/jejjohnson/geostack/issues/83))

| Issue | What | Stage |
| --- | --- | --- |
| [geostack#84](https://github.com/jejjohnson/geostack/issues/84) | [Design] the `stat` / `moment` dim for uncertainty outputs (CS-F15) | A |
| [geostack#85](https://github.com/jejjohnson/geostack/issues/85) | Neural-field entries and gates for case 9, case 11 and methane stage 3 | A–C |
| [geostack#86](https://github.com/jejjohnson/geostack/issues/86) | Research: neural-field compression vs SZ3 / ZFP at matched maximum error | D |

Carrier wrappers stay out of geonnax (L1 only). The `GeoTracks` → field operator for case 9 goes behind `vardax[geo]`, and those for case 11 and stage 3 go behind `pyrox.adapters.pipekit`.

## Stages and gates

```text
  A · plumbing
    vardax#83 point obs · pyrox#316 per-obs σ · geonnax#132 diffops · geostack#84 stat dim
    ── GATE: SH-SIREN on case 9 tracks matches OI RMSE;
             BNF + σₙ on case 11 ≤ GP CRPS
        │
        ▼
  B · uncertainty + methane
    pyrox#317, #318, #319, #320, #321 · geonnax#133, #134, #135
    ── GATE: stage-3 XCH₄ gap-fill beats kriging;
             90% intervals cover 85–95%
        │
        ▼
  C · physics
    vardax#84, #85, #86, #87, #88 · geonnax#136 · pyrox#322, #323
    ── GATE: reproduce Oh (2025) on 2-D Kolmogorov flow;
             case 9 with a QG residual beats the stage-A field
        │
        ▼
  D · research bets
    geonnax#137–#141 · geostack#86
    ── GATE: each beats its stage A–C baseline on one anchor,
             or it stays a notebook
```

The gates are scored in [geostack#85](https://github.com/jejjohnson/geostack/issues/85). Stage A is low risk to start now, as every item in it also helps models without neural fields.

## Open questions

| ID | Question | Recommendation |
| --- | --- | --- |
| NF-O1 | A column averaging kernel needs the field on pressure levels, $x(\lambda, \phi, t, p; \gvect{\theta})$, or a single collapsed weight per pixel | Support both ([pyrox#318](https://github.com/jejjohnson/pyrox/issues/318)); start with the collapsed column, measure the profile version on synthetic data |
| NF-O2 | pyrox's `Parameterized` registry lives outside the pytree, so `eqx.tree_at` or `filter_jit` rebuilds lose it; `BayesianSlepianEncoder` uses it | Test composition inside wrapped cores early; move it to the `pyrox_sample` pattern if it breaks |
| NF-O3 | pyrox-nn swaps geonnax fields by name (`W`, `b`, `Omega`, `phi`, `layers`, …) | Wave 6 adds fields, never renames them |
| NF-O4 | Spectral bias as a prior gives no calibrated error bars | Report ensemble spread from the start in neural-field 4D-Var ([vardax#87](https://github.com/jejjohnson/vardax/issues/87)) |
| NF-O5 | SWOT KaRIn noise below 30–50 km is wave-correlated and a high-ω₀ SIREN will fit it as signal | Hold out SWOT swaths in case 9 and use a correlated-noise $\gmat{\Sigma}_\varepsilon$ |
| NF-O6 | Who owns field derivatives: geonnax `diffops` (autodiff on continuous fields), xrtoolz `xrgrad` (grids) or spectraldiffx (spectral)? | Different objects; keep `diffops` a submodule so the capability index shows no clash |
| NF-O7 | Is compression worth building? | Only if [geostack#86](https://github.com/jejjohnson/geostack/issues/86) shows a win at matched maximum error, or for random-access and derivative queries |

## Relationships

- Wave epic: [geostack#82](https://github.com/jejjohnson/geostack/issues/82)
- Literature: [report 18](../reports/18_neural_fields.md)
- Case studies: [case 9, case 11](./case_studies.md); stage 3 in the [overview](../overview.md)
- Layers and tiers: [levels across the stack](./stack_levels.md), [dependency graph](../api/dependency_graph.md)
