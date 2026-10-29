---
title: "Ocean state estimation with esmax: a primer series"
subtitle: "esmax · ocean · Ten pages from the six-step cycle to 4D-Var, hybrid models, learned assimilation, learned advection, variational DA as probabilistic programs and non-dimensional models"
short_title: Overview
description: "A reading order through ten self-contained pages on ocean state estimation for esmax's ocean component: the six-step cycle in probabilities, the Gulf Stream SSH-mapping problem, strong- and weak-constraint 4D-Var, a taxonomy of hybrid models, learned data assimilation, advection on the C-grid, a learned WENO scheme, the variational estimators written as numpyro programs, and non-dimensional models with standardised states."
date: 2026-10-24
keywords:
  - esmax
  - somax
  - ocean
  - data assimilation
  - 4D-Var
  - hybrid models
  - emulators
  - advection
---

+++ {"part": "abstract"}

esmax's ocean component exists to be the forward model inside short-window data assimilation and the physics inside hybrid models ([esmax workspace](../../../design/esmax.md)). These ten pages build the background for that job on one running example, which is mapping sea-surface height over a Gulf Stream window with a quasi-geostrophic model. We start with the generative story and the six-step cycle, then the classical estimator (4D-Var, strong and weak), then what can be learned in the model and in the assimilation, one concrete hybrid, a learned advection scheme, and finally the variational estimators again, this time written as probabilistic programs. Page 10 is the scaling layer under all of them: models built from dimensionless numbers, and states standardised by the same affine map.

+++

## Reading order

::::{grid} 1 1 2 2

:::{card} 1 · The six-step cycle
:link: 01_six_step_cycle.md
Simulate, infer, emulate, amortize, improve: every box as a distribution, with linear and nonlinear examples underneath. Start here.
:::

:::{card} 2 · Ocean state estimation
:link: 02_ocean_state_estimation.md
The Gulf Stream SSH problem, written once and then read box by box as operators with inputs, outputs and shapes.
:::

:::{card} 3 · 4D-Var
:link: 03_strong_4dvar.md
Strong constraint: the cost, the adjoint, and the classic, incremental and reduced-basis algorithms, with JAX and diffrax options for the gradient.
:::

:::{card} 4 · Weak-constraint 4D-Var
:link: 04_weak_4dvar.md
Model error as a control: the forcing, state and reduced formulations, and how to choose the model-error covariance $\gmat{\Sigma}_\eta$.
:::

:::{card} 5 · Hybrid model taxonomy
:link: 05_hybrid_model_taxonomy.md
Ten things that can be learned in an ocean model, from the parameters to the rollout, and what each one keeps.
:::

:::{card} 6 · Learned data assimilation
:link: 06_learned_data_assimilation.md
Eleven ways to learn a piece of the analysis: the objective, the gradient, the solver, the output, or the whole inverse.
:::

:::{card} 7 · Advection on the C-grid
:link: 07_advection_c_grid.md
Flux, advective and Arakawa Jacobian forms in finitevolX's conventions, and what each one conserves.
:::

:::{card} 8 · Learned WENO
:link: 08_learned_weno.md
Fixed WENO stencils with learned weights, which stay conservative and mask-aware and have a trustworthy adjoint.
:::

:::{card} 9 · Variational DA as programs
:link: 09_variational_programs.md
3D-Var and strong, incremental and weak-constraint 4D-Var, each full and reduced, as numpyro models with gaussx, kernellib and geonnax priors. Read it after 3 and 4.
:::

:::{card} 10 · Non-dimensional models
:link: 10_nondim_standardisation.md
Build models from dimensionless numbers with resolution guards, and scale states with one affine map which the model, the DA and the emulators share.
:::

::::

## Conventions shared by all ten pages

**Notation.** We write $\vect{x}$ for the state, $\vect{u}$ for the exogenous inputs (covariates, forcing and controls), $\vect{y}$ for the observations, $\vect{m}$ for the observation mask, $\gvect{\theta}$ for the physical parameters and $\gvect{\alpha}$ for the hyperparameters, whereas $\gvect{\theta}_\text{emu}$ and $\gvect{\nu}$ are the parameters of the learned densities (the emulator and the amortized posterior). Page 9 also has its own letters for a method's whitened controls and for the reduced coefficients, and its notation table says so. [CHECK: the full pages under `pages/` still use the older letters ($z$ for the latent state, $x$ for covariates, $u$ for controls and forcing, $\phi$ and $\psi$ for the learned densities, and on page 9 $v$ for whitened controls and $u$ for reduced coefficients); they need the same pass before this paragraph describes them.]

**Operators.** Every step is drawn as an operator box with its inputs and outputs. Each one has a glyph for its kind (scalar, field, cube, ensemble, mask), a shape written with named dimensions (`(time, height, width)`, `(state,)`, `(basis,)`) rather than sizes, and units where they apply.

**Code sketches.** Each sketch opens with a shapes header, and each operation has its equation as a comment above it and its output shape in an aligned column. Markers say what exists: `today` ships on the named library's `main`, `proposed` does not exist yet, and `not pinned` marks a third-party API which was not checked against a release.

**Grids.** Stencils follow finitevolX, so arrays are `[Ny, Nx]` with a ghost ring, and `T[j, i]` (centre), `U[j, i]` at $(j, i + 1/2)$, `V[j, i]` at $(j + 1/2, i)$ and `X[j, i]` (corner) share an index.

**Numbers.** The Lorenz-96 and one-dimensional advection results on the 4D-Var, weak-constraint, advection, learned-WENO and variational-programs pages come from small JAX and NumPy experiments which we ran for these pages in double precision, and each page states its settings next to the numbers. They are not yet stages of this repository's DVC pipeline.

## Where the pieces live

| Library | Used on these pages for | Pages |
|---|---|---|
| somax (the future esmax ocean component) | the QG model, its forcing and the forcing-basis design; scales, `StateAffine` and `from_nondimensional` | 2, 3, 4, 5, 6, 8, 10 |
| finitevolX | C-grid operators, advection forms and WENO reconstructions | 2, 5, 6, 7, 8 |
| vardax | strong, incremental and weak-constraint 4D-Var | 2, 3, 4, 6 |
| gaussx | covariance operators, their square roots and solves, Gaussian distributions | 2, 5, 6, 9 |
| kernellib | kernels for the covariances, truncated Fourier features, spectral densities | 9 |
| geonnax | Laplacian eigenfunction and Legendre bases | 9 |
| pipekit | operator protocols for the learned-assimilation types | 6 |
| paramax, flowjax | constrained parameters (core), learned priors (optional extra) | 10 |
