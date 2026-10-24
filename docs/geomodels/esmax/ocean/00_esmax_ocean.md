---
title: "Ocean state estimation with esmax: a primer series"
subtitle: "esmax · ocean · Eight pages from the six-step cycle to 4D-Var, hybrid models, learned assimilation and learned advection"
short_title: Overview
description: "A reading order through eight self-contained pages on ocean state estimation for esmax's ocean component: the six-step cycle in probabilities, the Gulf Stream SSH-mapping problem, strong- and weak-constraint 4D-Var, a taxonomy of hybrid models, learned data assimilation, advection on the C-grid and a learned WENO scheme."
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

esmax's ocean component exists to be the forward model inside short-window
data assimilation and the physics inside hybrid models
([esmax workspace](../../../design/esmax.md)). These eight pages build the
background for that job on one running example, mapping sea-surface height
over a Gulf Stream window with a quasi-geostrophic model: first the
generative story and the six-step cycle, then the classical estimator
(4D-Var, strong and weak), then what can be learned in the model and in the
assimilation, and finally one concrete hybrid, a learned advection scheme.

+++

## Reading order

::::{grid} 1 1 2 2

:::{card} 1 · The six-step cycle
:link: 01_six_step_cycle.md
Simulate, infer, emulate, amortize, improve: every box as a distribution,
with linear and nonlinear examples underneath. Start here.
:::

:::{card} 2 · Ocean state estimation
:link: 02_ocean_state_estimation.md
The Gulf Stream SSH problem, written once, then read box by box as operators
with inputs, outputs and shapes.
:::

:::{card} 3 · 4D-Var
:link: 03_strong_4dvar.md
Strong constraint: the cost, the adjoint, and classic, incremental and
reduced-basis algorithms, with JAX and diffrax gradient options.
:::

:::{card} 4 · Weak-constraint 4D-Var
:link: 04_weak_4dvar.md
Model error as a control: forcing, state and reduced formulations, and how
to choose $\mathbf Q$.
:::

:::{card} 5 · Hybrid model taxonomy
:link: 05_hybrid_model_taxonomy.md
Ten things that can be learned in an ocean model, from parameters to the
rollout, and what each keeps.
:::

:::{card} 6 · Learned data assimilation
:link: 06_learned_data_assimilation.md
Eleven ways to learn a piece of the analysis: objective, gradient, solver,
output, or the whole inverse.
:::

:::{card} 7 · Advection on the C-grid
:link: 07_advection_c_grid.md
Flux, advective and Arakawa Jacobian forms in finitevolX's conventions, and
what each conserves.
:::

:::{card} 8 · Learned WENO
:link: 08_learned_weno.md
Fixed WENO stencils with learned weights: conservative, mask-aware, and with
a trustworthy adjoint.
:::

::::

## Conventions shared by all eight pages

**Notation.** $z$ is the latent state, $x$ covariates, $u$ controls and
forcing, $y$ observations, $m$ the observation mask, $\theta$ physical
parameters, $\alpha$ hyperparameters, and $\phi$, $\psi$ the parameters of
learned densities (emulator and amortized posterior).

**Operators.** Every step is drawn as an operator box: its inputs and outputs,
each with a glyph for its kind (scalar, field, cube, ensemble, mask), a shape
written with named dimensions (`(time, height, width)`, `(state,)`,
`(basis,)`) rather than sizes, and units where they apply.

**Code sketches.** Each sketch opens with a shapes header; each operation has
its equation as a comment above it and its output shape in an aligned column.
Markers say what exists: `today` ships on the named library's `main`,
`proposed` does not exist yet, `not pinned` marks a third-party API that was
not checked against a release.

**Grids.** Stencils follow finitevolX: arrays are `[Ny, Nx]` with a ghost
ring, and `T[j, i]` (centre), `U[j, i]` at $(j, i+\tfrac12)$, `V[j, i]` at
$(j+\tfrac12, i)$ and `X[j, i]` (corner) share an index.

**Numbers.** The Lorenz-96 and one-dimensional advection results on the
4D-Var, weak-constraint, advection and learned-WENO pages come from small JAX
and NumPy experiments run for these pages in double precision; each page
states its settings next to the numbers. They are not yet stages of this
repository's DVC pipeline.

## Where the pieces live

| Library | Used on these pages for | Pages |
|---|---|---|
| somax (the future esmax ocean component) | the QG model, its forcing and the forcing-basis design | 2, 3, 4, 5, 6, 8 |
| finitevolX | C-grid operators, advection forms and WENO reconstructions | 2, 5, 6, 7, 8 |
| vardax | strong, incremental and weak-constraint 4D-Var | 2, 3, 4, 6 |
| gaussx | covariance operators and their solves | 2, 5, 6 |
| pipekit | operator protocols for the learned-assimilation types | 6 |
