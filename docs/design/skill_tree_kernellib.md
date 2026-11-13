---
title: "kernellib's skill shelf: kernels, graphs and estimators on top of gaussx"
subtitle: "Design · the skill tree for a ladder library one tier above the base"
short_title: Skill tree · kernellib
description: "kernellib's shelf of agent skills: 17 use skills tagged by role, grouped into seven units that each carry a contract card, two of them listed, with the six contributor skills as its skeleton. kernellib is the first library that both receives cards (from gaussx) and gives them (to pyrox-gp, pyrox-lgm and manipy). Measured at the submodule pins by scripts/skill_shelf.py."
date: 2026-11-12
keywords:
  - agents
  - skills
  - kernellib
  - role ladder
  - skill tree
numbering:
  heading_1: true
  heading_2: true
  table: true
---

(sk-decision)=
## Decision

Recommended: kernellib's shelf is 17 use skills and six extend skills, each tagged with its role, grouped into seven units; two skills are listed, today's `kernel-methods-with-kernellib` as the front door and a new `kernel-estimators-and-tests`.
Alternative: one listed skill over the whole shelf, as today, which costs less in every session and leaves graphs and estimators to be found through the front door.
Status: open, 12 Nov 2026.

What does not change: kernellib's public API, its six contributor skills, and the name of today's skill.
This page follows the pattern of [gaussx's shelf](./skill_tree_gaussx.md) and says only where kernellib differs.

(sk-motivation)=
## Motivation

kernellib has seven kinds of thing, each of its dependents uses one to three of them, and one skill of 181 lines covers all seven.

:::{table} kernellib's public names by unit, at the submodule pin. The rungs follow the layers of its `AGENTS.md` (0 functions and kernels, 1 operators and features, 2 estimators).
:label: tbl-sk-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Kernels | ⓪ ① | ML | `kernellib.functional` 31, `kernellib._kernels` 29, `kernellib._derivatives` 6, `kernellib._heuristics` 4, `kernellib` 2 | 72 |
| Geostatistical kernels | ⓪ | ML, geoscience | `kernellib._geo` 41 | 41 |
| Operators and feature maps | ⓪, setup | numerics, ML | `kernellib._spectral` 19, `kernellib._operators` 16 | 35 |
| Graphs | ⓪ ① | ML | `kernellib._graph` 36 | 36 |
| Estimators | ③ ④ | ML | `kernellib._regression` 15, `kernellib.sklearn` 5 | 20 |
| Decomposition | ③ | ML | `kernellib._decomposition` 14 | 14 |
| Dependence measures | ④ | ML | `kernellib._dependence` 12 | 12 |
| **All** | | | | **230** |
<!-- shelf:units:end -->
:::

:::{table} Which part of kernellib each dependent package uses.
:label: tbl-sk-consumers

<!-- shelf:uses:begin -->
| Dependent package | Names used | Kernels | Geostatistical kernels | Operators and feature maps | Graphs | Estimators | Decomposition | Dependence measures |
|---|---|---|---|---|---|---|---|---|
| pyrox-gp | 21 | 14 | 0 | 3 | 2 | 0 | 2 | 0 |
| manipy-jax | 13 | 3 | 0 | 0 | 6 | 0 | 4 | 0 |
| pyrox-lgm | 5 | 0 | 0 | 0 | 5 | 0 | 0 | 0 |
<!-- shelf:uses:end -->
:::

Three packages depend on kernellib, and none of them uses its estimators, its dependence measures or its geostatistical kernels.
pyrox-gp takes kernels and a few feature maps; manipy takes graphs, decompositions and a few kernels; pyrox-lgm takes graphs only.
The 73 names of the estimators, the dependence measures and the geostatistical kernels are used by people, not by another library.

(sk-shelf)=
## The shelf

The contributor skills are the skeleton: each `add-<thing>` names something kernellib lets you add, and so a part of the shelf.

:::{table} kernellib's shelf. Each use skill is one file; the extend skills exist today in `.claude/skills/`.
:label: tbl-sk-shelf

| Role | Skill | Covers | Subpackage |
| --- | --- | --- | --- |
| ⓪ structure | Kernels and composition | `RBF`, `Matern`, `Periodic` and the rest; `+`, `*`, `.select`, `.stretch`, warping and modulation | `_kernels` |
| ⓪ structure | Derivative kernels | `Derivative`, `DerivativeIndexed`, derivative observations | `_kernels` |
| ⓪ structure | Kernels on the sphere and on meshes | great-circle, chordal, sphere-series and mesh kernels | `_geo` |
| ⓪ structure | Vector-field and space-time kernels | curl-free, divergence-free, `Gneiting`, `Advected`, geometric anisotropy | `_geo` |
| ⓪ structure | Kernel operators | `to_operator`, `KernelOperator`, implicit and cross operators, batched matvecs | `_operators` |
| ⓪ structure | Graphs and neighbour search | `KNNGraph`, `GridGraph`, mesh graphs, Laplacians, eigenpairs | `_graph` |
| ⓪ structure | Graph and stream-network kernels | `GraphMatern`, `GraphHeat`, diffusion and commute-time kernels, tail-up and tail-down | `_graph` |
| ⓪ structure | `add-kernel`, `add-kernel-operator`, `add-graph-component` | extend | |
| ① primitive | Kernel functions on arrays | `rbf_kernel` and the other pure functions, matrix-level HSIC and CKA | `functional` |
| ① primitive | Derivatives of a predictor | `gradient`, `jacobian`, `hessian`, `laplacian`, `divergence`, `curl` | `_derivatives` |
| ① primitive | Lengthscale heuristics | `estimate_lengthscale`, `lengthscale_grid`, the gamma conversions | `_heuristics` |
| setup | Feature maps | random Fourier, orthogonal, FastFood, Nyström, Laplace-eigenfunction and Slepian features | `_spectral` |
| setup | Low-rank operators and landmarks | `nystrom_operator`, `rff_operator`, `fastfood_operator`, landmark selection | `_operators`, `_spectral` |
| setup | `add-feature-map` | extend | |
| ③ model | Kernel ridge regression at scale | `KRR`, `Falkon`, `EigenPro` and their preconditioners | `_regression` |
| ③ model | Kernel PCA and eigenmaps | `KernelPCA`, Laplacian and Schrödinger eigenmaps, locality-preserving projections | `_decomposition` |
| ③ model | scikit-learn adapters | the estimators, transformers and tests as scikit-learn objects, behind the `[sklearn]` extra | `sklearn` |
| ③ model | `add-estimator` | extend | |
| ④ algorithm | Dependence measures | `hsic`, `cka`, `kernel_alignment`, distance covariance | `_dependence` |
| ④ algorithm | Two-sample and permutation tests | `mmd_squared`, `energy_distance`, `permutation_test`, streaming CKA | `_dependence` |
| ④ algorithm | `add-dependence-measure` | extend | |
:::

kernellib has no ② of its own: its estimators take gaussx's solver strategies, so "how the solve is done" is a card from below, not a skill here.
One entry is a judgement call: feature maps are tagged setup, because a map is fitted once on a kernel and inputs and then reused, although the fitted object is a Module.

(sk-units)=
## Units and cards

A dependent package receives the card of each unit it uses at least three names of.

:::{table} The cards each dependent package would receive from kernellib.
:label: tbl-sk-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
| pyrox-gp | Kernels; Operators and feature maps | Graphs (2); Decomposition (2) |
| manipy-jax | Kernels; Graphs; Decomposition |  |
| pyrox-lgm | Graphs |  |
<!-- shelf:cards:end -->
:::

kernellib is the first library with cards flowing both ways.
From below it receives two of gaussx's seven: structures and primitives, and strategies ([](#tbl-sg-cards)).
Upwards it gives one to three of its own seven to each dependent.
So pyrox-gp's block holds kernellib's kernels card, and the gaussx cards pyrox-gp earns by its own imports, never the gaussx cards kernellib received.
A card goes exactly one edge up.

(sk-listed)=
## Listed skills

:::{table} The listed use skills. Line budgets are proposals.
:label: tbl-sk-skills

| Skill | Units | Floor it carries cards for | Asked for as |
| --- | --- | --- | --- |
| `kernel-methods-with-kernellib`, the front door, at most 150 lines | kernels; geostatistical kernels; operators and feature maps; graphs | gaussx: structures and primitives, strategies | "build a kernel", "a Gram matrix as an operator", "random features" |
| `kernel-estimators-and-tests`, at most 150 lines | estimators; decomposition; dependence measures | kernels; operators and feature maps | "kernel ridge regression", "is X independent of Y", "kernel PCA" |
:::

The second skill is listed because its audience differs: it is used mostly by people running an analysis, and among the libraries only manipy takes its decompositions.

(sk-below)=
## What kernellib needs from below

| From | Surfaces to a caller | Card in |
| --- | --- | --- |
| gaussx | yes: a Gram matrix is a gaussx operator, and estimators take a gaussx solver strategy | the front door |
| geonnax | yes, narrowly: the random-feature and basis functions the feature maps build on | operators and feature maps, once geonnax has a shelf |
| lineax, equinox | yes: operators and kernels are lineax operators and Modules | covered by gaussx's card and by `role-ladder` |

(sk-differs)=
## Where kernellib bends the pattern

1. **Units that no library uses still need a card.** The estimators, the dependence measures and the geostatistical kernels have no dependent package. Their cards are for the band front doors and the capstones, not for another library's block.
2. **A library can have no rung of its own.** kernellib's strategies are gaussx's. The shelf says so in one line and points down.
3. **The public surface is wider than the package root.** `kernellib.functional` and `kernellib.sklearn` hold 36 names the root does not export, so the measurement reads public submodules as well.

(sk-gates)=
## Gates

Gates GX-0 to GX-6 of [gaussx's shelf](./skill_tree_gaussx.md#sg-gates) apply with kernellib's names.
`tests/test_skill_shelf.py` already checks the first two: the tables above are current, and every public name belongs to a unit.

(sk-risks)=
## Risks

- **A name is counted twice or missed.** A name exported by both the root and a submodule is counted once, at the root. Fallback: the test fails if any public name belongs to no unit, so a gap is visible.

## Relationships

- Parent: jejjohnson/geostack#150
- Related: jejjohnson/geostack#151 (the design), jejjohnson/geostack#155 (shelves for the other libraries)
- Related: [skill tree](./skill_tree.md), [gaussx's shelf](./skill_tree_gaussx.md)
