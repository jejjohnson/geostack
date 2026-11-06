---
title: "argminax: optimisation with uncertainty for JAX"
short_title: "argminax"
description: "argminax is the stack's optimisation library with uncertainty at the optimum: it finds x̂ = argmin J(x), and then says how sharp that minimum is (posterior covariance, averaging kernel, degrees of freedom) and how x̂ depends on what went in. It sits at Core tier 1 on gaussx, extends optimistix with the stack's solvers and adjoints, and takes over the curvature code now rebuilt in seven places. Status: proposed, at design-sketch depth."
date: 2026-11-05
keywords:
  - argminax
  - design
  - GeoStack
---

argminax is the stack's optimisation library with uncertainty at the optimum: it finds x̂ = argmin J(x), and then says how sharp that minimum is (posterior covariance, averaging kernel, degrees of freedom) and how x̂ depends on what went in. It sits at Core tier 1 on gaussx, extends optimistix with the stack's solvers and adjoints, and takes over the curvature code now rebuilt in seven places. Status: proposed, at design-sketch depth.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | Core, tier 1 |
| Imports | gaussx; extends optimistix; JAX, Equinox, lineax |
| Imported by | radtrax, plumax, vardax, pyrox-gp, pyrox-lgm, pyrox-nn |
| Takes over | vardax's adjoint strategies and posterior adapters; radtrax's retrieval diagnostics; the Newton loops of gaussx, pyrox-gp and pyrox-lgm; plumax's 4D-Var posterior |
| Does not own | training over batches (optax, optax\_bayes); sampling (NumPyro, pyrox); adjoints through time (integrix) |
| Status | proposed; no repository yet |
| Tracked in | the [layer map](../stack_layer_map.md); the companion [functax](functax.md) and [integrix](integrix.md) designs |

```{figure} ../../images/libraries/argminax.svg
:alt: where argminax sits · Core tier 1
:width: 100%

Where argminax sits · core tier 1.
```

argminax sits one tier above gaussx, whose single steps it iterates, so every library that minimises can import it. This page is the design; core types, algorithms and the code survey are in [argminax: design details](argminax_details.md).

## The problem

A methane retrieval is not finished when the optimiser stops. radtrax's 3D-Var (`retrieval/var3d`) finds the column enhancement that best fits a hyperspectral pixel, and then needs three more things at that minimum: how uncertain the answer is, how much of it came from the measurement rather than the prior, and how its gradient flows back to the plume model that called it. Those three are the curvature at the minimum, and every library that optimises rebuilds them:

| Library | Minimises | Rebuilds at the minimum |
| --- | --- | --- |
| radtrax `retrieval/var3d` | 3D-Var by L-BFGS, Gauss–Newton or dual PSAS (optimistix) | `degrees_of_freedom_for_signal`, `posterior_covariance_proxy` (the averaging kernel appears only in docstrings) |
| plumax `les_fvm/fourdvar.py` | 4D-Var for an emission field (optimistix) | `PosteriorCovariance`, `_gauss_newton_hessian` |
| plumax `assimilation/solve.py` | the same 3D-Var solvers radtrax ported from it | — |
| vardax | 3D-Var, strong, weak and incremental 4D-Var (optimistix) | `GaussNewtonHessian`, `LaplaceCovariance`; `KStepAdjoint`, `OneStepAdjoint` |
| pyrox-gp | non-Gaussian GP likelihoods | the Newton iterations of `LaplaceInference`, `GaussNewtonInference`, `QuasiNewtonInference` |
| pyrox-lgm | `inla()`: an inner Newton fit, an outer `optax.lbfgs` search | the Laplace approximation per hyperparameter |
| gaussx | `laplace_mode` | its own Newton loop |
| pyrox (main) | `last_layer_laplace` | GGN precision, evidence, prior-precision search |

The solvers mostly come from optimistix already. What each library rewrites is what optimistix leaves out: Gauss–Newton curvature as a structured operator, its inverse as a posterior covariance, the degrees of freedom of a retrieval, and adjoints for differentiating through a minimiser that stops early. Seven copies of that machinery, across GeoModels and Core, with plumax and radtrax each holding the same 3D-Var solvers.

## The cost, its minimum, and the curvature there

A retrieval minimises a cost that balances the measurement against the prior. Here y is the measured radiance, F the forward model, x the state (the methane column), xₐ the prior state, Sε the measurement-error covariance and Sₐ the prior covariance:

```{math}
J(x) = \frac{1}{2} (y - F(x))^\top S_\varepsilon^{-1} (y - F(x)) + \frac{1}{2} (x - x_a)^\top S_a^{-1} (x - x_a)
```

Three jobs share the word optimisation, and only one belongs here:

| Job | Question | Home |
| --- | --- | --- |
| Solve | where is the minimum x̂? | optimistix, extended by argminax's solvers |
| Curvature | how sharp is the minimum, and what does that say about x̂? | **argminax** |
| Train | how do I fit a network's weights over many batches? | optax and optax\_bayes |

At the minimum x̂, with K = ∂F/∂x the Jacobian there, the Gauss–Newton curvature gives the posterior covariance Ŝ and the averaging kernel A:

```{math}
\hat S = (K^\top S_\varepsilon^{-1} K + S_a^{-1})^{-1}
```

```{math}
A = \hat S K^\top S_\varepsilon^{-1} K = I - \hat S S_a^{-1}
```

A says how much a change in the true state moves the retrieved one: A = I means the measurement decides everything, A = 0 means the prior does. Its trace is the retrieval's degrees of freedom for signal. The same Ŝ is the inverse Gauss–Newton Hessian of 4D-Var, the Laplace approximation in pyrox and the precision in gaussx; only the names differ.

The last piece is the derivative of x̂ with respect to what went in (the prior, the parameters of F). At an exact minimum it comes from the implicit function theorem; for a solver stopped early, as in learned 4D-Var, it comes from differentiating the last k iterations. Both are adjoint strategies of the minimiser.

## Modules and boundaries

Four modules: the solvers optimistix lacks, the curvature at the minimum, the retrieval diagnostics built on it, and the adjoints of a minimiser. argminax extends optimistix as gaussx extends lineax: its solvers are optimistix solvers and its adjoints are optimistix adjoints, so `optx.minimise` and `optx.least_squares` keep working.

| Module | Owns | Comes from |
| --- | --- | --- |
| `argminax.solve` | Gauss–Newton and Levenberg–Marquardt with a matrix-free inner CG (incremental 4D-Var), trust regions, the dual (observation-space) solve, positivity and bounds by reparametrisation | radtrax and plumax `run_gauss_newton`, `run_dual_psas`; pyrox-gp's and gaussx's Newton loops |
| `argminax.curvature` | Gauss–Newton, Fisher and quasi-Newton curvature, Laplace covariance Ŝ, its low-rank and diagonal forms, all as gaussx operators | vardax `GaussNewtonHessian`, `LaplaceCovariance`; plumax `PosteriorCovariance`; pyrox `last_layer_laplace`'s GGN |
| `argminax.diagnose` | averaging kernel A, degrees of freedom for signal, information content, evidence at the minimum | radtrax `var3d/diagnostics.py`; pyrox's evidence and prior-precision search |
| `argminax.adjoint` | adjoints of a minimiser stopped early: k-step and one-step. At a converged minimum optimistix's own `ImplicitAdjoint` already does the job, so argminax adds none. | vardax `KStepAdjoint`, `OneStepAdjoint`, `to_optimistix_adjoint` |

### What argminax owns, and what it hands off

The test for every name: does it find a minimum, or say something about the curvature at one? Then argminax owns it.

| Neighbour | argminax owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| optimistix | the stack's solvers and adjoints, as optimistix subclasses | `minimise`, `least_squares`, `root_find`, BFGS, its solver and adjoint protocols | argminax extends optimistix |
| gaussx | the iteration to a tolerance | one step's algebra (`newton_update`, `gauss_newton_precision`), structured solves, its Chandrupatla root finder | a single step is gaussx's; the loop around it is argminax's |
| integrix | adjoints of a minimiser | adjoints of a march: checkpointed backward sweeps through a forward run | through the iterations of an optimiser is argminax's; through time is integrix's |
| optax, optax\_bayes | deterministic solves to a tolerance | stochastic training over batches; the Bayesian learning rule | many batches and a learning rate are optax's |
| pyrox, NumPyro | the Laplace approximation as curvature | sampling, variational inference, priors | a Gaussian at the mode is argminax's; anything that samples is pyrox's |
| radtrax, plumax, vardax | the solver, curvature and diagnostics | the cost: forward model, observation operator, B and R | the library passes J (or its residuals) in |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| vardax | `KStepAdjoint`, `OneStepAdjoint`, `to_optimistix_adjoint`; `GaussNewtonHessian`, `LaplaceCovariance` | argminax: vardax, plumax |
| radtrax `retrieval/var3d` | `run_gauss_newton`, `run_dual_psas`; `degrees_of_freedom_for_signal`, `posterior_covariance_proxy` | argminax: radtrax; plumax drops its copy in `assimilation/solve.py` |
| plumax `les_fvm/fourdvar.py` | `PosteriorCovariance`, `_gauss_newton_hessian` | argminax |
| pyrox-gp | the iterations of `LaplaceInference`, `GaussNewtonInference`, `QuasiNewtonInference` and their Markov forms | argminax; pyrox-gp keeps the likelihoods and the GP algebra |
| pyrox-lgm | the inner Newton fit and the outer search of `inla()` | argminax |
| gaussx | `laplace_mode`, the loop and its entry point; gaussx keeps one step's algebra and its root finder | argminax (`amx.solve.laplace_mode`); gaussx's copy is deprecated without forwarding |
| pyrox (after pyrox#332, via pyrox-nn) | `last_layer_laplace`'s curvature, evidence and prior-precision search | argminax; pyrox-nn keeps the last-layer wrapper |

## In use

The methane retrieval end to end: solve for the column, then read the uncertainty, the averaging kernel and the degrees of freedom off the curvature at the minimum, and differentiate the answer with respect to the prior. Every argminax name is proposed; the solve itself is ordinary optimistix.

```python
# Shapes: W spectral channels; L layers in the state; Sε is (W, W), Sₐ is (L, L)

import argminax as amx
import jax
import optimistix as optx


# user
# r(x) = (Sε^(-1/2) (y − F(x)), Sₐ^(-1/2) (x − xₐ)), so J(x) = ½ ‖r(x)‖²
def residuals(state, args): ...  # (L,) → (W + L,)


# proposed (argminax)
# Gauss–Newton with an inner CG on (KᵀSε⁻¹K + Sₐ⁻¹) δ = −∇J
solver = amx.solve.GaussNewtonCG(rtol=1e-6, atol=1e-6)
# not pinned (optimistix)
# d x̂ / d(xₐ, θ) by the implicit function theorem at x̂
adjoint = optx.ImplicitAdjoint()
# not pinned (optimistix)
# x̂ = argmin J(x)
solution = optx.least_squares(  # (L,) → (L,)
    residuals, solver, prior_state, args=problem, adjoint=adjoint
)
retrieved = solution.value  # → (L,)

# --- curvature at the minimum ---
# proposed (argminax)
# Ŝ = (KᵀSε⁻¹K + Sₐ⁻¹)⁻¹, a gaussx operator, never formed densely
posterior = amx.curvature.gauss_newton(residuals, retrieved, problem)  # → (L, L) op
# proposed (argminax)
# A = I − Ŝ Sₐ⁻¹
kernel = amx.diagnose.averaging_kernel(posterior, prior_covariance)  # → (L, L) op
# proposed (argminax)
# d_s = tr(A), the degrees of freedom for signal
dofs = amx.diagnose.degrees_of_freedom(kernel)  # → ()

# --- the retrieval inside a larger model ---
# user: retrieve(xₐ) wraps the least_squares call above
# x̂ is differentiable: ∂x̂/∂xₐ flows through the implicit adjoint
sensitivity = jax.jacrev(retrieve)(prior_state)  # (L,) → (L, L)
```

A plume inversion calls `retrieve` per pixel and differentiates through it; the implicit adjoint costs one more linear solve with the same curvature, not a pass through every iteration. A learned 4D-Var that stops after a fixed number of steps swaps in `amx.adjoint.KStep(k=3)`.

## Integration across the stack

Six libraries would import argminax. For radtrax, plumax and vardax the solve call barely changes, because they already use optimistix: what changes is that the curvature, diagnostics and adjoints come from one place.

| Consumer | Uses today | With argminax |
| --- | --- | --- |
| radtrax | `retrieval/var3d`: `run_lbfgs`, `run_gauss_newton`, `run_dual_psas` on optimistix, but each returns NumPy arrays, so no retrieval is differentiable today; its own degrees of freedom and posterior proxy, with no averaging-kernel function | `amx.solve` for Gauss–Newton and the dual solve; `amx.curvature` and `amx.diagnose` for Ŝ, A and d\_s; keeps the cost and the control transforms |
| plumax | `les_fvm/fourdvar.py` with its own `PosteriorCovariance`; `assimilation/solve.py`, a copy of radtrax's solvers | `amx.curvature` for the 4D-Var posterior; drops `assimilation/solve.py` in favour of argminax (or radtrax) |
| vardax | optimistix BFGS and Gauss–Newton; its own adjoint strategies and posterior adapters | imports them from argminax; keeps the 3D-Var and 4D-Var costs and models |
| pyrox-gp | Newton, Gauss–Newton and quasi-Newton loops inside its non-Gaussian inference | `amx.solve` for the loop; keeps likelihoods, sites and GP algebra |
| pyrox-lgm | `inla()`: an inner Newton fit and an outer `optax.lbfgs` search | `amx.solve` for both; keeps the GMRF components and PC priors |
| pyrox-nn | `last_layer_laplace` (after pyrox#332) | `amx.curvature` and `amx.diagnose` for the GGN posterior and evidence; keeps the last-layer selection |

vardax's adjoint strategies are the clearest move: they are already `optimistix.AbstractAdjoint` subclasses, so for vardax only the import changes. They are not yet usable elsewhere: their `apply()` raises `NotImplementedError`, and they work only inside vardax's own unrolled solver. Wave 1 implements `apply`, so they work under any optimistix solve.

```python
# Today:
# today (vardax)
from vardax.adjoints import KStepAdjoint

# After:
# proposed (argminax)
from argminax.adjoint import KStep
```

vardax keeps a deprecated alias for one release, and `FourDVarNet1D(solver_adjoint=...)` call sites do not change.

## Decisions and open questions

Five decided, three open. Each entry gives the position first, then the trade-off.

### 1. Name: argminax

**Decided (5 Nov 2026).** `argminax`, free on PyPI when checked. It extends optimistix, as gaussx extends lineax.

### 2. Scope: solve and curvature, not training

**Decided (5 Nov 2026).** argminax owns deterministic solves to a tolerance and what the curvature at the minimum says. Stochastic training over batches stays with optax and optax\_bayes; sampling and variational inference stay with NumPyro and pyrox.

### 3. Layer: Core tier 1

**Decided (5 Nov 2026, on the layer map).** argminax imports gaussx (Core tier 0) and extends optimistix. vardax, pyrox-gp, pyrox-lgm, pyrox-nn, radtrax and plumax sit above it.

### 4. gaussx keeps one step; argminax owns the loop

**Decided (5 Nov 2026; revised 6 Nov 2026).** gaussx keeps `newton_update`, `gauss_newton_precision` and its Chandrupatla root finder, because they are single-step algebra a tier below. `laplace_mode`, the loop that iterates them, moves to argminax whole, entry point included, and its callers (pyrox-gp, pyrox-lgm) import it from there. gaussx never calls argminax: a `laplace_mode` left in gaussx that forwarded to argminax would be an edge from Core tier 0 up to tier 1, and the repository fixes such an edge by moving code, never by a re-export (`AGENTS.md`, "Boundaries"). gaussx's own `laplace_mode` keeps its current loop for one release behind a `DeprecationWarning` that names `argminax.solve.laplace_mode`, then goes.

### 5. vardax's adjoint strategies move here

**Decided (5 Nov 2026).** `KStepAdjoint` and `OneStepAdjoint` are `optimistix.AbstractAdjoint` subclasses: they differentiate through a minimiser, so they are argminax's. Today their `apply()` raises `NotImplementedError` and they work only inside vardax's unrolled solver; the move implements `apply`, so they work under any `optx.minimise` or `optx.least_squares` call. Backward sweeps through a forward run stay integrix's (integrix decision 7).

### 6. Where radtrax's 3D-Var solvers live

**Open.** `run_gauss_newton` and `run_dual_psas` are general (any sum of squared residuals; any linear observation operator), so they fit `amx.solve`. `run_lbfgs` is a thin wrapper over optimistix and may simply go. The cost, control transforms and whitening stay in radtrax. Either way, plumax's copy in `assimilation/solve.py` goes.

### 7. Positivity and bounds

**Leaning: by reparametrisation.** Methane enhancements and emission rates are positive. A log or softplus transform keeps the solve unconstrained and the curvature well defined; the posterior is then reported in the transformed space, with a note on how to map it back. Projected methods stay out until a case needs them.

### 8. Diagnostics: argminax or radtrax?

**Open.** The averaging kernel, degrees of freedom and information content come from atmospheric retrieval [@rodgers2000], but they are properties of any linear-Gaussian posterior: the same A describes a 4D-Var analysis or a GP posterior. Leaning: argminax, with the retrieval vocabulary in its docstrings.

## Roadmap

Five waves, each a GitHub issue epic in the argminax repository once it exists. The adjoints and curvature come first: they already exist in vardax and need a home, plus the apply method they lack, and every later wave uses them.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| 0. Scaffold | the repository from the library template; CI, capability index; optimistix and SciPy reference tests | — | — |
| 1. Adjoints and curvature move in | `KStep`, `OneStep`, the optimistix mapping; Gauss–Newton and Laplace curvature as gaussx operators | vardax's copies, with deprecation aliases there | 0 |
| 2. Retrieval diagnostics | averaging kernel, degrees of freedom, information content, evidence | radtrax's `var3d/diagnostics.py`; pyrox's evidence code | 1 |
| 3. Solvers | Gauss–Newton and Levenberg–Marquardt with inner CG, the dual solve, trust regions, reparametrised bounds | radtrax's and plumax's 3D-Var solvers (decision 6); gaussx's Newton loop | 1 |
| 4. Consumers switch | radtrax, plumax, vardax, pyrox-gp, pyrox-lgm and pyrox-nn onto argminax | the remaining Newton loops and posterior adapters | 1–3 |

Wave 4 is one issue per consumer, filed in that consumer's repository and linked to the argminax epic. pyrox-nn's issue follows pyrox#332, which moves `last_layer_laplace` out of the pyrox core.

## Sources

- [Optimistix documentation](https://docs.kidger.site/optimistix/): the solver and adjoint protocols argminax extends.
- @rodgers2000: the averaging kernel, degrees of freedom and information content.
- @bolte2023onestep: one-step differentiation of iterative algorithms, the basis of vardax's k-step adjoint, as its docstring cites.
- [The GeoStack layer map](../stack_layer_map.md): argminax's layer, tier, imports and moves.
- integrix: the adjoint boundary between the two.
- [pyrox#332](https://github.com/jejjohnson/pyrox/issues/332): `last_layer_laplace` out of the pyrox core.
- Code read on 5 Nov 2026 at geostack's submodule pins: radtrax `retrieval/var3d/solve.py` and `diagnostics.py`; plumax `les_fvm/fourdvar.py` and `assimilation/solve.py`; vardax `adjoints/` and `posterior/`; pyrox-gp `_inference_nongauss.py`; pyrox-lgm `_inla.py`; gaussx `_inference/`.
- pyrox `last_layer_laplace` read on pyrox `main` at 2ac68a5; geostack's pyrox pin (94eef13) does not have it yet.
