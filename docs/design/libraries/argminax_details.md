---
title: "argminax: design details"
short_title: "argminax details"
description: "Core types, algorithms, JAX notes, tests and the code survey behind the argminax design."
date: 2026-11-05
keywords:
  - argminax
  - design
  - details
  - GeoStack
---

Core types, algorithms, JAX notes, tests and the code survey behind the [argminax design](argminax.md).

## Core types

Four modules, each a thin layer on optimistix, lineax or gaussx. Every solver is an optimistix solver and every adjoint an optimistix adjoint, so `optx.minimise` and `optx.least_squares` run them unchanged.

```python
# Shapes: N state size; M residuals; K probes or CG steps; n draws
from collections.abc import Callable

import equinox as eqx
import gaussx as gx
import lineax as lx
import optimistix as optx
from jaxtyping import Array, Float, PRNGKeyArray


# proposed (argminax.solve)
# (JᵀJ + λI) δ = −Jᵀr, by CG on δ ↦ Jᵀ(Jδ) + λδ; JᵀJ is never formed
class GaussNewtonCG(optx.AbstractGaussNewton):
    rtol: float
    atol: float
    norm: Callable = eqx.field(static=True, default=optx.max_norm)
    damping: float = 0.0  # λ > 0: Levenberg–Marquardt
    max_cg_steps: int = eqx.field(static=True, default=50)
    descent: optx.AbstractDescent = ...  # NewtonDescent(linear_solver=lx.CG)
    search: optx.AbstractSearch = ...  # LearningRate(1.0) or a trust region
    verbose: frozenset[str] = eqx.field(static=True, default=frozenset())


# proposed (argminax.solve)
# (H B Hᵀ + R) λ = d, then δx̂ = B Hᵀ λ: the solve in observation space
def dual_solve(
    jvp: Callable,  # δx ↦ H δx      (N,) → (M,)
    vjp: Callable,  # δy ↦ Hᵀ δy     (M,) → (N,)
    background_cov: lx.AbstractLinearOperator,  # B               (N, N), PSD
    obs_cov: lx.AbstractLinearOperator,  # R               (M, M), PSD
    innovation: Float[Array, " M"],  # d = y − H(x_b)
    *,
    solver: lx.AbstractLinearSolver,
) -> Float[Array, " N"]: ...  # δx̂


# proposed (argminax.curvature)
# the Gaussian at the minimum, N(x̂, Ŝ) with Ŝ = Ĥ⁻¹
class Laplace(eqx.Module):
    mode: Float[Array, " N"]  # x̂
    precision: lx.AbstractLinearOperator  # Ĥ  (N, N), PSD-tagged
    solver: gx.AbstractSolverStrategy = gx.DenseSolver()

    def covariance(self) -> lx.AbstractLinearOperator: ...  # Ŝ, lazy (N, N)
    def logdet_precision(self) -> Float[Array, ""]: ...  # log |Ĥ|
    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n N"]: ...


# proposed (argminax.curvature)
# Ĥ = JᵀJ at x̂ for whitened residuals r(x)
def gauss_newton(residuals: Callable, mode, args) -> Laplace: ...


# proposed (argminax.diagnose)
# a trace and its Monte Carlo standard error (0 when exact)
class TraceEstimate(eqx.Module):
    value: Float[Array, ""]
    stderr: Float[Array, ""]


# proposed (argminax.diagnose)
# A = I − Ŝ Sₐ⁻¹, lazy;  d_s = tr(A): exact for small N, Hutchinson above
def averaging_kernel(posterior: Laplace, prior_cov) -> lx.AbstractLinearOperator: ...
def degrees_of_freedom(
    kernel: lx.AbstractLinearOperator,
    *,
    key: PRNGKeyArray | None = None,  # None: exact trace
    num_probes: int = 32,  # K
) -> TraceEstimate: ...


# proposed (argminax.adjoint)
# ∂x̂/∂θ through the last k iterations; x_{T−k} under stop_gradient
class KStep(optx.AbstractAdjoint):
    k: int = eqx.field(static=True, default=1)

    def apply(self, primal_fn, rewrite_fn, inputs, tags): ...


# proposed (argminax.adjoint)
# k = 1 (Bolte, Pauwels and Vaiter, 2023)
class OneStep(KStep): ...


# proposed (argminax.adjoint)
# pipekit-cycle spec → optimistix adjoint, matched by class name;
# pipekit-cycle (Core tier 2) is never imported
def to_optimistix_adjoint(spec) -> optx.AbstractAdjoint: ...
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `GaussNewtonCG` | tolerances, norm, damping λ, CG budget, descent, search | an `optx.AbstractGaussNewton`; never materialises JᵀJ; λ = 0 is Gauss–Newton, λ > 0 Levenberg–Marquardt |
| `dual_solve` | (a function) | B and R PSD-tagged; on a linear H its δx̂ equals the primal Gauss–Newton step to the CG tolerance |
| `Laplace` | x̂, Ĥ, a gaussx solver strategy | Ĥ is a PSD-tagged lineax operator; Ŝ is formed only on request; sampling takes an explicit key |
| `TraceEstimate` | value, standard error | stderr = 0 exactly when the trace was computed exactly |
| `KStep`, `OneStep` | k | `apply` implemented; the forward value equals the plain solve; gradients flow through k solver steps |
| `Implicit` | a linear solver | `optx.ImplicitAdjoint` with CG on the curvature as default; valid only at a converged minimum |
| `to_optimistix_adjoint` | (a function) | `Implicit` → `Implicit`, `Truncated(k)` → `KStep(k)`, `RecursiveCheckpoint` → `optx.RecursiveCheckpointAdjoint`; `Direct` and `Backsolve` raise, being integrix's |

`amx.adjoint.Implicit` earns its name only if its default differs from `optx.ImplicitAdjoint` (CG with a PSD tag instead of a dense LU); otherwise users pass `optx.ImplicitAdjoint(linear_solver=lx.CG(...))` and argminax adds no name.

## Types to adhere to

- **optimistix callables.** `minimise` takes `fn(y, args) -> scalar`, `least_squares` takes `fn(y, args) -> residuals` (any pytree), both with `has_aux`. argminax adds no callable protocol of its own.
- **Whitened residuals.** The curvature and diagnostics assume r(x) = (Sε^(−1/2)(y − F(x)), Sₐ^(−1/2)(x − xₐ)), as in the Design sketch, so Ĥ = JᵀJ with no separate B or R. The whitening (radtrax's `WhiteningTransform`, plumax's χ-space) stays with the consumer; `averaging_kernel` takes Sₐ only to map back.
- **Linearised operators.** argminax takes a `jvp` / `vjp` pair or a residual function, never vardax's `ObservationOperator`; vardax's `obs_op.linearize` is turned into the pair on vardax's side, since vardax sits above.
- **gaussx operators and strategies.** Every curvature is a lineax operator with `lx.positive_semidefinite_tag`; solves, log-determinants and Cholesky factors go through gaussx and respect a `solver=` strategy (default `gx.DenseSolver()`).
- **gaussx single steps.** The damped Newton loop for a Gaussian prior and a factorised likelihood iterates `gx.newton_update`; the GGN low-rank form is `gx.gauss_newton_precision`.
- **pipekit-cycle adjoint specs.** `Implicit`, `Truncated`, `RecursiveCheckpoint`, `Direct`, `Backsolve` from `pipekit_cycle.adjoints` are the vocabulary. `to_optimistix_adjoint` matches them structurally, by class name and fields, as integrix's `to_diffrax_adjoint` does, and never imports pipekit-cycle: it sits at Core tier 2, above argminax, and a lazy or optional import still counts as an edge.
- **vardax's `Posterior`.** `Posterior(mean, cov, samples, provenance)` stays in vardax; vardax wraps a `Laplace` into it.

## Algorithms

Each algorithm below is written in whitened coordinates ξ, where the prior is N(0, I) and r(ξ) stacks the measurement and prior residuals, so J(ξ) = ½ ‖r(ξ)‖² and Ĥ = JᵀJ. The Jacobian J here is ∂r/∂ξ, not the cost.

### 1. Gauss–Newton and Levenberg–Marquardt with an inner CG

```{math}
(J_k^\top J_k + \lambda_k I) \delta_k = -J_k^\top r(\xi_k)
```

```{math}
\xi_{k+1} = \xi_k + \alpha_k \delta_k
```

1. Linearise r at ξₖ with `jax.linearize`, which gives v ↦ Jₖv; transpose it with `jax.linear_transpose` for w ↦ Jₖᵀw.
2. Wrap v ↦ Jₖᵀ(Jₖv) + λₖv as an `lx.FunctionLinearOperator` tagged PSD and solve with `lx.CG`, capped at `max_cg_steps`.
3. Take the step with the search: αₖ = 1 for Gauss–Newton, or a trust region (`optx.ClassicalTrustRegion`) that updates λₖ from the ratio of actual to predicted decrease.
4. Stop when the optimistix termination test passes (step and cost change within `rtol` and `atol` in `norm`).

**Cost.** Per outer step, one linearisation and K CG iterations of one JVP and one VJP each, so about 2K forward-model evaluations; memory O(N + M) plus the linearisation's stored primals.

**Pitfalls.**

- A forward model wrapped in diffrax's `custom_vjp` adjoints refuses forward mode (plumax hit this, hence its `jax.jacrev`). Then Jₖv comes from the transpose of the VJP, or the forward model is run with `diffrax.DirectAdjoint`.
- An inexact CG gives an inexact Newton step. Tie the CG tolerance to the outer residual (Eisenstat–Walker) rather than to a fixed number.
- Without damping, a rank-deficient Jₖ makes the CG system singular; the prior rows of the whitened residual keep Ĥ ≥ I, which is one more reason to whiten.

### 2. The dual (observation-space) solve

```{math}
(H B H^\top + R) \lambda = d
```

```{math}
\delta \hat x = B H^\top \lambda
```

1. Form the innovation d = y − H(x\_b) at the background state.
2. Solve for λ by CG in observation space on the PSD operator v ↦ H(B(Hᵀv)) + Rv, preconditioned by R⁻¹.
3. Map back: δx̂ = B Hᵀ λ.
4. For a nonlinear H, relinearise at x\_b + δx̂ and repeat: the incremental (outer-loop) form.

**Cost.** K CG steps, each one JVP, one VJP and one product with B and with R; memory O(M). It pays when M ≪ N, as for a few hundred spectral channels against a gridded state.

**Pitfalls.**

- radtrax's `run_dual_psas` takes R⁻¹ as one float (`obs_inv_variance`); the argminax version takes R as an operator.
- The dual and primal steps agree only for a linear H; the test suite checks that, not more.
- CG converges in the norm of the dual operator, so its tolerance does not translate directly into an error on δx̂.

### 3. Laplace covariance and sampling

```{math}
\hat S = \hat H^{-1} = (J^\top J)^{-1}
```

```{math}
\delta = \hat H^{-1} J^\top z \text{ for } z \sim N(0, I_{M+N})
```

1. At ξ̂, build Ĥ as a matrix-free operator (step 2 of algorithm 1 with λ = 0), or densely with `jax.jacrev` and `gx.cholesky` when N is small (up to a few thousand).
2. Read marginal variances from the Cholesky factor when dense; when matrix-free, estimate the diagonal of Ŝ by probing, or use a Lanczos low-rank form.
3. Sample densely as ξ̂ + L⁻ᵀz with Ĥ = LLᵀ. Matrix-free, sample by perturbation: draw z, solve Ĥδ = Jᵀz by CG. Then Cov(δ) = Ĥ⁻¹ JᵀJ Ĥ⁻¹ = Ĥ⁻¹, exact because the prior rows are part of J.
4. Map back to the physical state: Pₓ = Lₐ Ŝ Lₐᵀ, with Sₐ = LₐLₐᵀ (plumax's P\_S = L P\_χ Lᵀ).

**Cost.** Dense: one O(N³) factorisation, then O(N²) per sample. Matrix-free: one CG solve per sample or per probe.

**Pitfalls.**

- `jacrev` costs M VJPs; when M ≫ N and forward mode is allowed, N JVPs (`jacfwd`) are cheaper.
- A dense inverse drifts from symmetry; symmetrise before tagging PSD (plumax does).
- The Gauss–Newton Ĥ drops the residual-curvature term Σ rᵢ∇²rᵢ: fine at small residuals, optimistic at large ones.

### 4. Averaging kernel, degrees of freedom and information content

```{math}
A = I - \hat S S_a^{-1}
```

```{math}
A_\xi = I - \hat H_\xi^{-1}
```

```{math}
d_s = \operatorname{tr}(A) = \sum_i \frac{\sigma_i^2}{1 + \sigma_i^2}
```

```{math}
H_{\mathrm{info}} = \frac{1}{2} \log |\hat H_\xi| = \frac{1}{2} \sum_i \log(1 + \sigma_i^2)
```

Here σᵢ are the singular values of the whitened measurement Jacobian K̃ = Sε^(−1/2) K Sₐ^(1/2), so Ĥ\_ξ = I + K̃ᵀK̃ [@rodgers2000, chapter 2].

1. Exact path (small N): an SVD of K̃ or an `eigh` of Ĥ\_ξ gives A, d\_s and H\_info at once.
2. Large N: d\_s = N − tr(Ĥ\_ξ⁻¹). Estimate the trace by Hutchinson with K Rademacher probes zₖ drawn from a JAX key, each a CG solve with Ĥ\_ξ: tr(Ĥ⁻¹) ≈ (1/K) Σ zₖᵀĤ⁻¹zₖ.
3. Report the standard error, the sample standard deviation of zₖᵀĤ⁻¹zₖ over √K, in the `TraceEstimate`.
4. The averaging kernel itself stays lazy: A v = v − Ĥ⁻¹v in whitened space; rows of A are formed only on request.

**Cost.** Exact: O(MN²) for the SVD. Hutchinson: K CG solves, independent of N in count.

**Pitfalls.**

- radtrax draws its probes with `np.random.default_rng(seed)` on the host, so the estimate cannot run under `jit` or `vmap` and repeats the same probes in every pixel of a batch. JAX keys fix both.
- An inexact CG biases each probe; the standard error does not see that bias, so the CG tolerance must be tighter than the target error.
- Rademacher probes have lower variance than Gaussian ones for the same K; Hutch++ cuts K further when the spectrum decays fast.

### 5. The evidence at the minimum

```{math}
\log p(y) \approx -J(\hat \xi) - \frac{1}{2} \log |\hat H_\xi| - \frac{1}{2} \log |2 \pi S_\varepsilon|
```

1. J(ξ̂) comes from the solve; log |Ĥ\_ξ| from `gx.logdet` (dense) or stochastic Lanczos quadrature (matfree, as pyrox-lgm's PC priors use).
2. For a prior precision λ (pyrox's last-layer Laplace), maximise the evidence over log λ: a 1-D problem, cheap once the GGN's eigenvalues are known, since log |Ĥ| = Σ log(gᵢ + λ).

**Cost.** Dense: the factorisation already made for Ŝ. Matrix-free: K Lanczos runs of a few tens of steps each.

**Pitfalls.**

- The Gauss–Newton evidence inherits the Gauss–Newton curvature's optimism at large residuals.
- pyrox's `optimize_prior_precision` scans 4097 points on log λ, then refines by golden section: robust to a bimodal evidence, and argminax keeps that rather than a local 1-D minimiser.

### 6. Damped Newton for a Gaussian prior and a factorised likelihood

The loop inside gaussx's `laplace_mode`, pyrox-gp's `LaplaceInference` and `GaussNewtonInference`, and pyrox-lgm's inner fit. With prior N(m, K), per-site gradient gₜ and curvature Wₜ = max(−∇²log p, floor):

```{math}
\tilde f_t = (K^{-1} + W_t)^{-1} (W_t f_t + g_t + K^{-1} m)
```

```{math}
f_{t+1} = f_t + \eta (\tilde f_t - f_t)
```

1. Per-site gₜ and Wₜ by `vmap` of `grad` and second derivative of log p(yᵢ | fᵢ).
2. Floor Wₜ (pyrox-gp's `precision_floor`, 1e-3 for Gauss–Newton) so the step stays PSD for non-log-concave likelihoods such as Student-t.
3. One step's algebra is `gx.newton_update`; argminax owns the loop: `lax.while_loop` on ‖fₜ₊₁ − fₜ‖∞ < tol, with damping η ∈ (0, 1\].
4. Make the mode implicit for gradients (gaussx's `custom_root`), so a VJP costs one solve with K⁻¹ + W, not a pass through the iterations.

**Cost.** Per iteration one solve with K⁻¹ + W (O(N³) dense, less for a GMRF or a state-space prior); typically 5–20 iterations.

**Pitfalls.**

- pyrox-gp's loops are Python `for` loops with an early `break` on a traced value, so they run only eagerly: no `jit` around `fit`, no `vmap` over datasets.
- gaussx floors the tolerance at 64 machine epsilons; keep that, or float32 never converges.

### 7. The implicit adjoint

```{math}
\frac{\partial \hat \xi}{\partial \theta} = -(\nabla^2_{\xi \xi} J)^{-1} \nabla^2_{\xi \theta} J
```

1. Solve forward as usual; optimistix records the optimality condition ∇ξJ(ξ̂, θ) = 0.
2. On the backward pass, solve (∇²ξξJ)ᵀ w = ḡ for the incoming cotangent ḡ with the adjoint's linear solver (CG for argminax's default).
3. Pull −w through ∇²ξθJ by one VJP to get the cotangent of θ.

**Cost.** One linear solve (K Hessian-vector products) per VJP, whatever the number of forward iterations; no forward iterates stored.

**Pitfalls.**

- The exact Hessian is used, not the Gauss–Newton Ĥ; it is PSD only at a strict local minimum, so a PSD tag on it at a saddle makes CG fail silently. Tag it symmetric and check `solution.result`.
- The gradient is wrong if the solve did not converge. radtrax and plumax call `optx.least_squares(..., throw=False)` and keep going; with argminax, an unconverged solve either raises or masks its gradient.

### 8. k-step and one-step adjoints

```{math}
\hat \xi = \Phi^{k}(\bar \xi, \theta) \text{ with } \bar \xi = \mathrm{sg}(\xi_{T-k})
```

Here Φ is one solver step and sg is `stop_gradient`. For a Newton-like Φ, ∂Φ/∂ξ → 0 at the minimum, so the one-step derivative is first-order exact there [@bolte2023onestep].

1. In `KStep.apply`, run the solve to the end under `stop_gradient`.
2. From the stopped iterate, take k more `solver.step` calls, differentiably in θ.
3. Return the plain solve's value with the k-step path's gradient (straight-through: value + (Φᵏ − sg(Φᵏ))), so the forward value does not change.

**Cost.** k extra solver steps of compute and their memory; no linear solve.

**Pitfalls.**

- `apply` has to reach the solver, its state and `args` through optimistix's `inputs` tuple, which is private; pin optimistix's minor version and test the layout.
- A line search inside `step` makes the k steps branchy; with k small, keep the search but expect a noisier gradient.
- vardax's FourDVarNet unrolls its own inner solver and dispatches on `isinstance(adjoint, KStepAdjoint)`; that path keeps working, and the generic `apply` is new.

### 9. Positivity and bounds by reparametrisation

```{math}
x = g(u)
```

```{math}
G = \operatorname{diag}(g'(\hat u))
```

```{math}
\hat S_x \approx G \hat S_u G^\top
```

1. Solve, take curvature and diagnose in u, with g = softplus or exp and the prior defined in u.
2. Report Ŝᵤ, or push it forward linearly; better, sample in u and map each draw through g.

**Pitfalls.**

- The linear push-forward fails where g′ varies across the posterior width, which is near zero, exactly where bounds matter; prefer the mapped samples.
- The averaging kernel in u is not the averaging kernel in x; say which one is reported.

## JAX compatibility

| Name | `jit` | `vmap` | `grad` | Notes |
| --- | --- | --- | --- | --- |
| `GaussNewtonCG` (under `optx.least_squares`) | yes | yes | through the adjoint | needs Jₖv: forward mode, or the transposed VJP for a `custom_vjp` model |
| `dual_solve` | yes | yes | yes | lineax CG is differentiable; the CG budget is static |
| `gauss_newton`, `Laplace` | yes | yes | yes | the dense path fixes N at trace time; `sample` takes a key |
| `averaging_kernel` | yes | yes | yes | lazy; rows formed only on request |
| `degrees_of_freedom` | yes | yes | yes | exact path static on N; Hutchinson needs a key, split per pixel under `vmap` |
| damped Newton loop | yes | yes | implicit | `lax.while_loop`; the mode is a `custom_root` |
| `Implicit` | yes | yes | yes | valid only at a converged minimum |
| `KStep`, `OneStep` | yes | yes | yes | forward value unchanged; gradient from k steps |

- **x64.** Tests run with x64 on; constants are built in the input's dtype, so float32 callers stay float32, and tolerances have a floor of a few machine epsilons.
- **`vmap` over pixels.** A `while_loop` under `vmap` runs until the slowest pixel converges; per-pixel step counts come back in the solution's stats, not as a Python int.
- **Errors under `jit`.** `throw=True` raises through `eqx.error_if`, which works inside `jit`; with `throw=False` the caller must read `solution.result` before trusting the value or its gradient.
- **No host round trips.** No NumPy RNG, no `float(...)` on a traced value, no Python `break`: the three things that keep today's radtrax diagnostics, pyrox-gp loops and pyrox-lgm's outer search out of `jit`.

## Testing

- **Linear-Gaussian closed form.** For a linear F every quantity is exact: one Gauss–Newton step reaches x̂, and Ŝ, A, d\_s, H\_info and the evidence match the textbook formulas on a dense reference (`jnp.linalg` on the materialised matrices, which is fine in tests).
- **SciPy reference.** Small nonlinear problems against `scipy.optimize.least_squares` and `minimize` for x̂, to the solver tolerance.
- **Dual against primal.** On a linear H, `dual_solve` and `GaussNewtonCG` give the same δx̂ to the CG tolerance.
- **Hutchinson by its own distribution.** The estimate lies within four standard errors of the exact trace (a bound from its sampling distribution, stated in the test), and the reported stderr matches the spread over repeated keys.
- **Sampling.** The sample covariance of perturbation draws matches Ŝ within its Wishart sampling bound.
- **Adjoints.** `Implicit` against central finite differences and against `optx.RecursiveCheckpointAdjoint` on a converged problem; `KStep` leaves the forward value bit-identical, and `OneStep` on a quadratic with an exact Newton step equals the implicit gradient.
- **The mapping.** Every pipekit-cycle spec maps to the documented optimistix adjoint; `Direct` and `Backsolve` raise a `ValueError` that names integrix.
- **Ports.** Before deleting a copy, its results are reproduced: radtrax's and plumax's `run_gauss_newton` and `run_dual_psas`, plumax's `PosteriorCovariance`, vardax's `GaussNewtonHessian`, pyrox-gp's `GaussNewtonInference`; pyrox-lgm's R-INLA fixtures (slow tier) still pass after its inner and outer loops switch.
- **Transforms.** One smoke test per public name under `jit`, `vmap` and `grad`, in the fast tier.

## External dependencies

| Library | Used for | Constraint |
| --- | --- | --- |
| optimistix | solver and adjoint protocols, `minimise`, `least_squares`, `ImplicitAdjoint`, `RecursiveCheckpointAdjoint`, trust regions | pin the minor version: `KStep.apply` reads its private `inputs` layout |
| lineax | operators, tags, CG | operators carry `positive_semidefinite_tag` |
| gaussx | solves, `logdet`, `cholesky`, `inv`, `symmetrize`, `newton_update`, `gauss_newton_precision` | public API only; gaussx never imports argminax |
| equinox, jax, jaxtyping | modules, transforms, shape annotations | — |
| matfree | stochastic Lanczos quadrature for log-determinants | optional extra, imported lazily |
| pipekit-cycle | nothing: `to_optimistix_adjoint` matches its adjoint specs by class name | never imported; it sits at Core tier 2, above argminax |
| SciPy | reference solutions | tests only |

## Where it stands

Read on 5 Nov 2026 at geostack's submodule pins, except where noted.

| Code | Commit | What it does today | Moves to |
| --- | --- | --- | --- |
| vardax `adjoints/k_step.py` `KStepAdjoint`, `one_step.py` `OneStepAdjoint` | `90839c2` | `optx.AbstractAdjoint` subclasses with `k: int = 1`, but `apply()` raises `NotImplementedError`: they are markers that vardax's `_src/solver.py` dispatches on with `isinstance` | `amx.adjoint.KStep`, `OneStep`, with `apply` implemented |
| vardax `adjoints/mapping.py` `to_optimistix_adjoint` | `90839c2` | `Implicit` → `optx.ImplicitAdjoint`, `RecursiveCheckpoint` → `optx.RecursiveCheckpointAdjoint`, `Truncated(k)` → `KStepAdjoint(k)`; `Direct`, `Backsolve` raise | `amx.adjoint.to_optimistix_adjoint` |
| vardax `posterior/gauss_newton.py` `GaussNewtonHessian`, `posterior/laplace.py` `LaplaceCovariance` | `90839c2` | HᵀR⁻¹H + B⁻¹ from `obs_op.linearize` (`n_krylov=50`); the `Posterior` container and `PosteriorAdapter` protocol stay | `amx.curvature` |
| radtrax `retrieval/var3d/solve.py` | `ed04971` | `run_lbfgs`; `run_gauss_newton`, a thin `optx.GaussNewton` wrapper with `throw=False`; `run_dual_psas` with a scalar R⁻¹. All return a NumPy `SolveResult`, so no retrieval is differentiable today | `amx.solve` (decision 6) |
| radtrax `retrieval/var3d/diagnostics.py` | `ed04971` | `reduced_chi_squared`; `degrees_of_freedom_for_signal`, a Hutchinson estimate (32 probes) on `np.random.default_rng`; `posterior_covariance_proxy`, a CG matvec. No averaging-kernel function: A appears only in docstrings | `amx.diagnose` |
| plumax `assimilation/solve.py` | `28f257e` | `run_lbfgs`, `run_gauss_newton`, `run_dual_psas`; `run_gauss_newton` is identical to radtrax's (empty diff) | deleted |
| plumax `les_fvm/fourdvar.py` | `28f257e` | `solve_4dvar` by L-BFGS in whitened space; `_gauss_newton_hessian` dense by `jacrev` (diffrax's `custom_vjp` rejects `jacfwd`); `posterior_covariance` by `gx.inv` and symmetrise, P\_S = L P\_χ Lᵀ; `laplace_sample` | `amx.curvature`; plumax keeps the 4D-Var cost |
| gaussx `_inference/_laplace.py` `laplace_mode` | `7fa1a38` | damped Newton loop, mode implicit through `jax.lax.custom_root`, tolerance floored at 64 eps; `newton_update` and `gauss_newton_precision` (a `LowRankUpdate`) are the single steps; Chandrupatla is already an optimistix root finder | `laplace_mode` to `amx.solve`, deprecated in gaussx without forwarding; the steps and Chandrupatla stay |
| pyrox-gp `_inference_nongauss.py` | `94eef13` | `LaplaceInference`, `GaussNewtonInference` (`precision_floor=1e-3`, damping) as eager Python loops with `break`; `QuasiNewtonInference` by `optax.lbfgs` in a Python loop; hand-written `cho_solve`. `PosteriorLinearization` and EP are not Newton methods and stay | the loops to `amx.solve` |
| pyrox-lgm `_inla.py` | `94eef13` | inner fit by `gx.laplace_mode`; outer `_theta_mode` by `optax.lbfgs` in a Python loop with host `float` checks; R-INLA's grid / CCD design stays | the inner fit (`amx.solve.laplace_mode`) and the outer search to `amx.solve` |
| pyrox `inference/_laplace.py` `LastLayerLaplace`, `last_layer_laplace` | `2ac68a5` (pyrox main, 24 Oct 2026; not at geostack's pin `94eef13`) | dense J by forward mode; GGN; diagonal or dense precision; `log_marginal_likelihood`; `optimize_prior_precision` on a 4097-point log λ grid, then golden section | `amx.curvature`, `amx.diagnose`, after pyrox#332 |

Two facts set the order of work. vardax's adjoints are markers rather than working optimistix adjoints, so wave 1 must implement `apply`, not just move the files. And every 3D-Var retrieval leaves JAX at its last line, so the solvers and diagnostics need to return arrays and use keys before a plume model can differentiate through them.
