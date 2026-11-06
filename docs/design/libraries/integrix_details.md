---
title: "integrix: design details"
short_title: "integrix details"
description: "The implementation reference behind the integrix design: core types, the protocols callers implement, the algorithms per operation, JAX notes, tests, dependencies and the survey of the code that moves in. Every integrix name below is proposed."
date: 2026-11-05
keywords:
  - integrix
  - design
  - details
  - GeoStack
---

The implementation reference behind the [integrix design](integrix.md): core types, the protocols callers implement, the algorithms per operation, JAX notes, tests, dependencies and the survey of the code that moves in. Every integrix name below is proposed.

## Core types

Six kinds of object: the rule (functax's), what a rule computes, the steppers diffrax lacks, the adaptive rules, the adjoints of a rollout, and the expectation estimators. Every type is an `eqx.Module`; configuration that picks a code path is a static field, arrays are leaves.

```python
# Shapes: N nodes; D dims of the domain; C state size; S samples

import abc

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray, PyTree


# proposed (functax): integrix takes rules, never builds them
class Rule(eqx.Module):
    nodes: Float[Array, "N D"]
    weights: Float[Array, "N"]


# proposed (integrix)
class Integral(eqx.Module):
    # I ≈ ∑ₙ wₙ f(xₙ), with an error estimate when the rule has one
    value: Float[Array, "*out"]
    error: Float[Array, "*out"] | None
    n_evaluations: Int[Array, ""]
    converged: Bool[Array, ""]


# proposed (integrix)
class AbstractStepper(eqx.Module):
    """A multistep or split scheme with history diffrax's protocol lacks."""

    order: eqx.AbstractVar[int]

    @abc.abstractmethod
    def init(self, rhs, t0, y0: PyTree, dt, args) -> tuple[PyTree, PyTree]:
        """Bootstrap the history; returns (state, y₁)."""

    @abc.abstractmethod
    def step(self, rhs, t, y: PyTree, dt, state: PyTree, args):
        """One step; returns (yₙ₊₁, state)."""


# proposed (integrix)
class GaussKronrod(eqx.Module):
    # error ≈ |K₂ₙ₊₁(f) − Gₙ(f)| on each subinterval
    order: int = eqx.field(static=True, default=15)
    rtol: float = 1e-8
    atol: float = 0.0
    max_intervals: int = eqx.field(static=True, default=64)


# proposed (integrix)
class Checkpointed(eqx.Module):
    # backward march storing every k-th state, recomputing between
    every: int = eqx.field(static=True, default=50)


# proposed (integrix)
class Truncated(eqx.Module):
    # gradient flows through the carry of the last k steps only
    k: int = eqx.field(static=True, default=1)


# proposed (integrix)
class Expectation(eqx.Module):
    # 𝔼[f] ≈ (1/S) ∑ₛ f(xₛ), with its standard error
    value: Float[Array, "*out"]
    standard_error: Float[Array, "*out"]
    n_samples: int = eqx.field(static=True)
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `Rule` (functax) | nodes and weights of a rule on a domain | weights finite; for a positive rule, all weights ≥ 0 |
| `Integral` | value, error estimate, evaluation count, convergence flag | `converged` is false whenever the error exceeds the tolerance; never raised under `jit` |
| `AbstractStepper` | the scheme's coefficients; history lives in the returned state, not on the module | `init` then `step`; the module itself is never mutated |
| `GaussKronrod`, `TanhSinh`, `Romberg` | the pair or ladder of rules and the tolerances | the embedded rule's nodes are a subset of the outer rule's, so the error costs no extra evaluations |
| `Checkpointed`, `Truncated`, `Recursive` | the adjoint strategy of a rollout | forward values identical to an undifferentiated run; only gradients differ |
| `Expectation` | the estimate and its standard error | standard error reported from the same samples; quasi-Monte Carlo reports a randomised-shift estimate |

## Types to adhere to

What a caller passes in. integrix owns none of these; it only asks that they keep their shape.

- **An integrand** is a pure function `f(x, args)` that takes the rule's nodes with a leading axis N and returns values with the same leading axis, or an array of values already at the nodes. It must trace: no Python branching on values.
- **A right-hand side for a march** is a diffrax term: `dfx.ODETerm(vector_field)` with `vector_field(t, y, args)`, `dfx.MultiTerm(explicit, implicit)` for IMEX, or a drift and a `ControlTerm` for an SDE. The multistep steppers take the same `vector_field`, so a model writes its tendencies once.
- **A measure** is a functax `Rule`, a weights array over the nodes (an SRF, cell areas), or for `expect` a sampler `sample(key, n) -> (S, D)` with an optional `log_prob` for importance weights.
- **An adjoint spec** may be a pipekit-cycle spec (`RecursiveCheckpointAdjoint`, `DirectAdjoint`, `BacksolveAdjoint`, `ImplicitAdjoint`, `TruncatedAdjoint`), matched structurally by class name as pipekit-jax does today, or an integrix adjoint directly.

```python
# Shapes: N nodes; W wavelengths; Z layers

# user
# f(x) at the rule's nodes: (N, D) → (N,)
def integrand(nodes, args): ...


# user
# dy/dt = F(t, y): the same function for diffrax and for the multistep loop
def tendencies(t, state, args): ...  # pytree → pytree
```

## Algorithms

### A rule: weights against samples

```{math}
I \approx \sum_{n=1}^{N} w_n f(x_n)
```

1. Evaluate f at the rule's nodes, or take the values given.
2. Contract the node axis with the weights (`einx.dot("... n, n -> ...", values, weights)`), over one axis or several at once for a tensor rule.
3. For a weighted mean (band radiance), divide by ∑ wₙ R(xₙ) computed with the same rule.

Cost: N evaluations and one contraction. Pitfalls: build the weights in the input's dtype; sum in pairwise order, not left to right, so float32 sums over long spectra do not drift; on non-uniform nodes the trapezoid end weights are half-intervals, as radtrax's and plumax's SRF code already does.

### A scan: the cumulative integral

```{math}
F_k = F_{k-1} + \frac{1}{2} (x_k - x_{k-1}) (f_{k-1} + f_k)
```

1. Form the per-interval increments from the rule.
2. Take their prefix sum with `jax.lax.associative_scan`, forward for depth from the surface, reversed for optical depth from the top.

Cost: O(N) work in O(log N) depth. Pitfalls: a long float32 prefix sum loses the small increments at the end; a reversed scan must reverse the increments, not the nodes.

### Explicit Runge–Kutta

With Butcher coefficients aᵢⱼ, bᵢ, cᵢ and step h:

```{math}
k_i = F\left(t_n + c_i h, y_n + h \sum_{j<i} a_{ij} k_j\right)
```

```{math}
y_{n+1} = y_n + h \sum_i b_i k_i
```

1. Subclass `diffrax.AbstractERK` with the tableau, as finitevolX's `RK2Heun`, `RK3SSP`, `SSP_RK2` already do.
2. Fixed-step schemes without an embedded pair declare no error estimate, so they run only under `ConstantStepSize`.

Cost: s evaluations of F per step for s stages. Pitfalls: the strong-stability-preserving schemes (SSP-RK3, SSP-RK(10,4)) keep total-variation bounds only within their CFL limit; `stable_step` from sesmax-core is a first-order estimate, not a certificate for them.

### IMEX

For y' = Fᴇ(y) + Fᵢ(y), with Fᵢ stiff, each implicit stage solves

```{math}
(I - h \gamma J_I) \Delta = r
```

with Jᵢ the Jacobian of Fᵢ. Following sesmax-core's `imex_solver`, the solve is a Newton root-find with a matrix-free GMRES inner solve, so Jᵢ is never formed; `IMEX_SSP2` and diffrax's `KenCarp3` both fit. Pitfalls: a fixed step cannot reject a failed Newton solve, so IMEX runs under an adaptive PID controller with tolerances matching the Newton ones.

### Multistep and filtered leapfrog

```{math}
y_{n+1} = y_n + \frac{h}{2} (3 f_n - f_{n-1})
```

```{math}
y_{n+1} = \bar y_{n-1} + 2 h f_n
```

```{math}
\bar y_n = y_n + \alpha (\bar y_{n-1} - 2 y_n + y_{n+1})
```

1. Bootstrap the history with one Runge–Kutta step (`init`).
2. Each `step` returns the new state and the updated history; the stepper module is never mutated.
3. Run the loop under `jax.lax.scan`, with the same adjoint options as a diffrax solve.

Pitfalls: leapfrog's computational mode grows without the Robert–Asselin filter (α ≈ 0.01 to 0.1); a variable step breaks the Adams–Bashforth coefficients, so these schemes are fixed-step only.

### Split-explicit

The fast barotropic mode is advanced in M substeps of h / M inside each baroclinic step and averaged back, as finitevolX's `split_explicit_step` does. Pitfalls: the averaging filter sets the damping of the fast mode; M follows the external-wave CFL, not the internal one.

### Adaptive quadrature

On each subinterval, an embedded pair estimates the error:

```{math}
E \approx |K_{2n+1}(f) - G_n(f)|
```

1. Start from one interval; evaluate the Gauss–Kronrod pair.
2. Bisect the interval with the largest error until the total meets the tolerance or the buffer of `max_intervals` is full.
3. Return the sum, the error, the count and `converged`.

Under `jit` the buffer has a static size and the loop is a `jax.lax.while_loop`. Pitfalls: endpoint singularities need tanh–sinh, not Gauss–Kronrod; reverse-mode gradients through a `while_loop` are not defined, so the gradient is taken by differentiating the final rule on the final partition, treated as fixed (a custom JVP).

### Adjoints of a rollout

1. **Checkpointed:** store every k-th state on the forward run and recompute between checkpoints on the backward run; memory O(T / k) plus O(k), cost about one extra forward pass. diffrax's `RecursiveCheckpointAdjoint` is the default for a single solve.
2. **Truncated:** wrap the carry of every step before T − k in `stop_gradient`, as pipekit-jax's `truncated_scan` does; memory O(k), and for chaotic dynamics a regulariser, since the exact adjoint grows like exp(λₘₐₓ T).
3. **Backsolve** (continuous adjoint): offered, never a default; it diverges for chaotic or stiff dynamics.

### Expectations

With samples xₛ drawn from p:

```{math}
\hat I = \frac{1}{S} \sum_{s=1}^{S} f(x_s)
```

1. Plain Monte Carlo with the standard error σ̂ / √S.
2. Quasi-Monte Carlo on Sobol or Halton points with a random shift, so a standard error exists.
3. Self-normalised importance sampling with weights wₛ = p(xₛ) / q(xₛ), reporting the effective sample size (∑ wₛ)² / ∑ wₛ².

Pitfalls: an unshifted Sobol sequence has no error estimate; heavy-tailed weights make the importance estimate unreliable long before its variance looks large.

## JAX compatibility

| Function | jit | vmap | grad | Notes |
| --- | --- | --- | --- | --- |
| `rule.integrate`, `rule.weighted_mean` | yes | yes | yes | a contraction; gradients flow to values and weights |
| `scan.cumulative` | yes | yes | yes | `associative_scan`; reverse mode is a reversed prefix sum |
| `march` steppers under `diffeqsolve` | yes | yes | yes | diffrax's adjoints, or integrix's |
| `march` multistep loop | yes | yes | yes | `lax.scan`; the stepper is a pytree, its history in the carry |
| `adapt.gauss_kronrod` | yes | yes | forward and reverse via a custom JVP | the partition is not differentiated; the final rule is |
| `adjoint.Checkpointed`, `adjoint.Truncated` | yes | yes | yes, by construction | forward values identical to a plain run |
| `expect.monte_carlo`, `expect.qmc` | yes | yes | yes, by reparametrisation | the sampler must be reparametrisable for gradients in its parameters |

- **Static versus traced.** Stage counts, history lengths, `max_intervals`, checkpoint spacing and the number of samples are static fields; tolerances, weights, nodes and step sizes are traced, so they can be differentiated.
- **Custom derivatives.** Adaptive quadrature defines a custom JVP over its final rule; the multistep loop needs none, since `lax.scan` differentiates; truncated rollouts are a `stop_gradient` pattern, not a custom rule.
- **Dtypes.** Weights and coefficients are built in the dtype of the values; tests run in float64, and every rule is also checked in float32 for drift over long sums.
- **Keys.** Monte Carlo and quasi-Monte Carlo take a `key` keyword, last, and split it once per estimator; nothing seeds globally.

## Testing

- **Oracles.** Every rule on polynomials it must integrate exactly (degree 2N − 1 for Gauss–Legendre); ∫₀^π sin x dx = 2 and ∫ eˣ for the trapezoid's O(h²) convergence; `scipy.integrate.quad`, `cumulative_trapezoid` and `solve_ivp` as references; quadax's results for the adaptive rules, as a reference only.
- **Order of accuracy.** Each stepper on y' = λy and on a linear advection problem, with the error's slope against h matching its order; SSP schemes checked for total-variation bounds within their CFL limit.
- **Moved code.** finitevolX's existing tests for the ten steppers run unchanged against the integrix classes before the aliases go.
- **Gradients.** `jacrev` against central differences for a rule, a scan, a fixed-step march and an adaptive integral; a truncated rollout's gradient equal to the exact one when k ≥ T.
- **Transforms.** `jit`, `vmap` and a Python loop agree for every public function.
- **Statistics.** Monte Carlo estimates bounded by their own standard error (a number of standard errors, said in the test), not a tolerance tuned on one draw.

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| JAX, Equinox, jaxtyping | arrays, modules, shapes | required |
| diffrax | solver and adjoint protocols the steppers extend | required |
| lineax, optimistix | matrix-free Newton and GMRES for implicit stages | required |
| einx | contractions | required |
| functax | rules | required (Core tier 0) |
| gaussx | structured solves in implicit stages | required (Core tier 0) |
| SciPy, quadax | reference results in tests | test only |

## Where it stands

Code read on 5 Nov 2026. Each entry gives the commit, the names, what they do, how they differ from the proposal and what moves.

### finitevolX `82090a3` (0.1.0)

`_src/timestepping/diffrax_solvers.py` holds thirteen classes:

| Class | Base | What it is | Moves |
| --- | --- | --- | --- |
| `ForwardEulerDfx` | `dfx.AbstractSolver` | forward Euler | yes |
| `RK2Heun`, `RK3SSP`, `SSP_RK2` | `dfx.AbstractERK` | explicit Runge–Kutta by tableau | yes |
| `RK4Classic`, `SSP_RK104` | `_FixedStepERK` (an `AbstractERK` with no error estimate) | fixed-step Runge–Kutta | yes |
| `IMEX_SSP2` | `dfx.AbstractRungeKutta`, `dfx.AbstractImplicitSolver` | IMEX SSP2 | yes |
| `AB3Solver` | `dfx.AbstractSolver` | Adams–Bashforth 3, history in the solver state | yes |
| `AB2Solver`, `LeapfrogRAFSolver`, `SplitExplicitRKSolver` | `eqx.Module`, with `init` and `step` | multistep and split schemes outside `diffeqsolve` | yes, as `AbstractStepper`s |
| `SemiLagrangianSolver`, `SemiLagrangianTerm` | `dfx.AbstractSolver`, `dfx.AbstractTerm` | semi-Lagrangian advection | no: a field scheme |

The pure step functions beside them move with the classes: `euler_step`, `heun_step`, `rk3_ssp_step`, `rk4_step` (`explicit_rk.py`), `imex_ssp2_step` (`imex.py`), `ab2_step`, `ab3_step`, `leapfrog_raf_step` (`multistep.py`) and `split_explicit_step` (`split_explicit.py`). `semi_lagrangian_step` stays. `solve_ocean_pde` (`_solve.py`) wraps `diffeqsolve` with a boundary-condition hook, so it stays with finitevolX's boundary conditions and calls integrix. Difference from the Design tab: `AB3Solver` is new since geostack's pin, so eleven classes move, not ten (eight diffrax solvers, three steppers).

### pipekit-jax, pipekit `f2b3522`

`packages/pipekit-jax/src/pipekit_jax/adjoints.py`: `truncated_scan` (a `lax.scan` whose carry runs under `stop_gradient` before the last k steps) and `to_diffrax_adjoint` (maps pipekit-cycle's adjoint specs onto diffrax adjoints by class name, refusing `TruncatedAdjoint`). Both interpreters move. The specs themselves (`pipekit_cycle.adjoints`) stay in pipekit: they are configuration, and integrix accepts them structurally as pipekit-jax does.

### sesmax-core, sesmax `314de57`

`_src/solvers.py`: `imex_solver` builds diffrax's `KenCarp3` with an optimistix Newton and a lineax GMRES inner solve, so the implicit stage is matrix-free; `imex_stepsize_controller` returns `dfx.PIDController(rtol, atol)`; `stable_step` gives the forward-Euler CFL bound from spacings, speeds and diffusivities. All three are generic and move to `integrix.march`; the Design tab names somax for the controller, which sesmax-core has since taken over.

### plumax `28f257e`, radtrax `ed04971`, xrtoolz `b3855d7`

The hand-written quadrature is all trapezoidal; nothing in the stack uses Simpson's rule.

| Library | Where | What | Note |
| --- | --- | --- | --- |
| plumax | `gauss_plume/plume.py`, `gauss_puff/puff.py` | column totals by `np.trapezoid` over height | NumPy, so not traceable |
| plumax | `coupled/forward.py`, `population/fitting.py` | `jnp.trapezoid` over height and over a parameter | traceable |
| plumax | `population/paradox.py` | `np.trapezoid` against a Gaussian kernel | NumPy |
| plumax, radtrax | `radtran/srf.py`, `instrument/srf.py` | trapezoid weights for SRF band integration, half-intervals at the ends | the same code, ported from plumax to radtrax |
| radtrax | `geo/srf.py` | projected weights divided by trapezoid bin widths |  |
| xrops | `atm/_src/vertical.py` (`column_integral`), `transforms/_src/fourier.py` | trapezoid or sum along a coordinate; spectral moments by `DataArray.integrate` | xarray; xrops keeps the wrappers |

Each becomes an `ix.rule` call with a functax rule; the NumPy ones become traceable as a side effect.
