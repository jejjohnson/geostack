---
title: pipekit-cycle
subtitle: Report 10 · Time-stepping, data assimilation and observation operators, as shipped
short_title: 10 · pipekit-cycle
description: pipekit-cycle, the stack's cycles for forward simulation, data assimilation and iterative inference. The scoping report is checked against pipekit-cycle 0.0.2, which takes precedence; every stage of the running example is a real run, and the few things the code is missing become small proposals.
keywords:
  - pipekit-cycle
  - data assimilation
  - Kalman filter
  - EnKF
  - Recurrence
  - CarryState
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

pipekit-cycle turns a one-step transition into a run over time. A `Cycle`
repeats a step, a `DACycle` alternates forecast and analysis, and a
`Recurrence` iterates until a condition holds. Three protocols
(`ForwardModel`, `ObservationOperator`, `AnalysisStep`) let algorithm
libraries plug in without importing it. The scoping report described this
package before it was built. Version 0.0.2 ships everything the report
named and more: eleven further protocols, a back-and-forth nudging cycle and
adjoint specs that pipekit-jax reads. Where the two disagree, **the code
holds**. A Kalman filter written as a `DACycle` matches a hand-written loop
exactly, and an EnKF converges to it at the $1/\sqrt{N_e}$ rate. Two real
gaps remain: a stateful `Sequential` cannot be a cycle's step, and the
observation index restarts on every call, which silently feeds a restarted
cycle the wrong observations. This page proposes two small fixes.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Built. Originally a scoping proposal ("committed"); pipekit-cycle 0.0.2 is the design of record, and this page follows it |
| **Reading time** | about 25 minutes |
| **Audience** | anyone writing a forward simulation, a DA cycle or an iterative retrieval with the stack, or an adapter for an algorithm library |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit core, where `StatefulOperator` and `CarryState` live); [Report 9](09_use_cases.md) (use cases); [Report 13](13_state_catalog.md) (indexing the states a cycle produces); [Report 11](11_pipekit_train.md) (training the emulators of stage 4); [Report 12](#r10-ref-report-11) (the registry, forthcoming) |
| **Code checked against** | `pipekit-cycle` 0.0.2 and `pipekit` 0.0.2 at `ef3c17f`, the revision pinned in `references/pipekit`; its 100 tests pass. Read on 13 October 2026 |
| **Running example** | every output below is from one run on that revision of `notebooks/reports/report10_pipekit_cycle.py`, which seeds every draw and asserts each number the page shows |
:::

:::{important} The code takes precedence
The scoping report and pipekit-cycle 0.0.2 differ in a dozen places,
mostly in details the report guessed before the code existed: return
values, carrier shapes, the fields of the carry states, the extras. Each
difference has a **Differs** box below and the code's behaviour is what this
page documents. Only behaviour that is missing *and* needed by a real
workflow becomes a proposal ([](#r10-ws-a), [](#r10-ws-b)); the rest goes to
[](#r10-open-questions) or [](#r10-todos).
:::

(r10-summary)=
## Summary

**Decision.** pipekit-cycle 0.0.2 stays as it is: a separate package beside
pipekit-array, depending only on pipekit, with no JAX and no domain code.
Its protocols, cycles and helpers are the contract algorithm libraries
target. Two additions close the gaps the running example hits, and one more
is already proposed in Report 13.

- **A. Stateful pipelines as cycle steps.** `pipekit.Sequential` threads a
  carry state when it holds a stateful step, but it is not a
  `StatefulOperator`, so `Cycle` and its siblings reject it with a
  `TypeError`. Accept any `Sequential` that is stateful as a step.
- **B. Observation inputs on the cycle's clock.** `obs_source(i)` receives a
  call-local index that restarts at 0 on every call, and R comes only from
  the state. Let the index follow `state.cycle_count` (B1), and let a source
  set R per step (B2, the overview's `obs_err_source`).
- **Report 13 D1. Ensemble histories.** `EnsembleCycle` and `EnsembleDACycle`
  keep no trajectory; [Report 13](13_state_catalog.md#r13-ws-d) adds
  `save_history` to both, so cycle outputs can be written and indexed.

**End state.** A script composes a step from stateless and stateful
operators, cycles it, and restarts it from the last state without
re-reading old observations. Algorithm libraries supply forward models,
observation operators and analysis steps by structural typing. Every cycle
can keep its history, which Report 13's state profile indexes.

(r10-ownership)=
## Ownership rules

pipekit-cycle owns the shape of a run over time; the carrier, the physics
and the algorithms belong to whoever supplies the steps.

:::{figure} ../images/reports/r10_ownership.svg
:label: fig-r10-ownership
:alt: Scripts build cycles from pipekit-cycle; algorithm libraries satisfy its protocols by structural typing. pipekit-cycle groups cycles, DA cycles, protocols, observation and forward helpers, carry states and adjoint specs. It sits on pipekit core; pipekit-jax reads its adjoint specs.

pipekit-cycle 0.0.2 between pipekit core and the code that uses it. Dashed
arrows are structural: nothing is imported.
:::

1. **pipekit core** owns `StatefulOperator`, `CarryState` and `Sequential`'s
   state threading. It never imports pipekit-cycle.
2. **pipekit-cycle** owns the cycles, the DA cycles, the protocols, the
   generic observation and forward helpers, the carry states and the adjoint
   specs. It imports only pipekit, and never an array library, JAX or a
   domain package.
3. **Algorithm libraries** (filterax, vardax, plumax) own forward models,
   observation operators and analysis steps. They satisfy the protocols by
   structural typing and never need to import pipekit-cycle.
4. **pipekit-jax** owns everything JAX: it interprets the adjoint specs
   (`to_diffrax_adjoint`, `truncated_scan`). pipekit-cycle never imports it.
5. **Scripts** own the carrier, the initial state, the observations and
   every file a run writes ([Report 13](13_state_catalog.md#r13-ws-d)).

:::{table} The dependency tiers around pipekit-cycle.
:label: tbl-r10-tiers

| Tier | Packages | May import | Role |
|---|---|---|---|
| Entry points | scripts, notebooks | everything below | build and run cycles |
| Algorithms | filterax, vardax, plumax adapters | their own numerics; nothing from pipekit-cycle is required | forward models, observation operators, analysis steps |
| Cycles | pipekit-cycle | pipekit | runs over time and iterations |
| Framework | pipekit; pipekit-jax beside it | pipekit-jax: pipekit, JAX | operators, state; JAX execution |
:::

:::{warning} Differs from the plan
The scoping report listed three extras: `array`, `jax` and `catalog` (the
last for `statecatalog`). pipekit-cycle 0.0.2 has one, `array =
["pipekit-array"]`, and depends only on `pipekit`. There is no `jax` extra:
JAX execution lives in pipekit-jax, which reads pipekit-cycle's adjoint
specs instead of pipekit-cycle importing it. There is no `catalog` extra:
[Report 13](13_state_catalog.md) replaces `statecatalog` with a state
profile in geocatalog that scripts, not cycles, write to.
:::

(r10-surface)=
## What ships

:::{table} The public surface of pipekit-cycle 0.0.2, grouped. Rows in italics go beyond the scoping report.
:label: tbl-r10-surface

| Group | Names | Module |
|---|---|---|
| Cycles | `Cycle`, `EnsembleCycle`, `WindowedCycle`, `Recurrence` | `cycle` |
| DA cycles | `DACycle`, `EnsembleDACycle`, `SmootherCycle` | `da` |
| *Nudging* | *`BFNCycle`: back-and-forth nudging over a window, adjoint-free* | *`bfn`* |
| Core protocols | `ForwardModel`, `ObservationOperator`, `AnalysisStep` | `protocols` |
| *Cross-library protocols* | *`ReducedBasis`, `TangentLinearModel`, `ErrorSubspace`, `ReducedOrderModel`, `LatentMap`, `ObservationNoise`, `CovarianceLocalizer`, `EnsembleInflator`, `StepScheduler`, `IterativeProcess`, `SupportsAdjoint`* | *`protocols`* |
| Observation helpers | `IdentityObs`, `LinearObs`, `CallableObs`, `CompositeObs` | `obs` |
| Forward helpers | `CallableForward`, `CompositeForward`, `NeuralForward` | `forward` |
| Carry states | `DAState`, `IterationState`, `WindowState` | `state` |
| *Adjoint specs* | *`AdjointSpec`, `DirectAdjoint`, `RecursiveCheckpointAdjoint`, `TruncatedAdjoint`, `ImplicitAdjoint`, `BacksolveAdjoint`: frozen dataclasses with no behaviour here, interpreted by pipekit-jax* | *`adjoints`* |
:::

The eleven further protocols are the shared vocabulary for the algorithm
libraries: a tangent-linear model and its adjoint (M′, M\*), a low-rank error
subspace B ≈ LLᵀ, a reduced-order model with `encode`, `decode` and `step`,
covariance localisers, ensemble inflators, adaptive step schedulers, and
ensemble Kalman inversion as an `IterativeProcess`. None is used by a cycle
in pipekit-cycle; they exist so that, say, filterax's inflator and vardax's
error subspace agree on one signature.

(r10-running)=
## The running example: a methane plume advected along a transect

**The problem.** A methane enhancement $x \in \mathbb{R}^N$, in ppb above
background, sits on a periodic transect of $N = 40$ cells. Wind carries it
downstream one hour at a time. A column sensor sees every fourth cell, so
there are $p = 10$ observations per hour, for $T = 10$ hours. The plume
starts as a Gaussian of 50 ppb peak at cell 8.

**The model.** Let $M \in \mathbb{R}^{N \times N}$ be first-order upwind
advection with Courant number 0.4,
$M = 0.6 I + 0.4 S$ with $S$ the one-cell downstream shift; $Q = 0.01 I$ the
model-error covariance; $H \in \{0, 1\}^{p \times N}$ the selection of every
fourth cell; and $R = 0.1 I$ the observation-error covariance. The truth and
the observations are

$$
\begin{aligned}
x_t &= M x_{t-1} + \eta_t , & \eta_t &\sim \mathcal{N}(0, Q) , \\
y_t &= H x_t + \varepsilon_t , & \varepsilon_t &\sim \mathcal{N}(0, R) .
\end{aligned}
$$ (r10-eq-model)

The filter starts from $m_0 = 0$ with $P_0 = 100 I$: it knows nothing about
the plume. The data are synthetic, so every stage has an exact answer to
check against.

(r10-stage-1)=
### Stage 1 · Forecast only (pipekit-cycle)

**TL;DR.** Push the plume downstream for six hours and keep every second
state.

**Problem.** With no observations the forecast is
$x_{t} = M x_{t-1}$, repeated $n$ times; a history with stride $s$ keeps the
states after steps $1, 1 + s, 1 + 2s, \ldots$

```{code-block} python
:caption: A forecast is a Cycle over a ForwardModel; the history lives on the operator.
:linenos:

# Shapes: N = 40 cells
import pipekit_cycle as pc

class Advect:                                  # satisfies pc.ForwardModel    user
    dt = 1.0
    state_signature = None
    # x_t = M x_{t−1}
    def step(self, x, dt):
        return M @ x                           # (N,) → (N,)

fc = pc.Cycle(Advect(), n_steps=6, save_history=True, history_stride=2)  # today
x6, state = fc(x_true_0, pc.DAState())         # (N,) → (N,)                 today
len(fc.history)                                # → 3: after steps 1, 3, 5    today
[int(np.argmax(c)) for c, _ in fc.history]     # → [9, 10, 10]: the peak moves
state.t                                        # → 0.0: Cycle keeps no clock
```

A plain `Cycle` wraps the `ForwardModel` in an internal adapter and threads
the state through untouched, so `DAState.t` stays at 0; only the DA cycles
advance it.

:::{warning} Differs from the plan
The scoping report said a `Cycle` returns
`(final_carrier, final_state, history)`. It returns
`(final_carrier, final_state)`, and the history is the list
`Cycle.history` of `(carrier, state)` pairs, reset on every call. The last
step is kept only when $(n - 1)$ is a multiple of the stride.
:::

(r10-stage-2)=
### Stage 2 · A Kalman filter as a DACycle (pipekit-cycle)

**TL;DR.** Assimilate the ten hours of observations with an exact Kalman
filter, and check that the cycle changes nothing about the arithmetic.

**Problem.** The Kalman filter carries a mean $m_t$ and a covariance $P_t$:

$$
\begin{aligned}
m_t^b &= M m_{t-1} , & P_t^b &= M P_{t-1} M^\top + Q , \\
K_t &= P_t^b H^\top \left( H P_t^b H^\top + R \right)^{-1} , \\
m_t &= m_t^b + K_t \left( y_t - H m_t^b \right) , & P_t &= \left( I - K_t H \right) P_t^b .
\end{aligned}
$$ (r10-eq-kf)

```{code-block} python
:caption: The Kalman filter as a DACycle, against a hand-written loop over {eq}`r10-eq-kf`.
:linenos:

# Shapes: N = 40, p = 10, T = 10; the carrier is (m, P): (N,), (N, N)
class KFForward:                               # pc.ForwardModel              user
    dt = 1.0
    state_signature = None
    # m_b = M m;  P_b = M P Mᵀ + Q
    def step(self, carrier, dt):
        m, P = carrier
        return M @ m, M @ P @ M.T + Q          # ((N,), (N, N)) → same

class KalmanAnalysis:                          # pc.AnalysisStep              user
    # K = P_b Hᵀ (H P_b Hᵀ + R)⁻¹
    # m = m_b + K (y − H m_b);  P = (I − K H) P_b
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        m, P = forecast
        Hm = obs_op.linearize(m).H                                   # (p, N)
        K = P @ Hm.T @ np.linalg.inv(Hm @ P @ Hm.T + obs_err_cov)  # (N, p)
        return m + K @ (obs - obs_op(m)), (np.eye(N) - K @ Hm) @ P

da = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                obs_source=pk.Lambda(lambda i: y[i]), n_steps=T)    # today
(m, P), state = da((m0, P0), pc.DAState(obs_err_cov=R))           # today
# → max |m − m_hand| = 0.0, max |P − P_hand| = 0.0
# → RMSE against the truth 0.125 ppb (prior RMSE 17.1 ppb)
state.t, state.cycle_count, len(da.history)   # → 10.0, 10, 10       today
```

Each cycle steps the forward model, calls `obs_source(i)` for the step's
observations, and hands both to the analysis with `obs_err_cov` read from
the state. The result is bit-for-bit the hand-written filter. Two
conventions follow from the code and are worth stating once:

- **The covariance travels in the carrier.** `AnalysisStep.__call__`
  receives the forecast, the observations, the observation operator and R,
  but never the state. Anything the analysis updates, such as $P$ or an
  ensemble, must be part of the carrier; here it is a `(m, P)` tuple.
- **`ObservationOperator` requires `linearize`.** The protocol check fails
  for a class with only `__call__`. `LinearObs` and `IdentityObs` return
  themselves; `CallableObs` takes an optional `linearize_fn` and raises
  `NotImplementedError` without it.

:::{warning} Differs from the plan
The scoping report's `DAState` held `background_state` and
`ensemble_members` alongside `t` and `cycle_count`. The shipped one holds
`t`, `cycle_count`, `obs_err_cov` and a free-form `extras` dict. The
background and the ensemble are the carrier, which is why the analysis never
needs the state. `IterationState` is `iter_count`, `residual_norm`, `extras`
(no `lr`, no `history`, and `iter_count` rather than `iter`), and
`WindowState` is `window_index`, `last_analysis`, `extras`; no cycle uses
`WindowState` yet.
:::

(r10-stage-3)=
### Stage 3 · The same problem as a stochastic EnKF (pipekit-cycle)

**TL;DR.** Replace the exact covariance with an ensemble and check that the
ensemble mean approaches the Kalman mean as the ensemble grows.

**Problem.** With $N_e$ members $x^{(i)}$, the forecast covariance is the
sample covariance of the anomalies, and every member is updated against its
own perturbed observations, as in the overview ({eq}`eq-enkf`). For a
linear-Gaussian model the EnKF mean converges to the Kalman mean $m_t$, and
its error falls as $N_e^{-1/2}$ once the ensemble is large enough that the
gain is well estimated.

```{code-block} python
:caption: An EnKF as an EnsembleDACycle; the carrier is a list of members.
:linenos:

# Shapes: N_e members, each (N,); the carrier is a list of N_e arrays
class EnKF:                                    # pc.AnalysisStep              user
    # K = P_b Hᵀ (H P_b Hᵀ + R)⁻¹;  x⁽ⁱ⁾ ← x⁽ⁱ⁾ + K (y + ε⁽ⁱ⁾ − H x⁽ⁱ⁾)
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov): ...   # list → list

eda = pc.EnsembleDACycle(Member(), pc.LinearObs(H), EnKF(),
                         obs_source=obs_source, n_steps=T, n_members=N_e)  # today
members, state = eda(members_0, pc.DAState(obs_err_cov=R))   # N_e × (N,)     today
# RMS |mean(members) − m| , median over 10 seeds; KF posterior std 0.193 ppb
# → N_e =  50: 0.367   N_e = 200: 0.063   N_e = 800: 0.027
```

From 200 to 800 members the gap shrinks by a factor of 2.3, close to the
factor of 2 that $N_e^{-1/2}$ predicts. At 50 members it is six times larger
than at 200: with no localisation, 50 members estimate a 40 × 40 covariance
poorly, and the gain carries that error. Inflation and localisation are the
analysis step's business; pipekit-cycle names their signatures
(`EnsembleInflator`, `CovarianceLocalizer`) but does not apply them.

:::{warning} Differs from the plan
The scoping report described the ensemble carrier as "a stacked ensemble
(leading axis = ensemble member)". The shipped `EnsembleCycle` and
`EnsembleDACycle` take a **list** of members, length-checked against
`n_members`, and step each member separately; one carry state is threaded
through all members in turn. Vectorising over members is the user's
choice (a batched or vmapped step). Neither ensemble cycle keeps a history,
which Report 13's D1 adds.
:::

(r10-stage-4)=
### Stage 4 · Swap the forward model (pipekit-cycle)

**TL;DR.** Replace the physical model with an emulator and leave the cycle
unchanged.

**Problem.** The cycle only needs something that satisfies `ForwardModel`.
An emulator $\hat{M}$ trained to reproduce $M$ should therefore drop in with
no other change, and an exact emulator must give the same analysis.

```{code-block} python
:caption: The forward model swapped twice: a callable and a pipekit Operator.
:linenos:

# Shapes: carrier (m, P)
emulator = pc.CallableForward(lambda carrier, dt: step_fn(carrier), dt=1.0)  # today
neural = pc.NeuralForward(model_op=pk.Lambda(step_fn), dt=1.0)               # today
for fwd in (emulator, neural):
    da = pc.DACycle(fwd, pc.LinearObs(H), KalmanAnalysis(),
                    obs_source=obs_source, n_steps=T)                       # today
    (m_swap, _), _ = da((m0, P0), pc.DAState(obs_err_cov=R))
# → max |m_swap − m| = 0.0 for both
```

`NeuralForward(model_op, dt, state_signature=None)` is a thin, working
wrapper, not a stub: `step` calls `model_op(state)` and ignores `dt`, so any
pipekit `Operator` (a registry-loaded model, a `Sequential`) becomes a
forward model.

:::{warning} Differs from the plan
`CompositeForward` has **no `dt` argument**: its `dt` is the sum of its
components' `dt`, and its `step` ignores the `dt` it is passed, stepping
each component by its own. The scoping report said it "enforces consistent
total dt across components"; the code adds them instead. `BFNCycle` rejects
`CompositeForward` and `NeuralForward` unless a `backward_model` is given,
because it needs to step backwards in time.
:::

(r10-stage-5)=
### Stage 5 · A Gauss–Newton retrieval as a Recurrence (pipekit-cycle)

**TL;DR.** Retrieve a methane column enhancement from five-band
transmittances by iterating until the update is negligible.

**Problem.** Beer–Lambert absorption gives transmittances
$f_b(\alpha) = \exp(-k_b \alpha)$ in bands $b = 1, \ldots, 5$ with
absorption coefficients $k_b$, for a column enhancement $\alpha$. With
measured $y_b$ and Jacobian $J_b = -k_b \exp(-k_b \alpha)$, Gauss–Newton
iterates

$$
\alpha_{j+1} = \alpha_j + \frac{J^\top \left( y - f(\alpha_j) \right)}{J^\top J} ,
$$ (r10-eq-gn)

and stops when $|\alpha_{j+1} - \alpha_j| < 10^{-8}$.

```{code-block} python
:caption: An iterative retrieval is a Recurrence with a stateful step and a condition operator.
:linenos:

# Shapes: α scalar; y, f(α), J: (5,)
f = pc.CallableObs(lambda a: np.exp(-k * a),
                   linearize_fn=lambda a: -k * np.exp(-k * a))       # today

class GaussNewton(pk.StatefulOperator):                               # user
    # α ← α + Jᵀ (y − f(α)) / (Jᵀ J)
    def _apply(self, a, s):
        J, r = f.linearize(a), y_b - f(a)                     # (5,), (5,)
        step = float(J @ r / (J @ J))
        return a + step, pc.IterationState(iter_count=s.iter_count + 1,
                                           residual_norm=abs(step))

gn = pc.Recurrence(GaussNewton(),
                   condition_op=pk.Lambda(lambda s: s.residual_norm < 1e-8),
                   max_iters=50)                                      # today
a_hat, s = gn(0.0, pc.IterationState())                               # today
# → α̂ = 120.014 (truth 120), gn.iters = 6, gn.converged = True
# → Jᵀ (y − f(α̂)) = 9.9e-20: the normal equations hold at α̂
```

`Recurrence(step_op, condition_op, max_iters=1000)` calls `condition_op` on
the state after each step and stops on a truthy result; it records
`iters` and `converged` on the operator. The condition must be an
`Operator`, so a bare function is wrapped in `pk.Lambda`.

:::{warning} Differs from the plan
The scoping report's retrieval used
`condition_op=pk.qc.AssertCallable(predicate=…)`. `AssertCallable` is a QC
pass-through that **raises `QCError` when the predicate is false**, so that
recurrence stops with an exception on its first iteration. A condition is
a `pk.Lambda` that returns a boolean.
:::

(r10-stage-6)=
### Stage 6 · Restart the filter, and compose a step (the two gaps)

**TL;DR.** Run the ten hours as two five-hour calls, as an operational
cycle restarted from its last state would, and build a cycle step from a
`Sequential`.

**Problem.** Restarting a filter from its last carrier and state must give
the same analysis as one uninterrupted run, provided the second call reads
observations $y_5, \ldots, y_9$. A step that mixes a stateless operator and a
stateful one is a `Sequential`, which pipekit core threads state through.

```{code-block} python
:caption: Two gaps in 0.0.2: the observation index restarts, and a stateful Sequential is not a step.
:linenos:

# Shapes: as stage 2
da5 = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                 obs_source=obs_source, n_steps=5)                   # today
c, s = da5((m0, P0), pc.DAState(obs_err_cov=R))                     # hours 0–4
(m_restart, _), s = da5(c, s)                                       # hours 5–9?
# indices passed to obs_source → [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
# → max |m_restart − m| = 15.4 ppb, although s.cycle_count = 10

step = pk.Sequential([pk.Lambda(lambda x: M @ x), Count()])         # today
step(x, pc.DAState())          # → threads the state: cycle_count 1  today
pc.Cycle(step, n_steps=3)
# → TypeError: step_op must be a StatefulOperator or a ForwardModel, got Sequential.
```

The restart silently re-assimilates the first five hours of observations
and misplaces the plume by up to 15 ppb, with no warning: the state's
`cycle_count` is 10, but `obs_source` sees the call-local index. `BFNCycle`'s
docstring already tells users to "supply a stateful or clock-aware
`obs_source`" for a global clock. Workstream B makes the clock part of the
cycle. The `Sequential` case is workstream A.

(r10-names)=
## The scoping report's names, checked

:::{table} The stack names the scoping report's examples use, at the pinned revisions.
:label: tbl-r10-names

| Name in the report | At the pinned revisions |
|---|---|
| `pc.Cycle`, `pc.EnsembleCycle`, `pc.WindowedCycle`, `pc.Recurrence`, `pc.DACycle`, `pc.EnsembleDACycle`, `pc.SmootherCycle` | exist; signatures in [](#r10-audit) |
| `pc.NeuralForward(model_op=..., dt=...)` | exists, as a working wrapper |
| `pc.Checkpoint(cycle_op)` | does not exist; pipekit-train has a training `Checkpoint` callback, and `CarryState.to_dict` serialises a state |
| `pk.config_hash(da)` | does not exist; `pipekit.hashing.sha256_hex(stable_json(da.get_config()))`, or [Report 9](09_use_cases.md#r9-proposals)'s proposed `pipeline_hash` |
| `pk.qc.AssertCallable` as `condition_op` | exists, but raises instead of returning a boolean (stage 5) |
| `gc.CatalogTimeQuery`, `gz.tropomi.LoadL2`, `gz.qa.AssertValidFraction` | do not exist at the pinned revisions |
| `StateCatalog`, `state_cat.write(history, ...)` | not built; [Report 13](13_state_catalog.md) replaces them with a state profile in geocatalog |
| `pe.S3ModelRegistry`, `registry.load(name)` | `S3ModelRegistry` and `load(ref)` exist in pipekit-experiment, usually imported as `px`; `pe` is pipekit-evaluate elsewhere in these reports |
| `filterx.adapters.pipekit.EnKFAnalysis`, `plumax.adapters.pipekit.ChemistryForward` | not checked: those libraries are not pinned here. The library is filterax in the overview; pipekit-cycle's README quickstart still says `filterx` |
:::

(r10-audit)=
## Current-state audit

Two of the scoping report's pieces need action; the rest ship and work. The
audit read `references/pipekit` at `ef3c17f` on 13 October 2026.

:::{table} pipekit-cycle 0.0.2, piece by piece.
:label: tbl-r10-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `Cycle(step_op, n_steps, save_history=False, history_stride=1)` | `cycle.py:71` | `SimulationDataset` (pipekit-train), `Sequential` | **Healthy.** Returns `(carrier, state)`; history on `.history` |
| `EnsembleCycle(step_op, n_steps, n_members)` | `cycle.py:139` | | **Healthy.** List carrier, shared state; no history (Report 13 D1) |
| `WindowedCycle(step_op, window, stride=1)` | `cycle.py:202` | | **Healthy.** Input is a list of carriers; returns the window-final carriers |
| `Recurrence(step_op, condition_op, max_iters=1000)` | `cycle.py:265` | | **Healthy.** Records `iters`, `converged` |
| `step_op` coercion, `_coerce_step` | `cycle.py:50` | every cycle above | **Split ownership.** Accepts a `StatefulOperator` or a `ForwardModel`; rejects a stateful `Sequential` that pipekit core threads state through |
| `DACycle(forward_model, obs_op, analysis_step, obs_source=None, n_steps=1, save_history=True)` | `da.py:84` | the overview's stage 4 | **Healthy.** Exact against a hand-written Kalman filter |
| `EnsembleDACycle(..., n_steps=1, *, n_members)` | `da.py:184` | | **Healthy.** No `key`, no inflation argument, no history |
| `obs_source(i)` and `obs_err_cov` from the state | `da.py`, `bfn.py` | all DA cycles | **Split ownership.** Call-local index; R fixed per call (stage 6; the overview's `obs_err_source`) |
| `SmootherCycle(forward_model, obs_op, analysis_step, window, stride=1, obs_source=None)` | `da.py:284` | | **Healthy.** `stride` is windows per call; the analysis receives lists of forecasts and observations |
| `BFNCycle(forward_model, obs_op, nudging_gain, *, window, max_iters=10, tol=1e-3, …)` | `bfn.py:79` | | **Healthy.** Beyond the plan |
| `IdentityObs`, `LinearObs(H)`, `CallableObs(fn, linearize_fn=None)`, `CompositeObs(components)` | `obs.py` | | **Healthy.** `CompositeObs` applies `components[0]` first |
| `CallableForward(fn, dt)`, `CompositeForward(components)`, `NeuralForward(model_op, dt)` | `forward.py` | | **Healthy.** `CompositeForward` sums `dt` and ignores the one passed |
| the 14 protocols | `protocols.py` | algorithm adapters (not checked) | **Healthy.** All `runtime_checkable` |
| adjoint specs | `adjoints.py` | `pipekit_jax.adjoints` | **Healthy.** Specs only; pipekit-jax interprets them |
:::

Smaller findings:

- **The README is a release behind.** Its table says "What's shipped
  (v0.0.1)" and omits `BFNCycle`, the adjoint specs and the eleven further
  protocols. The CHANGELOG for 0.0.2 (11 July 2026) lists them.
- **The latent-space design is not built.** `docs/design/latent.md` (draft
  0.1.0) proposes a `latent.py` with `Encoder`, `Decoder`,
  `LatentForwardModel`, `LiftedObservationOperator`, `EncodedForwardModel`,
  `LatentDAState` and a `LatentDACycle` in three flavours (strong-latent,
  prior-only, hybrid). Only the `LatentMap` protocol ships.
- **`SmootherCycle` cannot see the window's start state.** It steps the
  model `window` times and passes the forecasts after each step, so a
  strong-constraint 4D-Var analysis, whose control is the start state, must
  carry it inside the carrier.
- **`WindowState` is unused**, and `Recurrence` never touches
  `IterationState`: the step owns the counters, as in stage 5.
- **Sequential threading works the other way round.** A cycle inside a
  `Sequential` is fine (`Inc() | Cycle(Inc(), 3)` runs); only a `Sequential`
  inside a cycle is rejected.

(r10-ws-a)=
## Workstream A · Stateful pipelines as cycle steps

**Decision.** A `Sequential` that holds at least one stateful step is a
valid `step_op` for every cycle.

:::{table} Workstream A's moves.
:label: tbl-r10-move-a

| Symbol | Destination | Reason |
|---|---|---|
| a `Sequential` branch in `_coerce_step` | `pipekit_cycle.cycle` | `Sequential._is_stateful` already identifies the case and `Sequential._apply_stateful` already threads the state; only the type check is missing |
| `Sequential` with no stateful step | still rejected, with a clearer message | a stateless pipeline is not a transition of the carry state; wrap it as a `CallableForward` or a `ForwardModel` instead |
:::

1. **Accept it.** In `_coerce_step`, return a `Sequential` unchanged when
   `step_op._is_stateful` is true, so `Cycle`, `EnsembleCycle`,
   `WindowedCycle` and `Recurrence` take it.
2. **Name the rule in the error.** A stateless `Sequential` gets a message
   that says which wrapper to use.
3. **Make the private check public** if the cycles are not the only callers:
   an `is_stateful` property on `Sequential` in pipekit core.

```{code-block} python
:caption: Stage 6's composed step, before and after workstream A.
:linenos:

# Shapes: x (N,)
step = pk.Sequential([pk.Lambda(lambda x: M @ x), Count()])   # stateful   today
# Today:
pc.Cycle(step, n_steps=3)                     # → TypeError                today
# After:
x3, s = pc.Cycle(step, n_steps=3)(x, pc.DAState())            # proposed (A)
# → x3 = M³ x, s.cycle_count = 3
```

(r10-ws-b)=
## Workstream B · Observation inputs on the cycle's clock

**Decision.** The DA cycles take their observation index, and optionally R,
from the cycle's own clock in the state.

:::{table} Workstream B's moves.
:label: tbl-r10-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `obs_clock: Literal["call", "state"] = "call"` | `DACycle`, `EnsembleDACycle`, `SmootherCycle`, `BFNCycle` | with `"state"`, `obs_source` receives `state.cycle_count` (or the window index), so a restarted cycle reads the right observations; the default keeps today's behaviour |
| `obs_err_source: Operator \| None` | the same cycles | returns $R_t$ for the step and writes it into `DAState.obs_err_cov` before the analysis; the overview's recommended answer to a per-step R ([overview](../overview.md), open questions) |
:::

### B1 · An index that survives restarts

The cycle already advances `cycle_count` in the state. With
`obs_clock="state"` it passes that count to `obs_source` instead of the
call-local `i`. Nothing else changes, and the default stays `"call"` so
existing scripts keep their meaning.

### B2 · A per-step observation error

Today a script can return `(y_t, R_t)` from `obs_source` and unpack it in
its own analysis step, because the analysis is user code. That works, but
it hides R from the cycle and from every analysis step that expects
`obs_err_cov`. `obs_err_source` puts $R_t$ where the protocol already
expects it.

```{code-block} python
:caption: Stage 6's restart, before and after workstream B.
:linenos:

# Shapes: as stage 2
# Today:
da5 = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                 obs_source=obs_source, n_steps=5)                    # today
# → second call reads y₀ … y₄ again; max |m_restart − m| = 15.4 ppb

# After:
da5 = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                 obs_source=obs_source, obs_err_source=r_source,
                 obs_clock="state", n_steps=5)                  # proposed (B1, B2)
# → second call reads y₅ … y₉; m_restart equals stage 2's m
```

(r10-gates)=
## Testing gates

Each workstream merges only when its gates pass; the existing gates are
checks the code passes today and must keep passing.

:::{table} Gates for the proposals and for what ships.
:label: tbl-r10-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Composed step | A | `Cycle(Sequential([Lambda(M @ ·), Count()]), n_steps=3)` returns $M^3 x$ to 0 and `cycle_count == 3`; a stateless `Sequential` still raises `TypeError` |
| B1 Restart | B1 | two 5-step calls with `obs_clock="state"` give the 10-step analysis of stage 2 to 0 (today 15.4 ppb); with the default the indices are `[0 … 4, 0 … 4]` as today |
| B2 Per-step R | B2 | with `obs_err_source` returning $R_t = (0.1 + 0.01 t) I$, the cycle equals a hand-written Kalman filter using the same $R_t$ to 0 |
| K1 Kalman exactness (existing) | — | stage 2: `DACycle` equals the hand-written filter to 0 in mean and covariance. Passes today |
| K2 EnKF rate (existing) | — | stage 3: the median RMS gap at $N_e = 800$ is $2 \pm 0.5$ times smaller than at $N_e = 200$, the $N_e^{-1/2}$ rate. Passes today: 2.3 |
| K3 Model swap (existing) | — | stage 4: `CallableForward` and `NeuralForward` wrapping the same step give the same analysis to 0. Passes today |
| K4 Retrieval (existing) | — | stage 5: Gauss–Newton converges in at most 10 iterations with $|J^\top (y - f(\hat{\alpha}))| < 10^{-12}$. Passes today: 6 iterations, 9.9e-20 |
:::

(r10-risks)=
## Sequencing and risks

The two workstreams are independent and small, so they ship together with
Report 13's ensemble histories; the documentation follows.

:::{figure} ../images/reports/r10_roadmap.svg
:label: fig-r10-roadmap
:alt: Phase 1: A1 stateful Sequential steps, B1 the observation clock, and Report 13's D1 ensemble histories, side by side, then gates A1, B1 and R13-D1. Phase 2: the README and the latent design.

Small fixes first, then the documentation and the latent design.
:::

- **A default that changes meaning.** Making `obs_clock="state"` the
  default would change every existing script that restarts a cycle on
  purpose with per-call observations. Fallback: keep `"call"` as the default
  and warn when a DA cycle starts with `cycle_count > 0` and the default
  clock.
- **Stateful detection leaks a private attribute.** Workstream A reads
  `Sequential._is_stateful`. Fallback: add the public `is_stateful` property
  in pipekit core in the same release, or duck-type on `_apply_stateful`.
- **Analysis steps that need the state.** Some analyses want the cycle
  count or the time (adaptive inflation, time-dependent localisation), and
  today the analysis never sees the state. Fallback: keep the carrier
  convention and pass time through `obs_source`'s output; revisit only if
  two adapters need it (open question 2).
- **Unverified adapters.** filterax, vardax and plumax are not pinned in
  `references/`, so whether their adapters satisfy the protocols is
  unchecked. Fallback: pin one of them and add an adapter test that runs
  stage 2 with its analysis step.

(r10-open-questions)=
## Open questions

1. **Should a plain `Cycle` advance the clock?** `DACycle` advances
   `DAState.t` by `dt`; `Cycle` leaves it at 0, so Report 13 computes valid
   times from the history index. Advancing `t` when the step is a
   `ForwardModel` and the state has `t` would make the two agree.
2. **Should `AnalysisStep` receive the state?** Passing `state=` as a
   keyword would let analyses read the clock, but it changes the protocol
   every adapter satisfies.
3. **A cycle checkpoint.** The scoping report's `Checkpoint(cycle_op)` does
   not exist. A checkpoint is the triple (config, `CarryState.to_dict()`,
   carrier); is it a pipekit-cycle operator, a pipekit-experiment artifact,
   or a script's job?
4. **The window's start state in `SmootherCycle`.** Should the analysis
   receive the state the window started from, so a 4D-Var analysis need not
   carry it inside the carrier?
5. **`CompositeForward` and `dt`.** Summing the components' steps is one
   reading of composition (operator splitting with sequential sub-steps).
   Should a composite that advances all components over one shared `dt`
   exist as well?
6. **The latent design.** Build `latent.py` as `docs/design/latent.md`
   drafts it, or retire the draft in favour of the shipped `LatentMap` and
   `ReducedOrderModel` protocols?

(r10-todos)=
## Upcoming work

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r10-todos

| Follow-up | Where | Why |
|---|---|---|
| Refresh the README for 0.0.2 and fix `filterx` → `filterax` in its quickstart | pipekit-cycle `README.md` | the table lists v0.0.1 and omits `BFNCycle`, the adjoint specs and eleven protocols |
| Document the carrier convention for covariances and ensembles | pipekit-cycle docs | the analysis never sees the state, so $P$ and the ensemble belong in the carrier (stage 2) |
| Add an adapter test against one algorithm library | pipekit-cycle tests, with filterax pinned | the protocols' main users are unchecked |
:::

(r10-references)=
## References

<!-- TODO(ref): placeholders. When a report page lands, point its links in this
file at the page and delete its entry below. -->

:::{note} Placeholders
:class: simple
Report 12 (pipekit-experiment) is still being written. Reports
[2](02_pipekit_surface.md), [9](09_use_cases.md), [11](11_pipekit_train.md) and
[13](13_state_catalog.md) are published.
:::

### Forthcoming reports

(r10-ref-report-11)=
Report 12 · pipekit-experiment
: *Forthcoming.* Registering the emulators that stage 4 swaps in.

### Sources

- The scoping report, "Report 10 — `pipekit-cycle`: time-stepping, DA, and
  observation operators" (supplied by the author).
- pipekit-cycle 0.0.2 in `references/pipekit/packages/pipekit-cycle`: its
  source, README, CHANGELOG, `docs/design/latent.md` and tests.
- The running example's script and outputs, described in [](#r10-running).
