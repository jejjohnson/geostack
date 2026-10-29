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

pipekit-cycle turns a one-step transition into a run over time. A `Cycle` repeats a step, a `DACycle` alternates forecast and analysis, and a `Recurrence` iterates until a condition holds. Three protocols (`ForwardModel`, `ObservationOperator`, `AnalysisStep`) let algorithm libraries plug in without importing it. The scoping report described this package before it was built, and version 0.0.2 ships everything the report named and more: eleven further protocols, a back-and-forth nudging cycle and adjoint specs that pipekit-jax reads. Where the two disagree, **the code holds** ([conventions](../overview.md#sec-conventions)). A Kalman filter written as a `DACycle` matches a hand-written loop exactly, and an EnKF converges to it at the $1/\sqrt{N_{\mathcal{E}}}$ rate (with $N_{\mathcal{E}}$ the number of members). However, two real gaps remain: 1) a stateful `Sequential` cannot be a cycle's step, and 2) the observation index restarts on every call, which silently feeds a restarted cycle the wrong observations. We propose two small fixes on this page.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Built. Originally a scoping proposal ("committed"); pipekit-cycle 0.0.2 is the design of record, and this page follows it |
| **Reading time** | about 25 minutes |
| **Audience** | anyone writing a forward simulation, a DA cycle or an iterative retrieval with the stack, or an adapter for an algorithm library |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit core, where `StatefulOperator` and `CarryState` live); [Report 9](09_use_cases.md) (use cases); [Report 13](13_state_catalog.md) (indexing the states a cycle produces); [Report 11](11_pipekit_train.md) (training the emulators of stage 4); [Report 12](12_pipekit_experiment.md) (the registry); [Report 16](16_geo_data_stack.md#r16-ownership) (the stack rules) |
| **Code checked against** | `pipekit-cycle` 0.0.2 and `pipekit` 0.0.2 at `ef3c17f`, the revision pinned in `references/geostack/pipekit`; its 100 tests pass. Read on 13 October 2026 |
| **Running example** | every output below is from one run on that revision of `notebooks/reports/report10_pipekit_cycle.py`, which seeds every draw and asserts each number the page shows |
:::

(r10-summary)=
## Summary

**Decision.** pipekit-cycle 0.0.2 stays as it is: a separate package beside pipekit-array, which depends only on pipekit and has no JAX and no domain code. Its protocols, cycles and helpers are the contract that algorithm libraries target. Two additions close the gaps the running example hits, and one more is already proposed in [Report 13](13_state_catalog.md#r13-ws-d).

- **A. Stateful pipelines as cycle steps.** `pipekit.Sequential` threads a carry state when it holds a stateful step, but it is not a `StatefulOperator`, so `Cycle` and its siblings reject it with a `TypeError`. We would accept any stateful `Sequential` as a step.
- **B. Observation inputs on the cycle's clock.** `obs_source(i)` receives a call-local index that restarts at 0 on every call, and the observation-error covariance $\gmat{\Sigma}_\varepsilon$ only comes from the state. We would let the index follow `state.cycle_count` (B1), and let a source set $\gmat{\Sigma}_\varepsilon$ per step (B2, the overview's `obs_err_source`).
- **Report 13 D1. Ensemble histories.** `EnsembleCycle` and `EnsembleDACycle` keep no trajectory, so [Report 13](13_state_catalog.md#r13-ws-d) adds `save_history` to both, which lets cycle outputs be written and indexed.

**End state.** A script composes a step from stateless and stateful operators, cycles it, and restarts it from the last state without re-reading old observations. Algorithm libraries supply forward models, observation operators and analysis steps by structural typing. Every cycle can keep its history, which Report 13's state profile indexes.

(r10-ownership)=
## Ownership rules

pipekit-cycle owns the shape of a run over time, whereas the carrier, the physics and the algorithms belong to whoever supplies the steps ([](#fig-r10-ownership)).

:::{figure} ../images/reports/r10_ownership.svg
:label: fig-r10-ownership
:alt: Scripts build cycles from pipekit-cycle; algorithm libraries satisfy its protocols by structural typing. pipekit-cycle groups cycles, DA cycles, protocols, observation and forward helpers, carry states and adjoint specs. It sits on pipekit core; pipekit-jax reads its adjoint specs.

pipekit-cycle 0.0.2 between pipekit core and the code that uses it. Dashed
arrows are structural: nothing is imported.
:::

**pipekit-cycle** owns the cycles, the DA cycles, the protocols, the generic observation and forward helpers, the carry states and the adjoint specs. It only imports pipekit, and never an array library, JAX or a domain package.

Neighbours follow [the stack rules](16_geo_data_stack.md#r16-ownership), and two of them matter here: 1) algorithm libraries (filterax, vardax, plumax) satisfy the protocols by structural typing without importing pipekit-cycle, and 2) scripts own the carrier, the initial state, the observations and every file a run writes ([Report 13](13_state_catalog.md#r13-ws-d)).

:::{warning} Differs from the plan
The scoping report listed three extras: `array`, `jax` and `catalog` (the last one for `statecatalog`). pipekit-cycle 0.0.2 has one, `array = ["pipekit-array"]`, and it only depends on `pipekit`. There is no `jax` extra, because JAX execution lives in pipekit-jax, which reads pipekit-cycle's adjoint specs instead of pipekit-cycle importing it. There is no `catalog` extra either, because [Report 13](13_state_catalog.md) replaces `statecatalog` with a state profile in geocatalog that scripts write to (not cycles).
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

The eleven further protocols (the italic rows of [](#tbl-r10-surface)) are the shared vocabulary for the algorithm libraries: a tangent-linear model and its adjoint (the Jacobian $\mat{J}_f$ of the dynamics $f$ and its transpose $\mat{J}_f^\top$), a low-rank error subspace $\gmat{\Sigma}_x \approx \mat{L} \mat{L}^\top$, a reduced-order model with `encode`, `decode` and `step`, covariance localisers, ensemble inflators, adaptive step schedulers, and ensemble Kalman inversion as an `IterativeProcess`. No cycle in pipekit-cycle uses any of them. They exist so that, for example, filterax's inflator and vardax's error subspace agree on one signature.

(r10-running)=
## The running example: a methane plume advected along a transect

**The problem.** A methane enhancement $\vect{x}_t \in \mathbb{R}^{D_x}$, in ppb above background, sits on a periodic transect of $D_x = 40$ cells, and $t$ indexes the hours. The wind carries it downstream one hour at a time. A column sensor sees every fourth cell, so there are $D_y = 10$ observations $\vect{y}_t \in \mathbb{R}^{D_y}$ per hour, for $T = 10$ hours. The plume starts as a Gaussian with a 50 ppb peak at cell 8.

**The model.** The dynamics are first-order upwind advection with Courant number 0.4, which is the linear map $\mat{F} = 0.6 \mat{I} + 0.4 \mat{S} \in \mathbb{R}^{D_x \times D_x}$, where $\mat{S} \in \{0, 1\}^{D_x \times D_x}$ is the one-cell downstream shift. The model-error covariance is $\gmat{\Sigma}_\eta = 0.01 \mat{I} \in \mathbb{R}^{D_x \times D_x}$, the observation operator $\mat{H} \in \{0, 1\}^{D_y \times D_x}$ selects every fourth cell, and the observation-error covariance is $\gmat{\Sigma}_\varepsilon = 0.1 \mat{I} \in \mathbb{R}^{D_y \times D_y}$. The truth and the observations follow {eq}`r10-eq-model` and {eq}`r10-eq-model-obs`,

$$
\vect{x}_t = \mat{F} \vect{x}_{t-1} + \gvect{\eta}_t ,
$$ (r10-eq-model)

$$
\vect{y}_t = \mat{H} \vect{x}_t + \gvect{\varepsilon}_t ,
$$ (r10-eq-model-obs)

where $\gvect{\eta}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\eta)$ is the model noise in $\mathbb{R}^{D_x}$ and $\gvect{\varepsilon}_t \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon)$ is the observation noise in $\mathbb{R}^{D_y}$. The filter starts from the prior mean $\gvect{\mu}_{x,0 \mid 0} = \vect{0}$ with the prior covariance $\gmat{\Sigma}_{x,0 \mid 0} = 100 \mat{I}$, so it knows nothing about the plume. The data are synthetic, so every stage has an exact answer to check against.

In the code sketches, the arrays keep the script's names: `M` is $\mat{F}$, `Q` is $\gmat{\Sigma}_\eta$, `R` is $\gmat{\Sigma}_\varepsilon$, `m` and `P` are the mean and covariance, and `N` is $D_x$.

(r10-stage-1)=
### Stage 1 · Forecast only (pipekit-cycle)

**TL;DR.** Push the plume downstream for six hours and keep every second
state.

**Problem.** With no observations, the forecast is simply $\vect{x}_t = \mat{F} \vect{x}_{t-1}$ repeated $n$ times, and a history with stride $s$ keeps the states after steps $1, 1 + s, 1 + 2s, \ldots$

```{code-block} python
:caption: A forecast is a Cycle over a ForwardModel; the history lives on the operator.
:linenos:

# Shapes: D_x = 40 cells (N in the code); M ≡ F
import pipekit_cycle as pc

class Advect:                                  # satisfies pc.ForwardModel    user
    dt = 1.0
    state_signature = None
    # x_t = F x_{t−1}
    def step(self, x, dt):
        return M @ x                           # (D_x,) → (D_x,)

fc = pc.Cycle(Advect(), n_steps=6, save_history=True, history_stride=2)  # today
x6, state = fc(x_true_0, pc.DAState())         # (D_x,) → (D_x,)             today
len(fc.history)                                # → 3: after steps 1, 3, 5    today
[int(np.argmax(c)) for c, _ in fc.history]     # → [9, 10, 10]: the peak moves
state.t                                        # → 0.0: Cycle keeps no clock
```

A plain `Cycle` wraps the `ForwardModel` in an internal adapter and threads the state through untouched, so `DAState.t` stays at 0. Only the DA cycles advance it.

:::{warning} Differs from the plan
The scoping report said a `Cycle` returns `(final_carrier, final_state, history)`. It returns `(final_carrier, final_state)`, and the history is the list `Cycle.history` of `(carrier, state)` pairs, which is reset on every call. The last step is only kept when $(n - 1)$ is a multiple of the stride.
:::

(r10-stage-2)=
### Stage 2 · A Kalman filter as a DACycle (pipekit-cycle)

**TL;DR.** Assimilate the ten hours of observations with an exact Kalman
filter, and check that the cycle changes nothing about the arithmetic.

**Problem.** The Kalman filter carries a mean $\gvect{\mu}_{x,t \mid t} \in \mathbb{R}^{D_x}$ and a covariance $\gmat{\Sigma}_{x,t \mid t} \in \mathbb{R}^{D_x \times D_x}$, where $t \mid t-1$ marks the forecast and $t \mid t$ the analysis. The forecast step propagates both through the dynamics,

$$
\gvect{\mu}_{x,t \mid t-1} = \mat{F} \gvect{\mu}_{x,t-1 \mid t-1} ,
$$ (r10-eq-kf)

$$
\gmat{\Sigma}_{x,t \mid t-1} = \mat{F} \gmat{\Sigma}_{x,t-1 \mid t-1} \mat{F}^\top + \gmat{\Sigma}_\eta ,
$$ (r10-eq-kf-cov)

and the analysis step corrects them with the gain $\mat{G}_t \in \mathbb{R}^{D_x \times D_y}$,

$$
\mat{G}_t = \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top \left( \mat{H} \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top + \gmat{\Sigma}_\varepsilon \right)^{-1} ,
$$ (r10-eq-kf-gain)

$$
\gvect{\mu}_{x,t \mid t} = \gvect{\mu}_{x,t \mid t-1} + \mat{G}_t \left( \vect{y}_t - \mat{H} \gvect{\mu}_{x,t \mid t-1} \right) ,
$$ (r10-eq-kf-mean)

$$
\gmat{\Sigma}_{x,t \mid t} = \left( \mat{I} - \mat{G}_t \mat{H} \right) \gmat{\Sigma}_{x,t \mid t-1} .
$$ (r10-eq-kf-cov-a)

```{code-block} python
:caption: The Kalman filter as a DACycle, against a hand-written loop over {eq}`r10-eq-kf` to {eq}`r10-eq-kf-cov-a`.
:linenos:

# Shapes: D_x = 40, D_y = 10, T = 10; the carrier is (m, P): (D_x,), (D_x, D_x)
class KFForward:                               # pc.ForwardModel              user
    dt = 1.0
    state_signature = None
    # μ_{t|t−1} = F μ_{t−1|t−1};  Σ_{t|t−1} = F Σ_{t−1|t−1} Fᵀ + Σ_η
    def step(self, carrier, dt):
        m, P = carrier
        return M @ m, M @ P @ M.T + Q          # ((D_x,), (D_x, D_x)) → same

class KalmanAnalysis:                          # pc.AnalysisStep              user
    # G = Σ_{t|t−1} Hᵀ (H Σ_{t|t−1} Hᵀ + Σ_ε)⁻¹
    # μ_{t|t} = μ_{t|t−1} + G (y − H μ_{t|t−1});  Σ_{t|t} = (I − G H) Σ_{t|t−1}
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        m, P = forecast
        Hm = obs_op.linearize(m).H                                   # (D_y, D_x)
        K = P @ Hm.T @ np.linalg.inv(Hm @ P @ Hm.T + obs_err_cov)  # (D_x, D_y)
        return m + K @ (obs - obs_op(m)), (np.eye(N) - K @ Hm) @ P

da = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                obs_source=pk.Lambda(lambda i: y[i]), n_steps=T)    # today
(m, P), state = da((m0, P0), pc.DAState(obs_err_cov=R))           # today
# → max |m − m_hand| = 0.0, max |P − P_hand| = 0.0
# → RMSE against the truth 0.125 ppb (prior RMSE 17.1 ppb)
state.t, state.cycle_count, len(da.history)   # → 10.0, 10, 10       today
```

Each cycle steps the forward model, calls `obs_source(i)` for the step's observations, and hands both to the analysis with `obs_err_cov` read from the state. The result is bit-for-bit the hand-written filter. Two conventions follow from the code, and they are worth stating once:

- **The covariance travels in the carrier.** `AnalysisStep.__call__` receives the forecast, the observations, the observation operator and $\gmat{\Sigma}_\varepsilon$, but never the state. So anything the analysis updates, such as $\gmat{\Sigma}_{x,t \mid t}$ or an ensemble, must be part of the carrier, which here is a `(m, P)` tuple.
- **`ObservationOperator` requires `linearize`.** The protocol check fails for a class with only `__call__`. `LinearObs` and `IdentityObs` return themselves, whereas `CallableObs` takes an optional `linearize_fn` and raises `NotImplementedError` without it.

:::{warning} Differs from the plan
The scoping report's `DAState` held `background_state` and `ensemble_members` alongside `t` and `cycle_count`. The shipped one holds `t`, `cycle_count`, `obs_err_cov` and a free-form `extras` dict. The background and the ensemble are the carrier, which is why the analysis never needs the state. `IterationState` is `iter_count`, `residual_norm` and `extras` (no `lr`, no `history`, and `iter_count` rather than `iter`), and `WindowState` is `window_index`, `last_analysis` and `extras`. No cycle uses `WindowState` yet.
:::

(r10-stage-3)=
### Stage 3 · The same problem as a stochastic EnKF (pipekit-cycle)

**TL;DR.** Replace the exact covariance with an ensemble and check that the
ensemble mean approaches the Kalman mean as the ensemble grows.

**Problem.** With $N_{\mathcal{E}}$ members $\vect{x}^{(e)} \in \mathbb{R}^{D_x}$, $e \in \mathcal{E}$, the forecast covariance is the sample covariance of the anomalies, and every member is updated against its own perturbed observations, as in the overview ({eq}`eq-enkf`). For a linear-Gaussian model, the EnKF mean converges to the Kalman mean $\gvect{\mu}_{x,t \mid t}$, and its error falls as $N_{\mathcal{E}}^{-1/2}$ once the ensemble is large enough that the gain is well estimated.

```{code-block} python
:caption: An EnKF as an EnsembleDACycle; the carrier is a list of members.
:linenos:

# Shapes: N_e members, each (D_x,); the carrier is a list of N_e arrays
class EnKF:                                    # pc.AnalysisStep              user
    # G = Σ_{t|t−1} Hᵀ (H Σ_{t|t−1} Hᵀ + Σ_ε)⁻¹;  x⁽ᵉ⁾ ← x⁽ᵉ⁾ + G (y + ε⁽ᵉ⁾ − H x⁽ᵉ⁾)
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov): ...   # list → list

eda = pc.EnsembleDACycle(Member(), pc.LinearObs(H), EnKF(),
                         obs_source=obs_source, n_steps=T, n_members=N_e)  # today
members, state = eda(members_0, pc.DAState(obs_err_cov=R))   # N_e × (D_x,)   today
# RMS |mean(members) − μ_{t|t}| , median over 10 seeds; KF posterior std 0.193 ppb
# → N_e =  50: 0.367   N_e = 200: 0.063   N_e = 800: 0.027
```

From 200 to 800 members the gap shrinks by a factor of 2.3, which is close to the factor of 2 that $N_{\mathcal{E}}^{-1/2}$ predicts. The log-log slope is −0.61, and a seeded bootstrap over the ten seeds puts its 99 % interval at [−0.96, −0.38], which contains −0.5. At 50 members the gap is six times larger than at 200, because with no localisation 50 members estimate a 40 × 40 covariance poorly and the gain carries that error. Inflation and localisation are the analysis step's business, so pipekit-cycle names their signatures (`EnsembleInflator`, `CovarianceLocalizer`) but does not apply them.

:::{warning} Differs from the plan
The scoping report described the ensemble carrier as "a stacked ensemble (leading axis = ensemble member)". However, the shipped `EnsembleCycle` and `EnsembleDACycle` take a **list** of members, length-checked against `n_members`, and they step each member separately, with one carry state threaded through all the members in turn. Vectorising over members is the user's choice (a batched or vmapped step). Neither ensemble cycle keeps a history, which [Report 13](13_state_catalog.md#r13-ws-d)'s D1 adds.
:::

(r10-stage-4)=
### Stage 4 · Swap the forward model (pipekit-cycle)

**TL;DR.** Replace the physical model with an emulator and leave the cycle
unchanged.

**Problem.** The cycle only needs something that satisfies `ForwardModel`. So an emulator $\hat{f}$ trained to reproduce the dynamics $\mat{F}$ should drop in with no other change, and an exact emulator must give the same analysis.

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

`NeuralForward(model_op, dt, state_signature=None)` is a thin wrapper that works, not a stub. Its `step` calls `model_op(state)` and ignores `dt`, so any pipekit `Operator` (a registry-loaded model, a `Sequential`) becomes a forward model.

:::{warning} Differs from the plan
`CompositeForward` has **no `dt` argument**. Its `dt` is the sum of its components' `dt`, and its `step` ignores the `dt` it is passed and steps each component by its own. The scoping report said it "enforces consistent total dt across components", whereas the code adds them. `BFNCycle` rejects `CompositeForward` and `NeuralForward` unless a `backward_model` is given, because it needs to step backwards in time.
:::

(r10-stage-5)=
### Stage 5 · A Gauss–Newton retrieval as a Recurrence (pipekit-cycle)

**TL;DR.** Retrieve a methane column enhancement from five-band
transmittances by iterating until the update is negligible.

**Problem.** Beer–Lambert absorption gives the transmittance $h_b(\alpha) = e^{-\sigma_b \alpha}$ in each band $b = 1, \ldots, 5$, where $\alpha \in \mathbb{R}$ is the column enhancement and $\sigma_b \in \mathbb{R}_{>0}$ is the band's absorption coefficient. We stack the bands into the forward model $h(\alpha) \in \mathbb{R}^5$ and the measured transmittances into $\vect{y} \in \mathbb{R}^5$ (a separate observation vector from the transect's $\vect{y}_t$). With the Jacobian $\vect{J}_h \in \mathbb{R}^5$, whose entries are $-\sigma_b e^{-\sigma_b \alpha}$, Gauss–Newton iterates {eq}`r10-eq-gn`,

$$
\alpha^{(j+1)} = \alpha^{(j)} + \frac{\vect{J}_h^\top \left( \vect{y} - h(\alpha^{(j)}) \right)}{\vect{J}_h^\top \vect{J}_h} ,
$$ (r10-eq-gn)

where $j$ counts the iterations and $\vect{J}_h$ is evaluated at $\alpha^{(j)}$. It stops when $\vert \alpha^{(j+1)} - \alpha^{(j)} \vert < 10^{-8}$.

```{code-block} python
:caption: An iterative retrieval is a Recurrence with a stateful step and a condition operator.
:linenos:

# Shapes: α scalar; y, h(α), J_h: (5,)
h = pc.CallableObs(lambda a: np.exp(-sigma * a),
                   linearize_fn=lambda a: -sigma * np.exp(-sigma * a))  # today

class GaussNewton(pk.StatefulOperator):                               # user
    # α ← α + J_hᵀ (y − h(α)) / (J_hᵀ J_h)
    def _apply(self, a, s):
        J, r = h.linearize(a), y_b - h(a)                     # (5,), (5,)
        step = float(J @ r / (J @ J))
        return a + step, pc.IterationState(iter_count=s.iter_count + 1,
                                           residual_norm=abs(step))

gn = pc.Recurrence(GaussNewton(),
                   condition_op=pk.Lambda(lambda s: s.residual_norm < 1e-8),
                   max_iters=50)                                      # today
a_hat, s = gn(0.0, pc.IterationState())                               # today
# → α̂ = 120.014 (truth 120), gn.iters = 6, gn.converged = True
# → J_hᵀ (y − h(α̂)) = 9.9e-20: the normal equations hold at α̂
```

`Recurrence(step_op, condition_op, max_iters=1000)` calls `condition_op` on the state after each step and stops on a truthy result, and it records `iters` and `converged` on the operator. The condition must be an `Operator`, so a bare function is wrapped in `pk.Lambda`.

:::{warning} Differs from the plan
The scoping report's retrieval used `condition_op=pk.qc.AssertCallable(predicate=…)`. However, `AssertCallable` is a QC pass-through that **raises `QCError` when the predicate is false**, so that recurrence stops with an exception on its first iteration. A condition is a `pk.Lambda` that returns a boolean.
:::

(r10-stage-6)=
### Stage 6 · Restart the filter, and compose a step (the two gaps)

**TL;DR.** Run the ten hours as two five-hour calls, as an operational
cycle restarted from its last state would, and build a cycle step from a
`Sequential`.

**Problem.** Restarting a filter from its last carrier and state must give the same analysis as one uninterrupted run, provided that the second call reads the observations $\vect{y}_5, \ldots, \vect{y}_9$. In addition, a step that mixes a stateless operator and a stateful one is a `Sequential`, which pipekit core threads the state through.

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

The restart silently re-assimilates the first five hours of observations and misplaces the plume by up to 15 ppb with no warning, because the state's `cycle_count` is 10 but `obs_source` sees the call-local index. `BFNCycle`'s docstring already tells users to "supply a stateful or clock-aware `obs_source`" for a global clock. Workstream B makes the clock part of the cycle, and the `Sequential` case is workstream A.

(r10-names)=
## The scoping report's names, checked

[](#tbl-r10-names) checks the stack names that the scoping report's examples use.

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

Two of the scoping report's pieces need action, and the rest ship and work ([](#tbl-r10-audit)). [](#fig-r10-readiness) counts the page's calls by marker. The audit read `references/geostack/pipekit` at `ef3c17f` on 13 October 2026.

:::{figure} ../images/reports/r10_readiness.svg
:label: fig-r10-readiness
:alt: A grid with one row per stage of the running example and per workstream, and one square per stack call in that row's sketches, coloured by its marker. Of 27 calls, 20 run today, 2 are proposed by this report's workstreams and 5 are code the reader writes.

What this page's sketches call: of 27 stack calls, 20 run today, 2 are
proposed by this report's workstreams and 5 are code the reader writes
([how to read it](../overview.md#sec-conventions)).
:::

:::{table} pipekit-cycle 0.0.2, piece by piece.
:label: tbl-r10-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `Cycle(step_op, n_steps, save_history=False, history_stride=1)` | `cycle.py:71` | `SimulationDataset` (pipekit-train), `Sequential` | **Healthy.** Returns `(carrier, state)`; history on `.history` |
| `EnsembleCycle(step_op, n_steps, n_members)` | `cycle.py:139` | | **Healthy.** List carrier, shared state; no history ([Report 13 D1](13_state_catalog.md#r13-ws-d)) |
| `WindowedCycle(step_op, window, stride=1)` | `cycle.py:202` | | **Healthy.** Input is a list of carriers; returns the window-final carriers |
| `Recurrence(step_op, condition_op, max_iters=1000)` | `cycle.py:265` | | **Healthy.** Records `iters`, `converged` |
| `step_op` coercion, `_coerce_step` | `cycle.py:50` | every cycle above | **Split ownership.** Accepts a `StatefulOperator` or a `ForwardModel`; rejects a stateful `Sequential` that pipekit core threads state through |
| `DACycle(forward_model, obs_op, analysis_step, obs_source=None, n_steps=1, save_history=True)` | `da.py:84` | the overview's stage 4 | **Healthy.** Exact against a hand-written Kalman filter |
| `EnsembleDACycle(..., n_steps=1, *, n_members)` | `da.py:184` | | **Healthy.** No `key`, no inflation argument, no history |
| `obs_source(i)` and `obs_err_cov` from the state | `da.py`, `bfn.py` | all DA cycles | **Split ownership.** Call-local index; $\gmat{\Sigma}_\varepsilon$ fixed per call (stage 6; the overview's `obs_err_source`) |
| `SmootherCycle(forward_model, obs_op, analysis_step, window, stride=1, obs_source=None)` | `da.py:284` | | **Healthy.** `stride` is windows per call; the analysis receives lists of forecasts and observations |
| `BFNCycle(forward_model, obs_op, nudging_gain, *, window, max_iters=10, tol=1e-3, …)` | `bfn.py:79` | | **Healthy.** Beyond the plan |
| `IdentityObs`, `LinearObs(H)`, `CallableObs(fn, linearize_fn=None)`, `CompositeObs(components)` | `obs.py` | | **Healthy.** `CompositeObs` applies `components[0]` first |
| `CallableForward(fn, dt)`, `CompositeForward(components)`, `NeuralForward(model_op, dt)` | `forward.py` | | **Healthy.** `CompositeForward` sums `dt` and ignores the one passed |
| the 14 protocols | `protocols.py` | algorithm adapters (not checked) | **Healthy.** All `runtime_checkable` |
| adjoint specs | `adjoints.py` | `pipekit_jax.adjoints` | **Healthy.** Specs only; pipekit-jax interprets them |
:::

Smaller findings:

- **The README is a release behind.** Its table says "What's shipped (v0.0.1)" and it omits `BFNCycle`, the adjoint specs and the eleven further protocols, which the CHANGELOG for 0.0.2 (11 July 2026) lists.
- **The latent-space design is not built.** `docs/design/latent.md` (draft 0.1.0) proposes a `latent.py` with `Encoder`, `Decoder`, `LatentForwardModel`, `LiftedObservationOperator`, `EncodedForwardModel`, `LatentDAState` and a `LatentDACycle` in three flavours (strong-latent, prior-only, hybrid). Only the `LatentMap` protocol ships.
- **`SmootherCycle` cannot see the window's start state.** It steps the model `window` times and passes the forecasts after each step, so a strong-constraint 4D-Var analysis, whose control is the start state, must carry that state inside the carrier.
- **`WindowState` is unused**, and `Recurrence` never touches `IterationState`, so the step owns the counters (as in stage 5).
- **Sequential threading works the other way round.** A cycle inside a `Sequential` is fine (`Inc() | Cycle(Inc(), 3)` runs), and only a `Sequential` inside a cycle is rejected.

(r10-ws-a)=
## Workstream A · Stateful pipelines as cycle steps

**Decision.** A `Sequential` that holds at least one stateful step is a valid `step_op` for every cycle ([](#tbl-r10-move-a)).

:::{table} Workstream A's moves.
:label: tbl-r10-move-a

| Symbol | Destination | Reason |
|---|---|---|
| a `Sequential` branch in `_coerce_step` | `pipekit_cycle.cycle` | `Sequential._is_stateful` already identifies the case and `Sequential._apply_stateful` already threads the state; only the type check is missing |
| `Sequential` with no stateful step | still rejected, with a clearer message | a stateless pipeline is not a transition of the carry state; wrap it as a `CallableForward` or a `ForwardModel` instead |
:::

1. **Accept it.** In `_coerce_step`, we return a `Sequential` unchanged when `step_op._is_stateful` is true, so that `Cycle`, `EnsembleCycle`, `WindowedCycle` and `Recurrence` all take it.
2. **Name the rule in the error.** A stateless `Sequential` gets a message that says which wrapper to use.
3. **Make the private check public** if the cycles are not the only callers, with an `is_stateful` property on `Sequential` in pipekit core.

```{code-block} python
:caption: Stage 6's composed step, before and after workstream A.
:linenos:

# Shapes: x (D_x,); M ≡ F
step = pk.Sequential([pk.Lambda(lambda x: M @ x), Count()])   # stateful   today
# Today:
pc.Cycle(step, n_steps=3)                     # → TypeError                today
# After:
x3, s = pc.Cycle(step, n_steps=3)(x, pc.DAState())            # proposed (A)
# → x3 = F³ x, s.cycle_count = 3
```

(r10-ws-b)=
## Workstream B · Observation inputs on the cycle's clock

**Decision.** The DA cycles take their observation index, and optionally the observation-error covariance $\gmat{\Sigma}_{\varepsilon,t}$, from the cycle's own clock in the state ([](#tbl-r10-move-b)).

:::{table} Workstream B's moves.
:label: tbl-r10-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `obs_clock: Literal["call", "state"] = "call"` | `DACycle`, `EnsembleDACycle`, `SmootherCycle`, `BFNCycle` | with `"state"`, `obs_source` receives `state.cycle_count` (or the window index), so a restarted cycle reads the right observations; the default keeps today's behaviour |
| `obs_err_source: Operator \| None` | the same cycles | returns $\gmat{\Sigma}_{\varepsilon,t}$ for the step and writes it into `DAState.obs_err_cov` before the analysis; the overview's recommended answer to a per-step $\gmat{\Sigma}_\varepsilon$ ([overview](../overview.md), open questions) |
:::

### B1 · An index that survives restarts

The cycle already advances `cycle_count` in the state. With `obs_clock="state"`, it passes that count to `obs_source` instead of the call-local `i`. Nothing else changes, and the default stays `"call"` so that existing scripts keep their meaning.

### B2 · A per-step observation error

Today a script can return `(y_t, R_t)` from `obs_source` and unpack it in its own analysis step, because the analysis is user code. That works, but it hides $\gmat{\Sigma}_{\varepsilon,t}$ from the cycle and from every analysis step that expects `obs_err_cov`. So `obs_err_source` puts $\gmat{\Sigma}_{\varepsilon,t}$ where the protocol already expects it.

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

Each workstream only merges when its gates pass, and the existing gates are checks the code passes today and must keep passing ([](#tbl-r10-gates)).

:::{table} Gates for the proposals and for what ships.
:label: tbl-r10-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Composed step | A | `Cycle(Sequential([Lambda(M @ ·), Count()]), n_steps=3)` returns $\mat{F}^3 \vect{x}$ to 0 and `cycle_count == 3`; a stateless `Sequential` still raises `TypeError` |
| B1 Restart | B1 | two 5-step calls with `obs_clock="state"` give the 10-step analysis of stage 2 to 0 (today 15.4 ppb); with the default the indices are `[0 … 4, 0 … 4]` as today |
| B2 Per-step $\gmat{\Sigma}_\varepsilon$ | B2 | with `obs_err_source` returning $\gmat{\Sigma}_{\varepsilon,t} = (0.1 + 0.01 t) \mat{I}$, the cycle equals a hand-written Kalman filter using the same $\gmat{\Sigma}_{\varepsilon,t}$ to 0 |
| K1 Kalman exactness (existing) | — | stage 2: `DACycle` equals the hand-written filter to 0 in mean and covariance. Passes today |
| K2 EnKF rate (existing) | — | stage 3: the log-log slope of the median RMS gap from $N_{\mathcal{E}} = 200$ to 800 is consistent with the $N_{\mathcal{E}}^{-1/2}$ rate: $-0.5$ lies inside its 99 % interval from a seeded bootstrap over the 10 seeds (4,000 resamples). Passes today: slope −0.61, interval [−0.96, −0.38] |
| K3 Model swap (existing) | — | stage 4: `CallableForward` and `NeuralForward` wrapping the same step give the same analysis to 0. Passes today |
| K4 Retrieval (existing) | — | stage 5: Gauss–Newton converges in at most 10 iterations with $\vert \vect{J}_h^\top (\vect{y} - h(\hat{\alpha})) \vert < 10^{-12}$. Passes today: 6 iterations, 9.9e-20 |
:::

(r10-risks)=
## Sequencing and risks

The two workstreams are independent and small, so they ship together with [Report 13](13_state_catalog.md#r13-ws-d)'s ensemble histories, and the documentation follows ([](#fig-r10-roadmap)).

:::{figure} ../images/reports/r10_roadmap.svg
:label: fig-r10-roadmap
:alt: Phase 1: A1 stateful Sequential steps, B1 the observation clock, and Report 13's D1 ensemble histories, side by side, then gates A1, B1 and R13-D1. Phase 2: the README and the latent design.

Small fixes first, then the documentation and the latent design.
:::

- **A default that changes meaning.** Making `obs_clock="state"` the default would change every existing script that restarts a cycle on purpose with per-call observations. Fallback: we keep `"call"` as the default and warn when a DA cycle starts with `cycle_count > 0` and the default clock.
- **Stateful detection leaks a private attribute.** Workstream A reads `Sequential._is_stateful`. Fallback: we add the public `is_stateful` property in pipekit core in the same release, or duck-type on `_apply_stateful`.
- **Analysis steps that need the state.** Some analyses want the cycle count or the time (adaptive inflation, time-dependent localisation), and today the analysis never sees the state. Fallback: we keep the carrier convention and pass the time through `obs_source`'s output, and only revisit this if two adapters need it (open question 2).
- **Unverified adapters.** filterax, vardax and plumax are not pinned in `references/`, so whether their adapters satisfy the protocols is unchecked. Fallback: we pin one of them and add an adapter test that runs stage 2 with its analysis step.

(r10-open-questions)=
## Open questions

1. **Should a plain `Cycle` advance the clock?** `DACycle` advances `DAState.t` by `dt`, whereas `Cycle` leaves it at 0, so [Report 13](13_state_catalog.md#r13-ws-d) computes valid times from the history index. Advancing `t` when the step is a `ForwardModel` and the state has `t` would make the two agree.
2. **Should `AnalysisStep` receive the state?** Passing `state=` as a keyword would let analyses read the clock, but it changes the protocol that every adapter satisfies.
3. **A cycle checkpoint.** The scoping report's `Checkpoint(cycle_op)` does not exist. A checkpoint is the triple (config, `CarryState.to_dict()`, carrier), so is it a pipekit-cycle operator, a pipekit-experiment artifact, or a script's job?
4. **The window's start state in `SmootherCycle`.** Should the analysis receive the state the window started from, so that a 4D-Var analysis need not carry it inside the carrier?
5. **`CompositeForward` and `dt`.** Summing the components' steps is one reading of composition (operator splitting with sequential sub-steps). Should a composite that advances all components over one shared `dt` exist as well?
6. **The latent design.** Do we build `latent.py` as `docs/design/latent.md` drafts it, or retire the draft in favour of the shipped `LatentMap` and `ReducedOrderModel` protocols?

(r10-todos)=
## Upcoming work

[](#tbl-r10-todos) lists the follow-ups.

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r10-todos

| Follow-up | Where | Why |
|---|---|---|
| Refresh the README for 0.0.2 and fix `filterx` → `filterax` in its quickstart | pipekit-cycle `README.md` | the table lists v0.0.1 and omits `BFNCycle`, the adjoint specs and eleven protocols |
| Document the carrier convention for covariances and ensembles | pipekit-cycle docs | the analysis never sees the state, so $\gmat{\Sigma}_{x,t \mid t}$ and the ensemble belong in the carrier (stage 2) |
| Add an adapter test against one algorithm library, and pin filterax in `references/` and `UPSTREAM` | pipekit-cycle tests; `references/filterax`; `scripts/capabilities.py` | the protocols' main users are unchecked; once filterax is pinned, the notebook's `KalmanAnalysis` and `EnKF` (user-written `AnalysisStep`s, as the report's running example) are replaced by imports of filterax's |
:::

(r10-references)=
## References

### Sources

- The scoping report, "Report 10 — `pipekit-cycle`: time-stepping, DA, and
  observation operators" (supplied by the author).
- pipekit-cycle 0.0.2 in `references/geostack/pipekit/packages/pipekit-cycle`: its
  source, README, CHANGELOG, `docs/design/latent.md` and tests.
- The running example's script and outputs, described in [](#r10-running).
