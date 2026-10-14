---
title: The six-step cycle
subtitle: Report 17 · The data-driven modeling cycle is the benchmark ladder, run on one retrieval
short_title: 17 · The six-step cycle
description: The scoping report names a third organising axis, the six-step cycle from a simple model to an amortized predictor, makes model-based inference the permanent oracle, and proposes a CycleStageBenchmark and a model-role registry. pipekit-evaluate already ships the cycle as its benchmark ladder. A run of all six steps on one methane retrieval shows what the ladder scores today, why each step needs its own reference, and what the registry cannot yet record. This page keeps the cycle and the oracle, drops the new classes, and puts lineage inside the registry's hash.
keywords:
  - six-step cycle
  - oracle
  - emulator
  - amortized inference
  - benchmark ladder
  - lineage
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

A research idea in geoscience usually travels the same road: a forward model
you can simulate, a slow classical inversion of it, a neural emulator of the
forward model, the inversion rerun with the emulator, and finally a network
that maps observations straight to an answer. The scoping report names this
road the **six-step cycle**, makes step 2 (the classical inversion) the
permanent **oracle** that validates steps 3 to 5, and argues that writing the
benchmark for each step is how a team states what it understands. It proposes
a `CycleStageBenchmark`, a model-role enum and a lineage record. This page
runs all six steps on one row of the cycle matrix, the L1 → L2 retrieval of a
methane column from five-band transmittances, on the pinned code. The cycle
is already there: pipekit-evaluate ships it as its **benchmark ladder**, with
six rungs and a reference rule that scores the emulator against the
simulator and both fast inversions against the oracle. The run shows why the
rule matters. An emulator within 3.2 % of the simulator moves the posterior by
up to 1.9 standard deviations; an amortized predictor that is calibrated
against the truth is 55 % too wide against the oracle where the column is
small. What is missing is small: a cell that names a transition, an oracle
strength, a way to score the emulator on parameters, and lineage the registry
cannot overwrite. Every number below is from a real run.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Mostly built. Originally a scoping proposal ("committed": the cycle as a third organising axis, step 2 as oracle, a `CycleStageBenchmark` in pipekit-evaluate v0.2); the cycle and the oracle are kept, the new classes are replaced by the shipped ladder and three small extensions |
| **Reading time** | about 25 minutes |
| **Audience** | anyone taking a method from "we have a forward model" to "we have a deployed predictor", and anyone deciding what a step must reproduce before the next one is trusted |
| **Companions** | [Report 14](14_pipekit_evaluate.md) (pipekit-evaluate and the ladder); [Report 15](15_benchmarking.md) (the benchmark contract); [Report 11](11_pipekit_train.md) (emulators and amortized retrievals on the same sensor); [Report 10](10_pipekit_cycle.md) (the Gauss–Newton retrieval); [Report 12](12_pipekit_experiment.md) (the registry); the [overview](../overview.md#sec-six-steps) and the [benchmarks gallery](../benchmarks_gallery.md) |
| **Code checked against** | pipekit 0.0.2 at `ef3c17f` (pipekit, pipekit-cycle, pipekit-train, pipekit-experiment, pipekit-evaluate and its `benchmark_ladder.md`); xrtoolz 0.0.5 at `308c7f2`; geotoolz 0.10.2 at `088475e`. Read on 14 October 2026 |
| **Running example** | every output is from one run of `notebooks/reports/report17_cycle.py`, which draws the scenes from fixed seeds in memory and asserts each number the page shows |
| **Sources** | the scoping report [@geostack_cycle_draft] and the gallery draft it extends [@geostack_gallery_draft], and pipekit-evaluate's design note on the ladder (`benchmark_ladder.md`) |
:::

:::{important} What changed from the scoping report
The scoping report's **central claims stand**: the six-step cycle orders the
road from a simple model to an amortized predictor, step 2 stays as the
oracle, and every step needs a benchmark against a reference. The running
example reproduces each claim on real numbers. Four parts **do not**. The
**cycle is not new code**: pipekit-evaluate already ships it as the
benchmark ladder (`RUNG_NAMES`, `DEFAULT_STAGED`, `Benchmark`), so a
`CycleStageBenchmark` would duplicate it. The **oracle rule** is stated three
ways in the source; the shipped rule (the emulator against the simulator,
both fast inversions against the oracle) is the consistent one. The
**amortized predictor** is trained on simulations, not on step 4's outputs.
And **lineage in registry tags** can be overwritten, so it moves inside the
hash. A **Differs** box marks each place.
:::

(r17-summary)=
## Summary

**Decision.** The six-step cycle is pipekit-evaluate's benchmark ladder:
rungs 1 to 6 are the steps, and its reference rule is the cycle's oracle
rule. No new class or package. The ladder gains a cell that names a
transition, an oracle strength and rung-3 scoring on the parameters;
pipekit-experiment records every artifact's parents inside its content hash
and answers what depends on an artifact; xrtoolz scores a posterior against
the oracle with weights; and the docs name the cycle once, consistently.

- **A. The cycle cell.** A ladder cell names its transition (L1 → L2), not one
  level, and the ladder's L2 is renamed to what its own description says, a
  retrieval (A1); a contract's oracle carries a strength, from a known forward
  model down to a heuristic, which every report row it scores records (A2);
  the emulator rung is scored on the parameters it emulates, not on the
  observations every other rung receives (A3).
- **B. Lineage in the hash.** An artifact's parents (the simulator, the
  emulator, the oracle it was scored against) are part of its stored payload
  and its hash, so they cannot be overwritten (B1); the registry answers which
  artifacts depend on a given one, so a new emulator marks exactly the stale
  artifacts (B2).
- **C. Scoring against the oracle.** xrtoolz's `wasserstein_1` accepts
  weights, so a posterior on quadrature nodes or importance samples is scored
  without resampling (C1); the amortized rung is scored against the oracle
  and the truth in one run, through [Report 14](14_pipekit_evaluate.md#r14-ws-a)'s
  A4 (C2).
- **D. The cycle in the docs.** The overview's oracle rule is corrected to the
  shipped one (D1); Reports 11, 14 and 15 point to the step each of their
  stages fills (D2).

**End state.** A team writes one contract per cell of its cycle row, runs
the ladder, and reads one report: each rung against the reference it claims
to reproduce, each row tagged with the oracle's strength. When it improves a
step, the registry lists the artifacts that must be rerun, and no others.

(r17-framework)=
## The framework, as the scoping report put it

This section restates the scoping report's framework; the rest of the page
keeps it except where a box says otherwise.

**The six steps.** The scoping report says it follows plumax's framing of
the same steps (plumax is not pinned here), and the
[overview](../overview.md#sec-six-steps) adopted it as the third axis.

```{figure} ../images/overview/six_steps.svg
:label: fig-r17-six-steps
:alt: Five steps in a row, from simple model to amortized inference. Step 2 validates steps 3 to 5. A shared improve step sits under steps 2 to 5 with two-way links.
:width: 100%

**The six-step cycle, as the overview draws it.**

Step 1 is the generative story, step 2 its classical inversion and the
oracle, step 3 an emulator of the forward model, step 4 the inversion rerun
with the emulator, step 5 an amortized predictor, and step 6 improves any of
them with the previous step as ground truth.
```

**Step 2 as oracle.** Classical inversions (the EnKF and 4D-Var in filterax
and vardax, Gaussian processes in pyrox) are not legacy code waiting to be
replaced. They are the only thing a neural surrogate can be validated
against, so they stay runnable on production data, and their compute is a
permanent cost.

**Benchmarks as formalised understanding.** A benchmark contract declares
the carriers, the reference, the metrics, the splits, the baselines and the
expected failure modes. If a team can write all six, it understands the
problem; the one it cannot write is what it has to learn next.

**The matrix.** The cycle applies at every L0–L4 transition, so the levels
and the steps form a matrix whose cells are concrete, benchmarkable methods.

```{figure} ../images/overview/axes_matrix.svg
:label: fig-r17-matrix
:alt: A grid with lifecycle transitions as rows, from calibration to forecast, and steps 1 to 5 as columns. Each cell names a method; the step-2 column is highlighted as the oracle.
:width: 100%

**The cycle matrix, as the overview draws it.**

The running example is one cell of each column in the L1 → L2 retrieval
row.
```

**Physics-first and data-first.** In a physics-first cycle step 1 is a known
forward model and step 2 a classical inversion of it, an oracle in the
strong sense. In a data-first cycle step 1 is a labelled dataset and step 2
a classical supervised baseline, an oracle in the weak sense: matching it
does not make a method right if the labels are wrong.

:::{warning} Differs from the plan
The scoping report proposes the cycle as new structure. pipekit-evaluate
0.0.2 already ships it: `RUNG_NAMES` names the six steps
(`benchmark/cube.py`), `DEFAULT_STAGED` encodes which reference each rung is
scored against (`benchmark/reference.py:41`), and `Benchmark` runs the
estimators and scores each against its reference. Its design note,
`benchmark_ladder.md`, draws the same staircase and adds a second axis, the
**complexity** of each rung (a MAP fit below a full posterior, a linear
emulator below a neural operator), with the same rule along it. The
overview's [three axes](../overview.md#sec-axes) already include the cycle, so the scoping report's edits to a vision document and a
"Report 0" land there; neither file exists in this repository.
:::

(r17-ownership)=
## Ownership rules

The cycle is orchestrated by pipekit-evaluate's ladder; each step runs in the
library that owns its kind of computation.

:::{figure} ../images/reports/r17_ownership.svg
:label: fig-r17-ownership
:alt: Domain projects run the steps and read cycle reports through pipekit-evaluate, which ships RUNG_NAMES, DEFAULT_STAGED, Cell, Benchmark and Estimator and would add transition cells, an oracle strength and rung-3 scoring on parameters. Below it, pipekit-cycle runs steps 1, 2 and 4, pipekit-train trains steps 3 and 5, xrtoolz scores posteriors against the oracle, and pipekit-experiment keeps parents inside the hash. Domain libraries supply physics and classical oracles through adapters.

The ladder in pipekit-evaluate is the cycle; the steps run where their
computation lives.
:::

1. **pipekit-evaluate** owns the cycle's vocabulary (rungs, levels,
   transitions, oracle strength), the reference rule and the runner. It
   imports pipekit-cycle only through an optional extra today (`cycle`), and
   would import pipekit-experiment and xrtoolz the same way.
2. **pipekit-cycle** owns steps 1, 2 and 4 as operators: the forward model
   (`CallableForward`, `NeuralForward`), the iterative inversion
   (`Recurrence`) and the assimilation cycles (`DACycle`). It never imports
   pipekit-evaluate.
3. **pipekit-train** owns steps 3 and 5: training data drawn from the step-1
   simulator (`SimulationDataset`) and the training loop. It never parses a
   contract or a cell.
4. **xrtoolz** owns the scores that compare two distributions, including a
   posterior against the oracle's (`wasserstein_1`), beside its other
   probabilistic metrics.
5. **pipekit-experiment** owns persistence and lineage: every artifact stored
   by content hash with its parents in the payload. It never interprets a
   rung.
6. **Domain libraries** (plumax, somax, rtmx for steps 1; filterax, vardax,
   pyrox for step 2) own the physics and the classical oracles, and reach
   pipekit only through `adapters/pipekit.py`.

(r17-running)=
## The running example: one retrieval, six steps

**The problem.** Retrieve a methane column enhancement from a multiband
sensor, and take the retrieval through all six steps of the cycle: the cell
of the matrix the scoping report works through in its section 4.1. The
sensor is Reports 10 and 11's, so their results carry over.

**The data.** A sensor measures transmittances in B = 5 bands. With α the
column enhancement, k ∈ ℝ⁵ the band absorption coefficients, k = (0.002,
0.004, 0.006, 0.008, 0.010), and ε the noise, the generative model is

$$
y_b = f_b(\alpha) + \varepsilon_b , \qquad f_b(\alpha) = e^{-k_b \alpha} , \qquad \varepsilon_b \sim \mathcal{N}(0, \sigma^2) , \qquad \alpha \sim \mathcal{U}(0, 300) ,
$$ (r17-eq-model)

with σ = 10⁻³. The test set is n = 500 scenes drawn from the prior with a
fixed seed. This is the L1 → L2 row: structured observations (transmittances)
in, a retrieved geophysical variable (α) out.

**What exists.** Every step runs on the pinned code except where a line is
marked `user`: the ladder's task, estimates and scorers are user code, as in
[Report 14](14_pipekit_evaluate.md#r14-stage-4), and the polynomial emulator
and the Gaussian head are small least-squares fits, so that the whole cycle
runs in under a minute and every number is checkable. Report 11 trains an
amortized retrieval for the same sensor with Equinox through `pipekit-train`
([its stage 6](11_pipekit_train.md#r11-stage-6)); the cycle does not depend
on which kind of model fills a step.

(r17-stage-1)=
### Stage 1 · The simple model, as a ladder task (pipekit-cycle → pipekit-evaluate)

**TL;DR.** Write the forward model as a pipekit-cycle operator and the
retrieval as a ladder task, and read the rule the ladder already applies to
each step.

**Problem.** A ladder `Task` bundles a prior, the step-1 simulator, an
observation operator and the data each rung is scored against: the known α
for the oracle, the simulator's outputs f(α) for the emulator, and the
observations y that every estimator receives.

```{code-block} python
:caption: The step-1 model and the task, and the rule the ladder applies to each rung, from a real run.
:linenos:

# Shapes: B = 5 bands; n = 500 test scenes
import pipekit_cycle as pc
from pipekit_evaluate.benchmark import (DEFAULT_STAGED, LEVEL_NAMES,
                                        RUNG_NAMES, Cell, Task)

# f_b(α) = exp(−k_b α)
forward = pc.CallableForward(lambda a, dt: np.exp(-K * a),
                             dt=1.0)                # (,) → (B,)    today
task = RetrievalTask()      # prior 𝒰(0, 300), simulator, obs     user
isinstance(task, Task)      # → True                              today
task.datasets()             # θ (n,), f(θ) (n, B), y (n, B)        user

RUNG_NAMES        # {1: simple_model, 2: model_based_inference,
                  #  3: emulator, 4: emulator_based_inference,
                  #  5: amortized_predictor, 6: improve}      today
DEFAULT_STAGED    # {1: none, 2: known_theta, 3: simulator_output,
                  #  4: oracle_posterior, 5: oracle_posterior,
                  #  6: best_validated_below}                 today
LEVEL_NAMES["L2"]           # → "gap_filled_obs"              today
Cell("L2", rung=4).label    # → "L2/r4/c0": not "L1 → L2"     today
```

The six steps are the ladder's six rungs, and its default rule is the
cycle's: the oracle (rung 2) against the known truth, the emulator (rung 3)
against the simulator it replaces, and both fast inversions (rungs 4 and 5)
against the oracle's posterior. Two things do not fit the cycle matrix. A
cell names one level, so the L1 → L2 retrieval, the L2 → L3 gap filling and
the L3 → L4 forecast cannot be told apart once two rows share an output
level. And the ladder names L2 "gap_filled_obs" while describing it, in the
same design note, as "per-pixel geophysical retrievals (gaps allowed)", which
is what the overview's levels call L2 (workstream A1).

:::{warning} Differs from the plan
The scoping report's `CycleStageBenchmark` carries `target_step`,
`oracle_step` and `l_transition`. The first two are the ladder's `rung` and
its reference rule, which already says which step is the reference; a
second field would let the two disagree. Only the transition is new.
:::

(r17-stage-2)=
### Stage 2 · The oracle, and a cheaper inversion below it (pipekit-cycle)

**TL;DR.** Compute the exact posterior for every scene, check it against the
truth, and compare it with a Gauss–Newton fit one step down the complexity
axis.

**Problem.** With the flat prior of {eq}`r17-eq-model`, the posterior is

$$
p(\alpha \mid y) \propto \mathbb{1}[0 \le \alpha \le 300] \prod_{b=1}^{5} \mathcal{N}\big(y_b ;\, f_b(\alpha),\, \sigma^2\big) .
$$ (r17-eq-posterior)

The **oracle** evaluates it on G = 30,001 nodes over [0, 300], a step of
0.01 against a smallest posterior standard deviation of 0.062, so it is
exact to quadrature. No unbiased estimator can beat the Cramér–Rao bound
σ_CR(α) = σ / ‖J(α)‖, with J_b = −k_b exp(−k_b α) the derivative of f_b. The
**Laplace** step below it fits α̂ by Gauss–Newton and reports the width
σ / ‖J(α̂)‖.

```{code-block} python
:caption: The oracle by quadrature and the Laplace step as a Recurrence, scored against the truth, from a real run.
:linenos:

# Shapes: n = 500 scenes; G = 30,001 nodes; B = 5
# log p(α_g | y) = −½ Σ_b (y_b − f_b(α_g))² / σ² + c, normalised
oracle = grid_posterior(y, f)                 # (n, B) → (n, G)   user
# α ← α + Jᵀ(y − f(α)) / JᵀJ  until |Δα| < 10⁻⁸
gn = pc.Recurrence(
    GaussNewtonStep(y),                                         # user
    condition_op=Lambda(lambda s: s.residual_norm < 1e-8),
    max_iters=50)                             # (n,) → (n,)       today
a_hat, _ = gn(a0, pc.IterationState())                        # today
# σ_L = σ / ‖J(α̂)‖
laplace = GaussEstimate(a_hat, SIG / norm(J(a_hat)))          # user
# RMSE vs α: oracle 0.270, Laplace 0.270; RMS Cramér–Rao 0.269
# 95 % coverage vs α: oracle 0.960, Laplace 0.958 (s.e. 0.010)
# Laplace mean within 0.1 oracle widths of the oracle's, every scene
```

The oracle sits on the Cramér–Rao bound and covers the truth at the nominal
rate, so on this problem step 2 is an oracle in the strong sense: the
forward model is known and the posterior is computed, not approximated. The
Laplace step agrees with it to a tenth of a standard deviation everywhere,
which is the complexity axis doing its job: a richer inversion must first
reproduce the simpler one where both are valid.

:::{warning} Differs from the plan
The scoping report calls step 2 "slow but exact". The overview already
corrects this ("the most trusted reference available, not because it is
exact"): a finite-ensemble EnKF or a 4D-Var is an approximation. This
example is one of the few where step 2 is exact, which is what lets the
later stages measure the other steps' errors without ambiguity. For the
methane plume case the source names, step 2 is a matched filter, a
classical method with known biases (`geotoolz.matched_filter.MatchedFilter`
ships today; the source's `CH4MatchedFilter` does not), so its oracle is a
weaker one.
:::

(r17-stage-3)=
### Stage 3 · Two emulators of the forward model (pipekit-train → numpy)

**TL;DR.** Fit two polynomial emulators to simulations of the forward model
and score them against the simulator, the reference rung 3 calls for.

**Problem.** An emulator f̂ of degree d writes each band as a polynomial in
α / 300 with coefficients C ∈ ℝ^{(d+1)×5}, fitted on the training split of
a `SimulationDataset` of 2,000 seeded draws:

$$
C^\star = \arg\min_C \sum_{i=1}^{1600} \big\lVert V(\alpha_i)\, C - f(\alpha_i) \big\rVert^2 , \qquad V(\alpha) = \big[(\alpha/300)^d, \ldots, \alpha/300, 1\big] .
$$ (r17-eq-emulator)

```{code-block} python
:caption: Training data from the step-1 simulator, two emulators, and their error against the simulator, from a real run.
:linenos:

# Shapes: 1,600 training draws; B = 5; C (d + 1, B)
sim = pt.SimulationDataset(forward, PriorDraw(), n_samples=2000,
                           seed=0)                            # today
len(sim)                       # → 1600: the train split      today
# C = argmin ‖V C − F‖²
C4 = fit_emulator(4)           # (1600, 5) → (5, B)           user
C6 = fit_emulator(6)           # (1600, 7) → (7, B)           user
# max_b,α |f̂_b(α) − f_b(α)| / σ over 3,001 values of α
# degree 4: 2.2 σ (3.2 % of the transmittance at worst)
# degree 6: 0.03 σ
# on the 500 test scenes only: degree 4, 2.13 σ (stage 6's figure)
```

Judged as a forward model, the degree-4 emulator looks good: its worst error is 3.2 % of the
transmittance. Judged in the unit the problem fixes, the noise σ, it is
2.2 σ off, and the next stage shows what that costs. This is the scoping
report's claim about benchmarks made concrete: the tolerance in a step-3
contract cannot be written until one knows the noise of the step-4 problem,
so writing it is the act of understanding it.

(r17-stage-4)=
### Stage 4 · The oracle's inversion, rerun with each emulator (pipekit-cycle → xrtoolz)

**TL;DR.** Rerun the oracle's quadrature with each emulator in place of the
simulator and score the result against the oracle's posterior.

**Problem.** Step 4 replaces f with f̂ in {eq}`r17-eq-posterior`. With F̂ and
F the distribution functions of the step-4 and oracle posteriors, the
1-Wasserstein distance

$$
W_1 = \int_0^{300} \big\lvert \hat{F}(\alpha) - F(\alpha) \big\rvert \, d\alpha
$$ (r17-eq-w1)

is zero only when the two posteriors agree; reported in units of the
oracle's standard deviation s, it reads as "how many widths apart".

```{code-block} python
:caption: Emulator-based inference against the oracle, exactly and through xrtoolz, from a real run.
:linenos:

# Shapes: n = 500 scenes; G = 30,001 nodes; 2,000 draws per scene
fast = grid_posterior(y, f_hat_4)             # (n, B) → (n, G)   user
# (m̂ − m) / s, the mean shift in oracle widths
# degree 4: RMS 0.45, max 1.89; width ratio 1.00; W₁ / s 0.38
# degree 6: max 0.02
# 95 % coverage vs α: degree 4 0.928, degree 6 0.960 (the oracle's)
# W₁ = ∫ |F̂ − F| dα, from samples
xrtoolz.metrics.wasserstein_1(
    fast.sample(2000), oracle.sample(2000))   # (n, 2000) → (n,)  today
#   each sample a (scene, member) DataArray     arguments sketched
# → 0.38 s; the oracle against itself 0.04 s: the floor at 2,000
```

A 3 % emulator moves the posterior mean by up to 1.9 widths and pulls the
coverage of the truth from 96 % to 93 %, while the width of the posterior
does not change at all: a check on the uncertainty alone would pass. The
degree-6 emulator, at 0.03 σ, reproduces the oracle to 0.02 widths. Scoring
step 4 against the oracle, as the ladder's rule does, is what makes this
visible; scored against the truth alone, the degree-4 inversion is a
retrieval with slightly low coverage, and nothing says why. xrtoolz's
`wasserstein_1` reproduces the exact W₁ but takes samples only, so an exact
posterior on nodes must be sampled first, and the sampling floor (0.04
widths at 2,000 draws) hides any distance below it, including the degree-6
emulator's 0.004 (workstream C1).

(r17-stage-5)=
### Stage 5 · An amortized predictor, against the truth and the oracle (numpy → xrtoolz)

**TL;DR.** Fit one Gaussian predictor for every spectrum, and see that
scoring it against the truth and scoring it against the oracle give
different verdicts.

**Problem.** The predictor is q(α | y) = 𝒩(μ(y), s(y)²), with μ and log s²
linear in features φ(y) ∈ ℝ⁸ built from the per-band inversions
z_b = −log y_b / k_b. Both are fitted once, by least squares on 20,000
simulated pairs (α, y) drawn from {eq}`r17-eq-model`:

$$
\beta_\mu = \arg\min_\beta \sum_i \big(\alpha_i - \varphi(y_i)^\top \beta\big)^2 , \qquad
\beta_s = \arg\min_\beta \sum_i \big(\log r_i^2 - \varphi(y_i)^\top \beta\big)^2 ,
$$ (r17-eq-head)

with r_i = α_i − φ(y_i)ᵀβ_μ, and log s(y)² = φ(y)ᵀβ_s + 1.27, since
𝔼[log χ²₁] = −1.27. A predictor calibrated against the truth has
𝔼[z²] = 1 for z = (α − μ) / s.

```{code-block} python
:caption: The amortized Gaussian head, scored two ways, from a real run.
:linenos:

# Shapes: 20,000 training pairs; φ(y) (n, 8); n = 500 test scenes
# φ(y) = [1, z₁ … z₅, z̄², z̄³],  z_b = −log y_b / (300 k_b)
# μ = φᵀβ_μ,  log s² = φᵀβ_s + 1.27
q = amortized(y)                      # (n, B) → μ (n,), s (n,)   user
# against α: RMSE 0.281; 𝔼[z²] 0.98; 95 % coverage 0.960
# against the oracle: width ratio 1.14; 1.55 for α < 50 (72 scenes)
# mean shift for α < 50: 1.24 widths RMS, 3.7 at worst; W₁ / s 0.43
```

Against the truth, the predictor is calibrated: 𝔼[z²] = 0.98 and 96 %
coverage. Against the oracle, it is a different estimator. Where the column
is small and the posterior narrow, its mean is noisier than the oracle's
(1.24 widths off) and it widens its intervals by 55 % to stay honest about
that. The truth only checks that the predictor knows how wrong it is on
average over the prior; the oracle checks that it reproduces the posterior
scene by scene, which is the question an amortized predictor must answer.
The ladder scores rung 5 against the oracle only, so today the two verdicts
need two runs ([Report 14](14_pipekit_evaluate.md#r14-stage-4)'s A4, gate C2
here).

:::{warning} Differs from the plan
The scoping report's L1 → L2 table trains the step-5 predictor "on Step 4
outputs", while its own section 1 and the overview's equation train it on
pairs drawn from the step-1 simulator. Stage 4 shows why the simulator is the
right source: step-4 outputs from the degree-4 emulator are up to 1.9 widths
off, and a predictor trained on them would learn the shift. Training on the
simulator also keeps step 5 independent of step 3, which stage 6 relies on.
:::

(r17-stage-6)=
### Stage 6 · The ladder and the lineage (pipekit-evaluate → pipekit-experiment)

**TL;DR.** Run every rung through the shipped ladder, store every artifact in
the registry with its parents, improve the emulator, and ask what must be
rerun.

**Problem.** **(a) The ladder.** Five estimators fill five cells at level L2:
Laplace (rung 2, complexity 0), the quadrature oracle (rung 2, complexity
1), the degree-4 emulator (rung 3), its inversion (rung 4) and the amortized
head (rung 5). **(b) The lineage.** Step 6 replaces the degree-4 emulator
with the degree-6 one. An artifact is stale when it was built from the old
emulator, directly or through another artifact; the set of stale artifacts
is the set of descendants of the old emulator in the graph of parents.

```{code-block} python
:caption: The cycle through the shipped runner and registry, from a real run.
:linenos:

# Shapes: n = 500; spectra (n, B) in σ; posteriors in oracle widths
report = pe.Benchmark(task=task, scorers=[Score()], level="L2",
    estimators=[Rung(2, 0, laplace), Rung(2, 1, oracle),
                EmulatorRung(3, 0, f_hat_4), Rung(4, 0, fast_4),
                Rung(5, 0, amortized)]).run()    # Rung: user     today
# L2/r2/c0  known_theta       max |α̂ − α| 1.23, RMSE 0.270
# L2/r2/c1  known_theta       max |α̂ − α| 1.23, RMSE 0.270
# L2/r3/c0  simulator_output  max |f̂ − f| 2.13 σ, RMSE 0.294 σ
# L2/r4/c0  oracle_posterior  max shift 1.89, width ratio 1.00
# L2/r5/c0  oracle_posterior  max shift 3.70, width ratio 1.14
EmulatorRung.fit(task)       # reads θ: rung 3 needs θ, not y  user

reg = px.LocalModelRegistry(root)                             # today
h1 = reg.store(Artifact("simulator", {...}),
               tags={"rung": 1})                              # today
h3 = reg.store(Artifact("poly_emulator", {...}, parents=[h1]),
               tags={"rung": 3, "oracle": h1})                # today
# … rungs 2, 4 and 5 likewise; then step 6: the degree-6 emulator
# → new hashes for rungs 3 and 4; rung 5's unchanged (parent h1)
dependents(h3)       # → [old rung 4]: a scan of configs      user
reg.store(emulator_6, tags={"oracle": "x"})
# → the same hash; its oracle tag is now "x"                  today
reg.list(tags={"parents": [h1]})
# → []: parents are config, not tags                          today
```

The ladder runs the whole cycle in one call and resolves each rung's
reference as stage 1 read it. After the improvement, rerunning it changes the
rung-3 and rung-4 rows (to 0.03 σ and a 0.02-width shift) and leaves rung 5's
unchanged, because rung 5 was trained on the simulator. Three things needed
glue. Rung 3 is a forward emulator, a map from α to y, but the runner calls
every estimator on the observations, so the emulator reads α from the task
inside `fit` (workstream A3). The registry can carry lineage two ways, and
neither is enough: tags are outside the hash, so storing the same emulator
again with another oracle tag silently replaces it, and a tag filter matches
whole values, so it cannot ask which artifacts list a given parent. Parents
in the configuration are hashed, which is right, and that is why a new
emulator gave rung 4 a new hash; but nothing indexes them, so "what depends
on this?" is a scan of every stored configuration (workstream B). And a
configuration that holds a mapping cannot be loaded back: `from_state`
refuses non-primitive values (`pipekit/_base/operator.py:251`).

:::{warning} Differs from the plan
The scoping report adds a `ModelRole` enum (oracle, emulator, production,
predictor, baseline) and an `ArtifactLineage` with an oracle hash, parents,
the training run and benchmark results. The roles are the ladder's rung
names already (`RUNG_NAMES`), so a rung tag is enough; a baseline is a lower
step on the complexity axis, not a role. The lineage is kept, but inside the
stored payload and its hash rather than in tags, for the reason stage 6
shows. The source's `pe.MultiTrackEvaluation` and `report.diff` do not exist;
[Report 15](15_benchmarking.md#r15-ws-d) drops the first, and comparing two
runs is a comparison of two reports' rows.
:::

(r17-names)=
## The scoping report's names, checked

:::{table} Every name the scoping report uses, against the pinned code.
:label: tbl-r17-names

| Scoping report | At the pinned revisions |
|---|---|
| the six steps; step 2 as oracle | `RUNG_NAMES` and `DEFAULT_STAGED` in `pipekit_evaluate.benchmark` (stage 1) |
| `CycleStageBenchmark(target_step, oracle_step, l_transition, oracle_artifact_hash, oracle_strength)` | does not exist; `Cell(level, rung, complexity)` and `ReferenceRule` cover the step and its reference; transition, oracle hash and strength are new (A1, A2) |
| `CycleBenchmark.run_full_cycle` | does not exist; `Benchmark(...).run()` scores every rung in one call (stage 6) |
| `OracleStrength` | does not exist; `ReferenceSource` says where a reference comes from, not how far to trust it |
| `ModelRole`, `ArtifactLineage` | do not exist; `LocalModelRegistry.store(..., tags=)` stores free tags outside the hash (stage 6) |
| `pc.DACycle(forward_model, obs_op, analysis_step, n_steps)` | exists, with `obs_source` and `save_history` as well (`da.py:84`) |
| `pc.NeuralForward(model_op, dt)` | exists (`forward.py:136`) |
| `pc.Recurrence` for iterative retrieval | exists (`cycle.py:265`); stage 2 runs Gauss–Newton with it |
| `pt.SimulationDataset(forward_model, prior, n_samples)`, `pt.CachedDataset(source, cache_dir)` | exist (`dataset.py:228`, `:393`) |
| `pt.TrainingLoop(..., n_epochs=, backend="lightning")` | `TrainingLoop` counts `max_steps`, not epochs; the Lightning backend raises `NotImplementedError` (Report 11) |
| `pt.MSE`, `pt.NLL` | exist (`loss.py`) |
| `pa.ModelOp` | no such alias; the Equinox adapter's `EquinoxModelOp` or pipekit-jax's `JaxModelOp` |
| `px.S3ModelRegistry` | exists (`registry.py`) |
| `pe.MultiTrackEvaluation`, `EvaluationReport.diff` | do not exist; Report 15 drops the first, Report 14 proposes the report (A3) |
| `geotoolz.matched_filter.CH4MatchedFilter` | `MatchedFilter`, `MatchedFilterPixel`, `MatchedFilterSNR` exist; no CH₄-specific class |
| BAEMR retrieval | not identified; the [gallery](../benchmarks_gallery.md#gal-3-1) found the same |
| `plumax.PlumeForward`, `ColumnObs`, `filterx…EnKFAnalysis`, `pyrox_nn…ConditionalNormalizingFlow`, `RTMX.RTM` | not pinned in `references/`, so not checked; the scoping report's `filterx` is filterax |
| "the 23 benchmarks" of the gallery | 18 entries; 23 contracts once entry 5.1 is split into its six ([gallery](../benchmarks_gallery.md#gal-5-1)) |
:::

(r17-audit)=
## Current-state audit

Four of the eight symbol groups need action. Read on 14 October 2026 at the
revisions in the report's header, on `main` of each submodule.

:::{table} The cycle's surface, by symbol group.
:label: tbl-r17-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `RUNG_NAMES`, `LEVEL_NAMES`, `Cell` | pipekit-evaluate `benchmark/cube.py` | `Benchmark`, Reports 14 and 15 | **Split ownership.** A cell names one level, not a transition; L2 is named `gap_filled_obs` but described as retrievals (`cube.py:28`) |
| `DEFAULT_STAGED`, `DEFAULT_COMPLEXITY`, `ReferenceRule`, `ReferenceSource` | `benchmark/reference.py` | `Benchmark` | **Healthy.** The cycle's oracle rule, as data a domain can override; no notion of how strong the oracle is |
| `Benchmark`, `CellResult` | `benchmark/run.py` | stage 6 | **Split ownership.** Every estimator is called on the observations (`run.py:108`), so rung 3 needs glue; one reference per cell (Report 14) |
| `Estimator`, `Estimate`, `Task`, `Oracle` | `benchmark/protocols.py` | stage 6 | **Healthy.** Enough for every rung once rung 3 is called on θ; `Oracle` is still not consumed (Report 14) |
| `CallableForward`, `NeuralForward`, `Recurrence`, `DACycle` | pipekit-cycle `forward.py`, `cycle.py`, `da.py` | stages 1, 2 and 6; Reports 10 and 11 | **Healthy.** Steps 1, 2 and 4 as operators; an emulator drops in through `NeuralForward` |
| `SimulationDataset`, `TrainingLoop` | pipekit-train `dataset.py`, `loop.py` | stage 3; Report 11 | **Healthy** for steps 3 and 5; the dataset hash ignores the forward model (Report 11's A3), so it cannot name step 1 as a parent |
| `LocalModelRegistry.store`, `list`, `tag` | pipekit-experiment `registry.py:113`, `:180` | stage 6; Report 12 | **Split ownership.** Tags are outside the hash and merged on re-store (`:159`); `list` matches whole tag values (`:195`); no reverse lookup |
| `wasserstein_1`, `Wasserstein1` | xrtoolz `metrics/_src/distributional.py:144` | stage 4 | **Healthy.** Samples only: no weights, so a posterior on nodes is resampled and floored by sampling noise |
:::

Smaller findings:

- **A dictionary in an operator's configuration does not reload.**
  `Operator.from_state` refuses non-primitive configuration values
  (`pipekit/_base/operator.py:251`), so an artifact whose configuration nests
  its settings stores but cannot be loaded back (stage 6). Flat keys or a
  JSON string work.
- **Skipping step 3 needs nothing.** A run with no rung-3 estimator simply
  has no rung-3 row; a run with no rung-2 estimator reports "no oracle
  (rung-2) estimator available" on rungs 4 and 5 (`run.py`), which is the
  honest answer. The scoping report's open question 3 is answered by the code.
- **The ladder's design note already plans the cycle report.** Its build
  order ends in "the level × rung × complexity `BenchmarkReport` + DVC
  wiring", and names the plume pilot as the first slice because "every rung
  already exists" (`benchmark_ladder.md` §7).

(r17-ws-a)=
## Workstream A · The cycle cell

**Decision.** A cell of the cycle matrix is a ladder cell with a transition,
and a contract names its oracle and how far to trust it; nothing else about
the ladder changes.

:::{table} Workstream A's moves.
:label: tbl-r17-move-a

| Symbol | Destination | Reason |
|---|---|---|
| `Cell(level, rung, complexity, source=None)` with label `"L1→L2/r4/c0"` | `benchmark/cube.py` | stage 1: one level cannot tell retrieval, gap filling and forecasting apart |
| `LEVEL_NAMES["L2"] = "retrieved_obs"` | `benchmark/cube.py` | the ladder's own description of L2, and the overview's |
| `OracleStrength` (`physics_derived`, `classical_method`, `human_labels`, `heuristic`) | `benchmark/reference.py`, beside `ReferenceSource` | the scoping report's strong and weak oracles; the reference rule says where, not how far to trust |
| `oracle: {artifact, strength}` on `BenchmarkContract`; `strength` on each `CellResult` scored against it | Report 15's contract; `benchmark/run.py` | a report row says what its reference was worth; a `heuristic` or `human_labels` oracle warns and still scores |
| rung 3 called on `task.datasets()[truth_key]` | `Benchmark.run` | stage 6: a forward emulator maps θ to y; the `fit` workaround goes |
:::

### A1 · Transition cells

A cell gains an optional source level, so a contract's task (Report 15's
`"L1->L2"`) and its cell use one vocabulary; without one, a cell is read as
today. The ladder's L2 is renamed to match its description; Report 15 and
the gallery use "L1 → L2" for gap filling, which on the renamed levels is
L2 → L3, and that relabelling is an open question below, not a silent change.

### A2 · Oracle strength

A contract names its oracle by artifact hash and strength; the runner copies
the strength onto every row it scores against that oracle and warns, without
refusing, when the strength is `human_labels` or `heuristic`. The scoping
report leans the same way (its open question 1).

### A3 · The emulator rung

`Benchmark` passes the truth, not the observations, to a rung-3 estimator,
and scores it against the simulator's outputs as today.

```{code-block} python
:caption: The emulator rung and an oracle with a strength, before and after workstream A.
:linenos:

# Shapes: θ (n,); f(θ) (n, B)
# Today:
EmulatorRung.fit(task)          # stores θ itself              user
pe.Benchmark(task, estimators, scorers, level="L2").run()     # today
# → "L2/r3/c0", ref simulator_output; no strength on any row
# After:
pe.Benchmark.from_contract(c, estimators=[...]).run()
# → "L1→L2/r3/c0": f̂(θ) against f(θ), 2.13 σ  proposed (A1, A3)
# c.oracle = {"artifact": h2, "strength": "physics_derived"}
# → rows r4 and r5 carry "physics_derived"        proposed (A2)
```

(r17-ws-b)=
## Workstream B · Lineage in the hash

**Decision.** An artifact's parents are part of what the registry hashes,
and the registry answers which artifacts descend from a given one.

:::{table} Workstream B's moves.
:label: tbl-r17-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `store(op, *, parents=(), ...)` writing `lineage.json` into the hashed payload | `pipekit_experiment.registry` | stage 6: a tag can be replaced without changing the hash |
| `ArtifactLineage(parents, oracle, training_artifact)` | `pipekit_experiment.artifacts`, beside `TrainingArtifact` | the scoping report's record, minus the benchmark results, which point the other way (a report names the model, Report 15's E2) |
| `dependents(hash)`, `descendants(hash)` | `ModelRegistry` protocol and both registries | stage 6: "what must be rerun?" is a scan today |
| a rung tag (`{"rung": 3}`) by convention | the ladder's docs | the role the scoping report's `ModelRole` would carry; `list(tags=)` filters it today |
:::

### B1 · Parents in the payload

A stored artifact's hash covers its operator state, its weights and its
lineage record. Two artifacts with the same weights and different parents
are different artifacts; nothing can rewrite a parent after the fact.
`TrainingArtifact.dataset_hash` becomes one parent among others, and Report
11's A3 (the simulator in the dataset hash) is what makes that parent
trustworthy.

### B2 · Dependents

The registry keeps a reverse index from parent to child, written at store
time, so `descendants(h)` is the stale set when h is replaced.

```{code-block} python
:caption: Asking what an improvement makes stale, before and after workstream B.
:linenos:

# Shapes: hashes of 64 hex characters
# Today:
reg.store(emu_6, tags={"oracle": h1})
reg.store(emu_6, tags={"oracle": "x"})
# → the same hash; the oracle tag is now "x"                  today
dependents(h3)              # scan every operator.json         user
# After:
h3 = reg.store(emu_4, parents=[h1])                 # proposed (B1)
reg.store(emu_4, parents=[h_other]) != h3           # proposed (B1)
reg.descendants(h3)         # → [h4]; rung 5 has parent h1
                            #                       proposed (B2)
```

(r17-ws-c)=
## Workstream C · Scoring against the oracle

**Decision.** The distance from a posterior to the oracle's is an xrtoolz
metric that takes weights, and the amortized rung is scored against the
oracle and the truth in one run.

:::{table} Workstream C's moves.
:label: tbl-r17-move-c

| Symbol | Destination | Reason |
|---|---|---|
| `wasserstein_1(a, b, *, weights_a=None, weights_b=None, ...)` | `xrtoolz.metrics` | stage 4: an exact posterior on nodes is resampled today, and the sampling floor (0.04 widths at 2,000 draws) hides smaller distances |
| two references for rung 5 (`oracle_posterior` and `known_theta`) | Report 14's A4 in `benchmark/run.py` | stage 5: the truth says "calibrated", the oracle says "55 % too wide where α is small"; both belong on the row |
:::

```{code-block} python
:caption: Distance to the oracle with weights, after workstream C.
:linenos:

# Shapes: nodes (G,); weights (n, G)
# W₁ = ∫ |F̂ − F| dα
xrtoolz.metrics.wasserstein_1(
    nodes, nodes, weights_a=fast.w,
    weights_b=oracle.w)            # (n, G) → (n,); 0.38 s  proposed (C1)
```

(r17-ws-d)=
## Workstream D · The cycle in the docs

**Decision.** The overview states the shipped oracle rule, and the reports
that already ran steps of the cycle say which. Both land with this page.

1. **Correct the overview's rule (D1).** The overview said every cell "uses
   the column before it as its oracle"; the shipped rule scores the emulator
   against the simulator and both fast inversions against step 2. This
   change corrects the sentence, and replaces the overview's planned
   `ModelRole`, `ArtifactLineage` and `CycleStageBenchmark` with workstreams
   A and B.
2. **Link the reports to the steps (D2).** Report 11's stage 2 is step 3 of a
   forecast row (its stage 3 runs that emulator forward, a forecast, not an
   inversion) and its stages 6 and 7 are steps 5 and 2 of this retrieval; Report 14's stage 4 fills rungs 2 and 5; Report 15's contract
   is what each cell is benchmarked by. This change adds one sentence to each
   that points here; the scoping report's 30 to 80 lines per report would
   repeat this page.

(r17-gates)=
## Testing gates

Each workstream merges only when its gates pass on the running example.

:::{table} Gates, by workstream.
:label: tbl-r17-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Transition cell | A | `Cell("L2", 4, source="L1").label == "L1→L2/r4/c0"`, a cell without a source keeps today's label, and `LEVEL_NAMES["L2"] == "retrieved_obs"` |
| A2 Oracle strength | A | stage 6's rows for rungs 4 and 5 carry `physics_derived`; with a `heuristic` oracle the same run emits one `UserWarning` and reproduces the same scores |
| A3 Emulator rung | A | stage 6's rung-3 estimator, with no `fit` workaround, scores 2.13 σ (degree 4) and 0.03 σ (degree 6) against the simulator |
| B1 Lineage hash | B | storing the stage-6 emulator with another parent gives another hash, and with the same parent and other tags the same hash |
| B2 Dependents | B | after stage 6's improvement, `descendants(h3)` is exactly the old rung-4 artifact, and the rung-5 artifact appears in no descendant set of an emulator |
| C1 Weighted W₁ | C | with weights, `wasserstein_1` reproduces stage 4's exact W₁ / s (0.38 for degree 4, 0.004 for degree 6) to 10⁻³, and is 0 for a posterior against itself |
| C2 Two references | C | one run reports rung 5's width ratio against the oracle (1.14) and its 𝔼[z²] against the truth (0.98) on the same row |
| D1 Docs | D | the overview's oracle sentence matches `DEFAULT_STAGED`, and `myst build --html` reports no warnings |
| D2 Links | D | Reports 11, 14 and 15 each link to this page (its running example, stage 6 and workstream A), and every link resolves in `myst build` |
:::

(r17-sequencing)=
## Sequencing and risks

The docs and the vocabulary first, because they cost nothing and every later
change uses them; then lineage and scoring; then a real cycle.

:::{figure} ../images/reports/r17_roadmap.svg
:label: fig-r17-roadmap
:alt: Phase 1: D1 and D2, the corrected oracle rule and the links, beside A1 and A2, transition cells and oracle strength; then gates D1, D2, A1, A2. Phase 2: B1 and B2, parents in the hash and the dependents query, beside A3, C1 and C2, rung 3 on parameters, weighted W1 and two references; then gates B1, B2, A3, C1, C2. Phase 3: a real methane L1 to L2 cycle with pinned radiative-transfer and plume code; then every gate rerun on real data.

Name the cycle, then record its lineage and score it, then run it on real
data.
:::

- **The oracle is expensive to keep.** Step 2 has to stay runnable on new
  data to score new submissions. Fallback: run it on a fixed, versioned
  validation set and at a lower cadence than production, and record the set's
  hash in the contract, so its cost is paid once per oracle version.
- **Many artifacts per row.** Five steps per transition, five transitions per
  domain. Fallback: the registry already content-addresses each one, and the
  lineage graph is a chain per row with few cross-links, so the dependents
  index stays small.
- **Weak oracles.** In data-first cycles step 2 is a supervised baseline on
  uncertain labels, and matching it is not being right. Fallback: the
  oracle's strength is on every row (A2), and the physical-constraint scorers
  of Report 14 are mandatory for rungs scored against a weak oracle.
- **Exact oracles are rare.** Stage 2's posterior is exact because α is one
  number. For a field or a plume, step 2 is an EnKF, a 4D-Var or MCMC.
  Fallback: the complexity axis (MAP below Laplace below NUTS) validates the
  oracle itself before it validates anything else, as stage 2 does with
  Laplace.
- **The physics libraries are not pinned.** plumax, rtmx, filterax and vardax
  are named but not in `references/`, so the real methane cycle cannot run
  here yet. Fallback: phase 3 waits for them; phases 1 and 2 need only the
  pinned packages.

(r17-open-questions)=
## Open questions

1. **Which levels do gap filling and retrieval connect?** The ladder (and
   Report 15 and the gallery, which follow it) call gap filling L1 → L2; the
   overview's levels and the scoping report's matrix call retrieval L1 → L2
   and gap filling L2 → L3. Recommended: the overview's levels, with Report
   15's and the gallery's tasks relabelled in one change. Alternative: keep
   the ladder's levels and rename the overview's.
2. **Does an oracle's strength propagate?** The scoping report scores its
   direct predictor against step 4 (its section 6.3), and a domain may
   override the default rule that way to save oracle runs; that reference
   carries step 4's emulator error. Should such a row record the weaker of
   the two strengths?
3. **Entering mid-cycle.** A team handed a trained emulator without its
   simulator can run steps 4 and 5 with the emulator as reference. Is that
   `heuristic` strength with no step-1 parent, or a strength of its own?
4. **Cross-domain cycles.** Methane retrieval feeding ocean carbon
   assimilation chains two rows. Is that the gallery's cascade contract
   ([entry 5.1](../benchmarks_gallery.md#gal-5-1)), with one cycle per row?
5. **When does pipekit-jax matter?** Step 4 with a differentiable emulator
   enables gradient-based inversions that stage 4's quadrature does not need.
   Should the first real cycle use one, to settle pipekit-jax's timing?

(r17-todos)=
## Upcoming work

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r17-todos

| Item | Where | Why |
|---|---|---|
| Map each gallery entry to its cells (transition, steps present, oracle strength) | [benchmarks gallery](../benchmarks_gallery.md) | the scoping report's appendix; needs open question 1 settled first |
| Pin plumax, rtmx, filterax and vardax | `references/`, `UPSTREAM` | the real methane cycle and the scoping report's step-1 and step-2 names cannot be checked otherwise |
| Flat configurations for stored artifacts | pipekit `Operator.from_state` or the registry's docs | stage 6: a nested configuration stores but does not reload |
| The plume pilot as the first real cycle | pipekit-evaluate `docs/design/plume_pilot.md` | the ladder's design note picks it because every rung exists |
:::

(r17-references)=
## References

:::{note} Companion reports
:class: simple
Reports [10](10_pipekit_cycle.md), [11](11_pipekit_train.md),
[12](12_pipekit_experiment.md), [14](14_pipekit_evaluate.md) and
[15](15_benchmarking.md) are published.
:::

### Sources

- The scoping report (numbered 16 in its draft, 17 here), "Report 16 — The data-driven modeling cycle and
  benchmark-gated L0–L4 refinement" [@geostack_cycle_draft], and the gallery
  draft it extends [@geostack_gallery_draft].
- The pinned code in `references/`: pipekit-evaluate (`benchmark/` and
  `docs/design/benchmark_ladder.md`), pipekit-cycle (`forward.py`,
  `cycle.py`, `da.py`), pipekit-train (`dataset.py`, `loop.py`),
  pipekit-experiment (`registry.py`, `artifacts.py`), pipekit
  (`_base/operator.py`), xrtoolz (`metrics/_src/distributional.py`) and
  geotoolz (`matched_filter/`).
- The running example's notebook, described in [](#r17-running).
