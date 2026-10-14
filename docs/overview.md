---
title: GeoStack overview
subtitle: A JAX-native research stack for geophysical modeling, inference and data assimilation
short_title: Overview
description: The entry point to GeoStack. Three organising axes, the seven-tier stack, an end-to-end methane walkthrough, the open questions and a plan.
keywords:
  - geophysics
  - data assimilation
  - JAX
  - operator graphs
  - remote sensing
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
abbreviations:
  DA: data assimilation
  GP: Gaussian process
  ML: machine learning
  RT: radiative transfer
  SST: sea surface temperature
  SSH: sea surface height
  LST: land surface temperature
  COG: Cloud-Optimized GeoTIFF
  CTD: conductivity, temperature and depth
  EnKF: ensemble Kalman filter
  TCCON: Total Carbon Column Observing Network
  CAMS: Copernicus Atmosphere Monitoring Service
  UQ: uncertainty quantification
---

+++ {"part": "abstract"}

There is no Keras for geophysics. GeoStack is the connective tissue that is
missing: one operator-graph abstraction that runs from structured linear
algebra up to end-to-end pipelines, with content-addressed data and models
throughout and machine learning pluggable at every step. This page is the
map. It ties together the fifteen scoping reports, the GeoModels pages and
the benchmarks gallery ([Where to read next](#sec-reading)), and it ends
with the open questions and a plan.

+++

:::{tip} How to read this page
:class: dropdown
- **Five minutes:** read the [elevator pitch](#sec-pitch), look at
  {numref}`fig-lifecycle` and {numref}`fig-tier-stack`, and skim the
  [plan](#sec-plan).
- **Building something:** start with the [walkthrough](#sec-walkthrough)
  and the [overview examples](overview_examples.md), then search the
  [capability index](api/capabilities.md) before writing new code.
- **Reviewing the design:** go straight to the [decision](#sec-decision),
  the [ownership rules](#sec-ownership), the [gates](#sec-gates) and the
  [open questions](#sec-open-questions).
:::

(sec-decision)=
## The decision

GeoStack is one operator algebra across three stacks: **GeoML** (numerics
and inference), **GeoModels** (forward models) and **GeoStack proper**
(data, pipelines, evaluation). None merges into another: the JAX libraries
stay pipeline-free and reach pipekit only through thin `adapters/pipekit.py`
modules.

The work splits into six workstreams. Each makes one decision, and the
letters tag every `proposed (X)` call on the
[examples page](overview_examples.md) and every gate in {numref}`tbl-gates`.

- **A · One framework.** Every library invents its own pipeline glue today.
  pipekit replaces it with one `Operator` protocol, `Sequential` and `Graph`
  composition, config round-tripping and sandboxed YAML.
- **B · Indexed data at both ends of the lifecycle.** Observations can be
  catalogued; model states are scattered files. `geocatalog` indexes both:
  L0–L2 observations as rows queried with `GeoSlice`s, and L3–L5 model
  states through its {term}`state profile` (run time, member, producer and
  parent columns, in `geocatalog.states`). The readers split by level:
  `georeader` below L3, `xrreader` from L3 up.
- **C · Two carriers, one algebra.** `geotoolz` (GeoTensor) and `xrtoolz`
  (xarray) share the operator algebra and the patcher; only the substrate
  differs.
- **D · Data assimilation as a cycle.** pipekit-cycle's `ForwardModel`,
  `ObservationOperator` and `AnalysisStep` protocols are the only contract.
  filterax, vardax and plumax satisfy them through adapters, so an emulator
  replaces a physics model without touching the cycle.
- **E · Training with lineage.** pipekit-train wraps Equinox + Optax today,
  with NumPyro and BlackJAX for Bayesian fits and Lightning and Keras 3
  planned; pipekit-experiment adds a tracker protocol and a content-addressed
  `ModelRegistry`.
- **F · Evaluation as infrastructure.** pipekit-evaluate scores every cell
  of the benchmark cube against an `Oracle`, with the metrics from
  `xrtoolz.metrics`; the Unit × Lens × Stage tags and a content-addressed
  evaluation report are proposed ([Report 14](reports/14_pipekit_evaluate.md)).

**End state.** GeoML owns linear algebra, kernels and inference and never
imports pipekit outside an adapter. GeoModels owns physics and exposes it
only through adapters. pipekit owns composition, cycles, training and
evaluation. geocatalog owns the one index, for observations and, through
its state profile, for model states; geotoolz and xrtoolz own the
carrier-specific operators. Orchestration and serving stay
external.

(sec-pitch)=
## The elevator pitch

> A vertically integrated, JAX-native research software stack for geophysical
> modeling, inference and data assimilation: from structured linear algebra
> primitives, through composable operator graphs, to indexed multi-tier data
> products, with ML pluggable at every level.

Great JAX numerics and great xarray pipelines exist today; what is missing is
the operator-graph abstraction that connects them, with content-addressed
data and models throughout.

(sec-problem)=
## The shape of the problem

Five gaps motivate everything below. The first four are old; the fifth has
only become visible recently.

::::{grid} 1 1 2 2

:::{card} 1 · Scattered numerics
The same linear algebra, kernel methods and filtering routines are
reimplemented in every project. There are no shared primitives.
:::

:::{card} 2 · ML tooling ignores classical numerics
Keras and PyTorch layers assume supervised learning on tensors. They do not
express linear operators, PDE solvers, state estimation or uncertainty
propagation.
:::

:::{card} 3 · Monolithic domain software
Ocean models, retrieval algorithms and DA systems are dense, tightly
coupled and hard to recombine.
:::

:::{card} 4 · No connective tissue
Even with good numerics and good ML, there is no Keras-like pipeline to
compose them for real geophysical work.
:::

:::{card} 5 · No indexed model-state tier
Observations are catalogable (STAC, GeoParquet). Forecasts, analyses and
reanalyses are scattered files with bespoke conventions: the L3–L5 half of
the lifecycle has nothing like the catalogs that organise L0–L2.
:::
::::

The stack addresses all five.

(sec-axes)=
## Three organising axes

Three axes structure the framework ([](#tbl-axes)). They are orthogonal:
every operation sits somewhere on all three.

:::{table} The three axes and what each one orders.
:label: tbl-axes

| Axis | Orders | Figure |
|---|---|---|
| 1 · Modeling cycle | the research loop: model, observe, compare, refine | {numref}`fig-modeling-cycle` |
| 2 · Data lifecycle | data maturity, from raw telemetry (L0) to derived products (L5) | {numref}`fig-lifecycle` |
| 3 · Six-step cycle | the journey from a simple model to amortized inference | {numref}`fig-six-steps` |
:::

(sec-modeling-cycle)=
### The modeling cycle (axis 1)

```{figure} images/overview/modeling_cycle.svg
:label: fig-modeling-cycle
:alt: The real-world system feeds two tracks. Modeling runs abstract, equations and code, predictions. Observing runs sensor, measurements, observations. Both meet at compare; refine loops back to abstract.
:width: 100%

**The modeling cycle runs on two parallel tracks.**

The modeling track abstracts the real-world system into equations and code
that make predictions. The observing track measures the same system with
sensors and turns measurements into observations. The tracks meet where
predictions are compared with observations, and the model is refined.
```

:::{important} Start simple
Add complexity only when validation demands it. This is the research loop,
and every package in the stack exists to make one or more of its steps
composable.
:::

(sec-lifecycle)=
### The L0–L5 data lifecycle (axis 2)

```{figure} images/overview/lifecycle.svg
:label: fig-lifecycle
:alt: Six maturity levels, L0 to L5, across the top. Below, the science axis fans radiances out into ocean, atmosphere, land and ice lanes; further below, the engineering axis shows which GeoStack tool transforms, patches, carries, reads, indexes and stores each level.
:width: 100%

**The data lifecycle on two perpendicular axes.**

The science axis fans one stream of radiances out into ocean, atmosphere,
land and ice at retrieval, and ends in derived products such as extremes
(L5, dashed: examples still to be defined). The engineering axis maps every
level to the tool that handles it, switching carrier from GeoTensor to
xarray between L2 and L3.
```

:::{table} The six levels of the data lifecycle.
:label: tbl-levels

| Level | Name | What it holds | Carrier |
|---|---|---|---|
| L0 | Unstructured observations | raw instrument output: telemetry, counts, packets | GeoTensor |
| L1 | Structured observations | calibrated, georeferenced radiances | GeoTensor |
| L2 | Retrieved observations | per-pixel geophysical variables, with gaps where retrieval fails | GeoTensor |
| L3 | Analysis | a regular, gap-filled grid: the "world right now" | xarray |
| L4 | Reanalysis and forecast | dynamical model output: data-constrained, fused or projected | xarray |
| L5 | Derived products | extremes, indicators and impacts computed from L4 | xarray |
:::

[](#tbl-levels) names the six levels.

:::{note} Two names for L2
pipekit-evaluate's `LEVEL_NAMES` calls L2 "gap-filled observations", where
[](#tbl-levels) has retrieved observations with gaps. Pages that follow the
code ([Report 15](reports/15_benchmarking.md#r15-framework), the
[gallery](benchmarks_gallery.md)) therefore label gap filling L1 → L2, while
this page puts it at L2 → L3 ([stage 3](#sec-stage-gapfill)). The two are
not reconciled yet; [Report 17 A1](reports/17_cycle.md#r17-ws-a) proposes
the rename to `"retrieved_obs"`.
:::

**The science axis** runs left to right. Until retrieval it is one stream:
the same L1 radiances feed every geoscience. Retrieval is where it fans out
into ocean, atmosphere, land and ice variables, and each domain then
carries its own analysis and reanalysis or forecast. L5 holds the products
derived from those states, such as extremes (marine heatwaves, droughts,
storms) or impact indicators. Which products belong in L5 is still open.

**The engineering axis** runs top to bottom, from transforming data down to
storing it, and maps every level to the GeoStack tool that handles it. It
splits where the {term}`carrier` changes:

- **Observations (L0–L2):** `georeader` reads them into a {term}`GeoTensor`,
  `geocatalog` indexes them as {term}`GeoSlice`s, `geotoolz` transforms
  them and `geocloud` stores them as COG on object storage.
- **Model states and derived products (L3–L5):** `xrreader` reads them into
  xarray, `geocatalog.states` indexes them in the same catalogs as state
  rows (its {term}`state profile`), `xrtoolz` transforms and verifies them,
  and they live as Zarr or NetCDF on object storage.
- **Derived products (L5):** `xtremax` computes extremes from L4 states.
- **Both carriers:** `geopatcher` patches fields whatever the carrier.

:::{aside}
`xrreader` exists today as the `xrtoolz-reader` package (it imports as
`xrreader`).
:::

ML enters at every transition.

(sec-two-streams)=
### Two streams converge

The lifecycle is not one chain. A top-down satellite stream and a bottom-up
in-situ stream converge between L2 and L3 ([](#fig-two-streams)).

```{figure} images/overview/two_streams.svg
:label: fig-two-streams
:alt: Satellite levels L0 to L4 on the left; in-situ networks for ocean, atmosphere, land and ice on the right. L2 and in-situ meet at a matchup node that feeds L3; in-situ also enters L4 through assimilation.
:width: 100%

**Satellite and in-situ observations converge through matchup.**

In-situ networks in every domain join the satellite chain through
spatiotemporal {term}`matchup` between L2 and L3, and enter L4 through
assimilation. They have no L0 or L1 equivalent.
```

The matchup pattern is canonical and deserves its own operator family
([stage 2](#sec-stage-matchup) runs it; where it lives is an
[open question](#sec-open-questions)).

(sec-three-layers)=
### Three layers of transformation

The cycle and the lifecycle are joined by a third lens: what we *do* to the
data on its way into a model ([](#fig-three-layers)).

```{figure} images/overview/three_layers.svg
:label: fig-three-layers
:alt: Scientific data becomes ML-ready data through harmonisation and tensorisation, which becomes embeddings through compression and latent mapping. Each layer lists the packages that own it.
:width: 100%

**Scientific data, ML-ready data, embeddings, and who owns each.**
```

(sec-six-steps)=
### The six-step cycle (axis 3)

The research-to-production journey ([](#fig-six-steps)). Every domain
library follows this cycle implicitly; the framework makes it explicit and
composable.

```{figure} images/overview/six_steps.svg
:label: fig-six-steps
:alt: Five steps in a row, from simple model to amortized inference. Step 2 validates steps 3 to 5. A shared improve step sits under steps 2 to 5 with two-way links.
:width: 100%

**From a simple model to amortized inference, with one shared improve step.**

Step 2, the classical method, is the {term}`oracle` that validates steps
3–5. Step 6 is shared: any of steps 2–5 can be improved, using the previous
step as ground truth.
```

In symbols: let θ be the quantity we want (an emission rate, a model
state), f the physical forward model, H the observation operator, y the
observations and ε their error with covariance R. Step 1 is the generative
story:

```{math}
:label: eq-generative
y = H\left(f(\theta)\right) + \varepsilon, ~~~ \varepsilon \sim \mathcal{N}(0, R)
```

Step 2 inverts it with classical, model-based inference. Bayes' rule with a
prior p(θ) defines the target:

```{math}
:label: eq-posterior
p(\theta \mid y) \propto p(y \mid \theta) ~ p(\theta)
```

Computing {eq}`eq-posterior` is exact only in tractable cases, such as a linear-Gaussian
model solved with the Kalman filter or a GP. The usual step-2 methods (a
finite-ensemble EnKF, 3D- or 4D-Var) approximate it, slowly. Step 2 is the
oracle because it is the most trusted reference available, not because it
is exact.

Step 3 trains an emulator $\hat f_\phi$ with weights φ on step-1
simulations by {eq}`eq-emulator`, and step 4 reruns step 2 with
$\hat f_\phi$ in place of f:

```{math}
:label: eq-emulator
\phi^\star = \arg\min_\phi \mathbb{E}_{\theta \sim p(\theta)} \left\| f(\theta) - \hat f_\phi(\theta) \right\|^2
```

Step 5 skips the inversion: it trains a conditional density $q_\psi$ with
weights ψ that maps observations straight to an approximate posterior, on
pairs (θ, y) drawn from the generative model {eq}`eq-generative`, by
{eq}`eq-amortized`:

```{math}
:label: eq-amortized
\psi^\star = \arg\min_\psi \mathbb{E}_{(\theta, y) \sim p(\theta, y)} \left[ -\log q_\psi(\theta \mid y) \right], ~~~ q_{\psi^\star}(\theta \mid y) \approx p(\theta \mid y)
```

[](#tbl-steps) says where each step lives in the stack. Two structural
commitments fall out of the cycle:

::::{grid} 1 1 2 2

:::{card} Step 2 is the oracle
Classical methods (`filterax`, `vardax`, `pyrox`) are not legacy code
waiting to be replaced by neural emulators. They become *reference
generators* that validate steps 3–5. The classical path stays alive,
promoted to oracle.
:::

:::{card} Benchmarks formalise understanding
Writing a benchmark contract (carriers, references, metrics, splits,
baselines, expected failure modes) *is* formalising what you understand
about the problem. Every transition between steps is a benchmarkable
contract.
:::
::::

:::{table} Where each step lives in the stack.
:label: tbl-steps

| Step | Lives in |
|---|---|
| 1 · Simple model | Tier 3 domain libraries (`somax`, `plumax`) |
| 2 · Model-based inference | `pipekit-cycle.DACycle` with classical analysis steps |
| 3 · Emulator | `pipekit-train.SimulationDataset`, registered in `pipekit-experiment.ModelRegistry` |
| 4 · Emulator-based inference | `pipekit-cycle.DACycle` with `NeuralForward` |
| 5 · Amortized inference | `pipekit-train` with `gauss_flows` or `pyrox` |
| 6 · Improve | `pipekit-evaluate.Benchmark` reruns, compared row by row, with lineage in `pipekit-experiment` |
:::

(sec-axes-relate)=
### How the axes relate

The axes form a cube of operations. At any point you are somewhere on the
modeling cycle (research stage), somewhere on the lifecycle (data maturity)
and somewhere on the six-step cycle (production maturity).
[](#fig-axes-matrix) crosses the lifecycle transitions with the steps.

```{figure} images/overview/axes_matrix.svg
:label: fig-axes-matrix
:alt: A grid with lifecycle transitions as rows, from calibration to forecast, and steps 1 to 5 as columns. Each cell names a method; the step-2 column is highlighted as the oracle.
:width: 100%

**Every transition crossed with every step is a concrete method.**
```

**Every cell is a benchmarkable artifact,** scored against the earlier step
that computes the same thing: an emulator against the simulator it
replaces, and emulator-based and amortized inference against the step-2
oracle's posterior. pipekit-evaluate's benchmark ladder ships this rule
([Report 17](reports/17_cycle.md#r17-stage-1)). The whole lifecycle is a nested set of six-step
cycles, each gated by benchmarks. That is what makes the framework
improvable in practice: improving a cell re-runs its downstream column,
lineage is content-addressed, and nothing is opaque.

(sec-stack)=
## The big picture stack

Two zoom levels: the tiers first, then the packages inside Tier 4.

(sec-tiers)=
### Seven tiers

```{figure} images/overview/tier_stack.svg
:label: fig-tier-stack
:alt: Seven stacked tiers, from the external ecosystem at the bottom through owned numerical, algorithm, domain and pipeline tiers to external orchestration and serving at the top.
:width: 100%

**The stack in seven tiers.**

Solid tiers are owned; dashed tiers are external. `georeader` sits in Tier 0
as the GeoTensor substrate.
```

(sec-tier4)=
### Tier 4, zoomed in

Most current work happens in the framework and infrastructure tier
([](#fig-tier4-map)).

```{figure} images/overview/tier4_map.svg
:label: fig-tier4-map
:alt: Three groups: the pipekit framework packages, the ML-loop packages, and the infrastructure peers geocatalog, geopatcher and geocatalog's state profile, with georeader as an external substrate.
:width: 100%

**The pipekit framework, the ML loops built on it, and the infrastructure peers.**
```

(sec-dependencies)=
### The dependency graph

[](#fig-dependency-graph) shows which reference packages import which.

```{figure} images/overview/dependency_graph.svg
:label: fig-dependency-graph
:alt: Three family panels. Every pipekit package depends on pipekit; xrtoolz builds on xrtoolz-core, which depends on pipekit; geotoolz depends on pipekit, and every geotoolz-family package depends on georeader. Dashed edges are optional extras.
:width: 100%

**How the reference packages import each other.**

Drawn from the packages' own metadata. Solid edges are dependencies, dashed
edges optional extras.
```

(sec-ownership)=
### Ownership rules

A package imports only from tiers below it, so there are no circular
dependencies; geotoolz and xrtoolz re-export the common framework symbols,
so user code usually imports a single package. The three stacks of [the decision](#sec-decision) each have
their rules: GeoStack's (pipekit and its packages, geocatalog, geopatcher,
geotoolz and xrtoolz) are [Report 16's](reports/16_geo_data_stack.md#r16-ownership),
and GeoModels' (somax, plumax, radtrax, on top of the Numerics layer) are on
[the package boundaries page](geomodels/package_boundaries.md#pb-rules).
GeoML's rules are stated only here: the JAX libraries touch pipekit only
through an optional adapter module, never in their core.

1. **gaussx** owns structured linear algebra and Gaussian primitives. It
   imports no GeoStack package.
2. **geonnax** owns basis functions, neural layers and coordinate
   transforms. somax and the other models compose them and never
   reimplement them.
3. **kernellib** owns covariance kernels and tapers. It imports no
   inference library.
4. **optax_bayes** owns the Bayesian learning rule as optax transforms. It
   imports no model library.
5. **spectraldiffx** and **finitevolx** own discretisations: spectral and
   finite-volume operators, no physics.
6. **pyrox** owns GP and Bayesian NN building blocks for NumPyro. Its
   pipekit surface (`KrigingOperator`) lives only in
   `pyrox.adapters.pipekit`.
7. **filterax** owns ensemble Kalman methods and **vardax** owns 3D/4D-Var.
   Each exposes an `AnalysisStep` only through `adapters/pipekit.py`.
8. **gauss_flows** owns Gaussianization and normalizing flows;
   **xtremax** owns extreme-value and point-process models. Neither imports
   pipekit outside an adapter.

(sec-walkthrough)=
## Walkthrough: methane attribution

The canonical example: a complete pipeline from satellite scenes to source
emission estimates, with ML at several stages.

```{figure} images/overview/methane_walkthrough.svg
:label: fig-methane-walkthrough
:alt: Eight steps: catalog, read and retrieve, match with in-situ, grid and gap-fill, assimilate, index, evaluate, register and serve; each linked to the package that handles it.
:width: 100%

**Eight steps from cataloguing scenes to serving estimates.**
```

:::{seealso}
The [full pipeline code](#ex-methane), step by step, on the overview
examples page.
:::

The data: EMIT imaging-spectrometer scenes (L1 radiances) over an area of
interest, TCCON column measurements as in-situ truth, and CAMS reanalysis and
aircraft campaigns as evaluation references. The dimensions used below: B
spectral bands, H × W pixels per scene, M observations, N grid cells,
$N_e = 40$ ensemble members and T = 24 hourly steps.

The eight steps of {numref}`fig-methane-walkthrough` group into five stages.
Each stage states the problem it solves before the code does; stages 3 and 4
have closed-form special cases that double as test oracles
({numref}`tbl-gates`).

(sec-stage-retrieve)=
### Stage 1 · Retrieve methane (steps 1–2: geocatalog → georeader → geotoolz)

**TL;DR.** Find every EMIT scene over the area, mask clouds, and turn each
pixel's spectrum into a methane enhancement.

**Problem.** For pixel i, $x_i \in \mathbb{R}^B$ is its radiance spectrum,
μ the scene-background mean, $\Sigma \in \mathbb{R}^{B \times B}$ the
background covariance and t the methane target signature (unit absorption
times μ). The matched filter's enhancement estimate $\hat\alpha_i$ is the
least-squares amplitude of t in the whitened residual:

```{math}
:label: eq-matched-filter
\hat\alpha_i = \frac{(x_i - \mu)^\top \Sigma^{-1} t}{t^\top \Sigma^{-1} t}
```

The classical filter {eq}`eq-matched-filter` is step 2 of the
[six-step cycle](#sec-six-steps) for the L1→L2 cell; its derivation is in
[radtrax · 1](geomodels/radtrax/01_column_retrieval.md#rx1-stage-5). A
learned retrieval takes the same slot in the pipeline, and the filter stays
as its oracle.

(sec-stage-matchup)=
### Stage 2 · Match with in-situ (step 3: geocatalog matchup)

**TL;DR.** Pair each TCCON measurement with the one satellite retrieval
nearest to it in space and time; measurements with nothing nearby are
dropped.

**Problem.** In-situ observation j sits at location $s_j$ and time $t_j$;
satellite retrieval k at $s_k$, $t_k$; d is great-circle distance. The
candidates for j are the retrievals inside both tolerances,
{eq}`eq-matchup-window`:

```{math}
:label: eq-matchup-window
W_j = \left\{ k : |t_j - t_k| \le 3~\mathrm{h} ~~\text{and}~~ d(s_j, s_k) \le 25~\mathrm{km} \right\}
```

If $W_j$ is empty, j has no matchup. Otherwise j keeps the single candidate
with the smallest scaled distance {eq}`eq-matchup`, so no TCCON record is
counted twice:

```{math}
:label: eq-matchup
k^\star(j) = \arg\min_{k \in W_j} \left(\frac{d(s_j, s_k)}{25~\mathrm{km}}\right)^2 + \left(\frac{|t_j - t_k|}{3~\mathrm{h}}\right)^2
```

Ties go to the earlier retrieval, then the smaller catalog index, so the
selection is deterministic.

The pairs serve two jobs, so TCCON is split once, by station, before
anything is fitted. Pairs from the training sites are the (label, feature)
data a learned retrieval trains on; pairs from the held-out sites validate
stage 1 and are the only TCCON references used in stage 5. No site appears
in both, so held-out scores never include a label the model saw. How much
of the pairs' disagreement is representativeness error is an
[open question](#sec-open-questions).

(sec-stage-gapfill)=
### Stage 3 · Grid and gap-fill (step 4: pyrox through pipekit)

**TL;DR.** Turn scattered retrievals into a regular map with an uncertainty
for every grid cell.

**Problem.** $y \in \mathbb{R}^M$ are the L2 retrievals, σ² their noise
variance and k a Matérn-5/2 kernel with a 50 km length scale. $K_{ff}$ is k
between the M observation locations, $K_{*f}$ between the N grid cells and
the observations, and $K_{**}$ between grid cells. The GP posterior mean m
and variance v on the grid are:

```{math}
:label: eq-kriging-mean
m = K_{*f} \left(K_{ff} + \sigma^2 I_M\right)^{-1} y
```

```{math}
:label: eq-kriging-var
v = \operatorname{diag}\left(K_{**} - K_{*f} \left(K_{ff} + \sigma^2 I_M\right)^{-1} K_{f*}\right)
```

Kriging is the oracle for the L2→L3 cell; DINEOF and neural gap-fillers are
judged against it. Stage 4 assimilates this map: m is its observation and
$\mathrm{diag}(v)$ its observation-error covariance.

(sec-stage-da)=
### Stage 4 · Assimilate hourly (step 5: pipekit-cycle → plumax → filterax)

**TL;DR.** Step a transport model forward each hour and pull it towards the
observations.

**Problem.** $x_t \in \mathbb{R}^N$ is the methane state at hour t, 𝓜 one
plumax transport step of 3,600 s, H the column observation operator, R the
observation-error covariance, here $R = \mathrm{diag}(v)$ from the stage-3
kriging, and $P^b$ the background covariance estimated
from the $N_e$ members. Each cycle forecasts every member $i = 1, \dots, N_e$
by {eq}`eq-forecast`:

```{math}
:label: eq-forecast
x_t^{b,(i)} = \mathcal{M}\left(x_{t-1}^{a,(i)}\right)
```

then updates each member against its own perturbed copy of the
observations, $\varepsilon^{(i)} \sim \mathcal{N}(0, R)$ drawn
independently per member (the stochastic EnKF):

```{math}
:label: eq-enkf
K = P^b H^\top \left(H P^b H^\top + R\right)^{-1}, ~~~ x_t^{a,(i)} = x_t^{b,(i)} + K \left(y_t + \varepsilon^{(i)} - H x_t^{b,(i)}\right)
```

The perturbations matter: updating every member with the same unperturbed
$y_t$ shrinks the anomalies by $(I - KH)$ and leaves the analysis ensemble
too narrow, which would bias the CRPS in stage 5. $P^b$ is inflated by 1.05
and localised with a Gaspari–Cohn taper; localisation trades exactness for
removing spurious long-range correlations, and the perturbations make the
estimate random. They are drawn from one key seeded from the run's config,
so a rerun reproduces them (gate G1). A square-root (ETKF) variant replaces
the perturbations with a deterministic anomaly transform. With a linear 𝓜
and H, no inflation and no localisation, its analysis mean equals the Kalman
update applied to the ensemble's own sample mean and covariance, which is
gate D2.

(sec-stage-evaluate)=
### Stage 5 · Index, evaluate, register (steps 6–8: geocatalog.states → pipekit-evaluate → pipekit-experiment)

**TL;DR.** Store every analysis, score it against three independent
references, and attach the score to the model that made it.

**Problem.** For a forecast distribution F, a reference value y, and
independent draws X, X′ from F, the continuous ranked probability score is
below; lower is better, and with one member it reduces to absolute error.
Mass closure checks that total methane is conserved to within 1 %.

```{math}
:label: eq-crps
\mathrm{CRPS}(F, y) = \mathbb{E}_F |X - y| - \frac{1}{2} \mathbb{E}_F |X - X'|
```

The same pipeline, written as YAML, becomes the regulatory artifact
([Report 9, use case 9](reports/09_use_cases.md#r9-case-9)). The same
operators run in a notebook ([case 1](reports/09_use_cases.md#r9-case-1)),
an ETL job ([case 2](reports/09_use_cases.md#r9-case-2)), a FastAPI endpoint
([case 5](reports/09_use_cases.md#r9-case-5)) or a Fused UDF (the Fused
integration design).

(sec-tier-briefs)=
## Per-tier briefs

A compact summary per package. Where a report or a GeoModels page covers a
package, the brief is a link to it.

::::{tab-set}

:::{tab-item} Tier 0 · Substrate
**georeader** (external, but central) owns `GeoTensor`, the raster carrier,
and the reader infrastructure (`RasterioReader`, `AsyncGeoTIFFReader`).
Everything raster-shaped builds on it.

**The xarray ecosystem** (external): `DataArray`, `Dataset` and `DataTree`,
plus rioxarray, xvec, cf-xarray, regionmask, xskillscore, dask and intake.
:::

:::{tab-item} Tier 1 · Numerics
| Package | What it does |
|---|---|
| `gaussx` | Structured linear algebra, Gaussian primitives, exponential family; wraps lineax and matfree |
| `geonnax` | Basis functions, neural layers and coordinate transforms (`lonlat_to_cartesian3d`) |
| `kernellib` | Covariance kernels and tapers |
| `spectraldiffx` | Pseudospectral differentiation, filters, PDE solvers |
| `finitevolx` | Arakawa C-grid finite-volume operators |
| `optax_bayes` | The Bayesian learning rule as optax gradient transformations |
| `eqx-trainer` | A minimal JAX / equinox training loop |

Pure math with no domain knowledge: all `eqx.Module`-based, jittable and
JAX-traceable. See the [gaussx example](#ex-gaussx).
:::

:::{tab-item} Tier 2 · Algorithms
| Package | What it does |
|---|---|
| `pyrox` | GP and Bayesian deep-learning building blocks for NumPyro hierarchical models (a split into `nnox` + `pyrox` is planned) |
| `filterax` | Differentiable ensemble Kalman methods |
| `vardax` | Learnable variational DA (3D- and 4D-Var) |
| `gauss_flows` | Rotation-based Gaussianization and normalizing flows |
| `xtremax` | Extreme-value and point-process models; computes the L5 extremes |
| `fairkl` | Fairness-constrained kernel learning (Keras 3) |
| `keras-nerf` | Neural fields for state estimation (Keras 3) |
| `keras-flows` | Normalizing flows (Keras 3) |

Keras 3 and Equinox are equal-status backends. Each algorithm's adapter
(`*.adapters.pipekit`) makes it composable into pipekit pipelines. See the
[filterax example](#ex-filterax).
:::

:::{tab-item} Tier 3 · Domain
| Package | What it does | Read |
|---|---|---|
| `somax` | Ocean and atmosphere model zoo | — |
| `plumax` | Atmospheric plume transport | [plumax · 1](geomodels/plumax/01_emission_rate.md), [package boundaries](geomodels/package_boundaries.md) |
| `radtrax` | Differentiable radiative transfer, formerly rtmx (proposed; does not exist yet) | [radtrax · 3](geomodels/radtrax/03_design.md) |
| `methanex` | End-to-end methane remote sensing (planned; no repository yet) | — |
| `geotoolz` | GeoTensor operators for remote sensing | [Report 3](reports/03_sister_libraries.md#r3-geotoolz) |
| `xrtoolz` | xarray operators for ocean, atmosphere and reanalysis | [Report 3](reports/03_sister_libraries.md#r3-xrtoolz) |

The forward models each ship a thin `adapters/pipekit.py` that exposes the
operator surface (`ForwardModel`, `ObservationOperator`, …) without touching
the algorithmic core. geotoolz and xrtoolz share the same Tier 4 peers: the
substrate differs, the operator algebra is identical.
:::

:::{tab-item} Tier 4 · Pipelines
**pipekit**, the framework: fourteen groups (A–N), listed in
[Report 2's inventory](reports/02_pipekit_surface.md#r2-inventory), with its
lineage in [Report 1](reports/01_toolz_primer.md).

| Package | Report |
|---|---|
| `pipekit-array` | [Report 3](reports/03_sister_libraries.md#r3-array) |
| `pipekit-jax` | [Report 5](reports/05_pipekit_jax.md): 0.0.2 ships a model wrapper and a diffrax forward model; the operator layer is proposed |
| `pipekit-cycle` | [Report 10](reports/10_pipekit_cycle.md) |
| `pipekit-train` | [Report 11](reports/11_pipekit_train.md) |
| `pipekit-experiment` | [Report 12](reports/12_pipekit_experiment.md) |
| `pipekit-evaluate` | [Report 14](reports/14_pipekit_evaluate.md), [Report 15](reports/15_benchmarking.md), [Report 17](reports/17_cycle.md) |
| `geocatalog` (ships as `geotoolz-catalog`) | [Report 6](reports/06_geocatalog.md) |
| `geopatcher` (ships as `geotoolz-patcher`) | [Report 7](reports/07_geopatcher.md) |
| `geocatalog.states` (proposed state profile) | [Report 13](reports/13_state_catalog.md) |
:::
::::

(sec-code-patterns)=
## Key code patterns

Each pattern has runnable-looking code on the
[overview examples](overview_examples.md) page.

::::{grid} 1 2 3 3

:::{card} Define an operator
:link: overview_examples.md#ex-operator
How an operator like `PercentileClip` is written.
:::

:::{card} Compose pipelines
:link: overview_examples.md#ex-compose
`Sequential`, the pipe syntax, and DAGs with `Graph`.
:::

:::{card} Catalogs
:link: overview_examples.md#ex-catalogs
Build, query and match a `geocatalog`.
:::

:::{card} Patcher
:link: overview_examples.md#ex-patcher
Geometry × sampler × window × aggregation.
:::

:::{card} Data assimilation cycle
:link: overview_examples.md#ex-da-cycle
Protocol-satisfying parts composed into a `DACycle`.
:::

:::{card} Swap in an emulator
:link: overview_examples.md#ex-emulator
Replace a classical forward model with a neural one.
:::

:::{card} Evaluation pipeline
:link: overview_examples.md#ex-evaluation
Multi-track evaluation as a content-addressed artifact.
:::

:::{card} Benchmark contract
:link: overview_examples.md#ex-benchmark
A pre-registerable, content-addressed benchmark.
:::

:::{card} Reproducibility artifact
:link: overview_examples.md#ex-repro
Freeze a pipeline for a regulatory cold rerun.
:::
::::

(sec-open-questions)=
## Open questions

None of these block work; all deserve thought before committing to a
decision.

:::{dropdown} Raised by the design review
:open:
- **What `InferenceArtifact` still needs for a cold rerun.** The shipped
  `InferenceArtifact` already saves and loads a pipeline's YAML, its pinned
  model hashes, the lockfile and metadata. A `rerun` that rebuilds the
  pipeline and pulls its frozen inputs is proposed. Open: whether that rerun
  resolves inputs from the pinned catalog URIs and SHA-256 digests alone,
  without cloud credentials, and how a `Benchmark` (not yet serialisable)
  is frozen next to it.
:::

:::{dropdown} Resolved elsewhere
- **Serialising a `Graph`.** Emitting a built pipeline as YAML that loads
  back is proposal A of [Report 9](reports/09_use_cases.md#r9-proposals);
  until it lands, gate A1 covers only YAML-safe operators.
- **A per-step observation error.** An `obs_err_source` is proposed in
  [Report 10, workstream B](reports/10_pipekit_cycle.md#r10-ws-b).
- **Two evaluation APIs.** [Report 15, workstream D](reports/15_benchmarking.md#r15-ws-d)
  builds the contract and the tracks on the shipped `Benchmark` cube
  (`from_contract`, several references per run), with no separate
  `MultiTrackEvaluation`.
- **Where `xrreader` stops and `georeader` starts.** A mission reader lives
  with its output carrier ([Report 16, workstream E](reports/16_geo_data_stack.md#r16-ws-e)).
- **Which stack somax and spectraldiffx belong to.** Rule 5 of the
  [package boundaries](geomodels/package_boundaries.md#pb-rules) puts somax
  in GeoModels and spectraldiffx and finitevolx in Numerics. methanex is
  still unplaced.
:::

:::{dropdown} Sequencing
:open:
- **When does pipekit-jax's operator layer land?** pipekit-jax 0.0.2 has
  shipped, as a model wrapper and a diffrax forward model
  ([Report 5](reports/05_pipekit_jax.md#r5-ws-b)); the operator layer is
  still proposed. The [six-step cycle](#sec-six-steps) gives it a concrete
  driver: differentiable step-3 emulators make step 4 (gradient-based
  inverse problems) tractable at scale. Open until such a project appears.
- **Which in-situ catalog backend comes first?**
  The scoping proposal behind [Report 6](reports/06_geocatalog.md) leaned
  to Argo (via argopy), but
  ICOADS, SOCAT and the World Ocean Database matter for other domains.
  Realistically one per quarter, driven by projects.
:::

:::{dropdown} Design
- **The Slice Protocol refactor.** `GeoSlice` is the only slice type today.
  Adding `ProfileSlice` and `TrajectorySlice` is the right structural move,
  but it is two weeks of work rippling across geocatalog, geotoolz, xrtoolz
  and pipekit-train. Better done before more code commits to `GeoSlice`.
- **Where matchup lives.** `geocatalog.queries.matchup()` is the framework
  function, with carrier-specific helpers in xrtoolz and geotoolz. Does
  matchup need its own catalog type? Probably; not yet scoped.
- **Backend choice in pipekit-train.** Equinox + Optax and the Bayesian
  backends ship; Lightning and Keras 3 are scaffolds ([Report 11](reports/11_pipekit_train.md)). The intended split: Lightning for PyTorch models, Equinox +
  Optax for JAX-traceable pipelines, Keras 3 for multi-backend. Worth stating
  per use case.
:::

:::{dropdown} Operational
- **Monorepo or polyrepo?** Monorepo for the framework and infrastructure
  (uv workspaces); the JAX-native algorithm and domain libraries stay in
  their own repositories and ship adapter modules.
- **One documentation site or many?** One site for the monorepo packages,
  separate sites for the algorithm libraries, and one research journal that
  cuts across everything.
- **CI per package or per monorepo?** Per-package CI inside the monorepo,
  plus one weekly integration matrix pinning every package to HEAD.
  Releases by git tag (`geocatalog-v0.2.0`).
:::

:::{dropdown} Scientific
- **Matchup UQ.** `MatchupPair.representativeness_uncertainty` is easy to
  declare and hard to compute; v0.1 ships a placeholder.
- **Cascading uncertainty.** A five-stage emission chain accumulates
  uncertainty across stages. Predicting that honestly is open research.
- **Probabilistic event detection.** "A plume happened" is binary; "a 73 %
  chance a plume happened" is much harder, and current event metrics assume
  the binary case.
:::

:::{dropdown} Sustainability
- **Solo maintenance.** About twenty packages is a lot even with monorepo
  discipline. Scope discipline is the operating constraint: each phase
  delivers standalone value, and no layer needs the full stack to exist.
- **Community adoption.** If oceanbench adopts pipekit-evaluate's benchmark
  contracts, the framework is validated; if not, it stays personal
  infrastructure. When to start reaching out is open.
:::

(sec-plan)=
## A plan

Each phase produces packages that are useful on their own ([](#tbl-phases)).

:::{table} Phases at a glance.
:label: tbl-phases

| Phase | Scope | Status | Time |
|---|---|---|---|
| 1 | Numerical foundation: gaussx (v0.0.6), geonnax, kernellib, spectraldiffx, finitevolx, optax_bayes, eqx-trainer | gaussx done; the rest drafts | ongoing |
| 2 | Algorithms: pyrox, filterax, vardax, gauss_flows, xtremax | in progress | each at its own pace |
| 3 | Domain models: somax, plumax, radtrax and methanex (both planned) | some done | project-driven |
| 4 | Pipeline framework | **current focus** | ~3–4 months |
| 5 | Evaluation, benchmarks and cycle discipline | next | ~2 months |
| 6 | Deferred items, each with a trigger | when projects demand | — |
| 7 | Polish and community | open-ended | — |
:::

### Phases 4 and 5, month by month

::::{tab-set}

:::{tab-item} Month 1
- pipekit 0.0.2: framework core, fourteen groups and state primitives
  (shipped; [Report 2](reports/02_pipekit_surface.md#r2-inventory))
- pipekit-array 0.0.2: Array API operators (shipped, phase A of four;
  [Report 3](reports/03_sister_libraries.md#r3-array))
- the pipekit-ecosystem monorepo with uv workspaces

**Gate:** A1 config round-trip ([](#tbl-gates)).
:::

:::{tab-item} Month 2
- geocatalog v0.1: InMemory and DuckDB backends, basic queries
- geopatcher v0.1: the four-axis framework and five Field adapters
- geotoolz and xrtoolz refactored onto the new framework

**Gates:** B1 backend parity, C1 patch identity.
:::

:::{tab-item} Month 3
- geocatalog's state profile (`geocatalog.states`) and xrtoolz forecast
  alignment
- pipekit-cycle 0.0.2: `Cycle`, `DACycle` and the protocols (shipped;
  [Report 10](reports/10_pipekit_cycle.md))
- adapter modules in filterax, vardax and plumax

**Gates:** B2 state rows, D1 kriging oracle, D2 Kalman oracle.
:::

:::{tab-item} Month 4
- pipekit-train 0.0.2 ships Equinox + Optax and the Bayesian backends;
  Lightning and Keras 3 are scaffolds ([Report 11](reports/11_pipekit_train.md))
- pipekit-experiment 0.0.2 ships the registry and the tracker protocol;
  MLflow and W&B trackers are planned ([Report 12](reports/12_pipekit_experiment.md))
- integration testing across all packages

**Gate:** E1 registry round-trip.

**After Phase 4:** the L0–L4 inference loop is structurally complete. ML
training and serving still need work, but the pipelines exist.
:::

:::{tab-item} Month 5
- pipekit-evaluate v0.1: framework, cheap metrics, `EvaluationReport`
- leakage-aware splits: block, platform and causal splits of catalog rows in
  geocatalog, and the window guard in time units in pipekit-train
- standard baselines in xrtoolz: persistence, climatology, OI
- lineage in pipekit-experiment's content hash, with a dependents query: the
  six-step cycle's lineage ([Report 17](reports/17_cycle.md#r17-ws-b))

**Gate:** F1 metric identities.
:::

:::{tab-item} Month 6
- particle tracking (`xrtoolz.lagrangian`) and event detection
  (`xrtoolz.events`)
- `BenchmarkContract` with transition cells and an oracle strength
  ([Report 17](reports/17_cycle.md#r17-ws-a))
- several references per run in the benchmark runner
  ([Report 15](reports/15_benchmarking.md#r15-ws-d))
- the first worked cycle (likely methane L1→L2 retrieval, steps 1–5)
- the first worked benchmark (SSH, from the [benchmarks gallery](benchmarks_gallery.md#gal-1-1))

**Gate:** G1 end to end.

**After Phase 5:** evaluation is first-class, benchmarks are
content-addressed contracts, and the six-step cycle is tractable end to end.
:::
::::

### Deferred until a project needs it

[](#tbl-deferred) lists what waits for a project, and what pulls it in.

:::{table} Phase 6: what is deferred, and what triggers it.
:label: tbl-deferred

| Item | Trigger |
|---|---|
| pipekit-jax's operator layer (operators as PyTrees; 0.0.2 ships only a model wrapper and a diffrax forward model, [Report 5](reports/05_pipekit_jax.md#r5-ws-b)) | the first project needing differentiable retrievals at scale or JIT-compiled inference |
| Deployment adapters (`pipekit.deploy.fused`, `.litserve`, `.modal`) | the first deployment beyond hand-shimming |
| Slice Protocol refactor (`ProfileSlice`, `TrajectorySlice`) | the first project needing first-class in-situ catalogs (likely the BGC benchmark) |
| More in-situ backends (Argo, ICOADS, SOCAT, …) | one per quarter, project-driven |
:::

Phase 7 covers the documentation site, a research journal, the first
external benchmark adoption (OceanBench, WeatherBench-Ocean?) and capstone
tutorials per domain.

(sec-gates)=
### Gates

A month is done when its gates pass, not when its packages exist
([](#fig-roadmap)). Every pass condition is a number or a closed form, and
several reuse the walkthrough's closed-form stages as oracles
([](#tbl-gates)).

```{figure} images/overview/roadmap.svg
:label: fig-roadmap
:alt: Phases 4 and 5 as six monthly boxes from top to bottom, each followed by a diamond naming the gates that must pass before the next month starts. Phases 1 to 3 run alongside and feed adapters into month 3; phases 6 and 7 follow the last gate.
:width: 100%

**The framework ships first; every month closes on its gates.**

Phases 1–3 run alongside and meet the chain when the DA adapters land in
month 3. Phases 6 and 7 wait for projects to pull them.
```

:::{table} Gates: what must pass before each month closes.
:label: tbl-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Config round-trip | A | For every shipped operator not marked `forbid_in_yaml`, `loads(dumps(op))` has the same config hash and returns the same output on a fixed input; once the topology serialiser lands, the same holds for a `Graph` (as [Report 9's A1](reports/09_use_cases.md#r9-gates)) |
| B1 Backend parity | B | The InMemory and DuckDB geocatalog backends return the identical slice set for the same bbox and time query |
| B2 State rows | B | A cycle's ensemble history, written as one file per run and member and indexed with `build_state_catalog`, loads back with `load_states` with no run dropped and equal to the history to 0 ([Report 13](reports/13_state_catalog.md#r13-gates), gates A2 and B1) |
| C1 Patch identity | C | An identity model through patch → stitch with Hann-window overlap-add reproduces the input to 1e-6 in float64, away from scene edges |
| D1 Kriging oracle | D | `KrigingOperator`'s mean and variance match {eq}`eq-kriging-mean` and {eq}`eq-kriging-var` to 1e-6 in float64 |
| D2 Kalman oracle | D | With linear 𝓜 and H, no inflation and no localisation, the square-root ensemble analysis mean matches the Kalman update computed from the same ensemble's sample mean and covariance, to 1e-6 in float64 |
| E1 Registry round-trip | E | Loading a stored operator returns one with the same content hash and bitwise-equal outputs |
| F1 Metric identities | F | CRPS {eq}`eq-crps` with one member equals absolute error exactly; RMSE of a field against itself is 0 |
| G1 End to end | all | Stages 1–5 of the walkthrough run from one YAML file. A cold rerun of the frozen inference artifact reproduces the analysis history's hash, and rerunning the frozen benchmark (its task, estimators, scorers and pinned TCCON references) on that history reproduces the report's hash |
:::

(sec-risks)=
### Risks

- **The schedule slips.** Six months for Phases 4–5 is optimistic for one
  person; eight to twelve is realistic. Fallback: every month ships a usable
  package, so nothing depends on the whole plan finishing.
- **Phase 6 is needed early.** A project may need pipekit-jax or in-situ
  catalogs before the framework is ready. Fallback: wrap jitted functions in
  pipekit-jax's `JaxModelOp` until the full backend lands, and jump phases;
  the sequence is a default, not a contract.
- **The Slice refactor ripples late.** The longer `GeoSlice` stays concrete,
  the more code binds to it. Fallback: introduce a `Slice` protocol in month
  2 with `GeoSlice` as its only implementation, and add profile and
  trajectory slices later.
- **pipekit-train becomes three half-backends.** Equal status for Lightning,
  Equinox and Keras 3 triples maintenance. Fallback: one default backend per
  use case; the others are community-supported.
- **Nobody else adopts it.** Adoption takes years. Fallback: GeoBench is the
  first consumer, so the substrate earns its keep without outside users.

(sec-reading)=
## Where to read next

(sec-conventions)=
### How the pages are written

Each page checks a design, a scoping report or a source note, against the
code pinned in `references/`, and the code takes precedence. A **Differs**
box marks each place where the code, or a check the page runs, disagrees
with the design. Gaps that matter become workstreams with proposals; the
rest go to open questions or upcoming work. Every number on a page comes
from a notebook that asserts it. Pseudocode follows one convention: a shapes
header names the dimensions, the equation sits in a comment above the line
that computes it, shapes go on the right, and every call into the stack
carries a status marker ([markers](overview_examples.md#ex-markers)).

### The pages

Start with the [capability index](api/capabilities.md) and the
[overview examples](overview_examples.md); then read by question.

**Across the stack**

- [Report 9](reports/09_use_cases.md): twenty use cases mapped onto the libraries as they ship.
- [Report 16](reports/16_geo_data_stack.md): the geo data stack, the ownership rules and the realignment plan across 18 distributions.

**pipekit**

- [Report 1](reports/01_toolz_primer.md): the toolz lineage, and why pipekit's operators are classes.
- [Report 2](reports/02_pipekit_surface.md): what pipekit 0.0.2 ships, in fourteen groups (A–N).
- [Report 5](reports/05_pipekit_jax.md): jit, vmap and grad through pipelines, and what pipekit-jax ships.
- [Report 10](reports/10_pipekit_cycle.md): pipekit-cycle, time-stepping and data assimilation.
- [Report 11](reports/11_pipekit_train.md): pipekit-train, training for emulators and amortized inference.
- [Report 12](reports/12_pipekit_experiment.md): pipekit-experiment, the registry, trackers and reproducibility artifacts.
- [Report 14](reports/14_pipekit_evaluate.md): pipekit-evaluate, multidimensional evaluation.

**geotoolz and xrtoolz**

- [Report 3](reports/03_sister_libraries.md): the sister libraries, pipekit-array, geotoolz and xrtoolz.
- [Report 6](reports/06_geocatalog.md): geocatalog, the spatiotemporal index.
- [Report 7](reports/07_geopatcher.md): geopatcher, the four-axis patcher.
- [Report 13](reports/13_state_catalog.md): L3–L5 indexing, as geocatalog's state profile.

**Benchmarking**

- [Report 15](reports/15_benchmarking.md): benchmarks as content-addressed contracts.
- [Report 17](reports/17_cycle.md): the six-step cycle as the benchmark ladder, step 2 as oracle.
- [Benchmarks gallery](benchmarks_gallery.md): eighteen draft benchmark entries in five domains.

**GeoModels**

- [Package boundaries](geomodels/package_boundaries.md): what moves from geotoolz and plumax into plumax and a new radtrax.
- [radtrax · 1](geomodels/radtrax/01_column_retrieval.md): the Beer–Lambert column retrieval, term by term.
- [radtrax · 2](geomodels/radtrax/02_mbmp.md): the SBMP, MBSP and MBMP family of multi-band retrievals.
- [radtrax · 3](geomodels/radtrax/03_design.md): radtrax's design and roadmap (formerly rtmx).
- [plumax · 1](geomodels/plumax/01_emission_rate.md): emission rate from first principles, from the plume to the IME.

(sec-summary)=
## In summary

GeoStack is one operator algebra from structured linear algebra to indexed
multi-tier data products, organised by three orthogonal axes and split into
the three stacks of [the decision](#sec-decision). Classical methods stay as
oracles, benchmarks are content-addressed contracts, and differentiability
lives only in the JAX packages. The plan is a default sequence that projects
will reshape: pick a page from [Where to read next](#sec-reading) and come
back with questions, since no part of the stack requires the rest.

## Glossary

:::{glossary}
carrier
: The data structure an operator acts on: NumPy or JAX arrays, a
  {term}`GeoTensor`, or an xarray object. pipekit operators are
  carrier-agnostic.

GeoTensor
: georeader's raster carrier: an array with its geotransform and CRS. The
  carrier for L0–L2.

GeoSlice
: geocatalog's wire format for one spatiotemporal slice of an observation
  catalog.

state profile
: geocatalog's reserved columns for model states (run time, ensemble
  member, producer hash and parent state) and the loader and queries in
  `geocatalog.states` that use them. It replaces the planned `statecatalog`
  package and its `StateSlice`.

matchup
: Pairing observations from different sources that coincide in space and
  time; the operation where the satellite and in-situ streams converge.

oracle
: The classical method (step 2) used as the trusted reference to validate emulators
  and amortized predictors.
:::
