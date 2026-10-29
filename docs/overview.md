---
title: GeoStack overview
subtitle: A JAX-native research stack for geophysical modeling, inference and data assimilation
short_title: Overview
description: The entry point to GeoStack. Three organising axes, the four-layer stack, an end-to-end methane walkthrough, the open questions and a plan.
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

There is no Keras for geophysics. GeoStack is our attempt at the missing connective tissue, which is one operator-graph abstraction that runs from structured linear algebra up to end-to-end pipelines, with content-addressed data and models throughout and machine learning pluggable at every step. This page is the map. It ties together the fifteen scoping reports, the GeoModels pages and the benchmarks gallery ([Where to read next](#sec-reading)), and it ends with the open questions and a plan.

+++

:::{tip} How to read this page
:class: dropdown
- **Five minutes:** read the [elevator pitch](#sec-pitch), look at {numref}`fig-lifecycle` and {numref}`fig-tier-stack`, and skim the [plan](#sec-plan).
- **Building something:** start with the [walkthrough](#sec-walkthrough) and the [overview examples](overview_examples.md), then search the [capability index](api/capabilities.md) before you write new code.
- **Reviewing the design:** go straight to the [decision](#sec-decision), the [ownership rules](#sec-ownership), the [gates](#sec-gates) and the [open questions](#sec-open-questions).
:::

(sec-decision)=
## The decision

GeoStack is one operator algebra across three stacks: **GeoML** (numerics and inference), **GeoModels** (forward models) and **GeoStack proper** (data, pipelines, evaluation). None of them merges into another. The JAX libraries stay pipeline-free, and they only reach pipekit through thin `adapters/pipekit.py` modules.

The work splits into six workstreams. Each one makes one decision, and its letter tags every `proposed (X)` call on the [examples page](overview_examples.md) and every gate in {numref}`tbl-gates`.

- **A · One framework.** Today every library invents its own pipeline glue. pipekit replaces it with one `Operator` protocol, `Sequential` and `Graph` composition, config round-tripping and sandboxed YAML.
- **B · Indexed data at both ends of the lifecycle.** Observations can be catalogued whereas model states are scattered files. `geocatalog` indexes both: L0 to L2 observations as rows queried with `GeoSlice`s, and L3 to L5 model states through its {term}`state profile` (run time, member, producer and parent columns, in `geocatalog.states`). The readers split by level, with `georeader` below L3 and `xrreader` from L3 up.
- **C · Two carriers, one algebra.** `geotoolz` (GeoTensor) and `xrtoolz` (xarray) share the operator algebra and the patcher, and only the substrate differs.
- **D · Data assimilation as a cycle.** The `ForwardModel`, `ObservationOperator` and `AnalysisStep` protocols of pipekit-cycle are the only contract. filterax, vardax and plumax satisfy them through adapters, so an emulator can replace a physics model without touching the cycle.
- **E · Training with lineage.** pipekit-train wraps Equinox + Optax today, with NumPyro and BlackJAX for Bayesian fits, and Lightning and Keras 3 are planned. pipekit-experiment adds a tracker protocol and a content-addressed `ModelRegistry`.
- **F · Evaluation as infrastructure.** pipekit-evaluate scores every cell of the benchmark cube against an `Oracle`, with the metrics from `xrtoolz.metrics`. The Unit × Lens × Stage tags and a content-addressed evaluation report are proposed ([Report 14](reports/14_pipekit_evaluate.md)).

**End state.** GeoML owns linear algebra, kernels and inference, and it never imports pipekit outside an adapter. GeoModels owns the physics and only exposes it through adapters. pipekit owns composition, cycles, training and evaluation. geocatalog owns the one index, for observations and (through its state profile) for model states, whereas geotoolz and xrtoolz own the carrier-specific operators. Orchestration and serving stay external.

(sec-pitch)=
## The elevator pitch

> A vertically integrated, JAX-native research software stack for geophysical
> modeling, inference and data assimilation: from structured linear algebra
> primitives, through composable operator graphs, to indexed multi-tier data
> products, with ML pluggable at every level.

Great JAX numerics and great xarray pipelines exist today. However, what is missing is the operator-graph abstraction which connects them, with content-addressed data and models throughout.

(sec-problem)=
## The shape of the problem

There are five gaps which motivate everything below. The first four are old, whereas the fifth has only become visible recently.

::::{grid} 1 1 2 2

:::{card} 1 · Scattered numerics
The same linear algebra, kernel methods and filtering routines are reimplemented in every project. There are no shared primitives.
:::

:::{card} 2 · ML tooling ignores classical numerics
Keras and PyTorch layers assume supervised learning on tensors. They do not express linear operators, PDE solvers, state estimation or uncertainty propagation.
:::

:::{card} 3 · Monolithic domain software
Ocean models, retrieval algorithms and DA systems are dense, tightly coupled and hard to recombine.
:::

:::{card} 4 · No connective tissue
Even with good numerics and good ML, there is no Keras-like pipeline to compose them for real geophysical work.
:::

:::{card} 5 · No indexed model-state tier
Observations are catalogable (STAC, GeoParquet). However, forecasts, analyses and reanalyses are scattered files with bespoke conventions, so the L3 to L5 half of the lifecycle has nothing like the catalogs that organise L0 to L2.
:::
::::

The stack addresses all five.

(sec-axes)=
## Three organising axes

Three axes structure the framework ([](#tbl-axes)). They are orthogonal, so every operation sits somewhere on all three.

:::{table} The three axes and what each one orders.
:label: tbl-axes

| Axis | Orders | Figure |
|---|---|---|
| 1 · Modeling cycle | the research loop: model, observe, compare, refine | {numref}`fig-modeling-cycle` |
| 2 · Data lifecycle | data maturity, from raw telemetry (L0) to derived products (L5) | {numref}`fig-lifecycle`, {numref}`fig-lifecycle-engineering` |
| 3 · Six-step cycle | the journey from a simple model to amortized inference | {numref}`fig-six-steps` |
:::

(sec-modeling-cycle)=
### The modeling cycle (axis 1)

```{figure} images/overview/modeling_cycle.svg
:label: fig-modeling-cycle
:alt: The real-world system feeds two tracks. Modeling runs conceptual model, equations and code, predictions. Observing runs sensor, measurements, observations. Both meet at compare. A random misfit goes to estimate (state by data assimilation, parameters by calibration), whose inner loop updates the predictions; a systematic misfit goes to refine the model, whose outer loop returns to the conceptual model and, dashed, to the sensor.
:width: 70%

**The modeling cycle runs on two parallel tracks and two loops.**

The modeling track turns a conceptual model of the real-world system into
equations and code that make predictions. The observing track measures the
same system with sensors and turns measurements into observations. The
tracks meet where predictions are compared with observations, and the misfit
decides which loop runs. A random misfit, one the error model explains, runs
the fast inner loop: **state estimation** (data assimilation) updates the
state $\vect{x} \in \mathbb{R}^{D_x}$, **parameter estimation** (calibration) updates the parameters
$\gvect{\theta} \in \mathbb{R}^{D_\theta}$, and the same model runs again. A systematic misfit runs the slow
outer loop: the model's structure changes, or the observing system does
(more or better data, better processing).
```

:::{important} Start simple
Add complexity only when validation demands it. This is the research loop, and every package in the stack exists to make one or more of its steps composable.
:::

(sec-lifecycle)=
### The L0–L5 data lifecycle (axis 2)

```{figure} images/overview/lifecycle_science.svg
:label: fig-lifecycle
:alt: Six maturity levels, L0 to L5, across the top, each step named on its arrow. Below, telemetry becomes one stream of radiances, which retrieval fans out into ocean, atmosphere, land and ice lanes that run through analysis and reanalysis to dashed derived products.
:width: 100%

**The data lifecycle, science axis.**

One stream of radiances fans out into ocean, atmosphere, land and ice at
retrieval, and ends in derived products such as extremes (L5, dashed:
examples still to be defined). {numref}`fig-lifecycle-engineering` maps the
same levels to the tools that handle them.
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
The `LEVEL_NAMES` of pipekit-evaluate call L2 "gap-filled observations", whereas [](#tbl-levels) has retrieved observations with gaps. So the pages that follow the code ([Report 15](reports/15_benchmarking.md#r15-framework), the [gallery](benchmarks_gallery.md)) label gap filling L1 → L2, while this page puts it at L2 → L3 ([stage 3](#sec-stage-gapfill)). The two are not reconciled yet, and [Report 17 A1](reports/17_cycle.md#r17-ws-a) proposes the rename to `"retrieved_obs"`.
:::

**The science axis** runs left to right. Until retrieval it is one stream, as the same L1 radiances feed every geoscience. Retrieval is where it fans out into ocean, atmosphere, land and ice variables, and each domain then carries its own analysis and reanalysis or forecast. L5 holds the products derived from those states, such as extremes (marine heatwaves, droughts, storms) or impact indicators. Which products belong in L5 is still an open question.

```{figure} images/overview/lifecycle_engineering.svg
:label: fig-lifecycle-engineering
:alt: The same six levels across the top. Below, a compute row names what performs the step into each level (radtrax at L2, GeoML at L3, GeoModels with data assimilation at L4, xtremax at L5), then rows for compose (pipekit, all levels), transform (geoops, xrtoolz), patch (geopatcher), carry (GeoTensor, xarray), read (georeader, xrreader), index (geocatalog, geocatalog.states) and store (geocloud, all levels), split by a dashed line where the carrier switches from GeoTensor to xarray.
:width: 100%

**The data lifecycle, engineering axis.**

The top row names what performs each step; the rows below map every level
to the tool that composes, transforms, patches, carries, reads, indexes and
stores it, switching carrier from GeoTensor to xarray between L2 and L3.
```

**The engineering axis** runs top to bottom, from computing a level down to storing it, and it maps every level to the GeoStack tool that handles it.

- **Compute:** what performs the step into each level. radtrax retrieves (L1 → L2). GeoML grids and gap-fills (L2 → L3: pyrox, kernellib, the optimal interpolation of vardax). GeoModels with data assimilation model and assimilate (L3 → L4: somax and plumax with filterax and vardax), which is the inner loop of {numref}`fig-modeling-cycle`. Lastly, xtremax derives extremes (L4 → L5).
- **Compose:** `pipekit` wires every level into operator graphs, cycles and training loops, whatever the carrier.

The rows below split where the {term}`carrier` changes:

- **Observations (L0 to L2):** `georeader` reads them into a {term}`GeoTensor`, `geocatalog` indexes them as {term}`GeoSlice`s and `geotoolz` transforms them.
- **Model states and derived products (L3 to L5):** `xrreader` reads them into xarray, `geocatalog.states` indexes them in the same catalogs as state rows (its {term}`state profile`), and `xrtoolz` transforms and verifies them.
- **Both carriers:** `geopatcher` patches fields whatever the carrier, and `geocloud` stores both on object storage (COG for observations, Zarr or NetCDF for states).

:::{aside}
`xrreader` exists today as the `xrtoolz-reader` package (it imports as `xrreader`).
:::

ML enters at every transition.

(sec-two-streams)=
### Two streams converge

The lifecycle is not one chain. A top-down satellite stream and a bottom-up in-situ stream converge between L2 and L3 ([](#fig-two-streams)).

```{figure} images/overview/two_streams.svg
:label: fig-two-streams
:alt: The satellite chain L0 to L5 on the left; in-situ networks for ocean, atmosphere, land and ice at the bottom right. L2 and in-situ meet at a matchup node that feeds L3 and yields matchup pairs for validation and bias correction; in-situ also enters L4 directly through assimilation.
:width: 100%

**Satellite and in-situ observations converge through matchup.**

In-situ networks in every domain join the satellite chain through
spatiotemporal {term}`matchup` between L2 and L3, and enter L4 directly
through assimilation. They have no L0 or L1 equivalent. Matchup also yields
the pairs that validate and bias-correct the satellite stream: the misfits
that drive the loops of [](#fig-modeling-cycle).
```

The matchup pattern is canonical and it deserves its own operator family ([stage 2](#sec-stage-matchup) runs it, and where it lives is an [open question](#sec-open-questions)).

(sec-three-layers)=
### Three layers of transformation

A third lens joins the cycle and the lifecycle, which is what we *do* to the data on its way into a model ([](#fig-three-layers)).

```{figure} images/overview/three_layers.svg
:label: fig-three-layers
:alt: Scientific data (observations as GeoTensors, states as xarray) becomes ML-ready tensors with provenance through harmonisation and patching (geopatcher), which geonnax encoders compress into latent embeddings; a dashed decode arrow maps latents back to fields. Each layer lists what it is, its array shape and the packages that own it.
:width: 100%

**Scientific data, ML-ready data, embeddings, and who owns each.**

The first step is data engineering (GeoStack), the second representation
learning (GeoML). The middle layer is the handoff that makes training
reproducible: cached, CF-compliant tensors with content-addressed splits.
Decoders run the chain backwards, which is what emulators and generative
models do.
```

(sec-six-steps)=
### The six-step cycle (axis 3)

The six-step cycle is the journey from research to production ([](#fig-six-steps)). Every domain library follows this cycle implicitly, and the framework makes it explicit and composable.

```{figure} images/overview/six_steps.svg
:label: fig-six-steps
:alt: Five steps in a row, from simple model to amortized inference, each with its defining equation. Step 1 simulates (θ, y) pairs that train steps 3 and 5; step 2 validates steps 3 to 5. A shared improve step under steps 2 to 5 lists each step's ground truth, the step before it. An axis below shows the cost per query falling from hours to milliseconds.
:width: 100%

**From a simple model to amortized inference, with one shared improve step.**

Step 2, the classical method, is the {term}`oracle` that validates steps
3–5; it is the inner loop of [](#fig-modeling-cycle). Step 1 also
simulates the (θ, y) pairs that train the emulator and the amortized map.
Step 6 is shared: any of steps 2–5 can be improved, using the previous step
as ground truth, and changing the structure is the outer loop. Moving right,
the cost of each query falls from hours to milliseconds because the training
cost is paid once, up front.
```

In symbols, let $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ be the quantity we want (an emission rate, a model state), $f$ the physical forward model, $h$ the observation operator, $\vect{y} \in \mathbb{R}^{D_y}$ the observations and $\gvect{\varepsilon} \in \mathbb{R}^{D_y}$ their error, with covariance $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{D_y \times D_y}$. In the six-step cycle (and in its figures), $\gvect{\theta}$ stands for whatever we infer, so it covers both the state $\vect{x}$ and the parameters of [](#fig-modeling-cycle). Step 1 is the generative story:

$$
\vect{y} = h(f(\gvect{\theta})) + \gvect{\varepsilon} ,
$$ (eq-generative)

where the error is Gaussian,

$$
\gvect{\varepsilon} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon) .
$$ (eq-generative-noise)

Step 2 inverts it with classical, model-based inference. Bayes' rule with a prior $p(\gvect{\theta})$ defines the target:

$$
p(\gvect{\theta} \mid \vect{y}) \propto p(\vect{y} \mid \gvect{\theta}) p(\gvect{\theta}) .
$$ (eq-posterior)

Computing {eq}`eq-posterior` is only exact in tractable cases, such as a linear-Gaussian model solved with the Kalman filter or a GP. The usual step-2 methods (a finite-ensemble EnKF, 3D- or 4D-Var) approximate it, and slowly. So step 2 is the oracle because it is the most trusted reference we have, and not because it is exact.

Step 3 trains an emulator $\hat{f}(\cdot; \gvect{\theta}_\text{emu})$ with weights $\gvect{\theta}_\text{emu} \in \mathbb{R}^{D_\text{emu}}$ on step-1 simulations by {eq}`eq-emulator`, and step 4 reruns step 2 with $\hat{f}$ in place of $f$:

$$
\gvect{\theta}_\text{emu}^\star = \mathrm{argmin}_{\gvect{\theta}_\text{emu}} \ \mathbb{E}_{\gvect{\theta} \sim p(\gvect{\theta})} \left[ \left\Vert f(\gvect{\theta}) - \hat{f}(\gvect{\theta}; \gvect{\theta}_\text{emu}) \right\Vert_2^2 \right] .
$$ (eq-emulator)

Step 5 skips the inversion. It trains a conditional density $q(\gvect{\theta} \mid \vect{y}; \gvect{\nu})$ with weights $\gvect{\nu} \in \mathbb{R}^{D_\nu}$, which maps observations straight to an approximate posterior, on pairs $(\gvect{\theta}, \vect{y})$ drawn from the generative model {eq}`eq-generative`, by {eq}`eq-amortized`:

$$
\gvect{\nu}^\star = \mathrm{argmin}_{\gvect{\nu}} \ \mathbb{E}_{(\gvect{\theta}, \vect{y}) \sim p(\gvect{\theta}, \vect{y})} \left[ -\log q(\gvect{\theta} \mid \vect{y}; \gvect{\nu}) \right] ,
$$ (eq-amortized)

so that the trained density approximates the posterior,

$$
q(\gvect{\theta} \mid \vect{y}; \gvect{\nu}^\star) \approx p(\gvect{\theta} \mid \vect{y}) .
$$ (eq-amortized-approx)

[](#tbl-steps) says where each step lives in the stack. There are two structural commitments which fall out of the cycle:

::::{grid} 1 1 2 2

:::{card} Step 2 is the oracle
Classical methods (`filterax`, `vardax`, `pyrox`) are not legacy code waiting to be replaced by neural emulators. They become *reference generators* that validate steps 3 to 5. So the classical path stays alive, promoted to oracle.
:::

:::{card} Benchmarks formalise understanding
Writing a benchmark contract (carriers, references, metrics, splits, baselines, expected failure modes) *is* formalising what you understand about the problem. Every transition between steps is a benchmarkable contract.
:::
::::

:::{table} Where each step lives in the stack.
:label: tbl-steps

| Step | Lives in |
|---|---|
| 1 · Simple model | GeoModels (`somax`, `plumax`, `radtrax`) |
| 2 · Model-based inference | `pipekit-cycle.DACycle` with classical analysis steps |
| 3 · Emulator | `pipekit-train.SimulationDataset`, registered in `pipekit-experiment.ModelRegistry` |
| 4 · Emulator-based inference | `pipekit-cycle.DACycle` with `NeuralForward` |
| 5 · Amortized inference | `pipekit-train` with `gauss_flows` or `pyrox` |
| 6 · Improve | `pipekit-evaluate.Benchmark` reruns, compared row by row, with lineage in `pipekit-experiment` |
:::

(sec-axes-relate)=
### How the axes relate

The axes form a cube of operations. At any point you are somewhere on the modeling cycle (research stage), somewhere on the lifecycle (data maturity) and somewhere on the six-step cycle (production maturity). [](#fig-axes-matrix) crosses the lifecycle transitions with the steps.

```{figure} images/overview/axes_matrix.svg
:label: fig-axes-matrix
:alt: A grid with the five lifecycle steps as rows, from read and calibrate (detector counts to radiance) to derive (heatwave return levels), each with a running example, and steps 1 to 5 as columns headed by their equations. Each cell names a method and what it does on the example; the step-2 column is highlighted as the oracle. Arrows above show step 3 scored against step 1, and steps 4 and 5 against step 2.
:width: 100%

**Every lifecycle step crossed with every six-step stage is a concrete method.**

Each row follows one running example from a simple model to an amortized
map: radiance from detector counts, methane from shortwave infrared, sea
level from altimeter tracks, the ocean state from sea-level maps, and
heatwave return levels.
```

**Every cell is a benchmarkable artifact,** scored against the earlier step that computes the same thing (an emulator against the simulator it replaces, and emulator-based and amortized inference against the posterior of the step-2 oracle). The benchmark ladder of pipekit-evaluate ships this rule ([Report 17](reports/17_cycle.md#r17-stage-1)). The whole lifecycle is a nested set of six-step cycles, each gated by benchmarks. This is what makes the framework improvable in practice, since improving a cell re-runs its downstream column, the lineage is content-addressed, and nothing is opaque.

(sec-stack)=
## The big picture stack

We show the stack at two zoom levels: the layers first, then the packages inside GeoStack.

(sec-tiers)=
### Four layers

```{figure} images/overview/tier_stack.svg
:label: fig-tier-stack
:alt: Four owned layers between external foundations at the bottom and external orchestration and serving at the top. GeoStack (data) and GeoNumerics (numerics) sit side by side at the bottom, GeoML (learning and inference) on both, GeoModels (physics) on top; imports point down. georeader is marked third-party and read-only.
:width: 100%

**The stack in four layers.**

Each layer has one job: GeoStack moves data in and out, GeoNumerics does
structured numerics, GeoML learns and infers, GeoModels simulates physics.
Imports point down, and GeoStack and GeoNumerics never import each other
([](#fig-dependency-graph) and `tests/test_dependency_graph.py` check it).
Dashed rows are external; `georeader` belongs to GeoStack but is
spaceml-org's, read here and never written.
```

(sec-tier4)=
### GeoStack, zoomed in

Most of the current work happens in the GeoStack layer ([](#fig-tier4-map)).

```{figure} images/overview/tier4_map.svg
:label: fig-tier4-map
:alt: The GeoStack layer. The pipekit core (pipekit, pipekit-array, pipekit-jax) at the bottom; above it two carrier columns, GeoTensor for observations and xarray for states, in rows by job: read (georeader, geoproducts; xrreader), index (geocatalog; the proposed geocatalog.states), transform (geoops; xrtoolz, xrcore, xrgrad, xreinx, xrsklearn), patch (geopatcher, either carrier) and store (geocloud, either carrier). The carrier-neutral ML loops (pipekit-cycle, -train, -experiment, -evaluate) on top.
:width: 100%

**One pipekit core, two carriers, the same jobs on both.**

Row by row, the GeoTensor side and the xarray side do the same job on a
different carrier; patching and storage serve both. The ML loops on top
never see which carrier they run on.
```

(sec-dependencies)=
### The dependency graph

[](#fig-dependency-graph) shows which repositories import which.

```{figure} images/overview/dependency_graph.svg
:label: fig-dependency-graph
:alt: The 18 repositories in four layer bands: GeoModels on top, GeoML below, GeoNumerics and GeoStack side by side at the bottom. Edges point down from each repository to those it depends on or imports; dashed edges are optional extras; the geotoolz and pipekit cycle is drawn in red. georeader is marked third-party.
:width: 100%

**How the repositories import each other.**

Drawn from the generated graph on the
[dependency-graph page](api/dependency_graph.md), which
`tests/test_dependency_graph.py` keeps current; that page has the
package-level detail and the table of known violations. Solid edges are
dependencies or imports, dashed edges optional extras, and red edges a cycle.
```

(sec-ownership)=
### Ownership rules

A package only imports from the layers below it, so there are no circular dependencies. geotoolz and xrtoolz re-export the common framework symbols, so user code usually imports a single package. Each of the three stacks of [the decision](#sec-decision) has its own rules. The GeoStack rules (pipekit and its packages, geocatalog, geopatcher, geotoolz and xrtoolz) are in [Report 16](reports/16_geo_data_stack.md#r16-ownership), and the GeoModels rules (somax, plumax, radtrax, on top of the GeoNumerics layer) are on [the package boundaries page](geomodels/package_boundaries.md#pb-rules). The GeoML rules are only stated here: the JAX libraries touch pipekit only through an optional adapter module, and never in their core.

1. **gaussx** owns structured linear algebra and Gaussian primitives. It imports no GeoStack package.
2. **geonnax** owns basis functions, neural layers and coordinate transforms. somax and the other models compose them and never reimplement them.
3. **kernellib** owns covariance kernels and tapers. It imports no inference library.
4. **optax_bayes** owns the Bayesian learning rule as optax transforms. It imports no model library.
5. **spectraldiffx** and **finitevolx** own discretisations (spectral and finite-volume operators, no physics).
6. **pyrox** owns GP and Bayesian NN building blocks for NumPyro. Its pipekit surface (`KrigingOperator`) only lives in `pyrox.adapters.pipekit`.
7. **filterax** owns ensemble Kalman methods and **vardax** owns 3D/4D-Var. Each one only exposes an `AnalysisStep` through `adapters/pipekit.py`.
8. **gauss_flows** owns Gaussianization and normalizing flows, and **xtremax** owns extreme-value and point-process models. Neither imports pipekit outside an adapter.

(sec-walkthrough)=
## Walkthrough: methane attribution

This is the canonical example, a complete pipeline from satellite scenes to source emission estimates, with ML at several stages.

```{figure} images/overview/methane_walkthrough.svg
:label: fig-methane-walkthrough
:alt: Five science steps down the left, grouped into stages 1 to 4: catalog the scenes, retrieve methane (L1 to L2), match with in-situ, grid and gap-fill (L2 to L3), assimilate hourly (L3 to L4). Each arrow names the data that flows, with its level, carrier and shape; each step names its packages and where ML enters. EMIT scenes, TCCON columns and CAMS and aircraft references come in from the right. Along the bottom, stage 5's engineering steps: index outputs, evaluate, register and serve; evaluation feeds the misfit back to assimilation.
:width: 100%

**Eight steps from cataloguing scenes to serving estimates.**

Stages 1–4 are the science, stage 5 the engineering around it. Each arrow
names what flows to the next step; the dashed arrow is the inner loop of
[](#fig-modeling-cycle), the misfit that updates the state and parameters.
```

:::{seealso}
The [full pipeline code](#ex-methane), step by step, on the overview examples page.
:::

For the data, we use EMIT imaging-spectrometer scenes (L1 radiances) over an area of interest, TCCON column measurements as in-situ truth, and CAMS reanalysis and aircraft campaigns as evaluation references. The dimensions we use below are $B$ spectral bands, $N_\text{pix}$ pixels per scene (`H × W` in the code sketches), $M$ observations, $N$ grid cells, $N_\mathcal{E} = 40$ ensemble members $e \in \mathcal{E}$ (`N_e` in the code) and $T = 24$ hourly steps.

The eight steps of {numref}`fig-methane-walkthrough` group into five stages. Each stage states the problem it solves before the code does, and stages 3 and 4 have closed-form special cases which double as test oracles ({numref}`tbl-gates`).

(sec-stage-retrieve)=
### Stage 1 · Retrieve methane (steps 1–2: geocatalog → georeader → geotoolz)

**TL;DR.** Find every EMIT scene over the area, mask clouds, and turn each pixel's spectrum into a methane enhancement.

**Problem.** For pixel $n$, $\vect{y}_n \in \mathbb{R}^B$ is its radiance spectrum, $\gvect{\mu}_y \in \mathbb{R}^B$ is the scene-background mean, $\gmat{\Sigma}_y \in \mathbb{R}^{B \times B}$ is the background covariance and $\vect{J}_h \in \mathbb{R}^B$ is the methane target signature (unit absorption times $\gvect{\mu}_y$). The matched-filter enhancement estimate $\hat{\alpha}_n \in \mathbb{R}$ is the least-squares amplitude of $\vect{J}_h$ in the whitened residual:

$$
\hat{\alpha}_n = \frac{\vect{J}_h^\top \gmat{\Sigma}_y^{-1} (\vect{y}_n - \gvect{\mu}_y)}{\vect{J}_h^\top \gmat{\Sigma}_y^{-1} \vect{J}_h} .
$$ (eq-matched-filter)

The matched filter {eq}`eq-matched-filter` is the fast classical enhancement estimate, and its derivation is in [radtrax · 1](geomodels/radtrax/01_column_retrieval.md#rx1-stage-5). The full-physics retrieval (optimal estimation on a radiative-transfer model) is step 2 of the [six-step cycle](#sec-six-steps) for the L1→L2 cell ([](#fig-axes-matrix)). A learned retrieval takes the same slot in the pipeline, and optimal estimation stays as its oracle.

(sec-stage-matchup)=
### Stage 2 · Match with in-situ (step 3: geocatalog matchup)

**TL;DR.** Pair each TCCON measurement with the one satellite retrieval nearest to it in space and time; measurements with nothing nearby are dropped.

**Problem.** In-situ observation $j$ sits at location $\vect{s}_j \in \mathbb{R}^2$ and time $t_j \in \mathbb{R}$, satellite retrieval $r$ sits at $\vect{s}_r$ and $t_r$, and $d(\cdot, \cdot) \in \mathbb{R}_{\ge 0}$ is the great-circle distance. The candidates for $j$ are the retrievals inside both tolerances, {eq}`eq-matchup-window`:

$$
\mathcal{C}_j = \left\{ r : \vert t_j - t_r \vert \le 3\ \mathrm{h} \text{ and } d(\vect{s}_j, \vect{s}_r) \le 25\ \mathrm{km} \right\} .
$$ (eq-matchup-window)

If $\mathcal{C}_j$ is empty, then $j$ has no matchup. Otherwise $j$ keeps the single candidate with the smallest scaled distance {eq}`eq-matchup`, so no TCCON record is counted twice:

$$
r^\star(j) = \mathrm{argmin}_{r \in \mathcal{C}_j} \ \left(\frac{d(\vect{s}_j, \vect{s}_r)}{25\ \mathrm{km}}\right)^2 + \left(\frac{\vert t_j - t_r \vert}{3\ \mathrm{h}}\right)^2 .
$$ (eq-matchup)

Ties go to the earlier retrieval and then to the smaller catalog index, so the selection is deterministic.

The pairs serve two jobs, so we split TCCON once, by station, before anything is fitted. The pairs from the training sites are the (label, feature) data which a learned retrieval trains on, whereas the pairs from the held-out sites validate stage 1 and are the only TCCON references we use in stage 5. No site appears in both, so the held-out scores never include a label the model saw. How much of the disagreement between the pairs is representativeness error is an [open question](#sec-open-questions).

(sec-stage-gapfill)=
### Stage 3 · Grid and gap-fill (step 4: pyrox through pipekit)

**TL;DR.** Turn scattered retrievals into a regular map with an uncertainty for every grid cell.

**Problem.** $\vect{y} \in \mathbb{R}^M$ are the L2 retrievals, $\sigma_\varepsilon^2 \in \mathbb{R}_{>0}$ is their noise variance and $k$ is a Matérn-5/2 kernel with a 50 km length scale. $\mat{K}_{SS} \in \mathbb{R}^{M \times M}$ is $k$ between the $M$ observation locations $\mat{S}$, $\mat{K}_{*S} \in \mathbb{R}^{N \times M}$ is $k$ between the $N$ grid cells and the observation locations (with $\mat{K}_{S*} = \mat{K}_{*S}^\top$), and $\mat{K}_{**} \in \mathbb{R}^{N \times N}$ is $k$ between the grid cells. The GP posterior mean $\gvect{\mu}_* \in \mathbb{R}^N$ and variance $\vect{v}_* \in \mathbb{R}^N$ on the grid are:

$$
\gvect{\mu}_* = \mat{K}_{*S} (\mat{K}_{SS} + \sigma_\varepsilon^2 \mat{I}_M)^{-1} \vect{y} ,
$$ (eq-kriging-mean)

$$
\vect{v}_* = \op{diag}(\mat{K}_{**} - \mat{K}_{*S} (\mat{K}_{SS} + \sigma_\varepsilon^2 \mat{I}_M)^{-1} \mat{K}_{S*}) .
$$ (eq-kriging-var)

Kriging is the oracle for the L2→L3 cell, and DINEOF and neural gap-fillers are judged against it. Stage 4 assimilates this map, with $\gvect{\mu}_*$ as its observation and $\op{diag}(\vect{v}_*)$ as its observation-error covariance.

(sec-stage-da)=
### Stage 4 · Assimilate hourly (step 5: pipekit-cycle → plumax → filterax)

**TL;DR.** Step a transport model forward each hour and pull it towards the observations.

**Problem.** $\vect{x}_t \in \mathbb{R}^N$ is the methane state at hour $t$, $f$ is one plumax transport step of 3,600 s, $\mat{H} \in \mathbb{R}^{N \times N}$ is the column observation operator, $\vect{y}_t \in \mathbb{R}^N$ is the stage-3 map for hour $t$, and $\gmat{\Sigma}_\varepsilon \in \mathbb{R}^{N \times N}$ is the observation-error covariance (here $\gmat{\Sigma}_\varepsilon = \op{diag}(\vect{v}_*)$ from the stage-3 kriging). The background covariance $\gmat{\Sigma}_{x,t \mid t-1} \in \mathbb{R}^{N \times N}$ is estimated from the $N_\mathcal{E}$ members. Each cycle forecasts every member $e = 1, \dots, N_\mathcal{E}$ by {eq}`eq-forecast`:

$$
\vect{x}_{t \mid t-1}^{(e)} = f(\vect{x}_{t-1 \mid t-1}^{(e)}) ,
$$ (eq-forecast)

and then updates each member against its own perturbed copy of the observations (the stochastic EnKF), with $\gvect{\varepsilon}^{(e)} \sim \mathcal{N}(\vect{0}, \gmat{\Sigma}_\varepsilon)$ drawn independently per member:

$$
\vect{x}_{t \mid t}^{(e)} = \vect{x}_{t \mid t-1}^{(e)} + \mat{G}_t (\vect{y}_t + \gvect{\varepsilon}^{(e)} - \mat{H} \vect{x}_{t \mid t-1}^{(e)}) ,
$$ (eq-enkf)

where the gain $\mat{G}_t \in \mathbb{R}^{N \times N}$ is

$$
\mat{G}_t = \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top (\mat{H} \gmat{\Sigma}_{x,t \mid t-1} \mat{H}^\top + \gmat{\Sigma}_\varepsilon)^{-1} .
$$ (eq-enkf-gain)

The perturbations matter, because if we update every member with the same unperturbed $\vect{y}_t$, the anomalies shrink by $(\mat{I} - \mat{G}_t \mat{H})$ and the analysis ensemble is too narrow, which would bias the CRPS in stage 5. $\gmat{\Sigma}_{x,t \mid t-1}$ is inflated by 1.05 and localised with a Gaspari–Cohn taper. The localisation trades exactness for removing spurious long-range correlations, and the perturbations make the estimate random. However, they are drawn from one key seeded from the config of the run, so a rerun reproduces them (gate G1). A square-root (ETKF) variant replaces the perturbations with a deterministic anomaly transform. With a linear $f$ and $\mat{H}$, no inflation and no localisation, its analysis mean equals the Kalman update applied to the sample mean and covariance of the ensemble itself, which is gate D2.

(sec-stage-evaluate)=
### Stage 5 · Index, evaluate, register (steps 6–8: geocatalog.states → pipekit-evaluate → pipekit-experiment)

**TL;DR.** Store every analysis, score it against three independent references, and attach the score to the model that made it.

**Problem.** For a forecast distribution with CDF $F$, a reference value $y \in \mathbb{R}$, and independent draws $y', y'' \in \mathbb{R}$ from $F$, the continuous ranked probability score is given below. Lower is better, and with one member it reduces to the absolute error. Mass closure checks that the total methane is conserved to within 1 %.

$$
\mathrm{CRPS}(F, y) = \mathbb{E}_F \vert y' - y \vert - \frac{1}{2} \mathbb{E}_F \vert y' - y'' \vert
$$ (eq-crps)

The same pipeline, written as YAML, becomes the regulatory artifact ([Report 9, use case 9](reports/09_use_cases.md#r9-case-9)). The same operators run in a notebook ([case 1](reports/09_use_cases.md#r9-case-1)), an ETL job ([case 2](reports/09_use_cases.md#r9-case-2)), a FastAPI endpoint ([case 5](reports/09_use_cases.md#r9-case-5)) or a Fused UDF (the Fused integration design).

(sec-tier-briefs)=
## Per-layer briefs

Here we give a compact summary per package, by layer ([](#fig-tier-stack)). Where a report or a GeoModels page covers a package, the brief is a link to it.

::::{tab-set}

:::{tab-item} External
**The JAX ecosystem**: JAX, Equinox, lineax, optax, diffrax, NumPyro and matfree.

**The xarray ecosystem**: `DataArray`, `Dataset` and `DataTree`, plus rioxarray, xvec, cf-xarray, regionmask, xskillscore, dask and intake.
:::

:::{tab-item} GeoStack
**georeader** (third-party, spaceml-org's, and read-only here) owns `GeoTensor`, the raster carrier, and the reader infrastructure (`RasterioReader`, `AsyncGeoTIFFReader`). Everything raster-shaped builds on it.

| Package | What it does | Read |
|---|---|---|
| `pipekit` | The operator framework: fourteen groups (A–N) | [Report 2](reports/02_pipekit_surface.md#r2-inventory), [Report 1](reports/01_toolz_primer.md) |
| `pipekit-array` | Array-API operators | [Report 3](reports/03_sister_libraries.md#r3-array) |
| `pipekit-jax` | A model wrapper and a diffrax forward model; the operator layer is proposed | [Report 5](reports/05_pipekit_jax.md) |
| `pipekit-cycle` | Time stepping and DA cycles | [Report 10](reports/10_pipekit_cycle.md) |
| `pipekit-train` | Training pipelines | [Report 11](reports/11_pipekit_train.md) |
| `pipekit-experiment` | Model registry and tracking | [Report 12](reports/12_pipekit_experiment.md) |
| `pipekit-evaluate` | The benchmark ladder | [Report 14](reports/14_pipekit_evaluate.md), [Report 15](reports/15_benchmarking.md), [Report 17](reports/17_cycle.md) |
| `geotoolz` | GeoTensor operators for remote sensing | [Report 3](reports/03_sister_libraries.md#r3-geotoolz) |
| `geocatalog` (ships as `geotoolz-catalog`) | Find, index, join and load files | [Report 6](reports/06_geocatalog.md) |
| `geopatcher` (ships as `geotoolz-patcher`) | Split, process and stitch patches | [Report 7](reports/07_geopatcher.md) |
| `geocatalog.states` (proposed state profile) | A catalog for model states | [Report 13](reports/13_state_catalog.md) |
| `xrtoolz` | xarray operators for ocean, atmosphere and reanalysis | [Report 3](reports/03_sister_libraries.md#r3-xrtoolz) |

geotoolz and xrtoolz share the same pipekit operator algebra, and only the carrier differs.
:::

:::{tab-item} GeoNumerics
| Package | What it does |
|---|---|
| `gaussx` | Structured linear algebra, Gaussian primitives, exponential family; wraps lineax and matfree |
| `spectraldiffx` | Pseudospectral differentiation, filters, PDE solvers |
| `finitevolx` | Arakawa C-grid finite-volume operators |

This layer is pure math with no domain knowledge, and all of it is `eqx.Module`-based, jittable and JAX-traceable. See the [gaussx example](#ex-gaussx).
:::

:::{tab-item} GeoML
| Package | What it does |
|---|---|
| `geonnax` | Basis functions, neural layers and coordinate transforms (`lonlat_to_cartesian3d`) |
| `kernellib` | Covariance kernels and scalable kernel methods |
| `pyrox` | GP and Bayesian deep-learning building blocks for NumPyro models |
| `vardax` | Learnable variational DA (3D- and 4D-Var) |
| `filterax` | Differentiable ensemble Kalman methods |
| `gauss_flows` | Rotation-based Gaussianization and normalizing flows |
| `xtremax` | Extreme-value and point-process models; computes the L5 extremes |
| `optax_bayes` | The Bayesian learning rule as optax gradient transformations |

The adapter of each library (`*.adapters.pipekit`) makes it composable into pipekit pipelines, whereas the core never imports pipekit. See the [filterax example](#ex-filterax).
:::

:::{tab-item} GeoModels
| Package | What it does | Read |
|---|---|---|
| `somax` | Ocean and atmosphere model zoo | — |
| `plumax` | Atmospheric plume transport | [plumax · 1](geomodels/plumax/01_emission_rate.md), [package boundaries](geomodels/package_boundaries.md) |
| `radtrax` | Differentiable radiative transfer (formerly rtmx) | [radtrax · 3](geomodels/radtrax/03_design.md) |

The forward models each ship a thin `adapters/pipekit.py` which exposes the operator surface (`ForwardModel`, `ObservationOperator`, …) without touching the algorithmic core.
:::
::::

(sec-code-patterns)=
## Key code patterns

Each pattern has runnable-looking code on the [overview examples](overview_examples.md) page.

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

None of these block the work, but all of them deserve some thought before we commit to a decision.

:::{dropdown} Raised by the design review
:open:
- **What `InferenceArtifact` still needs for a cold rerun.** The shipped `InferenceArtifact` already saves and loads the YAML of a pipeline, its pinned model hashes, the lockfile and metadata. A `rerun` which rebuilds the pipeline and pulls its frozen inputs is proposed. What is still open is 1) whether that rerun resolves inputs from the pinned catalog URIs and SHA-256 digests alone, without cloud credentials, and 2) how a `Benchmark` (not yet serialisable) is frozen next to it.
:::

:::{dropdown} Resolved elsewhere
- **Serialising a `Graph`.** Emitting a built pipeline as YAML that loads back is proposal A of [Report 9](reports/09_use_cases.md#r9-proposals). Until it lands, gate A1 only covers YAML-safe operators.
- **A per-step observation error.** An `obs_err_source` is proposed in [Report 10, workstream B](reports/10_pipekit_cycle.md#r10-ws-b).
- **Two evaluation APIs.** [Report 15, workstream D](reports/15_benchmarking.md#r15-ws-d) builds the contract and the tracks on the shipped `Benchmark` cube (`from_contract`, several references per run), with no separate `MultiTrackEvaluation`.
- **Where `xrreader` stops and `georeader` starts.** A mission reader lives with its output carrier ([Report 16, workstream E](reports/16_geo_data_stack.md#r16-ws-e)).
- **Which stack somax and spectraldiffx belong to.** Rule 5 of the [package boundaries](geomodels/package_boundaries.md#pb-rules) puts somax in GeoModels, and spectraldiffx and finitevolx in GeoNumerics. methanex is still unplaced.
:::

:::{dropdown} Sequencing
:open:
- **When does pipekit-jax's operator layer land?** pipekit-jax 0.0.2 has shipped as a model wrapper and a diffrax forward model ([Report 5](reports/05_pipekit_jax.md#r5-ws-b)), whereas the operator layer is still proposed. The [six-step cycle](#sec-six-steps) gives it a concrete driver, since differentiable step-3 emulators make step 4 (gradient-based inverse problems) tractable at scale. It stays open until such a project appears.
- **Which in-situ catalog backend comes first?** The scoping proposal behind [Report 6](reports/06_geocatalog.md) leaned to Argo (via argopy), but ICOADS, SOCAT and the World Ocean Database matter for other domains. Realistically, we can do one per quarter, driven by projects.
:::

:::{dropdown} Design
- **The Slice Protocol refactor.** `GeoSlice` is the only slice type today. Adding `ProfileSlice` and `TrajectorySlice` is the right structural move, but it is two weeks of work which ripples across geocatalog, geotoolz, xrtoolz and pipekit-train. So it is better done before more code commits to `GeoSlice`.
- **Where matchup lives.** `geocatalog.queries.matchup()` is the framework function, with carrier-specific helpers in xrtoolz and geotoolz. Does matchup need its own catalog type? Probably, but it is not scoped yet.
- **Backend choice in pipekit-train.** Equinox + Optax and the Bayesian backends ship, whereas Lightning and Keras 3 are scaffolds ([Report 11](reports/11_pipekit_train.md)). The intended split is Lightning for PyTorch models, Equinox + Optax for JAX-traceable pipelines and Keras 3 for multi-backend. This is worth stating per use case.
:::

:::{dropdown} Operational
- **Monorepo or polyrepo?** A monorepo for the framework and infrastructure (uv workspaces), whereas the JAX-native algorithm and domain libraries stay in their own repositories and ship adapter modules.
- **One documentation site or many?** One site for the monorepo packages, separate sites for the algorithm libraries, and one research journal which cuts across everything.
- **CI per package or per monorepo?** Per-package CI inside the monorepo, plus one weekly integration matrix which pins every package to HEAD. Releases go by git tag (`geocatalog-v0.2.0`).
:::

:::{dropdown} Scientific
- **Matchup UQ.** `MatchupPair.representativeness_uncertainty` is easy to declare and hard to compute, so v0.1 ships a placeholder.
- **Cascading uncertainty.** A five-stage emission chain accumulates uncertainty across stages. Predicting that honestly is still open research.
- **Probabilistic event detection.** "A plume happened" is binary, whereas "a 73 % chance a plume happened" is much harder, and the current event metrics assume the binary case.
:::

:::{dropdown} Sustainability
- **Solo maintenance.** About twenty packages is a lot, even with monorepo discipline. So scope discipline is the operating constraint: each phase delivers standalone value, and no layer needs the full stack to exist.
- **Community adoption.** If oceanbench adopts the benchmark contracts of pipekit-evaluate, the framework is validated, and if not, it stays personal infrastructure. When to start reaching out is still open.
:::

(sec-plan)=
## A plan

Each phase produces packages which are useful on their own ([](#tbl-phases)).

:::{table} Phases at a glance.
:label: tbl-phases

| Phase | Scope | Status | Time |
|---|---|---|---|
| 1 | GeoNumerics: gaussx (v0.0.6), spectraldiffx, finitevolx | gaussx done; the rest drafts | ongoing |
| 2 | GeoML: geonnax, kernellib, optax_bayes, pyrox, filterax, vardax, gauss_flows, xtremax | in progress | each at its own pace |
| 3 | GeoModels: somax, plumax, radtrax | some done | project-driven |
| 4 | Pipeline framework | **current focus** | ~3–4 months |
| 5 | Evaluation, benchmarks and cycle discipline | next | ~2 months |
| 6 | Deferred items, each with a trigger | when projects demand | — |
| 7 | Polish and community | open-ended | — |
:::

### Phases 4 and 5, month by month

::::{tab-set}

:::{tab-item} Month 1
- pipekit 0.0.2: framework core, fourteen groups and state primitives (shipped; [Report 2](reports/02_pipekit_surface.md#r2-inventory))
- pipekit-array 0.0.2: Array API operators (shipped, phase A of four; [Report 3](reports/03_sister_libraries.md#r3-array))
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
- geocatalog's state profile (`geocatalog.states`) and xrtoolz forecast alignment
- pipekit-cycle 0.0.2: `Cycle`, `DACycle` and the protocols (shipped; [Report 10](reports/10_pipekit_cycle.md))
- adapter modules in filterax, vardax and plumax

**Gates:** B2 state rows, D1 kriging oracle, D2 Kalman oracle.
:::

:::{tab-item} Month 4
- pipekit-train 0.0.2 ships Equinox + Optax and the Bayesian backends, whereas Lightning and Keras 3 are scaffolds ([Report 11](reports/11_pipekit_train.md))
- pipekit-experiment 0.0.2 ships the registry and the tracker protocol, and the MLflow and W&B trackers are planned ([Report 12](reports/12_pipekit_experiment.md))
- integration testing across all packages

**Gate:** E1 registry round-trip.

**After Phase 4:** the L0 to L4 inference loop is structurally complete. ML training and serving still need work, but the pipelines exist.
:::

:::{tab-item} Month 5
- pipekit-evaluate v0.1: framework, cheap metrics, `EvaluationReport`
- leakage-aware splits: block, platform and causal splits of catalog rows in geocatalog, and the window guard in time units in pipekit-train
- standard baselines in xrtoolz: persistence, climatology, OI
- lineage in pipekit-experiment's content hash, with a dependents query (the six-step cycle's lineage, [Report 17](reports/17_cycle.md#r17-ws-b))

**Gate:** F1 metric identities.
:::

:::{tab-item} Month 6
- particle tracking (`xrtoolz.lagrangian`) and event detection (`xrtoolz.events`)
- `BenchmarkContract` with transition cells and an oracle strength ([Report 17](reports/17_cycle.md#r17-ws-a))
- several references per run in the benchmark runner ([Report 15](reports/15_benchmarking.md#r15-ws-d))
- the first worked cycle (likely methane L1→L2 retrieval, steps 1 to 5)
- the first worked benchmark (SSH, from the [benchmarks gallery](benchmarks_gallery.md#gal-1-1))

**Gate:** G1 end to end.

**After Phase 5:** evaluation is first-class, benchmarks are content-addressed contracts, and the six-step cycle is tractable end to end.
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

Phase 7 covers the documentation site, a research journal, the first external benchmark adoption (OceanBench, WeatherBench-Ocean?) and capstone tutorials per domain.

(sec-gates)=
### Gates

A month is done when its gates pass, and not when its packages exist ([](#fig-roadmap)). Every pass condition is a number or a closed form, and several of them reuse the closed-form stages of the walkthrough as oracles ([](#tbl-gates)).

```{figure} images/overview/roadmap.svg
:label: fig-roadmap
:alt: Phases 4 and 5 as six monthly boxes from top to bottom, each with its status (shipped, in progress or planned) and followed by a diamond naming the gates, with their pass conditions, that close it. Phases 1 to 3, one per lower layer (GeoNumerics, GeoML, GeoModels), run alongside and feed the DA adapters in month 3 and the models trained in month 4; phases 6 and 7 follow the last gate.
:width: 100%

**The framework ships first; every month closes on its gates.**

Phases 1–3, one per lower layer, run alongside and meet the chain twice:
the DA adapters land in month 3 and the GeoML models are trained in month 4.
Phases 6 and 7 wait for projects to pull them.
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
| D2 Kalman oracle | D | With linear $f$ and $\mat{H}$, no inflation and no localisation, the square-root ensemble analysis mean matches the Kalman update computed from the same ensemble's sample mean and covariance, to 1e-6 in float64 |
| E1 Registry round-trip | E | Loading a stored operator returns one with the same content hash and bitwise-equal outputs |
| F1 Metric identities | F | CRPS {eq}`eq-crps` with one member equals absolute error exactly; RMSE of a field against itself is 0 |
| G1 End to end | all | Stages 1–5 of the walkthrough run from one YAML file. A cold rerun of the frozen inference artifact reproduces the analysis history's hash, and rerunning the frozen benchmark (its task, estimators, scorers and pinned TCCON references) on that history reproduces the report's hash |
:::

(sec-risks)=
### Risks

- **The schedule slips.** Six months for Phases 4 and 5 is optimistic for one person, and eight to twelve is realistic. Fallback: every month ships a usable package, so nothing depends on the whole plan finishing.
- **Phase 6 is needed early.** A project may need pipekit-jax or in-situ catalogs before the framework is ready. Fallback: we wrap jitted functions in pipekit-jax's `JaxModelOp` until the full backend lands, and we jump phases, since the sequence is a default and not a contract.
- **The Slice refactor ripples late.** The longer `GeoSlice` stays concrete, the more code binds to it. Fallback: we introduce a `Slice` protocol in month 2 with `GeoSlice` as its only implementation, and add profile and trajectory slices later.
- **pipekit-train becomes three half-backends.** Equal status for Lightning, Equinox and Keras 3 triples the maintenance. Fallback: one default backend per use case, and the others are community-supported.
- **Nobody else adopts it.** Adoption takes years. Fallback: GeoBench is the first consumer, so the substrate earns its keep without outside users.

(sec-reading)=
## Where to read next

(sec-conventions)=
### How the pages are written

Each page checks a design, a scoping report or a source note against the code pinned in `references/`, and the code takes precedence. A **Differs** box marks each place where the code (or a check the page runs) disagrees with the design. The gaps that matter become workstreams with proposals, and the rest go to open questions or upcoming work. Every number on a page comes from a notebook which asserts it. Pseudocode follows one convention: 1) a shapes header names the dimensions, 2) the equation sits in a comment above the line that computes it, 3) shapes go on the right, and 4) every call into the stack carries a status marker ([markers](overview_examples.md#ex-markers)).

### The pages

We suggest you start with the [capability index](api/capabilities.md) and the [overview examples](overview_examples.md), and then read by question.

**Across the stack**

- [Report 9](reports/09_use_cases.md): twenty use cases mapped onto the libraries as they ship.
- [Report 16](reports/16_geo_data_stack.md): the geo data stack, the ownership rules and the realignment plan across 18 distributions.

**pipekit**

- [Report 1](reports/01_toolz_primer.md): the toolz lineage, and why pipekit's operators are classes; now pipekit's [Background](https://jejjohnson.github.io/pipekit/background/toolz/) page.
- [Report 2](reports/02_pipekit_surface.md): what pipekit 0.0.2 ships, in fourteen groups (A to N); the tour is now pipekit's [tour of the core](https://jejjohnson.github.io/pipekit/core-tour/), and the planning stays here.
- [Report 5](reports/05_pipekit_jax.md): jit, vmap and grad through pipelines, and what pipekit-jax ships.
- [Report 10](reports/10_pipekit_cycle.md): pipekit-cycle, time-stepping and data assimilation.
- [Report 11](reports/11_pipekit_train.md): pipekit-train, training for emulators and amortized inference.
- [Report 12](reports/12_pipekit_experiment.md): pipekit-experiment, the registry, trackers and reproducibility artifacts.
- [Report 14](reports/14_pipekit_evaluate.md): pipekit-evaluate, multidimensional evaluation.

**geotoolz and xrtoolz**

- [Report 3](reports/03_sister_libraries.md): the sister libraries, pipekit-array, geotoolz and xrtoolz.
- [Report 6](reports/06_geocatalog.md): geocatalog, the spatiotemporal index.
- [Report 7](reports/07_geopatcher.md): geopatcher, the four-axis patcher.
- [Report 13](reports/13_state_catalog.md): L3 to L5 indexing, as geocatalog's state profile.

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

GeoStack is one operator algebra from structured linear algebra to indexed multi-tier data products, which is organised by three orthogonal axes and split into the three stacks of [the decision](#sec-decision). Classical methods stay as oracles, benchmarks are content-addressed contracts, and differentiability only lives in the JAX packages. The plan is a default sequence which projects will reshape. So pick a page from [Where to read next](#sec-reading) and come back with questions, since no part of the stack requires the rest.

## Glossary

:::{glossary}
carrier
: The data structure an operator acts on: NumPy or JAX arrays, a {term}`GeoTensor`, or an xarray object. pipekit operators are carrier-agnostic.

GeoTensor
: georeader's raster carrier: an array with its geotransform and CRS. The carrier for L0 to L2.

GeoSlice
: geocatalog's wire format for one spatiotemporal slice of an observation catalog.

state profile
: geocatalog's reserved columns for model states (run time, ensemble member, producer hash and parent state) and the loader and queries in `geocatalog.states` that use them. It replaces the planned `statecatalog` package and its `StateSlice`.

matchup
: Pairing observations from different sources that coincide in space and time; the operation where the satellite and in-situ streams converge.

oracle
: The classical method (step 2) used as the trusted reference to validate emulators and amortized predictors.
:::
