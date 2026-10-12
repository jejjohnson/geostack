---
title: GeoStack overview
subtitle: A JAX-native research stack for geophysical modeling, inference and data assimilation
short_title: Overview
description: The entry point to GeoStack. Three organising axes, the six-tier stack, an end-to-end methane walkthrough, the open questions and a plan.
keywords:
  - geophysics
  - data assimilation
  - JAX
  - operator graphs
  - remote sensing
numbering:
  heading_2: true
  heading_3: true
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
map. It ties together the vision document and the sixteen scoping reports,
and it ends with the open questions and a plan.

+++

:::{tip} How to read this page
:class: dropdown
- **Five minutes:** read the [elevator pitch](#sec-pitch), look at
  {numref}`fig-lifecycle` and {numref}`fig-tier-stack`, and skim the
  [plan](#sec-plan).
- **Building something:** start with the [walkthrough](#sec-walkthrough)
  and the [overview examples](overview_examples.md), then search the
  [capability index](api/capabilities.md) before writing new code.
- **Reviewing the design:** go straight to the
  [open questions](#sec-open-questions).
:::

(sec-pitch)=
## The elevator pitch

> A vertically integrated, JAX-native research software stack for geophysical
> modeling, inference and data assimilation: from structured linear algebra
> primitives, through composable operator graphs, to indexed multi-tier data
> products, with ML pluggable at every level.

In one sentence: **the operator-graph abstraction, extended from numerical
primitives all the way up to end-to-end pipelines, with content-addressed
data and models throughout.**

You can build great JAX numerics (gaussX, lineax, equinox) and great xarray
pipelines today. What you cannot do is connect them. GeoStack fills that gap
with five building blocks:

::::{grid} 1 1 2 3

:::{card} Framework
`pipekit`: a carrier-agnostic operator graph. One algebra for NumPy, JAX,
xarray and GeoTensor.
:::

:::{card} Infrastructure
`geocatalog`, `statecatalog`, `geopatcher`: index observations and model
states, and cut fields into patches.
:::

:::{card} Training and evaluation
`pipekit-train`, `pipekit-evaluate`, `pipekit-experiment`: train emulators,
evaluate along many axes, register every artifact.
:::

:::{card} Domain libraries
`geotoolz` for GeoTensor and `xrtoolz` for xarray: the carriers people
actually use.
:::

:::{card} Adapters
filterX, vardaX, plumax, pyrox-gp and gaussFlowX plug in through thin
adapter modules, without coupling.
:::
::::

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

Three axes structure the framework. They are orthogonal: every operation sits
somewhere on all three.

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
  xarray, `statecatalog` indexes them as {term}`StateSlice`s, `xrtoolz`
  transforms them, and they live as Zarr or NetCDF on object storage.
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
in-situ stream converge between L2 and L3.

```{figure} images/overview/two_streams.svg
:label: fig-two-streams
:alt: Satellite levels L0 to L4 on the left; in-situ networks for ocean, atmosphere, land and ice on the right. L2 and in-situ meet at a matchup node that feeds L3; in-situ also enters L4 through assimilation.
:width: 100%

**Satellite and in-situ observations converge through matchup.**

In-situ networks in every domain join the satellite chain through
spatiotemporal {term}`matchup` between L2 and L3, and enter L4 through
assimilation. They have no L0 or L1 equivalent.
```

The matchup pattern is canonical and deserves its own operator family; the
data-lifecycle report (`data_ml_lifecycle.md`) covers it in depth.

(sec-three-layers)=
### Three layers of transformation

The cycle and the lifecycle are joined by a third lens: what we *do* to the
data on its way into a model.

```{figure} images/overview/three_layers.svg
:label: fig-three-layers
:alt: Scientific data becomes ML-ready data through harmonisation and tensorisation, which becomes embeddings through compression and latent mapping. Each layer lists the packages that own it.
:width: 100%

**Scientific data, ML-ready data, embeddings, and who owns each.**
```

(sec-six-steps)=
### The six-step cycle (axis 3)

The research-to-production journey. Every domain library follows this cycle
implicitly; the framework makes it explicit and composable.

```{figure} images/overview/six_steps.svg
:label: fig-six-steps
:alt: Five steps in a row, from simple model to amortized inference. Step 2 validates steps 3 to 5. A shared improve step sits under steps 2 to 5 with two-way links.
:width: 100%

**From a simple model to amortized inference, with one shared improve step.**

Step 2, the classical method, is the {term}`oracle` that validates steps
3–5. Step 6 is shared: any of steps 2–5 can be improved, using the previous
step as ground truth.
```

Two structural commitments fall out of the cycle:

::::{grid} 1 1 2 2

:::{card} Step 2 is the oracle
Classical methods (`filterX`, `vardaX`, `pyrox-gp`) are not legacy code
waiting to be replaced by neural emulators. They become *ground-truth
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
| 5 · Amortized inference | `pipekit-train` with `gaussFlowX` or `pyrox-nn` |
| 6 · Improve | `pipekit-evaluate.EvaluationReport` diffs |
:::

(sec-axes-relate)=
### How the axes relate

The axes form a cube of operations. At any point you are somewhere on the
modeling cycle (research stage), somewhere on the lifecycle (data maturity)
and somewhere on the six-step cycle (production maturity).

```{figure} images/overview/axes_matrix.svg
:label: fig-axes-matrix
:alt: A grid with lifecycle transitions as rows, from calibration to forecast, and steps 1 to 5 as columns. Each cell names a method; the step-2 column is highlighted as the oracle.
:width: 100%

**Every transition crossed with every step is a concrete method.**
```

**Every cell is a benchmarkable artifact,** and every cell uses the column
before it as its oracle. The whole lifecycle is a nested set of six-step
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

Most current work happens in the framework and infrastructure tier.

```{figure} images/overview/tier4_map.svg
:label: fig-tier4-map
:alt: Three groups: the pipekit framework packages, the ML-loop packages, and the infrastructure peers geocatalog, geopatcher and statecatalog, with georeader as an external substrate.
:width: 100%

**The pipekit framework, the ML loops built on it, and the infrastructure peers.**
```

(sec-dependencies)=
### The dependency graph

```{figure} images/overview/dependency_graph.svg
:label: fig-dependency-graph
:alt: Three family panels. Every pipekit package depends on pipekit; xrtoolz builds on xrtoolz-core, which depends on pipekit; geotoolz depends on pipekit, and every geotoolz-family package depends on georeader. Dashed edges are optional extras.
:width: 100%

**How the reference packages import each other.**

Drawn from the packages' own metadata. Solid edges are dependencies, dashed
edges optional extras.
```

:::{note} No circular dependencies
Each layer depends only on the layers below it. The domain libraries
(geotoolz, xrtoolz) re-export the common framework symbols, so user code
usually imports a single package.
:::

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

The same pipeline, written as YAML, becomes the regulatory artifact (Report 9,
use case 9). The same operators run in a notebook (case 1), an ETL job
(case 2), a FastAPI endpoint (case 5) or a Fused UDF (Report 15 and the Fused
integration design).

(sec-tier-briefs)=
## Per-tier briefs

A compact summary per package. The deep dives are in the numbered reports.

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
| `gaussX` | Structured linear algebra, Gaussian primitives, exponential family; wraps lineax and matfree |
| `spectralDiffX` | Pseudospectral differentiation, filters, PDE solvers |
| `finiteVolX` | Arakawa C-grid finite-volume operators |
| `optax-bayes` | The Bayesian learning rule as optax gradient transformations |
| `eqx-trainer` | A minimal JAX / equinox training loop |

Pure math with no domain knowledge: all `eqx.Module`-based, jittable and
JAX-traceable. See the [gaussX example](#ex-gaussx).
:::

:::{tab-item} Tier 2 · Algorithms
| Package | What it does |
|---|---|
| `pyrox-gp` | GP building blocks for NumPyro hierarchical models |
| `pyrox-nn` | Bayesian deep learning with equinox and NumPyro |
| `filterX` | Differentiable ensemble Kalman methods |
| `vardaX` | Learnable variational DA (3D- and 4D-Var) |
| `gaussFlowX` | Rotation-based Gaussianization and normalizing flows |
| `fairkl` | Fairness-constrained kernel learning (Keras 3) |
| `keras-nerf` | Neural fields for state estimation (Keras 3) |
| `keras-flows` | Normalizing flows (Keras 3) |

Keras 3 and Equinox are equal-status backends. Each algorithm's adapter
(`*.adapters.pipekit`) makes it composable into pipekit pipelines. See the
[filterX example](#ex-filterx).
:::

:::{tab-item} Tier 3 · Domain
| Package | What it does |
|---|---|
| `somax` | Ocean and atmosphere model zoo |
| `xtremax` | Extreme value modeling |
| `methanex` | End-to-end methane remote sensing |
| `plumax` | Atmospheric plume transport |
| `RTMX` | Radiative transfer operators |
| `geotoolz` | GeoTensor operators for remote sensing: readers, retrieval, cloud, radiometry, indices, spectral, plume, viz |
| `xrtoolz` | xarray operators for ocean, atmosphere and reanalysis: validation, CRS, subsetting, masks, interpolation, metrics, kinematics |

The forward models each ship a thin `adapters/pipekit.py` that exposes the
operator surface (`ForwardModel`, `ObservationOperator`, …) without touching
the algorithmic core. geotoolz and xrtoolz share the same Tier 4 peers: the
substrate differs, the operator algebra is identical.
:::

:::{tab-item} Tier 4 · Pipelines
**pipekit**, the framework (~2,150 lines), in twelve semantic groups
(Reports 1 and 2):

| Group | Operators |
|---|---|
| Foundations | `Operator`, `Sequential`, `Graph`, `Input`, `Node`, `ConfigMixin` |
| Convenience | `pipe`, `compose`, `juxt`, `complement` |
| Building blocks | `Identity`, `Const`, `Lambda`, `Sink` |
| Control flow | `Branch`, `Switch`, `Try`, `Coalesce`, `Retry` |
| Observers | `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram` |
| Combination | `Fanout` |
| Caching | `Cache`, `Memoize` |
| QC | `Quarantine`, `AssertShape`, `AssertDType`, `AssertCallable` |
| Shape inference | `Signature`, `compute_output_signature` |
| Parallelism | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap` |
| Serialisation | `dumps`, `loads`, sandboxed loader |
| State | `StatefulOperator`, `CarryState` |

| Package | Role | Report |
|---|---|---|
| `pipekit-array` | Array API operators across NumPy, JAX, CuPy and PyTorch (`StackAlong`, `ConcatenateAlong`, `MeanScalar`) | 3 |
| `pipekit-jax` | Equinox-backed, JAX-traceable operators; deferred to v0.3+ | 5 |
| `pipekit-cycle` | Time-stepping and DA: `Cycle`, `EnsembleCycle`, `DACycle` and three Protocols | 10 |
| `pipekit-train` | Training for emulators and amortized inference over Lightning, Equinox + Optax or Keras 3 | 11 |
| `pipekit-experiment` | Tracker adapters (MLflow, W&B, DVC, Hydra) and a content-addressed `ModelRegistry` | 12 |
| `pipekit-evaluate` | Evaluation on a Unit × Lens × Stage matrix; benchmark contracts as artifacts | 14, 15 |
| `geocatalog` | Spatiotemporal index for L0–L2 (InMemory and DuckDB backends; `GeoSlice`) | 6 |
| `geopatcher` | Four-axis patcher: geometry × sampler × window × aggregation (~60 classes) | 7 |
| `statecatalog` | Spatiotemporal index for L3–L5 (`StateSlice` with valid time, run time, ensemble, lineage) | 13 |
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

:::{dropdown} Sequencing
:open:
- **Does pipekit-jax have to wait until v0.3+?** Report 5 says yes. But the
  [six-step cycle](#sec-six-steps) surfaces a concrete driver:
  differentiable step-3 emulators are what makes step 4 (gradient-based
  inverse problems) tractable at scale. A project that needs amortized
  inference with end-to-end differentiability would trigger pipekit-jax
  earlier. Open until such a project appears.
- **Which in-situ catalog backend comes first?** Report 6 leans to Argo (via
  argopy), but ICOADS, SOCAT and the World Ocean Database matter for other
  domains. Realistically one per quarter, driven by projects.
:::

:::{dropdown} Design
- **The Slice Protocol refactor.** `GeoSlice` is the only slice type today.
  Adding `ProfileSlice` and `TrajectorySlice` is the right structural move,
  but it is two weeks of work rippling across geocatalog, geotoolz, xrtoolz
  and pipekit-train. Better done before more code commits to `GeoSlice`.
- **Where matchup lives.** `geocatalog.queries.matchup()` is the framework
  function, with carrier-specific helpers in xrtoolz and geotoolz. Does
  matchup need its own catalog type? Probably; not yet scoped.
- **Backend choice in pipekit-train.** Lightning, Equinox + Optax and Keras 3
  are all supported. In practice: Lightning for PyTorch models, Equinox +
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

Each phase produces packages that are useful on their own.

:::{table} Phases at a glance.
:label: tbl-phases

| Phase | Scope | Status | Time |
|---|---|---|---|
| 1 | Numerical foundation: gaussX (v0.0.6), spectralDiffX, finiteVolX, optax-bayes, eqx-trainer | gaussX done; the rest drafts | ongoing |
| 2 | Algorithms: pyrox-gp, pyrox-nn, filterX, vardaX, gaussFlowX | in progress | each at its own pace |
| 3 | Domain models: somax, plumax, xtremax, methanex, RTMX | some done | project-driven |
| 4 | Pipeline framework | **current focus** | ~3–4 months |
| 5 | Evaluation, benchmarks and cycle discipline | next | ~2 months |
| 6 | Deferred items, each with a trigger | when projects demand | — |
| 7 | Polish and community | open-ended | — |
:::

### Phases 4 and 5, month by month

::::{tab-set}

:::{tab-item} Month 1
- pipekit v0.1: framework core, twelve groups and state primitives
- pipekit-array v0.1: Array API operators
- the pipekit-ecosystem monorepo with uv workspaces
:::

:::{tab-item} Month 2
- geocatalog v0.1: InMemory and DuckDB backends, basic queries
- geopatcher v0.1: the four-axis framework and five Field adapters
- geotoolz and xrtoolz refactored onto the new framework
:::

:::{tab-item} Month 3
- statecatalog v0.1
- pipekit-cycle v0.1: `Cycle`, `DACycle` and three Protocols
- adapter modules in filterX, vardaX and plumax
:::

:::{tab-item} Month 4
- pipekit-train v0.1: Lightning and Equinox backends
- pipekit-experiment v0.1: MLflow and S3 registry adapters
- integration testing across all packages

**After Phase 4:** the L0–L4 inference loop is structurally complete. ML
training and serving still need work, but the pipelines exist.
:::

:::{tab-item} Month 5
- pipekit-evaluate v0.1: framework, cheap metrics, `EvaluationReport`
- leakage-aware splitters in pipekit-train
- standard baselines: persistence, climatology, OI
- `ModelRole` and `ArtifactLineage` in pipekit-experiment, the six-step
  cycle's content-addressed lineage
:::

:::{tab-item} Month 6
- particle tracking (`xrtoolz.lagrangian`) and event detection
  (`xrtoolz.events`)
- `BenchmarkContract` and `CycleStageBenchmark` artifacts
- multi-track evaluation as a built-in pattern
- the first worked cycle (likely methane L1→L2 retrieval, steps 1–5)
- the first worked benchmark (SSH, from the benchmarks gallery)

**After Phase 5:** evaluation is first-class, benchmarks are
content-addressed contracts, and the six-step cycle is tractable end to end.
:::
::::

### Deferred until a project needs it

:::{table} Phase 6: what is deferred, and what triggers it.
:label: tbl-deferred

| Item | Trigger |
|---|---|
| pipekit-jax: Equinox-backed, JAX-traceable operators | the first project needing differentiable retrievals at scale or JIT-compiled inference |
| Deployment adapters (`pipekit.deploy.fused`, `.litserve`, `.modal`) | the first deployment beyond hand-shimming |
| Slice Protocol refactor (`ProfileSlice`, `TrajectorySlice`) | the first project needing first-class in-situ catalogs (likely the BGC benchmark) |
| More in-situ backends (Argo, ICOADS, SOCAT, …) | one per quarter, project-driven |
:::

Phase 7 covers the documentation site, a research journal, the first
external benchmark adoption (OceanBench, WeatherBench-Ocean?) and capstone
tutorials per domain.

:::{warning} Reality check
1. **Six months for Phases 4–5 is optimistic for one person;** eight to
   twelve is realistic. Each month still produces usable artifacts.
2. **Phase 6 may come early.** If a project needs pipekit-jax or in-situ
   catalogs first, jump phases: the sequence is a default, not a contract.
3. **Phase 7 is open-ended.** Adoption takes years. The substrate has to be
   useful without it; adoption is upside.
:::

(sec-reading)=
## Where to read next

:::{note}
These documents are not in this repository yet. They live with the migration
notes and will move into `docs/` as they are written up. Until then, the
[capability index](api/capabilities.md) and the reference repositories under
`references/` are the source of truth.
:::

| To understand… | Read |
|---|---|
| the big picture | `vision_v2.md` |
| why pipekit exists | Report 1: the toolz lineage |
| what pipekit ships | Report 2: twelve semantic groups |
| the sister libraries | Report 3: pipekit-array, geotoolz, xrtoolz |
| the JAX-native future | Report 5: pipekit-jax and the Equinox backend |
| spatiotemporal catalogs | Report 6: geocatalog |
| the patcher | Report 7: geopatcher |
| xrtoolz's role | Report 8: substrate asymmetry with geotoolz |
| concrete use cases | Report 9: thirteen cases mapped to the stack |
| time-stepping and DA | Report 10: pipekit-cycle |
| training pipelines | Report 11: pipekit-train |
| registry and tracking | Report 12: pipekit-experiment |
| L3–L5 indexing | Report 13: statecatalog |
| evaluation | Report 14: pipekit-evaluate |
| benchmark infrastructure | Report 15: contracts, leakage, baselines |
| the six-step cycle | Report 16: step 2 as oracle, benchmarks as understanding |
| the data lifecycle in detail | `data_ml_lifecycle.md` |
| concrete benchmarks | `benchmarks_gallery.md`: 23 benchmarks in 6 domains |

(sec-summary)=
## In summary

GeoStack is a vertically integrated, JAX-native research stack whose
operator-graph composition runs from structured linear algebra, through
pipeline infrastructure, to indexed multi-tier data products. It is
organised by three orthogonal axes: the modeling cycle, the L0–L5 data
lifecycle and the six-step cycle.

What makes it specific:

- **Operator algebra at every level,** with the right Protocol per layer and
  no universal base class.
- **A carrier-agnostic framework** bridging JAX numerics and xarray /
  GeoTensor data.
- **Content-addressed everything:** data slices, model artifacts, evaluation
  reports, benchmark contracts, reproducibility artifacts and cycle lineage.
- **ML pluggable at every transition,** through one training, registry and
  inference machinery.
- **Classical methods as oracles,** kept as ground-truth generators rather
  than replaced.
- **Benchmarks as operationalised hypotheses,** pre-registerable and
  content-addressed, not leaderboards.
- **A two-stream lifecycle,** satellite and in-situ converging through
  matchup, with depth-axis support.
- **Evaluation as infrastructure,** multidimensional from the start.
- **Gradients only where they belong:** differentiability lives in the JAX
  packages; xarray, pipekit and geotoolz orchestrate without gradients.
- **Math-first documentation:** the code is the paper.

The vision document is the manifesto, the sixteen reports are the deep
dives, and this page is the entry point. The plan is a default sequence that
projects will reshape. If a use case pulls you toward the stack, pick its
report from [Where to read next](#sec-reading) and come back with questions:
the stack is modular, and no part requires the rest.

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

StateSlice
: statecatalog's wire format for one model state, with valid time, run
  time, ensemble member and lineage.

matchup
: Pairing observations from different sources that coincide in space and
  time; the operation where the satellite and in-situ streams converge.

oracle
: The classical method (step 2) used as ground truth to validate emulators
  and amortized predictors.
:::
