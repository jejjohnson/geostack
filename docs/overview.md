---
title: GeoStack overview
---


## What this document is

A synthesis. The vision document (`vision_v2.md`) is the manifesto; the numbered reports (1–16) are scoping deep-dives for each piece; this document weaves them together into a single readable starting point. Lots of diagrams, lots of code snippets, lots of cross-references. Open questions and a plan at the end.

If you only have time for one document, this is the one. Otherwise, this is the map.

-----

## 1. The elevator pitch

> A vertically integrated, JAX-native research software stack for geophysical modeling, inference, and data assimilation — from structured linear algebra primitives through composable operator graphs to indexed multi-tier data products, with ML pluggable at every level.

In one sentence: **the operator-graph abstraction extended from numerical primitives all the way up to end-to-end pipelines, with content-addressed data and models throughout**.

In one paragraph: this stack solves a specific problem — there’s no Keras-equivalent for geophysical work. You can build great JAX numerics (gaussX, lineax, equinox), and you can build great xarray pipelines, but there’s no connective tissue between them. The GeoStack fills that gap with a:
- carrier-agnostic framework (`pipekit`)
- peer infrastructure for indexing observations and model states (`geocatalog`, `statecatalog`, `geopatcher`)
- training and evaluation machinery (`pipekit-train`, `pipekit-evaluate`, `pipekit-experiment`)
- domain libraries for the carriers people actually use (`geotoolz` for GeoTensor, `xr-toolz` for xarray)
- clean adapter pattern so existing algorithm libraries (filterX, vardaX, plumax, pyrox-gp, gaussFlowX) plug in without coupling.

-----

## 2. The shape of the problem

Five gaps motivate everything below. The first four are old; the fifth has only become visible recently.

1. **Scattered numerics.** Same linear algebra, kernel methods, filtering routines reimplemented in every project. No shared primitives.
2. **ML tooling doesn’t serve classical numerics.** Keras/PyTorch layers assume supervised learning on tensors. They don’t naturally express linear operators, PDE solvers, state estimation, uncertainty propagation.
3. **Monolithic domain software.** Ocean models, retrieval algorithms, DA systems are dense, tightly coupled, hard to recombine.
4. **No connective tissue.** Even with good numerics and good ML, there’s no Keras-like pipeline to compose them for real geophysical work.
5. **No indexed model-state tier.** Observations are catalogable (STAC, GeoParquet). Model outputs (forecasts, analyses, reanalysis) aren’t — they’re scattered files with bespoke conventions. The L3–L4 half of the lifecycle has no equivalent of the catalogs that organise L0–L2.

The stack addresses all five.

-----

## 3. Three organising axes

Three axes structure the whole framework. They’re orthogonal: every operation happens *somewhere* on all three. Axis 1 is the research loop. Axis 2 is the data lifecycle. Axis 3 is the research-to-production journey.

### 3.1 The modeling cycle (axis 1)

```{figure} images/overview/modeling_cycle.svg
:name: fig-modeling-cycle
:alt: The modeling cycle runs on two parallel tracks. The modeling track abstracts the real-world system into equations and code that make predictions; the observing track measures the same system with sensors and turns measurements into observations. The tracks meet where predictions are compared with observations, and the model is refined.
:width: 100%

The modeling cycle runs on two parallel tracks. The modeling track abstracts the real-world system into equations and code that make predictions; the observing track measures the same system with sensors and turns measurements into observations. The tracks meet where predictions are compared with observations, and the model is refined.
```

The cycle has two parallel tracks: modeling turns the system into predictions, observing turns it into observations through sensors and measurements, and the two meet at validation.

**Start simple. Add complexity only when validation demands it.** This is the research loop. Every package in the stack exists to make one or more steps of this cycle composable.

### 3.2 The L0–L5 data tier (axis 2)

```{figure} images/overview/lifecycle.svg
:name: fig-lifecycle
:alt: The L0–L5 lifecycle on two perpendicular axes. The science axis fans one stream of radiances out into ocean, atmosphere, land and ice at retrieval, and ends in derived products such as extremes (L5, dashed: examples still to be defined). The engineering axis maps every level to the GeoStack tool that transforms, patches, carries, reads, indexes and stores it, switching carrier from GeoTensor to xarray between L2 and L3.
:width: 100%

The L0–L5 lifecycle on two perpendicular axes. The science axis fans one stream of radiances out into ocean, atmosphere, land and ice at retrieval, and ends in derived products such as extremes (L5, dashed: examples still to be defined). The engineering axis maps every level to the GeoStack tool that transforms, patches, carries, reads, indexes and stores it, switching carrier from GeoTensor to xarray between L2 and L3.
```

**Two perpendicular axes.** The science axis runs left to right, from raw
telemetry to reanalysis and forecast. Until retrieval it is one stream: the
same L1 radiances feed every geoscience. Retrieval is where it fans out, into
ocean, atmosphere, land and ice variables at L2, and each domain then carries
its own L3 analysis and L4 reanalysis or forecast. L5 holds the products
derived from those states, such as extremes (marine heatwaves, droughts,
storms) or impact indicators; which products belong here is still open.

The engineering axis runs top to bottom, from transforming data down to
storing it, and maps every level to the GeoStack tool that handles it. It
splits at the L2/L3 boundary, where the carrier changes from GeoTensor to
xarray:

- **Observations (L0–L2):** `georeader` reads them into a GeoTensor,
  `geocatalog` indexes them as GeoSlices, `geotoolz` transforms them and
  `geocloud` stores them (object storage, Cloud-Optimized GeoTIFF).
- **Model states and derived products (L3–L5):** `xrreader` (today the
  `xrtoolz-reader` package) reads them into xarray, `statecatalog` indexes them as StateSlices,
  `xrtoolz` transforms them, and they live as Zarr or NetCDF on object
  storage.
- **Derived products (L5):** `xtremax` computes extremes from L4 states.
- **Both carriers:** `geopatcher` patches fields whatever the carrier.

ML enters at every transition.

### 3.3 The two-stream convergence

L0–L4 isn’t one chain. It’s two chains converging at L2/L3 via matchup:

```{figure} images/overview/two_streams.svg
:name: fig-two-streams
:alt: Two streams converge. The satellite chain runs L0→L4; in-situ networks in every domain (ocean, atmosphere, land, ice) join it through spatiotemporal matchup between L2 and L3, and enter L4 through assimilation.
:width: 100%

Two streams converge. The satellite chain runs L0→L4; in-situ networks in every domain (ocean, atmosphere, land, ice) join it through spatiotemporal matchup between L2 and L3, and enter L4 through assimilation.
```

The matchup pattern is canonical and deserves a real operator family. Covered in `data_ml_lifecycle.md`.

### 3.4 The three-layer model (transformations along the lifecycle)

The cycle and the L0–L4 axis are joined by a third lens — what we *do* to the data:

```{figure} images/overview/three_layers.svg
:name: fig-three-layers
:alt: Three layers of transformation along the lifecycle, and the packages that own each.
:width: 100%

Three layers of transformation along the lifecycle, and the packages that own each.
```

### 3.5 The six-step data-driven cycle (axis 3)

The research-to-production journey. **Every domain library implicitly follows this cycle**; the framework makes it composable.

```{figure} images/overview/six_steps.svg
:name: fig-six-steps
:alt: The six-step cycle. Steps 1–5 run from simple model to amortized inference, with step 2, the classical method, as the oracle that validates steps 3–5. Step 6, improve, is shared: any of steps 2–5 can be improved, using the previous step as ground truth.
:width: 100%

The six-step cycle. Steps 1–5 run from simple model to amortized inference, with step 2, the classical method, as the oracle that validates steps 3–5. Step 6, improve, is shared: any of steps 2–5 can be improved, using the previous step as ground truth.
```

**Two structural commitments fall out:**

1. **Step 2 as oracle.** Classical methods (`filterX`, `vardaX`, `pyrox-gp`) are not legacy code to be replaced by neural emulators. They become *ground-truth generators* against which Steps 3–5 are validated. The classical path stays alive forever, promoted to oracle status.
2. **Benchmarks as formalised understanding.** The act of writing a benchmark contract (carriers, references, metrics, splits, baselines, expected failure modes) is the act of formalising what you understand about the problem. Each cycle-step transition is itself a benchmarkable contract. Covered in Report 16 and `benchmarks_gallery.md`.

The cycle maps directly onto stack pieces:
- Step 1 → Tier 3 domain libraries (`somax`, `plumax`)
- Step 2 → `pipekit-cycle.DACycle` with classical analysis steps
- Step 3 → `pipekit-train.SimulationDataset` + emulator in `pipekit-experiment.ModelRegistry`
- Step 4 → `pipekit-cycle.DACycle` with `NeuralForward`
- Step 5 → `pipekit-train` with `gaussFlowX` / `pyrox-nn`
- Step 6 → `pipekit-evaluate.EvaluationReport` diffs.

### 3.6 How the three axes relate

The three axes form a cube of operations: at any point in the framework you’re somewhere on the modeling cycle (research stage), somewhere on the L0–L4 axis (data maturity), and somewhere on the six-step cycle (production maturity).

```{figure} images/overview/axes_matrix.svg
:name: fig-axes-matrix
:alt: The axes as a matrix: every L0–L4 transition crossed with the six-step cycle is a concrete, benchmarkable method, validated against the step-2 oracle column.
:width: 100%

The axes as a matrix: every L0–L4 transition crossed with the six-step cycle is a concrete, benchmarkable method, validated against the step-2 oracle column.
```

**Every cell is a concrete benchmarkable artifact.** Every cell uses the previous column as oracle. The whole L0–L4 chain is a nested set of six-step cycles, each gated by benchmarks. This is what makes the framework operationally improvable: improving any cell triggers re-running its downstream column; lineage is content-addressed; nothing is opaque.

-----

## 4. The big picture stack

Two granularities. First the high-level (six tiers), then the package map (~20 packages).

### 4.1 High-level — six tiers

```{figure} images/overview/tier_stack.svg
:name: fig-tier-stack
:alt: The stack in seven tiers. Solid tiers are owned; dashed tiers are external (georeader sits in Tier 0 as the GeoTensor substrate).
:width: 100%

The stack in seven tiers. Solid tiers are owned; dashed tiers are external (georeader sits in Tier 0 as the GeoTensor substrate).
```

### 4.2 The package map — Tier 4 zoomed in

The framework + infrastructure tier is where most current work happens. Detailed view:

```{figure} images/overview/tier4_map.svg
:name: fig-tier4-map
:alt: Tier 4 zoomed in: the pipekit framework, the ML-loop packages built on it, and the infrastructure peers.
:width: 100%

Tier 4 zoomed in: the pipekit framework, the ML-loop packages built on it, and the infrastructure peers.
```

### 4.3 The dependency graph

How they import each other, from the packages' metadata (`A ──► B`: A
depends on B; `A ┄► B`: through an optional extra):

```{figure} images/overview/dependency_graph.svg
:name: fig-dependency-graph
:alt: How the reference packages import each other, from their metadata. Solid edges are dependencies, dashed edges optional extras.
:width: 100%

How the reference packages import each other, from their metadata. Solid edges are dependencies, dashed edges optional extras.
```

**No circular dependencies.** Each layer only depends on layers below it. Domain libraries (geotoolz, xr-toolz) re-export common framework symbols so user code typically only imports one package.

-----

## 5. End-to-end walkthrough: methane attribution

The canonical example. Builds a complete L0→L4 pipeline from satellite scenes to source emission estimates, with ML at multiple stages.

```{figure} images/overview/methane_walkthrough.svg
:name: fig-methane-walkthrough
:alt: The methane attribution walkthrough: eight steps from cataloguing scenes to serving estimates, and the package that handles each.
:width: 100%

The methane attribution walkthrough: eight steps from cataloguing scenes to serving estimates, and the package that handles each.
```

The full pipeline, step by step, is in the [overview examples](#ex-methane).

The same pipeline, written as a YAML, becomes the regulatory artifact (per Report 9 use case 9). The same operators run in a notebook (Case 1), an ETL job (Case 2), a FastAPI endpoint (Case 5), or a Fused UDF (Reports 15 + the Fused integration design).

-----

## 6. Per-tier briefs

A compact summary per package. The deep dives are in the numbered reports.

### 6.1 — Tier 0: Substrate

**georeader** (external, but central). Owns `GeoTensor` (the raster carrier) and the reader infrastructure (`RasterioReader`, `AsyncGeoTIFFReader`). The bottom of the stack everything raster-shaped builds on.

**xarray ecosystem** (external). `DataArray` / `Dataset` / `DataTree` carriers plus rioxarray / xvec / cf-xarray / regionmask / xskillscore / dask / intake. Broad, community-standard.

### 6.2 — Tier 1: Numerical Foundation

| Package | What it does |
|---|---|
| `gaussX` | Structured linear algebra; Gaussian primitives; exponential family. Wraps lineax, matfree. |
| `spectralDiffX` | Pseudospectral differentiation, filters, PDE solvers. |
| `finiteVolX` | Arakawa C-grid finite-volume operators. |
| `optax-bayes` | Bayesian learning rule as optax GradientTransformations. |
| `eqx-trainer` | Minimal JAX/equinox training loop. |

Each is pure math. No domain knowledge. All `eqx.Module`-based; all jittable; all JAX-traceable.

The [overview examples](#ex-gaussx) show a gaussX kernel matrix.

### 6.3 — Tier 2: Algorithms

Inference / learning / model families. Domain-agnostic but problem-structure-aware.

| Package | What it does |
|---|---|
| `pyrox-gp` | GP building blocks for NumPyro hierarchical models. |
| `pyrox-nn` | Bayesian deep learning with equinox + NumPyro. |
| `filterX` | Differentiable ensemble Kalman methods. |
| `vardaX` | Learnable variational data assimilation (3D/4D-Var). |
| `gaussFlowX` | Rotation-based Gaussianization, normalizing flows. |
| `fairkl` | Fairness-constrained kernel learning (Keras 3). |
| `keras-nerf` | Neural fields for state estimation (Keras 3). |
| `keras-flows` | Normalizing flows (Keras 3). |

Keras 3 and Equinox are **equal-status backends**. Each algorithm’s adapter (`*.adapters.pipekit`) makes it composable into pipekit pipelines.

The [overview examples](#ex-filterx) show a filterX adapter used as a DA analysis step.

### 6.4 — Tier 3: Domain Models

Geophysical forward models and parameterizations.

| Package | What it does |
|---|---|
| `somax` | Ocean/atmosphere model zoo. |
| `xtremax` | Extreme value modeling. |
| `methanex` | End-to-end methane remote sensing. |
| `plumax` | Atmospheric plume transport simulation. |
| `RTMX` | Radiative transfer operators. |

Each has a thin `adapters/pipekit.py` exposing the operator surface (`ForwardModel`, `ObservationOperator`, etc.) without polluting the algorithmic core.

### 6.5 — Tier 4: Pipeline Infrastructure (the headline of this work)

**pipekit** — the framework. ~2,150 LOC. Twelve semantic groups (Reports 1, 2):

| Group | Operators |
|---|---|
| Foundations | Operator, Sequential, Graph, Input, Node, ConfigMixin |
| Convenience | pipe, compose, juxt, complement |
| Building blocks | Identity, Const, Lambda, Sink |
| Control flow | Branch, Switch, Try, Coalesce, Retry |
| Observers | Tap, Snapshot, ShapeTrace, Profile, Histogram |
| Combination | Fanout |
| Caching | Cache, Memoize |
| QC | Quarantine, AssertShape, AssertDType, AssertCallable |
| Shape inference | Signature, compute_output_signature |
| Parallelism | ThreadMap, ProcessMap, AsyncMap, BatchedMap |
| Serialisation | dumps, loads, sandboxed loader |
| State primitives | StatefulOperator, CarryState |

**pipekit-array** — Array API operators. Multi-backend (numpy/JAX/CuPy/PyTorch). `StackAlong`, `ConcatenateAlong`, `MeanScalar`; `ModelOp` lives in geotoolz. (Report 3)

**pipekit-jax** — Equinox-backed, JAX-traceable operators. Deferred to v0.3+. (Report 5)

**pipekit-cycle** — Time-stepping and DA. `Cycle`, `EnsembleCycle`, `DACycle`, three Protocols (`ForwardModel`, `ObservationOperator`, `AnalysisStep`). (Report 10)

**pipekit-train** — Training pipelines for emulators and amortized inference. Thin adapter over Lightning / Equinox+Optax / Keras 3. (Report 11)

**pipekit-experiment** — Tracker adapters (MLflow, W&B, DVC, Hydra) + content-addressed `ModelRegistry`. (Report 12)

**pipekit-evaluate** — Multidimensional evaluation framework. Three-axis matrix (Unit × Lens × Stage). Benchmark contracts as content-addressed artifacts. (Reports 14, 15)

**georeader** — substrate; covered above.

**geocatalog** — Spatiotemporal index for L0–L2 observations. Two backends (InMemory + DuckDB). `GeoSlice` as wire format. (Report 6)

**geopatcher** — Four-axis patcher framework. `Field` / `Domain` Protocols. ~60 classes for Geometry × Sampler × Window × Aggregation. (Report 7)

**statecatalog** — Spatiotemporal index for L3–L4 model states. `StateSlice` with valid_time/run_time/ensemble/lineage. (Report 13)

### 6.6 — Tier 3 (siblings): Domain Pipeline Libraries

**geotoolz** — GeoTensor-flavoured operators. Remote-sensing focus: sensor readers, retrieval, cloud, radiometry, indices, spectral, viz, plume, presets. (Reports 3, 8)

**xr-toolz** — xarray-flavoured operators. Ocean/atmosphere/reanalysis focus: validation, CRS, subset, masks, detrend, interpolate, metrics, kinematics, viz, calc, data sources, atm/ocn/ice. (Report 8)

Both consume the same Layer 4 peers. The substrate differs (georeader vs xarray ecosystem); the operator algebra is identical.

-----

## 7. Key code patterns

The code for each pattern is on the [overview examples](overview_examples.md) page:

- [Define an operator](#ex-operator)
- [Compose pipelines](#ex-compose)
- [Catalogs](#ex-catalogs)
- [Patcher (geopatcher)](#ex-patcher)
- [Data assimilation cycle](#ex-da-cycle)
- [Swap classical forward model for a neural emulator](#ex-emulator)
- [Evaluation pipeline](#ex-evaluation)
- [Benchmark contract](#ex-benchmark)
- [Reproducibility artifact (regulatory)](#ex-repro)

-----

## 8. Open questions

Honest list. None are blockers; all deserve thought before committing to specific decisions.

### 8.1 — Sequencing

- **Does pipekit-jax really need to wait until v0.3+?** Report 5 says yes. But the six-step cycle (§3.5) surfaces a concrete driver: differentiable Step 3 emulators are exactly what makes Step 4 (gradient-based inverse problems) tractable at scale. If a project demands amortized inference (Step 5) with end-to-end differentiability, that becomes the trigger for pipekit-jax — earlier than the v0.3+ default. Open until there’s a concrete project driver.
- **In-situ catalog backends — which one first?** Report 6 leans Argo (via argopy) but ICOADS, SOCAT, World Ocean Database all matter for different domains. One per quarter realistically. Choice is driven by project needs.

### 8.2 — Design

- **The Slice Protocol refactor.** Currently `GeoSlice` is the only slice type. Adding `ProfileSlice` and `TrajectorySlice` (per `data_ml_lifecycle.md`) is the right structural move but the refactor is 2 weeks of work that ripples across geocatalog + geotoolz + xr-toolz + pipekit-train. Worth doing before too much code commits to the current GeoSlice.
- **Where matchup lives.** `geocatalog.queries.matchup()` is the framework function; carrier-specific helpers in `xr_toolz.matchup` / `geotoolz.matchup`. But does matchup need its own catalog type (a “matchup catalog”)? Probably yes; not yet scoped.
- **Backend choice in pipekit-train.** Lightning / Equinox+Optax / Keras 3 are all supported. Are they truly equal-status or does one become the “blessed” choice for most use cases? Honest answer: Lightning is most likely to be the default for PyTorch models; Equinox+Optax for JAX-traceable / differentiable pipelines; Keras 3 for multi-backend. Worth being explicit per use case.

### 8.3 — Operational

- **Monorepo vs polyrepo.** Decision is monorepo (`pipekit-ecosystem` with uv/hatch workspaces); but the JAX-native algorithm libraries (gaussX, filterX, vardaX, plumax) are *outside* the monorepo. Where’s the line? Lean: framework + infrastructure in monorepo; algorithm + domain libraries are separate repos that ship adapter modules.
- **Documentation: one site or per-package sites?** Lean: one Sphinx (or mkdocs) site for the monorepo packages; separate sites for the JAX-native algorithm libraries (they have their own audiences). Plus a single JupyterBook research journal cross-cutting everything.
- **CI: per-package or per-monorepo?** Lean: per-package CI inside the monorepo + a single integration matrix that pins all packages to HEAD weekly. Releases via git tags (`geocatalog-v0.2.0`).

### 8.4 — Open scientific questions

- **Matchup uncertainty quantification.** `MatchupPair.representativeness_uncertainty` is easy to declare and hard to compute. Real estimation is its own research problem; v0.1 ships with a placeholder.
- **Cascade-uncertainty in multi-stage benchmarks.** The 5-stage emission estimation chain accumulates uncertainty across stages. Predicting this honestly is open research.
- **Probabilistic event detection.** Detecting “a plume happened” is binary; saying “there’s a 73% chance a plume happened” is much harder. Current event-detection metrics assume binary; probabilistic variants matter operationally but aren’t standard.

### 8.5 — Sustainability

- **Solo-maintainer feasibility.** ~20 packages, even with monorepo discipline, is a lot. The “scope discipline” principle in vision_v2 §7 is the operating constraint: each phase produces standalone value; the full stack doesn’t need to exist for individual layers to deliver.
- **Community adoption.** If `oceanbench` adopts pipekit-evaluate’s benchmark contracts, the framework is validated. If not, it stays personal infrastructure. Open question: at what point do you start actively reaching out to existing benchmark efforts?

-----

## 9. A plan

Concrete sequencing. Each phase produces usable, valuable packages on its own.

### Phase 1 — Numerical Foundation (done in part)

- gaussX (v0.0.6, done)
- spectralDiffX (draft)
- finiteVolX (draft)
- optax-bayes (draft)
- eqx-trainer (draft)

Time: ongoing.

### Phase 2 — Algorithms (in progress)

- pyrox-gp
- pyrox-nn
- filterX
- vardaX
- gaussFlowX

Time: ongoing, each at its own pace.

### Phase 3 — Domain Models (some done, some in progress)

- somax (PDE solver zoo)
- plumax (atmospheric plume transport)
- xtremax (extreme value)
- methanex (end-to-end methane)
- RTMX (radiative transfer)

Time: project-driven; each tied to a specific science question.

### Phase 4 — Pipeline Framework (load-bearing, current focus) — ~3-4 months

**Month 1**

- pipekit v0.1 (framework core, 12 groups + state primitives)
- pipekit-array v0.1 (Array API operators)
- Setup pipekit-ecosystem monorepo with uv/hatch workspaces

**Month 2**

- geocatalog v0.1 (InMemory + DuckDB backends; basic queries)
- geopatcher v0.1 (four-axis framework; 5 Field adapters)
- Refactor geotoolz and xr_toolz to depend on the new framework

**Month 3**

- statecatalog v0.1
- pipekit-cycle v0.1 (Cycle, DACycle, three Protocols)
- Adapter modules in filterX, vardaX, plumax

**Month 4**

- pipekit-train v0.1 (Lightning + Equinox backends)
- pipekit-experiment v0.1 (MLflow + S3 registry adapters)
- Initial integration testing across all packages

After Phase 4: the L0–L4 inference loop is structurally complete. ML training and serving still need work but the pipelines exist.

### Phase 5 — Evaluation + Benchmarks + Cycle Discipline — ~2 months

**Month 5**

- pipekit-evaluate v0.1 (framework + cheap metrics + EvaluationReport)
- Leakage-aware splitters in pipekit-train
- Standard baselines (Persistence, Climatology, OI)
- ModelRole enum + ArtifactLineage in pipekit-experiment (the six-step cycle's content-addressed lineage tracking)

**Month 6**

- xr_toolz.lagrangian (particle tracking)
- xr_toolz.events (event detection)
- BenchmarkContract + CycleStageBenchmark artifacts
- MultiTrackEvaluation built-in pattern
- First worked cycle (probably methane L1→L2 retrieval, end-to-end Steps 1-5)
- First worked benchmark instantiation (SSH from benchmarks_gallery)

After Phase 5: evaluation is first-class, benchmarks are content-addressable contracts, the full L0–L4 cycle has multidimensional eval discipline, **and the six-step research-to-production cycle is operationally tractable end-to-end**.

### Phase 6 — Deferred (when projects demand)

| Deferred item | Trigger |
|---|---|
| pipekit-jax (Equinox-backed, JAX-traceable operators) | first project needing differentiable retrievals at scale or JIT-compiled inference paths |
| Deployment adapters (pipekit.deploy.fused, .litserve, .modal) | first deployment use case beyond hand-shimming |
| Slice Protocol refactor (GeoSlice → Protocol + ProfileSlice + TrajectorySlice) | first project needing first-class in-situ catalog support (probably BGC benchmark from gallery) |
| Additional in-situ catalog backends (Argo, ICOADS, SOCAT, ...) | one per quarter, project-driven |

### Phase 7 — Polish + community

- Comprehensive documentation site (Sphinx-based)
- Research journal as JupyterBook (cross-cutting narrative)
- First external benchmark adoption (OceanBench? WeatherBench-Ocean?)
- Capstone tutorials per domain

### Reality check on the plan

Three honest pushbacks on my own sequencing:

1. **6 months for Phase 4-5 is optimistic for solo work.** Realistically 8-12 months. Mitigation: each month produces usable artifacts; nothing requires the full plan to finish to be valuable.
2. **Phase 6 might come earlier than planned.** If a project demands pipekit-jax or in-situ catalogs first, jump phases. The sequence is a default, not a contract.
3. **Phase 7 is open-ended.** Community adoption is years not months. Plan for the substrate to be useful even without external adoption; treat adoption as upside.

-----

## 10. Where to read next

Depending on what you’re trying to understand. These documents are not in
this repository yet (they live with the migration notes and will move into
`docs/` as they are written up); until then, the
[capability index](api/capabilities.md) and the reference repositories under
`references/` are the source of truth.

|If you want to understand…      |Read                                                                        |
|--------------------------------|----------------------------------------------------------------------------|
|**The big picture vision**      |`vision_v2.md`                                                              |
|**Why pipekit exists**          |Report 1 (toolz lineage, background)                                        |
|**What pipekit actually ships** |Report 2 (12 semantic groups, ~2150 LOC)                                    |
|**The sister-library structure**|Report 3 (pipekit-array, geotoolz, xr-toolz refactor)                       |
|**Concrete use cases**          |Report 9 (13 cases mapped to the full stack)                                |
|**The JAX-native future**       |Report 5 (pipekit-jax, Equinox backend)                                     |
|**Spatiotemporal catalogs**     |Report 6 (geocatalog)                                                       |
|**The patcher framework**       |Report 7 (geopatcher, four-axis, 60+ classes)                               |
|**xr-toolz’s role**             |Report 8 (substrate asymmetry vs geotoolz)                                  |
|**Time-stepping + DA**          |Report 10 (pipekit-cycle)                                                   |
|**Training pipelines**          |Report 11 (pipekit-train)                                                   |
|**Model registry + tracking**   |Report 12 (pipekit-experiment)                                              |
|**L3-L4 indexing**              |Report 13 (statecatalog)                                                    |
|**Multidimensional evaluation** |Report 14 (pipekit-evaluate, three-axis matrix)                             |
|**Benchmark infrastructure**    |Report 15 (benchmark contracts, leakage, baselines)                         |
|**The six-step modeling cycle** |Report 16 (cycle + Step 2 as oracle + benchmarks as understanding)          |
|**The data lifecycle in detail**|`data_ml_lifecycle.md` (three-layer model, two streams, depth axis, matchup)|
|**Concrete benchmarks**         |`benchmarks_gallery.md` (23 benchmarks across 6 domains)                    |

-----

## 11. Closing summary

The GeoStack is **a vertically integrated, JAX-native research software stack with operator-graph composition extending from structured linear algebra through pipeline infrastructure to indexed multi-tier data products** — structured by three orthogonal axes (the modeling cycle, the L0–L4 data lifecycle, and the six-step research-to-production cycle).

What makes it specific:

- **Three organising axes** — the research loop (modeling cycle), the data lifecycle (L0–L4 with two-stream convergence), and the production journey (six-step cycle from Simple Model to Amortized Predictor)
- **Operator algebra at every level**, with appropriate Protocols per layer (no universal base class)
- **Carrier-agnostic framework** (pipekit) bridging JAX-native numerics and xarray/GeoTensor data
- **Content-addressed everything** — data slices, model artifacts, evaluation reports, benchmark contracts, reproducibility artifacts, cycle-step lineage
- **ML pluggable at every L0–L4 transition** via the same training + registry + inference machinery
- **Step 2 as oracle** — classical methods (filterX, vardaX, pyrox-gp) stay alive forever as ground-truth generators, not as legacy code to be replaced
- **Benchmarks as formalised understanding** — contracts are operationalised hypotheses, not just leaderboards
- **Two-stream data lifecycle** (satellite + in-situ converging via matchup) with depth-axis support
- **Multidimensional evaluation** as first-class infrastructure, not an afterthought
- **Pre-registerable benchmarks** as content-addressable contracts
- **JAX-only differentiability** (gradients live in the JAX-level packages; xarray/pipekit/geotoolz are gradient-free orchestration)
- **Math-first documentation, code IS the paper** — every operator documented in unicode with theory

The vision document is the manifesto. The 16 reports are the scoping deep-dives. This document is the entry point. The plan above is a default sequence; project demands will modify it. The bones are right; what remains is execution at scope-disciplined pace.

If a specific use case is pulling you toward this stack: pick the relevant deep-dive from §10, read it, and come back with questions. The structure is intentionally modular — you don’t need to internalise everything to use any part.
