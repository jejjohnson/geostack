---
title: Benchmarking
subtitle: Report 15 · Benchmarks as content-addressed contracts, checked against what the stack runs today
short_title: 15 · Benchmarking
description: The scoping report argues that a benchmark is a frozen contract, not a scored run, and scopes five infrastructure pieces. A run of one SSH mapping benchmark on the pinned code shows which parts exist (a hash, a runner, the levels, the gap-fillers, CRPS) and which do not. This page keeps the contract, puts each of its terms in the library that owns the data it constrains, and corrects the split rule.
keywords:
  - benchmark
  - contract
  - pre-registration
  - leakage
  - block cross-validation
  - baselines
  - multi-track evaluation
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

Two teams that report "a 10 % improvement" on the same ocean dataset are often not comparing the same thing, because they split the data differently, use different baselines, score against different references, or keep iterating on the test set. The scoping report calls these **contract failures**, and it proposes that a benchmark be a frozen, content-addressed **contract** (task, carriers, splits, references, metrics, baselines and submission rules) with five pieces of infrastructure to enforce it. In this page, we run one sea surface height (SSH) mapping benchmark through the pinned code. The contract's hash and the runner already exist: pipekit hashes any configuration, and pipekit-evaluate ships a `Benchmark` runner and the L0 to L4 levels which the scoping report's task list restates. The data and the baselines are also within reach, as xrtoolz ships the gap-fillers, CRPS and spread-skill. What is missing is the contract itself, splits that know about space and platforms, and a report that names the contract it scored. The run also corrects the scoping report's split rule: **the held-out set must reproduce the gap the method will face in use**, which is a withheld altimeter for mapping, whereas a block larger than the correlation scale is only right for generalization to unseen regions or periods. On the same method, this choice moves the error from 1.68 to 4.18 cm. Every number below is from a real run.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Partly built. Originally a scoping proposal ("committed": a framework, with a gallery, now the [benchmarks gallery](../benchmarks_gallery.md)); the framework is kept, the infrastructure is redistributed across the libraries that own each term |
| **Reading time** | about 30 minutes |
| **Audience** | anyone designing a benchmark, entering one, or turning a paper's evaluation into one: paper authors, curators, OceanBench or WeatherBench adopters |
| **Companions** | [Report 14](14_pipekit_evaluate.md) (pipekit-evaluate, whose runner and report this extends); [Report 11](11_pipekit_train.md) (the training splits); [Report 12](12_pipekit_experiment.md) (the registry); [Report 13](13_state_catalog.md) (analyses, reanalyses and forecasts as catalog rows); [Report 6](06_geocatalog.md) (catalogs and matchups) |
| **Code checked against** | pipekit 0.0.2 at `ef3c17f` (pipekit, pipekit-evaluate, pipekit-train, pipekit-experiment, pipekit-cycle); xrtoolz 0.0.5 at `308c7f2`; geotoolz 0.10.2 with geocatalog 0.5.1 and geopatcher at `088475e`. Read on 13 October 2026 |
| **Running example** | every output is from one run of `notebooks/reports/report15_benchmarking.py`, which builds the field and the altimeters from fixed seeds in memory and asserts each number the page shows |
| **Sources** | the scoping report's source material: external writing on ocean ML benchmarks, OceanBench's multi-track pattern [@johnson2023oceanbench], and WeatherBench 2 [@rasp2024weatherbench] as the second community benchmark it names |
:::

:::{important} What changed from the scoping report
The **central claim stands**: a benchmark is a contract, and stages 1 to 5 reproduce each contract failure it lists. However, three parts do not:

- **Task list:** the tasks become the L0 to L4 levels that pipekit-evaluate already ships, read as transitions ([](#r15-framework)).
- **Split rule:** blocks above the correlation scale overstate the mapping error by 55 %, so the held-out set should reproduce the gap in use ([](#r15-stage-2)).
- **Package plan:** splits go to geocatalog and baselines go beside xrtoolz's gap-fillers, following [Report 14](14_pipekit_evaluate.md#r14-ownership)'s rule ([](#r15-stage-3)).
:::

(r15-summary)=
## Summary

**Decision.** A benchmark is a `BenchmarkContract`, which is frozen data in pipekit-evaluate, hashed with pipekit's `stable_json` and `sha256_hex` and recorded on every report it produces. We implement each term of the contract where its data lives: splits of catalog rows in geocatalog and splits of training windows in geopatcher (pipekit-train until [Report 16's D2](16_geo_data_stack.md#r16-ws-d) moves the window dataset), baselines and probabilistic scores in xrtoolz, and the content-addressed store for contracts and submissions in pipekit-experiment. There is no new package and no hosted scoring service.

- **A. The contract.** A frozen `BenchmarkContract` with a task given as a level transition (L1 → L2 for mapping), carriers, splits, references, scorers, baselines and a status (pre-registered or exploratory), which is hashed and round-tripped through YAML (A1), and whose hash is recorded on every report (A2).
- **B. Splits as contract terms.** Block, platform and causal splits of catalog rows in geocatalog, where each one refuses a configuration it cannot honour (B1, B2), and the training window's guard band in time units, which refuses an empty split (B3). The contract states the gap that the split reproduces.
- **C. Baselines with the carriers.** Space-time optimal interpolation with a valid covariance in `xrtoolz.interpolate` (C1), persistence with an explicit reference time and gap rule (C2), and a climatology fitted on the training period only (C3). The contract pins each configuration.
- **D. Variants and tracks in the runner.** Entries and scorers declare a variant, which the report records, and point forecasts are scored under CRPS as the degenerate distributions they are (D1). One run scores against several references and reports a missing one as unavailable (D2), and entries are named rather than keyed by their ladder cell (D3).
- **E. Registry and submissions.** Contracts and reports are stored content-addressed in pipekit-experiment (E1), and a submission pins the contract, the model and the report hashes and verifies against them (E2).

**End state.** A curator writes a contract and publishes its hash, and then anyone can run it. The splits refuse configurations that would leak, the baselines are the same code for every entry, and the runner records each entry's variant and scores every track. The report names the contract and the data, and each of its rows names the model it scored. Running the leaderboard, a blind test service and its queue stays with the projects that host one. Each transition of the [six-step cycle](../overview.md#sec-six-steps) is benchmarked by one such contract ([Report 17](17_cycle.md#r17-ws-a)).

(r15-framework)=
## The framework, as the scoping report put it

This section restates the scoping report's framework, and the rest of the page keeps it except where a box says otherwise.

**Benchmark versus evaluation.** An evaluation scores one model, once, with whatever data and choices are at hand. A benchmark, however, is a shared contract which fixes everything a comparison depends on before anyone submits ([](#tbl-r15-bench-vs-eval)).

:::{table} The scoping report's distinction between an evaluation and a benchmark.
:label: tbl-r15-bench-vs-eval

| | Evaluation | Benchmark |
|---|---|---|
| What it is | the act of scoring a model | a shared, pre-registered evaluation contract |
| Scope | one project, one run | across projects and teams |
| Permanence | ephemeral | versioned and immutable |
| Reference data | whatever is at hand | curated, frozen, content-addressed |
| Preprocessing, splits | each team's choice | fixed and distributed as code |
| Metrics | each team's choice | pre-registered before submissions open |
| Baselines | optional | mandatory, one shared implementation |
:::

**Tasks as carrier transformations.** Each task maps an input carrier to an output carrier, so a model is simply anything that performs the map: 1) discretization (points to a sparse grid), 2) gap filling (to a dense grid), 3) analysis (to a state), 4) reanalysis (to a smoothed state trajectory), 5) forecast (a state to a future state) and 6) representation (a state to a latent and back). Every task comes in a **deterministic** and a **probabilistic** variant, which are scored by different metrics.

**Three tracks.** The same predictions are scored against a reanalysis, an analysis and the raw observations. A model that wins on the reanalysis and loses on the observations has learned the assimilation product, not the ocean.

**Five pieces of infrastructure.** These are 1) leakage-aware splitters, 2) the contract artifact, 3) a baseline registry, 4) pre-registration by content address, and 5) a blind test set behind catalog access control, plus multi-track evaluation as a built-in pattern.

:::{warning} Differs from the plan
The six tasks are the transitions between the levels that pipekit-evaluate already names (`benchmark/cube.py`, `LEVEL_NAMES`; the levels are listed in the overview's [](../overview.md#tbl-levels)). However, the two lists disagree on L2: `LEVEL_NAMES` calls it gap-filled observations, whereas the overview calls it retrieved observations, and this page keeps the code's meaning until [Report 17 A1](17_cycle.md#r17-ws-a)'s rename. Discretization is L0 → L1, gap filling is L1 → L2, analysis is L2 → L3, and reanalysis and forecast end at L4. The contract names its task by that transition instead of by a second `Literal` list, so a benchmark cell and a contract use the same vocabulary. Representation is the exception, as it is not a level transition and nothing in pipekit-train ships an encoder or a decoder to map it to.
:::

(r15-ownership)=
## Ownership rules

The contract is data in pipekit-evaluate, and every term it fixes is implemented by the library that owns the data which that term constrains ([](#fig-r15-ownership)).

:::{figure} ../images/reports/r15_ownership.svg
:label: fig-r15-ownership
:alt: Benchmark projects write contracts and read reports through pipekit-evaluate, which ships Benchmark, ReferenceRule, Scorer, Task and BenchmarkReport and would add BenchmarkContract, variant and track checks and a contract hash on reports. Below it, geocatalog holds block, platform and causal splits of rows, geopatcher the window splits and their guard (in pipekit-train until Report 16's D2 moves them), xrtoolz the baselines and CRPS, and pipekit-experiment the content-addressed store. All four use pipekit's hashing.

The contract lives in pipekit-evaluate; its splits, baselines and storage
live with their data.
:::

1. **pipekit-evaluate** owns the contract, the runner, the scorer tags (variant, beside [Report 14](14_pipekit_evaluate.md#r14-ws-a)'s unit, lens and stage) and the report. It imports geocatalog, xrtoolz and pipekit-experiment only through optional extras, as it does with pipekit-cycle today.
2. **geocatalog** owns splits of catalog rows (blocks in space and time, a held-out platform, a causal cut). A row already carries the footprint, the time interval and, with [Report 13](13_state_catalog.md#r13-ws-a)'s profile, the producer, which is all a split needs. It never imports pipekit-evaluate.
3. **The window dataset** owns splits of training windows inside a dataset, and it reads the contract's split as plain arguments (fractions, a guard in time units), so it never parses a contract. It is pipekit-train's `XarrayWindowDataset` today and moves to `geopatcher.integrations.train` with [Report 16](16_geo_data_stack.md#r16-ws-d)'s D2, and B3 lands wherever it is when B3 is done.

The baselines and probabilistic scores go to xrtoolz (geotoolz for GeoTensor carriers; [](#r15-ws-c)) and the content-addressed store goes to pipekit-experiment ([](#r15-ws-e)), whereas curation, the leaderboard and any blind service stay with benchmark projects such as OceanBench, WeatherBench and the [gallery](../benchmarks_gallery.md)'s entries. Otherwise, neighbours follow [the stack rules](16_geo_data_stack.md#r16-ownership).

(r15-running)=
## The running example: mapping sea surface height from three altimeters

**The problem.** We want to map SSH anomalies every day from along-track altimetry, and to decide how to rank mapping methods fairly. This is the task that OceanBench's first edition poses on real data [@johnson2023oceanbench]. Here the field is synthetic, so the truth is known and every score can be checked against theory.

**The data.** The field $\eta(\vect{s}, t) \in \mathbb{R}$ is an SSH anomaly in metres at the location $\vect{s} \in \mathbb{R}^2$ on a doubly periodic 800 km square, with 32 × 32 cells of $\Delta = 25$ km, daily from 1 January 2023 to 30 December 2025 ($T = 1{,}095$ days, so the test year 2025 has 364). It is a seasonal cycle of amplitude $a = 4$ cm plus a mesoscale anomaly with standard deviation $\sigma_k = 8$ cm and the Gaussian kernel

$$
k((\vect{s}, t), (\vect{s}', t')) = \sigma_k^2 \exp\left(-\frac{r^2}{2 \ell_s^2} - \frac{(t - t')^2}{2 \ell_t^2}\right) ,
$$ (r15-eq-cov)

where $r = \Vert \vect{s} - \vect{s}' \Vert_2 \in \mathbb{R}_{\ge 0}$ is the distance between the two points, and $\ell_s = 100$ km and $\ell_t = 30$ days are the space and time lengthscales (the mesoscale values that the scoping report gives for SSH), all of them scalars. Space is periodic, but time is not. The field is drawn by FFT over a year longer than the record and the extra year is dropped, so the end of 2025 is uncorrelated with the start of 2023. Three altimeters s1, s2 and s3 each fly one diagonal ground track across the domain every third day and measure $\eta$ with a noise of $\sigma_\varepsilon = 2$ cm, which gives 35,040 observations (11,680 per satellite). The task is gap filling, L1 → L2, so from the observations we produce $\eta$ on the full grid.

**What exists.** The contract in stage 1 is a dictionary hashed with pipekit's own primitives. Where the stack has no mapping method, we write it in the notebook: an inverse-distance mapper in space and time (xrtoolz's `idw_to_points` only weighs in two dimensions) and an optimal interpolation (proposed as C1). Lines marked `user` ([markers](../overview_examples.md#ex-markers)) are code that the workstreams move upstream.

(r15-stage-1)=
### Stage 1 · Write the contract and hash it (pipekit → pipekit-evaluate)

**TL;DR.** Write down everything the comparison depends on and give it a fingerprint that changes when anything in it changes.

**Problem.** A contract $c$ is a nested mapping of the task, carriers, splits, references, metrics and baselines. Its address {eq}`r15-eq-hash` is

$$
h(c) = \mathrm{SHA256}\left(\mathrm{json}_\text{sorted}(c)\right) ,
$$ (r15-eq-hash)

which is the hash of its JSON encoding with sorted keys. So two contracts with the same content get the same $h$ whatever the order they were written in, and any edit gives a new $h$.

```{code-block} python
:caption: The stage-1 contract as data, from a real run.
:linenos:

# Shapes: input the s2, s3 obs binned to the grid (1,095, 32, 32), NaN off-track;
#         output (364, 32, 32) maps of 2025
from pipekit.hashing import sha256_hex, stable_json
contract = {
    "task": "L1->L2", "operation": "gap filling", "variant": "deterministic",
    "input": {"carrier": "sparse grid", "sources": ["s2", "s3"], "units": "m",
              "aggregate": "mean"},  # coincident s2/s3 samples in one cell
    "output": {"carrier": "grid", "shape": [364, 32, 32], "units": "m"},
    "splits": {"kind": "leave_one_platform_out", "platform": "s1",
               "period": ["2025-01-01", "2025-12-30"]},
    "references": {                 # each derived, with its settings
        "truth": {"from": "data.generator", "period": [...]},
        "reanalysis": {"from": "truth", "op": "scipy.ndimage.gaussian_filter",
                       "sigma_cells": [0, 2, 2], "mode": "wrap", "code": "scipy@…"},
        "observations": {"from": "data.generator", "platform": "s1", ...}},
    "data": {"generator": GEN},     # grid, ℓ_s, ℓ_t, σ_k, noise, orbits, seed
    "metrics": [{"name": "rmse", "code": "xrtoolz.metrics.rmse@308c7f2"}],
    "scoring": {                    # how each track is matched and reduced
        "truth": {"carrier": "grid", "dims": ["time", "y", "x"]},
        "reanalysis": {"carrier": "grid", "dims": ["time", "y", "x"]},
        "observations": {"carrier": "points", "matchup": "cell_of_obs",
                         "match_on": ["time", "y", "x"], "dims": ["obs"]}},
    "baselines": [                  # every setting that changes a score
        {"name": "linear_in_time", "method": "linear", "edge_fill_m": 0.0,
         "code": "xrtoolz.interpolate.fillnan_temporal@308c7f2"},
        {"name": "oi_100km", "length_km": 100, "time_days": 30, "window_days": 15,
         "space_kernel": "gaussian_periodic_images", "images": [-3, 3],
         "code": "notebook source sha256:5288366f…", ...},
        ...],                       # six in all: stage 3's
}
# h = SHA-256 of the key-sorted JSON
h = sha256_hex(stable_json(contract))           # → "4f0a8b663178…"     today
# reverse key order → same h; withhold s2 → "8d701687cece…"; seed 1, another
# OI kernel or another edge fill → a new h each
[f.name for f in dataclasses.fields(pe.BenchmarkReport)]
# → ["task", "domain", "axis", "results"]: no contract, no model   today
```

The fingerprint costs nothing, as pipekit already uses `stable_json` and `sha256_hex` for the registry and the dataset hashes, and the whole contract is about 3 KB of JSON. However, it must identify the data and the baselines and not only name them. Labels such as `"osse"` and `"s1"` say nothing about which arrays they stand for, so this contract hashes the generator's full configuration and seed, and a contract over real data would carry the catalog's content hash for each input and reference. In the same way, each baseline carries every setting that changes its score (down to the kernel and the fill for cells with no observation on one side) and the revision of the code that implements it. This is a library at a pinned commit or, for the OI that lives in the notebook until workstream C1, the SHA-256 of its source, so that editing it changes the hash. Each reference carries its derivation (the reanalysis stand-in is a Gaussian filter of the truth, and its `sigma` and boundary `mode` are in the contract), and each track says how it is scored: the grids cell by cell, and the withheld s1 observations by sampling the map at each observation's cell and reducing over the observations. Stage 3 shows that each of these settings moves the result. What is missing is a place to put it. pipekit-evaluate's `BenchmarkReport` records the task name, the domain, the axis and the cell results, so a report cannot say which contract, which model or which data it scored, and a submission cannot be checked against anything (workstreams A and E).

:::{warning} Differs from the plan
The scoping report's sketch also carried `block_size_spatial_km` and `block_size_temporal_days` beside `splitter_config`. Here they live only in the split configuration, so the two cannot disagree, and the overview's [contract example](../overview_examples.md#ex-benchmark) had already made that change. The report expected about 50 KB of YAML, whereas a full contract for this benchmark is about 3 KB.
:::

(r15-stage-2)=
### Stage 2 · One method, six splits (numpy and scipy, then pipekit-train)

**TL;DR.** Score one mapping method under six ways of holding data out, and see which of them predicts the error the method will make in use.

**Problem.** Let $\mathcal{D} = \{(\vect{s}_n, t_n, y_n)\}$ be the set of observations, where $y_n \in \mathbb{R}$ is the SSH that sample $n$ measured at the location $\vect{s}_n \in \mathbb{R}^2$ and the time $t_n \in \mathbb{R}$. A split divides $\mathcal{D}$ into a training set $\mathcal{D}_\text{train}$ and a test set $\mathcal{D}_\text{test}$, and the method sees $\mathcal{D}_\text{train}$ and predicts $\hat{\eta}_n = \hat{\eta}(\vect{s}_n, t_n) \in \mathbb{R}$ at each test point. The score is

$$
\mathrm{RMSE} = \left(\frac{1}{\vert \mathcal{D}_\text{test} \vert} \sum_{n \in \mathcal{D}_\text{test}} (\hat{\eta}_n - y_n)^2\right)^{1/2} ,
$$ (r15-eq-rmse)

which we compute against the noisy observation $y_n$ and, with $\eta(\vect{s}_n, t_n)$ in place of $y_n$, against the true field. The method is inverse-distance weighting of the 10 nearest training observations in the scaled coordinates $(\vect{s} / \ell_s, t / \ell_t)$.

```{code-block} python
:caption: The same method under six splits, from a real run (RMSE in cm, against the observation and the truth).
:linenos:

# Shapes: 35,040 obs; scaled coordinates (n, 3) = (s_1 / ℓ_s, s_2 / ℓ_s, t / ℓ_t)
# η̂ = Σ w_n y_n / Σ w_n,  w_n = 1 / d_n over the 10 nearest      user
splits = {
    "random 80/20": random_split(0.2),                  # 2.62 / 1.73   user
    "withhold s1": platform_split("s1"),                # 3.32 / 2.70   user
    "withhold s2": platform_split("s2"),                # 2.75 / 1.92   user
    "blocks 50 km × 7 d": blocks(50, 7),                # 2.61 / 1.68   user
    "blocks 200 km × 60 d": blocks(200, 60),            # 4.58 / 4.18   user
    "last 20 % of 2025": causal_split(0.8),             # 8.65 / 8.39   user
}
```

:::{table} The six splits of stage 2: what each reproduces, and the method's error under it.
:label: tbl-r15-splits

| Split | Reproduces | RMSE vs observations (cm) | RMSE vs truth (cm) |
|---|---|---|---|
| random 80/20 | nothing: test points sit beside training points on the same track | 2.62 | 1.73 |
| withhold s1 | mapping where one altimeter flew, from the other two | 3.32 | 2.70 |
| withhold s2 | the same, for s2's orbit | 2.75 | 1.92 |
| blocks 50 km × 7 d | gaps smaller than ℓ_s and ℓ_t | 2.61 | 1.68 |
| blocks 200 km × 60 d | a region and season with no data at all | 4.58 | 4.18 |
| last 20 % of 2025 | a forecast of the last 73 days of 2025, with no data | 8.65 | 8.39 |
:::

The splits disagree by a factor of five ([](#tbl-r15-splits)), and no single one is "honest". A mapping product is used where an altimeter did not fly while the others did, so the withheld-satellite rows predict its error in use, which is 2.7 cm against the truth for s1. The random split is optimistic by 36 % against that, because each test point has a training neighbour 25 km away on the same track and the same day. The 200 km × 60 day blocks that the scoping report prescribes for SSH are pessimistic by 55 %, because they ask the method to fill a region and a season with no data, which is a task it is not used for. In addition, which satellite is withheld matters (2.70 against 1.92 cm), as s1's orbit leaves wider gaps between the other two. A fourth satellite or a different orbit would change every row. So the split is a term of the contract, not a team's choice.

```{code-block} python
:caption: The training loader's own split, from a real run.
:linenos:

# Shapes: one cell's daily series, T = 1,095; input stencil −2 … 0 d, target +1 d
ds = pt.XarrayWindowDataset(source={"input": cell, "target": cell},
                            stencils=stencils, store_version="v1")    # today
ds._split_guard, [len(ds.with_split(s)) for s in ("train", "val", "test")]
# → 2, [871, 105, 108]: contiguous 80/10/10, a guard of 2 origins
# 2 origins dropped each side: targets 5 d apart, ρ(5 d) = exp(−25 / 1,800) = 0.986
pt.XarrayWindowDataset(..., split_guard=60)       # 60 d = 2 ℓ_t        today
# → 60, [813, 0, 50]: the validation split is empty, silently
```

pipekit-train's window dataset already splits contiguously in time, which is the right shape for a forecast. Its guard band is the stencil's reach (2 origins, dropped on each side of a boundary), so the last training target and the first validation target are five days apart and correlated at 0.986. However, a guard of two correlation times leaves the 105-day validation split (which has a boundary on each side) empty, and the dataset returns zero windows instead of refusing (workstream B3).

:::{warning} Differs from the plan
The scoping report's rule is that "the block size for splitting must exceed the integral spatial/temporal scale of the variable". This holds for generalization to unseen regions or periods, and for forecasts (the causal split). However, for mapping it overstates the error by 55 % here, because the deployment gaps are the spaces between tracks and not whole regions. So we correct the rule to **the held-out set reproduces the gap the method will face in use**, and the contract states which gap that is. Blocks above the correlation scale are one answer to that question, not the rule.
:::

(r15-stage-3)=
### Stage 3 · The baselines a contract must pin (xrtoolz)

**TL;DR.** Score the simple methods every paper compares against, and see how much "the baseline" depends on how it was written.

**Problem.** The input is the 2025 maps gridded from s2 and s3, with no value where neither flew, and s1 is withheld. Where s2 and s3 sample the same cell on the same day (242 cells in 2025), the grid holds their mean. Every baseline, OI included, reads this grid rather than the raw tracks, so they all see the same input. A baseline fills every empty cell. Optimal interpolation (OI) is the best linear estimate under the kernel {eq}`r15-eq-cov`, with its spatial factor summed over the domain's periodic images so that it stays a valid covariance. Let $\vect{y} \in \mathbb{R}^{D_y}$ be the $D_y$ non-empty cells in a window of ±15 days and $\vect{x} \in \mathbb{R}^{D_x}$ the day's map on the $D_x$ grid cells. We write $\gmat{\Sigma}_y \in \mathbb{R}^{D_y \times D_y}$ for the covariance of the observations (the kernel between the observed cells plus $\sigma_\varepsilon^2 \mat{I}$, with $\mat{I} \in \mathbb{R}^{D_y \times D_y}$ the identity), $\gmat{\Sigma}_{xy} \in \mathbb{R}^{D_x \times D_y}$ for the covariance between the grid cells and the observations, and $\mu \in \mathbb{R}$ for the window mean, with $\vect{1}$ a vector of ones (in $\mathbb{R}^{D_x}$ or $\mathbb{R}^{D_y}$, to match). The estimate is {eq}`r15-eq-oi`

$$
\hat{\vect{x}} = \mu \vect{1} + \gmat{\Sigma}_{xy} \gmat{\Sigma}_y^{-1} (\vect{y} - \mu \vect{1}) ,
$$ (r15-eq-oi)

and the variance of grid cell $d$ is

$$
\hat{\sigma}_d^2 = \sigma_k^2 - [\gmat{\Sigma}_{xy} \gmat{\Sigma}_y^{-1} \gmat{\Sigma}_{xy}^\top]_{dd} ,
$$ (r15-eq-oi-var)

where $\hat{\vect{x}} \in \mathbb{R}^{D_x}$ and $\hat{\sigma}_d^2 \in \mathbb{R}_{\ge 0}$.

```{code-block} python
:caption: Six baselines on the 2025 maps, from a real run (RMSE in cm against the truth and against the withheld s1).
:linenos:

# Shapes: G (T = 1,095, 32, 32) with NaN off-track, coincident s2/s3 samples
#         averaged; every baseline, OI included, reads G; scored on 2025's 364 days
from xrtoolz.interpolate import fillnan_climatology, fillnan_temporal
G.ffill("time")      # pandas; persistence (carry fwd) 7.98 / 7.49  user
fillnan_temporal(G, method="nearest")   # nearest in time    5.59 / 5.72  today
fillnan_temporal(G, method="linear")    # linear in time     4.77 / 5.02  today
fillnan_climatology(G, group="month", residual="zero")
                                        # monthly clim.      8.35 / 7.87  today
# x̂ = μ + Σ_xy Σ_y⁻¹ (y − μ), one solve per day, ±15 d
oi(G, length_km=50, time_days=30)       # OI, ℓ_s = 50 km     2.66 / 3.20  user
oi(G, length_km=100, time_days=30)      # OI, ℓ_s = 100 km    1.81 / 2.62  user
# 1 − 1.81 / RMSE(baseline): 77 % over persistence; 68 % and 62 % over
# nearest and linear interpolation in time
```

Only carry forward is persistence, as it repeats the last observation. Nearest and linear interpolation in time also look at later observations, so they are interpolation baselines and not persistence, and a forecast contract could not use them. Against these three simple baselines, OI improves by 77, 68 and 62 %, so "an improvement over the baseline" means nothing until the contract pins which baseline it is and how it handles gaps (all three leave cells empty before the first or after the last observation, which the notebook fills with zero). OI itself depends on its assumptions, and half the right correlation length costs 47 %. Furthermore, the assumption can break the method. The usual Gaussian over the shortest distance on a periodic domain is not a covariance at all: before the noise term is added, its smallest eigenvalue on a day of 2025 is −9.1 × 10⁻⁶ m² at $\ell_s = 100$ km and −1.7 × 10⁻³ m² at $\ell_s = 150$ km. The noise variance $\sigma_\varepsilon^2 = 4 \times 10^{-4}$ m² hides the first and not the second. The Gaussian summed over the domain's periodic images is a valid covariance (its smallest eigenvalue is zero to rounding at both lengths), and it is what the notebook's OI uses. So a baseline must also fix a valid kernel for its domain. xrtoolz ships every gap-filler above except carry forward and OI, and it has `fillnan_rbf` and inverse-distance weighting but no covariance-based interpolation (workstreams C1 and C2).

:::{warning} Differs from the plan
The scoping report puts the baselines in a new `pipekit_evaluate.baselines` as `Operator` subclasses. However, a gap-filling baseline is simply a gap-filler, and xrtoolz already holds seven of them (`fillnan_temporal`, `fillnan_climatology`, `fillnan_spatial`, `fillnan_laplacian`, `fillnan_biharmonic`, `fillnan_rbf`, `fillnan_idw`) with the xarray carrier they need. A second home would split one concept across two packages, which [Report 14](14_pipekit_evaluate.md#r14-ownership) refused for metrics. So the baselines go into `xrtoolz.interpolate` (OI) and beside `xrtoolz.geo`'s climatology (persistence, climatology forecasts), and the contract pins their configuration by name and arguments.
:::

(r15-stage-4)=
### Stage 4 · A deterministic and an ensemble entry under CRPS (xrtoolz)

**TL;DR.** Score a map and an ensemble of maps with the same probabilistic score, and see whether the score tells them apart.

**Problem.** The continuous ranked probability score of a forecast distribution $F$ against the true value $\eta \in \mathbb{R}$, which we estimate from $N_{\mathcal{E}}$ members $\eta^{(e)} \in \mathbb{R}$ with $e \in \mathcal{E}$, is {eq}`r15-eq-crps`:

$$
\mathrm{CRPS}(F, \eta) = \mathbb{E}_F \vert \zeta - \eta \vert - \frac{1}{2} \mathbb{E}_F \vert \zeta - \zeta' \vert ,
$$ (r15-eq-crps)

where $\zeta, \zeta' \in \mathbb{R}$ are independent draws from $F$. For a point forecast, the second term is zero and the CRPS is the mean absolute error. The ensemble has $N_{\mathcal{E}} = 20$ members $\eta^{(e)} = \hat{\eta} + \hat{\sigma} z^{(e)}$, with $z^{(e)} \sim \mathcal{N}(0, 1)$, drawn from the $\ell_s = 100$ km OI of stage 3 (its mean $\hat{\eta}$ and standard deviation $\hat{\sigma}$ at each cell, from {eq}`r15-eq-oi` and {eq}`r15-eq-oi-var`).

```{code-block} python
:caption: CRPS of an ensemble and of a deterministic map, from a real run.
:linenos:

# Shapes: m25 (364, 32, 32); ens (N_ℰ = 20, 364, 32, 32)
from xrtoolz.metrics import crps_ensemble, spread_skill_ratio
crps_ensemble(ens, truth25, dims=dims)         # → 0.96 cm              today
spread_skill_ratio(ens, truth25, dims=dims)    # → 0.96                 today
crps_ensemble(m25.expand_dims(member=1), truth25, dims=dims)
# → ValueError: Ensemble/sample dim 'member' has size 1; need at least 2.
crps_ensemble(xr.concat([m25, m25], "member"), truth25, dims=dims)
# → 1.29 cm, equal to the mean absolute error of m25                    today
```

A point forecast is a valid (if degenerate) predictive distribution, and its CRPS is its mean absolute error. WeatherBench 2 scores deterministic baselines in that way beside ensembles [@rasp2024weatherbench]. So the comparison is meaningful: the map scores 1.29 cm and the calibrated ensemble 0.96 cm, and the proper score rewards the ensemble for its spread. The defect is in the plumbing. xrtoolz refuses a one-member ensemble, so a point entry has to be entered twice to be scored. The fix (workstream D1) scores point forecasts directly and records each entry's variant on the report, so a reader sees which rows are deterministic, whereas refusing them would remove the baselines that a probabilistic benchmark needs.

(r15-stage-5)=
### Stage 5 · Three references (xrtoolz → pipekit-evaluate)

**TL;DR.** Score the same maps against a reanalysis, the truth and the withheld observations, and see whether the ranking survives.

**Problem.** A reanalysis is the output of an assimilation system, and it inherits that system's smoothing. Here the stand-in is the true field smoothed with a Gaussian of 50 km (two cells). "Learned the reanalysis" is the $\ell_s = 100$ km OI map smoothed in the same way, as a model trained against that product would be. Each entry is scored with {eq}`r15-eq-rmse` on three tracks.

```{code-block} python
:caption: One set of maps, three tracks, from a real run (RMSE in cm).
:linenos:

# Shapes: each entry (364, 32, 32); s1 track: the withheld 2025 observations
rean = truth.copy(data=gaussian_filter(eta, sigma=(0, 2, 2), mode="wrap"))  # user
for name, p in entries.items():                     # reanalysis · truth · s1
    scores[name] = (rmse(p, rean), rmse(p, truth), rmse_at(p, s1))      # user
# OI, ℓ_s = 100 km        2.27 · 1.81 · 2.62
# OI, ℓ_s = 50 km         2.82 · 2.66 · 3.20
# learned the reanalysis  1.24 · 2.66 · 3.27
```

The entry that learned the reanalysis wins its track by a factor of 1.8 and loses the other two, coming last on the observations. The truth (which a real benchmark does not have) and the withheld observations agree on the winner, the $\ell_s = 100$ km OI, whereas the reanalysis does not. This reproduces the scoping report's three-track argument, as a single reanalysis track would have put first the map which comes last on the observations. Today each track is a separate loop that the user writes, because the runner takes one reference per cell (workstream D2).

(r15-stage-6)=
### Stage 6 · The runner pipekit-evaluate ships (pipekit-evaluate)

**TL;DR.** Run the baselines through the benchmark runner that exists, and see what it records.

**Problem.** pipekit-evaluate's `Benchmark` takes 1) a `Task` whose `datasets()` returns the observations and the truth, 2) a list of `Estimator`s, each placed at a ladder cell (level, rung, complexity), and 3) `Scorer`s. It resolves each cell's reference with its `ReferenceRule` and then scores the prediction.

```{code-block} python
:caption: Three baselines as estimators in the shipped runner, from a real run.
:linenos:

# Shapes: obs G (1,095, 32, 32); θ the true 2025 field (364, 32, 32)
ests = [Baseline("linear in time", rung=2, complexity=0),             # user
        Baseline("OI, L = 50 km", rung=2, complexity=1),
        Baseline("OI, L = 100 km", rung=2, complexity=2)]
rep = pe.Benchmark(task=MappingTask(), estimators=ests,
                   scorers=[RMSEScorer()]).run()                      # today
# → L4/r2/c0 known_theta 4.77 · L4/r2/c1 known_theta 2.66
#   L4/r2/c2 known_theta 1.81
pe.Benchmark(..., estimators=[*ests, Baseline("nearest in time", 2, 0)])
# → ValueError: duplicate benchmark cells (rung, complexity): [(2, 0)]  today
```

The runner works on a mapping task as written, because the protocols are structural, so a user's task and estimators satisfy them without importing pipekit-evaluate. However, three things stop it from running a contract. 1) Its cells are ladder coordinates, so two baselines of the same rung and complexity collide, and the report names cells rather than methods. 2) Its level defaults to L4 although mapping ends at L2, as nothing sets it from the task. 3) `Task` has no notion of a split. `datasets()` is one dictionary, so the train and test sets of stage 2 are simply a convention between the user's task and estimators (workstreams A and D).

(r15-names)=
## The scoping report's names, checked

In [](#tbl-r15-names), we set every name that the scoping report uses against the pinned code.

:::{table} Every name the scoping report uses, against the pinned code.
:label: tbl-r15-names

| Scoping report | At the pinned revisions |
|---|---|
| `BenchmarkContract`, `content_hash`, `to_yaml`, `from_yaml` | do not exist; `pipekit.hashing.stable_json` and `sha256_hex` hash any mapping today (stage 1) |
| task `Literal["discretization", …, "representation"]` | pipekit-evaluate's `LEVEL_NAMES` (L0 to L4) name the same carriers; representation has no level |
| `Metric.differentiable`, `stage_compatibility`, `variant` | do not exist; [Report 14](14_pipekit_evaluate.md#r14-ws-a)'s A1 proposes unit, lens, stage and differentiable tags on `Scorer` |
| `pipekit_train.splitters` (`SpatioTemporalBlockSplit`, `LeaveOnePlatformOut`, `CausalTemporalSplit`, `LeaveOneRegionOut`) | do not exist; `XarrayWindowDataset` and `SimulationDataset` split 80/10/10 in fixed tenths, the first with a guard band |
| `PersistenceBaseline`, `ClimatologyBaseline`, `LinearTrendBaseline`, `OptimalInterpolationBaseline` | do not exist as baselines; xrtoolz ships `fillnan_temporal`, `fillnan_climatology`, `CalculateClimatology` and `fillnan_rbf`, and no OI |
| `MultiTrackEvaluation` | does not exist; Report 14's A4 proposes several references per cell |
| `registry.register_contract`, `load_contract` | do not exist; `ModelRegistry.store` takes an `Operator`, and `_JsonArtifact` gives JSON `save` and `load` to the two artifacts |
| `BenchmarkSubmission.verify`, `EvaluationReport.was_produced_from` | do not exist; `BenchmarkReport` records no contract or model (stage 1) |
| `AccessGatedGeoCatalog` | does not exist; geocatalog has no access control |
| `pipekit-cycle.DACycle`, `SmootherCycle`, `Cycle` for analysis, reanalysis, forecast | exist, as named ([Report 10](10_pipekit_cycle.md)) |
| `xr_toolz.interpolate` for discretization and gap filling | exists as `xrtoolz.interpolate`: `bin_2d`, `points_to_grid`, `idw_to_grid`, the `fillnan_*` family |
| pyrox-gp, gaussx for Gaussian-process gap filling | not pinned in `references/`, so not checked |
:::

(r15-audit)=
## Current-state audit

Four of the nine symbol groups need action ([](#tbl-r15-audit)), and [](#fig-r15-readiness) shows what the page's sketches call. We read the code on 13 October 2026 at the revisions in the report's header, on `main` of each submodule.

:::{figure} ../images/reports/r15_readiness.svg
:label: fig-r15-readiness
:alt: A grid with one row per stage of the running example and per workstream, and one square per stack call in that row's sketches, coloured by its marker. Of 45 calls, 14 run today, 15 are proposed by this report's workstreams, 1 by another report and 15 are code the reader writes.

Of the page's 45 marked stack calls, 14 run today, 15 are proposed by this
report's workstreams, 1 by another report and 15 are code the reader writes
(how to read it: [conventions](../overview.md#sec-conventions)).
:::

:::{table} The benchmark surface, by symbol group.
:label: tbl-r15-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `stable_json`, `sha256_hex` | pipekit `hashing.py` | the registry, `hash_config` in pipekit-train | **Healthy.** Deterministic; hashes the stage-1 contract as is |
| `Benchmark`, `BenchmarkReport`, `CellResult` | pipekit-evaluate `benchmark/run.py` | stage 6 | **Split ownership.** Cells keyed by `(rung, complexity)`; the report records no contract, model or data |
| `Task`, `Estimator`, `Scorer`, `Estimate` | `benchmark/protocols.py` | stage 6 | **Split ownership.** No split, no variant, no estimator name |
| `LEVEL_NAMES`, `Cell` | `benchmark/cube.py` | `Benchmark(level="L4")` | **Healthy.** The vocabulary the contract's task should use; the level is never set from the task |
| `XarrayWindowDataset` split, `_split_origins`, `split_guard` | pipekit-train `xarray_window.py:113-133` | [Report 11](11_pipekit_train.md)'s window dataset | **Split ownership.** Temporal only, fixed 80/10/10; the guard counts origins and can empty a split (stage 2) |
| `SimulationDataset` split | pipekit-train `dataset.py:220-226`, `:295-299` | training on simulations | **Healthy.** Disjoint seed ranges per split, so no leakage between draws |
| `fillnan_temporal`, `fillnan_climatology`, `fillnan_rbf`, `idw_to_points`, `CalculateClimatology` | xrtoolz `interpolate/_src/gap_fill.py`, `knn.py`, `geo/operators.py` | stage 3 | **Healthy.** The baselines a gap-filling benchmark needs, minus OI |
| `crps_ensemble`, `spread_skill_ratio`, `rank_histogram`, `ensemble_coverage`, `rank_methods`, `rmse_skill_scores` | xrtoolz `metrics/_src/` | stage 4 | **Healthy.** Probabilistic scores and a leaderboard table; the variant is checked on array shape only |
| `ModelRegistry`, `LocalModelRegistry`, `_JsonArtifact` | pipekit-experiment `protocols.py:51`, `registry.py`, `artifacts.py` | [Report 12](12_pipekit_experiment.md) | **Split ownership.** Content-addressed storage for operators only; artifacts save as JSON but are not addressed by hash |
:::

Smaller findings:

- **`rank_methods` already builds a leaderboard.** xrtoolz's `metrics/_src/leaderboard.py` sorts a per-method score dataset into a table, and a report's `to_pandas` ([Report 14](14_pipekit_evaluate.md#r14-ws-a)'s A3) can feed it.
- **`rmse_skill_scores` fixes one convention.** It pins `ddof=0` to match an upstream OSSE report, which is exactly the kind of choice a contract has to record.
- **`fillnan_climatology` computes its climatology from its input.** On a test year this is legitimate (the inputs are observations of that year). However, a climatology forecast must use the training years only, which needs a separate fit (workstream C3).
- **The window dataset's empty split is silent.** `XarrayWindowDataset` keeps a zero-length split rather than raising, so a guard set from the physics can remove a split without notice (stage 2).

(r15-ws-a)=
## Workstream A · The contract

**Decision.** A benchmark is a frozen dataclass of plain data in `pipekit_evaluate.benchmark.contract`, hashed with pipekit's primitives, and its hash travels with every report it produces ([](#tbl-r15-move-a)).

:::{table} Workstream A's moves.
:label: tbl-r15-move-a

| Symbol | Destination | Reason |
|---|---|---|
| `BenchmarkContract(name, version, task, operation, input, output, variant, splits, references, data, scorers, baselines, submission, status)` | `pipekit_evaluate.benchmark.contract` | the scoping report's artifact; `task` is a level transition (`"L1->L2"`) and `operation` what it does (`"gap filling"`); `data` holds the content hashes of the inputs and references (or, for an OSSE, the generator and its seed); each baseline lists every setting that changes its score and its code revision; `status` is `"pre-registered"` or `"exploratory"` |
| `content_hash`, `to_yaml`, `from_yaml` | the same module, over `pipekit.hashing` | stage 1's hash, with YAML for people and JSON for the hash |
| `contract_hash` on `BenchmarkReport` (and [Report 14](14_pipekit_evaluate.md#r14-ws-a)'s `EvaluationReport`), and `model_hash` on each of its results | `benchmark/run.py`, `report.py` | stage 1: a report names nothing it scored; one run scores several entries (stage 6 runs three), so the model identity belongs to the row, not the report |
| `Benchmark.from_contract(contract, estimators)` | `benchmark/run.py` | sets the level, the scorers and the references from the contract (stage 6) |
| templates per task | `pipekit_evaluate.benchmark.templates` | the scoping report's mitigation for long contracts |
:::

1. **Make the contract data.** Every field is a string, a number, a list or a mapping, so the contract round-trips through YAML and hashes stably. We give splits, scorers and baselines by `_target_` and arguments, as Hydra would build them, so the contract never holds an object.
2. **Hash the content, not the file.** `content_hash` is `sha256_hex(stable_json(asdict(self)))`, so formatting and key order never change it.
3. **Record the hash everywhere.** The runner copies the contract hash onto the report, and a report without one is exploratory by construction.

```{code-block} python
:caption: Stage 1's contract, before and after workstream A.
:linenos:

# Shapes: as stage 1
# Today:
h = sha256_hex(stable_json(contract))                   # a dict        today
# After:
c = pe.BenchmarkContract(name="ssh-mapping-synthetic", version="1.0.0",
                         task="L1->L2", variant="deterministic",
                         splits={"_target_": "geocatalog.splits.split_by",
                                 "column": "producer", "test": "s1"},  # (B2)
                         status="pre-registered")        # proposed (A1)
c.content_hash()                                         # proposed (A1)
rep = pe.Benchmark.from_contract(c, estimators=ests).run()  # proposed (A2)
rep.contract_hash == c.content_hash()                    # proposed (A2)
```

(r15-ws-b)=
## Workstream B · Splits as contract terms

**Decision.** A split is a function of catalog rows in geocatalog, or of windows in `geopatcher.integrations.train` (pipekit-train until [Report 16's D2](16_geo_data_stack.md#r16-ws-d) moves the window dataset), which the contract configures and which refuses what it cannot honour. The contract also says which gap the split reproduces ([](#tbl-r15-move-b)).

:::{table} Workstream B's moves.
:label: tbl-r15-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `split_blocks(catalog, *, km, days, test_fraction, seed, min_scale=None, crossing="drop")` | `geocatalog.splits` (new) | stage 2's blocks; rows carry footprints and intervals, so a row is assigned only when it lies wholly inside one block (`crossing="drop"`, or `"group"` to merge the blocks it touches); raises when a block is smaller than the declared scale or the domain cannot hold one |
| `split_by(catalog, column, *, test)` | `geocatalog.splits` | stage 2's withheld satellite, over a platform or producer column ([Report 13](13_state_catalog.md#r13-ws-a)'s `producer`) |
| `split_causal(catalog, *, train_end, val_end)` | `geocatalog.splits` | the forecast split: every test row after every training row |
| `split_regions(catalog, regions)` | `geocatalog.splits` | the scoping report's leave-one-region-out, as footprint intersection |
| `fractions` and `guard` as a `pd.Timedelta` | `XarrayWindowDataset`, `_split_origins`: in `geopatcher.integrations.train` after Report 16's D2, in pipekit-train before it | stage 2: the guard counts origins, and a long one empties the validation split silently |
| `gap` field on the contract's split | `BenchmarkContract.splits` | stage 2's correction: the split states the deployment gap it reproduces |
:::

### B1 · Blocks in space and time

A block split tiles space and time into blocks and holds out whole blocks. A row whose footprint or time interval crosses a block boundary would carry observations from one block into the other, so by default such rows go to neither side (or, as an option, they are grouped with every block they touch). The split refuses a block smaller than the declared correlation scale when the contract asks for generalization, and it refuses a block that does not fit the catalog's domain.

### B2 · Platform, causal and region splits

The platform split is the one that stage 2 shows a mapping benchmark needs, which is to hold out every row of one producer. The causal split is the forecast's. Both are row filters over columns that geocatalog already has, or that Report 13's state profile adds.

### B3 · The window guard in time

`XarrayWindowDataset` (in `geopatcher.integrations.train` once [Report 16](16_geo_data_stack.md#r16-ws-d)'s D2 moves it there) takes its guard as a duration and converts it to origins with the source's step. It raises when any split ends up empty, and the error names the guard and the split's length.

```{code-block} python
:caption: Stage 2's splits, before and after workstream B.
:linenos:

# Shapes: 35,040 obs rows; the window dataset as stage 2
# Today:
tr, te = sat != "s1", sat == "s1"                                     # user
pt.XarrayWindowDataset(..., split_guard=60)    # → val has 0 windows   today
# After:
train, test = gc.splits.split_by(obs_cat, "producer", test="s1")  # proposed (B2)
gc.splits.split_blocks(obs_cat, km=200, days=60, test_fraction=0.2, seed=0,
                       min_scale=(100, 30))                       # proposed (B1)
from geopatcher.integrations.train import XarrayWindowDataset  # proposed (R16 D2)
XarrayWindowDataset(..., guard=pd.Timedelta("60D"))
# → ValueError: guard 60 days leaves split 'val' (105 days) empty  proposed (B3)
```

(r15-ws-c)=
## Workstream C · Baselines with the carriers

**Decision.** Baselines are ordinary operators in the carrier library, beside the gap-fillers that xrtoolz ships, and the contract names them and pins every argument ([](#tbl-r15-move-c)).

:::{table} Workstream C's moves.
:label: tbl-r15-move-c

| Symbol | Destination | Reason |
|---|---|---|
| `optimal_interpolation(obs, grid, *, kernel, length, time_scale, noise, window)` and `OptimalInterpolation` | `xrtoolz.interpolate` | stage 3's best baseline, absent from the stack; the kernel is valid on the grid (images summed on periodic domains, or a compactly supported kernel), checked on the prior before the noise is added by its smallest eigenvalue against a scale-aware tolerance (λ_min ≥ −10⁻¹⁰ σ_k²); a strict Cholesky would reject a valid kernel, since the prior is only semidefinite (colocated observations make it singular) |
| `persistence(field, *, lead, reference="last_available", gap="nan")` | `xrtoolz.geo` (forecast baselines) | stage 3: persistence is carry-forward with an explicit reference time; nearest and linear interpolation in time are interpolation baselines (`fillnan_temporal`), not persistence, and the improvement over each differs by up to 14 points |
| `ClimatologyForecast(train_period, freq)` | `xrtoolz.geo`, over `CalculateClimatology` | a climatology fitted on the training years only, unlike `fillnan_climatology` |
| `linear_trend(field, *, n_recent)` | `xrtoolz.geo` | the scoping report's trend baseline |
:::

```{code-block} python
:caption: Stage 3's OI baseline, before and after workstream C.
:linenos:

# Shapes: G (1,095, 32, 32); output (364, 32, 32)
# Today:
mean, var = oi(G, length_km=100, time_days=30)                        # user
# After:
# x̂ = μ + Σ_xy Σ_y⁻¹ (y − μ); σ̂_d² = σ_k² − [Σ_xy Σ_y⁻¹ Σ_xyᵀ]_dd
mean, var = xi.optimal_interpolation(G, grid, kernel="gaussian", length=100e3,
                                     time_scale="30D", noise=0.02,
                                     window="15D")              # proposed (C1)
```

(r15-ws-d)=
## Workstream D · Variants and tracks in the runner

**Decision.** The runner checks an entry against the contract before scoring it, scores every declared reference, and names entries ([](#tbl-r15-move-d)).

:::{table} Workstream D's moves.
:label: tbl-r15-move-d

| Symbol | Destination | Reason |
|---|---|---|
| `variant` on `Scorer` and on the entry's `Estimate` (`"deterministic"`, `"probabilistic"`) | `benchmark/protocols.py`, beside [Report 14](14_pipekit_evaluate.md#r14-ws-a)'s A1 tags | stage 4: the report should say which rows are point forecasts |
| point forecasts under CRPS | `xrtoolz.metrics.crps_ensemble` and the runner | a point forecast is scored as a degenerate distribution (its CRPS is its MAE) without duplicating it; stage 4's duplication goes |
| `references: Mapping[str, ReferenceSource \| str]` | `Benchmark`, building on Report 14's A4 | stage 5's three tracks in one run; a track with no data reports "unavailable", as `CellResult.note` already does for a missing oracle |
| per-track scoring from the contract's `scoring` (carrier, matchup rule, reduction dims) | `Benchmark.from_contract` | stage 5: a gridded track is scored over `(time, y, x)`, the s1 track only after sampling the map at each observation and reducing over `obs`; without the rule the runner cannot reproduce the observation scores |
| `MultiTrackEvaluation` | not added | the multi-reference runner is the same object; a second class would duplicate it |
| `name` on `Estimator`; results keyed by name | `benchmark/protocols.py`, `run.py` | stage 6: two baselines in one cell collide |
| `datasets(split)` on `Task` | `benchmark/protocols.py` | stage 6: a task has no train and test sets today |
:::

```{code-block} python
:caption: Stages 4 to 6, after workstream D.
:linenos:

# Shapes: as stages 4 and 5
rep = pe.Benchmark.from_contract(
    c, estimators=[oi_100, oi_50, learned_reanalysis],          # named entries
    references={"reanalysis": rean_cat, "observations": s1_cat,
                "analysis": None},          # → "unavailable"   proposed (D2)
).run()                                                          # proposed (D3)
pe.Benchmark.from_contract(c_probabilistic, estimators=[oi_100_map]).run()
# → scored as a point forecast (CRPS = MAE), row tagged "deterministic"  proposed (D1)
```

(r15-ws-e)=
## Workstream E · Registry and submissions

**Decision.** pipekit-experiment stores contracts, reports and submissions by content hash, and a submission verifies by recomputing hashes rather than by trust ([](#tbl-r15-move-e)).

:::{table} Workstream E's moves.
:label: tbl-r15-move-e

| Symbol | Destination | Reason |
|---|---|---|
| `ArtifactStore.put(obj) -> hash`, `get(hash)` for JSON artifacts | `pipekit_experiment` (beside `ModelRegistry`, same layout) | `store` takes only operators; contracts and reports are data |
| `BenchmarkSubmission(contract_hash, model_hash, report_hash)` with `verify(store)` | `pipekit_evaluate.benchmark.submission` | one submission is one entry: checks that the report's `contract_hash` matches, that a result row carries the submitted `model_hash`, that every row carries a model hash, and that the contract is pre-registered |
| `max_test_evaluations` | `BenchmarkContract.submission` | recorded and counted per model hash; enforcing it needs a service, which stays with the project |
:::

```{code-block} python
:caption: Publishing a contract and verifying a submission, after workstream E.
:linenos:

# Shapes: three hashes of 64 hex characters
store = px.ArtifactStore("registry")                             # proposed (E1)
h = store.put(c)                       # = c.content_hash()       proposed (E1)
sub = pe.BenchmarkSubmission(contract_hash=h, model_hash=m_hash,
                             report_hash=store.put(rep))         # proposed (E2)
sub.verify(store)                      # → True                   proposed (E2)
```

(r15-gates)=
## Testing gates

Each workstream merges only when its gates ([](#tbl-r15-gates)) pass on the running example.

:::{table} Gates, by workstream.
:label: tbl-r15-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Contract hash | A | the stage-1 contract as a `BenchmarkContract` hashes to the same value after a YAML round trip and in any key order; changing any one field (the withheld platform, a block size, a baseline argument) changes the hash |
| A2 Report hash | A | `Benchmark.from_contract(c, …).run().contract_hash == c.content_hash()`, the report's level is the contract's (L2 for stage 6, not L4), and each of its three rows carries its entry's model hash |
| B1 Blocks | B | on stage 2's observations, `split_blocks(km=200, days=60, seed=0)` reproduces the notebook's test set and its RMSE (4.18 cm against the truth) to 1e-4, assigning no row that crosses a block boundary; `km=50` with `min_scale=(100, 30)` raises `ValueError` |
| B2 Platform and causal | B | `split_by("producer", test="s1")` reproduces stage 2's 2.70 cm, and every test time of `split_causal` is later than every training time |
| B3 Window guard | B | `XarrayWindowDataset(guard=pd.Timedelta("60D"))`, imported from wherever [Report 16's D2](16_geo_data_stack.md#r16-ws-d) has left it, on stage 2's series raises `ValueError` naming the empty split; with `guard="2D"` its splits are 871, 105 and 108 windows |
| C1 OI | C | `optimal_interpolation` reproduces stage 3's 1.81 and 2.62 cm to 1e-4 m; with ℓ_s = 150 km on the 800 km periodic grid its prior covariance, before the noise term, has smallest eigenvalue λ_min ≥ −10⁻¹⁰ σ_k² (semidefinite to rounding), where the minimum-image Gaussian gives −1.7 × 10⁻³ m² |
| C2 Persistence | C | `persistence` reproduces stage 3's carry-forward 7.98 cm and uses no observation after its reference time; nearest and linear interpolation in time (5.59 and 4.77 cm) are separate interpolation baselines, and each configuration hashes differently |
| C3 Climatology | C | `ClimatologyForecast(train_period=("2023", "2024"))` uses no 2025 value: perturbing every 2025 input changes its output by 0 |
| D1 Variant | D | on a probabilistic contract, stage 4's map is scored without duplication at its MAE, 1.29 cm, its row is tagged deterministic, and the 20-member ensemble scores 0.96 cm CRPS |
| D2 Tracks | D | one run, driven by the contract's `scoring`, reproduces stage 5's nine scores to 1e-4 m (the s1 track through its matchup, over `obs`), and a track with no reference appears with the note "unavailable" |
| D3 Names | D | stage 6's four baselines, two at the same cell, run without error and the report is keyed by their names |
| E1 Store | E | `store.put(c) == c.content_hash()` and `store.get(h)` returns a contract equal to `c` |
| E2 Verify | E | each of the stage-6 report's three entries verifies with its own model hash, and a submission fails after any change to the contract, its model hash or the report, or when a row has no model hash |
:::

(r15-sequencing)=
## Sequencing and risks

The contract and the splits come first, because every later check needs something to check against ([](#fig-r15-roadmap)), and the hosted service never comes into the framework.

:::{figure} ../images/reports/r15_roadmap.svg
:label: fig-r15-roadmap
:alt: Phase 1: A1 and A2, the contract and its hash on every report, beside B1 to B3, the block, platform and causal splits and the guard in days; then gates A1, A2, B1, B2, B3. Phase 2: C1 to C3, the OI, persistence and climatology baselines, beside D1 to D3, the variant check, tracks and named entries; then gates C1 to D3. Phase 3: E1 and E2, the contract and submission store and verification; then gates E1 and E2.

The contract and its splits first, then the baselines and runner checks, then
the store.
:::

- **The contract outgrows its readers.** A full contract with carriers, splits, scorers and baselines is long for a person to write. Fallback: templates per task (A), where a contract can reference a template by hash and override fields.
- **Splits over rows need rows.** A dataset that is one Zarr store has one catalog row, so a block split cannot see inside it. Fallback: split the windows inside the dataset (B3) with the same block rule, or index the store per time step with [Report 13](13_state_catalog.md#r13-ws-a)'s per-step option.
- **A split that matches use needs knowledge of use.** Stage 2 shows that the right split depends on the gap the method will face, which is a scientific judgement. Fallback: the contract's `gap` field is required and free text, so at least the choice is stated and reviewable.
- **OI does not scale as written.** One dense solve per day is fine for 320 to 704 observations, but not for a global product. Fallback: a localized OI (a radius per grid cell) behind the same function, or the Gaussian-process tools once pyrox-gp or gaussx is pinned.
- **Blind tests need a service.** Without one, `max_test_evaluations` is recorded but not enforced. Fallback: the contract marks its test set `blind`, and the projects that run a service enforce it there.
- **Two report types.** [Report 14](14_pipekit_evaluate.md#r14-ws-a)'s `EvaluationReport` and the ladder's `BenchmarkReport` could both grow hashes. Fallback: the hash fields go on the one report that Report 14's A3 leaves standing.

(r15-open-questions)=
## Open questions

1. **Does a contract pin code or names?** A baseline pinned as `xrtoolz.interpolate.optimal_interpolation` with arguments still changes if xrtoolz changes. Should a contract also pin the library versions, as `TrainingArtifact.deps_lock` does?
2. **Who sets the gap?** Stage 2's withheld satellite is right for mapping with today's constellation. Should a contract carry several splits (a withheld satellite and a withheld region) and report both?
3. **Pre-registered means what?** The hash proves that the contract did not change, but it does not prove that it was published before the results. Is a timestamped record outside the repository (a DOI, a tagged release) required for the `pre-registered` status?
4. **Representation tasks.** They are not a level transition. Do they need a contract kind of their own, or are they scored only through a downstream task?
5. **Where does a reanalysis track come from?** [Report 13](13_state_catalog.md#r13-ws-a)'s state profile indexes reanalyses as catalog rows. Should a contract's references be catalog hashes rather than URIs?

(r15-todos)=
## Upcoming work

We list the follow-ups that fall outside the workstreams in [](#tbl-r15-todos).

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r15-todos

| Item | Where | Why |
|---|---|---|
| Review the gallery's eighteen draft contracts with domain experts, then pre-register them | [benchmarks gallery](../benchmarks_gallery.md) | the drafts hash today, but their metrics, baselines and splits need review before their hashes mean anything |
| Raise on an empty split | `XarrayWindowDataset` (pipekit-train today, `geopatcher.integrations.train` after [Report 16's D2](16_geo_data_stack.md#r16-ws-d)) | stage 2: a long guard empties a split silently; independent of B3's units |
| Index xrtoolz submodules in the capability index | `scripts/capabilities.py` (`UPSTREAM`) | the index lists only top-level `xrtoolz`, so `fillnan_*`, `crps_ensemble` and `rank_methods` are not searchable ([Report 14](14_pipekit_evaluate.md#r14-todos) found the same) |
| Pin pyrox-gp or gaussx | `references/`, `UPSTREAM` | the scoping report maps Gaussian-process gap filling to them; neither is checked |
| Write the representation contract | [open question 4](#r15-open-questions) | no level transition to name it by |
:::

(r15-references)=
## References

### Sources

- The scoping report, "Report 15 — Benchmarking as first-class infrastructure" (supplied by the author).
- The pinned code in `references/`: pipekit (`hashing.py`), pipekit-evaluate (`benchmark/`), pipekit-train (`xarray_window.py`, `dataset.py`), pipekit-experiment (`registry.py`, `artifacts.py`), xrtoolz (`interpolate/`, `metrics/`, `geo/`) and geocatalog.
- OceanBench, the SSH mapping benchmark that the running example imitates [@johnson2023oceanbench], and WeatherBench 2 [@rasp2024weatherbench].
- Block cross-validation for spatially and temporally structured data [@roberts2017], the standard reference for the leakage argument of stage 2.
- The running example's notebook, described in [](#r15-running).
