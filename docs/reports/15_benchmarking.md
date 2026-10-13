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

Two teams that report "a 10 % improvement" on the same ocean dataset are
often not comparing the same thing: they split the data differently, use
different baselines, score against different references, or keep iterating
on the test set. The scoping report calls these **contract failures** and
proposes that a benchmark be a frozen, content-addressed **contract** (task,
carriers, splits, references, metrics, baselines and submission rules), with
five pieces of infrastructure to enforce it. This page runs one sea surface
height (SSH) mapping benchmark through the pinned code. The contract's hash
and the runner exist: pipekit hashes any configuration, and
pipekit-evaluate ships a `Benchmark` runner and the L0–L4 levels the
scoping report's task list restates. The data and the baselines are in reach:
xrtoolz ships the gap-fillers, CRPS and spread-skill. What is missing is the
contract itself, splits that know about space and platforms, and a report that
names the contract it scored. The run also corrects the scoping report's
split rule: **the held-out set must reproduce the gap the method will face in
use**, which is a withheld altimeter for mapping and a block larger than the
correlation scale only for generalization to unseen regions or periods. On the
same method, the choice moves the error from 1.68 to 4.26 cm. Every number
below is from a real run.

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
The scoping report's **central claim stands**: a benchmark is a contract and
evaluation is the machinery that runs it, and stages 1 to 5 reproduce each
contract failure it lists. Three parts **do not**. Its **task list** restates
the L0–L4 levels pipekit-evaluate already ships, as transitions between
them. Its **split rule** (block size above the correlation scale) is right
for generalization to unseen regions and periods but wrong for mapping, where
it overstates the error in use by 56 % (stage 2). And
its **package plan** puts splitters and baselines in pipekit-train and
pipekit-evaluate, while the stack's rule ([Report 14](14_pipekit_evaluate.md#r14-ownership))
is that computations live with their carriers: splits over catalog rows
belong to geocatalog, and gap-filling baselines sit beside xrtoolz's
gap-fillers. A **Differs** box marks each place.
:::

(r15-summary)=
## Summary

**Decision.** A benchmark is a `BenchmarkContract`: frozen data in
pipekit-evaluate, hashed with pipekit's `stable_json` and `sha256_hex`, and
recorded on every report it produces. Each term of the contract is
implemented where its data lives: splits of catalog rows in geocatalog and of
training windows in pipekit-train, baselines and probabilistic scores in
xrtoolz, and the content-addressed store for contracts and submissions in
pipekit-experiment. No new package, and no hosted scoring service.

- **A. The contract.** A frozen `BenchmarkContract` with a task given as a
  level transition (L1 → L2 for mapping), carriers, splits, references,
  scorers, baselines and a status (pre-registered or exploratory), hashed
  and round-tripped through YAML (A1); the hash recorded on every report
  (A2).
- **B. Splits as contract terms.** Block, platform and causal splits of
  catalog rows in geocatalog, each refusing a configuration it cannot honour
  (B1, B2); the training window's guard band in time units, refusing an empty
  split (B3). The contract states the gap the split reproduces.
- **C. Baselines with the carriers.** Space-time optimal interpolation with a
  valid covariance in `xrtoolz.interpolate` (C1); persistence with an explicit
  reference time and gap rule (C2); climatology fitted on the training period
  only (C3). The contract pins each configuration.
- **D. Variants and tracks in the runner.** Entries and scorers declare a
  variant, which the report records, and point forecasts are scored under
  CRPS as the degenerate distributions they are (D1); one run scores against
  several references and reports a missing one as unavailable (D2); entries
  are named, not keyed by their ladder cell (D3).
- **E. Registry and submissions.** Contracts and reports stored
  content-addressed in pipekit-experiment (E1); a submission that pins the
  contract, the model and the report hashes, and verifies against them (E2).

**End state.** A curator writes a contract, publishes its hash, and anyone
can run it: the splits refuse configurations that would leak, the baselines
are the same code for every entry, the runner records each entry's variant
and scores every track, and the report names the contract, the model
and the data. Running the leaderboard, a blind test service and its queue
stays with the projects that host one.

(r15-framework)=
## The framework, as the scoping report put it

This section restates the scoping report's framework; the rest of the page
keeps it except where a box says otherwise.

**Benchmark versus evaluation.** An evaluation scores one model, once, with
whatever data and choices are at hand. A benchmark is a shared contract that
fixes everything a comparison depends on before anyone submits.

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

**Tasks as carrier transformations.** Each task maps an input carrier to an
output carrier, so a model is anything that performs the map: discretization
(points to a sparse grid), gap filling (to a dense grid), analysis (to a
state), reanalysis (to a smoothed state trajectory), forecast (a state to a
future state) and representation (a state to a latent and back). Every task
comes in a **deterministic** and a **probabilistic** variant, scored by
different metrics.

**Three tracks.** The same predictions are scored against a reanalysis, an
analysis and the raw observations. A model that wins on the reanalysis and
loses on observations has learned the assimilation product, not the ocean.

**Five pieces of infrastructure.** Leakage-aware splitters, the contract
artifact, a baseline registry, pre-registration by content address, and a
blind test set behind catalog access control, plus multi-track evaluation as
a built-in pattern.

:::{warning} Differs from the plan
The six tasks are the transitions between the levels pipekit-evaluate
already names (`benchmark/cube.py`, `LEVEL_NAMES`): L0 unstructured
observations, L1 structured observations, L2 gap-filled observations, L3
analysis, L4 reanalysis or forecast. Discretization is L0 → L1, gap filling
L1 → L2, analysis L2 → L3, reanalysis and forecast end at L4. The contract
names its task by that transition instead of a second `Literal` list, so a
benchmark cell and a contract use the same vocabulary. Representation is the
exception: it is not a level transition, and nothing in pipekit-train ships
an encoder or a decoder to map it to.
:::

(r15-ownership)=
## Ownership rules

The contract is data in pipekit-evaluate; every term it fixes is implemented
by the library that owns the data the term constrains.

:::{figure} ../images/reports/r15_ownership.svg
:label: fig-r15-ownership
:alt: Benchmark projects write contracts and read reports through pipekit-evaluate, which ships Benchmark, ReferenceRule, Scorer, Task and BenchmarkReport and would add BenchmarkContract, variant and track checks and a contract hash on reports. Below it, geocatalog holds block, platform and causal splits of rows, pipekit-train the window splits and their guard, xrtoolz the baselines and CRPS, and pipekit-experiment the content-addressed store. All four use pipekit's hashing.

The contract lives in pipekit-evaluate; its splits, baselines and storage
live with their data.
:::

1. **pipekit-evaluate** owns the contract, the runner, the scorer tags
   (variant beside Report 14's unit, lens and stage) and the report. It
   imports geocatalog, xrtoolz and pipekit-experiment only through optional
   extras, as it imports pipekit-cycle today.
2. **geocatalog** owns splits of catalog rows: blocks in space and time, a
   held-out platform, a causal cut. A row already carries the footprint, the
   time interval and, with Report 13's profile, the producer, which is all a
   split needs. It never imports pipekit-evaluate.
3. **pipekit-train** owns splits of training windows inside a dataset, and
   reads the contract's split as plain arguments (fractions, a guard in time
   units). It never parses a contract.
4. **xrtoolz** owns the baselines for xarray carriers, next to the
   gap-fillers it ships (`fillnan_temporal`, `fillnan_climatology`,
   `fillnan_rbf`, `idw_to_points`), and the probabilistic scores. Baselines
   for GeoTensor carriers go to geotoolz the same way.
5. **pipekit-experiment** owns persistence: contracts, reports and
   submissions stored by content hash beside the model registry. It never
   interprets what it stores.
6. **Benchmark projects** (OceanBench, WeatherBench, the gallery's entries)
   own curation, reference data, the leaderboard and any blind scoring
   service. The framework gives them the contract and the checks.

(r15-running)=
## The running example: mapping sea surface height from three altimeters

**The problem.** Map SSH anomalies every day from along-track altimetry, and
decide how to rank mapping methods fairly. This is the task OceanBench's
first edition poses on real data [@johnson2023oceanbench]; here the field is
synthetic, so the truth is known and every score can be checked against
theory.

**The data.** η(t, y, x) is an SSH anomaly in metres on a doubly periodic
800 km square, 32 × 32 cells of Δ = 25 km, daily from 1 January 2023 to
30 December 2025 (T = 1,095 days, so the test year 2025 has 364). It is a seasonal cycle of amplitude a = 4 cm plus a
mesoscale anomaly with standard deviation σ = 8 cm and a Gaussian covariance

$$
C(r, \Delta t) = \sigma^2 \exp\left(-\frac{r^2}{2 L^2} - \frac{\Delta t^2}{2 \tau^2}\right) ,
$$ (r15-eq-cov)

with r the distance between two points, Δt the time between them,
L = 100 km and τ = 30 days, the mesoscale values the scoping report gives for
SSH. Three altimeters s1, s2 and s3 each fly one diagonal ground track across
the domain every third day and measure η with noise σₒ = 2 cm: 35,040
observations, 11,680 per satellite. The task is gap filling, L1 → L2: from
the observations, produce η on the full grid.

**What exists.** The contract in stage 1 is a dictionary hashed with
pipekit's own primitives. The mapping methods are written in the notebook
where the stack has none: an inverse-distance mapper in space and time
(xrtoolz's `idw_to_points` weighs in two dimensions only) and an optimal
interpolation (proposed as C1). Lines marked `user` are code the workstreams
move upstream.

(r15-stage-1)=
### Stage 1 · Write the contract and hash it (pipekit → pipekit-evaluate)

**TL;DR.** Write down everything the comparison depends on and give it a
fingerprint that changes when anything in it changes.

**Problem.** A contract c is a nested mapping of the task, carriers, splits,
references, metrics and baselines. Its address is

$$
h(c) = \mathrm{SHA256}\left(\mathrm{json}_{\mathrm{sorted}}(c)\right) ,
$$ (r15-eq-hash)

the hash of its JSON encoding with sorted keys, so two contracts with the same
content get the same h whatever the order they were written in, and any edit
gives a new h.

```{code-block} python
:caption: The stage-1 contract as data, from a real run.
:linenos:

# Shapes: input the s2, s3 obs binned to the grid (1,095, 32, 32), NaN off-track;
#         output (364, 32, 32) maps of 2025
from pipekit.hashing import sha256_hex, stable_json
contract = {
    "task": "L1->L2", "operation": "gap filling", "variant": "deterministic",
    "input": {"carrier": "sparse grid", "sources": ["s2", "s3"], "units": "m"},
    "output": {"carrier": "grid", "shape": [364, 32, 32], "units": "m"},
    "splits": {"kind": "leave_one_platform_out", "platform": "s1",
               "period": ["2025-01-01", "2025-12-30"]},
    "references": {"truth": "osse", "reanalysis": "smoothed-50km",
                   "observations": "s1"},
    "data": {"generator": GEN},     # grid, L, τ, σ, noise, orbits, seed
    "metrics": [...], "baselines": [...],                  # arguments sketched
}
# h = SHA-256 of the key-sorted JSON
h = sha256_hex(stable_json(contract))           # → "9677c757a296…"     today
# reverse key order → same h; withhold s2 → "aa7284f9dc8b…"; seed 1 → new h
[f.name for f in dataclasses.fields(pe.BenchmarkReport)]
# → ["task", "domain", "axis", "results"]: no contract, no model   today
```

The fingerprint costs nothing: pipekit already uses `stable_json` and
`sha256_hex` for the registry and the dataset hashes, and the whole contract
is about 1 KB of JSON. It must identify the data, not only name it: labels
such as `"osse"` and `"s1"` say nothing about which arrays they stand for,
so this contract hashes the generator's full configuration and seed, and a
contract over real data would carry the catalog's content hash for each
input and reference. What is missing is the place to put it. pipekit-evaluate's
`BenchmarkReport` records the task name, the domain, the axis and the cell
results, so a report cannot say which contract, which model or which data it
scored, and a submission cannot be checked against anything (workstreams A
and E).

:::{warning} Differs from the plan
The scoping report's sketch also carried `block_size_spatial_km` and
`block_size_temporal_days` beside `splitter_config`. They live only in the
split configuration here, so the two cannot disagree; the overview's
[contract example](../overview_examples.md#ex-benchmark) had already made that
change. The report expected about 50 KB of YAML; a full contract for this
benchmark is under 1 KB.
:::

(r15-stage-2)=
### Stage 2 · One method, six splits (numpy and scipy, then pipekit-train)

**TL;DR.** Score one mapping method under six ways of holding data out, and
see which of them predicts the error the method will make in use.

**Problem.** Let 𝒟 be the set of observations (tᵢ, yᵢ, xᵢ, ηᵢ). A split
divides 𝒟 into a training set 𝒟_train and a test set 𝒟_test; the method sees
𝒟_train and predicts η̂ at each test point. The score is

$$
\mathrm{RMSE} = \left(\frac{1}{|\mathcal{D}_{\mathrm{test}}|} \sum_{i \in \mathcal{D}_{\mathrm{test}}} (\hat{\eta}_i - \eta_i)^2\right)^{1/2} ,
$$ (r15-eq-rmse)

against the noisy observation ηᵢ and against the true field. The method is
inverse-distance weighting of the k = 10 nearest training observations in the
scaled coordinates (x/L, y/L, t/τ).

```{code-block} python
:caption: The same method under six splits, from a real run (RMSE in cm, against the observation and the truth).
:linenos:

# Shapes: 35,040 obs; scaled coordinates (n, 3) = (x / L, y / L, t / τ)
# ŷ = Σ wᵢ yᵢ / Σ wᵢ,  wᵢ = 1 / dᵢ over the 10 nearest      user
splits = {
    "random 80/20": random_split(0.2),                  # 2.61 / 1.68   user
    "withhold s1": platform_split("s1"),                # 3.37 / 2.73   user
    "withhold s2": platform_split("s2"),                # 2.75 / 1.91   user
    "blocks 50 km × 7 d": blocks(50, 7),                # 2.62 / 1.75   user
    "blocks 200 km × 60 d": blocks(200, 60),            # 4.70 / 4.26   user
    "last 20 % of 2025": causal_split(0.8),             # 8.98 / 8.78   user
}
```

:::{table} The six splits of stage 2: what each reproduces, and the method's error under it.
:label: tbl-r15-splits

| Split | Reproduces | RMSE vs observations (cm) | RMSE vs truth (cm) |
|---|---|---|---|
| random 80/20 | nothing: test points sit beside training points on the same track | 2.61 | 1.68 |
| withhold s1 | mapping where one altimeter flew, from the other two | 3.37 | 2.73 |
| withhold s2 | the same, for s2's orbit | 2.75 | 1.91 |
| blocks 50 km × 7 d | gaps smaller than L and τ | 2.62 | 1.75 |
| blocks 200 km × 60 d | a region and season with no data at all | 4.70 | 4.26 |
| last 20 % of 2025 | a forecast of the last 73 days of 2025, with no data | 8.98 | 8.78 |
:::

The splits disagree by a factor of five, and no single one is "honest". A
mapping product is used where an altimeter did not fly while the others did,
so the withheld-satellite rows predict its error in use: 2.7 cm against the
truth for s1. The random split is optimistic by 38 % against that, because
each test point has a training neighbour 25 km away on the same track and the
same day. The 200 km × 60 day blocks the scoping report prescribes for SSH
are pessimistic by 56 %, because they ask the method to fill a region and a
season with no data, a task it is not used for. And which satellite is
withheld matters (2.73 against 1.91 cm), because s1's orbit leaves wider gaps
between the other two. A fourth satellite or a different orbit would change
every row: the split is a term of the contract, not a team's choice.

```{code-block} python
:caption: The training loader's own split, from a real run.
:linenos:

# Shapes: one cell's daily series, T = 1,095; input stencil −2 … 0 d, target +1 d
ds = pt.XarrayWindowDataset(source={"input": cell, "target": cell},
                            stencils=stencils, store_version="v1")    # today
ds._split_guard, [len(ds.with_split(s)) for s in ("train", "val", "test")]
# → 2, [871, 105, 108]: contiguous 80/10/10, a guard of 2 origins
# 2 origins dropped each side: targets 5 d apart, ρ(5 d) = exp(−25 / 1,800) = 0.986
pt.XarrayWindowDataset(..., split_guard=60)       # 60 d = 2τ          today
# → 60, [813, 0, 50]: the validation split is empty, silently
```

pipekit-train's window dataset already splits contiguously in time, which is
the right shape for a forecast. Its guard band is the stencil's reach (2
origins, dropped on each side of a boundary), so the last training target and
the first validation target are five days apart and correlated at 0.986. A guard of two correlation times
leaves the 105-day validation split, which has a boundary on each side,
empty, and the dataset returns zero windows instead of refusing (workstream
B3).

:::{warning} Differs from the plan
The scoping report's rule is that "the block size for splitting must exceed
the integral spatial/temporal scale of the variable". That holds for
generalization to unseen regions or periods, and for forecasts (the causal
split). For mapping it overstates the error by 56 % here, because the
deployment gaps are the spaces between tracks, not whole regions. The
corrected rule: **the held-out set reproduces the gap the method will face in
use**, and the contract states which gap that is. Blocks above the
correlation scale are one answer to that question, not the rule.
:::

(r15-stage-3)=
### Stage 3 · The baselines a contract must pin (xrtoolz)

**TL;DR.** Score the simple methods every paper compares against, and see how
much "the baseline" depends on how it was written.

**Problem.** The input is the 2025 maps gridded from s2 and s3, with no value
where neither flew; s1 is withheld. A baseline fills every empty cell. Optimal
interpolation (OI) is the best linear estimate under the covariance
{eq}`r15-eq-cov`: with y the M observations in a window of ±15 days, C_oo
their M × M covariance plus σₒ² I, C_go the N × M covariance between the N
grid cells and the observations, μ the window mean, and the spatial factor of
{eq}`r15-eq-cov` summed over the domain's periodic images so that it stays a
valid covariance,

$$
\begin{aligned}
\hat{\eta} &= \mu + C_{go} C_{oo}^{-1} (y - \mu) , \\
\hat{\sigma}^2 &= \sigma^2 - \mathrm{diag}\left(C_{go} C_{oo}^{-1} C_{og}\right) .
\end{aligned}
$$ (r15-eq-oi)

```{code-block} python
:caption: Six baselines on the 2025 maps, from a real run (RMSE in cm against the truth and against the withheld s1).
:linenos:

# Shapes: G (T = 1,095, 32, 32) with NaN off-track; scored on the 364 days of 2025
from xrtoolz.interpolate import fillnan_climatology, fillnan_temporal
G.ffill("time")      # pandas; persistence (carry fwd) 7.96 / 7.41  user
fillnan_temporal(G, method="nearest")   # nearest in time    5.76 / 5.80  today
fillnan_temporal(G, method="linear")    # linear in time     4.97 / 5.18  today
fillnan_climatology(G, group="month", residual="zero")
                                        # monthly clim.      8.40 / 7.92  today
# η̂ = μ + C_go C_oo⁻¹ (y − μ), one solve per day, ±15 d
oi(G, length_km=50, time_days=30)       # OI, L = 50 km      2.67 / 3.22  user
oi(G, length_km=100, time_days=30)      # OI, L = 100 km     1.78 / 2.59  user
# 1 − 1.78 / RMSE(baseline): 78 % over persistence; 69 % and 64 % over
# nearest and linear interpolation in time
```

Only carry forward is persistence: it repeats the last observation. Nearest
and linear interpolation in time also look at later observations, so they are
interpolation baselines, not persistence, and a forecast contract could not
use them. Against these three simple baselines OI improves by 78, 69 and
64 %, so "an improvement over the baseline" means nothing until the contract
pins which baseline and its gap handling (all three leave cells empty before
the first or after the last observation, which the notebook fills with zero).
OI itself depends on its assumptions: half the right correlation length costs
50 %. And the assumption can break the method. The usual Gaussian over the
shortest distance on a periodic domain is not a covariance at all: before the
noise term is added, its smallest eigenvalue on a day of 2025 is
−9.4 × 10⁻⁶ m² at L = 100 km and −1.8 × 10⁻³ m² at L = 150 km. The noise
σₒ² = 4 × 10⁻⁴ m² hides the first and not the second. The Gaussian summed
over the domain's periodic images is a valid covariance (smallest eigenvalue
zero to rounding at both lengths), and it is what the notebook's OI uses, so
a baseline must also fix a valid kernel for its domain. xrtoolz ships every gap-filler above except carry
forward and OI; it has `fillnan_rbf` and inverse-distance weighting but no
covariance-based interpolation (workstreams C1 and C2).

:::{warning} Differs from the plan
The scoping report puts the baselines in a new `pipekit_evaluate.baselines`
as `Operator` subclasses. A gap-filling baseline is a gap-filler, and xrtoolz
already holds seven (`fillnan_temporal`, `fillnan_climatology`,
`fillnan_spatial`, `fillnan_laplacian`, `fillnan_biharmonic`, `fillnan_rbf`,
`fillnan_idw`) with the xarray carrier they need; a second home
would split one concept across two packages, which Report 14 refused for
metrics. The baselines go into `xrtoolz.interpolate` (OI) and beside
`xrtoolz.geo`'s climatology (persistence, climatology forecasts); the
contract pins their configuration by name and arguments.
:::

(r15-stage-4)=
### Stage 4 · A deterministic and an ensemble entry under CRPS (xrtoolz)

**TL;DR.** Score a map and an ensemble of maps with the same probabilistic
score, and see whether the score tells them apart.

**Problem.** The continuous ranked probability score of a forecast
distribution F against the truth y, estimated from N_e members xᵢ, is

$$
\mathrm{CRPS}(F, y) = \mathbb{E}_F |X - y| - \frac{1}{2} \mathbb{E}_F |X - X'| ,
$$ (r15-eq-crps)

with X and X′ independent draws from F. For a point forecast the second term
is zero and CRPS is the mean absolute error. The ensemble is 20 draws
xᵢ = η̂ + σ̂ εᵢ, εᵢ ~ 𝒩(0, 1), from the L = 100 km OI of stage 3.

```{code-block} python
:caption: CRPS of an ensemble and of a deterministic map, from a real run.
:linenos:

# Shapes: m25 (364, 32, 32); ens (N_e = 20, 364, 32, 32)
from xrtoolz.metrics import crps_ensemble, spread_skill_ratio
crps_ensemble(ens, truth25, dims=dims)         # → 0.94 cm              today
spread_skill_ratio(ens, truth25, dims=dims)    # → 0.98                 today
crps_ensemble(m25.expand_dims(member=1), truth25, dims=dims)
# → ValueError: Ensemble/sample dim 'member' has size 1; need at least 2.
crps_ensemble(xr.concat([m25, m25], "member"), truth25, dims=dims)
# → 1.26 cm, equal to the mean absolute error of m25                    today
```

A point forecast is a valid, degenerate predictive distribution, and its CRPS
is its mean absolute error; WeatherBench 2 scores deterministic baselines
that way beside ensembles [@rasp2024weatherbench]. So the comparison is
meaningful: the map scores 1.26 cm, the calibrated ensemble 0.94 cm, and the
proper score rewards the ensemble for its spread. The defect is in the
plumbing: xrtoolz refuses a one-member ensemble, so a point entry has to be
entered twice to be scored. The fix (workstream D1) scores point forecasts
directly and records each entry's variant on the report, so a reader sees
which rows are deterministic; refusing them would remove the baselines a
probabilistic benchmark needs.

(r15-stage-5)=
### Stage 5 · Three references (xrtoolz → pipekit-evaluate)

**TL;DR.** Score the same maps against a reanalysis, the truth and the
withheld observations, and see whether the ranking survives.

**Problem.** A reanalysis is the output of an assimilation system and inherits
its smoothing. Here the stand-in is the true field smoothed with a Gaussian of
50 km (two cells). "Learned the reanalysis" is the L = 100 km OI map smoothed
the same way, as a model trained against that product would be. Each entry is
scored with {eq}`r15-eq-rmse` on three tracks.

```{code-block} python
:caption: One set of maps, three tracks, from a real run (RMSE in cm).
:linenos:

# Shapes: each entry (364, 32, 32); s1 track: the withheld 2025 observations
rean = truth.copy(data=gaussian_filter(eta, sigma=(0, 2, 2), mode="wrap"))  # user
for name, p in entries.items():                     # reanalysis · truth · s1
    scores[name] = (rmse(p, rean), rmse(p, truth), rmse_at(p, s1))      # user
# OI, L = 100 km          2.23 · 1.78 · 2.59
# OI, L = 50 km           2.83 · 2.67 · 3.22
# learned the reanalysis  1.22 · 2.66 · 3.25
```

The entry that learned the reanalysis wins its track by a factor of 1.8 and
loses the other two, coming last on the observations. The truth (which a
real benchmark does not have) and the withheld observations agree on the
winner, the L = 100 km OI, and the reanalysis does not. That is the scoping
report's three-track argument, reproduced: a single reanalysis track would
have put first the map that comes last on the observations. Today each
track is a separate loop the user writes; the runner takes one reference per
cell (workstream D2).

(r15-stage-6)=
### Stage 6 · The runner pipekit-evaluate ships (pipekit-evaluate)

**TL;DR.** Run the baselines through the benchmark runner that exists, and see
what it records.

**Problem.** pipekit-evaluate's `Benchmark` takes a `Task` whose `datasets()`
returns the observations and the truth, a list of `Estimator`s each placed at
a ladder cell (level, rung, complexity), and `Scorer`s. It resolves each
cell's reference with its `ReferenceRule` and scores the prediction.

```{code-block} python
:caption: Three baselines as estimators in the shipped runner, from a real run.
:linenos:

# Shapes: obs G (1,095, 32, 32); θ the true 2025 field (364, 32, 32)
ests = [Baseline("linear in time", rung=2, complexity=0),             # user
        Baseline("OI, L = 50 km", rung=2, complexity=1),
        Baseline("OI, L = 100 km", rung=2, complexity=2)]
rep = pe.Benchmark(task=MappingTask(), estimators=ests,
                   scorers=[RMSEScorer()]).run()                      # today
# → L4/r2/c0 known_theta 4.97 · L4/r2/c1 known_theta 2.67
#   L4/r2/c2 known_theta 1.78
pe.Benchmark(..., estimators=[*ests, Baseline("nearest in time", 2, 0)])
# → ValueError: duplicate benchmark cells (rung, complexity): [(2, 0)]  today
```

The runner works on a mapping task as written: the protocols are structural,
so a user's task and estimators satisfy them without importing
pipekit-evaluate. Three things stop it running a contract. Its cells are
ladder coordinates, so two baselines of the same rung and complexity collide,
and the report names cells, not methods. Its level defaults to L4 although
mapping ends at L2, because nothing sets it from the task. And `Task` has no
notion of a split: `datasets()` is one dictionary, so the train and test sets
of stage 2 are a convention between the user's task and estimators
(workstreams A and D).

(r15-names)=
## The scoping report's names, checked

:::{table} Every name the scoping report uses, against the pinned code.
:label: tbl-r15-names

| Scoping report | At the pinned revisions |
|---|---|
| `BenchmarkContract`, `content_hash`, `to_yaml`, `from_yaml` | do not exist; `pipekit.hashing.stable_json` and `sha256_hex` hash any mapping today (stage 1) |
| task `Literal["discretization", …, "representation"]` | pipekit-evaluate's `LEVEL_NAMES` (L0 to L4) name the same carriers; representation has no level |
| `Metric.differentiable`, `stage_compatibility`, `variant` | do not exist; Report 14's A1 proposes unit, lens, stage and differentiable tags on `Scorer` |
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

Four of the nine symbol groups need action. Read on 13 October 2026 at the
revisions in the report's header, on `main` of each submodule.

:::{table} The benchmark surface, by symbol group.
:label: tbl-r15-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `stable_json`, `sha256_hex` | pipekit `hashing.py` | the registry, `hash_config` in pipekit-train | **Healthy.** Deterministic; hashes the stage-1 contract as is |
| `Benchmark`, `BenchmarkReport`, `CellResult` | pipekit-evaluate `benchmark/run.py` | stage 6 | **Split ownership.** Cells keyed by `(rung, complexity)`; the report records no contract, model or data |
| `Task`, `Estimator`, `Scorer`, `Estimate` | `benchmark/protocols.py` | stage 6 | **Split ownership.** No split, no variant, no estimator name |
| `LEVEL_NAMES`, `Cell` | `benchmark/cube.py` | `Benchmark(level="L4")` | **Healthy.** The vocabulary the contract's task should use; the level is never set from the task |
| `XarrayWindowDataset` split, `_split_origins`, `split_guard` | pipekit-train `xarray_window.py:113-133` | Report 11's window dataset | **Split ownership.** Temporal only, fixed 80/10/10; the guard counts origins and can empty a split (stage 2) |
| `SimulationDataset` split | pipekit-train `dataset.py:220-226`, `:295-299` | training on simulations | **Healthy.** Disjoint seed ranges per split, so no leakage between draws |
| `fillnan_temporal`, `fillnan_climatology`, `fillnan_rbf`, `idw_to_points`, `CalculateClimatology` | xrtoolz `interpolate/_src/gap_fill.py`, `knn.py`, `geo/operators.py` | stage 3 | **Healthy.** The baselines a gap-filling benchmark needs, minus OI |
| `crps_ensemble`, `spread_skill_ratio`, `rank_histogram`, `ensemble_coverage`, `rank_methods`, `rmse_skill_scores` | xrtoolz `metrics/_src/` | stage 4 | **Healthy.** Probabilistic scores and a leaderboard table; the variant is checked on array shape only |
| `ModelRegistry`, `LocalModelRegistry`, `_JsonArtifact` | pipekit-experiment `protocols.py:51`, `registry.py`, `artifacts.py` | Report 12 | **Split ownership.** Content-addressed storage for operators only; artifacts save as JSON but are not addressed by hash |
:::

Smaller findings:

- **`rank_methods` already builds a leaderboard.** xrtoolz's
  `metrics/_src/leaderboard.py` sorts a per-method score dataset into a
  table; a report's `to_pandas` (Report 14's A3) can feed it.
- **`rmse_skill_scores` fixes one convention.** It pins `ddof=0` to match an
  upstream OSSE report, the kind of choice a contract has to record.
- **`fillnan_climatology` computes its climatology from its input.** On a test
  year that is legitimate (the inputs are observations of that year), but a
  climatology forecast must use the training years only, which needs a
  separate fit (workstream C3).
- **The window dataset's empty split is silent.** `XarrayWindowDataset` keeps a
  zero-length split rather than raising, so a guard set from the physics can
  remove a split without notice (stage 2).

(r15-ws-a)=
## Workstream A · The contract

**Decision.** A benchmark is a frozen dataclass of plain data in
`pipekit_evaluate.benchmark.contract`, hashed with pipekit's primitives, and
its hash travels with every report it produces.

:::{table} Workstream A's moves.
:label: tbl-r15-move-a

| Symbol | Destination | Reason |
|---|---|---|
| `BenchmarkContract(name, version, task, input, output, variant, splits, references, scorers, baselines, submission, status)` | `pipekit_evaluate.benchmark.contract` | the scoping report's artifact; `task` is a level transition (`"L1->L2"`), `status` is `"pre-registered"` or `"exploratory"` |
| `content_hash`, `to_yaml`, `from_yaml` | the same module, over `pipekit.hashing` | stage 1's hash, with YAML for people and JSON for the hash |
| `contract_hash`, `model_hash` on `BenchmarkReport` (and Report 14's `EvaluationReport`) | `benchmark/run.py`, `report.py` | stage 1: a report names nothing it scored |
| `Benchmark.from_contract(contract, estimators)` | `benchmark/run.py` | sets the level, the scorers and the references from the contract (stage 6) |
| templates per task | `pipekit_evaluate.benchmark.templates` | the scoping report's mitigation for long contracts |
:::

1. **Make the contract data.** Every field is a string, a number, a list or a
   mapping, so the contract round-trips through YAML and hashes stably. Splits,
   scorers and baselines are given by `_target_` and arguments, as Hydra
   would build them; the contract never holds an object.
2. **Hash the content, not the file.** `content_hash` is
   `sha256_hex(stable_json(asdict(self)))`, so formatting and key order never
   change it.
3. **Record the hash everywhere.** The runner copies the contract hash onto
   the report; a report without one is exploratory by construction.

```{code-block} python
:caption: Stage 1's contract, before and after workstream A.
:linenos:

# Shapes: as stage 1
# Today:
h = sha256_hex(stable_json(contract))                   # a dict        today
# After:
c = pe.BenchmarkContract(name="ssh-mapping-synthetic", version="1.0.0",
                         task="L1->L2", variant="deterministic",
                         splits={"_target_": "geocatalog.splits.by_platform",
                                 "platform": "s1"},          # arguments sketched
                         status="pre-registered")        # proposed (A1)
c.content_hash()                                         # proposed (A1)
rep = pe.Benchmark.from_contract(c, estimators=ests).run()  # proposed (A2)
rep.contract_hash == c.content_hash()                    # proposed (A2)
```

(r15-ws-b)=
## Workstream B · Splits as contract terms

**Decision.** A split is a function of catalog rows in geocatalog, or of
windows in pipekit-train, configured by the contract and refusing what it
cannot honour; the contract also says which gap the split reproduces.

:::{table} Workstream B's moves.
:label: tbl-r15-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `split_blocks(catalog, *, km, days, test_fraction, seed, min_scale=None, crossing="drop")` | `geocatalog.splits` (new) | stage 2's blocks; rows carry footprints and intervals, so a row is assigned only when it lies wholly inside one block (`crossing="drop"`, or `"group"` to merge the blocks it touches); raises when a block is smaller than the declared scale or the domain cannot hold one |
| `split_by(catalog, column, *, test)` | `geocatalog.splits` | stage 2's withheld satellite, over a platform or producer column ([Report 13](13_state_catalog.md#r13-ws-a)'s `producer`) |
| `split_causal(catalog, *, train_end, val_end)` | `geocatalog.splits` | the forecast split: every test row after every training row |
| `split_regions(catalog, regions)` | `geocatalog.splits` | the scoping report's leave-one-region-out, as footprint intersection |
| `fractions` and `guard` as a `pd.Timedelta` | pipekit-train `XarrayWindowDataset`, `_split_origins` | stage 2: the guard counts origins, and a long one empties the validation split silently |
| `gap` field on the contract's split | `BenchmarkContract.splits` | stage 2's correction: the split states the deployment gap it reproduces |
:::

### B1 · Blocks in space and time

A block split tiles space and time into blocks and holds out whole blocks. A
row whose footprint or time
interval crosses a block boundary would carry observations from one block
into the other, so such rows go to neither side by default (or, as an option,
are grouped with every block they touch). It refuses a block smaller than the
declared correlation scale when the contract asks for generalization, and a
block that does not fit the catalog's domain.

### B2 · Platform, causal and region splits

The platform split is the one stage 2 shows a mapping benchmark needs: hold
out every row of one producer. The causal split is the forecast's. Both are
row filters over columns geocatalog already has, or that Report 13's state
profile adds.

### B3 · The window guard in time

`XarrayWindowDataset` takes its guard as a duration, converts it to origins
with the source's step, and raises when any split ends up empty, naming the
guard and the split's length.

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
pt.XarrayWindowDataset(..., guard=pd.Timedelta("60D"))
# → ValueError: guard 60 days leaves split 'val' (105 days) empty  proposed (B3)
```

(r15-ws-c)=
## Workstream C · Baselines with the carriers

**Decision.** Baselines are ordinary operators in the carrier library, beside
the gap-fillers xrtoolz ships; the contract names them and pins every
argument.

:::{table} Workstream C's moves.
:label: tbl-r15-move-c

| Symbol | Destination | Reason |
|---|---|---|
| `optimal_interpolation(obs, grid, *, kernel, length, time_scale, noise, window)` and `OptimalInterpolation` | `xrtoolz.interpolate` | stage 3's best baseline, absent from the stack; the kernel is valid on the grid (images summed on periodic domains, or a compactly supported kernel), checked with a Cholesky factorization of the prior before the noise is added |
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
# η̂ = μ + C_go C_oo⁻¹ (y − μ); σ̂² = σ² − diag(C_go C_oo⁻¹ C_og)
mean, var = xi.optimal_interpolation(G, grid, kernel="gaussian", length=100e3,
                                     time_scale="30D", noise=0.02,
                                     window="15D")              # proposed (C1)
```

(r15-ws-d)=
## Workstream D · Variants and tracks in the runner

**Decision.** The runner checks an entry against the contract before scoring
it, scores every declared reference, and names entries.

:::{table} Workstream D's moves.
:label: tbl-r15-move-d

| Symbol | Destination | Reason |
|---|---|---|
| `variant` on `Scorer` and on the entry's `Estimate` (`"deterministic"`, `"probabilistic"`) | `benchmark/protocols.py`, beside Report 14's A1 tags | stage 4: the report should say which rows are point forecasts |
| point forecasts under CRPS | `xrtoolz.metrics.crps_ensemble` and the runner | a point forecast is scored as a degenerate distribution (its CRPS is its MAE) without duplicating it; stage 4's duplication goes |
| `references: Mapping[str, ReferenceSource \| str]` | `Benchmark`, building on Report 14's A4 | stage 5's three tracks in one run; a track with no data reports "unavailable", as `CellResult.note` already does for a missing oracle |
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

**Decision.** pipekit-experiment stores contracts, reports and submissions by
content hash, and a submission verifies by recomputing hashes, not by trust.

:::{table} Workstream E's moves.
:label: tbl-r15-move-e

| Symbol | Destination | Reason |
|---|---|---|
| `ArtifactStore.put(obj) -> hash`, `get(hash)` for JSON artifacts | `pipekit_experiment` (beside `ModelRegistry`, same layout) | `store` takes only operators; contracts and reports are data |
| `BenchmarkSubmission(contract_hash, model_hash, report_hash)` with `verify(store)` | `pipekit_evaluate.benchmark.submission` | checks that the report's `contract_hash` and `model_hash` match, and that the contract is pre-registered |
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

Each workstream merges only when its gates pass on the running example.

:::{table} Gates, by workstream.
:label: tbl-r15-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Contract hash | A | the stage-1 contract as a `BenchmarkContract` hashes to the same value after a YAML round trip and in any key order; changing any one field (the withheld platform, a block size, a baseline argument) changes the hash |
| A2 Report hash | A | `Benchmark.from_contract(c, …).run().contract_hash == c.content_hash()`, and the report's level is the contract's (L2 for stage 6, not L4) |
| B1 Blocks | B | on stage 2's observations, `split_blocks(km=200, days=60, seed=0)` reproduces the notebook's test set and its RMSE (4.26 cm against the truth) to 1e-4, assigning no row that crosses a block boundary; `km=50` with `min_scale=(100, 30)` raises `ValueError` |
| B2 Platform and causal | B | `split_by("producer", test="s1")` reproduces stage 2's 2.73 cm, and every test time of `split_causal` is later than every training time |
| B3 Window guard | B | `XarrayWindowDataset(guard=pd.Timedelta("60D"))` on stage 2's series raises `ValueError` naming the empty split; with `guard="2D"` its splits are 871, 105 and 108 windows |
| C1 OI | C | `optimal_interpolation` reproduces stage 3's 1.78 and 2.59 cm to 1e-4 m; with L = 150 km on the 800 km periodic grid its prior covariance, before the noise term, has smallest eigenvalue ≥ 0 to rounding, where the minimum-image Gaussian gives −1.8 × 10⁻³ m² |
| C2 Persistence | C | `persistence` reproduces stage 3's carry-forward 7.96 cm and uses no observation after its reference time; nearest and linear interpolation in time (5.76 and 4.97 cm) are separate interpolation baselines, and each configuration hashes differently |
| C3 Climatology | C | `ClimatologyForecast(train_period=("2023", "2024"))` uses no 2025 value: perturbing every 2025 input changes its output by 0 |
| D1 Variant | D | on a probabilistic contract, stage 4's map is scored without duplication at its MAE, 1.26 cm, its row is tagged deterministic, and the 20-member ensemble scores 0.94 cm CRPS |
| D2 Tracks | D | one run reproduces stage 5's nine scores to 1e-4 m, and a track with no reference appears with the note "unavailable" |
| D3 Names | D | stage 6's four baselines, two at the same cell, run without error and the report is keyed by their names |
| E1 Store | E | `store.put(c) == c.content_hash()` and `store.get(h)` returns a contract equal to `c` |
| E2 Verify | E | a submission verifies for the stage-6 report, and fails after any change to the contract, the model hash or the report |
:::

(r15-sequencing)=
## Sequencing and risks

The contract and the splits come first, because every later check needs
something to check against; the hosted service never comes into the framework.

:::{figure} ../images/reports/r15_roadmap.svg
:label: fig-r15-roadmap
:alt: Phase 1: A1 and A2, the contract and its hash on every report, beside B1 to B3, the block, platform and causal splits and the guard in days; then gates A1, A2, B1, B2, B3. Phase 2: C1 to C3, the OI, persistence and climatology baselines, beside D1 to D3, the variant check, tracks and named entries; then gates C1 to D3. Phase 3: E1 and E2, the contract and submission store and verification; then gates E1 and E2.

The contract and its splits first, then the baselines and runner checks, then
the store.
:::

- **The contract outgrows its readers.** A full contract with carriers,
  splits, scorers and baselines is long for a person to write. Fallback:
  templates per task (A), and a contract can reference a template by hash and
  override fields.
- **Splits over rows need rows.** A dataset that is one Zarr store has one
  catalog row, so a block split cannot see inside it. Fallback: split the
  windows inside the dataset (B3) with the same block rule, or index the store
  per time step with Report 13's per-step option.
- **A split that matches use needs knowledge of use.** Stage 2 shows the right
  split depends on the gap the method will face, which is a scientific
  judgement. Fallback: the contract's `gap` field is required and free text,
  so at least the choice is stated and reviewable.
- **OI does not scale as written.** One dense solve per day is fine for 320 to
  704 observations, not for a global product. Fallback: a localized OI (a radius
  per grid cell) behind the same function, or the Gaussian-process tools once
  pyrox-gp or gaussx is pinned.
- **Blind tests need a service.** Without one, `max_test_evaluations` is
  recorded but not enforced. Fallback: the contract marks its test set
  `blind`, and projects that run a service enforce it there.
- **Two report types.** Report 14's `EvaluationReport` and the ladder's
  `BenchmarkReport` could both grow hashes. Fallback: the hash fields go on
  the one report Report 14's A3 leaves standing.

(r15-open-questions)=
## Open questions

1. **Does a contract pin code or names?** A baseline pinned as
   `xrtoolz.interpolate.optimal_interpolation` with arguments still changes if
   xrtoolz changes. Should a contract also pin the library versions, as
   `TrainingArtifact.deps_lock` does?
2. **Who sets the gap?** Stage 2's withheld satellite is right for mapping
   with today's constellation. Should a contract carry several splits (a
   withheld satellite and a withheld region) and report both?
3. **Pre-registered means what?** The hash proves the contract did not
   change. It does not prove it was published before the results. Is a
   timestamped record outside the repository (a DOI, a tagged release)
   required for the `pre-registered` status?
4. **Representation tasks.** They are not a level transition. Do they need a
   contract kind of their own, or are they scored only through a downstream
   task?
5. **Where does a reanalysis track come from?** Report 13's state profile
   indexes reanalyses as catalog rows. Should a contract's references be
   catalog hashes rather than URIs?

(r15-todos)=
## Upcoming work

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r15-todos

| Item | Where | Why |
|---|---|---|
| Review the gallery's eighteen draft contracts with domain experts, then pre-register them | [benchmarks gallery](../benchmarks_gallery.md) | the drafts hash today, but their metrics, baselines and splits need review before their hashes mean anything |
| Raise on an empty split | pipekit-train `XarrayWindowDataset` | stage 2: a long guard empties a split silently; independent of B3's units |
| Index xrtoolz submodules in the capability index | `scripts/capabilities.py` (`UPSTREAM`) | the index lists only top-level `xrtoolz`, so `fillnan_*`, `crps_ensemble` and `rank_methods` are not searchable (Report 14 found the same) |
| Pin pyrox-gp or gaussx | `references/`, `UPSTREAM` | the scoping report maps Gaussian-process gap filling to them; neither is checked |
| Write the representation contract | Report 15, open question 4 | no level transition to name it by |
:::

(r15-references)=
## References

:::{note} Companion reports
:class: simple
Reports [6](06_geocatalog.md), [10](10_pipekit_cycle.md),
[11](11_pipekit_train.md), [12](12_pipekit_experiment.md),
[13](13_state_catalog.md) and [14](14_pipekit_evaluate.md) are published.
:::

### Sources

- The scoping report, "Report 15 — Benchmarking as first-class
  infrastructure" (supplied by the author).
- The pinned code in `references/`: pipekit (`hashing.py`), pipekit-evaluate
  (`benchmark/`), pipekit-train (`xarray_window.py`, `dataset.py`),
  pipekit-experiment (`registry.py`, `artifacts.py`), xrtoolz
  (`interpolate/`, `metrics/`, `geo/`) and geocatalog.
- OceanBench, the SSH mapping benchmark the running example imitates
  [@johnson2023oceanbench]; WeatherBench 2 [@rasp2024weatherbench].
- Block cross-validation for spatially and temporally structured data
  [@roberts2017], the standard reference for the leakage argument of stage 2.
- The running example's notebook, described in [](#r15-running).
