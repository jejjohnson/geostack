---
title: The state catalog
subtitle: Report 13 · Indexing model states, reassessed against geocatalog, xrtoolz and pipekit-cycle
short_title: 13 · State catalog
description: The scoping report proposed statecatalog, a sister package for forecasts, analyses and reanalyses. A reassessment of the pinned code finds most of it already in place, split across geocatalog, xrtoolz and pipekit. This page replaces the package with a state profile in geocatalog and three small additions elsewhere, and checks every claim against a real run.
keywords:
  - model states
  - forecasts
  - lead time
  - ensembles
  - lineage
  - geocatalog
  - xrtoolz
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

A **model state** is a field a model produced: an analysis, a forecast step,
one ensemble member of a reanalysis. Unlike an observation, it has two times
(when the run started and when the state is valid), a member index, a
producer and a parent state. The scoping report proposed `statecatalog`, a new
sister package with its own slice, backends and queries for these states.
Reading the pinned code shows that most of that package already exists in
pieces. geocatalog rows can carry the extra columns and filter on them,
xrtoolz scores forecasts by lead time and by ensemble, and pipekit-cycle keeps
the trajectory in memory. What is missing is narrower: typed state columns,
loading that keeps runs apart, a handful of queries, forecast alignment and an
ensemble history, plus area weighting when producers run on different
grids, as in a CMIP ensemble. **This report recommends building those pieces
where their concepts already live, and not building `statecatalog`.**

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Redesign of a scoping proposal ("committed": a sister package `statecatalog`). The user asked for it to be reassessed against the geotoolz and xrtoolz stacks; the recommendation below replaces the package |
| **Reading time** | about 30 minutes |
| **Audience** | anyone who stores, queries or scores forecasts, analyses or reanalyses with the stack |
| **Companions** | [Report 6](06_geocatalog.md) (geocatalog); [Report 3](03_sister_libraries.md) (xrtoolz and the sister libraries); [Report 9](09_use_cases.md) (use cases); [Report 10](#r13-ref-report-10) (pipekit-cycle, forthcoming) |
| **Code checked against** | the revisions pinned in `references/`: pipekit and its sister packages at 0.0.2 (`ef3c17f`); geotoolz 0.10.2 with geocatalog 0.5.1 (`088475e`); xrtoolz 0.0.5 with xrreader 0.0.4 (`308c7f2`); pandas 3.0.6, as in `pixi.lock`. Read on 13 October 2026 |
| **Running example** | every output in the code blocks is from real runs over two synthetic archives: an 11-file regional forecast archive (stages 1 to 5) and a 6-file, three-model archive on three grids (stage 6); both are described in [](#r13-running), and `notebooks/reports/report13_state_catalog.py` regenerates them from fixed seeds and asserts every number shown |
:::

:::{important} What changed from the scoping report
The scoping report's vocabulary holds: valid time, run time, lead time,
ensemble member, producer hash and parent state are the right fields, and
ensembles and lineage belong in the first version. Its packaging does not.
The report argued that putting model-state fields into geocatalog "would
corrupt the abstraction". At the pinned revisions the fields fit in
geocatalog's **rows**, not in `GeoSlice`, and the slice stays what it is: a
request for a region and a time interval. Each place where the scoping report
and the code disagree has a **Differs** box, and the gaps go to
[](#r13-open-questions) or [](#r13-todos).
:::

(r13-summary)=
## Summary

**Decision.** Model states are indexed by geocatalog, through a new
`geocatalog.states` namespace over the existing backends, and scored by
xrtoolz's existing forecast metrics. There is no `statecatalog` package, no
`StateSlice` and no second GeoParquet format. The overview is updated to
match in the same change. No library starts importing
another: pipekit-cycle, geocatalog and xrtoolz stay independent, and scripts
join them.

- **A. State columns in geocatalog.** Today a forecast's run time and member
  can live in extra columns, but nothing names, types or fills them. Reserve
  four columns in schema v1 and add a builder that reads them from CF or
  cfgrib metadata.
- **B. Run-aware loading and typed queries.** `load_xarray` concatenates on
  time and drops duplicates, so overlapping runs lose steps silently. Add
  `load_states`, which stacks files into an (init time, member, lead time)
  cube, or into an `xr.DataTree` with one leaf per producer when producers
  use different grids, and typed query functions that replace hand-written
  `where` strings.
- **C. Forecast alignment in xrtoolz.** `SkillByLeadTime` and `CRPS` score a
  forecast cube today, but the caller builds the verifying reference by hand,
  and pixel metrics cannot reduce over a dimension the reference lacks. Add
  `align_reference`, broadcast the reference in the pixel metrics, and let
  `geo.Reduce` weight by cell area, so multi-model area means need no user
  code.
- **D. Ensemble histories in pipekit-cycle.** `Cycle` and `DACycle` keep
  their trajectory in `history`; `EnsembleCycle` and `EnsembleDACycle`
  keep nothing. Add `save_history` to both. Writing files stays in scripts, as AGENTS.md
  requires.

**End state.** A script runs a cycle and writes one file per run and member.
geocatalog indexes the files with typed state columns and loads them back as
a cube without mixing runs. xrtoolz aligns the cube with its analyses and
scores it by lead time. Runs from several models on several grids load as
a tree, are summarised on their own grids and are regridded conservatively
only for the comparison. pipekit-experiment's `Run.pipeline_hash` names the
producer. Every one of the scoping report's queries is either a function in
`geocatalog.states` or an existing call.

(r13-problem)=
## The problem, as the scoping report put it

An observation has one time. A model state has several, and the scoping
report named them well. Let $r$ be the **run time** (the initialisation time
of a forecast), $v$ the **valid time** (the time the state describes), and

$$
\tau = v - r
$$ (r13-eq-lead)

the **lead time**. An **analysis** is the state a data-assimilation step
produces at its own valid time, so $\tau = 0$. A forecast run started at $r$
produces states at $\tau = \Delta t, 2 \Delta t, \ldots$ for a model step
$\Delta t$. An ensemble repeats each run $N_e$ times, with members
$m = 0, \ldots, N_e - 1$. Each state also has a **producer**, the hash of
the model and configuration that made it, and its **parents**, the states
it was computed from: one for a forecast (the state its run started from), and
the whole forecast ensemble for an ensemble-Kalman analysis, whose gain is
estimated from all members at once.

:::{table} The fields of a model state, their CF names and where each one lives today.
:label: tbl-r13-fields

| Field | Symbol | CF / cfgrib name | CMIP6 attribute | Today |
|---|---|---|---|---|
| Valid time | $v$ | `time`, `valid_time` | `time` | geocatalog's row interval (`start_time`, `end_time`) |
| Run time | $r$ | `forecast_reference_time` / cfgrib `time` | | an extra column, if the user adds it |
| Lead time | $\tau = v - r$ | `forecast_period` / cfgrib `step` | | derived; xrtoolz's `lead_dim="lead_time"` |
| Member | $m$ | `realization` / cfgrib `number` | `member_id`, a variant label such as `r1i1p1f1` | an extra column; xrtoolz's `ensemble_dim="member"` |
| Producer | $h$ | | `source_id` | `Run.pipeline_hash` in pipekit-experiment; an extra column in the catalog |
| Parents | $P(s)$, a set | | | nowhere |
:::

The questions the scoping report wanted answered are the right ones: which
states are valid at a time, what the latest analysis before a time is, which
states one run produced, which members share a run, what a state descends
from, and which observations an analysis assimilated. The rest of this page
answers each one with the code that exists, and adds what is missing.

(r13-ownership)=
## Ownership rules

States are produced by pipekit-cycle, indexed by geocatalog and scored by
xrtoolz, and only scripts join the three.

:::{figure} ../images/reports/r13_ownership.svg
:label: fig-r13-ownership
:alt: Scripts at the top call four independent libraries: pipekit-cycle, which produces states; geocatalog.states, the only new surface, which indexes them; xrtoolz.metrics, which scores them; and pipekit-experiment, which names the producer. geocatalog.states sits on geocatalog's core, which sits on NetCDF, Zarr and GeoParquet files.

Ownership of model states. The dashed box is the only new namespace; the
other three already ship.
:::

1. **pipekit-cycle** owns producing states: the cycles and their in-memory
   `history`. It never imports geocatalog or xrtoolz, and never writes files.
2. **geocatalog** owns the index of states (`geocatalog.states`): the
   reserved columns, the state builder, run-aware loading and the state
   queries. It never imports pipekit or xrtoolz.
3. **xrtoolz** owns verifying states: alignment with a reference and the
   lead-time and ensemble metrics. It never imports geocatalog.
4. **pipekit-experiment** owns the producer's identity (`Run.pipeline_hash`,
   the registry's content hashes). It never imports a catalog.
5. **Scripts** (here, geostack's `scripts/`) own the side effects: running
   the cycle, writing one file per run and member, building the catalog and
   writing metrics.

:::{table} The dependency tiers for model states.
:label: tbl-r13-tiers

| Tier | Packages | May import | Role |
|---|---|---|---|
| Entry points | geostack scripts and notebooks | everything below | run, write, index, score |
| Libraries | pipekit-cycle; geocatalog; xrtoolz; pipekit-experiment | their own substrate only | produce; index; verify; name |
| Substrate | xarray, pandas, geopandas, pyarrow, DuckDB | | arrays, tables and files |
:::

:::{warning} Differs from the plan
The scoping report placed `statecatalog` as a **sibling of geocatalog** with
its own `StateSlice`, `StateRow`, `StateCatalog` Protocol, in-memory and
DuckDB backends, streaming writes and GeoParquet extension: about 1,100 lines
and 2.5 to 3 weeks. Every one of those pieces exists in geocatalog already,
for rows that carry extra columns ([](#r13-stage-1)). A second package would
duplicate the backends, the GeoParquet round trip and the schema
versioning, and give users two indexes to learn. The overview
([](../overview.md)) described the two-index design until this change; it
now describes one index with a state profile.
:::

(r13-running)=
## The running example: a regional methane forecast archive

**The archive.** A regional methane system covers a 4 × 4 grid over the
Permian Basin (31° to 34° N, 104° to 101° W). Each day it runs one
data-assimilation step, which writes an **analysis** of the column-methane
anomaly $a(v, x)$ in ppb at grid cell $x$. From each of the first three
analyses it launches a two-member ensemble **forecast** with leads of 0, 1
and 2 days. The archive therefore holds $5$ analysis files and
$R \times N_e = 3 \times 2 = 6$ forecast files, one per run and member, each
NetCDF with a `time` axis of valid times. Each file's attributes record its
kind, run time, member, producer hash and parent file.

**The data.** The truth is a random walk in time; the analyses add noise
with a standard deviation of 0.05 ppb, and the forecast error grows with lead
as $\sigma_\tau = 0.1 + 0.5 \tau$ (in days). The data are synthetic, so the
scores in stage 4 can be checked against their expected values.

The notebook [`notebooks/reports/report13_state_catalog.py`](https://github.com/jejjohnson/geostack/blob/main/notebooks/reports/report13_state_catalog.py)
builds the archive from a fixed seed in a temporary directory, runs every
stage, and asserts each number on this page; it runs with geocatalog 0.5.1
and xrtoolz 0.0.5 from `references/`. Stage 6 widens the example from one
regional system to a multi-model comparison, with its own six-file archive
in the same notebook. The blocks below are excerpts with their real outputs;
names such as `cat`, `t3` and `five_days` are defined in the notebook. Lines marked `user` are glue the user
writes today; the workstreams remove them.

(r13-stage-1)=
### Stage 1 · Index the archive (scripts → geocatalog)

**TL;DR.** Index the eleven files so that later questions never open a file.

**Problem.** Each file $f$ has a footprint, a valid-time interval
$[v_{\min}(f), v_{\max}(f)]$ and a path, which geocatalog's xarray kind
already records. A state index also needs $r(f)$, $m(f)$, $h(f)$ and the
parent $p(f)$ on every row.

```{code-block} python
:caption: Index the archive with today's xarray builder, then copy the state fields into extra columns.
:linenos:

# Shapes: R = 3 runs, N_e = 2 members, L = 3 leads, H = W = 4 cells; 11 files
from geocatalog.build import build_xarray_catalog
from geocatalog.storage import from_geoparquet, to_geoparquet

# one row per file: footprint, [v_min, v_max], path
cat = build_xarray_catalog(paths, crs="EPSG:4326")      # 11 files → 11 rows  today
# r(f), m(f), h(f), p(f): read from the file attributes
attrs = [xr.open_dataset(p).attrs for p in cat.gdf.filepath]           # user
cat.gdf["run_time"] = pd.to_datetime([a["run_time"] for a in attrs])   # user
cat.gdf["member"] = [int(a["member"]) for a in attrs]                  # user
cat.gdf["producer"] = [a["producer"] for a in attrs]                   # user
cat.gdf["parent_uri"] = [a["parent_uri"] for a in attrs]               # user
to_geoparquet(cat, "states.parquet")                     # extras kept          today
cat = from_geoparquet("states.parquet")                  # run_time datetime64  today
```

The extra columns survive the GeoParquet round trip with their types
(`run_time` as `datetime64[us]`, `member` as `int64`), because geocatalog
keeps every non-reserved column in `CatalogRow.extras`. What does not exist
is a builder that fills them: the xarray builder reads neither
`forecast_reference_time` nor `realization`, and a file without a `time`
coordinate gets the placeholder interval 1900 to 2100 with
`n_timesteps=0`. Workstream A adds the builder and reserves the names.

(r13-stage-2)=
### Stage 2 · Ask the scoping report's questions (geocatalog)

**TL;DR.** Find one run's forecasts, the latest analysis before a time, and
the members of a run, using only the index.

**Problem.** With $\mathcal{C}$ the set of rows, the three questions are
filters:

$$
\begin{aligned}
\mathrm{chain}(r_0, m_0) &= \{ f \in \mathcal{C} : r(f) = r_0, m(f) = m_0, f \text{ a forecast} \} , \\
\mathrm{latest}(t) &= \arg\max_{f \text{ an analysis}} \{ v_{\max}(f) : v_{\max}(f) \le t \} , \\
\mathrm{members}(r_0) &= \{ m(f) : r(f) = r_0, f \text{ a forecast} \} .
\end{aligned}
$$ (r13-eq-queries)

```{code-block} python
:caption: The three questions as where strings over the extra columns, with their outputs.
:linenos:

# chain(r₀ = 01-02, m₀ = 0)
cat.where("kind == 'forecast' and member == 0 and run_time == '2026-01-02'")
# → 0 rows: a string equality on a datetime column matches nothing    today
cat.where("kind == 'forecast' and member == 0"
          " and run_time >= '2026-01-02' and run_time < '2026-01-03'")
# → ['fc_20260102_m0.nc']                                               today

# latest(01-03 06Z): end times live in the IntervalIndex after a round trip
ana = cat.where("kind == 'analysis'").gdf                              # today
ana = ana[ana.index.right <= pd.Timestamp("2026-01-03 06:00")]         # user
ana.iloc[ana.index.right.argmax()].filepath   # → 'ana_20260103.nc'    user

# members(r₀ = 01-01)
cat.where("kind == 'forecast' and run_time < '2026-01-02'").gdf.member
# → [0, 1]                                                              today
```

Every answer is right, but each one is a string the user must get exactly
right. The first query shows why that matters: on pandas 3.0.6, the version
in `pixi.lock`, `DataFrame.query("t == '2026-01-02'")` on a datetime column
returns no rows, while `df[df.t == "2026-01-02"]` returns one. The failure
is silent. Typed functions that take a `pd.Timestamp` (workstream B2) remove
the trap.

(r13-stage-3)=
### Stage 3 · Load a forecast cube (geocatalog → xarray)

**TL;DR.** Load the six forecast files as one array indexed by run, member
and lead.

**Problem.** The cube xrtoolz expects is

$$
\begin{aligned}
F &\in \mathbb{R}^{R \times N_e \times L \times H \times W} , \\
F[r, m, \tau, x] &= \text{member } m \text{ of run } r \text{ at lead } \tau \text{, cell } x ,
\end{aligned}
$$ (r13-eq-cube)

with a two-dimensional coordinate $v = r + \tau$. Three runs overlap in valid
time: all three are valid on 3 January, at leads 2, 1 and 0.

```{code-block} python
:caption: Today's loader keeps one run per valid time; the cube has to be stacked by hand.
:linenos:

# Shapes: R = 3, N_e = 2, L = 3, H = W = 4
from geocatalog import GeoSlice
from geocatalog.load import load_xarray

rows = cat.where("kind == 'forecast'").gdf                # 6 rows   today
jan3 = GeoSlice(bounds=(-104.5, 30.5, -100.5, 34.5),
                interval=pd.Interval(t3, t3, closed="both"),
                resolution=(1.0, 1.0), crs="EPSG:4326")          # today
m0 = cat.where("kind == 'forecast' and member == 0").query(jan3)  # 3 rows   today
load_xarray(m0, jan3).sizes["time"]                  # → 1: two runs dropped  today

# F[r, m, τ, x]: one file per (r, m), relabelled from v to τ = v − r
pieces = []
for (r, m), row in rows.set_index(["run_time", "member"]).iterrows():  # user
    ds = xr.open_dataset(row.filepath).load()                         # user
    ds = (ds.assign_coords(lead_time=("time", ds.time.values - r))    # user
            .swap_dims(time="lead_time").drop_vars("time"))          # user
    pieces.append(ds.expand_dims(init_time=[r], member=[m]))          # user
fc = xr.combine_by_coords(pieces, combine_attrs="drop")              # user
fc = fc.assign_coords(valid_time=fc.init_time + fc.lead_time)        # user
# → sizes {'init_time': 3, 'member': 2, 'lead_time': 3, 'lat': 4, 'lon': 4}
```

`load_xarray` is right for observations, where two files covering the same
time are duplicates. For forecasts the same valid time from two runs is two
different states, and `drop_duplicates` on time keeps one of them without a
warning. The eight user lines are the cube builder that workstream B1 puts in
`load_states`.

(r13-stage-4)=
### Stage 4 · Score the forecasts by lead time (xrtoolz)

**TL;DR.** Compare each forecast with the analysis valid at the same time,
and report the error at each lead.

**Problem.** The verifying reference for run $r$ at lead $\tau$ is the
analysis at its valid time, $A[r, \tau, x] = a(r + \tau, x)$. With
$\bar{F}$ the ensemble mean over members and $N = R H W$ the number of
scored values per lead, the two scores are

$$
\begin{aligned}
\mathrm{RMSE}(\tau) &= \left( \frac{1}{N} \sum_{r, x} \left( \bar{F}[r, \tau, x] - A[r, \tau, x] \right)^2 \right)^{1/2} , \\
\mathrm{CRPS}(\tau) &= \frac{1}{N} \sum_{r, x} \mathrm{CRPS}\left( F[r, \cdot, \tau, x], A[r, \tau, x] \right) ,
\end{aligned}
$$ (r13-eq-skill)

with the CRPS of an ensemble as in the overview ({eq}`eq-crps`). For this
archive the expected ensemble-mean RMSE is
$(\sigma_\tau^2 / 2 + 0.05^2)^{1/2}$, which is 0.087, 0.427 and 0.779 ppb at
leads 0, 1 and 2 days.

```{code-block} python
:caption: Align the analyses with the cube, then score by lead with today's xrtoolz metrics.
:linenos:

# Shapes: F (R, N_e, L, H, W); A (R, L, H, W)
from xrtoolz.metrics import CRPS, RMSE, SkillByLeadTime

an = load_xarray(cat.where("kind == 'analysis'"), five_days)  # (T = 5, H, W)  today
# A[r, τ, x] = a(r + τ, x)
ref = an.sel(time=fc.valid_time).drop_vars("time")       # → (R, L, H, W)      user
dims = ("init_time", "lat", "lon")
# RMSE(τ) of the ensemble mean
SkillByLeadTime(RMSE("xch4", dims=dims))(fc.mean("member"), ref)  # → (L,)   today
# → [0.076, 0.467, 0.746]
# CRPS(τ) over members
SkillByLeadTime(CRPS("xch4", ensemble_dim="member", dims=dims))(fc, ref)  # today
# → [0.048, 0.356, 0.605]
# per-member RMSE: reduce over the member dimension too
SkillByLeadTime(RMSE("xch4", dims=("member", *dims)))(fc, ref)           # today
# → ValueError: tuple.index(x): x not in tuple
SkillByLeadTime(RMSE("xch4", dims=("member", *dims)))(*xr.broadcast(fc, ref))
# → [0.109, 0.617, 1.132]                                               user
```

The measured RMSE (0.076, 0.467 and 0.746) sits within sampling error of
the expected values, with $N = 48$ cells per lead. The per-member RMSE
matches its expected $(\sigma_\tau^2 + 0.05^2)^{1/2}$, which is 0.112, 0.602
and 1.101. The scoring is shipped. What is missing is the alignment line
(workstream C1) and the broadcast that lets pixel metrics reduce over a
dimension only the forecast has (C2).

:::{warning} Differs from the plan
The scoping report's verification example looped over forecasts, looked up
"the latest analysis before" each valid time, read both files and called a
`compute_rmse` it did not define. Two things changed. Verification is
already in xrtoolz (`SkillByLeadTime`, `CRPS`, `SpreadSkillRatio`,
`RankHistogram`, `EnsembleCoverage`, and `viz.validation.LeadTimeSkillPanel`
to plot it), so the catalog's job is only to load the cube. And "the latest
analysis before $v$" is the wrong reference: it is the analysis **at** $v$,
which is an exact selection on the aligned cube, not a query per forecast.
:::

(r13-stage-5)=
### Stage 5 · Trace lineage and the assimilation window (geocatalog)

**TL;DR.** Follow a state back through the runs it came from, and find the
observations an analysis used.

**Problem.** With $P(s)$ the set of parents of state $s$, the lineage of $s$
is every state reachable by following parents, $s$, $P(s)$, $P(P(s))$, and
so on, until no state has parents left: a directed acyclic graph, which is a
chain when every state has one parent. This archive is the simple case:
each analysis lists one parent, the member-0 forecast. An ensemble-Kalman
analysis would list every forecast member, because its gain is computed from
all of them. For an
analysis at time $t$ with an assimilation window of width $w$, the
observations used are the rows $o$ of an observation catalog whose
valid time falls in $[t - w/2, t + w/2]$.

```{code-block} python
:caption: Lineage is a walk over the parent column; the assimilation window is an ordinary query on the observation catalog.
:linenos:

# s, P(s), P(P(s)), … until P = ∅; one parent per state in this archive
by_uri = cat.gdf.set_index("filepath")                                  # user
uri, chain = "ana_20260104.nc", []                                      # user
while uri:                                                              # user
    chain.append(uri)                                                   # user
    uri = by_uri.loc[uri, "parent_uri"]                                 # user
# → ana_20260104 ← fc_20260103_m0 ← ana_20260103 ← fc_20260102_m0
#   ← ana_20260102 ← fc_20260101_m0 ← ana_20260101

# obs with v ∈ [t − w/2, t + w/2]                              # arguments sketched
window = GeoSlice(bounds=aoi, interval=pd.Interval(t - w / 2, t + w / 2,
                  closed="both"), resolution=res, crs="EPSG:4326")  # today
obs_used = obs_cat.query(window)                       # (M rows)          today
```

The lineage walk is five lines of user code when each state has one parent,
and a graph search when states have several. The scoping report's `assimilation_window` needs no new code: it is
`GeoCatalog.query` with the window as the interval, or `matchup` with
`WithinWindow(start, end)` when many analyses are matched at once. Workstream
B2 adds `lineage`; the window stays an existing call.

(r13-stage-6)=
### Stage 6 · Many producers on many grids: a CMIP-style comparison (geocatalog → xrtoolz)

**TL;DR.** Compare models that run on different grids, either by computing
each model's statistics on its own grid or by regridding all of them to one
grid first, and see when the two give different answers.

**Problem.** A multi-model ensemble such as CMIP6 [@eyring2016] holds runs
from many models $k \in \mathcal{K}$. In CMIP6 terms the model is the
`source_id`, a run is a `member_id` (a variant label such as `r1i1p1f1`),
and each model writes on its own grid (`grid_label`). Model $k$ has $N_k$
members and a grid of cells $i$ with areas $a_i$; the field is $x_k(t, i)$
at day $t$. The archive here is synthetic: model A on a 1° grid with 3
members, B on 2.5° with 1, C on 2° with 2, 90 days each, with independent
cell-scale weather of standard deviation 2 K on a zonal climatology. The six
files are indexed like stage 1, with `producer` and `member` columns.

The area mean of a field $y$ on a grid with cells $i$ is
$\langle y \rangle = \sum_i a_i y_i / \sum_i a_i$. A **conservative regrid**
$\mathcal{R}$ onto a common grid with cells $j$ and areas $A_j$ weights each
source cell by its overlap $w_{ji} = |c_j \cap c_i|$, so that the area mean
survives whenever the common grid covers the source grid:

$$
\begin{aligned}
(\mathcal{R} y)_j &= \frac{\sum_i w_{ji} y_i}{\sum_i w_{ji}} , \\
\langle \mathcal{R} y \rangle &= \langle y \rangle ,
\end{aligned}
$$ (r13-eq-regrid)

A statistic $S$ over time and members, applied cell by cell, either commutes
with $\mathcal{R}$ or does not. The time mean is linear, so
$S(\mathcal{R} x) = \mathcal{R} S(x)$. The 99th percentile $q_{0.99}$ is
not: averaging cells before taking the percentile removes the cell-scale
extremes that the percentile measures. That gives three pipelines:

$$
\begin{aligned}
\text{(1) native: } & \langle S(x_k) \rangle \text{ on model } k \text{'s grid} , \\
\text{(2) regrid first: } & \langle S(\mathcal{R} x_k) \rangle , \\
\text{(3) hybrid: } & \langle \mathcal{R} S(x_k) \rangle = \langle S(x_k) \rangle ,
\end{aligned}
$$ (r13-eq-pipelines)

where (3) equals (1) by {eq}`r13-eq-regrid`, and (2) equals (1) only when
$S$ is linear. Pipeline (3) is the usual way to draw a multi-model map: each
statistic is computed where the model resolved it, and only the result is
regridded. Combining the models needs one more choice: one vote per model,
$\bar{x} = |\mathcal{K}|^{-1} \sum_k \langle S(x_k) \rangle$, or one vote per
member, $\sum_k N_k \langle S(x_k) \rangle / \sum_k N_k$.

```{code-block} python
:caption: Three models on three grids, loaded as a DataTree and compared three ways, from a real run.
:linenos:

# Shapes: K = 3 models; model k: (N_k, T = 90, H_k, W_k), H_k × W_k its own grid
from xrtoolz.interpolate import RegridConservative

# one leaf per producer, members stacked inside it; the grids differ
leaves = {}
for k, rows in cat.gdf.groupby("producer"):                             # user
    runs = [xr.open_dataset(p).load() for p in rows.filepath]           # user
    leaves[f"/{k}"] = xr.concat(runs, dim="member")                     # user
tree = xr.DataTree.from_dict(leaves)                                    # today
# → A (3, 90, 180, 360) · B (1, 90, 72, 144) · C (2, 90, 90, 180)

# 𝓡: conservative regrid of every leaf onto the coarsest grid (2.5°)
regrid = RegridConservative(grid_25, mode="mean", normalize="destarea")  # today
common = regrid(tree)                       # every leaf → (N_k, 90, 72, 144)  today

# ⟨y⟩ = Σᵢ aᵢ yᵢ / Σᵢ aᵢ, with aᵢ ∝ sin φᵢ₊ − sin φᵢ₋
def area_mean(y): ...                                                   # user

# S = time mean (linear): (1) and (2) agree
# → native    A 277.9999 · B 279.4971 · C 277.2991 K
# → regridded A 277.9999 · B 279.4971 · C 277.2991 K   (max difference 1.2e-11)

# S = q₀.₉₉ − time mean (nonlinear), per cell, then ⟨·⟩
# → (1) native        A 4.289 · B 4.279 · C 4.288 K
# → (2) regrid first  A 1.544 · B 4.279 · C 2.564 K
# → (3) hybrid        A 4.289 · B 4.279 · C 4.288 K    (equals (1) to 1.2e-11)

# one vote per model vs one vote per member, S = time mean
# → 278.2654 K vs 278.0159 K
```

The tree, the regrid and the map over models all ship today: xrcore's
`Operator` maps any xrtoolz operator over the leaves of an `xr.DataTree`,
and `RegridConservative` keeps area means to 1.2e-11 K across a 1° → 2.5°
and a non-nested 2° → 2.5° regrid. The numbers show why the order of
operations matters. Regridding first leaves the time mean untouched but
removes 64 % of model A's warm extremes and 40 % of model C's, so the fine
models look less extreme than the coarse one only because they were
averaged. Model B is unchanged because it is already on the common grid.
Two things are user code: grouping the catalog into leaves, which workstream
B1 adds as `load_states(..., group_by="producer")`, and the area mean, which
`geo.Reduce` cannot weight (workstream C3). The vote is a choice the script
must state: here it moves the ensemble mean by 0.25 K.

Comparing each model with observations runs the other way: the
observations are regridded onto **each** model's native grid, so that the
statistic is computed where the model resolved it. `RegridConservative` and
`RegridLike` take their target grid when they are built, so today that is
one operator per leaf, built in a loop (`regrid_conservative(obs, leaf)`
inside `xr.map_over_datasets` does the same). This is also the step where a
pipekit `Graph` earns its place: two inputs, the model leaf and the
observations, feed one bias operator.

(r13-names)=
## The scoping report's names, checked

:::{table} Every stack name the scoping report's examples use, at the pinned revisions.
:label: tbl-r13-names

| Name in the report | At the pinned revisions |
|---|---|
| `statecatalog`, `StateSlice`, `StateRow`, `StateCatalog` | not built; named only in pipekit-train's design docs (`architecture.md`, `api/datasets.md`) |
| `InMemoryStateCatalog`, `DuckDBStateCatalog` | not built; geocatalog's `InMemoryGeoCatalog` and `DuckDBGeoCatalog` hold extra columns and filter them with `where` and `sql` |
| `CycleStateWriter`, `CycleStateReader` | not built; see [](#r13-ws-d) |
| `pk.config_hash(op)` | does not exist; `pipekit.hashing.sha256_hex(stable_json(op.get_config()))` computes it, and [Report 9](09_use_cases.md#r9-proposals) proposes `pipeline_hash(op)` (proposal B) |
| `georeader.read_from_url` | does not exist; states are xarray files, read with `xr.open_dataset` or geocatalog's `load_xarray` |
| `pc.Sequential` | does not exist. `pipekit.Sequential` threads a carry state when it holds stateful steps, but it is not a `StatefulOperator`, so `Cycle(step_op=pk.Sequential([...]))` raises `TypeError: step_op must be a StatefulOperator or a ForwardModel` even then ([Report 10](#r13-ref-report-10)) |
| `pc.EnsembleCycle(step_op, n_steps, n_members)` | exists, with that signature |
| `pt.SimulationDataset` | exists, but simulates on the fly from a forward model and a prior; it reads no stored states |
:::

(r13-audit)=
## Current-state audit

Eleven pieces of the proposed package were checked: seven are healthy as
they ship, three are split between a library that has them and a gap, and
one is missing. The audit read the revisions pinned in
`references/` on 13 October 2026.

:::{table} What each piece of the proposed statecatalog maps to today.
:label: tbl-r13-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| bounds, CRS, valid-time interval; `query`, `GeoSlice` | `geocatalog` (`InMemoryGeoCatalog.query`, `DuckDBGeoCatalog.query`) | every catalog workflow | **Healthy.** Answers "valid at $t$" at file granularity |
| extra columns; `CatalogRow.extras`; `where`, `sql` | `geocatalog._src.base`, `backends.memory`, `backends.duckdb_backend` | STAC `extra_properties`, user filters | **Healthy.** Holds $r$, $m$, $h$, $p$, but untyped and unnamed |
| xarray builder; `build_xarray_catalog(time_var=...)` | `geocatalog._src.formats.xarray_backend` | xarray-kind catalogs | **Split ownership.** One row per file; reads no forecast metadata |
| loader; `load_xarray` | same module | xarray-kind catalogs | **Split ownership.** Right for observations; for forecasts, drops overlapping runs ([](#r13-stage-3)) |
| schema versioning; `SCHEMA_VERSION_CURRENT = 0`, `parquet._MIGRATIONS` | `geocatalog._src._schema`, `storage.parquet` | GeoParquet round trip | **Healthy.** No migration registered yet; a reserved column is the first |
| temporal matching; `NearestInTime`, `WithinWindow`, `Synchronous`, `matchup` | `geocatalog._src.matchup` | matchup tables | **Healthy.** Covers the assimilation window, on row midpoints |
| lead-time skill; `SkillByLeadTime`, `skill_by_lead_time` | `xrtoolz.metrics._src.forecast` | `viz.validation.LeadTimeSkillPanel`, `FacetPanel` | **Healthy.** Needs a reference aligned on `lead_dim`, or none on it |
| ensemble scores; `CRPS`, `SpreadSkillRatio`, `RankHistogram`, `EnsembleCoverage` | `xrtoolz.metrics` (`ensemble_dim="member"`) | | **Healthy.** Pixel metrics such as `RMSE` cannot reduce over `member` when the reference lacks it |
| trajectories; `Cycle.history`, `DACycle.history` | `pipekit_cycle.cycle`, `pipekit_cycle.da` | | **Split ownership.** In memory; `EnsembleCycle`, `EnsembleDACycle`, `SmootherCycle` and `WindowedCycle` keep none |
| producer identity; `Run.pipeline_hash`, `ModelRegistry.store`, `sha256_hex` | `pipekit_experiment.run`, `.registry`; `pipekit.hashing` | `TrainingArtifact`, `InferenceArtifact` | **Healthy.** The hash a row's `producer` column records |
| lineage, forecast chains, typed state queries | | | **Missing.** Nothing anywhere |
:::

Smaller findings:

- **A plain `Cycle` does not advance time.** `_ForwardModelAdapter` passes
  the carry state through unchanged, so `DAState.t` stays 0.0. Only the DA
  cycles call `_advance_da_state`. A history entry's valid time must be
  computed from its index: entry $k$ with stride $s$ is valid at
  $r + (ks + 1) \Delta t$ (checked: a 4-step `Cycle` with stride 2 keeps the
  states after steps 1 and 3).
- **`EnsembleCycle` threads one carry state through all members.** Its loop
  calls `step_op(member_j, state)` member by member, so a step that counts
  calls cannot tell members apart. This is one reason to write from the
  history (workstream D) rather than from inside the step.
- **`DAState.t` is a float with no epoch.** There is no run time or calendar
  anywhere in pipekit-cycle; the script that writes files supplies $r$.
- **`CatalogDataset` is documented but not built.** pipekit-train's
  `docs/design/api/datasets.md` already says its catalog can be "A
  `geocatalog.GeoCatalog` (or StateCatalog) reference". With this redesign
  the two are the same object.
- **xrreader has no forecast products.** Its CDS presets are ERA5 reanalyses
  (`product_type="reanalysis"`) and its CMEMS presets are reanalyses and NRT
  observations. `_TIME_COORDS` in `cds/source.py` includes
  `forecast_reference_time` only to trim on time. The download cache maps a
  request to a path; it keeps no index.
- **pipekit-evaluate has the level but not the scoring.** `LEVEL_NAMES["L4"]`
  is `"reanalysis_forecast"`, and references come from `Task.datasets()`.
  No lead-time or CRPS scoring ships there; xrtoolz's does.

(r13-options)=
## Options considered

:::{table} Four ways to index model states.
:label: tbl-r13-options

| Option | What it is | Cost | Verdict |
|---|---|---|---|
| 1 · `statecatalog` package | the scoping report: a second slice, Protocol, backends and parquet format | about 1,100 lines, 2.5 to 3 weeks; two indexes to learn and keep in step | not recommended: duplicates geocatalog |
| 2 · A state profile in geocatalog | reserved columns, a builder, `load_states` and queries in `geocatalog.states` | an estimated 400 lines over the existing backends | **recommended** |
| 3 · One Zarr cube per producer, no catalog | a single store with `init_time`, `member` and `lead_time` dimensions | none in the libraries | fine for one producer writing one store; no answer for many files, producers or lineage. Workstreams B1 and C produce exactly this cube in memory |
| 4 · An `xr.DataTree` of producers | one tree leaf per producer, members inside it | none: xrcore operators already map over tree leaves | a way to hold a loaded result, not an index; **adopted** as `load_states(group_by="producer")` for producers on different grids ([](#r13-stage-6)) |
:::

**Recommended:** option 2. **Alternative:** option 3 for a single producer
with one store, which needs only workstream C.

(r13-ws-a)=
## Workstream A · State columns in geocatalog

**Decision.** The state fields are reserved row columns of geocatalog's
schema, not fields of a new slice.

:::{table} Workstream A's moves.
:label: tbl-r13-move-a

| Symbol | Destination | Reason |
|---|---|---|
| `run_time`, `member`, `producer`, `parents` | reserved columns, geocatalog schema v1 (`_schema.py`, `parquet._MIGRATIONS`) | named and typed once: `datetime64[us]`; `string` (a CMIP variant label such as `r1i1p1f1`, or a CF realization number written as text; null for a deterministic run); `string`; `list<string>` (one URI for a forecast, every forecast member for an ensemble analysis) |
| `build_state_catalog(paths, *, crs, names)` | `geocatalog.build` | fills the four columns from file metadata; one row per file |
| `CF_NAMES`, `CFGRIB_NAMES` | `geocatalog.states` | where each field is read from: CF attributes and coordinates, or cfgrib's `time`, `step`, `valid_time` and `number` |
:::

### A1 · Reserve the columns

1. **Register the migration.** Bump `SCHEMA_VERSION_CURRENT` to 1 and
   register a v0 → v1 function in `parquet._MIGRATIONS` that adds the four
   columns as nulls. A v0 catalog then reads as a catalog of states with no
   state fields.
2. **Add the names to the reserved set.** `CatalogRow.extras` stops carrying
   them, and a `CatalogRow.state` property returns them typed.
3. **Document them** in `docs/catalog/schema-versions.md`, the place the
   schema module points to.

Lead time is not stored: a row covers $[v_{\min}, v_{\max}]$, so its leads
are $[v_{\min} - r, v_{\max} - r]$, derived when needed.

### A2 · Build the columns from metadata

```{code-block} python
:caption: Stage 1's index, before and after workstream A.
:linenos:

# Shapes: 11 files → 11 rows
# Today:
cat = build_xarray_catalog(paths, crs="EPSG:4326")                     # today
attrs = [xr.open_dataset(p).attrs for p in cat.gdf.filepath]           # user
cat.gdf["run_time"] = pd.to_datetime([a["run_time"] for a in attrs])   # user
# … member, producer, parents the same way                            # user

# After:
from geocatalog.build import build_state_catalog
from geocatalog.states import CF_NAMES
cat = build_state_catalog(paths, crs="EPSG:4326", names=CF_NAMES)  # proposed (A2)
cat.gdf[["run_time", "member", "producer", "parents"]]            # proposed (A1)
```

`names` maps each field to where it is found: a coordinate, a variable, a
global attribute or a callable on the path (for archives that encode the run
in the file name, such as `fc_20260102_m0.nc`). `producer` and `parents`
have no CF name, so they come from attributes the script writes. The member
column is a string, not the integer of stage 1's glue: a CF realization is a
number, but a CMIP member is a variant label such as `r1i1p1f1` whose four
indices (realization, initialization, physics, forcing) all distinguish runs,
so an integer column would lose them. Stage 1's members become `"0"` and
`"1"`, and its analyses, written with `member=-1`, get a null.

(r13-ws-b)=
## Workstream B · Run-aware loading and typed queries

**Decision.** `geocatalog.states` holds the loader and the queries that know
what a run is; the observation loader stays as it is.

:::{table} Workstream B's moves.
:label: tbl-r13-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `load_states(catalog, slice_, *, data_vars)` | `geocatalog.states` | stacks one file per (run, member) into an (`init_time`, `member`, `lead_time`, y, x) cube with a `valid_time` coordinate; never drops overlapping runs |
| `forecast_chain`, `latest_analysis`, `ensemble_at`, `lead_window`, `at_time` | `geocatalog.states` | typed filters on the reserved columns; `pd.Timestamp` and `pd.Timedelta` arguments, so no query strings |
| `load_states(..., group_by="producer")` | `geocatalog.states` | returns an `xr.DataTree` with one leaf per producer and members stacked in each; stage 6's grouping loop. Without `group_by`, rows on different grids raise instead of being interpolated silently |
| `lineage(catalog, uri, *, depth)` | `geocatalog.states` | the walk over `parents`, breadth first, with a depth limit; returns every ancestor once |
| `assimilation_window` | not added | it is `query` with an interval, or `matchup` with `WithinWindow` |
:::

:::{table} Where every new name in workstreams A to C lives, and what it returns.
:label: tbl-r13-placement

| Object | Lives in | Name | Kind | Event shape |
|---|---|---|---|---|
| State builder | `geocatalog.build` | `build_state_catalog` | function | a catalog: one row per file |
| Run-aware loader | `geocatalog.states` | `load_states` | function | `xr.Dataset` (R, N_e, L, H, W) |
| Grouped loader | `geocatalog.states` | `load_states(group_by="producer")` | function | `xr.DataTree`, leaf $k$: (N_k, T, H_k, W_k) |
| Forecast chain | `geocatalog.states` | `forecast_chain` | function | a catalog: one run, one member |
| Latest analysis | `geocatalog.states` | `latest_analysis` | function | one `CatalogRow`, or `None` |
| Members of a run | `geocatalog.states` | `ensemble_at` | function | a catalog: one run, all members |
| Lead window | `geocatalog.states` | `lead_window` | function | a catalog: rows whose leads overlap $[\tau_0, \tau_1]$ |
| Lineage | `geocatalog.states` | `lineage` | function | a list of `CatalogRow`, newest first |
| Reference alignment | `xrtoolz.metrics` | `align_reference` | function | `xr.Dataset` (R, L, H, W) |
| Area-weighted reduction | `xrtoolz.geo` | `Reduce(..., weights="cell_area")` | `Operator` option | the input without the reduced dims |
:::

### B1 · Load without mixing runs

`load_states` is stage 3's eight user lines, with a check that each
(run, member) pair appears once. Members missing from a run come back as NaN,
not as a shorter cube. With `group_by="producer"` it is stage 6's grouping
loop instead: one leaf per producer, each on its own grid, so that every
xrtoolz operator maps over the models unchanged. The return type follows the
argument, never the data: a plain call on rows with different grids raises
rather than returning a tree by surprise.

### B2 · Typed queries

```{code-block} python
:caption: Stages 2, 3 and 5, before and after workstream B.
:linenos:

# Shapes: R = 3, N_e = 2, L = 3, H = W = 4
import geocatalog.states as gs

# Today:
cat.where("kind == 'forecast' and member == 0"
          " and run_time >= '2026-01-02' and run_time < '2026-01-03'")  # today

# After:
gs.forecast_chain(cat, run_time=pd.Timestamp("2026-01-02"), member=0)
# → 1 row, fc_20260102_m0.nc                                       proposed (B2)
gs.latest_analysis(cat, before=pd.Timestamp("2026-01-03 06:00"))
# → ana_20260103.nc                                               proposed (B2)
gs.ensemble_at(cat, run_time=pd.Timestamp("2026-01-01"))  # 2 rows  proposed (B2)
fc = gs.load_states(gs.forecasts(cat), aoi)    # → (3, 2, 3, 4, 4)  proposed (B1)
gs.lineage(cat, "ana_20260104.nc")             # → 7 rows           proposed (B2)
```

`kind` (analysis or forecast) is not a reserved column: an analysis is a row
whose leads are all zero, $v_{\min} = v_{\max} = r$, so
`latest_analysis` can tell without one. Whether to reserve `kind` anyway is
open question 2.

:::{warning} Differs from the plan
The scoping report's queries returned "a subset `StateCatalog`" or a list of
`StateSlice`s. Here they return geocatalog catalogs and rows, so every
existing geocatalog call (`query`, `matchup`, `stage`, `to_geoparquet`,
`CatalogBundle`) works on states with no bridge module. The scoping report's
`geocatalog → state` bridge, `states_that_used(obs_slice)`, needs the
assimilated observations recorded per analysis; it goes to
[](#r13-open-questions).
:::

(r13-ws-c)=
## Workstream C · Forecast alignment in xrtoolz

**Decision.** xrtoolz builds the verifying reference and broadcasts it;
the metrics stay as they are.

:::{table} Workstream C's moves.
:label: tbl-r13-move-c

| Symbol | Destination | Reason |
|---|---|---|
| `align_reference(forecast, reference, *, init_dim, lead_dim, time_dim)` | `xrtoolz.metrics` (beside `skill_by_lead_time`) | selects $A[r, \tau, x] = a(r + \tau, x)$; NaN where the reference has no valid time |
| reference broadcasting in the pixel metrics | `xrtoolz.metrics._src` (the `apply_ufunc` wrappers) | `RMSE(dims=("member", …))` raises `ValueError` when the reference lacks `member`; broadcasting first gives the per-member score |
| `forecast_reference_time` → `lead_time` | `align_reference` (accepts either layout) | CF files store $r$ and $\tau$, or $r$ and $v$; nothing converts today |
| `weights="cell_area"` | `xrtoolz.geo.Reduce` | stage 6's area mean is user code: `Reduce` calls the plain `ds.mean`. Exact spherical cell areas from the latitude bounds, as `regrid_conservative` already computes them |
:::

```{code-block} python
:caption: Stage 4, before and after workstream C.
:linenos:

# Shapes: F (R, N_e, L, H, W); a (T, H, W)
# Today:
ref = an.sel(time=fc.valid_time).drop_vars("time")    # (R, L, H, W)     user
SkillByLeadTime(RMSE("xch4", dims=("member", *dims)))(*xr.broadcast(fc, ref))  # user

# After:
from xrtoolz.metrics import align_reference
# A[r, τ, x] = a(r + τ, x)
ref = align_reference(fc, an, init_dim="init_time", lead_dim="lead_time",
                      time_dim="time")               # → (R, L, H, W) proposed (C1)
SkillByLeadTime(RMSE("xch4", dims=("member", *dims)))(fc, ref)  # (L,)  proposed (C2)
# → [0.109, 0.617, 1.132]
```

### C3 · Area-weighted reductions

```{code-block} python
:caption: Stage 6's area mean, before and after C3.
:linenos:

# Shapes: leaf k: (N_k, T, H_k, W_k) → (N_k, T)
# ⟨y⟩ = Σᵢ aᵢ yᵢ / Σᵢ aᵢ, aᵢ ∝ Δλ (sin φᵢ₊ − sin φᵢ₋)
# Today:
geo.Reduce("mean", dim=("lat", "lon"))(tree)   # unweighted: wrong on a sphere  today
def area_mean(y): ...                          # sin-latitude weights            user

# After:
geo.Reduce("mean", dim=("lat", "lon"), weights="cell_area")(tree)  # proposed (C3)
```

Because `Reduce` is an xrcore `Operator`, the weighted form maps over the
tree like every other operator, and each leaf is weighted by its own cell
areas.

(r13-ws-d)=
## Workstream D · Ensemble histories in pipekit-cycle

**Decision.** The cycle's `history` is the write seam; files are written by
the script, after the call.

:::{table} Workstream D's moves.
:label: tbl-r13-move-d

| Symbol | Destination | Reason |
|---|---|---|
| `save_history`, `history_stride`, `history` | `pipekit_cycle.EnsembleCycle` (as on `Cycle`: one `(members, state)` pair per kept step, where `members` is the list of $N_e$ carriers) | the ensemble forecast is the case the scoping report wrote for, and it keeps no trajectory |
| `save_history`, `history` | `pipekit_cycle.EnsembleDACycle` (as on `DACycle`: one (forecast, analysis) pair per cycle, here of member lists) | the overview's methane walkthrough runs an `EnsembleDACycle` and indexes its analyses |
| writing one file per (run, member) | geostack `scripts/` | AGENTS.md: side effects live in scripts; the library computes |
| `CycleStateWriter`, `CycleStateReader` | not added | a writer inside `step_op` cannot tell members apart and has no clock ([](#r13-audit)); reading an initial state is `latest_analysis` and `xr.open_dataset` |
:::

```{code-block} python
:caption: Writing an ensemble forecast, after workstream D. The script owns the files.
:linenos:

# Shapes: T steps, N_e members, H × W cells; Δt the model step
fcst = pc.EnsembleCycle(step_op=model, n_steps=T, n_members=N_e,
                        save_history=True)                       # proposed (D1)
members, _ = fcst([x_a] * N_e, pc.DAState())                     # today
# v_k = r + (k + 1) Δt for history entry k (stride 1)
for m in range(N_e):                                             # user (script)
    # history entry h = (members, state), so h[0][m] is member m's carrier
    ds = to_dataset([h[0][m] for h in fcst.history], r, dt)      # arguments sketched
    ds.attrs |= {"forecast_reference_time": str(r), "realization": m,
                 "producer": h_hash, "parents": [x_a_uri]}      # user (script)
    ds.to_zarr(root / f"fc_{r:%Y%m%dT%H}_m{m}.zarr")             # user (script)
cat = build_state_catalog(sorted(root.glob("fc_*.zarr")),
                          crs=crs, names=CF_NAMES)               # proposed (A2)
```

`to_dataset` is the model's own carrier-to-xarray step, which differs per
model; it is not a library function. `h_hash` is the producer:
`sha256_hex(stable_json(fcst.get_config()))` today, or Report 9's proposed
`pipeline_hash`.

:::{warning} Differs from the plan
The scoping report made **pipekit-cycle the primary writer and reader** of
the state catalog, through `CycleStateWriter` placed inside the cycle's step
(`pc.Sequential([chemistry_forward, writer])`). At the pinned revisions that
composition raises `TypeError`, pipekit-cycle has no per-step hook, and its
states carry no run time. Writing from `history` after the call needs one
flag on each ensemble cycle and keeps the cycle free of I/O. The cost is memory:
the whole trajectory is held until the script writes it (see
[](#r13-risks)).
:::

(r13-gates)=
## Testing gates

Each workstream merges only when its gates pass. The fixtures are the
running example's eleven files.

:::{table} One gate per proposal, each with a pass condition.
:label: tbl-r13-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Schema round trip | A1 | a v1 catalog round-trips through GeoParquet with `run_time` as `datetime64[us]` and `member` as `string` (a CMIP variant label `r1i1p1f1` and a null both survive) and `parents` as `list<string>`; a v0 file reads with the four columns all null, and every other column unchanged |
| A2 Builder | A2 | `build_state_catalog` on the 11 files fills 11 of 11 rows, equal to stage 1's hand-filled columns; with `CFGRIB_NAMES` on a cfgrib-decoded file it reads $r$, $\tau$ and $m$ |
| B1 No dropped runs | B1 | `load_states` on the 6 forecast files returns sizes (3, 2, 3, 4, 4), equal to stage 3's hand-stacked cube with a maximum absolute difference of 0; the slice for 3 January returns 3 runs (today: 1) |
| B2 Typed queries | B2 | on the archive, `forecast_chain` returns 1 row, `latest_analysis` `ana_20260103.nc`, `ensemble_at` members `["0", "1"]`, and `lineage` 7 rows in stage 5's order; `forecast_chain` with an exact `pd.Timestamp` returns the row the string query misses |
| C1 Alignment | C1 | `align_reference` equals `an.sel(time=fc.valid_time)` exactly, and `SkillByLeadTime(RMSE)` on it reproduces [0.076, 0.467, 0.746] to 1e-12; a lead past the last analysis is NaN |
| C2 Broadcasting | C2 | `RMSE` over `("member", "init_time", "lat", "lon")` with a member-less reference equals the `xr.broadcast` result [0.109, 0.617, 1.132] to 1e-12; results for inputs that already share dimensions are unchanged to 0 |
| B3 Producers as leaves | B1 | `load_states(group_by="producer")` on stage 6's 6 files returns 3 leaves of sizes (3, 90, 180, 360), (1, 90, 72, 144) and (2, 90, 90, 180), equal to stage 6's hand-grouped tree to 0; the same call without `group_by` raises |
| C3 Area weights | C3 | `Reduce("mean", weights="cell_area")` equals the sin-latitude weighted mean to 1e-12 on each leaf, and the global mean of a constant field is that constant to 1e-12 on every grid |
| C4 Conservation (existing) | — | `RegridConservative(mode="mean", normalize="destarea")` keeps each model's time-mean area mean to 1e-10 K; the hybrid pipeline equals the native one to 1e-10 K. It passes today: 1.2e-11 ([](#r13-stage-6)) |
| D1 Ensemble history | D1 | with `save_history=True`, `EnsembleCycle` keeps `len(history) == ceil(T / s)` `(members, state)` entries with `len(members) == N_e`, and `EnsembleDACycle` one (forecast, analysis) pair of $N_e$ members per cycle; with it off, outputs equal today's to 0 and `history` is empty |
:::

(r13-risks)=
## Sequencing and risks

The three phase-1 pieces are independent, so they ship side by side; the
builder, loader and queries follow because they use A1's columns.

:::{figure} ../images/reports/r13_roadmap.svg
:label: fig-r13-roadmap
:alt: Phase 1: A1 reserved state columns, C1 to C3 alignment, broadcasting and area weights, and D1 ensemble histories, side by side, then gates A1, C1 to C3 and D1. Phase 2: A2 the state builder and B1 and B2 the loader and queries, then gates A2, B1 to B3. Phase 3: the consumers.

Foundations first, then the state profile, then its consumers.
:::

- **The first schema migration.** A1 is geocatalog's first registered
  migration, and a mistake in it touches every saved catalog. Fallback:
  ship A2 and B with the four names as documented extra columns and no
  schema bump, and reserve them once the API has settled.
- **geocatalog keeps growing.** [Report 6](06_geocatalog.md) found it had
  already outgrown its scope. Fallback: if `geocatalog.states` passes about
  500 lines, move it to its own distribution in the geotoolz monorepo that
  depends on geocatalog, still with one index format.
- **Metadata conventions vary.** CF files, cfgrib-decoded GRIB, and
  archives that encode the run in the file name all store $r$ and $m$
  differently. Fallback: `names` accepts a callable on the path, so any
  archive can be indexed without changing the builder.
- **One row per file is coarse.** A file per (run, member) answers
  "valid at $t$" at file granularity, and a per-step query must open the
  file. Fallback: when steps are separate files (common for GRIB), rows are
  per step without any change; a per-step explode option is open question 1.
- **Histories held in memory.** A 72-step, 40-member trajectory at 0.25°
  does not fit in memory. Fallback: run the cycle in chunks of steps from
  the last state and write after each chunk; a streaming hook on the cycles
  is open question 3.
- **Pandas query semantics.** Users who keep writing `where` strings still
  hit stage 2's silent miss. Fallback: B2's functions use boolean masks, and
  the `where` docstring gets a warning about datetime equality.
- **Curvilinear model grids.** `regrid_conservative` works along two
  rectilinear dimensions, and many CMIP ocean models use curvilinear or
  tripolar native grids (`grid_label="gn"`). Fallback: use the regridded
  products the modelling centres publish (`grid_label="gr"`), or regrid
  those models with an external tool such as xESMF in the script; open
  question 5.

(r13-open-questions)=
## Open questions

1. **One row per file, or per step?** Per file keeps catalogs small (40 rows
   for a 40-member run instead of the scoping report's 2,880). Per step
   answers "valid at $t$" without opening files. Should `build_state_catalog`
   offer an `explode="step"` option?
2. **Reserve `kind`?** Analyses can be told apart by zero leads. A
   reanalysis written as a run with $\tau > 0$ cannot. Is a fifth column,
   `kind`, worth it?
3. **A streaming hook on the cycles.** Should `Cycle` and `EnsembleCycle`
   take an `on_step(k, carriers, state)` callable, so scripts can write while
   the cycle runs? It would also give `SmootherCycle` and `WindowedCycle` a
   history.
4. **Recording assimilated observations.** The scoping report's
   `states_that_used(obs_slice)` needs each analysis to record which
   observation rows it used. A `CatalogBundle` matchup table between the
   analysis and observation catalogs would do it; is that the right home?
5. **Curvilinear grids.** Should xrtoolz grow a conservative regrid for
   curvilinear grids (cell polygons rather than two 1-D weight matrices), or
   should the stack defer to xESMF for them?
6. **Targets that differ per leaf.** `RegridConservative` and `RegridLike`
   fix their target grid when built, so regridding one observation field onto
   each model's own grid needs one operator per leaf. Should they accept the
   target as a second input, so the step maps over a tree of targets?

(r13-todos)=
## Upcoming work

:::{table} Follow-ups from the reassessment.
:label: tbl-r13-todos

| Follow-up | Where | Why |
|---|---|---|
| Build `CatalogDataset` over `load_states` | pipekit-train | documented in `api/datasets.md`, not built; it is the emulator-training consumer the scoping report wanted from `SimulationDataset` |
| Score L4 references from a state catalog | pipekit-evaluate | `LEVEL_NAMES["L4"]` exists; `Task.datasets()` could return `load_states` output as its `truth_key` |
| Add forecast presets with run and lead dimensions | xrreader (CDS, CMEMS forecast products) | its presets are reanalyses and observations; downloaded forecasts would be indexed with `build_state_catalog` |
| Replace `statecatalog` in pipekit-train's design docs | pipekit-train `docs/design/architecture.md`, `api/datasets.md` | they list it as a sibling package |
| Warn about datetime equality in `where` | geocatalog `InMemoryGeoCatalog.where` docstring | stage 2's silent miss on pandas 3.0.6 |
:::

(r13-references)=
## References

<!-- TODO(ref): placeholders. When a report page lands, point its links in this
file at the page and delete its entry below. -->

:::{note} Placeholders
:class: simple
Report 10 (pipekit-cycle) and the reports on the other pipekit sister packages
are still being written. Reports [3](03_sister_libraries.md),
[6](06_geocatalog.md) and [9](09_use_cases.md) are published.
:::

### Forthcoming reports

(r13-ref-report-10)=
Report 10 · pipekit-cycle
: *Forthcoming.* The cycles, DA cycles and states that produce model states.

(r13-ref-report-11)=
Reports 11 and 12 · pipekit-train, pipekit-evaluate, pipekit-experiment
: *Forthcoming.* The consumers of a state catalog and the producer hash.

### Sources

- The scoping report, "Report 13 — `statecatalog`: catalog for model
  states" (supplied by the author).
- The pinned code in `references/`: geocatalog (`geotoolz-catalog`), xrtoolz
  and xrreader, pipekit-cycle, pipekit-train, pipekit-experiment and
  pipekit-evaluate.
- CF conventions, [standard name table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
  (`forecast_reference_time`, `forecast_period`, `realization`).
- The running example's script and outputs, described in [](#r13-running).
