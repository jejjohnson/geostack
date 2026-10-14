---
title: Package boundaries for plume and radiative-transfer code
subtitle: GeoModels · What moves from geotoolz and plumax to plumax and a new radtrax, and why the package graph stays acyclic
short_title: Package boundaries
description: Plume post-processing lives in geotoolz, and radiative transfer, the matched filter and a 3D-Var retrieval live in plumax. This page proposes moving plume code to plumax and radiative-transfer code and retrieval algorithms to a new radtrax, checks each move against the imports and package metadata, and shows the resulting dependency graph has no cycle unless geotoolz keeps re-exports.
keywords:
  - radtrax
  - plumax
  - geotoolz
  - package boundaries
  - dependency graph
  - matched filter
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

Methane code is split today between two places that do not match what it is.
geotoolz, a GeoTensor operator library, holds plume post-processing (masks,
footprints, the integrated mass enhancement, fluxes) and a matched filter.
plumax, a JAX dispersion library, holds band-integrated radiative transfer, a
HITRAN look-up-table pipeline, a 3D-Var retrieval and a second matched
filter. The proposal moves **plume code to plumax** and **radiative transfer
and retrieval algorithms to a new radtrax**. Read from the code, the move is
clean. plumax's radiative-transfer group (radtran, hapi_lut, assimilation and
matched_filter, 6,100 of its 18,880 lines) imports nothing else in plumax.
geotoolz's `plume` and `matched_filter` are called by nothing else in
geotoolz. The package graph after the moves has no cycle, and every edge
between stacks points from GeoModels down to Numerics and GeoStack. It gets
one cycle (geotoolz, plumax, radtrax) if geotoolz keeps re-exports of what
moved, so it must not. **Three matched filters, two spectral-response
implementations and two air-mass factors become one each.**

+++

:::{note} About this page
:class: dropdown

| | |
|---|---|
| **Status** | Proposal. Nothing has moved; radtrax does not exist yet |
| **Reading time** | about 15 minutes |
| **Audience** | anyone deciding where a plume, radiative-transfer or retrieval function goes, and anyone reviewing a dependency change in the GeoModels packages |
| **Companions** | [radtrax · 1](radtrax/01_column_retrieval.md) (the retrieval radtrax implements); [radtrax · 2](radtrax/02_mbmp.md) (what `plume.SBMP` computes); [radtrax · 3](radtrax/03_design.md) (radtrax's design); [plumax · 1](plumax/01_emission_rate.md) (the plume operators that move to plumax); [Report 16](../reports/16_geo_data_stack.md) (the same kind of realignment for pipekit, xrtoolz and geotoolz, and one methane scene through these operators); [Report 9](../reports/09_use_cases.md#r9-running) (the use cases that call them); the [overview](../overview.md) (the three stacks) |
| **Code checked against** | geotoolz 0.10.2 at `088475e`, xrtoolz 0.0.5 at `308c7f2`, pipekit 0.0.2 at `ef3c17f` and georeader-spaceml 2.4.1, from `references/`; plumax 0.1.2 at `28f257e`, gaussx 0.6.5 at `7fa1a38`, finitevolX 0.0.47 at `32ffb45` and spectraldiffx 0.1.1 at `b5f3dc4`, from clones (they are not submodules yet). Read on 14 October 2026 |
| **Running example** | every number and figure is from one run of `notebooks/geomodels/package_boundaries.py`, which reads source files and `pyproject.toml` metadata only, imports and installs nothing, and asserts each number the page shows |
:::

(pb-today)=
## Where the code is today

[](#tbl-pb-today) is the inventory the radtrax and plumax pages point to:
each module, its size and what it holds.

:::{table} Plume, radiative-transfer and retrieval code in the stack, by package.
:label: tbl-pb-today

| Package | Module | Lines | What it holds |
|---|---|---|---|
| geotoolz | `plume` | 1,724 | `SBMP` (an MBMP without scale factors, [radtrax · 2](radtrax/02_mbmp.md#rx2-stack)), `PlumeMask`, `PlumeContours`, `PlumeFootprint`, `PlumeShapeFilter`, `PlumeColumnStats`, `PlumeQNDFeatures`, `WindAdvectionCone`, `IMEEstimate`, `CrossSectionalFlux`, `ColumnToMass`, on GeoTensors |
| geotoolz | `matched_filter` | 2,622 | NumPy matched filter: backgrounds (empirical, shrunk, low-rank, adaptive window, cluster, streaming), targets from an observation model, `ColumnEnhancement` |
| plumax | `radtran` | 2,392 | band-integrated Beer–Lambert forward models (exact, Maclaurin, Taylor), `SpectralResponseFunction`, a normalised-brightness LUT, targets, a JAX matched filter, a gaussx Woodbury solve |
| plumax | `hapi_lut` | 1,001 | HITRAN cross-section tables through HAPI (behind `[hapi]`), Beer–Lambert transmittance, the air-mass factor |
| plumax | `assimilation` | 1,237 | 3D-Var methane retrieval: the radiance observation operator, background covariances, control transforms, three solvers, diagnostics |
| plumax | `matched_filter` | 1,470 | a second JAX matched filter, on gaussx operators, with scikit-learn background estimators |
| plumax | dispersion (`gauss_plume`, `gauss_puff`, `lagrangian`, `les_fvm`, `met`, `population`, `coupled`) | 12,307 | plume physics, the plume-to-radiance coupling, emission inversion |
:::

(pb-stage-1)=
## plumax's radiative-transfer group is closed

**Problem.** A group of modules can leave a package if it imports nothing
else in that package. The check reads every import in plumax, lazy ones
included, and keeps those between subpackages.

:::{figure} ../images/package_boundaries/plumax_modules.svg
:label: fig-pb-plumax
:alt: plumax's subpackages as boxes with their line counts. The top row, framed as becoming radtrax, holds hapi_lut, radtran, assimilation and matched_filter; matched_filter imports assimilation and assimilation imports radtran. Below, coupled imports radtran, gauss_plume and lagrangian; gauss_puff imports gauss_plume; les_fvm imports gauss_puff and lagrangian; met imports les_fvm; population imports lagrangian.

plumax's subpackage imports, from the notebook. The package root, which
re-exports four dispersion models and radtran, is not drawn.
:::

```{code-block} python
:caption: The import check on plumax, from a real run.
:linenos:

# Shapes: 11 subpackages, 18,880 lines
# E = {(s, t) : a module of s imports plumax.t, s ≠ t}
edges = {s: {t for m in imports_of(f) for t in [m.split(".")[1]]
             if m.startswith("plumax.") and t != s} for s in SUBS}       # user
# C = {radtran, hapi_lut, assimilation, matched_filter};  E ∩ (C × C̄) = ∅
out_of_C = {s: edges[s] - C for s in C}          # → all empty            user
into_C = {s: edges[s] & C for s in SUBS if edges[s] & C}
# → coupled: {radtran}, and the package root: {radtran}
```

The group imports only itself ([](#fig-pb-plumax)): `matched_filter` imports `assimilation`
(for the radiance observation model its targets use), which imports
`radtran`. Nothing in it touches the dispersion models. Two things outside
it import `radtran`: `coupled`, which turns a simulated plume into radiance,
and the package root's re-exports. After the move both import radtrax, which
is the direction the graph needs: plume physics uses radiative transfer, not
the other way round.

(pb-stage-2)=
## geotoolz's plume and matched-filter modules stand alone

**Problem.** A module can leave geotoolz if no other geotoolz module calls
it, and it can live elsewhere if what it imports from geotoolz is public.

:::{table} What `plume` and `matched_filter` import from the rest of geotoolz.
:label: tbl-pb-private

| Module | geotoolz modules it imports | Private (`_src`) |
|---|---|---|
| `plume` | `_src.bands`, `_src.config`, `_src.geo`, `_src.labels`, `_src.optional`, `_src.shape`, `_src.valid`, `_src.wrap`, `segment._src.array` | 9 of 9 |
| `matched_filter` | `_src.config`, `_src.fitted`, `_src.samples`, `_src.shape`, `_src.valid`, `_src.wrap` | 6 of 6 |
:::

Only the package root imports either module, for its re-exports;
`compositing`, `measure` and `segment` mention them in docstrings only. So
neither has a caller to break. Both depend entirely on geotoolz's private
helpers, though ([](#tbl-pb-private)): nodata handling (`invalid_values`, `wrap_filled`), carrier
wrapping (`wrap_like`), band names (`resolve_band`), geometry checks
(`require_projected_crs`), labelling and thresholds. Code that moves out
must not import `geotoolz._src`, so those helpers become public first, as
[Report 16](../reports/16_geo_data_stack.md#r16-ws-a)'s A3 did for pipekit's.

(pb-stage-3)=
## Implementations that exist more than once

:::{table} The same computation in several places, found by name in the code.
:label: tbl-pb-dupes

| Computation | Where | After the move |
|---|---|---|
| matched filter, pixel and image, and its SNR | geotoolz `matched_filter/_src/array.py`; plumax `matched_filter/core.py`; plumax `radtran/matched_filter.py` | one JAX core in radtrax, from plumax's `matched_filter` (it already dispatches on gaussx operators); GeoTensor wrappers behind `radtrax[geo]` |
| spectral response functions | georeader `reflectance.py` (`srf`, Gaussian, and the Sentinel-2 table); plumax `radtran/srf.py` (`SpectralResponseFunction`) | radtrax owns the band integration; georeader keeps the agency tables as data, read through `radtrax[geo]` |
| air-mass factor | plumax `hapi_lut/beers.py` (`air_mass_factor`); plumax `radtran/config.py` (`ObservationGeometry.air_mass_factor`) | one function in radtrax |
:::

Of the duplicates in [](#tbl-pb-dupes), the two plumax matched filters
share their formula:
`radtran.matched_filter` inverts a dense covariance, and
`matched_filter.core` dispatches on gaussx operators and is the one the
3D-Var derivation links to. geotoolz's NumPy filter and plumax's newer one
both carry empirical, shrunk, low-rank, cluster and streaming backgrounds. One
core keeps the estimators and the operator dispatch; the
[radtrax page](radtrax/01_column_retrieval.md#rx1-stage-5) runs geotoolz's
filter today, and [Report 16](../reports/16_geo_data_stack.md#r16-stage-2)
runs it chip by chip.

(pb-moves)=
## The moves

:::{table} What moves where. Everything not listed stays.
:label: tbl-pb-moves

| From | To | What | Dependency it adds |
|---|---|---|---|
| plumax `radtran`, `hapi_lut`, `assimilation`, `matched_filter` | radtrax | radiative transfer, cross-section tables, the 3D-Var retrieval, the matched filter | plumax → radtrax |
| geotoolz `matched_filter` | radtrax `[geo]` | GeoTensor carriers and the background estimators, merged into one core | radtrax → geotoolz, georeader, as an extra |
| geotoolz `plume.SBMP` | radtrax `[geo]` | an MBMP without scale factors ([radtrax · 2](radtrax/02_mbmp.md#rx2-stack)), beside the single-pass MBSP of [radtrax · 1](radtrax/01_column_retrieval.md#rx1-stage-3) | |
| geotoolz `plume` (the other ten operators) | plumax `[geo]` | masks, contours, footprints, shape filters, column statistics, wind cones, the integrated mass enhancement, cross-sectional flux, unit conversion | plumax → geotoolz, as an extra |
| geotoolz `_src` helpers used by the above | geotoolz public API | nodata, wrapping, bands, geometry, labels, thresholds | none |
:::

What stays, outside [](#tbl-pb-moves): georeader's readers and SRF tables (sensor knowledge),
geotoolz's generic spectral operators (`spectral.BandRatio`,
`spectral.ContinuumRemoval`), segmentation and measurement, xrtoolz's
column and unit conversions (`atm.gas.ch4`), and geotoolz-products'
Carbon Mapper readers, which read plume products rather than make them.

(pb-graph)=
## The package graph, today and after

**Problem.** The packages' declared dependencies, base and extras, form a
directed graph. A strongly connected component with more than one node is a
cycle: two packages that each need the other to install or import. The
check builds the graph from each `pyproject.toml` and runs Tarjan's
algorithm on it today, after the moves, and in a counterfactual where
geotoolz keeps re-exports of what moved, behind extras.

:::{figure} ../images/package_boundaries/package_graph.svg
:label: fig-pb-graph
:alt: Ten packages in four layers. Bottom: pipekit, georeader, gaussx. Next: pipekit-cycle, xrtoolz, geotoolz, spectraldiffx. Next: finitevolx and radtrax, which is new. Top: plumax. plumax depends on finitevolx and radtrax; radtrax on pipekit and gaussx, and on geotoolz through an extra; finitevolx on spectraldiffx; spectraldiffx on gaussx; geotoolz on pipekit and georeader; xrtoolz and pipekit-cycle on pipekit.

The stack's package graph after the moves, from the notebook. Solid arrows
are base dependencies, dashed ones extras; an arrow implied by a longer path
is not drawn.
:::

```{code-block} python
:caption: The cycle check on the package graph, from a real run.
:linenos:

# Shapes: 10 packages; edges from [project.dependencies] and the extras
# G = (V, E),  (a, b) ∈ E  ⇔  a declares b, as a base or an extra dependency
today = {n: deps(pyproject) for n, pyproject in META.items()}            # user
after = today | {"radtrax": ({"pipekit", "gaussx"},
                             {"geotoolz", "georeader-spaceml"}),
                 "plumax": (today["plumax"][0] | {"radtrax"}, {"geotoolz"})}
# cycles = strongly connected components with |S| > 1 (Tarjan)
sccs(today), sccs(after)                 # → [], []                      user
# geotoolz[plume] → plumax and geotoolz[methane] → radtrax, as re-exports
sccs(shim)                               # → [{geotoolz, plumax, radtrax}] user
# layers: pipekit, georeader, gaussx 0 · geotoolz, xrtoolz, spectraldiffx 1
#         finitevolx, radtrax 2 · plumax 3
```

The graph is acyclic today and after the moves ([](#fig-pb-graph)). Its longest chain is
plumax → radtrax → geotoolz → georeader. Every edge between stacks points
from GeoModels to Numerics (gaussx, finitevolx, spectraldiffx) or to GeoStack (pipekit, geotoolz,
georeader), and none points back. The one way to break that is the usual
way a move is softened: if geotoolz kept `geotoolz.plume` and
`geotoolz.matched_filter` as re-exports behind extras, then geotoolz would
need plumax and radtrax, and both need geotoolz through their `[geo]`
extras. That is a cycle of three packages, so the old paths raise an
`ImportError` naming the new home instead, as
[Report 16](../reports/16_geo_data_stack.md#r16-ws-d)'s D2 does for the
window dataset.

(pb-rules)=
## Ownership rules

1. **geotoolz never imports plumax or radtrax**, not even lazily behind an
   extra. The moved names leave with an `ImportError` that says where they
   went.
2. **radtrax never imports plumax.** Radiative transfer does not know about
   plumes; the plume-to-radiance coupling (`coupled`) stays in plumax and
   imports radtrax.
3. **Geo carriers arrive through extras.** `radtrax[geo]` and `plumax[geo]`
   add geotoolz and georeader; the base installs stay JAX, xarray and
   pipekit, as plumax's are today.
4. **Moved code imports only public geotoolz names.** The private helpers it
   needs become public in geotoolz before the move.
5. **Edges between stacks point down.** Numerics (gaussx, spectraldiffx,
   finitevolx, and external ones such as diffrax) are the foundational
   operators and import no GeoModels or GeoML package. GeoModels (plumax,
   somax, radtrax) build on Numerics and GeoStack. Whether GeoModels and GeoML
   (geonnax and the learning libraries) may import each other is
   [open](#pb-open-questions).

(pb-gates)=
## Testing gates

[](#tbl-pb-gates) states each [rule](#pb-rules) as a check.

:::{table} Each rule as a check.
:label: tbl-pb-gates

| Gate | Passes when |
|---|---|
| P1 No cycle | this page's notebook, run in CI on the pinned revisions, finds no strongly connected component; the counterfactual still finds one, so the check is live |
| P2 geotoolz stays below | no module of geotoolz imports `plumax` or `radtrax` (an import-linter contract in geotoolz), and `from geotoolz.plume import PlumeMask` raises `ImportError` naming `plumax.geo` |
| P3 radtrax stays below plumax | no module of radtrax imports `plumax` |
| P4 One matched filter | geotoolz's matched-filter tests pass against radtrax's core, and the two plumax filters are gone |
| P5 Light base installs | `pip install radtrax` and `pip install plumax` install neither geotoolz nor georeader |
| P6 Public helpers | `plume` and `matched_filter` import no `geotoolz._src` module after the move |
:::

(pb-open-questions)=
## Open questions

1. **Does the 3D-Var retrieval belong in radtrax or in vardax?**
   `assimilation` is a variational retrieval with a radiance observation
   operator. Its operator is radiative transfer; its solvers are generic
   variational data assimilation, which is vardax's job. Splitting it would
   make radtrax depend on vardax; whether that edge is allowed depends on
   which stack vardax sits in (question 5).
2. **Is the matched filter radiative transfer?** It is a detector whose
   target radtrax computes. radtrax is its natural home while it is the only
   consumer; a detection library in GeoML would be one if a second gas or
   sensor family needs it.
3. **`plumax.geo` or `plumax.integrations.geotoolz`?** geopatcher's
   `integrations/` pattern names the carrier library; `geo` is shorter and
   matches `radtrax[geo]`.
4. **Who owns the SRF tables?** georeader ships Sentinel-2's as package
   data; EMIT's and PRISMA's come with their products. radtrax could read
   them all through georeader, or carry its own copies and drop the extra.
5. **May GeoModels and GeoML import each other?** An emulator of plumax in
   GeoML would import plumax; a radtrax retrieval with a learned prior would
   import a GeoML package. Allowing one direction only keeps the graph
   acyclic; the data-assimilation libraries (vardax, filterax) need a stack
   first.

(pb-todos)=
## Upcoming work

[](#tbl-pb-todos) lists the follow-ups this page leaves.

:::{table} Follow-ups.
:label: tbl-pb-todos

| Follow-up | Where | Why |
|---|---|---|
| Add plumax, gaussx, finitevolX and spectraldiffx as submodules, in `UPSTREAM` | geostack `references/`, `scripts/capabilities.py` | this page reads them from clones; the capability index does not list them |
| Publish the helpers `plume` and `matched_filter` use | geotoolz `_src` | rule 4 |
| Delete `radtran.matched_filter` in favour of `matched_filter.core` | plumax | two filters in one package, before either moves |
| Make the cycle check a test | geostack `tests/` | gate P1 |
:::
