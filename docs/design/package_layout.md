---
title: "Package layout across geotoolz and xrtoolz"
subtitle: "Design · one job per distribution, matching family names, readers named as products"
short_title: Package layout
description: "xrtoolz adopts geotoolz's split: a code-free umbrella, an xrops package one tier above geoops, backend bridges folded into the shared machinery, xrproducts as the twin of geoproducts, geocatalog indexing states, and one carrier per geometry kind in each lane."
date: 2026-10-23
keywords:
  - xrtoolz
  - geotoolz
  - xrops
  - geoops
  - xrproducts
  - geoproducts
  - geocatalog
  - package layout
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

We adopt geotoolz's split for xrtoolz: distributions by job, modules inside the ops package by operation kind, and heavy backends as extras. geotoolz keeps its shape apart from a slimmer geoops base install, and no code moves between the two repos except bridges that become shared machinery. xrops sits one tier above geoops ([the tiers](#design-tiers)), so every kernel lives in geoops or below and xrops adds only what needs xarray.

- **PL-A. xrtoolz becomes a meta package.** Today `xrtoolz` is both the ops code and the umbrella that hard-depends on every sibling, so we move its code to a new `xrtoolz-ops` (import `xrops`) and leave `xrtoolz` with no code, as `geotoolz` has.
- **PL-B. Backend bridges stop being distributions.** `xreinx` and `xrsklearn` give way to one named-dims bridge each in pipekit-array (OP-O2, OP-O3), whereas `xrgrad` becomes `xrops.Gradient` over the geoops kernel (OP-O7).
- **PL-C. One home for the xarray Operator base.** The `Operator` of `xrcore` (with its `DataTree` dispatch) folds into `xrops._src`, and its `Signature` is replaced by `pipekit_ops.Signature`, with no re-export (PL-Q5; [pipekit kernels](./pipekit_kernels.md)).
- **PL-D. Matching family names.** geoops is organised by operation kind, whereas xrtoolz is organised by domain plus a `geo` grab-bag, so `xrops` takes the family names of geoops wherever the job is the same, keeps the state-physics families, and drops the empty `rs` and `ice` stubs.
- **PL-E. Readers named as products.** `xrreader` (CMEMS, CDS, AEMET) is the xarray twin of `geoproducts`, so it becomes `xrtoolz-products` (import `xrproducts`).

In the end state each repo has a code-free umbrella and the same job-shaped distributions, and `geoops` and `xrops` share family names where the job is the same.

## Ownership rules

We use one axis per level: distributions split by job, modules inside an ops package split by operation kind, and heavy backends sit behind extras.

| Job | geotoolz | xrtoolz |
| --- | --- | --- |
| meta | `geotoolz` | `xrtoolz` (no code) |
| transform | `geoops` (tier 1) | `xrops` (tier 2, new) |
| products | `geoproducts` | `xrproducts` (was `xrreader`) |
| carry | `geoarray` | xarray |
| shared | `geocatalog` · `geocloud` · `geopatcher` | `geocatalog` · `geocloud` · `geopatcher` |

Each row is one job, and only the names on the right change. The shared row already serves both lanes, and pipekit-array (tier 0) sits under both ops packages.

1. **geotoolz** and **xrtoolz** hold no code, as each depends on the distributions of its lane and documents the import table.
2. **geoops** holds every kernel operator, written once, and every per-item geometry operator. Its base install is slim, since rasterio, geopandas and shapely, scikit-image and georeader sit behind `[raster]`, `[vector]`, `[morph]` and `[sensors]`, and it has no `[xarray]` extra and never imports xrops or xarray.
3. **xrops** holds xarray-native work (selection by value, many files, `Dataset` and `DataTree`), `xrops.lift` for geoops operators over cubes, and state physics (`atm`, `ocn`, `budgets`). It may import geoops and geoarray, and crossing lanes goes through the converters of geoarray ([labeled lane](./labeled_lane.md)).
4. **Domain physics** on model states lives in `xrops`, whereas on observations it belongs to GeoModels (plumax, radtrax) and never to geoops ([stack levels](./stack_levels.md), SL-O1).
5. **geoproducts** and **xrproducts** read provider data into the carrier of their lane and never transform it.
6. **Heavy backends** (JAX, einx, scikit-learn) are extras and never base dependencies.
7. **Every package writes [the three levels](#design-levels)** the same way: primitives in `_src/array.py`, operators on carriers above them, and carrier code only where the carrier may be imported.

## Current-state audit

Six mismatches need action. We read the `main` branches of geotoolz (umbrella 0.12.0) and xrtoolz (0.0.5) on 21 Oct 2026 for this audit, and the call-site counts are still to do.

| Symbols | Defined in | Used by | Verdict |
| --- | --- | --- | --- |
| Umbrella | `geotoolz/__init__.py` (no code); `xrtoolz` (code + hard deps on all five siblings) | every install | **Split ownership.** `xrtoolz` is both the ops package and the meta package |
| Operator base | `pipekit_ops.Operator` + `geoops._src.contract`; `xrcore.operator.Operator` (DataTree dispatch) | all operators | **Split ownership.** Base lives in the ops package on one side, its own distribution on the other |
| einx bridge | `geoops.einx`; `xreinx` (dist), with a `xrtoolz.einx` shim | pipelines | **True duplicate.** Same job, module vs distribution |
| sklearn bridge | `geoops.learn`; `xrsklearn` (dist) | pipelines | **True duplicate.** Module vs distribution, and `learn` vs `sklearn` |
| Geometry | `geoops.geom` + `geoops._src.engine`; `xrtoolz.geo` (crs, mosaic, rasterize, vectorize, footprint, masks, wavelets, extremes, detrend, …) | both lanes | **Split ownership.** `xrtoolz.geo` mixes geometry with analysis |
| RS physics | `geoops.indices`, `geoops.radiometry`; `xrtoolz.rs` (empty, plans NDVI and radiance-to-reflectance) | none yet | **True duplicate** in plan. Observation kernels belong to geoops and already run on cubes |

Smaller findings:

- **Domain modules.** `xrtoolz.atm`, `ocn` and `budgets` act on gridded states, whereas `xrtoolz.ice` is an empty stub. The `plume` and `matched_filter` modules of geoops are already scheduled to leave for plumax and radtrax (radtrax backlog RTX-34, RTX-30).
- **Readers.** `xrreader` reads CMEMS, CDS and AEMET archives into xarray, which is the job of `geoproducts`, but it is named like a different kind of package.
- **Shim policy.** `xrtoolz._operator`, `xrtoolz.signature` and `xrtoolz.einx` say they are removed at 1.0. However, the stack bumps 0.X.0 for breaking changes, so the removal needs a named 0.X release.
- **Shared packages stay shared.** `geocatalog` (with `geocatalog.states`), `geocloud` and `geopatcher` already serve both lanes, and nothing in xrtoolz duplicates them.

## Target layout and moves

Every job has the same distribution shape in both repos, and every ops family has the same name where the job is the same.

| Job | geotoolz | xrtoolz today | xrtoolz after |
| --- | --- | --- | --- |
| meta | `geotoolz` | `xrtoolz` (code + meta) | `xrtoolz` (no code) |
| transform | `geoops`, slim base plus `[raster]`, `[vector]`, `[morph]`, `[sensors]` | `xrtoolz` | `xrops` |
| Operator base | `pipekit_ops.Operator` in geoops | `xrcore` | `xrops._src`; `pipekit_ops.Signature` (PL-Q2) |
| tensor algebra | `geoops.einx` | `xreinx` | pipekit-array `[einx]`, shared (OP-O2) |
| ML bridge | `geoops.learn` | `xrsklearn` | pipekit-array `to_rows` / `from_rows` and one estimator wrapper (OP-O3) |
| calculus | `geoops.geom.Gradient` | `xrgrad` | `xrops.Gradient` over the geoops kernel (OP-O7) |
| lifting over cubes | — | — | `xrops.lift` (OP-O4) |
| geometry | `geoops.geom` | `xrtoolz.geo` | `xrops.geom` + analysis families (PL-Q4) |
| RS physics | `geoops.indices`, `radiometry` | `xrtoolz.rs` (empty) | removed |
| state physics | — | `xrtoolz.atm`, `ocn`, `budgets` | `xrops.atm`, `ocn`, `budgets` (PL-Q6) |
| products | `geoproducts` | `xrreader` | `xrproducts` (PL-Q3) |
| carry | `geoarray` | xarray | unchanged |
| read + index, store, patch | `geocatalog`, `geocloud`, `geopatcher` | shared | unchanged |

1. **Create `xrtoolz-ops`.** We move every module of the `xrtoolz` distribution into `xrops` with history, and the tests stay unchanged apart from imports.
2. **Empty the umbrella.** `xrtoolz` depends on `xrtoolz-ops` and `xrtoolz-products`, re-exports nothing but `__version__`, and documents the import table, as `geotoolz/__init__.py` does.
3. **Retire the bridges.** Per PL-Q1, `xreinx` and `xrsklearn` are removed once the bridges of pipekit-array ship (OP-O2, OP-O3), and `xrgrad` becomes `xrops.Gradient`.
4. **Rename the families.** Per PL-Q4, we split `xrtoolz.geo` and delete `rs` and `ice`.
5. **Rename the reader.** Per PL-Q3, `xrtoolz-reader` becomes `xrtoolz-products`.
6. **No shims.** Old import paths are removed outright, as geotoolz does (PL-Q5), so the release that moves a module updates every caller in the stack and nothing re-exports the old path.

## Boundary: geoproducts and xrproducts

The line is what one read returns: geoproducts reads one acquisition into a geoarray carrier, whereas xrproducts reads anything stacked over time, members or levels, or a station series, into xarray (PL-Q7). This follows the workload rule of [the crossover](#design-crossover), and it replaces the 21 Oct wording of observations versus states drawn at an L3 carrier switch.

Provenance alone does not split cleanly, as CMEMS serves both gridded satellite maps and the GLORYS reanalysis, and CDS serves ERA5 and satellite products. So a provenance rule would split the reader of one provider across two packages.

| Product | Level | One read returns | Package |
| --- | --- | --- | --- |
| Sentinel-2 L1C, Landsat 8/9 L1 | L1 | one scene on its grid | geoproducts |
| EMIT L1B, L2B | L1, L2 | one scene in sensor geometry (`GeoSwath`) | geoproducts |
| GOES ABI, Himawari AHI | L1 | one scan | geoproducts |
| Gridded SLA maps (DUACS) | L4 | a cube over time | xrproducts |
| GLORYS12 reanalysis | L4 | a state cube | xrproducts |
| ERA5 | L4 | a state cube | xrproducts |
| AEMET stations | in situ | a station series | xrproducts |

1. **Both read, neither transforms.** A reader returns the carrier of its lane on the native grid of the provider, and regridding and derived fields are operators.
2. **Neither imports the other.** A product that needs both (EMIT scenes matched to ERA5 winds) is joined in geocatalog or a pipekit graph.
3. **Rename the `catalog` module of `xrreader`.** `xrreader._src.catalog` is a short-name registry (`glorys12.daily` → source and dataset ID) and not a catalog, so it becomes `xrproducts.registry` and "catalog" means geocatalog everywhere.

### Note: how geocatalog handles states

geocatalog already indexes states on `main` through `geocatalog.states`, so this plan only adds the hand-off from xrproducts.

- **Rows.** A state catalog is an xarray-kind catalog with five reserved columns (schema v1): `kind`, `run_time`, `member`, `producer`, `parents`. `build_state_catalog` fills them from the metadata of each file, with `CF_NAMES` and `CFGRIB_NAMES` as name maps.
- **Queries.** `forecasts`, `ensemble_at`, `forecast_chain`, `lead_window`, `at_time` and `latest_analysis` work on either backend.
- **Loading.** `load_states` stacks one file per (run, member) into an `(init_time, member, lead_time, y, x)` cube, lazily, behind the `[xarray-raster]` extra, and overlapping runs are never dropped.

The split of work:

1. **xrproducts fetches.** It handles authentication, the provider request, download, cache and opening one dataset.
2. **geocatalog indexes and loads.** Many files become rows, the rows are queried, and `load_states` builds the cube, as it does today.
3. **The hand-off is rows, not imports.** xrproducts writes state-catalog rows (or calls `build_state_catalog` on what it downloaded) so a fetch lands indexed, and geocatalog never imports xrproducts, whereas xrproducts depends on geocatalog only through an extra (PL-Q8).

## The elephants: points, polygons, rasters, states

Each geometry kind gets one carrier per lane, and changing kind (rasterize, sample, bin) is an operator and never the job of the carrier. On the array side the carriers are `GeoArray` (grids), `GeoSwath` (sensor geometry) and `GeoPoints` (point sets), with stored dims and time stamps on a `TimeAxis` ([geoarray](./geoarray.md)), whereas xarray formalizes neither geometry nor time meaning and conventions do (CF, xvec).

| Kind | Examples | Array lane (NumPy / JAX) | Labeled lane (xarray) | At rest |
| --- | --- | --- | --- | --- |
| Raster | Sentinel-2 scene, EMIT L1B | `GeoArray`, `GeoSwath` | `DataArray` on `(y, x)` or `(row, col)` | COG |
| State | GLORYS12, ERA5, a somax run | shaped arrays in the model; a carrier slice at its boundary, e.g. for H(x) | `Dataset` on `(time, member, level, y, x)` | Zarr, NetCDF |
| Polygons | plume outlines, AOIs, field boundaries | rasterized to `GeoArray` by an operator | xvec `DataArray` with a geometry dim | GeoParquet |
| Points | Argo profiles, along-track SSH, AEMET stations, plume source points | `GeoPoints` | `Dataset` on one `obs` dim (CF DSG) | GeoParquet, NetCDF DSG |

### Point clouds: the gap

Points have three shapes today and no carrier:

- The `RasterToPoints`, `PointsToRaster`, `RasterToPointCloud` and `PointCloudToRaster` operators of geoops take a GeoDataFrame or an xvec array.
- The along-track helpers of `xrtoolz.geo` take bare 1-D lon and lat arrays.
- `geocatalog.matchup` pairs catalog rows and not points.

The fix, one carrier per lane (PL-Q9):

1. **Labeled lane: a CF Discrete Sampling Geometry dataset.** One `obs` dim, with lon, lat, time and depth as coordinates on it, and `featureType` naming the layout (point, timeSeries, trajectory, profile). Ragged profiles use the contiguous ragged layout of DSG, one `row_size` per feature, and Argo and CMEMS along-track files already ship this way, so xrproducts opens them with no conversion.
2. **xvec only on demand.** We build an xvec geometry index on `obs` only when a spatial query needs it, since one shapely object per sample is the cost to avoid at 10⁷ along-track points.
3. **Array lane: `GeoPoints` in geoarray.** It holds horizontal `xy` `(obs, 2)`, an optional `z` `(obs,)` with its own vertical descriptor, values, per-sample time, a CRS and a fill, as a pytree; the [geoarray page](./geoarray.md) defines it and says why `z` is never a third column. vardax and filterax need it, because an observation operator maps a gridded state to values at points inside `jit` and `grad`, and a GeoDataFrame cannot cross `jit`.
4. **Kind changes are operators with matching names in both lanes.** These are sample (raster to points), bin or grid (points to raster), and rasterize and vectorize (polygons to and from raster). The converters of geoarray cross lanes (`GeoPoints` to and from a DSG dataset), as for grids ([labeled lane](./labeled_lane.md)).
5. **At rest.** We use GeoParquet for point tables that geocatalog queries with DuckDB, and NetCDF DSG or Zarr when a profile or a trajectory is the unit.

Alternatives considered:

- **One table carrier** (GeoDataFrame or GeoArrow) for points in both lanes. It is best for huge sets and DuckDB, but it is not labeled N-D and not `jit`-able.
- **xvec for points too.** It gives one vector story, but one shapely object per point at scale.

When to leave a carrier for xarray is the subject of [the crossover](#design-crossover).

(design-one-workspace)=
## One workspace for both lanes

xrtoolz's three packages move into the geotoolz repository as workspace members, so that one uv workspace, one lock and one CI hold both lanes (PL-Q10, decided 13 Nov 2026). The workstream is PL-F. It changes where the packages live and how their releases are cut, but it changes none of the ownership rules above: xrops still sits one tier above geoops, geoops still never imports xarray, and every distribution and import name stays as it is. The summary's "no code moves between the two repos" is about which package owns a job, and it still holds, because no module changes package.

### Why one workspace

We read the `main` branches of geotoolz and xrtoolz on 12 Nov 2026.

- **xrops already builds on geoops.** `geotoolz-ops` and `geotoolz-array` are base dependencies of `xrtoolz-ops`, not extras. OP-O1 (xrops never re-implements a geoops kernel), OP-O4 (`xrops.lift`) and gate D1 (`scripts/family_parity.py` in xrtoolz) tie the two packages together, so a change to a geoops kernel or family name usually needs a change in xrops as well.
- **Two repos cost a sync step each time.** xrtoolz's root `pyproject.toml` takes `geotoolz-ops` and `geotoolz-array` from git with no tag, so the commit it tests against moves whenever someone relocks. Its pipekit override still names the pre-0.2.0 `packages/pipekit`, and its migration to pipekit-ops (xrtoolz#400) waited on geotoolz#651. A break in xrops caused by a geoops change surfaces only in xrtoolz's CI, after the geoops release.
- **geotoolz already has the machinery.** Its CI runs one test job per package, only for the packages a change reaches, through the `RUNTIME` map in `ci.yml` (geotoolz#659). Its generated `test_dependency_rules.py` already checks every package's imports. Both repos are uv workspaces over `packages/*` with one release-please component per package, both pin functax at `v0.0.6`, and both build their docs with mkdocs.
- **The size is modest.** geotoolz has about 105,000 lines of source in seven packages and xrtoolz about 45,000 in three. In the month to 12 Nov 2026, xrtoolz had 35 commits on `main` and geotoolz 97.

### What stays the same

- **Names.** `xrtoolz`, `xrtoolz-ops` (import `xrops`) and `xrtoolz-products` (import `xrproducts`) keep their distribution and import names, and their version series continue from `xrtoolz` 0.4.1, `xrtoolz-ops` 0.5.1 and `xrtoolz-products` 0.5.0. No Python import anywhere in the stack changes.
- **Two umbrellas.** `geotoolz` stays the meta package of the array lane and `xrtoolz` the meta package of the labeled lane, each with no code.
- **The lanes and their tiers.** The xarray lane stays one tier above the array lane, now as a rule between packages rather than between repositories (see below).
- **The slim bases.** A fresh `pip install geotoolz-ops` still installs no xarray, and the base-install workflow keeps checking it.

### What changes

| | Today | After PL-F |
| --- | --- | --- |
| Where xrops gets geoops | git source on geotoolz `main`, no tag; the lock holds whichever commit was current at the last relock | workspace member, always the same commit |
| A PR that changes a geoops kernel | runs geoops tests only; xrops breaks later, in xrtoolz | runs the xrops tests too, through the `RUNTIME` map |
| pipekit and functax pins | set and bumped in each repo | set once in the geotoolz root |
| Releases | release-please in each repo | one release-please config with ten components |
| Docs | two mkdocs sites | one site with a section for the xarray lane |
| Issues | two trackers | one tracker; xrtoolz is archived |
| geostack | two submodules, `TIERS` lists both repos | one submodule, package tiers inside geotoolz |

### The tier inside one repository

geostack's `scripts/dependency_graph.py` gives each repository one tier (`TIERS`), and between the packages of one workspace it checks only for cycles. A cycle check catches geoops importing xrops, because xrops already imports geoops, but it misses geocatalog or geoproducts importing xrops or xrproducts, since xrops depends on neither of them. The tier therefore has to move from the repository to the package.

1. **Package tiers.** A new `PACKAGE_TIERS` table in `scripts/dependency_graph.py` gives a tier to the packages of a repository that do not take the repository's own: `{"geotoolz": {"xrtoolz": 2, "xrtoolz-ops": 2, "xrtoolz-products": 2}}`. Every other geotoolz package takes geotoolz's tier, 1, and `xrtoolz` leaves `TIERS`.
2. **A check between packages.** An edge between two packages of one repository must point down a package tier or stay level, and a new violation kind, `package-tier`, reports one that points up. The edges between repositories keep using each package's tier, so pipekit (tier 0) still may not import xrops.
3. **The generated guard.** `python scripts/dependency_graph.py --guard geotoolz` then writes the rule into geotoolz's `test_dependency_rules.py` ("no `geotoolz-*` package imports or depends on `xrtoolz`, `xrtoolz-ops` or `xrtoolz-products`"), and xrtoolz's own guard test is deleted.
4. **The page and the tests.** The tier table on [the dependency graph](../api/dependency_graph.md) lists the xrtoolz packages at GeoStack tier 2 under geotoolz, and `tests/test_dependency_graph.py` states its two xrtoolz tests (the hypothetical cycle through xrtoolz, geotoolz and pipekit, and "xrtoolz may import geotoolz but not the reverse") over packages instead of repositories. The rule in `AGENTS.md` that "each repository has a tier" becomes "each repository, or a group of packages inside one, has a tier".

### Migration steps

Each step is one PR unless it says otherwise, and the geotoolz steps land as a stack.

1. **Clear the way (done 13 Nov 2026).** xrtoolz#400 (pipekit-ops) merged first as xrtoolz#407, so the move carries no pin change. The two draft PRs on xrtoolz (#212 and #213, opened by the Copilot agent on 22 May 2026) were closed, since a pull request cannot be transferred. Only dependabot PRs remain open on xrtoolz, and dependabot opens their equivalents in geotoolz after the move.
2. **Rewrite xrtoolz's history for the move.** In a scratch clone, `git filter-repo` keeps `packages/xrtoolz*` where they are, moves `docs/` to `docs/xarray/`, `notebooks/` to `notebooks/xarray/` and `scripts/family_parity.py` to `scripts/`, and drops the root files whose content is merged by hand in step 4 (`pyproject.toml`, `uv.lock`, `AGENTS.md`, `CLAUDE.md`, `mkdocs.yml`, `Makefile`, `.github/`). The rewrite carries the `xrtoolz*-v*` tags along to the rewritten commits.
3. **Merge the history into geotoolz.** On a geotoolz branch, `git merge --allow-unrelated-histories` brings in the rewritten history, and the rewritten tags are pushed with it, so that `git log --follow` and blame reach back past the move and release-please finds each xrtoolz component's last release.
4. **Join the workspace.** The geotoolz root `pyproject.toml` gains `{ workspace = true }` sources for the three packages and their dev dependencies (dask, xarray, rioxarray, regionmask, xrft, xskillscore, netcdf4, scikit-learn and the test fixtures), and the git sources of geotoolz that xrtoolz carried go away. The xrtoolz root `AGENTS.md` becomes `packages/xrtoolz-ops/AGENTS.md`, as each geotoolz package has its own. One relock produces one `uv.lock` for ten packages.
5. **Releases.** `release-please-config.json` and `.release-please-manifest.json` gain the three components at their current versions. The first release of each from geotoolz is a patch with no API change.
6. **CI.** `ci.yml` gains a path filter per xrtoolz package and `RUNTIME` entries for them: `xrtoolz-ops` is reached by `geotoolz-array`, `geotoolz-ops` and their closure, and by `xrtoolz-products`. `base-install.yml` gains a job per xrtoolz package, and xrtoolz's `tests-extended.yml` merges into geotoolz's.
7. **The tier rule.** The geostack changes of the section above land, and the regenerated guard is committed to geotoolz in the same stack.
8. **Docs.** geotoolz's `mkdocs.yml` gains an "xarray lane" section for `docs/xarray/`, and the xrtoolz site keeps a single page that points to it.
9. **Issues.** The open xrtoolz issues (53 on 13 Nov 2026) move with `gh issue transfer`, after a trial on one issue confirms that its GeoStack project (#13) fields and its sub-issue and blocked-by links survive the transfer. Closed issues stay in the archived repository.
10. **geostack.** The `references/geostack/xrtoolz` submodule is removed, the six xrtoolz paths in `pixi.toml` point into `references/geostack/geotoolz/packages/`, `scripts/capabilities.py` keeps its two xrtoolz entries unchanged (it names distributions, which pixi now installs from the geotoolz submodule), and `make capabilities` and `make dependency-graph` regenerate the indexes and the graph.
11. **Downstream.** plumax's `io` extra still names `xrtoolz` and `xrtoolz-reader` at rev `a94d9a1` of the six-package layout, which predates PL-A to PL-E, so it moves to `xrtoolz-products` from the geotoolz repository as part of plumax#249. A search of the stack for `jejjohnson/xrtoolz` finds any other git source.
12. **Archive.** xrtoolz's README points to geotoolz, and the repository is archived, which can be undone; it is not deleted.

### Gates

| Gate | Passes when |
| --- | --- |
| F1 History | `git log --follow` on a moved file in geotoolz lists its commits from before the move |
| F2 One lock | one `uv.lock` resolves all ten packages, with `geotoolz-ops` and `geotoolz-array` from the workspace |
| F3 Test parity | each xrtoolz package collects as many tests in geotoolz as it did on xrtoolz `main` before the move |
| F4 Tier | the guard test fails on a seeded `import xrops` in geoops, and geostack's graph reports it as `package-tier` |
| F5 CI reach | a PR that touches only `packages/geotoolz-ops/src` runs the `xrtoolz-ops` tests |
| F6 Release | release-please proposes `xrtoolz-ops` 0.5.2 for a fix, from the geotoolz repository |
| F7 Slim base | a fresh `pip install geotoolz-ops` still installs no xarray |

### Risks

- **The name covers more than geo arrays.** The geotoolz repository would hold ocean, atmosphere and budget physics under `xrops`. We keep the repository name, because renaming it touches every git source in the stack for no gain in function, and the README and docs say that the repository holds both lanes.
- **A heavier dev environment.** The dev group gains the xarray stack listed in step 4. Base installs do not change, because each package's install is checked on its own.
- **Issue numbers change.** A transferred issue gets a new number, and GitHub redirects its old URL, but a reference written as plain text, such as "xrtoolz#355" in another issue body, still points at the archived repository.
- **xrops may shrink later.** The diagnostix design (proposed) moves the state-physics maths that xrops wraps into a GeoNumerics library. That would make xrops thinner, and it neither blocks nor depends on PL-F.

The alternative is to keep two repositories and pin xrtoolz's geotoolz sources to a tag, bumped by hand after each geotoolz release. That removes the moving lock, but not the late breakage or the second set of pins, CI and release configuration.

## Open questions

Each question has a recommendation and its alternative, and we can flip any of them before the issues open.

| ID | Question | Recommended | Alternative | Status |
| --- | --- | --- | --- | --- |
| PL-Q1 | Backend bridges: extras or distributions? | Retire `xreinx` and `xrsklearn` in favour of pipekit-array's shared bridges (OP-O2, OP-O3); `xrgrad` becomes `xrops.Gradient` (OP-O7) | Fold them into `xrops` behind extras (the 21 Oct recommendation), or split `geoops.einx` and `geoops.learn` out into distributions | superseded by OP-O2, OP-O3, OP-O7 |
| PL-Q2 | Where the xarray Operator base lives | **Decided 29 Oct 2026:** `xrcore` folds into `xrops._src`, as geoops keeps its base; `Signature` is `pipekit_ops.Signature` | Keep `xrcore` if something outside xrtoolz needs the base without the ops | decided 29 Oct 2026 (`Signature` 21 Oct, GA-Q9) |
| PL-Q3 | Reader name | **Decided 29 Oct 2026:** `xrtoolz-products` (import `xrproducts`), matching `geoproducts` | Keep `xrreader` | decided 29 Oct 2026 |
| PL-Q4 | How to split `xrtoolz.geo` | **Decided 29 Oct 2026:** geometry (crs, mosaic, rasterize, vectorize, footprint, subset) → `xrops.geom`; analysis (detrend, wavelets, extremes, regimes) into named families | Keep `geo` as the generic family and rename `geoops.geom` to match | decided 29 Oct 2026 |
| PL-Q5 | Shim lifetime | **Decided 29 Oct 2026:** no shims; old import paths are removed outright in the release that moves them, and the stack's callers move in the same release, as in geotoolz | a one-release `DeprecationWarning` shim (the earlier recommendation), or shims through the whole 0.x series | decided 29 Oct 2026 |
| PL-Q6 | Domain families | `xrops.atm`, `ocn`, `budgets` stay, because they act on model states; geoops stays domain-neutral | Move them to GeoModels packages next to their physics | decided 22 Oct 2026 (OP-O1); the split by discretisation is SL-O1 |
| PL-Q7 | geoproducts versus xrproducts | **Decided 29 Oct 2026:** what one read returns: one acquisition → geoproducts; a cube or a station series → xrproducts | Strict provenance (observations versus model output), splitting each provider's reader across both | decided 29 Oct 2026 |
| PL-Q8 | Who assembles state cubes | **Decided 29 Oct 2026:** `load_states` stays in geocatalog, which already serves both lanes; xrproducts hands off rows | Move cube assembly into xrproducts and keep geocatalog index-only | decided 29 Oct 2026 |
| PL-Q9 | Point carriers | CF DSG `Dataset` in the labeled lane, `GeoPoints` in the array lane | One table carrier for both lanes, or xvec for points | `GeoPoints` decided (geoarray page); DSG open |
| PL-Q10 | One repository or two for the two lanes | **Decided 13 Nov 2026:** xrtoolz's three packages join the geotoolz workspace, with the xarray lane at package tier 2 ([one workspace](#design-one-workspace)) | Keep two repositories and pin xrtoolz's geotoolz sources to a tag | decided 13 Nov 2026 |
| PL-Q11 | One products package for both lanes | geoproducts absorbs xrproducts, and each reader returns the carrier its geometry needs ([products](./products.md)) | Keep the two packages, split by what one read returns (PL-Q7) | open |
| PL-Q12 | xrops as the xarray layer | kernels run through `KernelOp` and move to the libraries below, and xrops keeps the work that needs labels ([xrops layer](./xrops_layer.md)) | Keep state-domain kernels in xrops (OP-O1) | open |

## Gates, sequencing and risks

Each workstream merges only when its gate passes; PL-A ships first because every other move lands inside `xrops`.

| Gate | Workstream | Passes when |
| --- | --- | --- |
| A1 Port parity | PL-A | the moved tests pass with only import changes, and the test count equals the old `xrtoolz` suite's |
| A2 Empty umbrella | PL-A | `xrtoolz/__init__.py` exports only `__version__`, and `import xrtoolz` imports no ops module |
| B1 Light base | PL-B | a fresh `pip install xrtoolz-ops` installs none of jax, einx, scikit-learn |
| C1 One base | PL-C | every `xrops` operator is an instance of the one base class, and `xrcore` no longer exists |
| D1 Name parity | PL-D | a script lists the families of `geoops` and `xrops`; every name in both means the same job, and the list is checked into the docs |
| E1 Reader rename | PL-E | every public name of `xrreader` is importable from `xrproducts`, and `import xrreader` fails |

The order is PL-A, then PL-B and PL-E in parallel, then PL-C, then PL-D. PL-C waits for PL-B: `xreinx` and `xrsklearn` subclass `xrcore.Operator` and `xrops` depends on `xrsklearn`, so folding `xrcore` into `xrops` before those bridges are deleted would make a cycle. PL-B waits for the pipekit-array release that ships the shared bridges (OP-O2, OP-O3), and `xrops.lift` waits for the slim base of geoops ([operators](./operators.md)). Each is one minor bump of `xrtoolz` (0.1.0, then 0.2.0).

- **Downstream imports break.** pipekit, geostack notebooks and GeoBench import `xrtoolz.*` today. Mitigation: there are no shims (PL-Q5), so each release greps the stack for the old paths and updates every caller in the same release.
- **The shared bridges slip.** If the einx and scikit-learn bridges of pipekit-array are late, PL-B blocks. Fallback: we keep `xreinx` and `xrsklearn` as they are for one more release, and PL-D still applies.
- **`xrtoolz.geo` has no clean split.** Some functions are both geometry and analysis. Fallback: we place each such function by its main job and list it in the checked-in table of D1.
