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

xrtoolz adopts geotoolz's split: distributions by job, modules inside the ops package by operation kind, heavy backends as extras.
geotoolz keeps its shape apart from a slimmer geoops base install, and no code moves between the two repos except bridges that become shared machinery.
xrops sits one tier above geoops ([the tiers](#design-tiers)), so every kernel lives in geoops or below and xrops adds only what needs xarray.

- **PL-A. xrtoolz becomes a meta package.**
  Today `xrtoolz` is both the ops code and the umbrella that hard-depends on every sibling.
  Its code moves to a new `xrtoolz-ops` (import `xrops`), and `xrtoolz` holds no code, as `geotoolz` does.
- **PL-B. Backend bridges stop being distributions.**
  `xreinx` and `xrsklearn` give way to one named-dims bridge each in pipekit-array (OP-O2, OP-O3); `xrgrad` becomes `xrops.Gradient` over the geoops kernel (OP-O7).
- **PL-C. One home for the xarray Operator base.**
  `xrcore`'s `Operator` (with its `DataTree` dispatch) folds into `xrops._src`; its `Signature` becomes a re-export of `pipekit.Signature` for one release, then goes ([pipekit kernels](./pipekit_kernels.md)).
- **PL-D. Matching family names.**
  geoops is organised by operation kind, xrtoolz by domain plus a `geo` grab-bag.
  `xrops` takes geoops's family names wherever the job is the same, keeps state-physics families, and drops the empty `rs` and `ice` stubs.
- **PL-E. Readers named as products.**
  `xrreader` (CMEMS, CDS, AEMET) is the xarray twin of `geoproducts` and becomes `xrtoolz-products` (import `xrproducts`).

End state: each repo has a code-free umbrella and the same job-shaped distributions, and `geoops` and `xrops` share family names where the job is the same.

## Ownership rules

One axis per level: distributions split by job, modules inside an ops package split by operation kind, and heavy backends sit behind extras.

| Job | geotoolz | xrtoolz |
| --- | --- | --- |
| meta | `geotoolz` | `xrtoolz` (no code) |
| transform | `geoops` (tier 1) | `xrops` (tier 2, new) |
| products | `geoproducts` | `xrproducts` (was `xrreader`) |
| carry | `geoarray` | xarray |
| shared | `geocatalog` · `geocloud` · `geopatcher` | `geocatalog` · `geocloud` · `geopatcher` |

Each row is one job; only the names on the right change.
The shared row already serves both lanes, and pipekit-array (tier 0) sits under both ops packages.

1. **geotoolz** and **xrtoolz** hold no code.
   Each depends on its lane's distributions and documents the import table.
2. **geoops** holds every kernel operator, written once, and every per-item geometry operator.
   Its base install is slim: rasterio, geopandas and shapely, scikit-image and georeader sit behind `[raster]`, `[vector]`, `[morph]` and `[sensors]`; it has no `[xarray]` extra and never imports xrops or xarray.
3. **xrops** holds xarray-native work (selection by value, many files, `Dataset` and `DataTree`), `xrops.lift` for geoops operators over cubes, and state physics (`atm`, `ocn`, `budgets`).
   It may import geoops and geoarray; crossing lanes goes through geoarray's converters ([labeled lane](./labeled_lane.md)).
4. **Domain physics** on model states lives in `xrops`; on observations it belongs to GeoModels (plumax, radtrax), never to geoops ([stack levels](./stack_levels.md), SL-O1).
5. **geoproducts** and **xrproducts** read provider data into their lane's carrier and never transform it.
6. **Heavy backends** (JAX, einx, scikit-learn) are extras, never base dependencies.
7. **Every package writes [the three levels](#design-levels)** the same way: primitives in `_src/array.py`, operators on carriers above them, carrier code only where the carrier may be imported.

## Current-state audit

Six mismatches need action.
The audit read the `main` branches of geotoolz (umbrella 0.12.0) and xrtoolz (0.0.5) on 21 Oct 2026; call-site counts are still to do.

| Symbols | Defined in | Used by | Verdict |
| --- | --- | --- | --- |
| Umbrella | `geotoolz/__init__.py` (no code); `xrtoolz` (code + hard deps on all five siblings) | every install | **Split ownership.** `xrtoolz` is both the ops package and the meta package |
| Operator base | `pipekit.Operator` + `geoops._src.contract`; `xrcore.operator.Operator` (DataTree dispatch) | all operators | **Split ownership.** Base lives in the ops package on one side, its own distribution on the other |
| einx bridge | `geoops.einx`; `xreinx` (dist), with a `xrtoolz.einx` shim | pipelines | **True duplicate.** Same job, module vs distribution |
| sklearn bridge | `geoops.learn`; `xrsklearn` (dist) | pipelines | **True duplicate.** Module vs distribution, and `learn` vs `sklearn` |
| Geometry | `geoops.geom` + `geoops._src.engine`; `xrtoolz.geo` (crs, mosaic, rasterize, vectorize, footprint, masks, wavelets, extremes, detrend, …) | both lanes | **Split ownership.** `xrtoolz.geo` mixes geometry with analysis |
| RS physics | `geoops.indices`, `geoops.radiometry`; `xrtoolz.rs` (empty, plans NDVI and radiance-to-reflectance) | none yet | **True duplicate** in plan. Observation kernels belong to geoops and already run on cubes |

Smaller findings:

- **Domain modules.**
  `xrtoolz.atm`, `ocn` and `budgets` act on gridded states; `xrtoolz.ice` is an empty stub.
  geoops's `plume` and `matched_filter` are already scheduled to leave for plumax and radtrax (radtrax backlog RTX-34, RTX-30).
- **Readers.**
  `xrreader` reads CMEMS, CDS and AEMET archives into xarray.
  It does the job of `geoproducts` but is named like a different kind of package.
- **Shim policy.**
  `xrtoolz._operator`, `xrtoolz.signature` and `xrtoolz.einx` say they are removed at 1.0.
  The stack bumps 0.X.0 for breaking changes, so the removal needs a named 0.X release.
- **Shared packages stay shared.**
  `geocatalog` (with `geocatalog.states`), `geocloud` and `geopatcher` already serve both lanes; nothing in xrtoolz duplicates them.

## Target layout and moves

Every job has the same distribution shape in both repos, and every ops family has the same name where the job is the same.

| Job | geotoolz | xrtoolz today | xrtoolz after |
| --- | --- | --- | --- |
| meta | `geotoolz` | `xrtoolz` (code + meta) | `xrtoolz` (no code) |
| transform | `geoops`, slim base plus `[raster]`, `[vector]`, `[morph]`, `[sensors]` | `xrtoolz` | `xrops` |
| Operator base | `pipekit.Operator` in geoops | `xrcore` | `xrops._src`; `pipekit.Signature` (PL-Q2) |
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

1. **Create `xrtoolz-ops`.**
   Move every module of the `xrtoolz` distribution into `xrops` with history, tests unchanged apart from imports.
2. **Empty the umbrella.**
   `xrtoolz` depends on `xrtoolz-ops` and `xrtoolz-products`, re-exports nothing but `__version__`, and documents the import table, as `geotoolz/__init__.py` does.
3. **Retire the bridges.**
   Per PL-Q1, `xreinx` and `xrsklearn` are removed once pipekit-array's bridges ship (OP-O2, OP-O3), and `xrgrad` becomes `xrops.Gradient`.
4. **Rename the families.**
   Per PL-Q4, split `xrtoolz.geo`; delete `rs` and `ice`.
5. **Rename the reader.**
   Per PL-Q3, `xrtoolz-reader` becomes `xrtoolz-products`.
6. **No shims.**
   Old import paths are removed outright, as geotoolz does (PL-Q5): the release that moves a module updates every caller in the stack, and nothing re-exports the old path.

## Boundary: geoproducts and xrproducts

The line is what one read returns: geoproducts reads one acquisition into a geoarray carrier, and xrproducts reads anything stacked over time, members or levels, or a station series, into xarray (PL-Q7).
It follows the workload rule of [the crossover](#design-crossover), and it replaces the 21 Oct wording of observations versus states drawn at an L3 carrier switch.

Provenance alone does not split cleanly.
CMEMS serves both gridded satellite maps and the GLORYS reanalysis, and CDS serves ERA5 and satellite products.
A provenance rule would split one provider's reader across two packages.

| Product | Level | One read returns | Package |
| --- | --- | --- | --- |
| Sentinel-2 L1C, Landsat 8/9 L1 | L1 | one scene on its grid | geoproducts |
| EMIT L1B, L2B | L1, L2 | one scene in sensor geometry (`GeoSwath`) | geoproducts |
| GOES ABI, Himawari AHI | L1 | one scan | geoproducts |
| Gridded SLA maps (DUACS) | L4 | a cube over time | xrproducts |
| GLORYS12 reanalysis | L4 | a state cube | xrproducts |
| ERA5 | L4 | a state cube | xrproducts |
| AEMET stations | in situ | a station series | xrproducts |

1. **Both read, neither transforms.**
   A reader returns its lane's carrier on the provider's native grid; regridding and derived fields are operators.
2. **Neither imports the other.**
   A product that needs both (EMIT scenes matched to ERA5 winds) is joined in geocatalog or a pipekit graph.
3. **Rename `xrreader`'s `catalog` module.**
   `xrreader._src.catalog` is a short-name registry (`glorys12.daily` → source and dataset ID), not a catalog.
   It becomes `xrproducts.registry`, so "catalog" means geocatalog everywhere.

### Note: how geocatalog handles states

geocatalog already indexes states on `main` through `geocatalog.states`; this plan only adds the hand-off from xrproducts.

- **Rows.**
  A state catalog is an xarray-kind catalog with five reserved columns (schema v1): `kind`, `run_time`, `member`, `producer`, `parents`.
  `build_state_catalog` fills them from each file's metadata, with `CF_NAMES` and `CFGRIB_NAMES` as name maps.
- **Queries.**
  `forecasts`, `ensemble_at`, `forecast_chain`, `lead_window`, `at_time` and `latest_analysis` work on either backend.
- **Loading.**
  `load_states` stacks one file per (run, member) into an `(init_time, member, lead_time, y, x)` cube, lazily, behind the `[xarray-raster]` extra.
  Overlapping runs are never dropped.

The split of work:

1. **xrproducts fetches.**
   Authentication, the provider request, download, cache and opening one dataset.
2. **geocatalog indexes and loads.**
   Many files become rows, rows are queried, and `load_states` builds the cube, as it does today.
3. **The hand-off is rows, not imports.**
   xrproducts writes state-catalog rows (or calls `build_state_catalog` on what it downloaded) so a fetch lands indexed.
   geocatalog never imports xrproducts, and xrproducts depends on geocatalog only through an extra (PL-Q8).

## The elephants: points, polygons, rasters, states

Each geometry kind gets one carrier per lane.
Changing kind (rasterize, sample, bin) is an operator, never the carrier's job.
On the array side the carriers are `GeoArray` (grids), `GeoSwath` (sensor geometry) and `GeoPoints` (point sets), with stored dims and time stamps on a `TimeAxis` ([geoarray](./geoarray.md)); xarray formalizes neither geometry nor time meaning, conventions do (CF, xvec).

| Kind | Examples | Array lane (NumPy / JAX) | Labeled lane (xarray) | At rest |
| --- | --- | --- | --- | --- |
| Raster | Sentinel-2 scene, EMIT L1B | `GeoArray`, `GeoSwath` | `DataArray` on `(y, x)` or `(row, col)` | COG |
| State | GLORYS12, ERA5, a somax run | shaped arrays in the model; a carrier slice at its boundary, e.g. for H(x) | `Dataset` on `(time, member, level, y, x)` | Zarr, NetCDF |
| Polygons | plume outlines, AOIs, field boundaries | rasterized to `GeoArray` by an operator | xvec `DataArray` with a geometry dim | GeoParquet |
| Points | Argo profiles, along-track SSH, AEMET stations, plume source points | `GeoPoints` | `Dataset` on one `obs` dim (CF DSG) | GeoParquet, NetCDF DSG |

### Point clouds: the gap

Points have three shapes today and no carrier:

- geoops's `RasterToPoints`, `PointsToRaster`, `RasterToPointCloud` and `PointCloudToRaster` take a GeoDataFrame or an xvec array.
- `xrtoolz.geo` along-track helpers take bare 1-D lon and lat arrays.
- `geocatalog.matchup` pairs catalog rows, not points.

The fix, one carrier per lane (PL-Q9):

1. **Labeled lane: a CF Discrete Sampling Geometry dataset.**
   One `obs` dim, with lon, lat, time and depth as coordinates on it, and `featureType` naming the layout (point, timeSeries, trajectory, profile).
   Ragged profiles use DSG's contiguous ragged layout, one `row_size` per feature.
   Argo and CMEMS along-track files already ship this way, so xrproducts opens them with no conversion.
2. **xvec only on demand.**
   Build an xvec geometry index on `obs` only when a spatial query needs it; one shapely object per sample is the cost to avoid at 10⁷ along-track points.
3. **Array lane: `GeoPoints` in geoarray.**
   Horizontal `xy` `(obs, 2)`, an optional `z` `(obs,)` with its own vertical descriptor, values, per-sample time, a CRS and a fill, as a pytree; the [geoarray page](./geoarray.md) defines it and says why `z` is never a third column.
   vardax and filterax need it: an observation operator maps a gridded state to values at points inside `jit` and `grad`, and a GeoDataFrame cannot cross `jit`.
4. **Kind changes are operators with matching names in both lanes.**
   Sample (raster to points), bin or grid (points to raster), rasterize and vectorize (polygons to and from raster).
   geoarray's converters cross lanes (`GeoPoints` to and from a DSG dataset), as for grids ([labeled lane](./labeled_lane.md)).
5. **At rest.**
   GeoParquet for point tables geocatalog queries with DuckDB; NetCDF DSG or Zarr when a profile or a trajectory is the unit.

Alternatives considered:

- **One table carrier** (GeoDataFrame or GeoArrow) for points in both lanes.
  It is best for huge sets and DuckDB, but not labeled N-D and not `jit`-able.
- **xvec for points too.**
  One vector story, but one shapely object per point at scale.

When to leave a carrier for xarray is the subject of [the crossover](#design-crossover).

## Open questions

Each has a recommendation and its alternative; flip any before the issues open.

| ID | Question | Recommended | Alternative | Status |
| --- | --- | --- | --- | --- |
| PL-Q1 | Backend bridges: extras or distributions? | Retire `xreinx` and `xrsklearn` in favour of pipekit-array's shared bridges (OP-O2, OP-O3); `xrgrad` becomes `xrops.Gradient` (OP-O7) | Fold them into `xrops` behind extras (the 21 Oct recommendation), or split `geoops.einx` and `geoops.learn` out into distributions | superseded by OP-O2, OP-O3, OP-O7 |
| PL-Q2 | Where the xarray Operator base lives | **Decided 29 Oct 2026:** `xrcore` folds into `xrops._src`, as geoops keeps its base; `Signature` is `pipekit.Signature` | Keep `xrcore` if something outside xrtoolz needs the base without the ops | decided 29 Oct 2026 (`Signature` 21 Oct, GA-Q9) |
| PL-Q3 | Reader name | **Decided 29 Oct 2026:** `xrtoolz-products` (import `xrproducts`), matching `geoproducts` | Keep `xrreader` | decided 29 Oct 2026 |
| PL-Q4 | How to split `xrtoolz.geo` | **Decided 29 Oct 2026:** geometry (crs, mosaic, rasterize, vectorize, footprint, subset) → `xrops.geom`; analysis (detrend, wavelets, extremes, regimes) into named families | Keep `geo` as the generic family and rename `geoops.geom` to match | decided 29 Oct 2026 |
| PL-Q5 | Shim lifetime | **Decided 29 Oct 2026:** no shims; old import paths are removed outright in the release that moves them, and the stack's callers move in the same release, as in geotoolz | a one-release `DeprecationWarning` shim (the earlier recommendation), or shims through the whole 0.x series | decided 29 Oct 2026 |
| PL-Q6 | Domain families | `xrops.atm`, `ocn`, `budgets` stay, because they act on model states; geoops stays domain-neutral | Move them to GeoModels packages next to their physics | decided 22 Oct 2026 (OP-O1); the split by discretisation is SL-O1 |
| PL-Q7 | geoproducts versus xrproducts | **Decided 29 Oct 2026:** what one read returns: one acquisition → geoproducts; a cube or a station series → xrproducts | Strict provenance (observations versus model output), splitting each provider's reader across both | decided 29 Oct 2026 |
| PL-Q8 | Who assembles state cubes | **Decided 29 Oct 2026:** `load_states` stays in geocatalog, which already serves both lanes; xrproducts hands off rows | Move cube assembly into xrproducts and keep geocatalog index-only | decided 29 Oct 2026 |
| PL-Q9 | Point carriers | CF DSG `Dataset` in the labeled lane, `GeoPoints` in the array lane | One table carrier for both lanes, or xvec for points | `GeoPoints` decided (geoarray page); DSG open |

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

Order: PL-A, then PL-B, PL-C and PL-E in parallel, then PL-D.
PL-B waits for the pipekit-array release that ships the shared bridges, and `xrops.lift` waits for geoops's slim base ([operators](./operators.md)).
Each is one minor bump of `xrtoolz` (0.1.0, then 0.2.0).

- **Downstream imports break.**
  pipekit, geostack notebooks and GeoBench import `xrtoolz.*` today.
  Mitigation: no shims (PL-Q5), so each release greps the stack for the old paths and updates every caller in the same release.
- **The shared bridges slip.**
  If pipekit-array's einx and scikit-learn bridges are late, PL-B blocks.
  Fallback: keep `xreinx` and `xrsklearn` as they are for one more release; PL-D still applies.
- **`xrtoolz.geo` has no clean split.**
  Some functions are both geometry and analysis.
  Fallback: place each such function by its main job and list it in D1's checked-in table.
