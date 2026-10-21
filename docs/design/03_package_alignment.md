---
title: "xrtoolz and geotoolz: one split across both lanes"
subtitle: "Design · one job per distribution, matching family names in geotoolz and xrtoolz"
short_title: Package alignment
description: "Realigns xrtoolz with geotoolz: a code-free umbrella, an xrops package mirroring geoops, backend bridges as extras, xrproducts as the twin of geoproducts, and the boundary between observations and states."
date: 2026-10-21
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

xrtoolz adopts geotoolz's split: distributions by job, modules inside the ops package by operation kind, heavy backends as extras. geotoolz does not change, and no code moves between the two repos.

- **A. xrtoolz becomes a meta package.**
  Today `xrtoolz` is both the ops code and the umbrella that hard-depends on every sibling.
  Its code moves to a new `xrtoolz-ops` (import `xrops`), mirroring `geotoolz-ops` / `geoops`, and `xrtoolz` holds no code.
- **B. Backend bridges become extras.**
  `xreinx`, `xrsklearn` and `xrgrad` are separate distributions; geoops keeps the same jobs as modules (`geoops.einx`, `geoops.learn`).
  They fold into `xrops.einx`, `xrops.learn` and `xrops.grad`, with `[jax]` keeping JAX out of the base install (open question Q1).
- **C. One home for the Operator base.**
  `xrcore` holds the xarray `Operator` and `Signature`; geoops subclasses `pipekit.Operator` directly.
  `xrcore` folds into `xrops._src` (open question Q2).
- **D. Matching family names.** geoops is organised by operation kind, xrtoolz by domain plus a `geo` grab-bag.
  `xrops` takes geoops's family names wherever the job is the same, keeps domain diagnostics only for model states, and drops the empty `rs` and `ice` stubs.
- **E. Readers named as products.**
  `xrreader` (CMEMS, CDS, AEMET) is the xarray twin of `geoproducts` and becomes `xrtoolz-products` (import `xrproducts`).

End state: each repo has a code-free umbrella and the same job-shaped distributions; `geoops` and `xrops` share family names; a reader can predict where a function lives in one lane from where it lives in the other.

:::{note} Read with the later pages
This page set the package split first.
Two of its points were refined by the pages that followed:
the observations-versus-states line drawn at the L3 carrier is superseded by the workload rule in [](./04_geoarray_named_carriers.md) (one item or a fixed batch is a geoarray carrier; many items selected by value is xarray),
and the operator machinery it mentions is specified in [](./05_operators.md) and [](./06_pipekit_named_arrays.md).
:::

## Ownership rules

One axis per level: distributions split by job, modules inside an ops package split by operation kind, and heavy backends sit behind extras.

| Job | geotoolz | xrtoolz |
| --- | --- | --- |
| meta | `geotoolz` | `xrtoolz` (no code) |
| transform | `geoops` | `xrops` (new) |
| products | `geoproducts` | `xrproducts` (was `xrreader`) |
| carry | `geoarray` | xarray · coordax |
| shared | `geocatalog` · `geocloud` · `geopatcher` | `geocatalog` · `geocloud` · `geopatcher` |

Each row is one job; only the names on the right change.
The shared row already serves both lanes.

1. **geotoolz** and **xrtoolz** hold no code.
   Each depends on its lane's distributions and documents the import table.
2. **geoops** and **xrops** own every operator in their lane.
   Family names match where the job matches; a family that exists in one lane only says why in its docstring.
3. **xrops** never imports geoops or geoarray, and geoops never imports xarray in its base install.
   Crossing lanes goes through geoarray's converters ([labeled lane](./02_labeled_lane.md)).
4. **Domain physics** acts on model states in `xrops` (atm, ocn, budgets); on observations it belongs to GeoModels (plumax, radtrax), never to geoops.
5. **geoproducts** and **xrproducts** read provider data into their lane's carrier and never transform it.
6. **Heavy backends** (JAX, einx, scikit-learn) are extras of the ops package, never base dependencies.

## Current-state audit

Six mismatches need action.
The audit read the `main` branches of geotoolz (umbrella 0.12.0) and xrtoolz (0.0.5) on 21 Oct 2026; call-site counts are still to do.

| Symbols | Defined in | Used by | Verdict |
| --- | --- | --- | --- |
| Umbrella | `geotoolz/__init__.py` (no code); `xrtoolz` (code + hard deps on all five siblings) | every install | **Split ownership.** `xrtoolz` is both the ops package and the meta package |
| Operator base | `pipekit.Operator` + `geoops._src.contract`; `xrcore.operator.Operator` (DataTree dispatch) | all operators | **Split ownership.** Base lives in the ops package on one side, its own distribution on the other |
| einx bridge | `geoops.einx`; `xreinx` (dist), with a `xrtoolz.einx` shim | pipelines | **Split ownership.** Same job, module vs distribution |
| sklearn bridge | `geoops.learn`; `xrsklearn` (dist) | pipelines | **Split ownership.** Module vs distribution, and `learn` vs `sklearn` |
| Geometry | `geoops.geom` + `geoops._src.engine`; `xrtoolz.geo` (crs, mosaic, rasterize, vectorize, footprint, masks, wavelets, extremes, detrend, …) | both lanes | **Split ownership.** `xrtoolz.geo` mixes geometry with analysis |
| RS physics | `geoops.indices`, `geoops.radiometry`; `xrtoolz.rs` (empty, plans NDVI and radiance-to-reflectance) | none yet | **True duplicate** in plan. Observations belong to the geoops lane |

Smaller findings:

- **Domain modules.**
  `xrtoolz.atm`, `ocn` and `budgets` act on gridded states; `xrtoolz.ice` is an empty stub. geoops's `plume` and `matched_filter` are already scheduled to leave for plumax and radtrax (radtrax backlog RTX-34, RTX-30).
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
| transform | `geoops` | `xrtoolz` | `xrops` |
| Operator base | `pipekit.Operator` in geoops | `xrcore` | `xrops._src` (Q2) |
| tensor algebra | `geoops.einx` | `xreinx` | `xrops.einx` (Q1) |
| ML bridge | `geoops.learn` | `xrsklearn` | `xrops.learn` (Q1) |
| calculus | — | `xrgrad` | `xrops.grad`, `[jax]` (Q1) |
| geometry | `geoops.geom` | `xrtoolz.geo` | `xrops.geom` + analysis families (Q4) |
| RS physics | `geoops.indices`, `radiometry` | `xrtoolz.rs` (empty) | removed |
| products | `geoproducts` | `xrreader` | `xrproducts` (Q3) |
| carry | `geoarray` | xarray | unchanged |
| read + index, store, patch | `geocatalog`, `geocloud`, `geopatcher` | shared | unchanged |

1. **Create `xrtoolz-ops`.**
   Move every module of the `xrtoolz` distribution into `xrops` with history, tests unchanged apart from imports.
2. **Empty the umbrella.**
   `xrtoolz` depends on `xrtoolz-ops` and `xrtoolz-products`, re-exports nothing but `__version__`, and documents the import table, as `geotoolz/__init__.py` does.
3. **Fold the bridges.**
   Per Q1, move `xreinx`, `xrsklearn` and `xrgrad` into `xrops` subpackages behind extras.
4. **Rename the families.**
   Per Q4, split `xrtoolz.geo`; delete `rs` and `ice`.
5. **Rename the reader.**
   Per Q3, `xrtoolz-reader` becomes `xrtoolz-products`.
6. **Shims.**
   Each old import path re-exports with a `DeprecationWarning` naming its new home, removed in a named 0.X release (Q5).

## Boundary: geoproducts and xrproducts

The line is observations versus states, drawn at the L3 carrier switch: geoproducts reads one acquisition into a `GeoArray`, and xrproducts reads anything already stacked over time, members or levels into xarray.

Provenance alone does not split cleanly.
CMEMS serves both gridded satellite maps and the GLORYS reanalysis, and CDS serves ERA5 and satellite products.
A provenance rule would split one provider's reader across two packages, and would put gridded L3 observations into `GeoArray`, which has no labeled time axis.

| Product | Level | Kind | Package |
| --- | --- | --- | --- |
| Sentinel-2 L1C, Landsat 8/9 L1 | L1 | observation, one scene | geoproducts |
| EMIT L1B, L2B | L1, L2 | observation, one scene | geoproducts |
| GOES ABI, Himawari AHI | L1 | observation, one scan | geoproducts |
| Gridded SLA maps (DUACS) | L4 | observation, stacked in time | xrproducts |
| GLORYS12 reanalysis | L4 | state | xrproducts |
| ERA5 | L4 | state | xrproducts |
| AEMET stations | in situ | observation, time series | xrproducts |

The test for a new reader: does one read return one acquisition on one grid or in sensor geometry?
Then geoproducts.
Does it return a cube with a time, member or level axis built from many files, or a station series?
Then xrproducts.

1. **Both read, neither transforms.**
   A reader returns its lane's carrier on the provider's native grid; regridding and derived fields are ops.
2. **Neither imports the other.**
   A product that needs both (EMIT scenes matched to ERA5 winds) is joined in geocatalog or a pipekit graph.
3. **Rename `xrreader`'s `catalog` module.**
   `xrreader._src.catalog` is a short-name registry (`glorys12.daily` → source and dataset ID), not a catalog.
   It becomes `xrproducts.registry`, so "catalog" means geocatalog everywhere.

Recommended: the L3 carrier rule above.
Alternative: strict provenance (observations versus model output), with each provider reader split across both packages.

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

The split of work, proposed:

1. **xrproducts fetches.**
   Authentication, the provider request, download, cache and opening one dataset.
2. **geocatalog indexes and loads.**
   Many files become rows, rows are queried, and `load_states` builds the cube, as it does today.
3. **The hand-off is rows, not imports.** proposed: xrproducts writes state-catalog rows (or calls `build_state_catalog` on what it downloaded) so a fetch lands indexed. geocatalog never imports xrproducts, and xrproducts depends on geocatalog only through an extra.

Recommended: `load_states` stays in geocatalog, which already serves both lanes (raster rows to `GeoArray`, xarray rows to xarray).
Alternative: move cube assembly into xrproducts and keep geocatalog index-only, which removes xarray from geocatalog's loaders but splits states across two packages.

## The elephants: points, polygons, rasters, states

Each geometry kind gets one carrier per lane.
Changing kind (rasterize, sample, bin) is an op, never the carrier's job.

One correction to the starting picture: geoarray itself only carries grids.
A `GeoArray` is `(time, band, y, x)`, with time as an axis position and no timestamps.
Rasterizing happens in ops that return a `GeoArray`: `geoops.geom.Rasterize`, `RasterizeLike` and `geocatalog`'s `load_vector`. xarray formalizes neither geometry nor time meaning; conventions do (CF, xvec).

| Kind | Examples | Array lane (NumPy / JAX) | Labeled lane (xarray) | At rest |
| --- | --- | --- | --- | --- |
| Raster | Sentinel-2 scene, EMIT L1B | `GeoArray` | `DataArray` on `(y, x)` | COG |
| State | GLORYS12, ERA5, a somax run | `xarray; a carrier slice for H(x) under JAX (see the crossover)` | `Dataset` on `(time, member, level, y, x)` | Zarr, NetCDF |
| Polygons | plume outlines, AOIs, field boundaries | rasterized to `GeoArray` by an op | xvec `DataArray` with a geometry dim | GeoParquet |
| Points | Argo profiles, along-track SSH, AEMET stations, plume source points | proposed `GeoPoints` | `Dataset` on one `obs` dim (CF DSG) | GeoParquet, NetCDF DSG |

### Point clouds: the gap

Points have three shapes today and no carrier:

- geoops's `RasterToPoints`, `PointsToRaster`, `RasterToPointCloud` and `PointCloudToRaster` take a GeoDataFrame or an xvec array.
- `xrtoolz.geo` along-track helpers take bare 1-D lon and lat arrays.
- `geocatalog.matchup` pairs catalog rows, not points.

Proposed, one carrier per lane, mirroring `GeoArray` and xarray:

1. **Labeled lane: a CF Discrete Sampling Geometry dataset.**
   One `obs` dim, with lon, lat, time and depth as coords on it.
   `featureType` names the layout (point, timeSeries, trajectory, profile).
   Ragged profiles use DSG's contiguous ragged layout, one `row_size` per feature.
   Argo and CMEMS along-track files already ship this way, so xrproducts opens them with no conversion.
2. **xvec only on demand.**
   Build an xvec geometry index on `obs` only when a spatial query needs it.
   One shapely object per sample is the cost to avoid at 10⁷ along-track points.
3. **Array lane: `GeoPoints` in geoarray.**
   It holds horizontal coords `xy` `(N, 2)`, an optional vertical coord `z` `(N,)`, values `(C, N)`, time `(N,)`, a CRS and fill, and is a pytree like `GeoArray`. vardax and filterax need it: an observation operator maps a gridded state to values at points inside `jit` and `grad`, and a GeoDataFrame cannot cross `jit`.
   - **Why `z` is separate, not a third column.**
     The vertical coordinate differs in kind, not just in axis: height above the ellipsoid (m, up), ocean depth (m, down), pressure (hPa, not a length), or a model level index.
     One `(N, 3)` array would mix units and directions, and horizontal ops (binning, nearest neighbour, rasterizing) would have to slice it off every time.
   - **What `z` carries.**
     A small descriptor: kind (height, depth, pressure, level), units, CF `positive` (up or down), and datum.
     The CRS is a pyproj compound CRS (horizontal plus vertical) when `z` is a length, e.g. EPSG:4979 for ellipsoidal heights.
   - **A metric 3-D cloud** (lidar such as ICESat-2 or GEDI) keeps `z` as a height and converts to `(N, 3)` ECEF inside the op that needs 3-D distances.
   - **The DSG side already has this.**
     A `z` coord with `axis="Z"`, `units` and `positive`, so `GeoPoints` and a DSG dataset convert one-to-one.
4. **Kind changes are ops with matching names in both lanes.**
   Sample (raster to points), bin or grid (points to raster), rasterize and vectorize (polygons to and from raster). geoarray's converters cross lanes (`GeoPoints` to and from a DSG dataset), as `GeoArray` to and from xarray does today.
5. **At rest.**
   GeoParquet for point tables geocatalog queries with DuckDB; NetCDF DSG or Zarr when a profile or a trajectory is the unit.

Recommended: a DSG dataset plus `GeoPoints`.
Alternatives:

- **One table carrier** (GeoDataFrame or GeoArrow) for points in both lanes.
  It is best for huge sets and DuckDB, but not labeled N-D and not `jit`-able.
- **xvec for points too.**
  One vector story, but one shapely object per point at scale.

The two points left open here, L2 swaths and timestamps on `GeoArray`, are settled in [geoarray: one carrier per geometry](./04_geoarray_named_carriers.md) (`GeoArray`, `GeoSwath`, `GeoPoints`), with named dims instead of positional ones.

## The crossover: when to leave geoarray

geoarray holds one acquisition while a NumPy or JAX kernel runs on it.
Anything indexed by a coordinate value, or assembled from more than one file, is xarray's; a model state inside a JAX model is shaped arrays (L1).
The split is by workload, not data maturity: one item or a fixed batch (possibly under jit) is a geoarray carrier; many items by value is xarray. geoarray stores dims and shares a short list of dim-level functions with xarray, but never selects by value, so it does not grow into a second xarray.

```text
 one item or a fixed batch
 (scene · swath · point set · patches · a state slice)
            │
            ▼
 geoarray carrier + geoops      (L3)
 per-item kernels, possibly under jit:
 radiometry, matched filter, H(x)
            │  to_xarray()   ↑ from_xarray()
            ▼
 xarray + xrops                 (L2)
 many items, by value, lazy:
 stack by date, sel, resample,
 climatology, align, write Zarr
            │  .data + axis
            ▼
 shaped arrays                  (L1)
 somax · vardax · filterax · radtrax
```

The test for any feature or op:

| If it needs | It belongs to |
| --- | --- |
| a coordinate value (a date, a depth, a station ID) | xarray |
| more than one file | xarray, via geocatalog |
| a labeled state under `jit`, `grad` or `vmap` | shaped arrays in the model (L1); a carrier only at its boundary |
| pixel georeferencing for one acquisition under `jit` | geoarray |

Hand-offs:

1. **xarray to geoarray at the `jit` boundary.**
   Select by value in xarray, then `GeoArray.from_xarray(cube.sel(time=...))`.
2. **A geoops op over a time stack** runs through `geoops.lift(op, dim="time")`, behind `geoops[xarray]`: each slice becomes a geoarray carrier, the op runs, the result is restacked.
   It lives in geoops, not xrops, so xrops never imports geoops ([operators page](./05_operators.md), O4).
3. **Observations meet states** in DA as `GeoPoints` against the model state, sliced into a carrier; `geoarray.sample` is the bridge for `H(x)`.

Recommended: this three-way split, with a thin geoarray carrier and coordax only as an optional converter ([geoarray page](./04_geoarray_named_carriers.md)).
Alternative: xarray as the only NumPy-side container, with geometry checks rebuilt in an accessor.

## Open questions

Each has a recommendation and its alternative; flip any before the issues open.

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| Q1 | Backend bridges: extras or distributions? | Fold `xreinx`, `xrsklearn`, `xrgrad` into `xrops` behind extras, matching geoops; `[jax]` keeps JAX out of the base install | Split `geoops.einx` and `geoops.learn` out into distributions instead: more packages, but a symmetric graph |
| Q2 | Where the xarray Operator base lives | `xrops._src`, as geoops keeps its base | Keep `xrcore` if something outside xrtoolz needs the base without the ops |
| Q3 | Reader name | `xrtoolz-products` (import `xrproducts`), matching `geoproducts` | Keep `xrreader` and rename `geoproducts` to `georeader`-style naming instead (clashes with spaceml's georeader) |
| Q4 | How to split `xrtoolz.geo` | Geometry (crs, mosaic, rasterize, vectorize, footprint, subset) → `xrops.geom`; analysis (detrend, wavelets, extremes, regimes) into named families | Keep `geo` as the generic family and rename `geoops.geom` to match |
| Q5 | Shim lifetime | Removed in the release after the switch, named in each warning | Keep the shims through the whole 0.x series |
| Q6 | Domain families | `xrops.atm`, `ocn`, `budgets` stay, because they act on model states; geoops stays domain-neutral | Move them to GeoModels packages next to their physics (somax, plumax) |

## Gates, sequencing and risks

Each workstream merges only when its gate passes; A ships first because every other move lands inside `xrops`.

| Gate | Workstream | Passes when |
| --- | --- | --- |
| A1 Port parity | A | the moved tests pass with only import changes, and the test count equals the old `xrtoolz` suite's |
| A2 Empty umbrella | A | `xrtoolz/__init__.py` exports only `__version__`, and `import xrtoolz` imports no ops module |
| B1 Light base | B | a fresh `pip install xrtoolz-ops` installs none of jax, einx, scikit-learn |
| C1 One base | C | every `xrops` operator is an instance of the one base class, and `xrcore.Operator is xrops.Operator` through the shim |
| D1 Name parity | D | a script lists the families of `geoops` and `xrops`; every name in both means the same job, and the list is checked into the docs |
| E1 Reader shims | E | `import xrreader` warns and resolves every public name to `xrproducts` |

Order: A, then B, C and E in parallel, then D. Each is one minor bump of `xrtoolz` (0.1.0, then 0.2.0).

- **Downstream imports break.** pipekit, geostack notebooks and GeoBench import `xrtoolz.*` today.
  Fallback: shims for one release, and a grep of the stack for old paths before the release.
- **Q1 flips late.**
  If bridges stay distributions, B is a no-op and D still applies.
  Fallback: decide Q1 before A merges.
- **`xrtoolz.geo` has no clean split.**
  Some functions are both geometry and analysis.
  Fallback: keep a thin `xrops.geo` re-export until D1 lists every name.
