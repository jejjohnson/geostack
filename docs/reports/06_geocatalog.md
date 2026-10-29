---
title: geocatalog
subtitle: Report 6 · The standalone spatiotemporal index, as scoped and as built
short_title: 6 · geocatalog
description: geocatalog, the queryable spatiotemporal index over geospatial files. Its design and figures come from the geocatalog repo, its API from the code at geotoolz-catalog 0.5.1. The scoping report's claims are checked against both.
keywords:
  - geocatalog
  - GeoSlice
  - GeoParquet
  - DuckDB
  - catalog
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

A **catalog** is an index over geospatial files. Each row holds one file's footprint, time interval, CRS and path, so a question like "which files overlap this area between these dates?" is answered in milliseconds, **without opening any file**. This report scoped geocatalog as a standalone package extracted from geotoolz. It has been built and has outgrown the scope, since discovery sources, provenance bundles, matchups and staging now sit on top of the index. Most of the scoping decisions held (`GeoSlice` lives in the catalog, there is no pipekit dependency, and the patcher bridge is a Protocol). However, the migration back into geotoolz has not happened and the light base install did not survive, so we leave both as open questions at the end.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Built; originally a scoping proposal ("committed") for the catalog split |
| **Reading time** | about 20 minutes |
| **Audience** | anyone using or extending the catalog, or deciding how geotoolz and xrtoolz should consume it |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit core); [Report 3](03_sister_libraries.md) (sister libraries); [Report 7](07_geopatcher.md) (geopatcher); [Report 9](09_use_cases.md) (use cases); [Report 16](16_geo_data_stack.md) (the geo data stack) |
| **Design and figures** | the [geocatalog repo](https://github.com/jejjohnson/geocatalog)'s docs (concepts, quickstart, design notes) and figures |
| **Code checked against** | `geotoolz-catalog` 0.5.1, import name `geocatalog`, in the geotoolz monorepo at `088475e` (the revision pinned in `references/`). Examples are from real runs |
:::

:::{important} Which geocatalog this page describes
There are two copies of geocatalog. The **geocatalog repo** (to 0.0.3, May to July 2026) holds the design of record, which is its docs and figures. The **`geotoolz-catalog` distribution** in the geotoolz monorepo ([geotoolz#97](https://github.com/jejjohnson/geotoolz/issues/97), 13 July 2026) went through a breaking reorganisation in 0.3.0 (8 October) and is now at 0.5.1. It is the copy geostack pins, so it is the source of this page's **API and behaviour**. The repo's examples use the flat API from before 0.3.0 (`gc.load_raster`, `gc.to_geoparquet`), whereas the examples here use the current namespaced one (`geocatalog.load.load_raster`).
:::

(r6-stack)=
## Where geocatalog sits

The scoping report placed geocatalog beside pipekit and geopatcher, all three above georeader and below geotoolz and xrtoolz (the picture is drawn in [Report 7](07_geopatcher.md#r7-stack); the stack-wide rules are in [Report 16](16_geo_data_stack.md#r16-ownership)). That still holds. geocatalog owns the spatiotemporal index end to end, has no pipekit reference at all, loads geopatcher and geocloud only lazily inside the functions that need them, and consumes georeader rather than replacing it.

:::{warning} Differs from the plan
**Neither geotoolz nor xrtoolz imports geocatalog** at the pinned revisions. The consumer layer that was to "tie them together" (`CatalogPipeline`, `CatalogReader`, re-exports) does not exist yet ([](#r6-migration)).
:::

(r6-what)=
## What is in geocatalog

### The mental model

This is the picture from the repo's own docs. We build a `GeoCatalog` once, from a directory of files, a STAC search, a NASA CMR call or a hand-made list. We then hand it `GeoSlice`s to query, and it hands back smaller catalogs (the matching rows). When we want pixels, a loader takes the slice and the matching catalog and returns a `GeoTensor` or an `xr.Dataset`. At every step the geometry, time and CRS are explicit, so there is no hidden state ([](#fig-r6-dataflow)).

```{figure} ../images/reports/r6_dataflow.svg
:label: fig-r6-dataflow
:alt: geocatalog's data flow from top to bottom. Discovery sources (STACSource, EarthAccessSource, CMRSource, GEESource) and local files feed a CatalogBundle or the builders. Both produce a GeoCatalog, backed by InMemoryGeoCatalog or DuckDBGeoCatalog, which persists to GeoParquet 1.1 and is read back by open_catalog. Querying yields GeoSlices, which load_raster, load_xarray and load_vector turn into GeoTensor or xr.Dataset.
:width: 100%

**Catalogs are the index; `GeoSlice` is the unit of work; loaders materialise it.** Redrawn from the data-flow diagram in the geocatalog repo's concepts page.
```

It works in three layers, in order:

1. **Discover.** `Source` adapters yield `SourceRow`s, which are pre-catalog records that carry the query that produced them, whereas the local builders read filenames and metadata directly.
2. **Index.** A `GeoCatalog` Protocol implementation. Both backends expose the same surface: `query`, `intersect`, `union`, `iter_rows`, `iter_slices`, `total_bounds` and `temporal_extent`.
3. **Materialise.** Loaders take a `GeoSlice` plus a catalog and return a carrier.

### Source layout and public surface

The scoping report counted 11 files and about 3,600 lines from geotoolz's catalog module. The current package is about 14,300 lines, organised since 0.3.0 into ten namespaces by workflow step, each a thin facade over `geocatalog._src` ([](#tbl-r6-surface)):

:::{table} geocatalog's public namespaces at 0.5.1. Names in *italics* need an extra and import lazily.
:label: tbl-r6-surface

| Namespace | Public names |
|---|---|
| `geocatalog` (root) | `GeoCatalog`, `GeoSlice`, `open_catalog`, `query`, `intersect`, `union` |
| `geocatalog.backends` | `InMemoryGeoCatalog`, *`DuckDBGeoCatalog`*, `CatalogRow`, `GeoCatalogError`, `CatalogMetadataError`, `CatalogSchemaError`, `CatalogClosedError` |
| `geocatalog.build` | `build_raster_catalog`, *`build_vector_catalog`*, *`build_xarray_catalog`*, `append_files` |
| `geocatalog.load` | `load_raster`, `aload_raster`, `load_raster_timeseries`, *`load_vector`*, *`load_xarray`* |
| `geocatalog.grid` | `Align`, `GridAlignmentWarning`, `count_steps`, `is_grid_aligned`, `slice_to_window`, `window_to_slice`, `PIXEL_PRECISION` |
| `geocatalog.storage` | `to_geoparquet`, `from_geoparquet`, `migrate_geoparquet`, `sort_geoparquet`, `StreamingParquetWriter`, `CatalogBundle`, `QueryRecord`, *`to_stac_collection`*, `SCHEMA_VERSION_CURRENT` |
| `geocatalog.sources` | `Source`, `SourceRow`, `AuthStatus`, *`STACSource`*, *`EarthAccessSource`*, *`CMRSource`*, *`GEESource`*, *`from_stac_items`*, *`from_stac_search`*, `source_row_to_gdf_row` |
| `geocatalog.matchup` | `matchup`, `MatchupRow`; spatial strategies `Intersects`, `Contains`, `CentroidWithin`, `IouAtLeast`; temporal `Synchronous`, `WithinWindow`, `NearestInTime` |
| `geocatalog.staging` | `stage` |
| `geocatalog.patch` | `CatalogDomain`, `field_for` |
| `geocatalog.utils` | `parse_uri`, `ParsedURI`, `retry_transient_io`, time helpers (`to_naive_utc`, `to_rfc3339`, …) |
:::

There is also a `geocatalog` command line, with `build {raster, xarray, vector}`, `query`, `stats`, `migrate`, `convert` and `info`.

:::{warning} Differs from the plan
The scoping report's surface was **flat**: `from geocatalog import load_raster, to_geoparquet, CatalogDomain, …`. The 0.3.0 reorganisation removed the flat re-exports, with no aliases, so the root now keeps only the core types and the set algebra and everything else is imported from its namespace. The geocatalog repo's docs still show the flat form ([](#r6-todos)).
:::

(r6-geoslice)=
### `GeoSlice`, the unit of work

`GeoSlice` is a frozen dataclass of four fields (`bounds`, `interval` (a closed `pd.Interval`), `resolution` and `crs`), plus an `align` mode that is excluded from equality and hashing. Because it is frozen, a slice can be cached, hashed, sent across processes, or logged and replayed, and to change one we use `dataclasses.replace`. The repo's docs give the reason: "silent mutation of a query that's been logged is the worst kind of bug".

A slice carries everything the next stage needs to fetch a chip without consulting the catalog again. The `shape`, `height`, `width` and `transform` are derived from it, and `to_crs()` reprojects it.

**Grid alignment** is the one subtlety. The bounds are corner coordinates in the slice's CRS, $(x_0, y_0, x_1, y_1) \in \mathbb{R}^{4}$ (map coordinates, not inference variables), and the resolution is $(\Delta x, \Delta y) \in \mathbb{R}_{>0}^{2}$. The pixel counts $n_x, n_y \in \mathbb{N}$ along each axis are then

$$
n_x = \mathrm{round}\left(\frac{x_1 - x_0}{\Delta x}\right)
$$ (r6-eq-shape)

$$
n_y = \mathrm{round}\left(\frac{y_1 - y_0}{\Delta y}\right)
$$ (r6-eq-shape-y)

These expressions silently accept an extent that is not a whole number of pixels ({eq}`r6-eq-shape` rounds half up), which bites at matchup boundaries, where a chip is stacked against a label raster one pixel short. `aligned_shape()` is the strict version, and the `align=` mode chooses what construction does about a residual:

```{code-block} python
:caption: Grid alignment at geocatalog 0.5.1, from a real run.
:linenos:

import pandas as pd
from geocatalog import GeoSlice
from geocatalog.grid import is_grid_aligned

iv = pd.Interval(pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-05"),
                 closed="both")
s = GeoSlice(bounds=(0.0, 0.0, 105.0, 60.0), interval=iv,
             resolution=(10.0, 10.0), crs="EPSG:32630")   # x-extent 105 = 10.5 px

s.shape            # → (6, 11)        rounds half up, silently
s.aligned_shape()  # ValueError: x-extent=105.0 is not an integer multiple of
                   # step=10.0 (nearest is 10 -> residual -5). Use align='snap' ...

snap = GeoSlice(bounds=(0.0, 0.0, 105.0, 60.0), interval=iv,
                resolution=(10.0, 10.0), crs="EPSG:32630", align="snap")
snap.bounds        # → (0.0, 0.0, 110.0, 60.0)   rounds outward, keeps the origin

t = GeoSlice(bounds=(5.0, 0.0, 65.0, 60.0), interval=iv,
             resolution=(10.0, 10.0), crs="EPSG:32630")
is_grid_aligned(s, t, explain=True)
# → {'aligned': False, 'crs_match': True, 'x_res_match': True,
#    'y_res_match': True, 'x_origin_residual': -5.0, 'y_origin_residual': 0.0}
```

The modes are `"off"` (the default), `"warn"` (a `GridAlignmentWarning`), `"error"` and `"snap"`. Snapping extends `xmax` right and `ymin` down, so the affine origin (`xmin`, `ymax`) never moves. Two slices are on the same pixel lattice when they share a CRS and resolution and their origins agree modulo the resolution, which is exactly what `is_grid_aligned` checks.

### Why `GeoSlice` lives here

The scoping report locked in this decision, and the code bears it out:

1. **The catalog is the primary producer.** `catalog.iter_slices(...)` and `CatalogDomain.slices()` yield `GeoSlice`s, and every loader consumes one.
2. **The patcher does not use it.** geopatcher's unit is a `Patch` (data plus anchor plus neighbourhood), not a bounded request, and geopatcher does not import geocatalog.
3. **A separate package would be over-engineering.** `GeoSlice` is about 330 lines.

[](#tbl-r6-geoslice-consumers) shows who imports it, which so far is only geocatalog and user code.

:::{table} Who imports `GeoSlice`, planned and at the pinned revisions.
:label: tbl-r6-geoslice-consumers

| Consumer | Planned | At the pinned revisions |
|---|---|---|
| geocatalog | `geocatalog._src` | `geocatalog._src.geoslice` (exported from the root) |
| geotoolz | catalog-aware operators, samplers, `CatalogPipeline` | **no import**; geotoolz's ruff config bans `geocatalog._src` |
| xrtoolz | catalog-aware xarray loaders | **no import** |
| geopatcher | does not import it | does not import it, as planned |
| user code | `from geocatalog import GeoSlice` | the same |
:::

### The row schema

Every backend shares one row schema ([](#tbl-r6-schema)):

:::{table} The shared row schema, from the geocatalog repo's concepts page.
:label: tbl-r6-schema

| Column | Type | Meaning |
|---|---|---|
| `geometry` | Shapely polygon | the file's footprint, in the catalog's CRS |
| `start_time`, `end_time` (promoted to an `IntervalIndex`) | timestamp | the time interval, closed on both ends |
| `filepath` | str | path or URI to the source file |
| `crs` | str | the file's own CRS, which may differ from the catalog's |
:::

Each kind of catalog adds its own columns:
- **xarray** catalogs add `n_timesteps` and `time_var`;
- **vector** catalogs add `layer`;
- **STAC ingestion** adds `asset_key`, `stac_item_id` and `stac_collection`;
- **bundles** add `id`, `source`, `collection`, `assets` and `provenance`.

The `IntervalIndex` is what makes temporal queries cheap. Combined with an R-tree on `geometry` (in memory) or bbox-column pushdown (DuckDB), `query(slice)` is two cheap index lookups, intersected. Times are held as naive UTC throughout, and footprints ingested from STAC or CMR that cross the antimeridian are split into a MultiPolygon at ±180°.

(r6-backends)=
### Two backends, one Protocol

Both backends implement `GeoCatalog`, and the Protocol is the contract, so anywhere code accepts a `GeoCatalog` either one works (`isinstance` checks work against it). [](#tbl-r6-backends) compares the two, and [](#fig-r6-backends) shows where each one wins on latency.

:::{table} The two backends, from the geocatalog repo's concepts page.
:label: tbl-r6-backends

| | `InMemoryGeoCatalog` | `DuckDBGeoCatalog` |
|---|---|---|
| Install | base | the `[duckdb]` extra |
| Storage | a `GeoDataFrame` in memory | GeoParquet 1.1 on disk, S3 or Hugging Face |
| Indexing | R-tree plus `IntervalIndex` | GeoParquet 1.1 covering bbox column |
| Scale | up to about 10⁵ rows | 10⁶ rows and more |
| Remote URIs | read through fsspec | native, through DuckDB's `httpfs` |
| Mutation | a new instance per operation | a new instance per operation (a lazy SQL relation) |
| Build | eager | streaming, in bounded memory |
:::

```{figure} ../images/reports/r6_backend_comparison.png
:label: fig-r6-backends
:alt: Log-log plot of query latency against catalog rows. The in-memory backend is flat and fast up to about ten to the five rows, then rises steeply. DuckDB starts slower and stays nearly flat. The curves cross around half a million rows.
:width: 90%

**Where each backend wins.** From the geocatalog repo. The curves are illustrative, synthetic rather than measured; the shape is the point.
```

`open_catalog(source, *, engine="auto", …)` picks a backend for a GeoParquet artifact, DuckDB when the extra is installed and in memory otherwise, and `engine="duckdb"` or `"memory"` forces one. A remote DuckDB catalog reads only the row groups a query touches. With `strict=True` it raises `CatalogMetadataError` on incomplete metadata instead of inferring it.

(r6-algebra)=
### Set algebra

`query`, `intersect` and `union` all return **new** catalogs and leave their inputs untouched, so they are safe to compose and to cache ([](#fig-r6-algebra)).

```{figure} ../images/reports/r6_set_algebra.png
:label: fig-r6-algebra
:alt: Three panels. query returns the rows whose footprints overlap a slice. intersect returns the overlap of two catalogs, in space and time. union returns rows from either catalog, deduplicated by file path.
:width: 100%

**Catalog set algebra.** From the geocatalog repo.
```

```{code-block} python
:caption: An in-memory catalog end to end at geocatalog 0.5.1, from a real run.
:linenos:

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from geocatalog import GeoSlice, intersect, open_catalog, query, union
from geocatalog.backends import InMemoryGeoCatalog
from geocatalog.storage import from_geoparquet, to_geoparquet

gdf = gpd.GeoDataFrame(
    {"filepath": ["a.tif", "b.tif"],
     "start_time": pd.to_datetime(["2024-06-01", "2024-06-10"]),
     "end_time": pd.to_datetime(["2024-06-02", "2024-06-11"])},
    geometry=[box(0, 0, 100, 100), box(50, 50, 150, 150)],   # overlap in space,
    crs="EPSG:32630",                                         # not in time
)
cat = InMemoryGeoCatalog(gdf, kind="raster")
cat.total_bounds                  # → (0.0, 0.0, 150.0, 150.0)

s = GeoSlice(bounds=(0.0, 0.0, 60.0, 60.0),
             interval=pd.Interval(pd.Timestamp("2024-06-01"),
                                  pd.Timestamp("2024-06-05"), closed="both"),
             resolution=(10.0, 10.0), crs="EPSG:32630")
list(query(cat, s).gdf["filepath"])        # → ['a.tif']  b.tif misses the window

a = InMemoryGeoCatalog(gdf.iloc[[0]].copy(), kind="raster")
b = InMemoryGeoCatalog(gdf.iloc[[1]].copy(), kind="raster")
len(intersect(a, b))                       # → 0  disjoint in time
len(intersect(a, b, spatial_only=True))    # → 1  the (50, 50, 100, 100) overlap
len(union(a, b))                           # → 2

to_geoparquet(cat, "cat.parquet")          # GeoParquet 1.1, with a bbox column
len(from_geoparquet("cat.parquet"))        # → 2
open_catalog("cat.parquet", engine="memory")   # → an InMemoryGeoCatalog
```

`intersect(…, spatial_only=True)` is the tool for pairing imagery with static labels, which have no temporal overlap by construction.

### Why the loaders live here

The scoping report kept the loaders in geocatalog rather than georeader or geotoolz, and that held:
- **They are catalog API.** The pattern is "the catalog produces slices, a loader consumes them", and both sides ship together.
- **They keep georeader independent.** georeader stays a reader, writer and `GeoTensor` library that knows nothing about catalogs.

`load_raster` reprojects each source into the slice's CRS, range-reads remote files through GDAL, and has an async twin, `aload_raster`. `load_raster_timeseries` derives its time steps from the slice. The scoping report expected the loaders to be extras-gated, but at 0.5.1 only the xarray and vector loaders are lazy and the raster loader is part of the base install ([](#r6-deps)).

(r6-beyond)=
### What shipped beyond the scope

The scoping report described an index with builders and loaders, whereas the current package adds the pieces that turn it into a data-acquisition tool. Most of them come from the repo's [query → matchup → patch design](https://github.com/jejjohnson/geocatalog/blob/main/docs/design/query-matchup.md) ([](#tbl-r6-beyond)):

:::{table} Shipped features the scoping report did not cover.
:label: tbl-r6-beyond

| Feature | Where | What it does |
|---|---|---|
| Discovery sources | `geocatalog.sources` | `STACSource`, `EarthAccessSource`, `CMRSource`, `GEESource` behind one `Source` Protocol, yielding `SourceRow`s with their query |
| Provenance bundles | `geocatalog.storage.CatalogBundle` | items, query records and matchups persisted as one directory (`ingest`, `write_matchups`, `to_directory`, `from_directory`) |
| Matchups | `geocatalog.matchup` | pair rows across catalogs with a spatial strategy (`Intersects`, `IouAtLeast`, …) and a temporal one (`Synchronous`, `WithinWindow`, `NearestInTime`) |
| Staging | `geocatalog.staging.stage` | download a catalog's assets in parallel with retries, through geocloud's `LocalCache` |
| Exact grid alignment | `geocatalog.grid` | `align=` modes, `aligned_shape`, `is_grid_aligned` ([](#r6-geoslice)) |
| Hive partitioning and appends | `to_geoparquet(partition_by=…)`, `build.append_files` | archives that grow; only new rows are written |
| Schema versions | `storage.migrate_geoparquet`, `SCHEMA_VERSION_CURRENT` (0) | forward migrations of the GeoParquet schema |
| STAC export | `storage.to_stac_collection` | a catalog as a STAC collection |
| Remote I/O | `utils.parse_uri`, `utils.retry_transient_io` | URI resolution through GDAL `/vsi` paths or geocloud's filesystems; retries with backoff |
| CLI | `geocatalog …` | `build`, `query`, `stats`, `migrate`, `convert`, `info` |
:::

(r6-deps)=
## Dependencies and extras

The scoping report wanted an index-only install with no GDAL, made of `pandas`, `geopandas`, `pyproj`, `shapely` and `rasterio` (for `GeoSlice`'s affine transforms and windows). Loaders, DuckDB and streaming would be extras. [](#tbl-r6-base) and [](#tbl-r6-extras) set that against what shipped.

:::{table} The base install, scoped and shipped.
:label: tbl-r6-base

| Scoped base | Shipped base at 0.5.1 |
|---|---|
| pandas, geopandas, pyproj, shapely, rasterio | pandas, geopandas, pyproj, shapely, rasterio, **georeader-spaceml**, **pyarrow**, numpy, typing-extensions, loguru, cyclopts, tenacity |
:::

:::{table} The optional extras, scoped and shipped.
:label: tbl-r6-extras

| Scoped | Shipped |
|---|---|
| `duckdb = [duckdb]` | `duckdb = [duckdb>=1.1]` (1.1 for bbox-column pushdown) |
| `raster = [georeader]` | none: georeader is in the base |
| `xarray = [xarray, rioxarray]` | `xarray-raster = [xarray, rioxarray, h5netcdf, zarr]` |
| `vector = [pyogrio]` | none: pyogrio arrives with geopandas |
| `streaming = [pyarrow]` | `streaming = [duckdb]`: pyarrow is in the base |
| — | `cloud = [geotoolz-cloud[fsspec]]`, `stac`, `earthaccess`, `gee`, `sources-all`, `patch = [geotoolz-patcher>=0.6]`, `full` |
:::

:::{warning} Differs from the plan
The light install does not exist, as `pip install` of geocatalog brings in georeader, rasterio and pyarrow. So someone who wants only the index over their own files pays for the raster stack. Some docstrings still mention a `[vector]` extra that is not defined. See [](#r6-open-questions).
:::

**No pipekit dependency.** This held exactly, and it is the cleanest decision in the report. The catalog is not an `Operator` but a queryable index: pipekit composes operators that transform a carrier, whereas geocatalog finds which files match a query. We can open a catalog without pipekit, and a pipekit pipeline can read from a source that is not a catalog. The two meet in the consumer libraries, which is where the missing operators belong ([](#r6-migration)).

(r6-bridge)=
## The patcher bridge

`CatalogDomain` lets a geopatcher patcher walk a catalog without either package importing the other, and it shipped as designed in that respect. geopatcher's `Domain` is a `runtime_checkable` Protocol with two members, `crs` and `bounds`. `CatalogDomain` has both, so it satisfies `Domain` structurally: `isinstance(CatalogDomain(...), geopatcher.Domain)` returns `True` at runtime, and neither package imports the other at module level.

```{code-block} python
:caption: CatalogDomain at geocatalog 0.5.1, from a real run on the catalog above.
:linenos:

from geocatalog.patch import CatalogDomain

dom = CatalogDomain(cat, resolution=(10.0, 10.0))
dom.bounds, dom.crs.to_epsg()    # → ((0.0, 0.0, 150.0, 150.0), 32630)
len(dom.slices())                # → 2  one GeoSlice per catalog row
```

`field_for` is the second half of the bridge, which the scoping report did not have. With the `[patch]` extra it hands a catalog to geopatcher as a `RasterField` (or a list of them), importing geopatcher lazily inside the call:

```{code-block} python
:caption: Catalog-driven iteration meeting patch-driven iteration (a sketch; needs the [patch] extra).
:linenos:

from geocatalog.patch import field_for

field = field_for(catalog, slice_, asset="visual")   # → geopatcher.RasterField
# geopatcher's SpatialPatcher then splits `field` into patches.
```

:::{warning} Differs from the plan
The scoping report gave `CatalogDomain` a `rows_for(anchor)` method. The shipped class is a frozen dataclass `CatalogDomain(catalog, resolution)` with `slices()` instead, which returns every row's slice at that resolution. The geocatalog repo's concepts page names `geotoolz.patch.SpatialPatcher` as the canonical consumer, but the patcher lives in geopatcher and geotoolz only bridges to it through `patch_ops` ([Report 3](03_sister_libraries.md#r3-geotoolz)).
:::

(r6-migration)=
## Migration from geotoolz

The scoping report planned a transitional period with three steps:
- move `geotoolz.catalog` and `geotoolz.types.GeoSlice` into geocatalog;
- have geotoolz re-export everything (including `import geocatalog as catalog`) for two minor versions, then deprecate the re-exports;
- keep only the pipekit-shaped operators in a new `geotoolz.catalog_ops`.

```{code-block} python
:caption: The consumer operators the scoping report planned for geotoolz (not built).
:linenos:

class CatalogPipeline(Operator):
    """ETL driver: iterate a catalog, apply an op per scene, maybe in parallel."""
    def __init__(self, catalog: GeoCatalog, op: Operator, *, n_workers=1): ...

class CatalogReader(Operator):
    """Read one GeoSlice into a GeoTensor, inline in a Sequential."""
```

The first step happened and the rest did not ([](#tbl-r6-migration)):

:::{table} The migration steps at the pinned revisions.
:label: tbl-r6-migration

| Step | Status |
|---|---|
| Catalog code moves out of geotoolz into its own package | **done**: geocatalog, now `geotoolz-catalog` in the same monorepo |
| `GeoSlice` becomes `geocatalog.GeoSlice` | **done** |
| geotoolz re-exports the catalog (`gz.open_catalog`, `gz.catalog`, `geotoolz.types.GeoSlice`) | not done; geotoolz has no catalog names and no `types` module |
| Deprecation over two minor versions | not needed: there was no compatibility layer to deprecate |
| `CatalogPipeline`, `CatalogReader` in `geotoolz.catalog_ops` | not built |
| xrtoolz depends on geocatalog | not done |
:::

The effect is that geocatalog is used **on its own**: we open a catalog and load slices ourselves, then hand the results to geotoolz or xrtoolz operators. The scoping report's cross-library examples still work in that form, with today's names:

```{code-block} python
:caption: One catalog feeding both domain libraries, in today's names (a sketch).
:linenos:

from geocatalog import open_catalog
from geocatalog.load import load_raster, load_xarray

cat = open_catalog("s3://bucket/scenes.parquet")      # DuckDB if installed
for sl in cat.iter_slices(resolution=(60.0, 60.0)):   # one GeoSlice per row
    gt = load_raster(cat, sl)                         # → GeoTensor, for geotoolz
    ...

for sl in reanalysis.iter_slices(resolution=(0.25, 0.25)):
    ds = load_xarray(reanalysis, sl)                  # → xr.Dataset, for xrtoolz
    ...
```

`CatalogPipeline` is now assumed by three reports ([2](02_pipekit_surface.md#r2-group-j), [3](03_sister_libraries.md#r3-todos) and this one) and by use cases [2](09_use_cases.md#r9-case-2) and [12](09_use_cases.md#r9-case-12), and it exists nowhere, which makes it the most important open item here.

(r6-monorepo)=
## The monorepo development model

The scoping report chose to develop everything in one "pipekit-ecosystem" monorepo with uv workspaces and publish separate packages, for four reasons:
- cross-package changes are common early on;
- `GeoSlice` changes ripple through every consumer;
- fixtures can be shared;
- one CI can run the integration matrix.

What exists is **three** monorepos, each a uv workspace with per-package releases ([](#tbl-r6-monorepos); [Report 16](16_geo_data_stack.md) maps every distribution in them):

:::{table} The workspaces at the pinned revisions.
:label: tbl-r6-monorepos

| Workspace | Packages |
|---|---|
| pipekit | `pipekit`, `pipekit-array`, `pipekit-cycle`, `pipekit-evaluate`, `pipekit-experiment`, `pipekit-jax`, `pipekit-train` |
| geotoolz | `geotoolz`, `geotoolz-catalog` (geocatalog), `geotoolz-cloud`, `geotoolz-patcher`, `geotoolz-products` |
| xrtoolz | `xrtoolz`, `xrtoolz-core`, `xrtoolz-einx`, `xrtoolz-grad`, `xrtoolz-reader`, `xrtoolz-sklearn` |
:::

The operating pattern held inside each workspace. Packages resolve to local paths during development, and release-please cuts per-package tags (`geotoolz-catalog-v0.5.1`). Cross-package pins appear in published metadata, for example `patch = [geotoolz-patcher>=0.6]`. The catalog landed in the geotoolz workspace, next to the patcher and cloud I/O it works with most, rather than beside pipekit. geostack itself is where all three workspaces meet (as submodules in `references/`).

(r6-timing)=
## Effort and timing

The scoping report estimated a week of focused work, done in one coordinated v1.0 release alongside the pipekit and geopatcher extractions, so that users would absorb one breaking change rather than three. The record is in [](#tbl-r6-history).

:::{table} geocatalog's history.
:label: tbl-r6-history

| When | What |
|---|---|
| May 2026 | standalone geocatalog repo, through 0.0.2 (25 May): the index and both backends, builders, loaders, sources, bundles, matchups, staging |
| late May – June 2026 | Hive partitioning, fsspec URIs, retries, STAC builders, obstore, exact grid alignment (0.0.3, 9 June) |
| 13 July 2026 | geocatalog and geopatcher merge into the geotoolz monorepo (`geotoolz-catalog` 0.1.0) |
| 8 October 2026 | 0.3.0: organised by workflow step; flat re-exports removed (breaking) |
| 12 October 2026 | 0.5.1, the pinned revision |
:::

There was no single v1.0. The extraction was gradual, with breaking changes absorbed per package (geocatalog 0.3.0, geopatcher's matching reorganisation), and geotoolz is at 0.10.2. The split unblocked what the report hoped, which is that catalog backends, schema migrations and new sources are released on their own cycle. There have been twelve `geotoolz-catalog` releases since July (0.1.0 to 0.5.1), each tagged on its own.

(r6-tradeoffs)=
## Trade-offs

The scoping report's list is below, with how each item turned out.

**What got better:**
1. **Discoverability**, partly. The repo and the import name are `geocatalog`, but the distribution from the monorepo is `geotoolz-catalog`, which is not what someone searching for a catalog would find ([](#r6-open-questions)).
2. **A minimal install for narrow use**: this did not happen ([](#r6-deps)).
3. **`GeoSlice` as a cross-library type**: it is `geocatalog.GeoSlice`, but no other library imports it yet.
4. **Independent backend work**: this worked, as DuckDB parity, GeoParquet 1.1 pushdown and schema migrations all shipped in catalog releases.

**What got worse, and the mitigations:**
1. **One more package to release**: release-please makes it one tag.
2. **Version pinning between packages**: this is largely moot, because geotoolz does not depend on geocatalog yet.
3. **Documentation has to live somewhere**: it lives in the standalone repo (mkdocs, with the figures and the Lake Tahoe notebook), separate from the code since July.
4. **The Protocol bridge is implicit**: it is documented in the repo's concepts page, and we can check it with `isinstance`.

(r6-open-questions)=
## Open questions

1. **Which repo is canonical?** The code lives in the geotoolz monorepo, whereas the docs, figures and notebook live in the standalone repo, frozen at 0.0.3. Either the docs move into the monorepo (and the standalone repo is archived with a pointer), or the standalone repo becomes the docs site for `geotoolz-catalog`.
2. **`geocatalog` or `geotoolz-catalog` on PyPI?** Resolved: `geotoolz-catalog` ([Report 16 §F](16_geo_data_stack.md#r16-ws-f)).
3. **Should there be an index-only install?** The scoped base had no georeader, and no rasterio beyond what `GeoSlice` needs. If the narrow use case matters, we would move georeader and the raster loader behind an extra.
4. **Where do `CatalogPipeline` and `CatalogReader` live?** They could live in geotoolz, as scoped, which needs geotoolz to depend on geocatalog. Alternatively they could live in an optional `geocatalog[pipekit]` module, which breaks the no-pipekit rule only behind an extra. Reports [2](02_pipekit_surface.md#r2-group-j) and [3](03_sister_libraries.md#r3-todos) both depend on the answer.
5. **Should geotoolz and xrtoolz adopt geocatalog at all?** Today the catalog is used standalone and the domain libraries never see a `GeoSlice`. The scoped re-exports may not be worth a dependency, whereas catalog-aware operators might be.
6. **Is `slices()` the right bridge?** `CatalogDomain.slices()` returns every row's slice eagerly, whereas the scoped `rows_for(anchor)` was lazy and query-driven, which matters for large catalogs walked by a patcher.
7. **A third backend?** The scoping report floated PostGIS, but nothing in the design notes picks it up, and DuckDB over GeoParquet covers the scale cases so far.

(r6-todos)=
## Upcoming work

The follow-ups are listed in [](#tbl-r6-todos).

:::{table} Follow-ups from checking the scoping report against the repo and the code.
:label: tbl-r6-todos

| Follow-up | Where | Why |
|---|---|---|
| Fix the data-flow figure: the arrowheads cover the box labels | geocatalog repo, `docs/assets/make_diagrams.py` | this page redraws it instead |
| Regenerate the Lake Tahoe NDVI composite, which renders as an empty plot | geocatalog repo, `docs/assets/lake_tahoe_ndvi_composite.png` | the quickstart's result figure shows no data |
| Replace the illustrative backend curves with measured ones | geocatalog, from `tests/bench/` | the figure says "synthetic, not measured" |
| Update the docs' examples to the namespaced 0.3.0 API | geocatalog docs | `gc.load_raster`, `gc.to_geoparquet` no longer exist |
| Point the concepts page at `geopatcher.SpatialPatcher` | geocatalog docs | it names `geotoolz.patch.SpatialPatcher` |
| Remove the `[vector]` extra from docstrings | `geocatalog._src.formats.vector` | the extra is not defined |
| Index geocatalog's namespaces in the capability index | geostack `scripts/capabilities.py` | only the root's six names are listed ([Report 3](03_sister_libraries.md#r3-todos)) |
| Build `CatalogPipeline` / `CatalogReader` once question 4 is settled | geotoolz or geocatalog | use cases [2](09_use_cases.md#r9-case-2) and [12](09_use_cases.md#r9-case-12); Reports [2](02_pipekit_surface.md#r2-group-j), [3](03_sister_libraries.md#r3-todos) and 6 |
:::

(r6-references)=
## References

### Sources

- The [geocatalog repo](https://github.com/jejjohnson/geocatalog): concepts, quickstart, recipes and the design notes on [exact grid alignment](https://github.com/jejjohnson/geocatalog/blob/main/docs/design/exact-grid-alignment.md) and [query → matchup → patch](https://github.com/jejjohnson/geocatalog/blob/main/docs/design/query-matchup.md); we took the set-algebra and backend figures on this page from there.
- `geotoolz-catalog` 0.5.1 in the geotoolz monorepo, and its changelog.

*Citations to be added:* GeoParquet 1.1, DuckDB, STAC, and NASA CMR, as entries in `references.bib`.
