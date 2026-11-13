---
title: "One products package for both lanes"
subtitle: "Design · one reader per provider, shared requests, credentials and cache, and each reader returning the carrier its geometry needs"
short_title: Products
description: "geoproducts and xrproducts become one package: provider subpackages over a shared request vocabulary, credentials and cache, a carrier-free base install, and readers that return a GeoArray, a GeoSwath or an xarray Dataset by the geometry of what they read, with geoarray's converters for the other lane."
date: 2026-11-12
keywords:
  - geoproducts
  - xrproducts
  - readers
  - products
  - package layout
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

Today a provider's reader lives in geoproducts if it returns a `GeoArray` and in xrproducts if it returns xarray, so the same plumbing (requests, credentials, caching) is written twice, and a product's home depends on its carrier rather than on its provider. We propose one package that owns every provider, keeps the carrier at the very end of each read, and lets the caller cross lanes with the converters geoarray already has.

- **PD-D1. One package.** `xrtoolz-products` folds into `geotoolz-products` (import `geoproducts`), with one subpackage per provider: `goes`, `himawari`, `carbonmapper`, `toy_sensor`, `cmems`, `cds`, `aemet` and `local`.
- **PD-D2. The geometry picks the carrier.** A reader returns the carrier that fits what it read: a lazy `RasterSource` or `GeoArray` for one acquisition on an affine grid, a `GeoSwath` for a swath, and an `xr.Dataset` for cubes, station series, profiles and trajectories. Crossing lanes is an explicit call to geoarray's converters.
- **PD-D3. A carrier-free base.** The base install holds the request vocabulary, credentials and cache on NumPy and pandas only, and each provider's extra brings its carrier (geoarray, geocloud or xarray) and its client.
- **PD-D4. One request vocabulary.** xrproducts' `BBox`, `TimeRange`, `Request`, `Station` and CF `Variable` registry become `geoproducts.types` and replace geoproducts' tuple bounding box and time helpers.

This page answers PL-Q11 on [package layout](./package_layout.md). If it is decided, it supersedes PL-Q7 (geoproducts or xrproducts by what one read returns) and PL-E (the rename of `xrreader` to `xrproducts`), and the "products" row of the ownership table there has one name in both lanes.

## The packages today

We read the `main` branches of geotoolz and xrtoolz on 12 Nov 2026. Line counts are of the source, and the splits between kinds of code are estimates from per-file counts.

| | geoproducts | xrproducts |
| --- | --- | --- |
| Size | about 13,000 lines | about 9,100 lines |
| Providers | GOES-R ABI, Himawari AHI, Carbon Mapper, a toy sensor | CMEMS, CDS (reanalysis and in situ), AEMET, local methane L2 (TROPOMI, EMIT, GHGSat) and inventories (EDGAR, GFEI, EPA) |
| Base class | `ProductReader`, a `geoarray.RasterSource` on `Grid.from_affine(transform, crs, shape)` | `DataSource`, whose `open()` returns an `xr.Dataset` |
| Returns | lazy windowed reads (`read_window`, `read_from_bounds`, `load`) as `GeoArray`; `GeoSwath` for the toy swath | `xr.Dataset` for every source, of a kind in `DatasetKind`: gridded, alongtrack, profiles, trajectory, stations, swath, scene |
| Request vocabulary | a `(W, S, E, N)` tuple checked by `validate_lonlat_bbox`, which rejects boxes that cross the antimeridian; `as_utc`, `rfc3339_utc`, `time_interval` | `BBox` with longitude conventions and the antimeridian, `TimeRange`, `Request`, `Station`, and a 1,414-line CF `Variable` registry |
| Credentials | `~/.geoproducts/auth_<provider>.json` | loaders for environment variables, `~/.cmems` and `~/.cdsapirc` |
| Cache | none of its own; downloads go through `geocloud.files` | `$XRPRODUCTS_CACHE` or `~/.cache/xrproducts/<source>/<dataset>/<sha1(request)>.nc` |
| Base dependencies | `geotoolz-array[georeader]`, `georeader-spaceml`, rasterio, affine, NumPy | NumPy, pandas, xarray |
| Imported by | nothing outside its own package | seven modules of xrops, all for `xrproducts.types` |

Three findings drive the proposal.

1. **Most of each package is provider code, not carrier code.** About 45% of geoproducts is provider access and IO (the Carbon Mapper API and downloads, the AWS fetchers, the HSD decoder, credentials), against about 40% that builds geoarray carriers. In xrproducts about 1,900 lines of types are carrier-free and most of the rest is provider clients; only the end of each `open()`, `validation` and `subset` need xarray.
2. **The shared plumbing is written twice and disagrees.** The two bounding boxes differ on the antimeridian, the credentials have two designs, and only one package has a cache.
3. **The carrier rule does not sort the products cleanly.** An EMIT or GHGSat scene is one acquisition, which PL-Q7 would put in geoproducts, but its opener sits in xrproducts. Methane products are split across the two packages: Carbon Mapper in one, TROPOMI, EMIT and GHGSat in the other.

## Why the output is not a pipekit-array carrier

The first idea was a reader that knows no carrier and hands its result through pipekit-array, so that the caller picks a `GeoArray` or a `DataArray`. The adapter protocol cannot do that today. `NamedArrayAdapter` in pipekit-ops has `dims`, `data`, `labels`, `isel`, `rewrap(x, data, dims)`, `check_compatible` and `signature`, and the optional `LabelledAdapter` adds `relabel(x, labels)`. Every method takes an existing carrier `x`, so an adapter can rewrap a carrier like one it already has, but it cannot build a `GeoArray` or a `DataArray` from data, dims and coordinates. The neutral carrier, `DimArray`, holds data, dims and labels but no CRS or transform, so a reader that returned one would drop the georeferencing that the caller then has to rebuild.

Adding a constructor to the protocol would not fix the second problem. Station series, profiles, trajectories and swaths on 2-D latitude and longitude have no `GeoArray` form at all, so a reader that promises "any carrier" would fail for half the providers. The geometry of the data, not the caller, decides which carriers can hold it.

## The proposal

(pd-layout)=
### Layout

```text
geoproducts/
├── types/          # BBox, TimeRange, Request, Station, Variable registry (from xrproducts.types)
├── _src/           # credentials, cache, net, s3, query helpers, ProductReader, DataSource
├── goes/           # [goes]
├── himawari/       # [himawari]
├── carbonmapper/   # [carbonmapper]
├── toy_sensor/
├── cmems/          # [cmems]
├── cds/            # [cds], [cds-insitu], [grib]
├── aemet/          # [aemet]
└── local/          # [local]: TROPOMI, EMIT, GHGSat, inventories
```

`import geoproducts` imports only `types` and `_src`, and each provider subpackage is imported on its own, so the base install never loads a carrier or a client.

### The geometry picks the carrier

| What a read returns | Carrier | Providers today |
| --- | --- | --- |
| one acquisition on an affine grid | lazy `RasterSource`, read into `GeoArray` | GOES, Himawari, Carbon Mapper rasters |
| one acquisition on a swath (2-D latitude and longitude) | `GeoSwath` | the toy swath; EMIT and GHGSat scenes are candidates (PD-Q2) |
| a cube over time, levels or members | `xr.Dataset` | CMEMS, CDS reanalysis |
| stations, profiles, trajectories, along-track | `xr.Dataset` (CF discrete sampling geometry, PL-Q9) | AEMET, CDS in situ, CMEMS in situ |
| vectors and tables | GeoDataFrame or shapely | Carbon Mapper plumes and sources, inventories |

The other lane is one call away: `GeoArray.to_xarray()` and `GeoArray.from_xarray(da)`, with the same pair on `GeoSwath` and `GeoPoints` and `from_dataset`, `to_dataset`, `from_datatree` and `to_datatree` for whole datasets. They round-trip through a CF `spatial_ref` coordinate (`crs_wkt`, `CRS.to_cf()` attributes and a GDAL `GeoTransform`), keep the spectral, variable, vertical and time axes as CF coordinates, and keep `_FillValue` and attributes. Their limits stay visible to the caller, which is why the conversion is explicit rather than a flag on every reader:

- `from_xarray` computes dask data, so converting a lazy cube loads it into memory;
- a batched `GeoArray` (one transform per item) raises;
- a grid that is not affine raises.

### A carrier-free base install

The base install depends on NumPy and pandas only and holds `types`, credentials and the cache. Each provider's extra brings its carrier and its client.

| Extra | Brings | Today |
| --- | --- | --- |
| `[goes]`, `[himawari]` | `geotoolz-array`, `geotoolz-cloud[hdf5]` | geoarray and georeader are base dependencies of geoproducts |
| `[carbonmapper]` | `geotoolz-array[georeader]`, rasterio, requests, pydantic, shapely, geopandas | the same, minus the carrier |
| `[cmems]` | xarray, `copernicusmarine` | xarray is a base dependency of xrproducts |
| `[cds]`, `[cds-insitu]`, `[grib]`, `[aemet]`, `[local]` | xarray and the existing clients | unchanged apart from xarray |

The base install is therefore lighter than either package's base today. geotoolz's base-install workflow checks it, with one assertion that neither geoarray, rasterio nor xarray is installed.

### One request vocabulary

`geoproducts.types` is xrproducts' `types` package moved as it is. Its `BBox` replaces the tuple and `validate_lonlat_bbox`, so GOES, Himawari and Carbon Mapper queries gain antimeridian handling, and `TimeRange` replaces `as_utc`, `rfc3339_utc` and `time_interval`. Each provider translates the shared request into its own form, as `BBox.as_cmems()` and `BBox.as_cds_area()` already do; a Carbon Mapper STAC query gets an `as_stac()` of the same kind.

Credentials keep reading the locations users already have (`~/.cdsapirc` is cdsapi's own convention, `~/.cmems` and `~/.geoproducts/auth_<provider>.json`), behind one loader per provider in `_src/credentials.py`. The cache becomes one module for every provider, so the GOES and Himawari downloads and the CDS requests land in the same tree.

### Where it sits

The merged package sits below both ops packages. xrops imports `geoproducts.types` where it imports `xrproducts.types` today, and geoops never imports geoproducts. That needs [geotoolz#641](https://github.com/jejjohnson/geotoolz/issues/641) first: today geoproducts' `[operators]` extra imports geoops so that each sensor's presets can build Operators, which puts geoproducts above geoops. Once the presets are plain data, the merged package depends on geoarray, geocloud and pipekit-ops only, and only through extras.

The stack map proposes `xrcf`, a floor package for the xarray lane that would hold the CF `Variable` registry, CF validation and coordinate conventions. This page keeps the registry in `geoproducts.types`, because the readers use it and its base is light; whether it moves later to a floor package is PD-Q3.

## Migration

The steps assume [one workspace](#design-one-workspace) (PL-F) has landed, so every move is one PR in one repository. Without PL-F the same steps work, but each one that touches xrops is a pair of PRs in two repositories, released in order.

1. **Presets become data.** geotoolz#641 lands, and geoproducts drops its `[operators]` extra.
2. **Types move.** `xrproducts.types` moves to `geoproducts.types` with history, and the seven xrops modules import it from there. There are no shims (PL-Q5): `xrproducts.types` is removed in the same release.
3. **Shared plumbing.** The credentials loaders and the cache merge into `geoproducts._src`, and geoproducts' bounding-box and time helpers give way to `BBox` and `TimeRange`.
4. **The base goes light.** geoarray, georeader and rasterio move from geoproducts' base dependencies into the provider extras, and `geoproducts/__init__.py` stops importing provider subpackages.
5. **The xarray providers move.** `cmems`, `cds`, `aemet` and `local` move from xrproducts into geoproducts with history, each with its tests and its extra.
6. **xrproducts goes.** The `xrtoolz-products` distribution is removed, the `xrtoolz` umbrella depends on `geotoolz-products` instead, and plumax's `io` extra (still pinned to the old `xrtoolz-reader` at rev `a94d9a1`) moves to `geotoolz-products[cds,local]` as part of plumax#249.
7. **Docs.** One products section in the geotoolz docs, organised by provider, with the geometry-to-carrier table above.

## Gates

| Gate | Passes when |
| --- | --- |
| D1 One package | `import xrproducts` fails, every public name it had is importable from `geoproducts`, and the moved tests pass with only import changes |
| D2 Light base | a fresh `pip install geotoolz-products` installs none of geoarray, georeader, rasterio or xarray, and `import geoproducts` works |
| D3 Carrier by geometry | every reader's return type matches the table above, checked by one parametrized test over the providers' fixtures |
| D4 One vocabulary | no module defines a bounding-box or time-range type other than `geoproducts.types`, and for each provider whose coverage crosses the antimeridian, a box across it returns the same data as the two boxes on either side |
| D5 Layering | geostack's dependency graph has no edge from geoproducts to geoops, and no edge from geoops to geoproducts |
| D6 Round trip | for each provider that returns a geoarray carrier, `from_xarray(to_xarray(x))` equals `x` on its fixture |

## Risks

- **User-visible names change.** `XRPRODUCTS_CACHE` and `~/.cache/xrproducts/` become geoproducts names (PD-Q4), so existing caches are downloaded again once.
- **Two lanes' extras in one package.** Installing `geotoolz-products[goes,cmems]` installs both carriers. That is the caller's choice, and the base stays light.
- **The geometry rule has edge cases.** A swath scene can be read either as a `GeoSwath` or as a CF `Dataset` with 2-D coordinates (PD-Q2). The rule picks the carrier that keeps the geometry with the least loss, and the reader documents it.

## Open questions

| ID | Question | Recommended | Alternative | Status |
| --- | --- | --- | --- | --- |
| PD-Q1 | The merged package's name | keep `geotoolz-products` (import `geoproducts`), which already ships at 0.7.1 | a lane-neutral new name such as `products`, which renames every caller of both packages | open |
| PD-Q2 | What EMIT and GHGSat scenes return | `GeoSwath`, as one acquisition on 2-D coordinates, with `to_xarray()` for the labeled lane | keep the flat `xr.Dataset` they return today | open |
| PD-Q3 | Where the CF `Variable` registry lives | `geoproducts.types`, beside the readers that use it | a floor package for the xarray lane (`xrcf` on the stack map) that xrops and geoproducts both import | open |
| PD-Q4 | The cache variable and directory | `GEOPRODUCTS_CACHE` and `~/.cache/geoproducts/<provider>/…`, renamed in the release that merges the cache, with a changelog note | keep the xrproducts names for the merged cache | open |
| PD-Q5 | A carrier constructor in pipekit-ops' adapter protocol | not needed for this page: readers return a concrete carrier and convert through geoarray | add `build(data, dims, labels)` to `NamedArrayAdapter`, so a carrier-neutral reader could target any registered carrier | open |
