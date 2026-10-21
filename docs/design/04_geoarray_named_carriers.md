---
title: "geoarray: one carrier per geometry"
subtitle: "Design · named dims, standard coordinates and readable types for GeoArray, GeoSwath and GeoPoints"
short_title: Named carriers
description: "geoarray stores named dims, ships standard coordinates and readable geo types, keeps geometry as pytree leaves so a new scene never recompiles, and hands over to xarray for anything selected by value."
date: 2026-10-21
keywords:
  - geoarray
  - named dims
  - GeoImage
  - GeoPoints
  - GeoSwath
  - JAX
  - xarray
  - coordax
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Decision

geoarray is the geometry vocabulary of the stack: one family of thin carriers for NumPy and JAX, standard coordinates, readable type names, and a short list of dim-level functions that also work on xarray carriers.
Two independent questions organise everything: what an operation needs to know (its layer, a row) and how much data it runs on (the carrier family, a column).

### Two words: carriers and operations

A **carrier** is a data structure: values plus their description.
There are two families, and xarray's data structures are carriers just as much as geoarray's.

| Family | Carriers | Holds | Good at |
| --- | --- | --- | --- |
| geoarray | `GeoArray`, `GeoSwath`, `GeoPoints` | NumPy or JAX data, dims, static coordinates, a geometry leaf (transform, lat/lon, xy), CRS | one item or a fixed batch: strict grids, compiles under `jit`, \~1–10 µs per op |
| xarray | `DataArray`, `Dataset`, `DataTree` | NumPy or dask data, dims, coordinate values, attributes; `Dataset` groups variables on shared dims, `DataTree` nests groups | many items: selection by value, laziness, out-of-core, variables with different dims |

geoarray has no `Dataset` or `DataTree` equivalent on purpose: a carrier holds one stack of values (several bands or variables share one `band` or `variable` dim).
Collections of differently shaped variables are xarray's job.

### Operations live in their own docs

This page covers the data.
How operations are organised (the grid of layers × carrier families, what geoops and xrops each write, kernels written once) is in [GeoStack operators: geoops and xrops on one kernel model](./05_operators.md).
The shared machinery (the named-array adapter, the single `Signature`, the dim-level functions, `KernelSpec` and `KernelOp`) is in [pipekit: named arrays, one Signature, kernels written once](./06_pipekit_named_arrays.md).

In one line: layers are rows (L1 shapes, L2 dim names, L3 geometry), carrier families are columns (geoarray, xarray), and an L1 kernel wrapped as a `KernelOp` runs in both columns.

### Indicative costs (not yet reproducible)

One exploratory run on a CPU sandbox, a 4-op normalized difference and a window, a mock thin NumPy carrier versus `xr.DataArray`.
No script or output from that run is committed, so these numbers motivate the design but are not results: gate G9 replaces them with a committed benchmark and its output, and the claims below stand or fall with it.

| Shape | ND op, carrier | ND op, xarray | Window, carrier | Window, xarray |
| --- | --- | --- | --- | --- |
| 2 × 64 × 64 | 12 µs | 2,257 µs | 1.4 µs | 80 µs |
| 2 × 256 × 256 | 275 µs | 2,779 µs | 1.0 µs | 63 µs |
| 2 × 2000 × 2000 | 48 ms | 52 ms | 2.4 µs | 96 µs |

- In that run, on a full scene the difference was noise; on patches xarray was 10–200× slower per op (to be confirmed by G9).
- Two grids whose x coordinates differ by 10⁻⁷ m, added in xarray, gave shape `(1, 4, 0)`: every pixel dropped, no error.
- `import xarray` took about 0.57 s, against 0.11 s for NumPy.

The carrier earns its place by what it refuses (mismatched grids, a transform drifting from its coordinates, ambiguous pixel edges) and by compiling, not by speed on large scenes.

:::{note} How this page relates to the earlier design pages
It builds on [](./01_geoarray.md) (the GeoTensor → GeoArray design) and supersedes its positional `DIMS[-ndim:]` convention with stored, named dims.
It also changes one decision in [](./02_labeled_lane.md): coordax is an optional converter only, not the foundation of the JAX lane, so that page's coordax `AffineGrid` workstream is superseded (static coordinates would recompile per scene).
Its CF `spatial_ref` + `GeoTransform` encoding still stands, and is the recommended answer to Q2 here.
:::

## The carrier

One class per geometry kind, each holding NumPy or JAX data, stored dims, static coordinates, a georeference leaf and a static CRS.
Today's `GeoArray` already carries either backend as a pytree; the change is replacing `DIMS[-ndim:]` with dims it stores.

```python
@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=["data", "transform"],
    meta_fields=["dims", "coords", "crs", "fill_value"],
)
@dataclasses.dataclass(frozen=True)
class GeoArray:
    # np.ndarray or jax.Array, ending in (y, x)
    data: Float[Array, "... H W"]
    # T = (a, b, c, d, e, f) per frame or per patch; a leaf, so it varies under jit
    transform: Float[Array, "*batch 6"]
    # ("time", "band", "y", "x"); static
    dims: tuple[str, ...]
    # one geoarray.coords coordinate per non-spatial dim; static
    coords: tuple[Coordinate, ...]
    # fixed for a compiled program
    crs: str
    fill_value: float | None = None
```

**Static or leaf.**
What fixes a compiled program (dims, sizes, band and level labels, CRS) is static.
Geometry that changes per call (the transform, swath lat/lon, point xy and z, time stamps) is a leaf, so a new scene, patch or day of observations never recompiles.

| Concern | Owner |
| --- | --- |
| dims, labels, `isel`, exact `sel`, `reduce`, `map_over` | the carrier itself (small, ours) |
| transform, CRS, grid checks, swath and point geometry | the carrier |
| many files, value selection, laziness, dask | xarray, via `to_xarray()` / `from_xarray()` (`[xarray]` extra) |
| labeled ops inside model code (`cmap`) | coordax, via `to_coordax()` / `from_coordax()` (`[coordax]` extra), only if a model wants it |

The backend is the data's type.
An op given one NumPy and one JAX carrier refuses and names `to_jax()`; nothing is promoted silently.

Dependencies: the base install is numpy, pyproj, affine.
`[jax]` adds jax; `[xarray]` adds xarray; `[coordax]` adds coordax and is not needed by any geotoolz package.

## One vocabulary over NumPy, JAX and xarray

The type names and a short list of dim-level functions work on a geoarray carrier (NumPy or JAX) and on an `xr.DataArray`.
The functions are written once, in pipekit-array, over a named-array adapter that each carrier family registers; geoarray registers its own and re-exports the functions, so `geoarray.reduce` and `pipekit_array.reduce` are the same object.

The full op surface is not shared: the backends differ in four behaviours, not one.

| Behaviour | Carrier, NumPy | Carrier, JAX | xarray |
| --- | --- | --- | --- |
| two inputs on different grids | refuses | refuses | inner-joins silently in plain arithmetic |
| laziness | eager | traced under `jit` | often dask-lazy; `.data` can trigger a full compute |
| compiles | no | yes | no |
| per-op overhead (indicative, G9) | \~1–10 µs | \~1–10 µs eager | \~60–2,000 µs |

### What geoarray contributes

| Piece | Where it lives |
| --- | --- |
| `isel`, `sel` (exact), `transpose`, `reduce`, `argreduce`, `take_along`, `map_over`, `stack`, `concat`, `combine` | pipekit-array, re-exported by geoarray |
| the geoarray adapter: dims, data, labels, a `rewrap` that keeps the georeference, a `check_compatible` that compares grids and raises `GridMismatchError`, `map_over` over data and transform rows together | geoarray |
| geometry functions: `sample` (H(x)), windows that move the transform, `to_rows` / `from_rows` for pixels as rows, `to_pixel` | geoarray |
| type names and `check_dims`, built on `pipekit.Signature` patterns | geoarray (names), pipekit (rule) |

On xarray, `combine` and `stack` use the xarray adapter's exact comparison, so the silent-alignment trap does not apply to them; the alignment difference is one explicit check per carrier family instead of a hidden default.
The rule for adding a shared function is in the [pipekit page](./06_pipekit_named_arrays.md): it never reads coordinate values or geometry and behaves identically on every carrier.

### Composites are two ops

```text
many items ──stack──▶ one series ──reduce over "time"──▶ one image
           assembly                   dim-level math
```

A median composite is `stack` then `reduce`.
A max-NDVI or cloud-free composite picks a time index per pixel from one quantity and gathers the other bands at it: `argreduce` then `take_along`.
The math is the same on every backend; the execution differs (eager in memory, or chunked with dask for 100 scenes of 10k × 10k).

## Geometry kinds and dims

A new class exists only when the georeferencing changes; everything else is a named dim.

| Carrier | Georeference leaf | Spatial dims (last) | Examples |
| --- | --- | --- | --- |
| `GeoArray` | transform `(…, 6)` | `y x` | Sentinel-2 scenes, gridded SLA maps, a model grid |
| `GeoSwath` | lat, lon `(…, row, col)` | `row col` | EMIT, TROPOMI, AVIRIS in sensor geometry |
| `GeoPoints` | xy `(obs, 2)`, optional z `(obs,)` | `obs` | Argo cycles, along-track SSH, stations, plume sources |

Other dims come before the spatial ones.
With coordax underneath, any name works; these are the conventions geoops and the converters expect:

| Dim | Means | Labels | xarray name |
| --- | --- | --- | --- |
| `time` | acquisition or valid time | `datetime64` | `time` |
| `member` | ensemble member | strings | `member` |
| `level` | a vertical grid shared by every pixel | layer index or nominal value, with a `Vertical` | `level` (maps CF `depth`, `lev`, `plev`, `layer`) |
| `band` | spectral band or variable | names, optional wavelength in nm | `band` |
| `patch` | patches from geopatcher | none | — |

**Level or z?**
The test is whether samples share the vertical grid:

- **Shared grid → a `level` dim.**
  TROPOMI's 12 retrieval layers, IASI or AIRS pressure levels, EN4 depths, GPM range bins.
  When layer pressures move with surface pressure, `level` is the layer index and the pressures are data in their own band.
- **Each sample its own position → `GeoPoints.z`.**
  Argo, CTD and glider samples at irregular depths, lidar returns.

`z` is never a third column of `xy`: height (m, up), depth (m, down) and pressure (hPa) differ in units and direction, so `z` carries a `Vertical` (kind, units, CF `positive`, datum).

## Standard coordinates and readable types

geoarray ships the coordinates every geoscience array needs, so users never build `LabeledAxis("band", …)` by hand, and a set of readable type names for the common carrier shapes.
Both follow the static-or-leaf rule: what is fixed for a compiled program lives on the coordinate; what varies per call stays a leaf on the carrier.

### Coordinates (`geoarray.coords`)

| Coordinate | Dims | Static content | Leaf on the carrier |
| --- | --- | --- | --- |
| `PixelGrid` | `y x` | shape, CRS, pixel convention (area or point) | transform `(…, 6)` |
| `SwathGrid` | `row col` | shape, CRS; rows along track, columns across | lat, lon `(…, row, col)` |
| `CurvilinearGrid` | `y x` | shape, CRS, fixed lat and lon (a model grid such as ORCA) | none |
| `PointAxis` | `obs` | N, CRS | xy `(obs, 2)`, optional z `(obs,)`, optional time `(obs,)` |
| `SpectralAxis` | `band` | names, centre wavelength (nm), optional FWHM (nm) | none |
| `VariableAxis` | `variable` | names and units of physical quantities (K, hPa, m s⁻¹) | none |
| `VerticalAxis` | `level` | level values + a `Vertical` (kind, units, CF `positive`, datum) | none; per-pixel pressures are data in a band |
| `TimeAxis` | `time` | size only, under `[jax]` | stamps as an `int64` leaf (`datetime64[ns]`) |

`band` and `variable` are separate on purpose.
Spectral channels share units and have wavelengths, so `NDVI` or a matched filter makes sense over them.
Station or model variables carry different units, and an op that mixes channels must refuse them.

**Several variables per point.**
A weather station reporting temperature, pressure and wind is one point with a `variable` dim: `GeoPoints["time variable obs"]` for 40 stations reporting hourly.
A variable a station does not report is the fill value, masked as today.

**Time as a dim or per sample.**
Shared stamps are a `time` dim with a `TimeAxis`: a stack of scenes, fixed stations reporting hourly.
A moving platform has one time per sample, so time is a `(obs,)` leaf on `GeoPoints`, not a dim: along-track altimetry, Argo, ship tracks.

`TimeAxis` is the one that splits.
Its stamps change from batch to batch, so under `[jax]` they are a leaf, never a static label; otherwise every new set of dates would recompile.
On the base install they are ordinary labels.

Each coordinate is a frozen, hashable dataclass with `dims` and `shape`, static under `jit`.
`to_xarray()` writes them as CF coordinates and attributes (`band` names with `wavelength`, `level` with `units` and `positive`, the grid mapping with the transform); `to_coordax()` maps them to coordax coordinates for model code that wants them.

```python
from geoarray.coords import PixelGrid, SpectralAxis, TimeAxis

# Sentinel-2 SWIR: names and centre wavelengths
swir = SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0))

# a 600 × 600 grid in UTM 13N, area-convention pixels
grid = PixelGrid(shape=(600, 600), crs="EPSG:32613")

# 12 acquisition dates
dates = TimeAxis(stamps=acquired)

# a carrier from coordinates, data and the transform leaf
stack: GeoImageSeries = GeoArray.from_coords(data, dates, swir, grid, transform=T)
```

### Readable types (`geoarray.types`)

The names are aliases for a carrier plus its dims, not subclasses.
Slicing a `GeoTimeSeries` at one date simply is a `GeoImage`, and `check_dims` verifies the dims at the call.

| Type | Carrier and dims | Typical data |
| --- | --- | --- |
| `GeoMap` | `GeoArray["y x"]` | one band: a matched-filter score, a mask, an XCH₄ map |
| `GeoImage` | `GeoArray["band y x"]` | one multispectral scene |
| `GeoImageSeries` | `GeoArray["time band y x"]` | frames over time, like a video: a Sentinel-2 stack |
| `GeoVolume` | `GeoArray["level y x"]` | gridded profiles on shared levels (EN4) |
| `GeoPatches` | `GeoArray["patch band y x"]` | a geopatcher batch |
| `GeoSwathImage` | `GeoSwath["band row col"]` | EMIT or AVIRIS L1B in sensor geometry |
| `GeoSwathProfile` | `GeoSwath["level row col"]` | TROPOMI averaging kernels and priors |
| `GeoTimeSeries` | `GeoPoints["time variable obs"]` | fixed locations over shared times: weather stations, pixel series sampled from a stack |
| `GeoTracks` | `GeoPoints["variable obs"]` + time leaf | moving platforms: along-track altimetry, ship tracks |
| `GeoProfiles` | `GeoPoints["variable obs"]` + z leaf | Argo, CTD, gliders |

**A time series is not a video.**
An image series is frames on a grid (`time band y x`).
A time series is values at fixed locations (`time variable obs`), which is what "time series" means in geo usage.
Sampling a `GeoImageSeries` at station points gives a `GeoTimeSeries`.

**Decided, 21 Oct 2026:** `GeoImageSeries` names frames on a grid and `GeoTimeSeries` names fixed locations; `GeoVideo` is not used.
`band` (spectral channels) and `variable` (physical quantities with their own units) stay separate dims.
Tracked in geotoolz #521 and #518.

```python
from geoarray.types import GeoImage, GeoImageSeries, GeoMap


@check_dims
def ndvi(scene: GeoImage) -> GeoMap:
    nir: GeoMap = scene.sel(band="B8")
    red: GeoMap = scene.sel(band="B4")
    return combine(_normalized_difference, nir, red)


# one date of a series is an image, with no conversion
first: GeoImage = stack.isel(time=0)

# map over time: a series of maps
series: GeoArray["time y x"] = geoarray.map_over(ndvi, stack, dim="time")
```

`GeoArray` is generic over its array type today (`GeoArray[np.ndarray]`).
The string subscript replaces that use, since the backend now comes from the field.

## Demo API

Names are proposed.
Running example: Sentinel-2 and EMIT for methane, TROPOMI for a level axis, along-track SSH and AEMET stations for the ocean and atmosphere.

### 1. The problem today

```python
# 12 monthly frames of one band, 600 × 600
data = np.zeros((12, 600, 600), dtype=np.float32)

# today a 3-D array is always (band, y, x)
monthly = GeoArray(data, transform, "EPSG:32613")

monthly.dims
# ('band', 'y', 'x'): 12 "bands", not 12 months

# so ops take a position, and every caller must know the layout
mf = geoops.matched_filter.MatchedFilter(target=ch4_target, axis=-3)
```

### 2. The carrier with stored dims (L3 type)

```python
from geoarray.coords import SpectralAxis
from geoarray.types import GeoImage, GeoMap

# Sentinel-2 SWIR
swir = SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0))

scene: GeoImage = GeoArray(
    swir_data,
    transform=(20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0),
    dims=("band", "y", "x"),
    coords=(swir,),
    crs="EPSG:32613",
)

# exact label lookup; the band dim is dropped
b12: GeoMap = scene.sel(band="B12")

# a window: the data is sliced and the transform's origin moves with it
chip: GeoImage = scene.isel(y=slice(0, 256), x=slice(0, 256))
```

Window georeference, from T = (a, b, c, d, e, f) and a window corner at row r₀, column c₀:

```text
c′ = c + a·c₀ + b·r₀
f′ = f + d·c₀ + e·r₀
```

### 3. One call, two backends

```python
import geoarray

# eager, in memory: a carrier series from the catalog
stack: GeoImageSeries = geocatalog.load_raster_timeseries(catalog, window)

# lazy, chunked: the same type name on an xarray cube
cube: GeoImageSeries = xr.open_zarr(store)["reflectance"]

# m̃(band, y, x) = median over t of s(t, band, y, x)
median: GeoImage = geoarray.reduce(stack, np.nanmedian, dim="time")
median_lazy: GeoImage = geoarray.reduce(cube, np.nanmedian, dim="time")
```

### 4. A max-NDVI composite: argreduce + take_along

```python
# NDVI per date: (time, y, x)
ndvi_series: GeoArray["time y x"] = geoarray.map_over(ndvi, stack, dim="time")

# t*(y, x) = argmax over t of NDVI(t, y, x)
best: GeoArray["y x"] = geoarray.argreduce(ndvi_series, np.argmax, dim="time")

# c(band, y, x) = s(t*(y, x), band, y, x)
composite: GeoImage = geoarray.take_along(stack, best, dim="time")
```

### 5. Two inputs are checked, never aligned

```python
# same grid: works on a carrier and on xarray
diff: GeoImage = geoarray.combine(np.subtract, median, reference)

# reference shifted by 10⁻⁷ m
geoarray.combine(np.subtract, median_lazy, shifted)
# GridMismatchError: x coordinates differ by up to 1e-07 m; regrid first.
# (plain xarray would return shape (1, 4, 0) here)
```

### 6. Patches under jit: one trace, many transforms

```python
# 64 chips of 256 × 256, each with its own transform: GeoPatches
patches: GeoPatches = geopatcher.batch_split(scene, sampler, batch_size=64)

# patches.transform.shape == (64, 6); cross the boundary once
jpatches: GeoPatches = patches.to_jax()

# vmap over the patch dim and the transform rows together
scores: GeoArray["patch y x"] = geoarray.map_over(detect_fn, jpatches, dim="patch")
```

### 7. A geo-algorithm calling an L1 kernel: the TROPOMI averaging kernel

```python
# L1, in radtrax: shapes only
def apply_column_kernel(
    ak: Float[Array, "L N"],
    model: Float[Array, "L N"],
    prior: Float[Array, "L N"],
    x_prior: Float[Array, "N"],
) -> Float[Array, "N"]:
    # xₗ − xₐ,ₗ
    departure = model - prior
    # ∑ₗ Aₗ (xₗ − xₐ,ₗ)
    correction = (ak * departure).sum(axis=0)
    # Xₘ = Xₐ + ∑ₗ Aₗ (xₗ − xₐ,ₗ)
    return x_prior + correction


# L3, in radtrax's optional [geo] extra: geometry in, geometry out.
# GeoModels may import GeoStack, never the reverse, so a wrapper that
# calls a radtrax kernel lives in radtrax, not in geotoolz.
def model_xch4(ak: GeoSwathProfile, model: GeoSwathProfile, prior: GeoSwathProfile, x_prior: GeoSwathImage) -> GeoSwathImage:
    # level first, pixels flattened: (L, N)
    ak_rows: Float[Array, "L N"] = geoarray.to_rows(ak, dim="level")
    model_rows: Float[Array, "L N"] = geoarray.to_rows(model, dim="level")
    prior_rows: Float[Array, "L N"] = geoarray.to_rows(prior, dim="level")
    x_prior_rows: Float[Array, "N"] = geoarray.to_rows(x_prior)
    # the L1 call
    xm: Float[Array, "N"] = apply_column_kernel(ak_rows, model_rows, prior_rows, x_prior_rows)
    # back onto the swath: same lat, lon
    return x_prior.with_rows(xm)
```

l is the retrieval layer (L = 12), A the column averaging kernel, x the model sub-column, xₐ the prior sub-column, Xₐ the prior XCH₄.

### 8. Points against a state: a data-assimilation cost

```python
# along-track SLA: values on obs; xy and time are leaves
obs: GeoTracks = GeoPoints(
    sla,
    dims=("obs",),
    xy=lonlat,
    time=t_obs,
    crs="EPSG:4326",
)


@jax.jit
def cost(ssh: GeoMap, obs: GeoTracks, sigma: float) -> Float[Array, ""]:
    # H(x): bilinear sample of the state at each observation
    predicted: GeoTracks = geoarray.sample(ssh, obs)
    # d = y − H(x)
    d: Float[Array, "N"] = obs.data - predicted.data
    # J(x) = ½ dᵀ R⁻¹ d, with R = σ² I
    return 0.5 * jnp.sum(d**2) / sigma**2


# ∇ₓJ, the gradient a 4D-Var minimizer needs
grad: GeoMap = jax.grad(cost)(ssh, obs, 0.03)
```

y is the observed SLA, H the sampling operator, R the observation-error covariance, σ the along-track noise in m.
Positions change daily but are a leaf, so no recompile; pad each day to a fixed N with a mask.

### 9. Stations with several variables

```python
from geoarray.coords import TimeAxis, VariableAxis

met = VariableAxis(names=("t2m", "psfc", "wind"), units=("K", "hPa", "m s-1"))

# 40 AEMET stations, hourly for a day: (time, variable, obs)
stations: GeoTimeSeries = GeoPoints(
    values,
    dims=("time", "variable", "obs"),
    coords=(TimeAxis(stamps=hours), met),
    xy=station_lonlat,
    crs="EPSG:4326",
)

# a daily mean per station and variable, on either backend
daily: GeoPoints["variable obs"] = geoarray.reduce(stations, np.nanmean, dim="time")
```

### 10. Many items: an L3 op lifted over a cube

```python
# every pass over the facility, lazy: (time, band, y, x)
cube: GeoImageSeries = site_cube

# run the geoops detector per date, per chunk; written once in geoops
per_date = geoops.lift(detect_plumes, dim="time")
masks: xr.DataArray = per_date(cube)

# by value: the 2026 dry season
dry: xr.DataArray = masks.sel(time=slice("2026-06-01", "2026-09-30"))

# f(y, x) = (1 / T) ∑ₜ 𝟙[mask(t, y, x)], the detection frequency
frequency: xr.DataArray = dry.mean("time")
```

T is the number of passes in the window.

## When to jit

JAX is for the compiled step: the model, the loss, its gradient and the optimizer, run many times on fixed shapes.
Pre-processing before it and post-processing after it stay NumPy and eager, run once per file.
The carrier crosses the boundary twice, with `to_jax()` and `to_numpy()`.

```text
PRE · NumPy, eager, once per file
read COG / NetCDF · reproject · mask clouds
rasterize polygons · split into patches
(geoops numpy ops, geopatcher, geocatalog)
        │
        │ to_jax()      fixed shapes from here
        ▼
COMPILED · JAX, jit + grad
normalize · augment · model forward
H(x) · averaging kernel · loss · ∇ · optimizer step
(GeoML, GeoModels, dual-backend geoops)
runs many times on the same program
        │
        │ to_numpy()
        ▼
POST · NumPy, eager, once per result
threshold · vectorize plumes · stitch patches
metrics · write COG / Zarr · plots
(geoops numpy ops, geopatcher, xrops)
```

The test for any op:

| Put it inside `jit` when | Keep it NumPy when |
| --- | --- |
| it runs many times on the same shapes (every training step, every DA iteration) | it runs once per file or per result |
| a gradient must flow through it (H(x), averaging kernels, a differentiable normalization) | nothing downstream needs its gradient |
| shapes are fixed or can be padded | shapes depend on the data (vectorized polygons, variable-length tracks) |
| it is array math (elementwise, einx, linear algebra) | it calls GDAL, rasterio, georeader, scipy, skimage or sklearn |
| it is cheap enough that fusing it saves a host round trip | it touches files, the network or Python objects |

Two running examples:

- **Methane, training a plume detector.**
  Pre: read Sentinel-2, mask clouds, rasterize labelled plume polygons, split 256 × 256 patches.
  Compiled: normalize, augment, forward, loss, gradient, optimizer.
  Post: stitch scores, threshold, vectorize plumes, write a COG.
- **Ocean, 4D-Var.**
  Pre: select one day of along-track passes in xarray, pad to a fixed N. Compiled: somax forward, H(x), J(x), ∇J, minimizer iterations.
  Post: write the analysis to Zarr, compute metrics against withheld tracks.

Some ops sit on both sides.
`normalize` runs in pre for a one-off product and inside the training step when its statistics are learned; that is why a handful of geoops ops are dual-backend.

## Implications

| Repo or package | Row (layer) | Column (carriers) | What changes |
| --- | --- | --- | --- |
| gaussx, finitevolx, spectraldiffx, somax, radtrax, plumax, geonnax | L1 core; L3 wrappers behind a `[geo]` extra | raw arrays in the core; geoarray carriers in `[geo]` | the core never imports geotoolz; carrier wrappers that call their own kernels (e.g. radtrax's `model_xch4`) sit behind `[geo]`, since GeoModels may import GeoStack but not the reverse; jaxtyping signatures are the contract; consumed axes last, so kernels lift through `KernelOp` |
| pipekit, pipekit-array | L2 machinery | every registered carrier | the adapter, one `Signature`, the dim-level functions, `KernelSpec` / `KernelOp` ([pipekit page](./06_pipekit_named_arrays.md)) |
| geoarray | L3 types, adapter, geometry functions | geoarray carriers | stored dims, standard coordinates, readable types, `GeoSwath`, `GeoPoints`, its adapter, lossless xarray converters |
| geocatalog, geocloud, geoproducts | produce carriers | both: acquisitions as geoarray carriers, cubes as xarray carriers | readers name their dims; stay JAX-free and xarray-optional |
| geopatcher | L3 | geoarray carriers | the batch is a `GeoPatches` carrier with a `(patch, 6)` transform; split and stitch stay NumPy; the silent `jnp = np` fallback goes |
| geoops | L2 kernel operators, L3 geometry | kernel operators in both; geometry on geoarray carriers, on cubes via `geoops.lift` | `dim=` replaces `axis=`; each op declares `backends` ([operators page](./05_operators.md)) |
| xrops | L2 kernel operators and value analysis, L3 cube geometry | xarray carriers; its kernel operators in both | built on native xarray operations; never imports geoops ([operators page](./05_operators.md)) |
| coordax | none | optional converter | for model code that wants `cmap`; no geotoolz package depends on it |

**Which geoops ops run on JAX.**
The ones that pass the when-to-jit test and call no CPU-only library: `indices`, `qa`, array-math `normalize`, `einx`, `compositing` (as `reduce` / `argreduce` + `take_along`), basic `augment`.
Everything wrapping georeader, rasterio, scipy, skimage or sklearn stays NumPy and refuses a JAX carrier with a `BackendError` naming the fix.
`matched_filter` and `plume` move to radtrax and plumax (#482).

**The hero figure.**
The vertical "carrier switch at L3" becomes horizontal rows by workload (per item and fast, many items and lazy); data maturity stays as the column axis and as each reader's default.

**geoproducts and xrproducts.**
Unchanged: one acquisition versus a cube is what the reader returns.

## Under `jit`

- **Leaves:** data and the georeference (transform, lat/lon, xy, z, time stamps).
  New scenes, patches and observation days reuse the compiled program as long as shapes match.
- **Static:** dims, sizes, coordinates (band and level labels), CRS, fill value.
  A change means a retrace.
- **Shapes are static too.**
  Pad point sets to a fixed N with a mask; geopatcher already fixes the patch size.
- **x64 stays required**, as today: in float32 a UTM northing near 3.5 × 10⁶ m resolves only to about 0.25 m, too coarse for sub-pixel coregistration.

## Migration

| Today | After |
| --- | --- |
| dims inferred from `ndim` | dims stored; inferred from `ndim` for one release with a `DeprecationWarning` |
| 3-D is always `(band, y, x)` | any dims ending in the spatial dims, e.g. `(time, y, x)` |
| ops take `axis=-3` | ops take `dim="band"` |
| band names in `attrs["band_names"]` | a `SpectralAxis` in `coords` |
| `GeoArray[np.ndarray]` generic | dims-string subscript (`GeoArray["band y x"]`) |
| swaths and points have no carrier | `GeoSwath`, `GeoPoints` |
| xarray conversions ad hoc per package | `to_xarray()` / `from_xarray()` in geoarray, lossless |

One 0.X.0 bump of geotoolz-array, like the GeoTensor migration.

## Spike gates

| Gate | Passes when | If it fails |
| --- | --- | --- |
| G1 One trace, many scenes | a jitted kernel on 10 same-shape scenes with different transforms compiles once | the transform is not a leaf; fix first |
| G2 Patches | 64 patches with different transforms run in one trace via `map_over`, and the stitch matches today's within float tolerance | keep patches as raw arrays inside the model, carriers outside |
| G3 DA gradient | `jax.grad(cost)` with respect to the state works with observations padded to a fixed N | sample returns raw arrays; the carrier stops at the boundary |
| G4 No JAX on I/O | geocatalog, geocloud and geoproducts import and read with JAX not installed | move the import behind `[jax]` |
| G5 Shared functions agree | each shared function gives identical results on NumPy, JAX and xarray (dask-backed included) | drop it from the shared set |
| G6 No silent alignment | `combine` and `stack` raise on a 10⁻⁷ m offset on every backend | the xarray path is not shared |
| G7 Lossless round trip | `from_xarray(to_xarray(g))` returns the same transform, dims and coordinates for random windows and rotated grids; the #463 and #464 regression cases pass | conventions are fixed before release |
| G8 Patch overhead | 10k patches through a 5-op chain cost no more than today's carrier plus a budget you set | profile and cut before release |
| G9 Reproducible costs | a committed benchmark script times the normalized difference and the window on the carrier and on xarray at the three shapes above, writes its output to a tracked file, and those numbers replace the indicative table | drop the per-op cost argument from this page |

## Plan

```text
P0 spike: carrier with stored dims, 3 kernels,
   shared functions on 3 backends
   gate: G1–G8
        │
        ▼
P1 geoarray 0.2.0: carrier, coords, types,
   shared functions, GeoSwath, GeoPoints,
   xarray converters
        │
   ┌────┴────┐
   ▼         ▼
P2 I/O     P3 geopatcher
name dims  GeoPatches
   │         │
   └────┬────┘
        ▼
P4 geoops: dim=, backends, KernelOps, lift
   xrops: KernelOps, on pipekit-array
        │
        ▼
P5 JAX-first kernels in GeoModels
   (radtrax, plumax), coordax converter if wanted
```

| Phase | Scope | Gate |
| --- | --- | --- |
| P0 Spike | the carrier with stored dims for a matched-filter kernel, a 64-patch `map_over` and the DA cost; the shared functions on NumPy, JAX and xarray; the cost benchmark | G1–G9 |
| P1 geoarray 0.2.0 | carrier, `geoarray.coords`, `geoarray.types`, `check_dims`, shared functions, `combine` / `stack`, `GeoSwath`, `GeoPoints`, `to_xarray` / `from_xarray`; `ndim` inference deprecated | the 0.1.x suite passes with only the deprecation warning |
| P2 I/O | readers name dims; swath readers return `GeoSwath` | G4 |
| P3 geopatcher | `batch_split` returns `GeoPatches`; stitch reads transform rows | G2 |
| P4 geoops and xrops | `dim=`, `backends`, kernel operators and `geoops.lift` ([operators page](./05_operators.md)); P4 waits for the pipekit-array release ([pipekit page](./06_pipekit_named_arrays.md)) | the contract test passes on both backends |
| P5 GeoModels | JAX-first matched filter and plume ops; `to_coordax()` only if somax or vardax ask | G1 and G3 rerun there |

## Open questions

Settle these before the geotoolz issues (#511–#534) are rewritten.

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| Q1 | What do somax and vardax take at their boundary? | Shaped arrays plus a grid object (L1); the carrier stops in the L3 wrapper, so `sample` and H(x) live in geotoolz | Models take carriers directly, which makes `to_coordax()` or carrier-aware models necessary |
| Q2 | How is the georeference written into xarray? | CF `grid_mapping` plus a `GeoTransform` attribute, written by geoarray and readable by rioxarray, with no rioxarray dependency | Depend on rioxarray (pulls in rasterio and GDAL) |
| Q3 | Where do the shared functions live? | **Decided, 21 Oct 2026:** pipekit-array, over a named-array adapter; geoarray registers its adapter and re-exports them | — |
| Q4 | What happens to geoops `compositing`? | Keep named operators as thin, serializable wrappers over `stack`, `reduce`, `argreduce`, `take_along` | Remove them; users compose the shared functions |
| Q5 | Do geoops operators accept a DataArray? | Kernel operators: yes, lazily through `apply_ufunc`. Geometry operators: no, they raise and name `geoops.lift` | Convert every operator at entry and back at exit |
| Q6 | `map_over` on xarray | Allowed via `apply_ufunc`, documented as per-item Python cost unless dask-parallelized | Not shared on xarray |
| Q7 | Dim names: closed vocabulary or conventions? | Free names; geoops and the converters rely on the conventions (`time`, `member`, `level`, `band`, `variable`, `patch`); `check_dims` enforces each signature | A closed list, validated at construction |
| Q8 | Curvilinear carrier name | `GeoSwath`, also used for fixed model grids | `GeoCurvilinear`, with `GeoSwath` an alias |
| Q9 | The dims checker | **Decided, 21 Oct 2026:** `check_dims` on `pipekit.Signature` patterns, the single `Signature` | — |
| Q10 | The existing issues | Edit #511–#534 in place (numbers and links survive); block themes B–D on the spike #516; add pipekit and pipekit-array issues for the shared machinery | Close and recreate |

Already decided (21 Oct 2026): `GeoImageSeries` for frames on a grid and `GeoTimeSeries` for fixed locations; `band` and `variable` are separate dims; `level` is allowed on observations; coordax is a converter, not a foundation.
