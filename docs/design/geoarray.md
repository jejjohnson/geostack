---
title: "geoarray: the carrier"
subtitle: "Design · one thin carrier family for NumPy and JAX: GeoArray, GeoSwath and GeoPoints, with stored dims, standard coordinates, readable types and the geometry that never changes values"
short_title: geoarray
description: "The design of geotoolz-array (import geoarray): frozen-dataclass carriers registered as JAX pytrees that replace georeader's GeoTensor, store their dims, ship standard coordinates and readable geo types, keep geometry as leaves so a new scene never recompiles, and hand over to xarray for anything selected by value. Includes the migration from GeoTensor and from positional dims, gates and open questions, checked on one methane plume seen by EMIT and Sentinel-2."
date: 2026-10-23
keywords:
  - geoarray
  - geotoolz
  - GeoTensor
  - named dims
  - GeoImage
  - GeoSwath
  - GeoPoints
  - JAX
  - NumPy
  - pytree
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Decision

geoarray is the geometry vocabulary of the stack.
It is one bottom package of geotoolz, `geotoolz-array` (import `geoarray`), holding one family of thin carriers whose data is a NumPy or a JAX array:

- **`GeoArray`** for gridded data with an affine transform;
- **`GeoSwath`** for sensor geometry with per-pixel lat and lon;
- **`GeoPoints`** for observations at scattered positions.

Each carrier stores its dims, carries standard coordinates and a geometry leaf, and is registered as a JAX pytree.
geoarray also ships readable type names for the common shapes (`GeoImage`, `GeoMap`, `GeoImageSeries`, `GeoTracks`, …) and every operation that re-indexes or re-describes pixels without changing their values.

It replaces `georeader.GeoTensor` as the type that flows between geotoolz packages; georeader becomes an optional reader adapter.
pipekit's `Operator` contract does not change.
The carrier rules in geotoolz's `AGENTS.md` keep their meaning (fresh `attrs`, NaN-aware nodata, exact grid matching), except that dims are stored instead of inferred from `ndim`.

Where the carrier sits in [the three levels](#design-levels): L2 operators and L3 algorithms take carriers or readable types; L1 primitives never see one.

### Carriers and operations

A **carrier** is a data structure: values plus their description.
There are two families, and xarray's data structures are carriers just as much as geoarray's.

| Family | Carriers | Holds | Good at |
| --- | --- | --- | --- |
| geoarray | `GeoArray`, `GeoSwath`, `GeoPoints` | NumPy or JAX data, dims, static coordinates, a geometry leaf (transform, lat/lon, xy), CRS | one item or a fixed batch: strict grids, compiles under `jit`, about 1–10 µs per op |
| xarray | `DataArray`, `Dataset`, `DataTree` | NumPy or dask data, dims, coordinate values, attributes; `Dataset` groups variables on shared dims, `DataTree` nests groups | many items: selection by value, laziness, out-of-core, variables with different dims |

geoarray has no `Dataset` or `DataTree` equivalent on purpose: a carrier holds one stack of values (several bands or variables share one `band` or `variable` dim).
Collections of differently shaped variables are xarray's job, and the crossing is the [labeled lane](./labeled_lane.md).

This page covers the data.
How operations are written once and run on both families is in [KernelOp](#design-kernelop); what geoops and xrops each own, and [the tiers](#design-tiers) they sit in, is in [operators](./operators.md).

### Ownership rules

Carriers flow down from geoarray; nothing in it reads files, and nothing in it changes pixel values.

```text
numpy · pyproj · affine · array-api-compat · pipekit-array
   ┊ extras: [jax] [xarray] [georeader] [coordax]
   ▼
geotoolz-array (import geoarray)
   carriers   GeoArray, GeoSwath, GeoPoints
   values     Grid, Window, FrozenAttrs, coords, types
   geometry   isel, crop, windows, valid_mask
   ▼
geocatalog · geocloud · geoproducts   native-grid reads → carriers
geopatcher                            plan on the host, crop
geoops                                KernelOps and geometry operators
   ▼
geotoolz (umbrella, no code)
   ┊ behind [geo] extras
   ▼
radtrax[geo] · plumax[geo] · esmax-core[geo]   downstream carriers
```

1. **geoarray** owns the carriers, `Grid`, `Window`, `FrozenAttrs`, the coordinates, the readable types, its named-array adapter, and every function that re-indexes or re-describes pixels.
   It never imports georeader, rasterio or coordax in its base modules, and imports JAX and xarray only behind their extras.
2. **geoops** owns every function that changes values or moves pixels to a new grid, as L1 kernels wrapped as `KernelOp`s or as hand-written geometry operators ([operators](./operators.md)).
   It never builds a carrier by hand; it rewraps through the adapter.
3. **geocatalog, geocloud and geoproducts** read bytes into a NumPy carrier on the file's native grid, with its dims named.
   They never reproject, since that would make them import geoops.
4. **geopatcher** plans patch offsets on the host and extracts them with `geoarray.crop`.
   It stops importing `georeader.window_utils` and `georeader.read`.
5. **georeader** appears only behind extras: `geoarray[georeader]` for the carrier adapter, and the readers' own extras for file access.

What the other packages change, in one table:

| Package | Takes or makes | What changes |
| --- | --- | --- |
| geocatalog, geocloud, geoproducts | acquisitions as geoarray carriers, cubes as xarray carriers | readers name their dims; swath readers return `GeoSwath`; stay JAX-free and xarray-optional |
| geopatcher | geoarray carriers | the batch is a `GeoPatches` carrier with a `(patch, 6)` transform; split and stitch stay NumPy; the silent `jnp = np` fallback goes |
| geoops | both families | `dim=` replaces `axis=`; each operator declares `backends`; no xarray in its base install ([operators](./operators.md)) |

### Indicative costs (not yet reproducible)

One exploratory run on a CPU sandbox compared a 4-op normalized difference and a window on a mock thin NumPy carrier and on `xr.DataArray`.
No script or output from that run is committed, so these numbers motivate the design but are not results: gate GA-G22 replaces them with a committed benchmark and its output, and the claims below stand or fall with it.

| Shape | ND op, carrier | ND op, xarray | Window, carrier | Window, xarray |
| --- | --- | --- | --- | --- |
| 2 × 64 × 64 | 12 µs | 2,257 µs | 1.4 µs | 80 µs |
| 2 × 256 × 256 | 275 µs | 2,779 µs | 1.0 µs | 63 µs |
| 2 × 2000 × 2000 | 48 ms | 52 ms | 2.4 µs | 96 µs |

- In that run, on a full scene the difference was noise; on patches xarray was 10–200× slower per op (to be confirmed by GA-G22).
- Two grids whose x coordinates differ by 10⁻⁷ m, added in xarray, gave shape `(1, 4, 0)`: every pixel dropped, no error.
- `import xarray` took about 0.57 s, against 0.11 s for NumPy.

The carrier earns its place by what it refuses (mismatched grids, a transform drifting from its coordinates, ambiguous pixel edges) and by compiling, not by speed on large scenes.

(design-carrier)=
## The carrier

Each carrier is a frozen dataclass that composes an array, never subclasses one, and registers as a JAX pytree when JAX is importable.

| Object | Lives in | Name | Kind |
| --- | --- | --- | --- |
| Carriers | `geoarray._src.array` | `GeoArray`, `GeoSwath`, `GeoPoints` | frozen dataclasses, pytrees |
| Pixel grid | `geoarray._src.grid` | `Grid` | frozen dataclass, hashable |
| Pixel window | `geoarray._src.grid` | `Window` | frozen dataclass of ints |
| Coordinates | `geoarray.coords` | `PixelGrid`, `SpectralAxis`, `TimeAxis`, … | frozen dataclasses, hashable |
| Metadata | `geoarray._src.attrs` | `FrozenAttrs` | hashable `Mapping` |
| Backend | `geoarray._src.backend` | `xp`, `to_jax`, `to_numpy` | functions |

Every field is either a leaf, which JAX traces, or static, which JAX hashes into the compiled function's cache key.
Changing a static field recompiles; changing a leaf does not.

| Field | Type | Pytree role | Changing it |
| --- | --- | --- | --- |
| `data` | `np.ndarray` or `jax.Array`, ending in the spatial dims | leaf | free (same shape and dtype) |
| `transform` | `(*batch, 6)` float64, (a, b, c, d, e, f), per frame or per patch | leaf | free; differentiable |
| `dims` | `tuple[str, ...]`, e.g. `("time", "band", "y", "x")` | static | recompiles |
| `coords` | one `geoarray.coords` coordinate per non-spatial dim | static | recompiles |
| `crs` | `pyproj.CRS`, parsed once; strings and EPSG ints are coerced | static | recompiles |
| `fill_value` | `float`, `int`, `bool` or `None` | static | recompiles |
| `attrs` | `FrozenAttrs` | static | recompiles |

`GeoSwath` replaces `transform` with lat and lon leaves `(…, row, col)`; `GeoPoints` replaces it with an xy leaf `(obs, 2)` and optional z, time and valid leaves `(obs,)`.

```python
# composition, frozen; pytree registration is by hand (see "Construction" below)
@dataclasses.dataclass(frozen=True, eq=False)
class GeoArray:
    # np.ndarray or jax.Array, ending in (y, x)
    data: Float[Array, "... H W"]
    # T = (a, b, c, d, e, f): (x, y) = (a·col + b·row + c, d·col + e·row + f), per frame or per patch
    transform: Float[Array, "*batch 6"]
    # ("time", "band", "y", "x"); static
    dims: tuple[str, ...]
    # one geoarray.coords coordinate per non-spatial dim; static
    coords: tuple[Coordinate, ...]
    # fixed for a compiled program
    crs: pyproj.CRS
    # nodata marker; NaN allowed; None means no fill
    fill_value: float | int | bool | None = None
    # hashable non-coordinate metadata: sensor, product, processing level
    attrs: FrozenAttrs = dataclasses.field(default_factory=FrozenAttrs)
```

**Static or leaf.**
What fixes a compiled program (dims, sizes, band and level labels, CRS, fill value) is static.
Geometry that changes per call (the transform, swath lat/lon, point xy and z, time stamps) is a leaf, so a new scene, patch or day of observations never recompiles.
Shapes are static too: pad point sets to a fixed N with a valid mask; geopatcher already fixes the patch size.

**The backend is the data's type.**
An operator given one NumPy and one JAX carrier refuses and names `to_jax()`; nothing is promoted silently.

### Compose, do not subclass (GA-A1)

JAX operations always return a plain `jax.Array`, so a subclass would lose its metadata on the first `jnp` call.
Composition costs one thing: `g + 1` no longer works (GA-B1).
In return, one class holds both backends, and `eq=False` avoids the ambiguous truth value that elementwise `==` would give a dataclass.
Equality of carriers is `g.identical(h)`, and equality of grids is `g.grid == h.grid`.

### The transform is a float64 leaf (GA-A2)

A leaf transform lets `jax.grad` reach the georeferencing (Stage 4 of the running example).
It also lets `vmap` batch patches whose origins differ, and lets `jit` reuse one compiled function across scenes.
The prototype confirmed both: a new transform did not retrace, while a new CRS or shape did.

float64 is required because map coordinates are large.
In float32 the spacing between representable numbers is 0.25 m at a northing of 3.55 × 10⁶ m, 0.5 m at 4.65 × 10⁶ m, and 7.6 × 10⁻⁶° (≈ 0.7 m at 32° N) at longitude −103.75°.
So `to_jax` checks `jax.config.jax_enable_x64` and raises with a message if it is off, rather than letting JAX silently round the transform to float32.
`data` keeps its own dtype, normally float32.

Recommended: require x64 for JAX carriers.
Alternative: store a static float64 origin (c₀, f₀) and a float32 leaf offset; no global flag, but every new origin recompiles.

### Static, frozen attrs (GA-A3)

Per-band metadata (names, wavelengths, FWHM) moved into a `SpectralAxis` in `coords` ([standard coordinates](#design-readable-types)), so `attrs` holds only what describes the whole carrier: sensor, product, processing level.
Static attrs must be hashable, so `FrozenAttrs` enforces three rules:

1. **Lists become tuples.** `["L1C", "B"]` is stored as `("L1C", "B")`.
2. **1-D numeric arrays become tuples of Python floats.** Building and hashing EMIT's 285 wavelengths this way took about 21 µs in the prototype.
3. **Anything else unhashable raises.** That covers dicts, 2-D arrays and live objects such as open file handles.

Updates return a new carrier: `g.with_attrs(sensor="EMIT")` and `g.drop_attrs("id")`.
Scene-specific keys such as acquisition time or scene ID would recompile per scene, so a pipeline strips non-geometric keys before `jit` and reattaches them after; acquisition times belong in a `TimeAxis` leaf instead.

### No fill by default (GA-A4)

`fill_value=None` means "no nodata marker"; `valid_mask` then tests finiteness only.
Readers set the fill explicitly: 0 for Sentinel-2 L1C digital numbers, NaN after conversion to reflectance.
The georeader adapter carries georeader's value across unchanged, so converting an existing `GeoTensor` changes nothing.

### Construction and unflattening

`__post_init__` validates: dims ending in the spatial dims of the carrier kind, data whose rank matches `dims`, a transform of shape `(*batch, 6)`, one coordinate per non-spatial dim with the right size, a parseable CRS, and attrs that freeze.
The pytree `unflatten` bypasses `__post_init__`, because JAX unflattens with placeholder leaves during tracing, and a shape check would fail on them.
That is why the registration is written by hand rather than with `jax.tree_util.register_dataclass`, which calls the constructor.
The pytree is registered at import time inside a `try: import jax`, so a NumPy-only install never imports JAX.

A bare `jax.vmap` over a carrier batches its leaves but leaves the static `dims` unchanged, so it would describe a batched array with the wrong dims.
Batching goes through `geoarray.map_over` or a function that adds the dim (`geopatcher.split`, Stage 5).

### The Grid value type

`Grid` is the transform, CRS and spatial shape without data.
It is hashable, so it can be a warp target, an operator's static config or a dictionary key.
geocatalog's `GeoSlice` and geopatcher's domains both produce one, so there is one "where are these pixels" object across the stack.

```python
# no data, so it is cheap, hashable and JSON-serialisable
@dataclasses.dataclass(frozen=True)
class Grid:
    # (a, b, c, d, e, f) as Python floats
    transform: tuple[float, float, float, float, float, float]
    crs: pyproj.CRS
    # (H, W)
    shape: tuple[int, int]

    # (|a|, |e|) for an axis-aligned grid
    @property
    def res(self) -> tuple[float, float]: ...

    # (x_min, y_min, x_max, y_max)
    @property
    def bounds(self) -> tuple[float, float, float, float]: ...

    # b = d = 0
    @property
    def is_axis_aligned(self) -> bool: ...

    # T⁻¹(bounds), rounded outward to whole pixels
    def window(self, bounds: tuple[float, float, float, float]) -> "Window": ...

    # exact equality is ==; this allows a tolerance in map units
    def isclose(self, other: "Grid", atol: float = 1e-6) -> bool: ...

    # strict JSON: CRS as its authority string or WKT
    def get_config(self) -> dict: ...

    @classmethod
    def from_bounds(cls, bounds, res, crs) -> "Grid": ...
```

## Named dims and geometry kinds

Dims are stored on the carrier.
They replace the positional convention `("time", "band", "y", "x")[-ndim:]`, under which a 3-D array was always `(band, y, x)` and every operator took `axis=-3`:

```python
# today: 12 monthly frames of one band, 600 × 600
data: Float[Array, "T H W"] = monthly_values
# dims inferred from ndim: ('band', 'y', 'x'), so 12 "bands", not 12 months
monthly: GeoArray = GeoArray(data, transform, "EPSG:32613")
# ops take a position, and every caller must know the layout
mf: MatchedFilter = geoops.matched_filter.MatchedFilter(target=ch4_target, axis=-3)
```

With stored dims the same frames are `GeoArray["time y x"]`, and operators take `dim="band"`.

A new class exists only when the georeferencing changes; everything else is a named dim.

| Carrier | Georeference leaf | Spatial dims (last) | Examples |
| --- | --- | --- | --- |
| `GeoArray` | transform `(…, 6)` | `y x` | Sentinel-2 scenes, gridded SLA maps, a model grid |
| `GeoSwath` | lat, lon `(…, row, col)` | `row col` | EMIT, TROPOMI, AVIRIS in sensor geometry |
| `GeoPoints` | xy `(obs, 2)`, optional z `(obs,)`, optional valid `(obs,)` for padding | `obs` | Argo cycles, along-track SSH, stations, plume sources |

Other dims come before the spatial ones.
Any name works; these are the conventions geoops and the converters expect:

| Dim | Means | Labels | xarray name |
| --- | --- | --- | --- |
| `time` | acquisition or valid time | `datetime64` | `time` |
| `member` | ensemble member | strings | `member` |
| `level` | a vertical grid shared by every pixel | layer index or nominal value, with a `Vertical` | `level` (maps CF `depth`, `lev`, `plev`, `layer`) |
| `band` | spectral band | names, optional wavelength in nm | `band` |
| `variable` | physical quantity with its own units | names and units | `variable` |
| `patch` | patches from geopatcher | none | — |

**Level or z?**
The test is whether samples share the vertical grid:

- **Shared grid → a `level` dim.**
  TROPOMI's 12 retrieval layers, IASI or AIRS pressure levels, EN4 depths, GPM range bins.
  When layer pressures move with surface pressure, `level` is the layer index and the pressures are data in their own band.
- **Each sample its own position → `GeoPoints.z`.**
  Argo, CTD and glider samples at irregular depths, lidar returns.

`z` is never a third column of `xy`: height (m, up), depth (m, down) and pressure (hPa) differ in units and direction, so `z` carries a `Vertical` (kind, units, CF `positive`, datum).

(design-readable-types)=
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
Station or model variables carry different units, and an operator that mixes channels must refuse them.

**Several variables per point.**
A weather station reporting temperature, pressure and wind is one point with a `variable` dim: `GeoPoints["time variable obs"]` for 40 stations reporting hourly.
A variable a station does not report is the fill value, masked as usual.

**Time as a dim or per sample.**
Shared stamps are a `time` dim with a `TimeAxis`: a stack of scenes, fixed stations reporting hourly.
A moving platform has one time per sample, so time is an `(obs,)` leaf on `GeoPoints`, not a dim: along-track altimetry, Argo, ship tracks.

`TimeAxis` is the one that splits.
Its stamps change from batch to batch, so under `[jax]` they are a leaf, never a static label; otherwise every new set of dates would recompile.
On the base install they are ordinary labels.

Each coordinate is a frozen, hashable dataclass with `dims` and `shape`, static under `jit`.
The converters write them as CF coordinates and attributes and read them back ([labeled lane](./labeled_lane.md)).

```python
from geoarray.coords import PixelGrid, SpectralAxis, TimeAxis

# Sentinel-2 SWIR: names and centre wavelengths, nm
swir: SpectralAxis = SpectralAxis(names=("B11", "B12"), wavelength_nm=(1610.0, 2190.0))
# a 600 × 600 grid in UTM 13N, area-convention pixels
grid: PixelGrid = PixelGrid(shape=(600, 600), crs="EPSG:32613")
# 12 acquisition dates
dates: TimeAxis = TimeAxis(stamps=acquired)
# raw values: 12 dates, 2 bands, 600 × 600
data: Float[Array, "T C H W"] = swir_values
# T = (a, b, c, d, e, f): 20 m pixels, origin at (600 000, 3 550 000) m
T: tuple[float, ...] = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
# a carrier from coordinates, data and the transform leaf; a GeoImageSeries
stack: GeoArray["time band y x"] = GeoArray.from_coords(
    data, dates, swir, grid, transform=T
)
```

### Readable types (`geoarray.types`)

The names are aliases for a carrier plus its dims, not subclasses.
Slicing a `GeoImageSeries` at one date simply is a `GeoImage`, and `check_dims` verifies the dims at the call, on `pipekit.Signature` patterns.

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

```python
from geoarray.types import GeoImage, GeoMap


@check_dims
def ndvi(scene: GeoImage) -> GeoMap:
    # ρ_NIR and ρ_red, selected by band name
    nir: GeoArray["y x"] = scene.sel(band="B8")
    red: GeoArray["y x"] = scene.sel(band="B4")
    # NDVI = (ρ_NIR − ρ_red) / (ρ_NIR + ρ_red); grids checked, never aligned
    return combine(_normalized_difference, nir, red)


# one date of a series is an image, with no conversion; a GeoImage
first: GeoArray["band y x"] = stack.isel(time=0)
# map over time: a series of maps
series: GeoArray["time y x"] = geoarray.map_over(ndvi, stack, dim="time")
```

`GeoArray` was generic over its array type (`GeoArray[np.ndarray]`).
The dims-string subscript replaces that use, since the backend now comes from the field.

## Geometry geoarray owns

geoarray owns every operation that re-indexes or re-describes pixels without computing new values, written once against the array namespace `xp`.
The membership test: if the output contains only input values, fill values or coordinates, the function belongs here.
Everything that computes new values (warping, resampling, CRS maths, padding by reflection) is a geoops operator: an L1 kernel wrapped as a [KernelOp](#design-kernelop), or a hand-written geometry operator that derives source coordinates and calls a primitive ([operators](./operators.md)).
geoarray holds none of them.

| `GeoTensor` symbol | Destination | Reason |
| --- | --- | --- |
| `.dims` (inferred from `ndim`) | `.dims` (stored) | named dims |
| `.transform` (rasterio `Affine`) | `.transform` array; `.grid.affine` for an `affine.Affine` | leaf for JAX, `Affine` on the host |
| `__getitem__`, `isel` | `isel(y=…, x=…)`, `sel(band=…)` by name | coordinates sliced with the data |
| `read_from_window(window, boundless)` | `geoarray.read_window` | crop with fill padding |
| `write_from_window` (in place) | `geoarray.write_window`, returns a new carrier | JAX arrays are immutable |
| `pad` with `mode="constant"` | `geoarray.pad` | fills with `fill_value` |
| `pad` with other modes, `resize` | geoops kernels | compute new values |
| `meshgrid()` in the native CRS | `geoarray.pixel_coords` | coordinates only |
| `meshgrid(dst_crs)` | a geoops geometry operator | needs CRS maths |
| `validmask`, `invalidmask` | `.valid_mask` | NaN-aware rule from `geoops._src.valid` |
| `same_extent` | `g.grid == h.grid`, `Grid.isclose` | exact by default, as geoops already requires |
| `stack`, `concatenate`, `squeeze`, `expand_dims`, `transpose` | shared dim-level functions, `dim=` | non-spatial dims only |
| `footprint`, `valid_footprint` | `geoarray.footprint` (NumPy only, shapely); `valid_footprint` in geoops | the second's output size depends on the data |
| `window_from_bounds`, `round_outer_window` | `Grid.window` | geopatcher stops importing georeader |

### Arithmetic through map and combine (GA-B1)

`GeoTensor` supports `gt + 1` and `gt1 - gt2` through NumPy's ufunc protocol, which JAX arrays do not offer to wrappers.
geoarray replaces it with two explicit forms.
`g.map(f)` applies f to the data and keeps the geometry, and raises if f changes the spatial shape.
`combine(f, g1, g2, …)` checks that every grid is equal, then applies f to the data arrays.
The grid check runs on concrete values; under `jit` the transforms are traced, so `combine` compares the static parts (dims, CRS and shape) and leaves transform equality to the caller.

```python
# g ↦ log(1 + g), geometry unchanged
logged: GeoArray["band y x"] = s2_jax.map(jnp.log1p)
# the two bands as separate maps on one grid
b12: GeoArray["y x"] = s2_jax.sel(band="B12")
b11: GeoArray["y x"] = s2_jax.sel(band="B11")
# r = B12 / B11; raises if the grids differ
ratio: GeoArray["y x"] = geoarray.combine(jnp.divide, b12, b11)
```

Recommended: no dunders.
Alternative: dunders that check grids eagerly and skip the transform check under `jit`.

### Slicing moves the transform and the coordinates

Slicing rows i₀ : i₁ with step s_r and columns j₀ : j₁ with step s_c changes the transform by a shift and a scale:

$$
\begin{aligned} a' &= a s_c & b' &= b s_r & c' &= c + a j_0 + b i_0 \\ d' &= d s_c & e' &= e s_r & f' &= f + d j_0 + e i_0 \end{aligned}
$$

Spatial dims take slices only, with positive steps; other dims take anything NumPy indexing allows.
Selecting along a dim selects the matching entries of its coordinate: `isel(band=slice(1, 2))` keeps `band` with a one-entry `SpectralAxis`, and `sel(band="B12")` drops the dim.
`g.shift_origin(δ)` is the traced counterpart for a sub-pixel move: it adds δ to (c, f), and gradients flow through it.

### Traced crops

Plain slicing needs concrete offsets.
`geoarray.crop(g, offset, size)` takes a traced `(row, col)` offset and a static `(h, w)` size, uses `lax.dynamic_slice` on JAX data and slicing on NumPy data, and shifts the transform as above.
`lax.dynamic_slice` clamps an out-of-range offset to stay inside the array instead of raising, so `crop` documents that and `read_window` handles out-of-bounds windows with padding.

```python
# x_p = x[…, o_p : o_p + (h, w)], T_p = T shifted by o_p
def crop(
    g: GeoArray["... y x"],
    offset: Int[Array, "2"],
    size: tuple[int, int],
) -> GeoArray["... y x"]: ...
```

### Windows

`Window(row_off, col_off, height, width)` is a frozen dataclass of Python ints, equivalent to a rasterio `Window` without importing rasterio.
`Grid.window(bounds)` rounds outward, as `round_outer_window` does today.
`Grid.bounds_of(window)` is its inverse on grid-aligned bounds.
`read_window(g, window, boundless=True)` pads the outside with the fill value, and requires a fill value when the window leaves the array.

### Shared dim-level functions

A short list of dim-level functions works on a geoarray carrier (NumPy or JAX) and on an `xr.DataArray`.
They are written once, in pipekit-array, over a named-array adapter that each carrier family registers; geoarray registers its adapter and re-exports them, so `geoarray.reduce` and `pipekit_array.reduce` are the same object.
The rule for adding one is on the [pipekit kernels](./pipekit_kernels.md) page: it never reads coordinate values or geometry and behaves identically on every carrier.

| Piece | Where it lives |
| --- | --- |
| `isel`, `sel` (exact), `transpose`, `reduce`, `argreduce`, `take_along`, `map_over`, `stack`, `concat`, `combine` | pipekit-array, re-exported by geoarray |
| the geoarray adapter: dims, data, labels, an `isel` that slices coordinates and moves the transform origin, a `rewrap` that keeps the georeference, a `check_compatible` that compares grids and raises `GridMismatchError`, `map_over` over data and transform rows together | geoarray |
| geometry functions: `sample` (H(x)), windows that move the transform, `to_rows` / `from_rows` for pixels as rows, `to_pixel` | geoarray |
| type names and `check_dims`, built on `pipekit.Signature` patterns | geoarray (names), pipekit (rule) |

The full operator surface is not shared, because the backends differ in four behaviours:

| Behaviour | Carrier, NumPy | Carrier, JAX | xarray |
| --- | --- | --- | --- |
| two inputs on different grids | refuses | refuses | inner-joins silently in plain arithmetic |
| laziness | eager | traced under `jit` | often dask-lazy; `.data` can trigger a full compute |
| compiles | no | yes | no |
| per-op overhead (indicative, GA-G22) | about 1–10 µs | about 1–10 µs eager | about 60–2,000 µs |

On xarray, `combine` and `stack` use the xarray adapter's exact comparison, so the silent-alignment trap does not apply to them; the alignment difference is one explicit check per carrier family instead of a hidden default.

**Composites are two ops.**
A median composite is `stack` then `reduce` over `time`.
A max-NDVI or cloud-free composite picks a time index per pixel from one quantity and gathers the other bands at it: `argreduce` then `take_along`.
The maths is the same on every backend; the execution differs (eager in memory, or chunked with dask for 100 scenes of 10k × 10k).

## Backends and crossings

Every move between backends or libraries is an explicit call, and each one is a function in geoarray.

| Call | From → to | Notes |
| --- | --- | --- |
| `g.to_jax(device=None)` | NumPy → JAX | `jax.device_put` for data and the geometry leaves; raises if x64 is off (GA-A2) |
| `g.to_numpy()` | JAX → NumPy | host copy of data and leaves |
| `np.asarray(g)` | NumPy carrier → `np.ndarray` | the data, so `np.asarray(x)` in user code keeps working |
| `np.asarray(g)` on a JAX carrier | JAX → `np.ndarray` | eager: a host copy; inside `jit`, `grad` or `vmap`: raises `GeoArrayTraceError` (GA-C1) |
| `from_georeader(gt)`, `to_georeader(g)` | `GeoTensor` ↔ `GeoArray` | `geoarray[georeader]`; NumPy only; values, transform, CRS, fill and attrs carried over, band names into a `SpectralAxis` |
| `Grid.affine`, `Grid.from_affine` | `affine.Affine` ↔ `Grid` | `affine` is a small BSD-licensed base dependency, already used by geoops |
| `to_xarray`, `from_xarray`, `to_coordax`, `from_coordax` | geoarray ↔ xarray, coordax | behind `[xarray]` and `[coordax]`; lossless for transform, dims and coordinates ([labeled lane](./labeled_lane.md)) |

`g.xp` returns the array namespace through `array_api_compat`, so a function written against `xp` runs on both backends.
`g.backend` returns `"numpy"` or `"jax"` for dispatch tables.

### NumPy and JAX, eagerly and under transforms (GA-C1)

JAX and NumPy already convert to each other outside a transform: `jnp` functions accept NumPy arrays, and `np.asarray` copies a JAX array to the host.
Inside `jit`, `grad` or `vmap` the data are tracers, which NumPy cannot read, so one check on the data catches all three transforms.
The prototype confirmed it: an eager `np.asarray` on JAX data returned a NumPy array, and the same call inside each transform raised the custom error instead of JAX's `TracerArrayConversionError`.

```python
# eager → host copy; traced → a geoarray error
def __array__(self, dtype=None, copy=None) -> np.ndarray:
    # tracers exist only if JAX is loaded, so NumPy-only installs never import it
    jax_loaded: bool = "jax" in sys.modules
    if jax_loaded and _is_tracer(self.data):
        raise GeoArrayTraceError(
            "a NumPy function received GeoArray data inside jit/grad/vmap; "
            "use a JAX-capable operator, or call it outside the transform"
        )
    return np.asarray(self.data, dtype=dtype)
```

That host copy is for user code and notebooks.
Operators do not rely on it: each declares its `backends`, and one given a carrier it cannot run refuses with a `BackendError` naming the fix (`to_numpy()`, or the JAX-capable operator).
Mixed NumPy and JAX inputs refuse and name `to_jax()` rather than promoting.

What runs where, for geoarray's own functions:

| Function | NumPy | JAX | Under `jit` |
| --- | --- | --- | --- |
| properties, `grid`, `bounds`, `res` | yes | yes | `grid` needs a concrete transform |
| `pixel_coords`, `valid_mask`, `map`, `combine` | yes | yes | yes |
| `isel`, `sel` | yes | yes | concrete slices and labels only |
| `crop` | yes | yes | traced offset, static size |
| `read_window`, `write_window`, `pad` | yes | yes | concrete window |
| `stack`, `concat`, `reduce`, `map_over` | yes | yes | yes |
| `footprint` | yes | no | no (shapely) |

### When to jit

JAX is for the compiled step: the model, the loss, its gradient and the optimizer, run many times on fixed shapes.
Pre-processing before it and post-processing after it stay NumPy and eager, run once per file.
The carrier crosses the boundary twice, with `to_jax()` and `to_numpy()`; the run modes themselves are defined in the [vision](./vision.md).

```text
PRE · NumPy, eager, once per file
read COG / NetCDF · reproject · mask clouds
rasterize polygons · split into patches
(geoops NumPy operators, geopatcher, geocatalog)
        │
        │ to_jax()      fixed shapes from here
        ▼
COMPILED · JAX, jit + grad
normalize · augment · model forward
H(x) · averaging kernel · loss · ∇ · optimizer step
(GeoML, GeoModels, dual-backend geoops)
        │
        │ to_numpy()
        ▼
POST · NumPy, eager, once per result
threshold · vectorize plumes · stitch patches
metrics · write COG / Zarr · plots
(geoops NumPy operators, geopatcher, xrops)
```

The test for any operator: put it inside `jit` when it runs many times on the same shapes, a gradient must flow through it, and it is array maths (elementwise, einx, linear algebra); keep it NumPy when it runs once per file, its shapes depend on the data (vectorized polygons, variable-length tracks), or it calls GDAL, rasterio, georeader, SciPy, scikit-image or scikit-learn.

Some operators sit on both sides.
`normalize` runs in pre for a one-off product and inside the training step when its statistics are learned; that is why a handful of geoops operators are dual-backend.
The ones that pass the test and call no CPU-only library run on JAX: indices, QA, array-maths `normalize`, compositing (as `reduce` / `argreduce` + `take_along`), basic augmentation.
`matched_filter` and `plume` move to radtrax and plumax (geotoolz #482).

## Running example: one methane plume, two sensors

The problem: a methane plume over a well pad in the Delaware Basin, seen by EMIT and by Sentinel-2 within the same week.
EMIT gives the spectral detection, and Sentinel-2 gives a finer grid to confirm it on.
The goal is a stack of co-registered 64 × 64 patches that a plume detector can score.

The data:

- **Sentinel-2 L1C**, one tile in UTM 13N (EPSG:32613), bands B11 (1,610 nm) and B12 (2,190 nm) at 20 m, so 5,490 × 5,490 pixels per band.
  Values are top-of-atmosphere reflectance, with fill 0 in the product and NaN after conversion to reflectance.
- **EMIT L1B radiance**, 285 bands from 381 to 2,493 nm, about 1,242 lines × 1,280 samples in sensor geometry: a `GeoSwath["band row col"]`.
  Its geometry lookup table (GLT) orthorectifies it onto an EPSG:4326 grid of about 0.00054° (≈ 60 m).
- **The area of interest** is a 12 km × 12 km box around the pad, which is 600 × 600 pixels on the Sentinel-2 grid.

Sizes used below: C bands, H × W pixels, P patches, h × w patch size; T = (a, b, c, d, e, f) is the affine transform.

### Stage 1: describe a chip (geocatalog → geoarray)

Let $x \in \mathbb{R}^{C \times H \times W}$ be the chip, with C = 2, H = W = 600.
Its grid is the triple (T, crs, (H, W)), where T maps pixel coordinates (column j, row i) to map coordinates.
Pixel (i, j) covers a square whose centre is

$$
\begin{pmatrix} x_{ij} \\ y_{ij} \end{pmatrix} = \begin{pmatrix} a & b \\ d & e \end{pmatrix} \begin{pmatrix} j + 1/2 \\ i + 1/2 \end{pmatrix} + \begin{pmatrix} c \\ f \end{pmatrix}
$$

For a north-up grid b = d = 0, a is the pixel width and e is minus the pixel height.
The window for a bounding box (x_min, y_min, x_max, y_max) is T⁻¹ applied to its corners, rounded outward to whole pixels.

```python
import geoarray as ga
import geocatalog as gc

# a NumPy carrier on the tile's native grid, dims named by the reader
s2: GeoArray["band y x"] = gc.read(item, bounds=aoi, bands=("B11", "B12"))
# the band coordinate: names and centre wavelengths, nm
swir: SpectralAxis = s2.coord("band")
# the grid is a hashable value, (T, crs, (H, W))
grid: ga.Grid = s2.grid
# pixel centres: x_ij = a(j + ½) + b(i + ½) + c, y_ij = d(j + ½) + e(i + ½) + f
xy: Float[np.ndarray, "2 600 600"] = ga.pixel_coords(s2)
# (x_min, y_min, x_max, y_max) from T and (H, W)
bounds: tuple[float, float, float, float] = grid.bounds
```

`GeoTensor` gives the same properties, but its `pixel_coords` equivalent lives in `georeader.griddata.meshgrid` and goes through rasterio.
geoarray keeps the maths and drops rasterio, so the same lines run on JAX data in Stage 4.

### Stage 2: valid pixels and band selection (geoarray)

Let φ be the fill value, possibly NaN.
A pixel (i, j) is valid when every band is finite and none equals the fill:

$$
v_{ij} = \bigwedge_{k=1}^{C} \left[ \mathrm{isfinite}(x_{kij}) \wedge x_{kij} \neq \phi \right]
$$

When φ is NaN the second term is already covered by the first.
georeader's `validmask()` computes `values != fill`, and NaN ≠ NaN is always true, so a NaN-filled raster reports every pixel valid.
geoops already works around this in `geoops._src.valid`; the correct rule moves into the type.

```python
# v_ij = ∧_k [isfinite(x_kij) ∧ x_kij ≠ φ], k over the band dim
valid: Bool[np.ndarray, "600 600"] = s2.valid_mask
# keep the band dim; the SpectralAxis is sliced with the data, so band names stay ("B12",)
b12: GeoArray["band y x"] = s2.isel(band=slice(1, 2))
# by label, dropping the band dim
b12_map: GeoArray["y x"] = s2.sel(band="B12")
# a window moves the origin: c′ = c + a·j₀ + b·i₀, f′ = f + d·j₀ + e·i₀
rows: slice = slice(150, 450)
cols: slice = slice(150, 450)
centre: GeoArray["band y x"] = s2.isel(y=rows, x=cols)
```

Under `GeoTensor`, `isel` passed `attrs` through unchanged and `wrap_like` dropped per-band keys later only if the band count changed; with the band names on a coordinate, there is nothing to keep in sync.

### Stage 3: one grid for both sensors (geoarray → a geoops geometry operator)

Let $z \in \mathbb{R}^{2 \times H_s \times W_s}$ hold two EMIT layers on its EPSG:4326 grid (T_s, crs_s): band `sim_B12` is EMIT radiance integrated over Sentinel-2's B12 spectral response, and band `dXCH4` is the matched-filter enhancement ΔX.
The GLT step that produced z from the swath is a gather, not an affine map, and runs as a geoops operator before this stage.
Let (T_d, crs_d, (H, W)) be the destination grid from Stage 1, and π the map from crs_d to crs_s.
Each destination pixel (i, j) is pulled back to fractional source coordinates

$$
u_{ij} = T_s^{-1}\left( \pi\left( T_d (j + 1/2, i + 1/2) \right) \right) - (1/2, 1/2)
$$

and the warped value is the bilinear blend of the four source pixels N(u_ij) around it, with weights $w_{kl} \ge 0$ that sum to one:

$$
\tilde{z}_{ij} = \sum_{(k, l) \in N(u_{ij})} w_{kl}(u_{ij}) z_{kl}
$$

```python
import geoops

# the two orthorectified EMIT layers; band = ("sim_B12", "dXCH4")
z: GeoArray["band y x"] = emit_layers
# move data and transform to the device; requires jax_enable_x64
s2_jax: GeoArray["band y x"] = s2.to_jax()
z_jax: GeoArray["band y x"] = z.to_jax()
# a geometry operator: u_ij from T_d, π and T_s⁻¹, then z̃_ij = ∑ w_kl z_kl
to_s2: Operator = geoops.geom.Warp(target=grid, order=1)
z_on_s2: GeoArray["band y x"] = to_s2(z_jax)
```

`Warp` is a hand-written geometry operator: it derives u from the two grids, with pyproj on NumPy data and a closed-form kernel on JAX data (EPSG:4326 to UTM here), then calls the L1 primitive `map_coordinates` (SciPy's or `jax.scipy.ndimage`'s) on each band.

### Stage 4: sub-pixel co-registration (leaf transform → jax.grad)

Let $r \in \mathbb{R}^{H \times W}$ be Sentinel-2's B12 on the Stage 1 grid, and m the warped simulated B12 from Stage 3.
The two differ in calibration and resolution, so both are standardised to zero mean and unit variance; write r̃ and m̃.
Let δ ∈ ℝ² shift the moving image's origin, so T_m(δ) = (a, b, c + δ_x, d, e, f + δ_y).
Let Ω be the interior pixels, 8 pixels away from the edges.
The shift estimate minimises the squared misfit:

$$
\hat{\delta} = \arg\min_{\delta \in \mathbb{R}^2} \sum_{(i, j) \in \Omega} \left( \tilde{m}_{ij}(\delta) - \tilde{r}_{ij} \right)^2,
\qquad
\tilde{m}(\delta) = \mathrm{std}\left( \mathcal{W}(m; T_m(\delta)) \right)
$$

Here 𝒲 is the Stage 3 warp, which is differentiable in δ because the transform is a JAX leaf, and std(·) subtracts the mean and divides by the standard deviation.
The bilinear loss is piecewise quadratic in δ, with kinks at whole-pixel shifts, so start within one pixel of the answer.
The same δ̂ then moves the enhancement ΔX.

```python
import einx
import jax
import jax.numpy as jnp
import lineax as lx


def standardise(d: Float[Array, "H W"]) -> Float[Array, "H W"]:
    # μ = (1 / HW) ∑ᵢⱼ dᵢⱼ
    mu: Float[Array, ""] = einx.mean("h w ->", d)
    # d − μ
    centred: Float[Array, "H W"] = d - mu
    # s, the standard deviation of the centred field
    s: Float[Array, ""] = jnp.std(centred)
    # std(d) = (d − μ) / s
    return centred / s


# the interior Ω: an 8-pixel border dropped
inner: slice = slice(8, -8)
# r: Sentinel-2 B12 on the Stage 1 grid
b12_jax: GeoArray["y x"] = s2_jax.sel(band="B12")
# r on Ω
b12_inner: GeoArray["y x"] = b12_jax.isel(y=inner, x=inner)
# r̃ = std(r)
ref: Float[Array, "H W"] = standardise(b12_inner.data)
# m: the simulated B12 on the Sentinel-2 grid
sim_b12: GeoArray["y x"] = z_on_s2.sel(band="sim_B12")


def misfit(delta: Float[Array, "2"]) -> Float[Array, ""]:
    # T_m(δ) = (a, b, c + δx, d, e, f + δy); a float64 leaf, so ∂ misfit / ∂δ exists
    shifted: GeoArray["y x"] = sim_b12.shift_origin(delta)
    # 𝒲(m; T_m(δ)) back onto the reference grid
    moved: GeoArray["y x"] = to_s2(shifted)
    # the warped field on Ω
    moved_inner: GeoArray["y x"] = moved.isel(y=inner, x=inner)
    # m̃(δ)
    moved_std: Float[Array, "H W"] = standardise(moved_inner.data)
    # m̃ − r̃
    resid: Float[Array, "H W"] = moved_std - ref
    # ∑_Ω (m̃ − r̃)²
    return einx.dot("h w, h w ->", resid, resid)


# ∇δ misfit and ∇²δ misfit
grad_fn: Callable = jax.grad(misfit)
hess_fn: Callable = jax.hessian(misfit)

# Newton steps from δ = 0
delta: Float[Array, "2"] = jnp.zeros(2)
for _ in range(20):
    # g = ∇δ misfit, H = ∇²δ misfit at the current δ
    g: Float[Array, "2"] = grad_fn(delta)
    h: Float[Array, "2 2"] = hess_fn(delta)
    # H s = g, solved, never inverted
    h_op: lx.MatrixLinearOperator = lx.MatrixLinearOperator(h)
    step: Float[Array, "2"] = lx.linear_solve(h_op, g).value
    # δ ← δ − s
    delta = delta - step

# T_m(δ̂) applied to the enhancement layer
enhancement: GeoArray["band y x"] = z_on_s2.isel(band=slice(1, 2))
shifted_final: GeoArray["band y x"] = enhancement.shift_origin(delta)
# ΔX aligned on the Sentinel-2 grid
aligned: GeoArray["band y x"] = to_s2(shifted_final)
```

On a synthetic 64 × 64 scene with a 20 m grid and a true 7.3 m shift along x, the prototype recovered 7.296 m in 20 Newton steps, an error of 0.0002 pixel.
That is the payoff of keeping the transform a leaf: the same code also batches scenes with different origins without recompiling.

### Stage 5: patches for the detector (geopatcher → geoarray.crop)

With patch size h = w = 64 and stride s = 32 on a 600 × 600 chip, the patch origins are $o_{mn} = (s m, s n)$ for $m, n \in \{0, 1, \ldots, 16\}$, which gives P = 17² = 289 patches.
Patch p is x[…, o_p : o_p + (h, w)], and its transform shifts the origin by o_p = (o_{p,1}, o_{p,2}) (row, column):

$$
c_p = c + a o_{p,2} + b o_{p,1}, \qquad f_p = f + d o_{p,2} + e o_{p,1}
$$

Offsets vary per patch and are traced; the size is static, which is what `lax.dynamic_slice` needs.

```python
import geopatcher as gp

# x = [B11, B12, ΔX] on one grid, C = 3; grids checked, never aligned
parts: list[GeoArray["band y x"]] = [s2_jax, aligned]
stack: GeoArray["band y x"] = ga.concat(parts, dim="band")
# offsets planned on the host from the grid alone, then ga.crop vmapped over them in one trace;
# the result gains the patch dim and a (289, 6) transform; a GeoPatches
patches: GeoArray["patch band y x"] = gp.split(stack, size=64, overlap=32)
```

Each patch keeps its own transform row, so a detection at patch pixel (i, j) maps straight back to map coordinates.
The prototype ran the same `vmap(crop)` pattern with three offsets in one trace and got the expected origins.

## Demo API

Further uses of the carrier, beyond the running example.
Names are proposed.

### A swath wrapper calling an L1 kernel: the TROPOMI averaging kernel

l is the retrieval layer (L = 12), N the swath pixels, A the column averaging kernel, x the model sub-column, xₐ the prior sub-column and Xₐ the prior XCH₄.

```python
# L1, in radtrax: shapes only
def apply_column_kernel(
    ak: Float[Array, "L N"],
    model: Float[Array, "L N"],
    prior: Float[Array, "L N"],
    x_prior: Float[Array, "N"],
) -> Float[Array, "N"]:
    # xₗ − xₐ,ₗ
    departure: Float[Array, "L N"] = model - prior
    # ∑ₗ Aₗ (xₗ − xₐ,ₗ)
    correction: Float[Array, "N"] = einx.dot("l n, l n -> n", ak, departure)
    # Xₘ = Xₐ + ∑ₗ Aₗ (xₗ − xₐ,ₗ)
    return x_prior + correction


# behind radtrax[geo]: GeoModels may import GeoStack, never the reverse,
# so a wrapper that calls a radtrax kernel lives in radtrax, not in geotoolz.
# GeoSwathProfile ≡ GeoSwath["level row col"]; the column map is a GeoSwath["row col"]
def model_xch4(
    ak: GeoSwathProfile,
    model: GeoSwathProfile,
    prior: GeoSwathProfile,
    x_prior: GeoSwath["row col"],
) -> GeoSwath["row col"]:
    # level first, pixels flattened: (L, N)
    ak_rows: Float[Array, "L N"] = geoarray.to_rows(ak, dim="level")
    model_rows: Float[Array, "L N"] = geoarray.to_rows(model, dim="level")
    prior_rows: Float[Array, "L N"] = geoarray.to_rows(prior, dim="level")
    x_prior_rows: Float[Array, "N"] = geoarray.to_rows(x_prior)
    # the L1 call
    xm: Float[Array, "N"] = apply_column_kernel(
        ak_rows, model_rows, prior_rows, x_prior_rows
    )
    # back onto the swath: same lat, lon
    xm_swath: GeoSwath["row col"] = x_prior.with_rows(xm)
    return xm_swath
```

### Points against a state: a data-assimilation cost

y is the observed SLA, H the sampling operator, R = σ²I the observation-error covariance, σ the along-track noise in m.

```python
# along-track SLA, padded to a fixed N: values on obs; xy, time and valid are leaves (a GeoTracks)
obs: GeoPoints["obs"] = GeoPoints(
    sla,
    dims=("obs",),
    xy=lonlat,
    time=t_obs,
    valid=is_real,
    crs="EPSG:4326",
)


@jax.jit
def cost(ssh: GeoMap, obs: GeoTracks, sigma: float) -> Float[Array, ""]:
    # H(x): bilinear sample of the state at each observation
    predicted: GeoPoints["obs"] = geoarray.sample(ssh, obs)
    # y − H(x)
    innovation: Float[Array, "N"] = obs.data - predicted.data
    # m ∈ {0, 1}: 1 for a real observation, 0 for a padded slot
    m: Bool[Array, "N"] = obs.valid
    # d = m ⊙ (y − H(x)); padded slots give 0, never NaN, so ∇ₓJ stays finite
    d: Float[Array, "N"] = jnp.where(m, innovation, 0.0)
    # dᵀd
    dd: Float[Array, ""] = einx.dot("n, n ->", d, d)
    # J(x) = ½ dᵀ R⁻¹ d over real observations only
    return 0.5 * dd / sigma**2


# ∇ₓJ, the gradient a 4D-Var minimizer needs
grad_cost: Callable = jax.grad(cost)
grad: GeoArray["y x"] = grad_cost(ssh, obs, 0.03)
```

Positions change daily but are a leaf, so nothing recompiles; each day is padded to a fixed N, with `valid` (a boolean leaf, False on padding) as the mask.
The mask goes through `jnp.where`, not a multiply: a NaN-filled pad times 0 is still NaN, and would poison both J and ∇ₓJ.

Running a geoops operator over a lazy cube (`xrops.lift`) is on the [operators](./operators.md) page.

## Migration

Two changes reach geotoolz, in order: the carrier replaces `GeoTensor`, then stored dims replace positional ones.

### What the audit found

Five overlaps between georeader and geotoolz need action, and a sixth is healthy for NumPy but blocks JAX.
The audit read georeader `main` at `9812be1` (v2.4.1, 7 Oct 2026) and geotoolz `main` at `48a705a` on 17 Oct 2026.

| Symbols | Defined in | Used by | Verdict |
| --- | --- | --- | --- |
| `GeoTensor` (ndarray subclass with `transform`, `crs`, `fill_value_default`, `attrs`) | `georeader/geotensor.py` (2,566 lines) | files naming it: geoops 50, geopatcher 19, geoproducts 12, geocloud 6, geocatalog 5 | **Split ownership.** The type is georeader's, but its rules (fresh attrs, NaN nodata, per-band keys) are geotoolz's, in `AGENTS.md` and `geoops._src` |
| `validmask`, `invalidmask` vs `valid_pixels`, `invalid_values` | `geotensor.py`; `geoops/_src/valid.py` | geoops (2 calls to the former) | **True duplicate.** georeader's `values != fill` misses NaN fills; geoops's version is correct |
| `same_extent` vs `require_grid_match`, `grid_matches` | `geotensor.py` (transform tolerance 1e-3); `geoops/_src/geo.py` (exact) | geoops | **True duplicate.** The two disagree for transforms that differ below 1e-3 |
| `window_from_bounds`, `round_outer_window`, `calculate_transform_window`, `read_from_window` | `georeader/read.py`, `georeader/window_utils.py` | geopatcher (7 import lines) | **Split ownership.** geopatcher reaches into georeader internals for window maths |
| `resize`, `pad`, `meshgrid(dst_crs)`, `read_reproject` | `geotensor.py` (via scikit-image), `griddata.py`, `read.py` | geoops 10 + 10 calls; geopatcher | **Split ownership.** These change values or grids, so they belong in geoops |
| `__array_ufunc__` and 12 arithmetic dunders | `geotensor.py` | implicitly, any `gt + x` | **Healthy** for NumPy, but they rely on subclassing, which JAX arrays do not support |

Smaller findings:

- **License.** georeader is LGPL-3.0 and geotoolz is MIT.
  geoarray is a fresh implementation of the behaviour, not a copy of `geotensor.py`; georeader stays importable behind an extra and serves as a test oracle.
- **JAX was barely in geotoolz.** Only `geopatcher._src.jax` (`batch_split`) and one training integration imported it, both optionally.
- **What geoops reads off the carrier.** `.transform` 82 times, `.crs` 82, `.attrs` 41, `.values` 31, `.bounds` 29, `.fill_value_default` 25, `stack` and `concatenate` 16 each, `resize` and `pad` 10 each, `read_from_window` 9, `isel` 8, `res` 7.
  These counts set the minimum API.
- **georeader's default fill is 0.** A float raster read without a fill therefore treats real zeros as nodata (GA-A4).
- **The xarray bridge stores CRS differently.** georeader's `toDataArray` writes `attrs["crs"]`, while xrtoolz uses rioxarray's `spatial_ref` coordinate; the [labeled lane](./labeled_lane.md) settles it.

### Step 1: one carrier, one coordinated release (GA-D1)

All five packages switch to `GeoArray` in one coordinated, breaking release; no operator ever accepts both carriers.

| Symbol | Destination | Reason |
| --- | --- | --- |
| `geoops._src.wrap.wrap_like` | the adapter's `rewrap`; the `array_as_geotensor` path is removed | one carrier, one rewrap |
| `geoops._src.bands.PER_BAND_KEYS` | a `SpectralAxis` built by the readers and the georeader adapter | band metadata lives on the coordinate |
| geocatalog, geocloud, geoproducts readers | return carriers | native grid, NumPy data |

1. **Switch every package on one branch per repo.** geoops rewraps through the adapter only, and its validity, grid and band helpers call geoarray.
   The readers return carriers, and geopatcher replaces its seven georeader import lines with `Grid.window` and `crop`.
   A NumPy `GeoArray` implements `__array__`, so `np.asarray(x)` inside each operator keeps working.
2. **Reject `GeoTensor` at the door.** A `GeoTensor` is an `np.ndarray` subclass, so `np.asarray` would accept it silently as a plain array and the output would lose its georeferencing.
   Each operator therefore checks its carriers before `np.asarray` and raises a `TypeError` naming `from_georeader`; there is no warning period.
3. **Drop georeader from the base dependencies.** It moves to extras in all five packages; import-linter enforces that base modules never import it.
4. **Release together.** The five packages and the umbrella publish in one release-please run, each with a `BREAKING CHANGE:` footer and a minor bump, and the umbrella pins the new set.

Downstream, `radtrax.geo` and `plumax.geo` take `GeoTensor`s (radtrax backlog RTX-30 and RTX-34).
They pin geotoolz below the breaking release until they migrate; each migration is one PR that calls `from_georeader` at its own boundary.

### Step 2: stored dims

| Before | After |
| --- | --- |
| dims inferred from `ndim` | dims stored; inferred from `ndim` for one release with a `DeprecationWarning` |
| 3-D is always `(band, y, x)` | any dims ending in the spatial dims, e.g. `(time, y, x)` |
| operators take `axis=-3` | operators take `dim="band"` |
| band names in `attrs["band_names"]` | a `SpectralAxis` in `coords` |
| `GeoArray[np.ndarray]` generic | dims-string subscript (`GeoArray["band y x"]`) |
| swaths and points have no carrier | `GeoSwath`, `GeoPoints` |
| xarray conversions ad hoc per package | `to_xarray()` / `from_xarray()` in geoarray, lossless |

This is one 0.X.0 bump of geotoolz-array, released like Step 1.

### Phases

```text
Step 1  GeoTensor → GeoArray
        type, geometry, interop: geotoolz-array 0.1.x
        gates GA-G1 to GA-G12
          │
          ▼
        one coordinated switch of ops, catalog, cloud,
        products, patcher; georeader to extras
        gates GA-G13, GA-G14
          │
          ▼
P0      spike: stored dims, 3 kernels,
        shared functions on 3 backends
        gates GA-G2, GA-G15 to GA-G22
          │
          ▼
P1      geoarray 0.2.0: stored dims, coords, types,
        check_dims, GeoSwath, GeoPoints, xarray converters;
        ndim inference deprecated; the 0.1.x suite passes
        with only the deprecation warning
          │
     ┌────┴────┐
     ▼         ▼
P2 I/O       P3 geopatcher
name dims,   split returns GeoPatches
GeoSwath     GA-G15
GA-G17
     └────┬────┘
          ▼
P4      geoops and xrops (operators page): dim=, backends,
        KernelOps, xrops.lift; after the pipekit-array release
        gates GA-G24 to GA-G26
          │
          ▼
P5      GeoModels: JAX-first kernels (radtrax, plumax),
        a coordax converter only if somax or vardax ask
        GA-G2 and GA-G16 rerun there
```

Versions only bump the minor number; nothing reaches 1.0 in this plan.
Step 1's type is the only piece that blocks everything else, and it is small: the prototype for it is under 200 lines.

## Testing gates

Each phase merges only when its gates pass.
The "Was" column gives the gate's ID on the two pages this one merges.

| Gate | Was | Passes when |
| --- | --- | --- |
| GA-G1 Pytree round trip | GA1 | `unflatten(flatten(g))` is identical to g, and `jit(lambda g: g)(g)` returns equal data and transform bit for bit; if not: fix before anything else |
| GA-G2 One trace, many scenes | GA2, G1 | one trace across 100 random transforms of the same shape, dims, CRS, fill and attrs; a new CRS or shape adds exactly one trace; if not: the transform is not a leaf; fix first |
| GA-G3 Attrs are hashable | GA3 | equal attrs hash equal; lists become tuples; a dict value raises `TypeError` |
| GA-G4 x64 guard | GA4 | `to_jax` raises with x64 off, and the transform is float64 with it on |
| GA-G5 Slice parity | GB1 | for 1,000 random slices with steps 1 to 3, the transform equals `GeoTensor.isel`'s to 1e-9 in map units |
| GA-G6 Crop equals slice | GB2 | `crop` with a traced offset equals concrete slicing exactly; `vmap` over P offsets traces once |
| GA-G7 Window round trip | GB3 | `Grid.bounds_of(Grid.window(b)) == b` exactly for grid-aligned b, and contains b otherwise |
| GA-G8 NaN nodata | GB4 | `valid_mask` equals `geoops._src.valid.valid_pixels` on every geoops fixture, including NaN fills, where naive `!=` fails (prototype: 15 valid of 16, versus 16) |
| GA-G9 Coordinates follow selection | GB5 | after any selection along a dim, its coordinate has one entry per remaining index |
| GA-G10 Backend parity | GC1 | every geoarray function gives equal results on NumPy and JAX: exactly for indexing, to 1e-12 in float64 for `pixel_coords`; eager `np.asarray(g)` equals `g.to_numpy().data`, the same call raises `GeoArrayTraceError` under `jit`, `grad` and `vmap`, and mixed backends raise |
| GA-G11 georeader round trip | GC2 | `from_georeader(to_georeader(g))` is identical to g, for all geoops fixtures |
| GA-G12 Light base install | GC3 | `pip install geotoolz-array` into a fresh environment installs none of jax, rasterio, georeader, xarray; if not: move the import behind its extra |
| GA-G13 One carrier | GD1 | the full suites of all five packages pass on `GeoArray` fixtures, and every geoops operator given a `GeoTensor` raises a `TypeError` naming `from_georeader` |
| GA-G14 No georeader in base | GD2 | import-linter: no base module of the five packages imports georeader |
| GA-G15 Patches | G2 | 64 patches with different transforms run in one trace via `map_over`, and the stitch matches today's within float tolerance; if not: keep patches as raw arrays inside the model, carriers outside |
| GA-G16 DA gradient | G3 | `jax.grad(cost)` with respect to the state works with observations padded to a fixed N, and J and ∇ₓJ are unchanged by the pad length and finite with NaN-filled padding; if not: `sample` returns raw arrays; the carrier stops at the boundary |
| GA-G17 No JAX on I/O | G4 | geocatalog, geocloud and geoproducts import and read with JAX not installed; if not: move the import behind `[jax]` |
| GA-G18 Shared functions agree | G5 | each shared function gives identical results on NumPy, JAX and xarray (dask-backed included); if not: drop it from the shared set |
| GA-G19 No silent alignment | G6 | `combine` and `stack` raise on a 10⁻⁷ m offset on every backend; if not: the xarray path is not shared |
| GA-G20 Lossless round trip | G7 | `from_xarray(to_xarray(g))` returns the same transform, dims and coordinates for random windows and rotated grids; the geotoolz #463 and #464 regression cases pass; if not: conventions are fixed before release |
| GA-G21 Patch overhead | G8 | 10k patches through a 5-op chain cost no more than today's carrier plus a budget you set; if not: profile and cut before release |
| GA-G22 Reproducible costs | G9 | a committed benchmark script times the normalized difference and the window on the carrier and on xarray at the three shapes above, writes its output to a tracked file, and those numbers replace the indicative table; if not: drop the per-op cost argument from this page |
| GA-G23 Primitives stay pure | G10 | a test in geostack's dependency-graph suite finds no import of geoarray, xarray, pyproj, rasterio, geopandas or shapely in any `*/_src/array.py`; if not: move the offending code up into the operator |
| GA-G24 Warp parity (geoops) | GE1 | JAX bilinear warp equals rasterio's bilinear reprojection to 1e-5 relative on interior pixels |
| GA-G25 CRS kernels (geoops) | GE2 | each closed-form projection equals pyproj to 1 mm over its valid domain, in float64; if not: the CRS stays NumPy-only |
| GA-G26 Co-registration (geoops) | GE3 | Stage 4 recovers a known sub-pixel shift to 0.01 pixel on a synthetic scene (prototype: 0.0002 pixel) |

The three geoops gates check the JAX kernels geoarray depends on: bilinear and cubic warp (JAX's `map_coordinates` supports order ≤ 1, so cubic needs a local kernel), mean and mode resampling with static block sizes and class counts, closed-form `to_crs` for EPSG:4326, UTM, Web Mercator, polar stereographic, sinusoidal and geostationary, and the GLT gather.
A kernel without a JAX version raises an error naming the missing backend; it never converts silently.

## Decisions and open questions

IDs carry the page prefix `GA-`, so they stay unique across the design pages ([decisions](./decisions.md)).
Decisions from the GeoTensor design (17 Oct 2026) that the stored-dims design (21 Oct 2026) or the [operators](./operators.md) page replaced are marked superseded.

### Decisions

| ID | Question | Status | Recommended or decided | Alternative |
| --- | --- | --- | --- | --- |
| GA-A1 | Subclass or compose | recommended, to confirm | compose: a frozen dataclass holding `data` | keep the ndarray subclass for NumPy and add a JAX wrapper: two carrier types forever |
| GA-A2 | Transform precision | recommended, to confirm | float64 array leaf; `to_jax` requires `jax_enable_x64` | static float64 origin plus a float32 leaf offset: no global flag, but a new origin recompiles |
| GA-A3 | Metadata | recommended, to confirm; narrowed 21 Oct | static `FrozenAttrs` of hashable values for whole-carrier metadata; per-band metadata on a `SpectralAxis` | no attrs: breaks geoops's metadata reads (41 places) |
| GA-A4 | Default fill value | recommended, to confirm | `None` (no fill), set explicitly by readers | keep georeader's `0` for parity |
| GA-B1 | Arithmetic | recommended, to confirm | `g.map(f)` and `combine(f, *gs)`; no dunders | dunders with an eager grid check that skips under `jit` |
| GA-B2 | Per-band attrs on band selection | superseded 21 Oct | band metadata is a `SpectralAxis` coordinate, sliced with the data | — |
| GA-C1 | NumPy interop for JAX data | decided 17 Oct; mixed-input promotion superseded 21 Oct | `np.asarray(g)` copies eagerly and raises `GeoArrayTraceError` inside `jit`, `grad` or `vmap`; mixed NumPy and JAX inputs refuse and name `to_jax()`; operators declare `backends` and raise `BackendError` | `np.asarray(g)` on a JAX carrier always raises |
| GA-D1 | Migration style | decided 17 Oct | one coordinated breaking release; `GeoTensor` is never accepted next to `GeoArray` | a dual-carrier release in geoops first: two code paths in every operator |
| GA-E1 | Where JAX kernels live | superseded 22 Oct | L1 kernels in geoops wrapped as `KernelOp`s or called by geometry operators ([operators](./operators.md)), not a `geoops.functional` module | — |
| GA-T1 | Series names | decided 21 Oct | `GeoImageSeries` for frames on a grid, `GeoTimeSeries` for fixed locations; `GeoVideo` is not used (geotoolz #521, #518) | — |
| GA-T2 | Spectral bands and variables | decided 21 Oct | `band` and `variable` are separate dims | one dim for both |
| GA-T3 | Levels on observations | decided 21 Oct | `level` is allowed on observations (TROPOMI layers) | levels on gridded data only |
| GA-T4 | coordax | decided 21 Oct | an optional converter, not the foundation of the JAX lane: its static coordinates would recompile per scene ([labeled lane](./labeled_lane.md)) | coordax `Field` as the JAX carrier |

### Open questions

Settle these before the geotoolz issues #511–#534 are rewritten.

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| GA-Q1 | What do somax (esmax) and vardax take at their boundary? | Shaped arrays plus a grid object (L1); the carrier stops in the wrapper behind `[geo]`, so `sample` and H(x) live in geotoolz | models take carriers directly, which makes `to_coordax()` or carrier-aware models necessary |
| GA-Q2 | How is the georeference written into xarray? | CF `grid_mapping` plus a `GeoTransform` attribute, written by geoarray and readable by rioxarray, with no rioxarray dependency ([labeled lane](./labeled_lane.md), where it is LL-F1) | depend on rioxarray (pulls in rasterio and GDAL) |
| GA-Q3 | Where do the shared functions live? | **Decided 21 Oct 2026:** pipekit-array, over a named-array adapter; geoarray registers its adapter and re-exports them | — |
| GA-Q4 | What happens to geoops `compositing`? | keep named operators as thin, serializable wrappers over `stack`, `reduce`, `argreduce`, `take_along` | remove them; users compose the shared functions |
| GA-Q5 | Do geoops operators accept a `DataArray`? | kernel operators: yes, lazily through `apply_ufunc`; geometry operators: no, they raise and name `xrops.lift` ([operators](./operators.md)) | convert every operator at entry and back at exit |
| GA-Q6 | `map_over` on xarray | allowed via `apply_ufunc`, documented as per-item Python cost unless dask-parallelized | not shared on xarray |
| GA-Q7 | Dim names: closed vocabulary or conventions? | free names; geoops and the converters rely on the conventions (`time`, `member`, `level`, `band`, `variable`, `patch`); `check_dims` enforces each signature | a closed list, validated at construction |
| GA-Q8 | Curvilinear carrier name | `GeoSwath`, also used for fixed model grids | `GeoCurvilinear`, with `GeoSwath` an alias |
| GA-Q9 | The dims checker | **Decided 21 Oct 2026:** `check_dims` on `pipekit.Signature` patterns, the single `Signature` | — |
| GA-Q10 | The existing issues | edit #511–#534 in place (numbers and links survive); block themes B–D on the spike #516; add pipekit and pipekit-array issues for the shared machinery | close and recreate |

Before the issues open:

- [ ] GA-A1 to GA-A4 and GA-B1 confirmed or flipped.
- [ ] Package name confirmed: `geotoolz-array`, import `geoarray`, with its own icon like the other subpackages.
- [ ] geotoolz, georeader (read only), radtrax and plumax checked for duplicate open issues.
- [ ] The [labeled lane](./labeled_lane.md) reviewed, so the converters' extras exist when geoarray 0.2.0 ships.

## Limitations

These come with the design; each has a workaround rather than a fix.

- **Recompiles per shape, dims, CRS, fill, dtype, coordinates and attrs.** These are static, so every distinct combination compiles once.
  Fixed chip sizes, padded point sets and stripped attrs keep that to a handful per pipeline.
- **Eager JAX is slower than NumPy.** JAX dispatch costs tens of microseconds per operation, so a notebook calling operators one by one on JAX data loses to NumPy.
  NumPy stays the default for readers, and a pipeline compiles the whole chain.
- **x64 is a global switch.** Turning it on makes new JAX arrays default to float64, which costs memory and speed in code that does not set dtypes.
  geoarray keeps `data` in its own dtype, but user code outside geoarray is affected, and downstream training code may refuse the flag; the GA-A2 alternative avoids it at the cost of recompiles, behind the same API (only `to_jax` and the transform updates change).
- **No arithmetic operators.** `g + 1` becomes `g.map(lambda d: d + 1)`, and existing notebooks need edits (GA-B1).
- **Data-dependent shapes stay on the host.** `footprint`, `valid_footprint`, vectorising and connected components produce outputs whose size depends on the values, so they run on NumPy only.
- **Rotated grids are carried, not processed.** The type holds any affine transform, but warp and resample kernels may require `Grid.is_axis_aligned` and raise otherwise.
- **JAX's CRS coverage is a subset.** Six closed-form projections run inside `jit`; any other CRS works only on NumPy data, through pyproj.
- **No in-place writes.** `write_window` returns a new carrier, as JAX requires.
  NumPy code that relied on `write_from_window` mutating a mosaic buffer must assign the result.

## Risks

- **The coordinated release breaks a downstream user.** radtrax, plumax or a notebook still passes `GeoTensor`s on the day it lands.
  Fallback: they pin geotoolz below the breaking release, and the zero-copy `from_georeader` at their boundary makes each migration a few lines.
- **A `GeoTensor` slips through as a plain array.** Because it is an ndarray, an operator that skips the carrier check would accept it and drop its georeferencing without an error.
  Fallback: the check lives in the one rewrap path every operator uses, and gate GA-G13 sends a `GeoTensor` to every operator.
- **Stored dims and positional callers disagree during the deprecation release.** A caller that still passes `axis=-3` to an operator whose input is `(time, y, x)` would reduce over the wrong dim.
  Fallback: operators that receive `axis=` on a carrier with stored dims raise instead of guessing.
- **License contamination.** Copying georeader code into geoarray would bring LGPL terms into an MIT package.
  Fallback: write geoarray from the behaviour and the tests on this page, and use georeader only as an oracle in tests.
- **Patch extraction clamps silently.** `lax.dynamic_slice` shifts an out-of-range offset back inside the array, which would misplace a patch without an error.
  Fallback: geopatcher's plan validates offsets on the host, and `crop` has a debug mode that checks bounds with `jax.experimental.checkify`.

## References

- georeader `GeoTensor`: [geotensor.py](https://github.com/spaceml-org/georeader/blob/main/georeader/geotensor.py), read at `9812be1`.
- geotoolz carrier contract: [AGENTS.md](https://github.com/jejjohnson/geotoolz/blob/main/AGENTS.md), "The two contracts", read at `48a705a`.
- Array API compatibility layer: [array-api-compat](https://data-apis.org/array-api-compat/).
- JAX pytrees and custom nodes: [Pytrees, "Custom pytree nodes"](https://docs.jax.dev/en/latest/101/pytrees.html).
- JAX 64-bit mode: [Double (64-bit) precision](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).
- coordax, checked against 0.2.9: [coordax](https://coordax.readthedocs.io/).
- Related design pages: [vision](./vision.md), [case studies](./case_studies.md), [labeled lane](./labeled_lane.md), [pipekit kernels](./pipekit_kernels.md), [operators](./operators.md), [stack levels](./stack_levels.md).
