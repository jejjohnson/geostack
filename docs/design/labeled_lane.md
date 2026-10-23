---
title: "Labeled lane: crossing to xarray"
subtitle: "Design · lossless converters between geoarray carriers and xarray, one CRS convention, and where each workload runs"
short_title: Labeled lane
description: "How geoarray carriers cross to xarray and back without losing grid, CRS, fill or per-band metadata (CF spatial_ref plus GDAL GeoTransform), when to cross (one item or a fixed batch on a carrier, many items by value in xarray), and the lane rules: kernel operators run on DataArray, geometry operators reach cubes through xrops.lift, coordax is an optional converter."
date: 2026-10-23
keywords:
  - geoarray
  - xarray
  - DataTree
  - CF conventions
  - xrops.lift
  - coordax
  - design
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

Labeled data (named dims selected by value, many files, lazy cubes) stays in xarray.
geoarray carriers (`GeoArray`, `GeoSwath`, `GeoPoints`, on the [geoarray page](./geoarray.md)) cross to xarray and back through converters behind geoarray's `[xarray]` extra.
Nobody builds a labeled array type of their own, and geoarray never selects by value.

- **LL-F. xarray converters.**
  Today only georeader converts, it writes the CRS as `attrs["crs"]`, and nothing in geotoolz or xrtoolz calls it.
  geoarray converts to and from `DataArray`, `Dataset` and `DataTree`, using the CF `grid_mapping` and GDAL `GeoTransform` convention that rioxarray, xrtoolz, geopatcher and geocloud already share.
- **LL-G. coordax converter.**
  coordax is an optional converter for model code that wants `cmap`, not the foundation of the JAX lane.
  Its coordinates are static, so a georeferenced coordax grid recompiles for every new transform; batching and gradients stay on geoarray carriers.
- **LL-H. Lane rules.**
  Every crossing is a named call.
  Kernel operators run on a `DataArray` lazily; geometry operators reach a cube through `xrops.lift`; nothing converts silently.

End state: a carrier crosses to xarray and back without losing its grid, CRS, fill value, dims or per-band metadata, and each workload runs on the side built for it.

## Ownership rules

The two lanes meet only at geoarray's converters and at `xrops.lift`.

```text
 geoarray carriers (NumPy or JAX)
 one item or a fixed batch
 geoops operators, jit, grad, vmap
            │
            │  carrier.to_xarray()
            │  GeoArray.from_xarray(da)
            │  geoarray[xarray] extra
            ▼
 xarray: DataArray · Dataset · DataTree
 many items, by value, lazy
 xrops operators; xrops.lift(geoops op)
            │
            ┊  optional, for model code
            ┊  carrier.to_coordax()
            ▼
 coordax Field (static coordinates)
```

1. **geoarray** owns the converters, in `geoarray.interop.xarray` (and `geoarray.interop.coordax`), exposed as `to_xarray()` / `from_xarray()` on the carriers.
   It imports xarray and coordax only inside those modules, and only behind their extras.
2. **pipekit-array** owns the xarray named-array adapter behind its `[xarray]` extra (PK-P4), so a [KernelOp](#design-kernelop) runs on a `DataArray` without geoops importing xarray.
3. **xrops** owns xarray-native work: selection by value, many-file assembly, `Dataset` and `DataTree` dispatch, state physics, and `xrops.lift`, which runs a per-item geoops operator over a cube.
   It may import geoops and geoarray, one tier above them ([the tiers](#design-tiers)).
4. **geoops** never imports xarray.
   Its kernel operators accept a `DataArray` through the adapter (GA-Q5); its geometry operators raise on one and name `xrops.lift`.
5. **coordax** is used, not wrapped, and no geotoolz package depends on it.

### Decisions this page takes

| ID | Question | Recommended | Alternative | Status |
| --- | --- | --- | --- | --- |
| LL-F1 | CRS on the xarray side | A scalar `spatial_ref` coordinate with `crs_wkt` and GDAL `GeoTransform` attributes, written without importing rioxarray | georeader's `attrs["crs"]`, which no other package reads | decided (GA-Q2) |
| LL-F2 | Per-band metadata on the xarray side | `band` holds the names; `wavelength` (and FWHM) are non-index coordinates on `band` with their own index, so `sel(wavelength=…)` works; other keys go to `.attrs` | Wavelengths as the `band` values (the 17 Oct proposal), or everything into `.attrs` as georeader does | open; changed 23 Oct to match `SpectralAxis` on the geoarray page |
| LL-F3 | Multi-resolution products | `to_datatree({"r10m": g10, "r20m": g20, …})`, one node per grid, each with its own `spatial_ref` | A `Dataset` per resolution, joined by the caller | open |
| LL-G1 | Where a coordax grid coordinate lives | `geoarray.interop.coordax`, built only when somax or vardax ask for it | Upstream in coordax, which ties our release to theirs | deferred |
| LL-G2 | Static grid in coordax | Accept it: one compile per grid; batching and gradients use geoarray carriers | A leaf-transform wrapper around `Field`, which breaks coordax's rule that coordinates are static | decided; coordax is a converter, not a foundation |
| LL-H1 | Operators on labeled data | Kernel operators directly on `DataArray` (GA-Q5); geometry operators through `xrops.lift(op, dim=…)` (OP-O4); xarray-native work in xrops | geoops accepts every labeled type; or a coordax `lift` helper (the 17 Oct proposal) | decided 22 Oct 2026 |
| LL-H2 | Who owns the converters | geoarray, behind extras | xrtoolz, which would then depend on geoarray | decided |

(design-crossover)=
## The crossover: when to leave geoarray

The split is by workload, not by data maturity.
One item or a fixed batch, possibly under `jit`, is a geoarray carrier.
Many items selected by value, or anything assembled from more than one file, is xarray.
A model state inside a JAX model is shaped arrays, and a carrier appears only at its boundary ([the three levels](#design-levels)).

```text
 one item or a fixed batch
 (scene · swath · point set · patches · a state slice)
            │
            ▼
 geoarray carrier + geoops
 per-item operators, possibly under jit:
 radiometry, matched filter, reprojection, H(x)
            │  to_xarray()   ▲ from_xarray()
            ▼                │
 xarray + xrops
 many items, by value, lazy:
 stack by date, sel, resample,
 climatology, align, write Zarr
            │  .data plus derived geometry
            ▼
 shaped arrays in model code
 somax · vardax · filterax · radtrax
```

The test for any feature or operation:

| If it needs | It belongs to |
| --- | --- |
| a coordinate value (a date, a depth, a station ID, a wavelength) | xarray |
| more than one file | xarray, via geocatalog |
| a labeled state under `jit`, `grad` or `vmap` | shaped arrays in the model; a carrier only at its boundary |
| pixel georeferencing for one acquisition, possibly under `jit` | a geoarray carrier |

Hand-offs:

1. **xarray to a carrier at the `jit` boundary.**
   Select by value in xarray, then `GeoArray.from_xarray(cube.sel(time=…))`.
2. **A per-item geometry operator over a stack** runs through `xrops.lift(op, dim="time")`: each slice becomes a carrier, the operator runs, and the result is restacked.
3. **Observations meet states** in data assimilation as `GeoPoints` against a carrier slice of the model state; `geoarray.sample` is the bridge for H(x).

Recommended: this three-way split, with a thin carrier that stores dims but never selects by value, so it does not grow into a second xarray.
Alternative: xarray as the only NumPy-side container, with geometry checks rebuilt in an accessor.

## Running example: one tile, its labels and its time stack

The data are those of the [geoarray page](./geoarray.md): EMIT's 285-band radiance, orthorectified, and Sentinel-2 L1C over a Delaware Basin well pad.
What changes is what the data carry: EMIT's bands have wavelengths λ₁ … λ₂₈₅ from 381 to 2,493 nm that analysis code selects by value, Sentinel-2 has three native resolutions that belong in one object, and a year of Sentinel-2 dates is a cube.

### Stage 1: EMIT radiance to a CF-compliant file (carrier → xarray)

**TL;DR.** Save the orthorectified EMIT cube so any CF reader recovers its grid, CRS and wavelengths.

Let $x \in \mathbb{R}^{C \times H \times W}$ be the radiance, C = 285, on a pixel grid with transform T = (a, b, c, d, e, f) and a CRS.
For an axis-aligned grid (b = d = 0) the x and y coordinates are pixel centres:

$$
x_j = c + a \left(j + \tfrac12\right), \qquad y_i = f + e \left(i + \tfrac12\right)
$$

A rotated grid has no 1-D x and y coordinates and is written with `GeoTransform` only.
GDAL orders the same six numbers as (c, a, b, f, d, e), and that string goes into `spatial_ref.attrs["GeoTransform"]`.

```python
# emit: GeoArray["band y x"], C = 285, 60 m UTM grid, SpectralAxis with names and λ in nm
# dims as stored; xⱼ, yᵢ as centre coordinates; spatial_ref holds crs_wkt and GeoTransform (LL-F1)
da: DataArray["band y x"] = emit.to_xarray()

# select by value on the indexed wavelength coordinate: λ ∈ [2,100, 2,450] nm (LL-F2)
swir: DataArray["band y x"] = da.sel(wavelength=slice(2100.0, 2450.0))

# CF output; grid_mapping = "spatial_ref" on the variable
swir.to_netcdf("emit_swir.nc")
```

The file opens in rioxarray, QGIS or geoops's `ReadNetCDF` with its grid intact, because all of them read `spatial_ref`.
georeader's `toDataArray` would have written the CRS as a plain attribute that none of them read.

### Stage 2: Sentinel-2's three resolutions in one tree (carriers → `DataTree`)

**TL;DR.** Hold the 10, 20 and 60 m bands together, and pull one band back out as a carrier.

Let g₁₀, g₂₀ and g₆₀ be three carriers on nested grids of the same tile.
Their transforms differ only in pixel size, a = −e = 10, 20 and 60 m, with the same origin (c, f).
A `DataArray` holds one grid, so the three need a tree with one node per grid.

```python
# one node per grid, each with its own spatial_ref (LL-F3)
grids: dict[str, GeoArray["band y x"]] = {"r10m": g10, "r20m": g20, "r60m": g60}
tree: DataTree = geoarray.to_datatree(grids)

# the 20 m node: one variable, dims (band, y, x)
node: Dataset["band y x"] = tree["r20m"].to_dataset()
refl: DataArray["band y x"] = node["reflectance"]

# one band, by name
b12_da: DataArray["y x"] = refl.sel(band="B12")

# back to a carrier: T rebuilt from GeoTransform and checked against the centres
b12: GeoArray["y x"] = GeoArray.from_xarray(b12_da)
```

xrops dispatches per node over a `DataTree`, so per-resolution operators work on the tree unchanged (PK-P5).

### Stage 3: a year of dates (xarray, crossing once)

**TL;DR.** Keep the stack lazy in xarray, run kernel operators on it directly, cross to a carrier only where `jit` or a geometry operator needs one.

Let the cube hold T = 52 Sentinel-2 dates on one 20 m grid.
NDVI is a kernel over the band dim, with ρ the reflectance:

$$
\mathrm{NDVI} = \frac{\rho_{\mathrm{NIR}} - \rho_{\mathrm{red}}}{\rho_{\mathrm{NIR}} + \rho_{\mathrm{red}}}
$$

```python
# many files, lazy: one dask chunk per date
tiles: Dataset["time band y x"] = xr.open_mfdataset(
    paths, combine="by_coords", chunks={"time": 1}
)
cube: DataArray["time band y x"] = tiles["reflectance"]

# a kernel operator: lazy on the cube through apply_ufunc, one task per chunk (GA-Q5)
ndvi: Operator = geoops.NDVI(dim="band")
ndvi_cube: DataArray["time y x"] = ndvi(cube)

# a geometry operator needs a carrier per item: xrops.lift turns each date into a GeoImage (LL-H1)
to_wgs84: Operator = geoops.geom.Reproject(crs="EPSG:4326", resolution=0.0002)
per_date: Operator = xrops.lift(to_wgs84, dim="time")
ndvi_wgs84: DataArray["time y x"] = per_date(ndvi_cube)

# one date for a compiled step: select by value, then cross once
scene_da: DataArray["band y x"] = cube.sel(time="2026-07-14")
scene: GeoArray["band y x"] = GeoArray.from_xarray(scene_da)
jscene: GeoArray["band y x"] = scene.to_jax()

# the same kernel operator, compiled on the JAX carrier
ndvi_jit: Callable = jax.jit(ndvi)
ndvi_scene: GeoArray["y x"] = ndvi_jit(jscene)
```

Passing the cube to `to_wgs84` directly raises an error that names `xrops.lift`; passing a JAX carrier and a NumPy carrier to one operator raises and names `to_jax()`.

## Current-state audit

Two CRS conventions exist on the xarray side, and coordax has no georeferenced coordinate.
The audit read georeader `9812be1`, geotoolz `48a705a` and xrtoolz `5a2ba2c` on 17 Oct 2026, and coordax 0.2.9 from PyPI.

| Symbols | Defined in | Used by | Verdict |
| --- | --- | --- | --- |
| `toDataArray`, `fromDataArray`, `coords_to_transform` | `georeader/dataarray.py` (145 lines) | no caller in geotoolz or xrtoolz | **Split ownership.** CRS and fill go to `attrs`; the transform is inferred from the first two coordinates, so a one-pixel axis cannot convert |
| `assign_crs`, `get_crs`, `reproject` | `xrtoolz/geo/_src/crs.py` (rioxarray `.rio`) | xrtoolz geo operators | **Healthy.** `spatial_ref` convention |
| `RioXarrayField` | `geopatcher/_src/fields/rio_xarray.py` | geopatcher | **Healthy.** Same convention, through `.rio` |
| `ReadNetCDF`, `read_netcdf` | `geoops/io/_src/operators.py`, `geocloud/_src/hdf.py` | geoops io | **Healthy.** Reads CF `grid_mapping` and GDAL `GeoTransform` |
| `from_xarray` default coordinate types | coordax 0.2.9 | — | **Split ownership.** x and y come back as `LabeledAxis`; the CRS is dropped |

The coordax check also found that coordinates must be static pytree nodes: shifting a grid coordinate by one pixel recompiled a jitted function (two traces), where the same shift on a `GeoArray` did not.
That finding is why coordax is a converter here and not the JAX lane.

## Workstream LL-F: xarray converters

geoarray converts both ways between its carriers and `DataArray`, `Dataset` and `DataTree`, using the CF and GDAL conventions the stack already reads.

| Symbol today | Destination | Reason |
| --- | --- | --- |
| `georeader.dataarray.toDataArray` | `carrier.to_xarray()` (`geoarray.interop.xarray`) | `spatial_ref` instead of `attrs["crs"]` (LL-F1) |
| `georeader.dataarray.fromDataArray` | `GeoArray.from_xarray(da)` | reads `spatial_ref`, falls back to `attrs["crs"]` |
| `coords_to_transform` | `GeoTransform` first, centre coordinates second | works for one-pixel axes and rotated grids |
| per-band attrs | `band` names, `wavelength` and FWHM as indexed coordinates (LL-F2) | selection by wavelength |
| — | `to_dataset`, `to_datatree`, `from_dataset`, `from_datatree` | several variables, several grids (LL-F3) |

What crosses, in each direction:

| Carrier | xarray | Back |
| --- | --- | --- |
| `data` and stored dims | values, the same dims in the same order | exact |
| `GeoArray` transform | `spatial_ref.attrs["GeoTransform"]`, plus x and y centre coordinates when axis-aligned | exact, to float64 round-off |
| `GeoSwath` lat, lon | 2-D auxiliary coordinates on `(row, col)`, CF `coordinates` attribute | exact |
| `GeoPoints` xy, z, time | a CF Discrete Sampling Geometry `Dataset` on one `obs` dim | exact |
| CRS | `spatial_ref.attrs["crs_wkt"]` | equal `pyproj.CRS` |
| `fill_value` | `_FillValue` encoding and `attrs["fill_value"]` | exact, including NaN |
| `SpectralAxis`, `VariableAxis`, `VerticalAxis`, `TimeAxis` | CF coordinates and attributes (`wavelength`, `units`, `positive`) | equal coordinates |
| other attrs | `.attrs` | hashable values only; others raise on the way back |

1. **Write `spatial_ref` by hand.**
   A scalar coordinate with `crs_wkt`, `spatial_ref` (the same WKT) and `GeoTransform` attributes, plus `grid_mapping="spatial_ref"` in each variable's encoding.
   No rioxarray import is needed.
2. **Read in order of trust.**
   `GeoTransform` first, then the centre coordinates, then georeader's `attrs["crs"]` with a deprecation warning.
3. **Check the coordinates against the transform.**
   When both exist, the centres must equal T applied to (j + ½, i + ½) within 10⁻⁹ of a pixel, or the read raises.
4. **Trees by grid.**
   `to_datatree` takes a mapping of names to carriers; carriers with equal grids may share a node as variables of one `Dataset`.

## The coordax converter (LL-G)

`carrier.to_coordax()` and `from_coordax(field)` sit behind geoarray's `[coordax]` extra and map geoarray's coordinates to coordax coordinates, with the pixel grid as a static coordinate that writes and reads `spatial_ref` like LL-F1.
They ship only when somax or vardax ask for named-axis model code.
The 17 Oct proposal of a coordax `lift` helper for geoops kernels is dropped: KernelOps and `xrops.lift` cover it.

## Lane rules (LL-H)

- **Explicit crossings.**
  No converter runs implicitly; an operator given the wrong carrier raises and names the fix (`xrops.lift`, `to_jax()`, `from_xarray`).
- **One per-band mapping.**
  The per-band coordinate list lives in geoarray and both directions read it, so a key that is per-band on one side is per-band on the other.
- **Kernels everywhere, geometry per item.**
  A kernel operator (only dim names as geometry) runs on carriers and on `DataArray`; a geometry operator runs on one carrier and reaches a cube only through `xrops.lift`.
- **xarray for files, dask and analysis.**
  Out-of-core reductions and CF output stay on xarray; the carrier lane is dense and in memory.
- **Never align silently.**
  Two inputs on different grids raise; xarray's join on mismatched coordinates is never the default in an operator.

## Testing gates

Each workstream merges only when its truth tests pass.

| Gate | Workstream | Passes when |
| --- | --- | --- |
| GF1 xarray round trip | LL-F | `from_xarray(to_xarray(g))` is identical to g for every geoarray fixture, including NaN fills, 285 wavelengths, swaths and point sets |
| GF2 Readers agree | LL-F | a file written by `to_netcdf` opens in rioxarray and in geoops's `ReadNetCDF` with an equal transform (to 10⁻⁹ of a pixel) and an equal CRS |
| GF3 Coordinates agree | LL-F | a `DataArray` whose centres disagree with its `GeoTransform` by more than 10⁻⁹ of a pixel raises |
| GF4 Tree round trip | LL-F | the three Sentinel-2 grids of Stage 2 come back from `to_datatree` with equal grids per node |
| GG1 coordax round trip | LL-G | `from_coordax(to_coordax(g))` is identical to g; runs only once the converter ships |
| GH1 No implicit crossing | LL-H | every geometry operator given a `DataArray` raises an error naming `xrops.lift`; every kernel operator gives the same values on a carrier and on its `to_xarray()` |
| GH2 Extras stay optional | LL-H | `import geoarray` and `import geoops` work with neither xarray nor coordax installed |

## Roadmap and risks

The xarray converters ship with geoarray 0.2.0, because the file format is what other tools see; the lane rules land with the operators.

```text
 geoarray 0.2.0: carriers, coordinates, converters
            │
 LL-F  xarray converters
            ◆ GF1–GF4
            │
 LL-H  lane rules in geoops and xrops
            ◆ GH1, GH2  (with pipekit-array [xarray])
            ┊
 LL-G  coordax converter, only if a model asks
            ◆ GG1
```

- **Two CRS conventions linger.**
  Files written by georeader's `toDataArray` keep `attrs["crs"]`.
  Fallback: `from_xarray` reads it with a deprecation warning.
- **Rotated grids on the xarray side.**
  They have no 1-D x and y coordinates.
  Fallback: write `GeoTransform` only and skip the centre coordinates; rioxarray reads that form.
- **Wavelength selection needs a coordinate index.**
  `sel(wavelength=…)` works only if `to_xarray` sets an index on the non-dimension coordinate.
  Fallback: GF1 checks it, and the docs show `swap_dims` for files written elsewhere.
- **coordax changes its API.**
  It is at 0.2.9 and describes itself as an experiment.
  Fallback: everything coordax-specific sits in one module behind an extra that nothing in geotoolz needs.

## References

- Carriers and coordinates: [geoarray](./geoarray.md); operators and `xrops.lift`: [operators](./operators.md); the adapter: [pipekit kernels](./pipekit_kernels.md).
- CF conventions, grid mappings: [CF 1.11, section 5.6](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#grid-mappings-and-projections).
- rioxarray reads CRS, transform and nodata from the standard CF and GDAL locations: [Getting started](https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html).
- coordax: [documentation](https://coordax.readthedocs.io/), checked against 0.2.9.
- xarray `DataTree`: [Hierarchical data](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html).
