---
title: "Labeled lane: xarray and coordax next to geoarray"
subtitle: Design · Converters between GeoArray and xarray or coordax, one CRS convention, and the rules that keep the two lanes apart
short_title: Labeled lane
description: The companion to the geoarray design. Labeled data stays in xarray on the host and coordax on JAX; geoarray adds converters behind extras, a CF spatial_ref and GDAL GeoTransform convention the stack already reads, and an AffineGrid coordax coordinate that round-trips losslessly through xarray.
date: 2026-10-17
keywords:
  - geoarray
  - xarray
  - coordax
  - DataTree
  - CF conventions
  - design
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

Labeled data (named dims, coordinates, attributes) stays in two existing libraries: xarray on the host and coordax on JAX.
geoarray adds converters between `GeoArray` and those two, behind extras, plus one coordax coordinate that carries an affine grid.
Three things do not change.
xrtoolz owns operators on xarray objects.
geoops owns operators on `GeoArray` and accepts nothing labeled.
Nobody builds a labeled array type of their own.

- **F. xarray converters.** Today only georeader converts, it writes the CRS as `attrs["crs"]`, and nothing in geotoolz or xrtoolz calls it.
  geoarray converts to and from `DataArray`, `Dataset` and `DataTree`, using the CF `grid_mapping` and GDAL `GeoTransform` convention that rioxarray, xrtoolz, geopatcher and geocloud already share.
- **G. coordax converters.** coordax has no georeferenced coordinate, so x and y round-trip as plain labels and the CRS is lost.
  geoarray adds an `AffineGrid` coordinate that round-trips losslessly through xarray, and a `lift` helper that maps geoops kernels over named axes.
- **H. Lane rules.** Four carrier types per operator would multiply every test.
  Conversions are explicit, labeled inputs to geoops raise an error naming the converter, and per-band metadata has one mapping in each direction.

End state: a `GeoArray` crosses to xarray and coordax and back without losing its grid, CRS, fill value or per-band metadata.
Each lane keeps its own operators.

## Ownership rules

The two lanes meet only at geoarray's converters.

```
                 host / NumPy             JAX
              ┌────────────────────┬────────────────────┐
  grid lane   │ GeoArray[np]       │ GeoArray[jax]      │  geoops
              ├────────────────────┼────────────────────┤
  labeled     │ xarray             │ coordax Field      │  xrtoolz / lift(geoops)
  lane        │ DataArray, Dataset,│ with AffineGrid    │
              │ DataTree           │                    │
              └────────────────────┴────────────────────┘
                      ▲                      ▲
        geoarray.interop.xarray   geoarray.interop.coordax
             [xarray] extra           [coordax] extra
```

1. **geoarray.interop** owns the converters and the `AffineGrid` coordinate.
   It imports xarray and coordax only inside those modules, and only behind their extras.
2. **xrtoolz** owns operators on xarray objects.
   It never needs geoarray; when a user wants a geoops operator on xarray data, they convert.
3. **coordax** is used, not wrapped.
   geoarray adds one `Coordinate` subclass and one helper, and nothing else.
4. **geoops** accepts `GeoArray` and plain arrays only.
   A `DataArray` or a coordax `Field` raises an error naming `from_xarray` or `from_coordax`.

### Decisions this doc takes

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| F1 | CRS on the xarray side | A scalar `spatial_ref` coordinate with `crs_wkt` and GDAL `GeoTransform` attributes, written without importing rioxarray | georeader's `attrs["crs"]`, which no other package reads |
| F2 | Per-band metadata on the xarray side | `wavelengths` become the `band` coordinate's values, `band_names` a non-index coordinate on `band`; other keys go to `.attrs` | Everything into `.attrs`, as georeader does |
| F3 | Multi-resolution products | `to_datatree({"r10m": g10, "r20m": g20, …})`, one node per grid, each with its own `spatial_ref` | A `Dataset` per resolution, joined by the caller |
| G1 | Where `AffineGrid` lives | `geoarray.interop.coordax` | Upstream in coordax, which ties our release to theirs |
| G2 | Static grid in coordax | Accept it: one compile per grid; batching and gradients use `GeoArray` | A leaf-transform wrapper around `Field`, which breaks coordax's rule that coordinates are static |
| H1 | Operators on labeled data | xrtoolz for xarray; `lift(kernel)` for coordax | geoops accepts `DataArray` and `Field` directly |
| H2 | Who owns the converters | geoarray, behind extras | xrtoolz, which would then depend on geoarray |

## Running example: the same plume, with its labels

The data are the same as in [the geoarray design](./geoarray.md): EMIT's 285-band radiance and a Sentinel-2 L1C tile over a Delaware Basin well pad.
What changes is what the data carry.
EMIT's bands have wavelengths λ₁ … λ₂₈₅ from 381 to 2,493 nm that analysis code wants to select by value.
Sentinel-2 has three native resolutions, 10, 20 and 60 m, that belong in one object.

### Stage 1: EMIT radiance to a CF-compliant file (geoarray → xarray)

**TL;DR.** Save the orthorectified EMIT cube so any CF reader recovers its grid, CRS and wavelengths.

**Problem.**
Let $x \in \mathbb{R}^{C \times H \times W}$ be the radiance, C = 285, on the grid (T, crs, (H, W)) with T = (a, b, c, d, e, f).
The x and y coordinates are pixel centres, and the band coordinate holds the wavelengths:

$$
x_j = c + a (j + 1/2)
$$

$$
y_i = f + e (i + 1/2)
$$

$$
\lambda_k = \mathrm{wavelengths}[k]
$$

These hold for an axis-aligned grid (b = d = 0); a rotated grid has no 1-D x and y coordinates and is written with `GeoTransform` only.
GDAL orders the same six numbers differently, as (c, a, b, f, d, e), and that string goes into `spatial_ref.attrs["GeoTransform"]`.

```python
import xarray as xr
from geoarray.interop.xarray import to_xarray

# proposed (F): dims ("band", "y", "x"); band coordinate λ_k from attrs["wavelengths"]
da: xr.DataArray = to_xarray(emit)

# today (xarray): λ selection by value, 2,100 nm to 2,450 nm
swir: xr.DataArray = da.sel(band=slice(2100.0, 2450.0))

# today (xarray): CF output; spatial_ref carries crs_wkt and GeoTransform
swir.to_netcdf("emit_swir.nc")
```

The file opens in rioxarray, QGIS or geoops's `ReadNetCDF` with its grid intact, because all of them read `spatial_ref`.
georeader's `toDataArray` would have written the CRS as a plain attribute that none of them read.

### Stage 2: Sentinel-2's three resolutions in one tree (geoarray → xarray.DataTree)

**TL;DR.** Hold the 10, 20 and 60 m bands together, and pull one resolution back out as a `GeoArray`.

**Problem.**
Let g₁₀, g₂₀ and g₆₀ be three `GeoArray`s on nested grids of the same tile.
Their transforms differ only in the pixel size: a = −e = 10, 20 and 60 m, with the same origin (c, f).
A `DataArray` holds one grid, so the three need a tree with one node per grid.

```python
import geoarray as ga
import xarray as xr
from geoarray.interop.xarray import from_xarray, to_datatree

# proposed (F): one node per grid, each with its own spatial_ref
tree: xr.DataTree = to_datatree({"r10m": g10, "r20m": g20, "r60m": g60})

# proposed (F): the 20 m node back as a GeoArray; T rebuilt from GeoTransform
node = tree["r20m"].to_dataset()
g20_back: ga.GeoArray = from_xarray(node["B12"])
```

xrtoolz's `Operator` already dispatches over `DataTree` nodes, so per-resolution operators work on the tree unchanged.

### Stage 3: named-axis processing on JAX (geoarray → coordax)

**TL;DR.** Run a geoops kernel band by band on JAX, with the band axis named and its wavelengths kept.

**Problem.**
Let f be a kernel that maps one band image to another on the same grid, $f: \mathbb{R}^{H \times W} \to \mathbb{R}^{H \times W}$.
Applying it to every band is a map over the named axis "band":

$$
y_{k} = f(x_{k}) \text{ for } k = 1, \ldots, C
$$

coordax's `cmap` vectorises over every named axis that the function does not consume, so the spatial axes are untagged first and tagged again after.

```python
import coordax as cx
import geoarray as ga
import geoops.functional as F
from geoarray.interop.coordax import AffineGrid, lift, to_coordax

# proposed (G): Field with axes (band: LabeledAxis of λ, y × x: AffineGrid)
fld: cx.Field = to_coordax(emit.to_jax())

# proposed (G): lift untags (y, x), applies f per band, and retags with the same grid
# F.gaussian_filter is proposed (E) in the geoarray design
smooth = lift(F.gaussian_filter, sigma=1.0)

# proposed (G): y_k = f(x_k) for k = 1, …, C
out: cx.Field = smooth(fld)
```

`lift` passes `out_axes="same_as_input"` to `cmap`, because the default, `"trailing"`, moves the band axis to the end.
In the check for this doc, the default turned `("band", "y", "x")` into `("y", "x", "band")`.

## Current-state audit

Two conventions exist for the CRS on the xarray side, and coordax has no georeferenced coordinate.
The audit read georeader `9812be1`, geotoolz `48a705a` and xrtoolz `5a2ba2c` on 17 Oct 2026, and coordax 0.2.9 from PyPI.

| Symbols | Defined in | Used by | Verdict |
| --- | --- | --- | --- |
| `toDataArray`, `fromDataArray`, `coords_to_transform` | `georeader/dataarray.py` (145 lines) | no caller in geotoolz or xrtoolz | **Split ownership.** CRS and fill go to `attrs`; the transform is inferred from the first two coordinates, so a one-pixel axis cannot convert |
| `assign_crs`, `get_crs`, `reproject` | `xrtoolz/geo/_src/crs.py` (rioxarray `.rio`) | xrtoolz geo operators | **Healthy.** `spatial_ref` convention |
| `RioXarrayField` | `geopatcher/_src/fields/rio_xarray.py` | geopatcher | **Healthy.** Same convention, through `.rio` |
| `ReadNetCDF`, `read_netcdf` | `geoops/io/_src/operators.py`, `geocloud/_src/hdf.py` | geoops io | **Healthy.** Reads CF `grid_mapping` and GDAL `GeoTransform` |
| `from_xarray` default coordinate types | coordax 0.2.9 | — | **Split ownership.** x and y come back as `LabeledAxis`; the CRS is dropped |

Smaller findings from the coordax check:

- **Coordinates are static.** coordax requires every `Coordinate` subclass to be a static pytree node.
  Shifting an `AffineGrid` by one pixel recompiled a jitted function in the check (two traces), where the same shift on a `GeoArray` did not.
- **`cx.wrap` is deprecated.** In 0.2.9 it warns and points to `cx.field`, with the same signature.
  Code in this doc uses `cx.field`.
- **There is a lossless hook.** `cx.from_xarray(da, coord_types=(AffineGrid, …))` asks each listed coordinate type to recognise the dims.
  With `AffineGrid.to_xarray` writing `spatial_ref` and `AffineGrid.from_xarray` reading it, a field went to xarray and back with an equal `AffineGrid` and equal data.

## Workstream F: xarray converters

geoarray converts both ways between `GeoArray` and `DataArray`, `Dataset` and `DataTree`, using the CF and GDAL conventions the stack already reads.

| Symbol | Destination | Reason |
| --- | --- | --- |
| `georeader.dataarray.toDataArray` | `geoarray.interop.xarray.to_xarray` | `spatial_ref` instead of `attrs["crs"]` (F1) |
| `georeader.dataarray.fromDataArray` | `geoarray.interop.xarray.from_xarray` | reads `spatial_ref`, falls back to `attrs["crs"]` |
| `coords_to_transform` | `GeoTransform` first, coordinates second | works for one-pixel axes and rotated grids |
| per-band attrs | `band` coordinate values and non-index coordinates (F2) | selection by wavelength |
| — | `to_dataset`, `to_datatree`, `from_dataset`, `from_datatree` | several variables, several grids (F3) |

What crosses, in each direction:

| GeoArray | xarray | Back |
| --- | --- | --- |
| `data` | values, dims `("time", "band", "y", "x")[-ndim:]` | exact |
| `transform` | `spatial_ref.attrs["GeoTransform"]`, plus x and y centre coordinates when axis-aligned | exact, to float64 round-off |
| `crs` | `spatial_ref.attrs["crs_wkt"]` | equal `pyproj.CRS` |
| `fill_value` | `_FillValue` encoding and `attrs["fill_value"]` | exact, including NaN |
| `attrs["wavelengths"]` | `band` coordinate values | tuple of floats |
| `attrs["band_names"]` and other per-band keys | non-index coordinates on `band` | tuples |
| other attrs | `.attrs` | hashable values only; others raise on the way back |

1. **Write `spatial_ref` by hand.** A scalar coordinate with `crs_wkt`, `spatial_ref` (the same WKT) and `GeoTransform` attributes, plus `grid_mapping="spatial_ref"` on each variable's encoding.
   No rioxarray import is needed.
2. **Read in order of trust.** `GeoTransform` first, then the centre coordinates, then georeader's `attrs["crs"]` with a deprecation warning.
3. **Check the coordinates against the transform.** When both exist, the centres must equal T applied to (j + ½, i + ½) within 1e-9 of a pixel, or the read raises.
4. **Trees by grid.** `to_datatree` takes a mapping of names to `GeoArray`s.
   Nodes with equal grids may share a node as variables of one `Dataset`.

## Workstream G: coordax converters

geoarray adds an `AffineGrid` coordinate and a `lift` helper; everything else is coordax as it ships.

| Object | Lives in | Name | Kind |
| --- | --- | --- | --- |
| Affine grid coordinate | `geoarray.interop.coordax` | `AffineGrid` | static `cx.Coordinate`, dims `("y", "x")` |
| Converters | `geoarray.interop.coordax` | `to_coordax`, `from_coordax` | functions |
| Kernel lifting | `geoarray.interop.coordax` | `lift` | `cx.cmap` with spatial axes untagged |
| xarray hook | `geoarray.interop.coordax` | `AffineGrid.to_xarray`, `AffineGrid.from_xarray` | `spatial_ref` and `GeoTransform`, as F1 |

```python
import dataclasses

import coordax as cx
import jax


# proposed (G): a static coordinate, as coordax requires
@jax.tree_util.register_static
@dataclasses.dataclass(frozen=True)
class AffineGrid(cx.Coordinate):
    # (a, b, c, d, e, f) as Python floats
    transform: tuple[float, float, float, float, float, float]
    # WKT, so the coordinate hashes without pyproj objects
    crs_wkt: str
    # (H, W)
    hw: tuple[int, int]

    @property
    def dims(self) -> tuple[str, str]:
        return ("y", "x")

    @property
    def shape(self) -> tuple[int, int]:
        return self.hw

    # x_j = c + a(j + ½), y_i = f + e(i + ½), as coordinate fields
    @property
    def fields(self) -> dict[str, cx.Field]: ...

    # x, y and spatial_ref with crs_wkt and GeoTransform (c, a, b, f, d, e)
    def to_xarray(self) -> dict: ...

    # rebuild from spatial_ref; NoCoordinateMatch if absent
    @classmethod
    def from_xarray(cls, dims, coords) -> "AffineGrid | cx.NoCoordinateMatch": ...
```

1. **`to_coordax(g)`.** Leading axes become `LabeledAxis` coordinates where attrs give their labels (wavelengths for band) and plain names otherwise.
   The spatial axes become one `AffineGrid`.
2. **`from_coordax(field)`.** Requires an `AffineGrid` on the last two axes, and raises if the field has none.
3. **`lift(kernel, **kwargs)`.** Untags `("y", "x")`, calls `cx.cmap(kernel, out_axes="same_as_input")`, and retags with the input's `AffineGrid`.
   A kernel that changes the grid returns a plain array and must be retagged by the caller.
4. **The xarray hook.** `to_xarray` on a field goes through `AffineGrid.to_xarray`, and geoarray's `from_xarray_field(da)` calls `cx.from_xarray(da, coord_types=(AffineGrid, cx.LabeledAxis, cx.DummyAxis))`.

## Workstream H: lane rules

Each operator family takes one lane, and every crossing is a named call.

- **Explicit conversion.** No converter runs implicitly, and no operator accepts the other lane's types.
  The error names the function that converts.
- **One per-band mapping.** The per-band key list (`PER_BAND_KEYS`) lives in geoarray, and both converters read it, so a key that is per-band on one side is per-band on the other.
- **Coordax is for named-axis code, GeoArray for grid code.** Use coordax where axis names carry meaning across a computation, such as band, time or ensemble member.
  Use `GeoArray` where the grid itself moves, such as warps, crops of many scenes, or gradients through the transform.
- **xarray is for files, dask and analysis.** Out-of-core reductions and CF output stay on xarray; the `GeoArray` lane is dense and in memory.

## Testing gates

Each workstream merges only when its truth tests pass.

| Gate | Workstream | Passes when |
| --- | --- | --- |
| GF1 xarray round trip | F | `from_xarray(to_xarray(g))` is identical to g for every geoarray fixture, including NaN fills and 285 wavelengths |
| GF2 Readers agree | F | a file written by `to_netcdf` opens in rioxarray and in geoops's `ReadNetCDF` with an equal transform (to 1e-9 of a pixel) and an equal CRS |
| GF3 Coordinates agree | F | a `DataArray` whose centres disagree with its `GeoTransform` by more than 1e-9 of a pixel raises |
| GF4 Tree round trip | F | the three Sentinel-2 grids of Stage 2 come back from `to_datatree` with equal grids per node |
| GG1 coordax round trip | G | `from_coordax(to_coordax(g))` is identical to g, and `cx.from_xarray` with `AffineGrid` in `coord_types` returns an equal grid (passed in the check for this doc) |
| GG2 lift keeps axes | G | `lift(f)(field)` has the input's dims, in order, and an equal `AffineGrid` |
| GG3 Static grid documented | G | a test pins the behaviour: a new `AffineGrid` adds one trace, so a coordax change to it fails loudly |
| GH1 No implicit crossing | H | every geoops operator given a `DataArray` or a `Field` raises an error naming the converter |
| GH2 Extras stay optional | H | `import geoarray` works with neither xarray nor coordax installed |

## Roadmap and risks

The xarray converters ship first, because the file format is what other tools see; coordax follows.

```
  geoarray 0.1.0 (type, geometry, interop core)
       │
  F  xarray converters  ┃  G  coordax AffineGrid
       │                ┃       │
       ◆ GF1–GF4        ┃       ◆ GG1–GG3
       │
  H  lane rules in geoops
       │
       ◆ GH1 GH2  → geotoolz-array 0.2.0
```

- **coordax changes its API.** It is at 0.2.9 and describes itself as an experiment; `cx.wrap` was deprecated within the 0.2 series.
  Fallback: everything coordax-specific sits in one module behind an extra, and gate GG3 pins the behaviour this doc relies on.
- **Two CRS conventions linger.** Files written by georeader's `toDataArray` keep `attrs["crs"]`.
  Fallback: `from_xarray` reads it with a deprecation warning until georeader writes `spatial_ref` too.
- **Rotated grids on the xarray side.** They have no 1-D x and y coordinates.
  Fallback: write `GeoTransform` only and skip the centre coordinates; rioxarray reads that form.
- **The static grid surprises users.** Someone batching many scenes as coordax fields would recompile per scene.
  Fallback: `lift` documents it, and the batching recipe converts to `GeoArray` first.

## References

- geoarray design: [geoarray](./geoarray.md).
- CF conventions, grid mappings: [CF 1.11, section 5.6](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#grid-mappings-and-projections).
- rioxarray reads CRS, transform and nodata from the standard CF and GDAL locations: [Getting started](https://corteva.github.io/rioxarray/stable/getting_started/getting_started.html).
- coordax: [documentation](https://coordax.readthedocs.io/), checked against 0.2.9.
- xarray `DataTree`: [Hierarchical data](https://docs.xarray.dev/en/stable/user-guide/hierarchical-data.html).
