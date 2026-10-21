---
title: "GeoStack operators: geoops and xrops on one kernel model"
subtitle: "Design · layers × carrier families, kernels written once"
short_title: Operators
description: "How geoops and xrops are organised on a grid of layers (L1 shapes, L2 dim names, L3 geometry) and carrier families (geoarray, xarray), with L1 kernels wrapped once as KernelOps that run in both."
date: 2026-10-21
keywords:
  - geoops
  - xrops
  - KernelOp
  - operators
  - apply_ufunc
  - JAX
  - pipekit
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Scope

This page decides how operations are organised across geoops (geotoolz-ops) and xrops (xrtoolz-ops): what each package writes, what it reuses, and where every kind of operation lives, so nothing is written twice.
It sits between two others:

- [geoarray: one carrier per geometry](./04_geoarray_named_carriers.md) defines the data: geoarray carriers, coordinates, readable types, static versus leaf.
- [pipekit: named arrays, one Signature, kernels written once](./06_pipekit_named_arrays.md) defines the machinery this page reuses: the named-array adapter, the single `Signature`, the dim-level functions, `KernelSpec` and `KernelOp`, `JaxOperator`.

This page adds no machinery of its own; where it needs some, the need goes to pipekit.

:::{note} How this page relates to the earlier design pages
It replaces the `geoops.functional` wording in [](./01_geoarray.md) with the kernel model below: L1 kernels wrapped as `KernelOp`s, defined in [](./06_pipekit_named_arrays.md).
:::

## The grid

Two independent questions place every operation: what it needs to know (the layer, a row) and how much data it runs on (the carrier family, a column).

| Needs | geoarray carriers (one item, eager or `jit`) | xarray carriers (many items, by value, lazy) |
| --- | --- | --- |
| **L3 geometry** | geoops: reproject, rasterize, sample H(x), plume detection, patch/stitch | xrops: regrid a cube, mosaic many files |
| **L2 dim names** | pipekit-array dim-level functions; `KernelOp`s over a named dim | the same functions and `KernelOp`s; native value operations (`resample`, `groupby`, ranges); xrops analysis |
| **L1 shapes** | kernels on `.data` | kernels via `apply_ufunc` |

The two carrier families:

| Family | Carriers | Good at |
| --- | --- | --- |
| geoarray | `GeoArray`, `GeoSwath`, `GeoPoints` | one item or a fixed batch: strict grids, compiles, \~1–10 µs per op |
| xarray | `DataArray`, `Dataset`, `DataTree` | many items: selection by value, laziness, out-of-core, variables with different dims |

The five kinds of operation and their owners:

| Kind | Owner | Runs on |
| --- | --- | --- |
| L1 kernels | gaussx, somax, radtrax, … and geoops's `array.py` modules | raw arrays |
| dim-level functions | pipekit-array | every registered carrier |
| `KernelOp`s | wrapped by geoops or xrops, machinery in pipekit-array | every registered carrier |
| geometry operators | geoops (per item), xrops (per cube) | their column |
| native xarray operations | xarray | xarray carriers |

**The consequence that matters most:** an L2 operation built as a `KernelOp` is not column-specific.
`NDVI(dim="band")` runs on a `GeoImage` eagerly, on a JAX carrier under `jit`, and on a lazy cube through `apply_ufunc`, from one definition.
Only L3 geometry and xarray's value operations stay tied to a column.

## Where each operation is written, once

| Operation | Written in | Runs on |
| --- | --- | --- |
| L1 kernel | its L1 package (radtrax, somax, …) or the `array.py` modules of geoops and xrops | raw arrays |
| dim-level function (`reduce`, `stack`, `map_over`, …) | pipekit-array | every registered carrier |
| `KernelOp`, observation and sensor domain (indices, radiometry, qa, normalize, spectral) | geoops | both columns |
| `KernelOp`, physical and state domain (atmosphere, ocean, budgets, metrics) | xrops | both columns |
| per-item geometry (reproject, rasterize, sample, windows, patches) | geoops | geoarray carriers; cubes through `geoops.lift` |
| many-item geometry (lazy regrid of a cube, mosaics of many files) | xrops | xarray carriers |
| value analysis (`resample`, `groupby`, climatology, alignment) | native xarray; xrops builds on it | xarray carriers |

The split of `KernelOp`s by domain decides who maintains them, not where they run.

### Dependency direction

```text
pipekit ──▶ pipekit-array ──▶ geoarray ──▶ geoops
                  │                         │
                  │ [xarray] adapter        │ [xarray] extra: geoops.lift
                  ▼                         ▼
                xrops ◀─────── never ─────── ✗
```

- **xrops never imports geoops or geoarray**, as the [alignment page](./03_package_alignment.md) already requires.
  It reaches xarray carriers through the pipekit-array adapter.
- **geoops may import xarray**, behind an extra, for one job: `geoops.lift(op, dim="time")` runs a per-item geometry operator on each slice of a cube.
  This replaces the `xrops.ApplyGeoOp` proposed earlier, which would have broken the rule above.
- **A cube user who needs NDVI imports geoops.**
  That costs geoarray's base install (NumPy, pyproj, affine), no JAX.

### Lifting a geometry operator over a cube

```python
# a per-item L3 operator, written once for geoarray carriers
detect = geoops.plume.DetectPlumes(threshold=tau)

# lifted over time: each slice becomes a GeoImage, runs, and is wrapped back
per_date = geoops.lift(detect, dim="time")

# lazy: one task per dask chunk
masks: xr.DataArray = per_date(cube)
```

Under the hood `lift` is `xr.apply_ufunc` with the spatial dims as core dims; each chunk is turned into a carrier with `GeoArray.from_xarray`, so the georeference comes from the cube's grid mapping.

## geoops

geoops holds three kinds of code: L1 kernels in its `array.py` modules, observation-domain `KernelOp`s that run on both columns, and per-item geometry operators that run on geoarray carriers.
Today it has 218 operator classes; 61 store a positional `self.axis`, and 106 kernel signatures default to `axis=-3`.

### Three operator shapes

| Shape | Base | Backends | Example |
| --- | --- | --- | --- |
| kernel operator (L2) | `KernelOp` | numpy, jax; xarray through the adapter | `NDVI(dim="band")` |
| geometry operator, array math (L3) | `Operator`, reads the georeference | numpy, jax | `Sample`, `Window`, `ShiftOrigin` |
| geometry operator, engine-bound (L3) | `Operator`, calls georeader or rasterio | numpy | `Reproject`, `Rasterize`, `Vectorize` |

```python
# L1 kernel in geoops/indices/_src/array.py: "(),()->()"
def normalized_difference(a: Float[Array, "*s"], b: Float[Array, "*s"], eps: float) -> Float[Array, "*s"]:
    # (a − b) / (a + b + ε)
    return (a - b) / (a + b + eps)


# L2 operator: two labels of one dim, then the kernel
class NDVI(Operator):
    backends = ("numpy", "jax", "xarray")

    def __init__(self, nir: str = "B8", red: str = "B4", dim: str = "band", eps: float = 1e-10) -> None:
        self.nir = nir
        self.red = red
        self.dim = dim
        self.eps = eps

    def _apply(self, scene):
        # exact label lookups, any carrier
        nir = pipekit_array.sel(scene, **{self.dim: self.nir})
        red = pipekit_array.sel(scene, **{self.dim: self.red})
        # the kernel, element-wise, through KernelOp
        nd = KernelOp(normalized_difference, "(),()->()", core={}, eps=self.eps)
        return nd(nir, red)
```

```python
# L3, engine-bound: carrier-only, NumPy-only
class Reproject(Operator):
    backends = ("numpy",)

    def _apply(self, scene: GeoImage) -> GeoImage:
        # georeader underneath; refuses JAX carriers and DataArrays with a BackendError
        ...
```

### Families

| Family | Shape | Backends |
| --- | --- | --- |
| `indices`, `qa` | kernel operators | numpy, jax, xarray |
| `normalize` | kernel operators; skimage-backed ones engine-bound | numpy, jax, xarray / numpy |
| `compositing` | thin named operators over `stack`, `reduce`, `argreduce`, `take_along` | numpy, jax, xarray |
| `einx` | kernel operators over einx patterns (see O2) | numpy, jax, xarray |
| `augment` | kernel operators for flips, crops, noise (PRNG key under JAX); scipy warps engine-bound | numpy, jax / numpy |
| `spectral` | kernel operators backed by scipy | numpy, xarray |
| `learn` | scikit-learn wrappers | numpy |
| `geom`, `io`, `viz` | geometry, engine-bound | numpy |
| `mask`, `measure`, `feature`, `segment`, `restore`, `radiometry` | mixed: vector and solar-geometry parts engine-bound, array parts kernel operators | per op |
| `matched_filter`, `plume` | leaving for radtrax and plumax (#482); rebuilt there as kernel operators | — |

### Rules

- **`dim=` replaces `axis=`** in every operator; kernels keep a positional axis and take it last.
- **Every operator declares `backends`.**
  Given a carrier it cannot run on, it raises `BackendError` naming the fix (`to_numpy()`, `geoops.lift`, or the right column).
- **Fitted operators** follow pipekit's `FittableTransformer`: `fit` returns a new operator with arrays filled; under JAX, `pipekit_jax.to_jax(op)` makes those arrays leaves.
- **No geoops operator reimplements a pipekit-array function or a native xarray operation.**

## xrops

xrops is the xarray column: value analysis built on native xarray operations, many-item geometry, and the physical- and state-domain `KernelOp`s, which also run on geoarray carriers.
It is the code of today's `xrtoolz` distribution, renamed per the [alignment page](./03_package_alignment.md), with `xrcore`'s operator base folded into `xrops._src`.

### What it writes and what it reuses

| xrops writes | xrops reuses |
| --- | --- |
| state-domain kernel operators (atmosphere, ocean, budgets, metrics, spectral transforms) | `KernelOp`, the dim-level functions (pipekit-array) |
| value analysis: climatologies, anomalies, detrending, regimes, extremes | `groupby`, `resample`, `rolling`, `sel` by value (xarray) |
| many-item geometry: lazy regrid of a cube, mosaics of many files, masks by region | `apply_ufunc`, dask (xarray) |
| `Dataset` and `DataTree` dispatch: run an operator per variable or per node | `pipekit.Signature`, `Operator` |

### A state-domain kernel operator on both columns

```python
# L1 kernel in xrops/ocn/_src/array.py: consumes the level axis, "(l),(l)->()"
def mixed_layer_depth(
    sigma0: Float[Array, "... L"],
    depth: Float[Array, "... L"],
    threshold: float,
) -> Float[Array, "..."]:
    # numpy or jax.numpy, chosen by the input
    xp = array_namespace(sigma0)
    # Δσ(z) = σ₀(z) − σ₀(z_ref), with z_ref the shallowest level
    delta = sigma0 - sigma0[..., :1]
    # first level where Δσ exceeds the threshold
    below = delta > threshold
    first = xp.argmax(below, axis=-1)
    # MLD = z at that level
    mld = xp.take_along_axis(depth, first[..., None], axis=-1)[..., 0]
    # argmax of an all-False row is 0: no crossing means no MLD, so NaN
    return xp.where(xp.any(below, axis=-1), mld, xp.nan)


mld = KernelOp(mixed_layer_depth, "(l),(l)->()", core={"l": "level"}, threshold=0.03)

# a GLORYS-like cube (time, level, y, x), lazy → (time, y, x)
mld_cube: xr.DataArray = mld(sigma0_cube, depth_cube)

# one gridded profile on a geoarray carrier, GeoVolume (level, y, x) → GeoMap (y, x)
mld_map: GeoMap = mld(sigma0_volume, depth_volume)
```

σ₀ is potential density anomaly (kg m⁻³), z depth (m), and the threshold 0.03 kg m⁻³ a common MLD criterion.
A profile that never crosses the threshold (a well-mixed column, or one too shallow) returns NaN rather than the shallowest depth; a caller that wants the bottom depth as a fallback fills the NaNs explicitly.

### Geometry written once at L1, wrapped per column

A gradient needs grid spacing, which is geometry.
The finite-difference kernel takes spacings as arrays; each column only derives the spacings:

| Column | Wrapper | Spacing from |
| --- | --- | --- |
| geoarray | `Gradient` on geoarray carriers | the transform (`a`, `e`), or geodesic distances for a geographic CRS |
| xarray | `Gradient` on xarray carriers | coordinate values, with spherical metrics on lat/lon |

Two thin wrappers, one kernel.
Where they live is constrained twice over: GeoStack and GeoNumerics never import each other, and geotoolz and xrtoolz share a tier, so geoops and xrops never import each other either.
That leaves two homes, by what the gradient is for (O7):

- **A GeoStack diagnostic** (a gradient of SSH or of a plume map for analysis): one small central-difference kernel in GeoStack's lowest shared tier, pipekit-array, with the two wrappers in geoops and xrops. Today's `xrtoolz-grad` (`xrgrad`) becomes the xarray wrapper.
- **Model-grade calculus** (staggered grids, finite volumes, boundary conditions): finitevolX's kernels, raw arrays only. Carrier wrappers for them live in the GeoML or GeoModels package that needs them (somax, vardax), which may import both GeoNumerics and GeoStack; neither GeoStack nor finitevolX holds them.

### Families

| Family (today in xrtoolz) | Becomes |
| --- | --- |
| `atm`, `ocn`, `budgets` | state-domain kernel operators + value analysis |
| `metrics` | kernel operators over named dims; skill scores by value |
| `transforms` (Fourier, wavelet, DCT) | kernel operators over named dims |
| `interpolate` | native `interp` plus many-item geometry |
| `geo` | split: geometry (crs, mosaic, rasterize, footprint) → many-item geometry; analysis (detrend, extremes, regimes) → value analysis |
| `einx` (`xreinx`), `sklearn` (`xrsklearn`) | see O2 and O3 |
| `rs`, `ice` | removed (empty stubs; observation indices live in geoops and run on cubes as they are) |

## Testing

| Test | Lives in | Proves |
| --- | --- | --- |
| operator contract | geoops (today's `test_operator_contract.py`), xrops | config round trip, graph mode, declared `backends`, `BackendError` on the rest |
| `check_kernel_op` | pipekit-array test kit, run by both packages | each kernel operator gives equal results on a geoarray carrier, a JAX carrier, an xarray cube and raw arrays |
| lift parity | geoops | `geoops.lift(op, dim="time")` on a cube equals the op on each slice, eagerly |
| one trace | geoops, xrops | kernel operators compile once under `jit` for same-shape carriers |
| summary | both | `Sequential.summary` prints dims for every step, no `?` |
| laziness | xrops | operators on dask-backed cubes compute nothing until asked |

## Migration

```text
1 pipekit, pipekit-array: adapter, Signature, functions, KernelOp
        │
        ▼
2 geoarray: adapter (geoarray epic, theme A)
        │
   ┌────┴────┐
   ▼         ▼
3 geoops   4 xrops (alignment doc A–E)
axis→dim   rename, fold xrcore,
KernelOps  Signature re-export,
backends   KernelOps for transforms,
lift       metrics, ocn, atm
```

Each step is a 0.X.0 bump. geoops and xrops can move in parallel once pipekit-array and geoarray are released.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| O1 | Where kernel operators live | by domain: observation and sensor in geoops, physical and state in xrops; both run on both columns | a third, neutral package of kernel operators |
| O2 | Two einx bridges (`geoops.einx`, `xreinx`) | one named-dims einx bridge in pipekit-array (`[einx]` extra); geoops keeps only its spatial-survival rule | keep both |
| O3 | Two scikit-learn bridges (`geoops.learn`, `xrsklearn`) | `to_rows` / `from_rows` as dim-level functions in pipekit-array, one estimator wrapper on pipekit's `FittableTransformer` | keep both |
| O4 | Lifting geometry over cubes | `geoops.lift` behind `geoops[xarray]`, so xrops never imports geoops | `xrops.ApplyGeoOp`, breaking the dependency rule |
| O5 | How operators use `KernelOp` | by composition: an operator keeps its own config and calls a `KernelOp` inside (as `NDVI` does) | every kernel operator subclasses `KernelOp` |
| O6 | The `backends` vocabulary | `"numpy"`, `"jax"`, `"xarray"` | backends and carrier families declared separately |
| O7 | Where grid calculus lives | diagnostic gradients: one central-difference kernel in pipekit-array, wrapped by geoops and xrops (`xrgrad` becomes the xarray wrapper); model-grade calculus: finitevolX, wrapped by the GeoML or GeoModels package that uses it | xrops keeps `xrgrad` as is and geoops ships no gradient; simpler, but observation-side users have none |
