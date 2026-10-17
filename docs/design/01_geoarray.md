---
title: "geoarray: one NumPy/JAX carrier for geotoolz"
subtitle: Design · A replacement for georeader's GeoTensor whose data is a NumPy or a JAX array, with its geometry, migration, gates and roadmap
short_title: geoarray
description: The design of geotoolz-array (import geoarray), a frozen-dataclass carrier registered as a JAX pytree that replaces georeader's GeoTensor across geotoolz. It audits GeoTensor and its users, proposes the type, its geometry primitives, backends and a five-step migration, and checks the claims against a prototype on one methane plume seen by EMIT and Sentinel-2.
date: 2026-10-17
keywords:
  - geoarray
  - geotoolz
  - GeoTensor
  - JAX
  - NumPy
  - pytree
  - design
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

geotoolz gets a new bottom package, `geotoolz-array` (import `geoarray`), holding one carrier type, `GeoArray`, whose data is either a NumPy array or a JAX array.
It replaces `georeader.GeoTensor` as the type that flows between every geotoolz package, and georeader becomes an optional reader adapter.
Three things do not change.
pipekit's `Operator` contract stays as it is.
The carrier rules in geotoolz's `AGENTS.md` keep their meaning: dims, band axis −3, fresh `attrs`, NaN-aware nodata and exact grid matching.
The labeled lane (xarray and coordax) stays outside the type and gets its own companion page, [Labeled lane](./02_labeled_lane.md).

- **A. The type.** `GeoTensor` is an `np.ndarray` subclass, and JAX arrays cannot be subclassed that way.
  `GeoArray` is a frozen dataclass that holds its data, registered as a JAX pytree whose leaves are the data and a float64 transform.
- **B. Geometry primitives.** Window maths, slicing and crops live partly in `GeoTensor` and partly in georeader internals that geopatcher imports.
  geoarray owns every operation that re-indexes or re-describes pixels without changing their values, written once for both backends.
- **C. Backends and interop.** Code has to cross between NumPy, JAX and georeader without guessing.
  `to_jax`, `to_numpy` and a georeader adapter are explicit, and nothing converts implicitly.
- **D. Carrier migration.** Five packages import `GeoTensor` directly.
  geoops learns both carriers first, then readers return `GeoArray`, then georeader leaves the base dependencies.
- **E. JAX kernels.** Value-changing kernels (warp, resample, CRS maths) get a JAX implementation in `geoops.functional`, next to the NumPy one.
  geoarray never holds them.

End state: geoarray owns the carrier and its geometry, imports only NumPy, pyproj, affine and array-api-compat, and imports JAX lazily.
geoops owns every value-changing operation on both backends.
geocatalog, geocloud and geoproducts read native grids into NumPy `GeoArray`s.
geopatcher plans on the host and crops with `geoarray.crop`.
georeader is an optional extra.

## Ownership rules

Carriers flow down from geoarray; nothing below it imports a reader, and nothing in it changes pixel values.

```
  numpy · pyproj · affine · array-api-compat
          ┊ extras: jax, georeader, xarray, coordax
          │
  geotoolz-array           GeoArray, Grid, Window, FrozenAttrs
  (geoarray)               slicing, crop, windows, valid_mask
          │
  ┌───────┬────────┬───────┴──────┬──────────────┐
  catalog  cloud    products       patcher        ops ◀── pipekit
  native-grid reads → GeoArray     plan + crop    functional/ modules
          │
  geotoolz (umbrella)
          ┊ optional
  radtrax[geo] · plumax[geo]       downstream carriers
```

1. **geoarray** owns `GeoArray`, `Grid`, `Window`, `FrozenAttrs` and every function that re-indexes or re-describes pixels without changing their values.
   It never imports georeader, rasterio, xarray or coordax, and it imports JAX only inside functions that need it.
2. **geoops** owns every function that changes values or moves pixels to a new grid, in `functional/` (pure functions, one implementation per backend) and as `Operator`s.
   It never builds a carrier by hand; it rewraps through `GeoArray.with_data`.
3. **geocatalog, geocloud and geoproducts** read bytes into a NumPy `GeoArray` on the file's native grid.
   They never reproject, since that would make them import geoops.
4. **geopatcher** plans patch offsets on the host and extracts them with `geoarray.crop`.
   It stops importing `georeader.window_utils` and `georeader.read`.
5. **georeader** appears only behind extras: `geoarray[georeader]` for the carrier adapter, and the readers' own extras for file access.

### Decisions this doc takes

Each is recommended with its alternative; flip any before the issues open.

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| A1 | Subclass or compose | Compose: a frozen dataclass holding `data` | Keep the ndarray subclass for NumPy and add a JAX wrapper: two carrier types forever |
| A2 | Transform precision | float64 array leaf; `to_jax` requires `jax_enable_x64` | Static float64 origin plus a float32 leaf offset: no global flag, but a new origin recompiles |
| A3 | Metadata | Static `FrozenAttrs` of hashable values; 1-D numeric lists become tuples | No attrs (my first proposal in chat): breaks geoops's band lookup, which reads `.attrs` in 41 places |
| A4 | Default fill value | `None` (no fill), set explicitly by readers | Keep georeader's `0` for parity |
| B1 | Arithmetic | `g.map(f)` and `geoarray.combine(f, *gs)`; no dunders | Dunders with an eager grid check that skips under `jit` |
| B2 | Per-band attrs on band selection | geoarray slices `band_names`, `wavelengths` and the other per-band keys | Leave it to geoops's `wrap_like`, as today |
| C1 | Implicit NumPy interop for JAX data | `np.asarray(g)` on a JAX `GeoArray` raises | Allow it with a host transfer, as `jax.Array.__array__` does |
| D1 | Migration style | Dual carrier in geoops for one release, then readers switch | Big-bang switch in one release |

## Running example: one methane plume, two sensors

The problem: a methane plume over a well pad in the Delaware Basin, seen by EMIT and by Sentinel-2 within the same week.
EMIT gives the spectral detection, and Sentinel-2 gives a finer grid to confirm it on.
The goal is a stack of co-registered 64 × 64 patches that a plume detector can score.

The data:

- **Sentinel-2 L1C**, one tile in UTM 13N (EPSG:32613), bands B11 (1,610 nm) and B12 (2,190 nm) at 20 m, so 5,490 × 5,490 pixels per band.
  Values are top-of-atmosphere reflectance, with fill 0 in the product and NaN after conversion to reflectance.
- **EMIT L1B radiance**, 285 bands from 381 to 2,493 nm, about 1,242 lines × 1,280 samples in sensor geometry.
  Its geometry lookup table (GLT) orthorectifies it onto an EPSG:4326 grid of about 0.00054° (≈ 60 m).
- **The area of interest** is a 12 km × 12 km box around the pad, which is 600 × 600 pixels on the Sentinel-2 grid.

Dimensions, used everywhere below: C bands, H × W pixels, P patches, h × w patch size, T the affine transform.

### Stage 1: describe a chip (geocatalog → geoarray)

**TL;DR.** Read the Sentinel-2 bands for the area of interest and know exactly where every pixel sits.

**Problem.**
Let $x \in \mathbb{R}^{C \times H \times W}$ be the chip, with C = 2, H = W = 600.
Its grid is the triple (T, crs, (H, W)), where T = (a, b, c, d, e, f) maps pixel coordinates (column j, row i) to map coordinates.
Pixel (i, j) covers a square whose centre is

$$
\begin{pmatrix} x_{ij} \\ y_{ij} \end{pmatrix} = \begin{pmatrix} a & b \\ d & e \end{pmatrix} \begin{pmatrix} j + 1/2 \\ i + 1/2 \end{pmatrix} + \begin{pmatrix} c \\ f \end{pmatrix}
$$

For a north-up grid b = d = 0, a is the pixel width and e is minus the pixel height.
The window for a bounding box (x_min, y_min, x_max, y_max) is T⁻¹ applied to its corners, rounded outward to whole pixels.

```python
import geoarray as ga
import geocatalog as gc
import numpy as np
from jaxtyping import Float

# proposed (D): the catalog returns a NumPy GeoArray on the tile's native grid
# arguments sketched
s2: ga.GeoArray[Float[np.ndarray, "2 600 600"]] = gc.read(
    item, bounds=aoi, bands=("B11", "B12")
)

# proposed (A): the grid is a hashable value, (T, crs, (H, W))
grid: ga.Grid = s2.grid

# proposed (B): pixel centres, x_ij = c + a(j + ½), y_ij = f + e(i + ½)
xy: Float[np.ndarray, "2 600 600"] = ga.pixel_coords(s2)

# proposed (A): (x_min, y_min, x_max, y_max) from T and (H, W)
bounds: tuple[float, float, float, float] = grid.bounds
```

Today `GeoTensor` gives the same properties, but `pixel_coords` lives in `georeader.griddata.meshgrid` and goes through rasterio.
The proposal keeps the maths and drops rasterio, so the same lines run on JAX data in Stage 4.

### Stage 2: valid pixels and band selection (geoarray)

**TL;DR.** Decide which pixels hold data, and pick bands by name without losing their metadata.

**Problem.**
Let φ be the fill value, possibly NaN.
A pixel (i, j) is valid when every band is finite and none equals the fill:

$$
v_{ij} = \bigwedge_{k=1}^{C} \left[ \mathrm{isfinite}(x_{kij}) \wedge x_{kij} \neq \phi \right]
$$

When φ is NaN the second term is already covered by the first.
georeader's `validmask()` computes `values != fill`, and NaN ≠ NaN is always true, so a NaN-filled raster reports every pixel valid.
geoops already works around this in `geoops._src.valid`; the proposal moves the correct rule into the type.

```python
import geoarray as ga
import numpy as np
from jaxtyping import Bool, Float

# proposed (A): attrs are frozen; band_names is a tuple, so the carrier hashes
s2 = s2.with_attrs(band_names=("B11", "B12"))

# proposed (A): v_ij = ∧_k [isfinite(x_kij) ∧ x_kij ≠ φ]
valid: Bool[np.ndarray, "600 600"] = s2.valid_mask

# proposed (B): band selection slices per-band attrs too (decision B2)
b12: ga.GeoArray[Float[np.ndarray, "1 600 600"]] = s2[1:2]

# proposed (B): a spatial slice shifts the origin: c′ = c + a·j₀, f′ = f + e·i₀
centre: ga.GeoArray[Float[np.ndarray, "2 300 300"]] = s2[..., 150:450, 150:450]
```

Band selection keeping `band_names == ("B12",)` is new.
Today `isel` passes `attrs` through unchanged, and `wrap_like` drops per-band keys later only if the band count changed.

### Stage 3: one grid for both sensors (geoarray → geoops.functional)

**TL;DR.** Put two orthorectified EMIT layers onto the Sentinel-2 grid of Stage 1.

**Problem.**
Let $z \in \mathbb{R}^{2 \times H_s \times W_s}$ hold two EMIT layers on its EPSG:4326 grid (T_s, crs_s).
Layer 0 is a simulated B12: EMIT radiance integrated over Sentinel-2's B12 spectral response.
Layer 1 is the matched-filter enhancement ΔX.
Let (T_d, crs_d, (H, W)) be the destination grid from Stage 1, and π the map from crs_d to crs_s.
Each destination pixel (i, j) is pulled back to fractional source coordinates

$$
u_{ij} = T_s^{-1}\left( \pi\left( T_d (j + 1/2, i + 1/2) \right) \right) - (1/2, 1/2)
$$

and the warped value is the bilinear blend of the four source pixels around $u_{ij}$:

$$
\tilde{z}_{ij} = \sum_{(k, l) \in N(u_{ij})} w_{kl}(u_{ij}) z_{kl}
$$

The weights $w_{kl}$ are non-negative and sum to one.

The GLT step that produced z is a gather, not an affine map, so it runs in geoops before this stage (see limitation L7).

```python
import geoarray as ga
import geoops.functional as F
import jax.numpy as jnp
from jaxtyping import Array, Float

# proposed (C): move data and transform to the device; requires jax_enable_x64
s2_jax: ga.GeoArray[Float[Array, "2 600 600"]] = s2.to_jax()

# proposed (C): the same for the two orthorectified EMIT layers
z_jax: ga.GeoArray[Float[Array, "2 Hs Ws"]] = z.to_jax()

# proposed (E): u_ij = T_s⁻¹(π(T_d(j + ½, i + ½))) − ½, then bilinear weights
z_on_s2: ga.GeoArray[Float[Array, "2 600 600"]] = F.warp(z_jax, grid, order=1)
```

`F.warp` dispatches on `z_jax.xp`: SciPy's `map_coordinates` for NumPy data, `jax.scipy.ndimage.map_coordinates` for JAX data.
The CRS map π uses pyproj on the NumPy path and a closed-form JAX kernel on the JAX path (EPSG:4326 to UTM here).

### Stage 4: sub-pixel co-registration (geoarray leaf transform → jax.grad)

**TL;DR.** Find the small shift that best aligns EMIT with Sentinel-2, by matching EMIT's simulated B12 to the real B12 and differentiating through the transform.

**Problem.**
Let $r \in \mathbb{R}^{H \times W}$ be Sentinel-2's B12 on the Stage 1 grid, and m the warped simulated B12 from Stage 3 (layer 0).
The two differ in calibration and resolution, so both are standardised to zero mean and unit variance; write r̃ and m̃.
Let δ ∈ ℝ² shift the moving image's origin, so T_m(δ) = (a, b, c + δ_x, d, e, f + δ_y).
Let Ω be the interior pixels, away from the edges.
The shift estimate minimises the squared misfit:

$$
\hat{\delta} = \arg\min_{\delta \in \mathbb{R}^2} \sum_{(i, j) \in \Omega} \left( \tilde{m}_{ij}(\delta) - \tilde{r}_{ij} \right)^2
$$

$$
\tilde{m}(\delta) = \mathrm{std}\left( \mathcal{W}(m; T_m(\delta)) \right)
$$

Here 𝒲 is the Stage 3 warp, which is differentiable in δ because the transform is a JAX leaf, and std(·) subtracts the mean and divides by the standard deviation.
The bilinear loss is piecewise quadratic in δ, with kinks at whole-pixel shifts, so start within one pixel of the answer.
The same δ̂ then moves layer 1, the enhancement ΔX.

```python
import geoarray as ga
import geoops.functional as F
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


# today (jax): std(d) = (d − mean d) / std d
def standardise(d: Float[Array, "H W"]) -> Float[Array, "H W"]:
    centred = d - jnp.mean(d)
    return centred / jnp.std(centred)


# proposed (B): r̃ = std(B12) on the interior Ω
b12 = s2_jax.data[1, 8:-8, 8:-8]
ref = standardise(b12)

# proposed (B): m, the simulated B12 on the Sentinel-2 grid
sim_b12 = z_on_s2[0:1]


# proposed (A): T is a float64 leaf, so ∂ misfit / ∂δ exists
def misfit(delta: Float[Array, "2"]) -> Float[Array, ""]:
    # T_m(δ) = (a, b, c + δx, d, e, f + δy)
    shifted = sim_b12.shift_origin(delta)
    # 𝒲(m; T_m(δ)) back onto the reference grid
    moved = F.warp(shifted, grid, order=1)
    # m̃(δ) on the interior Ω
    moved_std = standardise(moved.data[0, 8:-8, 8:-8])
    # ∑_Ω (m̃ − r̃)²
    return jnp.sum((moved_std - ref) ** 2)


# today (jax): ∇δ misfit and ∇²δ misfit
grad = jax.grad(misfit)
hess = jax.hessian(misfit)

# today (jax): Newton steps, δ ← δ − (∇²misfit)⁻¹ ∇misfit
delta = jnp.zeros(2)
for _ in range(20):
    step = jnp.linalg.solve(hess(delta), grad(delta))
    delta = delta - step

# proposed (B): T_m(δ̂) applied to the enhancement layer
enhancement = z_on_s2[1:2]
shifted_final = enhancement.shift_origin(delta)

# proposed (E): ΔX aligned on the Sentinel-2 grid
aligned = F.warp(shifted_final, grid, order=1)
```

On a synthetic 64 × 64 scene with a 20 m grid and a true 7.3 m shift along x, the prototype recovered 7.296 m in 20 Newton steps, an error of 0.0002 pixel.
That is the payoff of keeping the transform a leaf: the same code also batches scenes with different origins without recompiling.

### Stage 5: patches for the detector (geopatcher → geoarray.crop)

**TL;DR.** Cut the co-registered chip into overlapping 64 × 64 patches in one compiled call.

**Problem.**
With patch size h = w = 64 and stride s = 32 on a 600 × 600 chip, the patch origins are $o_{mn} = (s m, s n)$ for $m, n \in \{0, 1, \ldots, 16\}$, which gives P = 17² = 289 patches.
Patch p is x[:, o_p : o_p + (h, w)], and its transform shifts the origin by o_p:

$$
c_p = c + a o_{p,2} + b o_{p,1}
$$

$$
f_p = f + d o_{p,2} + e o_{p,1}
$$

Offsets vary per patch and are traced; the size is static, which is what `lax.dynamic_slice` needs.

```python
import geoarray as ga
import geopatcher as gp
import jax
from jaxtyping import Array, Float, Int

# proposed (B): x = [B11, B12, ΔX] on one grid, C = 3
stack: ga.GeoArray[Float[Array, "3 600 600"]] = ga.concatenate(
    [s2_jax, aligned], axis=-3
)

# proposed (D): offsets planned on the host from the grid alone
offsets: Int[Array, "289 2"] = gp.plan(stack.grid, size=(64, 64), stride=32)


# proposed (B): x_p = crop(x, o_p, (h, w))
def crop_one(offset: Int[Array, "2"]) -> ga.GeoArray[Float[Array, "3 64 64"]]:
    return ga.crop(stack, offset, size=(64, 64))


# today (jax): one compile for all P patches
patches: ga.GeoArray[Float[Array, "289 3 64 64"]] = jax.vmap(crop_one)(offsets)
```

Each patch keeps its own transform, so a detection at patch pixel (i, j) maps straight back to map coordinates.
The prototype ran the same `vmap(crop)` pattern with three offsets in one trace and got the expected origins.

## Current-state audit

Five overlaps need action, and a sixth is healthy for NumPy but blocks JAX.
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
  geoarray is a fresh implementation of the behaviour, not a copy of `geotensor.py`; georeader stays importable behind an extra.
- **JAX is barely in geotoolz today.** Only `geopatcher._src.jax` (`batch_split`) and one training integration import it, both optionally.
- **What geoops reads off the carrier.** `.transform` 82 times, `.crs` 82, `.attrs` 41, `.values` 31, `.bounds` 29, `.fill_value_default` 25, `stack` and `concatenate` 16 each, `resize` and `pad` 10 each, `read_from_window` 9, `isel` 8, `res` 7.
  These counts set the minimum API.
- **The carrier contract already says what the type must do.** Dims are `("time", "band", "y", "x")[-ndim:]`, so a carrier is 2-D to 4-D with the band axis at −3.
  Outputs get a fresh `attrs` whose per-band keys match the band count, and `wrap_like` rewraps through a duck-typed `array_as_geotensor` check.
- **georeader's default fill is 0.** A float raster read without a fill therefore treats real zeros as nodata.
- **The xarray bridge stores CRS differently.** georeader's `toDataArray` writes `attrs["crs"]`, while xrtoolz uses rioxarray's `spatial_ref` coordinate.
  The [labeled-lane page](./02_labeled_lane.md) settles this.

## Workstream A: the GeoArray type

`GeoArray` is a frozen dataclass that composes an array, never subclasses one, and registers as a JAX pytree when JAX is importable.

| Object | Lives in | Name | Kind |
| --- | --- | --- | --- |
| Carrier | `geoarray._src.array` | `GeoArray[A]` | frozen dataclass, pytree |
| Pixel grid | `geoarray._src.grid` | `Grid` | frozen dataclass, hashable |
| Pixel window | `geoarray._src.grid` | `Window` | frozen dataclass of ints |
| Metadata | `geoarray._src.attrs` | `FrozenAttrs` | hashable `Mapping` |
| Backend | `geoarray._src.backend` | `xp`, `to_jax`, `to_numpy` | functions |

Every field is either a leaf, which JAX traces, or static, which JAX hashes into the compiled function's cache key.
Changing a static field recompiles; changing a leaf does not.

| Field | Type | Pytree role | Changing it |
| --- | --- | --- | --- |
| `data` | `np.ndarray` or `jax.Array`, 2-D to 4-D | leaf | free (same shape and dtype) |
| `transform` | `(6,)` float64 array, (a, b, c, d, e, f) | leaf | free; differentiable |
| `crs` | `pyproj.CRS` | static | recompiles |
| `fill_value` | `float`, `int`, `bool` or `None` | static | recompiles |
| `attrs` | `FrozenAttrs` | static | recompiles |

```python
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import pyproj

A = TypeVar("A")


# proposed (A): composition, frozen, generic over the array type
@dataclass(frozen=True, eq=False)
class GeoArray(Generic[A]):
    # (..., H, W): dims are ("time", "band", "y", "x")[-ndim:]
    data: A
    # (6,) float64, (a, b, c, d, e, f): (x, y) = (a·col + b·row + c, d·col + e·row + f)
    transform: A
    # parsed once on construction; strings and EPSG ints are coerced
    crs: pyproj.CRS
    # nodata marker; NaN allowed; None means no fill
    fill_value: float | int | bool | None = None
    # hashable metadata: band_names, wavelengths, sensor, …
    attrs: "FrozenAttrs" = field(default_factory=lambda: FrozenAttrs())
```

### A1. Compose, do not subclass

JAX operations always return a plain `jax.Array`, so a subclass would lose its metadata on the first `jnp` call.
Composition costs one thing: `g + 1` no longer works (decision B1).
In return, the same class holds both backends, and `eq=False` avoids the ambiguous truth value that elementwise `==` would give a dataclass.
Equality of carriers is `g.identical(h)`, and equality of grids is `g.grid == h.grid`.

### A2. The transform is a float64 leaf

A leaf transform lets `jax.grad` reach the georeferencing (Stage 4).
It also lets `vmap` batch patches whose origins differ, and lets `jit` reuse one compiled function across scenes.
The prototype confirmed both: a new transform did not retrace, while a new CRS or shape did.

float64 is required because map coordinates are large.
In float32 the spacing between representable numbers is 0.25 m at a northing of 3.55 × 10⁶ m, 0.5 m at 4.65 × 10⁶ m, and 7.6 × 10⁻⁶° (≈ 0.7 m at 32° N) at longitude −103.75°.
So `to_jax` checks `jax.config.jax_enable_x64` and raises with a message if it is off, rather than letting JAX silently round the transform to float32.
`data` keeps its own dtype, normally float32.

Recommended: require x64 for JAX carriers.
Alternative: store a static float64 origin (c₀, f₀) and a float32 leaf offset; no global flag, but every new origin recompiles.

### A3. Static, frozen attrs

geoops resolves bands by name through `attrs["band_names"]`, so the carrier needs metadata.
Static attrs must be hashable, so `FrozenAttrs` enforces three rules:

1. **Lists become tuples.** `["B11", "B12"]` is stored as `("B11", "B12")`.
2. **1-D numeric arrays become tuples of Python floats.** EMIT's 285 wavelengths fit; building and hashing them took about 21 µs in the prototype.
3. **Anything else unhashable raises.** That covers dicts, 2-D arrays and live objects such as open file handles.

Updates return a new carrier: `g.with_attrs(sensor="EMIT")` and `g.drop_attrs("id")`.
Scene-specific keys such as acquisition time or scene ID recompile per scene; limitation L2 covers the fix.

### A4. No fill by default

`fill_value=None` means "no nodata marker"; `valid_mask` then tests finiteness only.
Readers set the fill explicitly: 0 for Sentinel-2 L1C digital numbers, NaN after conversion to reflectance.
The georeader adapter carries georeader's value across unchanged, so converting an existing `GeoTensor` changes nothing.

### A5. Construction and unflattening

`__post_init__` validates: 2-D to 4-D data, a `(6,)` transform, a parseable CRS, and attrs that freeze.
The pytree `unflatten` bypasses `__post_init__`, because JAX unflattens with placeholder leaves during tracing, and a shape check would fail on them.
The pytree is registered at import time inside a `try: import jax`, so a NumPy-only install never imports JAX.

```python
# proposed (A): flatten splits leaves from static aux data
def _flatten(g: GeoArray) -> tuple[tuple[Any, Any], tuple[Any, Any, Any]]:
    leaves = (g.data, g.transform)
    aux = (g.crs, g.fill_value, g.attrs)
    return leaves, aux


# proposed (A): unflatten never validates; placeholders must pass through
def _unflatten(aux: tuple[Any, Any, Any], leaves: tuple[Any, Any]) -> GeoArray:
    g = object.__new__(GeoArray)
    object.__setattr__(g, "data", leaves[0])
    object.__setattr__(g, "transform", leaves[1])
    object.__setattr__(g, "crs", aux[0])
    object.__setattr__(g, "fill_value", aux[1])
    object.__setattr__(g, "attrs", aux[2])
    return g
```

### A6. The Grid value type

`Grid` is the transform, CRS and spatial shape without data.
It is hashable, so it can be a warp target, an operator's static config or a dictionary key.
geocatalog's `GeoSlice` and geopatcher's domains can both produce one, so there is one "where are these pixels" object across the stack.

```python
from dataclasses import dataclass

import pyproj


# proposed (A): no data, so it is cheap, hashable and JSON-serialisable
@dataclass(frozen=True)
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

## Workstream B: geometry primitives

geoarray owns every operation that re-indexes or re-describes pixels without computing new values, and each one is written once against the array namespace `xp`.
The test for membership: if the output contains only input values, fill values or coordinates, the function belongs here.

| Symbol | Destination | Reason |
| --- | --- | --- |
| `GeoTensor.values` | `GeoArray.data` | composition (A1) |
| `.res`, `.bounds`, `.height`, `.width`, `.count`, `.dims` | `GeoArray` properties via `Grid` | derived from T and shape |
| `.transform` (rasterio `Affine`) | `.transform` array; `.grid.affine` for an `affine.Affine` | leaf for JAX, `Affine` on the host |
| `__getitem__`, `isel` | `GeoArray.__getitem__`, `GeoArray.isel` | same semantics, plus per-band attrs (B2) |
| `read_from_window(window, boundless)` | `geoarray.read_window` | crop with fill padding |
| `write_from_window` (in place) | `geoarray.write_window`, returns a new carrier | JAX arrays are immutable |
| `pad` with `mode="constant"` | `geoarray.pad` | fills with `fill_value` |
| `pad` with other modes, `resize` | `geoops.functional` | computes new values |
| `meshgrid()` in the native CRS | `geoarray.pixel_coords` | coordinates only |
| `meshgrid(dst_crs)` | `geoops.functional.to_crs` | needs CRS maths |
| `validmask`, `invalidmask` | `GeoArray.valid_mask` | NaN-aware rule from `geoops._src.valid` |
| `same_extent` | `g.grid == h.grid`, `Grid.isclose` | exact by default, as geoops already requires |
| `stack`, `concatenate`, `squeeze`, `expand_dims`, `transpose` | `geoarray` functions and methods | non-spatial axes only |
| `footprint` | `geoarray.footprint` | NumPy only, returns a shapely polygon |
| `valid_footprint` | `geoops` | output size depends on the data |
| `clip`, `astype` | `g.map(...)`, `g.astype(...)` | elementwise, geometry unchanged |
| `window_from_bounds`, `round_outer_window` | `Grid.window` | geopatcher stops importing georeader |
| `to_json`, `from_json` | `Grid.get_config`, `GeoArray.to_json` | strict JSON round trip |
| `load_file`, `load_bytes` | readers and the georeader adapter | I/O stays out of the type |

### B1. Arithmetic through map and combine

`GeoTensor` supports `gt + 1` and `gt1 - gt2` through NumPy's ufunc protocol, which JAX arrays do not offer to wrappers.
geoarray replaces it with two explicit forms.
`g.map(f)` applies f to the data and keeps the geometry, and raises if f changes the spatial shape.
`geoarray.combine(f, g1, g2, …)` checks that every grid is equal, then applies f to the data arrays.
The grid check runs on concrete values; under `jit` the transforms are traced, so `combine` compares the static parts (CRS and shape) and leaves transform equality to the caller.

```python
import geoarray as ga
import jax.numpy as jnp
from jaxtyping import Array, Float

# proposed (B): g ↦ log(1 + g), geometry unchanged
logged = s2_jax.map(jnp.log1p)

# proposed (B): the two bands as separate carriers on one grid
b12 = s2_jax[1:2]
b11 = s2_jax[0:1]

# proposed (B): r = B12 / B11; raises if the grids differ
ratio: ga.GeoArray[Float[Array, "1 600 600"]] = ga.combine(jnp.divide, b12, b11)
```

Recommended: no dunders.
Alternative: dunders that check grids eagerly and skip the transform check under `jit`.

### B2. Slicing updates the transform and the per-band attrs

Slicing rows i₀ : i₁ with step s_r and columns j₀ : j₁ with step s_c changes the transform by a shift and a scale:

$$
\begin{aligned} a' &= a s_c & b' &= b s_r & c' &= c + a j_0 + b i_0 \\ d' &= d s_c & e' &= e s_r & f' &= f + d j_0 + e i_0 \end{aligned}
$$

Spatial axes take slices only, with positive steps; leading axes (time, band) take anything NumPy indexing allows.
`g.shift_origin(δ)` is the traced counterpart for a sub-pixel move: it adds δ to (c, f), and gradients flow through it (Stage 4).
Selecting bands also selects the matching entries of every per-band key in `geoops._src.bands.PER_BAND_KEYS` (`band_names`, `descriptions`, `bands`, `band_descriptions`, `wavelengths`, `wavelengths_nm`).
That list moves into geoarray, and geoops imports it.

### B3. Traced crops

Plain slicing needs concrete offsets.
`geoarray.crop(g, offset, size)` takes a traced `(row, col)` offset and a static `(h, w)` size, uses `lax.dynamic_slice` on JAX data and slicing on NumPy data, and shifts the transform as in B2.
`lax.dynamic_slice` clamps an out-of-range offset to stay inside the array instead of raising, so `crop` documents that and `read_window` handles out-of-bounds windows with padding.

```python
import geoarray as ga
from jaxtyping import Array, Float, Int


# proposed (B): x_p = x[:, o_p : o_p + (h, w)], T_p = T · shift(o_p)
def crop(
    g: ga.GeoArray[Float[Array, "*lead H W"]],
    offset: Int[Array, "2"],
    size: tuple[int, int],
) -> ga.GeoArray[Float[Array, "*lead h w"]]: ...
```

### B4. Windows

`Window(row_off, col_off, height, width)` is a frozen dataclass of Python ints, equivalent to a rasterio `Window` without importing rasterio.
`Grid.window(bounds)` rounds outward, as `round_outer_window` does today.
`Grid.bounds_of(window)` is its inverse on grid-aligned bounds.
`read_window(g, window, boundless=True)` pads the outside with the fill value, and requires a fill value when the window leaves the array.

## Workstream C: backends and interop

Every move between backends or libraries is an explicit call, and each one is a function in geoarray.

| Call | From → to | Notes |
| --- | --- | --- |
| `g.to_jax(device=None)` | NumPy → JAX | `jax.device_put` for data and transform; raises if x64 is off (A2) |
| `g.to_numpy()` | JAX → NumPy | host copy of data and transform |
| `np.asarray(g)` | NumPy carrier → `np.ndarray` | the data, so geoops's `np.asarray(x)` keeps working |
| `np.asarray(g)` on a JAX carrier | — | raises, naming `to_numpy` (decision C1) |
| `from_georeader(gt)`, `to_georeader(g)` | `GeoTensor` ↔ `GeoArray` | `geoarray[georeader]`; NumPy only; values, transform, CRS, fill and attrs carried over |
| `Grid.affine`, `Grid.from_affine` | `affine.Affine` ↔ `Grid` | `affine` is a small BSD-licensed base dependency, already used by geoops |
| `to_xarray`, `to_coordax` and their inverses | labeled lane | [Labeled lane](./02_labeled_lane.md) |

```python
import geoarray as ga
from geoarray.interop.georeader import from_georeader, to_georeader

# proposed (C): GeoTensor → GeoArray, zero-copy for the data
g: ga.GeoArray = from_georeader(gt)

# proposed (C): explicit device move; no implicit conversion anywhere
g_jax = g.to_jax()

# proposed (C): back to a GeoTensor for code that still needs one
gt_back = to_georeader(g_jax.to_numpy())
```

`g.xp` returns the array namespace through `array_api_compat`, so a function written against `xp` runs on both backends.
`g.backend` returns `"numpy"` or `"jax"` for dispatch tables.
Mixing backends in one call (`combine` of a NumPy and a JAX carrier) raises, following the Array API rule against mixing namespaces.

What runs where, for geoarray's own functions:

| Function | NumPy | JAX | Under `jit` |
| --- | --- | --- | --- |
| properties, `grid`, `bounds`, `res` | yes | yes | `grid` needs a concrete transform |
| `pixel_coords`, `valid_mask`, `map`, `combine` | yes | yes | yes |
| `__getitem__`, `isel` | yes | yes | concrete slices only |
| `crop` | yes | yes | traced offset, static size |
| `read_window`, `write_window`, `pad` | yes | yes | concrete window |
| `stack`, `concatenate` | yes | yes | yes |
| `footprint` | yes | no | no (shapely) |

## Workstream D: carrier migration

geoops learns both carriers first, then each reader switches, so no package breaks in between.

| Symbol | Destination | Reason |
| --- | --- | --- |
| `geoops._src.wrap.wrap_like` | dispatch on `GeoArray.with_data`, then on `array_as_geotensor` | one rewrap call for both carriers |
| `geoops._src.valid.*` | wrappers over `GeoArray.valid_mask` | one nodata rule |
| `geoops._src.geo.require_grid_match`, `grid_matches` | `Grid.__eq__` | exact equality, as today |
| `geoops._src.bands.PER_BAND_KEYS` | `geoarray._src.attrs` | slicing needs it (B2) |
| geocatalog, geocloud, geoproducts readers | return `GeoArray` | native grid, NumPy data |
| geopatcher window helpers | `Grid.window`, `geoarray.crop` | no georeader internals |

1. **Teach geoops both carriers.** `wrap_like` tries `with_data` first, so every operator accepts `GeoArray` without edits.
   A NumPy `GeoArray` implements `__array__`, so `np.asarray(x)` inside each operator keeps working.
   Parametrise geoops's test fixtures over both carriers.
2. **Switch the readers.** geocatalog, geocloud and geoproducts return `GeoArray`.
   Callers that need a `GeoTensor` call `to_georeader`.
3. **Switch geopatcher.** Replace its seven georeader import lines with `Grid.window` and `crop`.
   Its `RasterField` adapter takes either carrier.
4. **Deprecate `GeoTensor` input.** geoops warns once per operator when handed a `GeoTensor`, naming `from_georeader`.
5. **Drop georeader from the base dependencies.** It moves to extras in all five packages; import-linter enforces that base modules never import it.

Downstream, `radtrax.geo` and `plumax.geo` take `GeoTensor`s today (radtrax backlog RTX-30 and RTX-34).
They follow after step 4, using the adapter until then.

## Workstream E: JAX kernels in geoops.functional

The payoff of a JAX carrier comes from kernels that run inside `jit`, so geoops gains a JAX implementation of each core kernel next to the NumPy one.
geoarray stays free of all of them.

| Kernel | NumPy | JAX |
| --- | --- | --- |
| `warp`, order 0–1 | `scipy.ndimage.map_coordinates` | `jax.scipy.ndimage.map_coordinates` |
| `warp`, cubic | SciPy | a local cubic kernel (JAX supports order ≤ 1 only) |
| `resample_mean` | block reshape and mean | the same, with static block sizes |
| `resample_mode` | `bincount` | one-hot sum then argmax, class count K static |
| `to_crs` | pyproj, any CRS | closed form: EPSG:4326, UTM, Web Mercator, polar stereographic, sinusoidal, geostationary |
| `glt_orthorectify` | fancy indexing | `jnp.take` gather |

A kernel without a JAX version raises an error naming the missing backend; it never converts silently.

## Testing gates

Each workstream merges only when its truth tests pass.
Gate IDs are "G", then the workstream letter, then a number.

| Gate | Workstream | Passes when |
| --- | --- | --- |
| GA1 Pytree round trip | A | `unflatten(flatten(g))` is identical to g, and `jit(lambda g: g)(g)` returns equal data and transform bit for bit |
| GA2 No retrace on transform | A | one trace across 100 random transforms of the same shape, CRS, fill and attrs; a new CRS or shape adds exactly one trace |
| GA3 Attrs are hashable | A | equal attrs hash equal; lists become tuples; a dict value raises `TypeError` |
| GA4 x64 guard | A | `to_jax` raises with x64 off, and the transform is float64 with it on |
| GB1 Slice parity | B | for 1,000 random slices with steps 1 to 3, the transform equals `GeoTensor.isel`'s to 1e-9 in map units |
| GB2 Crop equals slice | B | `crop` with a traced offset equals concrete slicing exactly; `vmap` over P offsets traces once |
| GB3 Window round trip | B | `Grid.bounds_of(Grid.window(b)) == b` exactly for grid-aligned b, and contains b otherwise |
| GB4 NaN nodata | B | `valid_mask` equals `geoops._src.valid.valid_pixels` on every geoops fixture, including NaN fills, where naive `!=` fails (prototype: 15 valid of 16, versus 16) |
| GB5 Per-band attrs | B | after any band selection, every per-band key has one entry per band |
| GC1 Backend parity | C | every geoarray function gives equal results on NumPy and JAX: exactly for indexing, to 1e-12 in float64 for `pixel_coords` |
| GC2 georeader round trip | C | `from_georeader(to_georeader(g))` is identical to g, for all geoops fixtures |
| GC3 Light base install | C | `pip install geotoolz-array` into a fresh environment installs none of jax, rasterio, georeader, xarray |
| GD1 Dual carrier | D | geoops's full suite passes with fixtures parametrised over `GeoTensor` and `GeoArray` |
| GD2 No georeader in base | D | import-linter: no base module of the five packages imports georeader |
| GE1 Warp parity | E | JAX bilinear warp equals rasterio's bilinear reprojection to 1e-5 relative on interior pixels |
| GE2 CRS kernels | E | each closed-form projection equals pyproj to 1 mm over its valid domain, in float64 |
| GE3 Co-registration | E | Stage 4 recovers a known sub-pixel shift to 0.01 pixel on a synthetic scene (prototype: 0.0002 pixel) |

## Roadmap

The type first, then the geometry and interop that sit on it, then the switch, so each release has one kind of change.
Versions only bump the minor number; nothing reaches 1.0 in this plan.
Milestone, in every repo: "geoarray: one NumPy/JAX carrier for geotoolz".

```
  1   geoarray type (A)
      GeoArray, Grid, Window, FrozenAttrs, pytree
       │
       ◆ GA1 GA2 GA3 GA4
       │
  2a  geometry (B)        ┃  2b  interop (C)
      slicing, crop,      ┃      to_jax, to_numpy,
      windows, valid_mask ┃      georeader adapter
       │
       ◆ GB1–GB5 GC1–GC3   → geotoolz-array 0.1.0
       │
  3   geoops dual carrier (D, step 1)
       │
       ◆ GD1               → geotoolz-ops 0.11.0
       │
  4a  readers return      ┃  4b  geopatcher on    ┃  4c  JAX kernels
      GeoArray (D, 2)     ┃      geoarray (D, 3)  ┃      in geoops (E)
       │
       ◆ GE1 GE2 GE3       → catalog 0.6, cloud 0.5, products 0.5,
       │                     patcher 0.11, ops 0.12
  5   deprecate GeoTensor input, drop georeader (D, 4–5)
       │
       ◆ GD2               → geotoolz 0.11.0; radtrax[geo], plumax[geo] follow
```

Phase 1 is the only one that blocks everything else, and it is small: the prototype for this doc is under 200 lines.
Phases 2a and 2b run in parallel, as do 4a, 4b and 4c.

## Limitations

These come with the design; each has a workaround rather than a fix.

- **L1. Recompiles per shape, CRS, fill, dtype and attrs.** These are static, so every distinct combination compiles once.
  Fixed chip sizes and stripped attrs keep that to a handful per pipeline.
- **L2. Scene-specific attrs recompile per scene.** An acquisition time or scene ID in `attrs` makes every scene a new cache key.
  geoops's `Pipeline` strips non-geometric keys before `jit` and reattaches them after, keeping only `band_names` and the other per-band keys inside the trace.
- **L3. Eager JAX is slower than NumPy.** JAX dispatch costs tens of microseconds per operation, so a notebook calling operators one by one on JAX data loses to NumPy.
  NumPy stays the default for readers, and `Pipeline` compiles the whole chain.
- **L4. x64 is a global switch.** Turning it on makes new JAX arrays default to float64, which costs memory and speed in code that does not set dtypes.
  geoarray keeps `data` in its own dtype, but user code outside geoarray is affected.
  The A2 alternative avoids the flag at the cost of recompiles.
- **L5. No arithmetic operators.** `g + 1` becomes `g.map(lambda d: d + 1)`, and existing notebooks need edits.
  Decision B1 holds the alternative.
- **L6. Data-dependent shapes stay on the host.** `footprint`, `valid_footprint`, vectorising and connected components produce outputs whose size depends on the values, so they run on NumPy only.
- **L7. Sensor geometry is not affine.** EMIT and PRISMA in sensor geometry have no affine transform, so they are not `GeoArray`s until orthorectified.
  The GLT and its source array travel as plain arrays into `geoops.functional.glt_orthorectify`, whose output is a `GeoArray`.
- **L8. Rotated grids are carried, not processed.** The type holds any affine transform, but warp and resample kernels may require `Grid.is_axis_aligned` and raise otherwise.
- **L9. JAX's CRS coverage is a subset.** Six closed-form projections run inside `jit`; any other CRS works only on NumPy data, through pyproj.
- **L10. No in-place writes.** `write_window` returns a new carrier, as JAX requires.
  NumPy code that relied on `write_from_window` mutating a mosaic buffer must assign the result.

## Risks

- **Two carriers for longer than planned.** The dual-carrier phase could stretch while downstream repos catch up.
  Fallback: the georeader adapter is zero-copy, so `GeoTensor` users can stay on it indefinitely without blocking geotoolz releases.
- **Per-band attrs disagree between geoarray and `wrap_like`.** Slicing (B2) and rewrapping could apply different rules during phase 3.
  Fallback: both import one `PER_BAND_KEYS` from geoarray, and gate GB5 runs inside geoops's suite too.
- **x64 rejected by users.** Downstream training code may refuse a global float64 switch.
  Fallback: implement the A2 alternative behind the same `GeoArray` API, since only `to_jax` and the transform update functions change.
- **coordax or Array API churn.** `array_api_compat` and coordax are young.
  Fallback: geoarray depends only on `array_namespace`, and coordax sits behind an extra on [the labeled-lane page](./02_labeled_lane.md).
- **License contamination.** Copying georeader code into geoarray would bring LGPL terms into an MIT package.
  Fallback: write geoarray from the behaviour and the tests in this doc, and use georeader only as an oracle in tests.
- **Patch extraction clamps silently.** `lax.dynamic_slice` shifts an out-of-range offset back inside the array, which would misplace a patch without an error.
  Fallback: geopatcher's plan validates offsets on the host, and `crop` has a debug mode that checks bounds with `jax.experimental.checkify`.

## Before opening the issues

- [ ] Decisions A1–A4, B1, B2, C1 and D1 confirmed or flipped.
- [ ] Package name confirmed: `geotoolz-array`, import `geoarray`, with its own icon like the other subpackages.
- [ ] geotoolz, georeader, radtrax and plumax checked for duplicate open issues.
- [ ] Companion page [Labeled lane](./02_labeled_lane.md) reviewed, so the converters' extras exist when geoarray 0.1.0 ships.

## References

- georeader `GeoTensor`: [geotensor.py](https://github.com/spaceml-org/georeader/blob/main/georeader/geotensor.py), read at `9812be1`.
- geotoolz carrier contract: [AGENTS.md](https://github.com/jejjohnson/geotoolz/blob/main/AGENTS.md), "The two contracts", read at `48a705a`.
- Array API compatibility layer: [array-api-compat](https://data-apis.org/array-api-compat/).
- JAX pytrees and custom nodes: [Pytrees, "Custom pytree nodes"](https://docs.jax.dev/en/latest/101/pytrees.html).
- coordax `Coordinate` as a static pytree node: [coordax](https://coordax.readthedocs.io/), checked against 0.2.9.
- JAX 64-bit mode: [Double (64-bit) precision](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).
- Companion page: [Labeled lane](./02_labeled_lane.md).
