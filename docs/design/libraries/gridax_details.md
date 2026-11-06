---
title: "gridax: design details"
short_title: "gridax details"
description: "The implementation reference behind the gridax design: the geometry and vertical types, the algorithms step by step, JAX and NumPy behaviour, tests, dependencies, and the survey of the code that moves in. Everything here is about analysis grids; model and spectral grids stay in finitevolX and spectraldiffx."
date: 2026-11-05
keywords:
  - gridax
  - design
  - details
  - GeoStack
---

The implementation reference behind the [gridax design](gridax.md): the geometry and vertical types, the algorithms step by step, JAX and NumPy behaviour, tests, dependencies, and the survey of the code that moves in. Everything here is about analysis grids; model and spectral grids stay in finitevolX and spectraldiffx.

## Core types

Three geometries, five vertical coordinates and a stencil spec. Like geophysix's records they are frozen dataclasses of arrays, registered as pytrees when JAX is importable, because geoarray holds them and must run on NumPy alone (Design decision 6). Coordinates are in degrees; nothing holds a CRS.

```python
# Shapes: Ny × Nx lon/lat cells; R × C curvilinear or swath cells; K levels

import dataclasses

from jaxtyping import Array, Bool, Float


# proposed (gridax)
@dataclasses.dataclass(frozen=True)
class LonLatGeometry:
    """A rectilinear lon/lat grid, uniform or not, from its cell edges."""

    lon_edges: Float[Array, "x+1"]
    lat_edges: Float[Array, "y+1"]
    periodic: bool = False

    @classmethod
    def from_axes(cls, lon, lat, *, periodic=False) -> "LonLatGeometry":
        # edges = midpoints of centres, extended by half a step, |φ| ≤ 90°
        ...


# proposed (gridax)
@dataclasses.dataclass(frozen=True)
class CurvilinearGeometry:
    """A fixed 2-D lat/lon grid (an ORCA-like ocean grid) with its corners."""

    lat: Float[Array, "row col"]
    lon: Float[Array, "row col"]
    lat_corners: Float[Array, "row+1 col+1"]
    lon_corners: Float[Array, "row+1 col+1"]


# proposed (gridax)
@dataclasses.dataclass(frozen=True)
class SwathGeometry:
    """One scene in sensor geometry: rows along track, columns across."""

    lat: Float[Array, "row col"]
    lon: Float[Array, "row col"]
    lat_corners: Float[Array, "row+1 col+1"]
    lon_corners: Float[Array, "row+1 col+1"]
    corners_estimated: bool = True


# proposed (gridax)
@dataclasses.dataclass(frozen=True)
class Hybrid:
    """Hybrid σ–p half levels: p_{k+½} = a_{k+½} + b_{k+½} p_s."""

    a_half: Float[Array, "k+1"]
    b_half: Float[Array, "k+1"]


# proposed (gridax)
@dataclasses.dataclass(frozen=True)
class Stencil:
    """Integer offsets for derivative n at a given accuracy order."""

    derivative: int
    accuracy: int
    method: str = "central"
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `LonLatGeometry` | 1-D lon and lat edges; whether longitude wraps | edges strictly monotonic; latitudes within \[−90°, 90°\]; a periodic grid spans exactly 360° |
| `CurvilinearGeometry` | 2-D centres and corners, fixed for a model or product | corners shaped (row+1, col+1); each cell's corners wind counter-clockwise |
| `SwathGeometry` | 2-D centres and corners of one scene; whether the corners were read or estimated | as above; corners may overlap between scans (bowtie), which areas tolerate and neighbour search flags |
| `Height`, `Pressure`, `Sigma`, `Hybrid`, `Isopycnal` | the level values of one vertical coordinate, and for `Sigma` and `Hybrid` the parameters that turn p\_s into levels | strictly monotonic; the direction (`positive="up"` or `"down"`) explicit |
| `Stencil` | derivative, accuracy, method; offsets and coefficients are derived and cached | derivative ≥ 1, accuracy ≥ 1; a central stencil is built at accuracy + 1, finitediffx's rule |

Errors subclass `GridaxError(ValueError)`: `GeometryError` for bad edges or corners, `LevelError` for a non-monotonic vertical coordinate, `StencilError` for too few points.

## Types to adhere to

| Protocol | What a caller's object or array must provide | Read by |
| --- | --- | --- |
| `HasCorners` | `lat_corners`, `lon_corners` of shape (row+1, col+1) | `metrics.cell_area`, neighbour search |
| `VerticalCoordinate` | `levels(**state)` returning (K, …) values, and `positive` | `vertical.remap` |
| `EarthLike` (geophysix) | `semi_major`, `flattening` | every metric on the sphere or ellipsoid |
| arrays | NumPy or JAX arrays; gaps as NaN | all kernels |

A gap is a NaN in the field, not a mask: that is how products arrive, and the differencing adapts to it (below). Every kernel computes with the namespace of its first array (`pipekit_array.array_namespace`), so a geoarray caller on NumPy never imports JAX:

```python
# proposed (gridax)
# Aᵢⱼ on WGS84, in the namespace of lat_edges: NumPy in, NumPy out
area = gridax.metrics.cell_area(geometry, earth=geophysix.earth.WGS84)  # → (Ny, Nx)
```

## Algorithms

### Corners of a curvilinear or swath grid

With cᵢⱼ the unit vector of centre (i, j) on the sphere, an interior corner is the normalised mean of the four centres around it:

```{math}
\mathbf{v}_{i+1/2, j+1/2} = \frac{\mathbf{c}_{i,j} + \mathbf{c}_{i+1,j} + \mathbf{c}_{i,j+1} + \mathbf{c}_{i+1,j+1}}{\left| \mathbf{c}_{i,j} + \mathbf{c}_{i+1,j} + \mathbf{c}_{i,j+1} + \mathbf{c}_{i+1,j+1} \right|}
```

1. Convert centres to unit vectors (functax's sphere frames).
2. Average the four neighbours of each interior corner and normalise.
3. Extrapolate the edge corners linearly from the two nearest interior ones, v₀ = 2 v₁ − v₂, then normalise.
4. Convert back to lat/lon.

Cost O(R C). Averaging in 3-D, not in degrees, is what keeps cells that straddle the dateline or a pole right. A reader that has corners (TROPOMI's bounds) skips this and sets `corners_estimated=False`; the test of the estimate compares the two on such a product (Design decision 8). Pitfall: in VIIRS and MODIS scans the edge rows of consecutive scans overlap (the bowtie), so estimated corners there are meaningful only after radiomax's bowtie geometry.

### Area of a corner polygon

Each cell is split into two triangles with great-circle edges. For unit vectors a, b, c the spherical excess of a triangle is (Van Oosterom and Strackee)

```{math}
\tan\frac{E}{2} = \frac{\mathbf{a} \cdot (\mathbf{b} \times \mathbf{c})}{1 + \mathbf{a} \cdot \mathbf{b} + \mathbf{b} \cdot \mathbf{c} + \mathbf{c} \cdot \mathbf{a}}
```

1. On the ellipsoid, map each corner's latitude to its authalic latitude (geophysix), so that a sphere of authalic radius Rₚ preserves area.
2. E for the two triangles, with `atan2` of numerator and denominator.
3. A = Rₚ² (E₁ + E₂).

Cost O(R C). For a rectilinear lon/lat grid, geophysix's closed form replaces this. Pitfalls: corners must wind consistently (the sign of E is the check); the great-circle edges differ from the rhumb lines of a lon/lat cell, an error below 10⁻⁶ relative for cells under 1°.

### Stencil coefficients on uneven spacing

For derivative n at a point with neighbours at signed distances δ₀, …, δ\_{K−1}, the coefficients cₖ make the stencil exact for polynomials of degree K − 1 (the Taylor conditions):

```{math}
\sum_{k=0}^{K-1} c_k \frac{\delta_k^m}{m!} = \delta_{mn} \text{ for } m = 0, \dots, K - 1
```

1. Offsets from finitediffx's rules: forward 0, …, n + p − 1; backward its mirror; central ±⌊(n + p)/2⌋, built at accuracy p + 1.
2. Uniform spacing: solve once on the host in float64 and cache (geoops' `difference_coeffs`).
3. Uneven spacing: scale each row by its widest distance and solve the K × K system per point on the array namespace (geoops' `nonuniform_coeffs`), or use Fornberg's recursion, O(K²) per point and better conditioned for wide stencils.
4. Edges: the stencil shifts one-sided and keeps its accuracy.
5. Gaps: run candidate stencils widest first and keep, per point, the first that returns a finite value (xrops' `_difference_adaptive`), so a NaN coast costs one stencil width of accuracy rather than a strip of data.

Cost O(K) per point for the uniform case; O(K³) per point for the per-point solve, which is negligible for K ≤ 9.

### Divergence and curl on the sphere and on curvilinear grids

On a lon/lat grid, with u, v eastward and northward components:

```{math}
\nabla \cdot \mathbf{u} = \frac{1}{R \cos\varphi} \left( \frac{\partial u}{\partial \lambda} + \frac{\partial (v \cos\varphi)}{\partial \varphi} \right)
```

```{math}
\zeta = \frac{1}{R \cos\varphi} \left( \frac{\partial v}{\partial \lambda} - \frac{\partial (u \cos\varphi)}{\partial \varphi} \right)
```

On a curvilinear grid the index derivatives map to physical ones through the Jacobian J of (i, j) → (x, y) in a local tangent plane: ∂/∂x = (y\_j ∂\_i − y\_i ∂\_j) / J and ∂/∂y = (x\_i ∂\_j − x\_j ∂\_i) / J. Pitfalls: the 1 / cos φ factor at the poles, where rows at |φ| = 90° return NaN rather than infinity; and the flux form ∂(v cos φ)/∂φ, which is what the integral of the divergence over the sphere needs to be zero.

### Vertical remapping

**Linear.** Per column, for each target level z\*: find the bracketing source levels zₖ ≤ z\* < zₖ₊₁, then q(z\*) = qₖ + (qₖ₊₁ − qₖ)(z\* − zₖ) / (zₖ₊₁ − zₖ). In ln p rather than p for pressure levels. Targets outside the column are NaN unless an extrapolation is asked for.

**Conservative.** With source layer means qₖ on edges zₖ₊½, the cumulative integral Q is piecewise linear and exact at the edges:

```{math}
Q(z_{k+1/2}) = \sum_{j \le k} q_j \Delta z_j
```

1. Q at the source edges, a cumulative sum along the level axis.
2. Interpolate Q linearly to the target edges.
3. Target means are differences of Q divided by the target thickness.

The column total is kept exactly, whatever the two grids. Cost O(K) per column for both methods, plus a search for the bracketing levels.

## JAX compatibility

| Function | jit | vmap | grad | Notes |
| --- | --- | --- | --- | --- |
| `LonLatGeometry.from_axes` | yes | yes | n/a | Edges from centres; `periodic` static |
| `cell_area` (lon/lat) | yes | yes | yes | Closed form from geophysix; grad in the edges |
| `estimate_corners` | yes | yes | yes | Pure array maths on unit vectors |
| `polygon_area` | yes | yes | yes | `atan2` keeps the gradient finite for small cells |
| `stencil_coeffs` (uniform) | static | n/a | no | Host float64 solve, cached; the result is a constant |
| `nonuniform_coeffs` | yes | yes | yes | Per-point solve on the namespace; spacing traced |
| `difference` | yes | yes | yes | Stencil spec, `axis`, `periodic` static |
| `difference` (NaN-adaptive) | yes | yes | yes | Every candidate runs, then a `where` picks; no branch on data |
| `divergence`, `curl` | yes | yes | yes | NaN rows at the poles, not infinities |
| `remap_linear` | yes | yes | yes | `searchsorted` per column; gradient zero in the level positions at the brackets |
| `remap_conservative` | yes | yes | yes | Cumulative sum, then linear interpolation of Q |
| `Hybrid.pressure`, `Sigma.pressure` | yes | yes | yes | p = a + b pₛ, traced in pₛ |

**Static or traced.** Static: every shape, the stencil spec (derivative, accuracy, method), `axis`, `periodic`, the number of levels, the remap method and whether to extrapolate. Traced: coordinates, edges, corners, spacing, surface pressure, field values. A new accuracy or a new axis is a recompile, a new grid is not.

**Dtypes.** Outputs follow the field's dtype. Uniform stencil coefficients are solved in float64 on the host and cast once; per-point solves run in the input's dtype, so float32 grids should keep stencils at accuracy ≤ 4, where the scaled Vandermonde system stays well conditioned. Corner averaging and spherical excess are done in the coordinates' dtype; for cells under about 1 km in float32, area comes from the closed form rather than from the excess.

**The xp rule.** Every function resolves its namespace with `pipekit_array.array_namespace` from its array arguments and never imports `jax` at module level, so gridax installs and runs on NumPy alone, as geoarray needs. JAX-only code paths (`lax.scan` instead of a cumulative sum, `custom_jvp` on `polygon_area`) sit behind the namespace check, not behind a separate module.

## Testing

- **Stencils.** Exact on polynomials up to the order they claim, on uniform, stretched and random spacings (hypothesis). Convergence: halving the spacing divides the error of a smooth test function by about 2ᵖ, checked as a band around the slope of a least-squares fit over four resolutions. One-sided edges keep the order. The NaN-adaptive path loses at most one stencil width next to a gap.
- **Areas.** The cells of a global lon/lat grid sum to 4πR² on the sphere and to the ellipsoid's area (geophysix) on WGS84, to 10⁻¹² relative. Corner-polygon areas match pyproj's `Geod.polygon_area_perimeter` on random quads, to 10⁻⁶ relative.
- **Corners.** Estimated corners of a stored TROPOMI granule match its bounds to within a tenth of a pixel inside the swath; a grid that straddles the dateline and one that touches a pole give the same areas as their rotated copies.
- **Vector calculus.** Solid-body rotation u = U cos φ has zero divergence and vorticity 2U sin φ / R; the divergence of any field integrates to zero over the sphere (the flux form); on a rotated curvilinear grid the results match the lon/lat ones.
- **Remap.** Conservative remap keeps every column total to round-off; linear remap reproduces linear profiles exactly; a remap to the source levels is the identity.
- **Against today's code.** Stored fixtures generated once from geoops' `difference` and xrops' `grid_metrics_from_coords` and `remap_axis_columns` (a script in `scripts/`), not imports of them, so gridax tests never depend on the layers that will depend on gridax.
- **Backends.** The fast tier runs in a NumPy-only job with JAX absent; a second job checks `jit`, `vmap` and `grad` against the NumPy results for every row of the table above.

## External dependencies

| Dependency | Why | Kind |
| --- | --- | --- |
| NumPy | Arrays and the host coefficient solves | required |
| pipekit-array | `array_namespace` | required |
| geophysix | Ellipsoid, authalic latitude, closed-form cell area | required |
| JAX | `jit`, `vmap`, `grad`, `lax.scan` paths | optional (`[jax]`) |
| pyproj | Reference geodesic areas in tests | test only |
| hypothesis | Property tests on spacings and grids | test only |

No xarray, no geoarray: geoarray and xrops import gridax, not the other way round.

## Where it stands

Read at geotoolz d2f53f4, xrtoolz b3855d7, finitevolX 82090a3, spectraldiffx d68f8f6, somax aea3a03 and sesmax 314de57.

| Today | Where | In gridax |
| --- | --- | --- |
| `difference_offsets` (finitediffx's offset rules), `difference_coeffs` (host float64 Vandermonde solve, `lru_cache`, rounded at 10⁻¹²), `nonuniform_coeffs` (per-row scaled Taylor solve on the namespace) | geoops `geom/_src/stencils.py` | `Stencil` and its coefficients, unchanged in method |
| `difference(f, axis, derivative, accuracy, method, step, coords)`, with bounded and periodic variants, dispatched by `array_namespace` | geoops `geom/_src/array.py` | `difference`; geoops keeps the operator wrapper |
| `_difference_adaptive`: "the widest stencil that fits on finite data" | xrops `grad/_src/cartesian.py` | The NaN-adaptive path of `difference` |
| Cartesian, rectilinear and spherical gradients, delegating to geoops | xrops `grad/_src/` | `divergence`, `curl`, spherical gradient; xrops keeps the xarray wrappers |
| `grid_metrics_from_coords(ds, lat, lon, depth, sphere, radius=EARTH_RADIUS)`: dx = R cos φ Δλ, area = dx dy | xrops `grad/_src/grid_metrics.py` | `cell_area` and the metrics, with the exact band area instead of the midpoint rule and geophysix's radius instead of 6,371,000 m |
| `remap_axis`, `remap_axis_columns` (NumPy `np.interp`) and the presets `ToHeight`, `ToPressureLevels`, `ToSigma`, `ToIsopycnal` | xrops `transforms/_src/_coord_remap_kernels.py` | `remap_linear` on any namespace, plus `remap_conservative`; the presets stay in xrops |
| `SwathGrid(shape, crs)`, `CurvilinearGrid(lat, lon, crs)`, `Vertical(kind, units, positive, datum)`, `VerticalAxis` | geoarray `_src/coords.py` | Stay as labels; gridax adds the corners, areas and metrics they lack |

Per library:

- **geoops.** The stencil engine is already namespace-generic and is the code gridax lifts. The `geom` operators keep their names and become thin wrappers over gridax.
- **xrops.** The midpoint-rule area overstates every cell by a factor of about 1 + Δφ²/24 (10⁻⁵ at 1°) and gives no area to a cell centred on a pole; the remap kernels are NumPy-only and so not differentiable; both are fixed by the move.
- **geoarray.** Its grid classes name a geometry but compute nothing from it; they stay that way, and gain metrics by calling gridax.
- **finitevolX, spectraldiffx, somax, sesmax.** Staggered C-grid operators and spectral derivatives on their own model grids; nothing moves. They can later read cell areas and metric terms from gridax instead of building them from `DEG2M`.

## Sources

- Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily spaced grids. *Mathematics of Computation* 51, 699–706.
- Van Oosterom, A. and Strackee, J. (1983). The solid angle of a plane triangle. *IEEE Transactions on Biomedical Engineering* BME-30, 125–126.
- Snyder, J. P. (1987). *Map Projections: A Working Manual*. USGS Professional Paper 1395 (authalic latitude).
- Lin, S.-J. (2004). A vertically Lagrangian finite-volume dynamical core for global models. *Monthly Weather Review* 132, 2293–2307 (conservative vertical remapping).
