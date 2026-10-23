---
title: "GeoStack operators: geoops and xrops on one kernel model"
subtitle: "Design · needs × carrier families, kernels written once, package tiers"
short_title: Operators
description: "How geoops and xrops are organised by what an operation needs (shapes, dim names, geometry) and by carrier family (geoarray, xarray), with L1 kernels wrapped once as KernelOps that run in both, and the package tiers pipekit-array → geoops → xrops."
date: 2026-10-23
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

- [geoarray](./geoarray.md) defines the data: geoarray carriers, coordinates, readable types, static versus leaf.
- [pipekit kernels](./pipekit_kernels.md) defines the machinery this page reuses: the named-array adapter, the single `Signature`, the dim-level functions, `KernelSpec` and `KernelOp`, `JaxOperator`.

This page adds no machinery of its own; where it needs some, the need goes to pipekit.

**Decided 22 Oct 2026:** xrops sits one tier above geoops ([package tiers](#design-tiers)); this settles OP-O1, OP-O4 and OP-O7.
The levels (L1 primitive, L2 operator, L3 algorithm) are defined once, in [the three levels](#design-levels); this page places operations within them.
Still open: the [case studies](./case_studies.md) findings CS-F1 to CS-F17.

## The grid

Two independent questions place every operation: what it needs to know (a row) and how much data it runs on (the carrier family, a column).

| Needs | geoarray carriers (one item, eager or `jit`) | xarray carriers (many items, by value, lazy) |
| --- | --- | --- |
| **geometry** | geoops: reproject, rasterize, sample H(x), plume detection, patch/stitch | xrops: regrid a cube, mosaic many files |
| **dim names** | pipekit-array dim-level functions; `KernelOp`s over a named dim | the same functions and `KernelOp`s; native value operations (`resample`, `groupby`, ranges); xrops analysis |
| **shapes only** | kernels on `.data` | kernels via `apply_ufunc` |

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

**The consequence that matters most:** an operation that needs only dim names, built as a [KernelOp](#design-kernelop), is not column-specific.
`NDVI(dim="band")` runs on a `GeoImage` eagerly, on a JAX carrier under `jit`, and on a lazy cube through `apply_ufunc`, from one definition.
Only geometry operators and xarray's value operations stay tied to a column.

(design-tiers)=
## Package tiers: xrops above geoops (decided)

Decided 22 Oct 2026: pipekit-array (tier 0) → geoops (tier 1) → xrops (tier 2).
Every kernel is written once, in geoops or below, and runs on xarray through the pipekit-array adapter; xrops keeps native-xarray work and the state-physics kernels.

- **xrops may import geoops; geoops never imports xrops or xarray.** geoops has no `[xarray]` extra: the adapter lives in pipekit-array's `[xarray]` extra, and lifting is `xrops.lift`.
- **xrops holds:** selection by value, many-file assembly, `Dataset` / `DataTree` dispatch, lifting geoops operators over cubes, and state-physics kernels (mixed-layer depth, budgets).
- **Prerequisite:** geoops's base install is slimmed first (below); it blocks the tier change.
- **In `scripts/dependency_graph.py`:** `TIERS["geostack"]` moves xrtoolz from tier 1 to tier 2, above geotoolz.

```text
pipekit-array        tier 0: adapters, KernelOp, dim-level functions
     │               runs any kernel on NumPy, JAX and xarray
     │   [xarray] extra ──▶ xarray
     ▼
geoops               tier 1: every kernel operator, written once;
     │               geometry on carriers; no xarray in its base
     ▼
xrops                tier 2: native-xarray work (sel by value, groupby,
                     many files, Dataset/DataTree), xrops.lift(geoops op),
                     state physics; imports geoops and xarray
```

Arrows point from a package to what it imports: geoops reaches xarray only through pipekit-array's `[xarray]` adapter, never directly.

### Carriers versus operations

Supporting an xarray carrier is not being an xarray operation: an operator unwraps the carrier, runs its kernel and rewraps the result.

| Thing | Examples | Who owns it | A GeoStack operator… |
| --- | --- | --- | --- |
| xarray data structures (carriers) | `DataArray`, `Dataset`, `DataTree` | xarray | accepts and returns them through the adapter |
| native xarray operations | `sel`, `groupby`, `resample`, `rolling`, `interp`, accessors such as `.rio`, `.xvec` | xarray and its extensions | calls them only inside xrops |
| GeoStack operators | `NDVI`, `MatchedFilter`, `Gradient`, `Reproject` | geoops (kernels, geometry), xrops (native-xarray work, state physics) | is a function or `Operator` taking a carrier, never an accessor |

No accessors: an accessor registers globally on every `DataArray` and hides the import that provides it.
The scikit-learn comparison holds on the main point and breaks on two details:

|  | scikit-learn `set_output(transform="pandas")` | pipekit-array adapter |
| --- | --- | --- |
| separate package per container | no (there is no "pdlearn") | no (xrops writes no kernel twice) |
| what is unwrapped | a 2-D table, converted to NumPy at entry | named N-D arrays, unwrapped through `apply_ufunc` |
| dask input | computed eagerly | stays lazy, one task per chunk |
| which axes a step consumes | always the feature axis | declared by the kernel's signature, e.g. `"(c)->()"` with `c` = `band` |
| output metadata | column names | coords and attrs of the surviving dims |

### Prerequisite: a slim geoops base install

The tier change is only cheap if `pip install geotoolz-ops` stops pulling the vector and sensor stack.
This split was taken as the recommended option without a strong preference either way, so it stays reversible until the geotoolz issue lands.

| Dependency (base today) | Home | Why |
| --- | --- | --- |
| numpy, scipy, jaxtyping, typing-extensions, pipekit, einx | base | kernels and `KernelOp`s |
| affine, pyproj | base | transforms and CRS on carriers |
| pandas | base, unless an import audit finds it only in vector code | small, often already installed |
| rasterio | `[raster]` | reprojection and rasterizing; xrtoolz already pulls it via rioxarray |
| geopandas, shapely | `[vector]` | polygons and points as tables |
| scikit-image | `[morph]` | morphology and segmentation |
| georeader-spaceml | `[sensors]` | sensor readers and geometry |

Gate: a fresh `pip install geotoolz-ops` installs none of rasterio, geopandas, shapely, scikit-image or georeader; `import geoops` succeeds; an operator that needs an extra raises an `ImportError` naming it.
A geotoolz issue for the slim base blocks the tier change.

### Costs of the choice

Tiering won on where code lives; its costs are install weight (removed by the slim base) and the lost symmetry of the two lanes.

|  | Chosen: tiered | Not taken: two lanes, same tier |
| --- | --- | --- |
| lifting geoops over cubes (OP-O4) | `xrops.lift`, in the package that knows xarray | `geoops.lift` behind `geoops[xarray]` |
| gradient (OP-O7) | kernel and carrier wrapper in geoops, xarray wrapper in xrops | kernel in pipekit-array, which stops being domain-free |
| xrops reusing geoops operators | imports them and can type-check them as arguments | only by composing in a pipekit pipeline |
| installing xrops alone | adds geoops's slim base | xarray stack only |
| state physics (MLD, budgets) on a geoarray carrier | lives in xrops, so xarray must be installed | same |
| `import xrops` | also imports geoops (not measured) | imports xarray only |
| runtime speed | same | same |
| symmetry of the lanes | gone: geoops can never use an xrops kernel | kept |

### Demo API

How one kernel runs on every carrier, eagerly, under `jit` and lazily through `apply_ufunc`, is shown once, in [KernelOp](#design-kernelop).
`NDVI` (defined in the geoops section) is written once in geoops; no `xrops.NDVI` exists, and the xarray call works because pipekit-array's `[xarray]` adapter is registered, not because xrops is installed.
What tiering adds is xrops composing geoops; lifting over a cube and the gradient have their own sections below.

**xrops composing geoops.**

```python
# monthly NDVI climatology over a lazy cube: a geoops kernel, then a by-value reduction
# clim(m, y, x) = (1 / |Tₘ|) ∑_{t ∈ Tₘ} NDVI(t, y, x), Tₘ the dates in month m
ndvi: Operator = geoops.NDVI()
climatology: Operator = xrops.Climatology(op=ndvi, freq="month")
clim: DataArray["month y x"] = climatology(cube)

# the same as a pipekit pipeline
monthly: Operator = xrops.Climatology(freq="month")
pipe: pipekit.Pipeline = ndvi | monthly
clim_2: DataArray["month y x"] = pipe(cube)
```

The pipeline form worked under the two-lane design too; tiering adds the first, where an xrops operator takes a geoops operator as an argument and type-checks it.

## The levels in practice

Every package that computes writes its numerics as L1 primitives on arrays, L2 operators on carriers above them, and L3 algorithms on readable geo types at the top; the rules are in [the three levels](#design-levels).
Three examples, one per level, from the GeoNumerics and GeoModels side of the stack.

### L1: a primitive (finitevolX)

```python
# finitevolx/_src/array.py — arrays in, arrays out, no carrier, no CRS
class CGrid(eqx.Module):
    dx_u: Float[Array, "H W"]  # zonal spacing at u points, m
    dy_v: Float[Array, "H W"]  # meridional spacing at v points, m
    area_t: Float[Array, "H W"]  # cell area at T points, m²
    mask_t: Bool[Array, "H W"]  # ocean = True


def divergence(
    u: Float[Array, "... H W"],  # zonal velocity on u points
    v: Float[Array, "... H W"],  # meridional velocity on v points
    grid: CGrid,
) -> Float[Array, "... H W"]:
    # ∇·𝐮 ≈ [δₓ(u Δy) + δᵧ(v Δx)] / A, zero on land
    ...
```

### L2: an operator on a carrier (somax)

```python
# somax/geo/operators.py — behind somax[geo]; GeoModels may import GeoStack and GeoNumerics
class Divergence(Operator):
    def __call__(
        self, u: GeoArray["... y x"], v: GeoArray["... y x"]
    ) -> GeoArray["... y x"]:
        # derived geometry, once per call: transform + CRS → spacings and areas (pyproj lives here)
        grid: CGrid = cgrid_from_transform(u.transform, u.crs, mask=u.valid)
        # the L1 call on raw arrays
        div: Float[Array, "... H W"] = divergence(u.data, v.data, grid)
        # rewrapped on the T-point grid
        return u.rewrap(div, dims=u.dims)
```

### L3: a simulator and an assimilator (somax, vardax)

```python
# somax/geo/simulate.py — linear shallow water, η the SSH anomaly
def simulate(
    eta0: GeoMap,  # GeoArray["y x"], initial state
    times: Int[np.ndarray, "T"],  # output times tₖ, s after eta0's time; concrete
    depth: float,  # H, m
    dt: float,  # Δt, s
    mask: Bool[Array, "H W"] | None = None,  # ocean cells; None means eta0.valid
) -> GeoImageSeries:  # GeoArray["time y x"], one frame per requested time
    # entry, before tracing: output k is reached after nₖ = tₖ / Δt steps; n = 0 is η₀
    n_out: Int[np.ndarray, "T"] = np.rint(times / dt).astype(int)
    off_grid: bool = bool(np.any(n_out * dt != times))
    gaps: Int[np.ndarray, "T-1"] = np.diff(n_out)
    unsorted: bool = bool(np.any(gaps < 0))
    if off_grid or unsorted or n_out[0] < 0:
        raise ValueError("times must be sorted, non-negative multiples of dt")
    n_total: int = int(n_out[-1])
    # entry: carrier → arrays and derived geometry, once; the domain is the mask the
    # caller passes, so finite placeholders on land never turn land into ocean
    ocean: Bool[Array, "H W"] = eta0.valid if mask is None else mask
    grid: CGrid = cgrid_from_transform(eta0.transform, eta0.crs, mask=ocean)
    eta_init: Float[Array, "H W"] = eta0.data
    zeros: Float[Array, "H W"] = jnp.zeros_like(eta_init)
    # (η, u, v), each (H, W)
    state0: tuple[Float[Array, "H W"], Float[Array, "H W"], Float[Array, "H W"]] = (
        eta_init,
        zeros,
        zeros,
    )

    def step(
        state: tuple[Float[Array, "H W"], ...], _: None
    ) -> tuple[tuple[Float[Array, "H W"], ...], Float[Array, "H W"]]:
        eta, u, v = state
        # ∂η/∂t = −H ∇·𝐮
        div: Float[Array, "H W"] = divergence(u, v, grid)
        eta_next: Float[Array, "H W"] = eta - dt * depth * div
        # ∇η = (∂η/∂y, ∂η/∂x)
        gy: Float[Array, "H W"]
        gx: Float[Array, "H W"]
        gy, gx = gradient(eta_next, grid)
        # ∂𝐮/∂t = −g ∇η, g = 9.81 m s⁻²
        u_next: Float[Array, "H W"] = u - dt * G * gx
        v_next: Float[Array, "H W"] = v - dt * G * gy
        return (eta_next, u_next, v_next), eta_next

    # one trace, no carrier inside the loop: η after each of the S = n_total steps, (S, H, W)
    etas: Float[Array, "S H W"]
    _, etas = jax.lax.scan(step, state0, None, length=n_total)
    # index n is the state after n steps, so η₀ goes first: (S + 1, H, W)
    first: Float[Array, "1 H W"] = eta_init[None]
    traj: Float[Array, "S+1 H W"] = jnp.concatenate([first, etas])
    # the frames at the requested times: (T, H, W)
    frames: Float[Array, "T H W"] = traj[n_out]
    # exit: arrays → a series on eta0's grid, stamped with the requested times;
    # TimeAxis stamps are int64 nanoseconds, so offsets in s become ns first
    offsets_ns: Int[np.ndarray, "T"] = times.astype(np.int64) * 1_000_000_000
    stamps: Int[np.ndarray, "T"] = eta0.time + offsets_ns
    return GeoImageSeries.from_frames(frames, like=eta0, time=stamps)


# vardax/geo/assimilate.py — 4D-Var over a window of along-track SLA.
# vardax is GeoML and may not import GeoModels, so the forward model M is an argument.
# M: an initial map, output times and the ocean mask → the trajectory at those times
ForwardModel = Callable[
    [GeoMap, Int[np.ndarray, "T"], Bool[Array, "H W"]], GeoImageSeries
]


def assimilate(
    background: GeoMap,  # xᵇ, GeoArray["y x"]
    obs: GeoTracks,  # y, GeoPoints["obs"] padded to N, with a valid mask
    times: Int[np.ndarray, "T"],  # the window's output times, s after xᵇ's time
    forward: ForwardModel,  # M, from the caller
    sigma_b: float,  # background error std, m
    sigma_o: float,  # observation error std, m
) -> GeoImageSeries:

    # m_b = 1 where xᵇ is valid (ocean); land and nodata hold 0 and never enter J
    valid_b: Bool[Array, "H W"] = background.valid
    xb: Float[Array, "H W"] = jnp.where(valid_b, background.data, 0.0)

    def cost(x0: Float[Array, "H W"]) -> Float[Array, ""]:
        # m_b ⊙ x₀: land cells are held at 0, so their gradient is 0 and they never move
        x0_ocean: Float[Array, "H W"] = jnp.where(valid_b, x0, 0.0)
        # M(x₀) on the background's domain: the mask is passed, not re-derived from data
        x0_map: GeoMap = background.with_data(x0_ocean)
        traj: GeoImageSeries = forward(x0_map, times, valid_b)
        # H(M(x₀)): sampled at the observations in space and time
        predicted: GeoTracks = geoarray.sample(traj, obs)
        # d = m ⊙ (y − H(M(x₀))); padded slots give 0, never NaN
        resid: Float[Array, "N"] = obs.data - predicted.data
        d: Float[Array, "N"] = jnp.where(obs.valid, resid, 0.0)
        # m_b ⊙ (x₀ − xᵇ): finite everywhere, zero off the valid cells
        dx_raw: Float[Array, "H W"] = x0_ocean - xb
        dx: Float[Array, "H W"] = jnp.where(valid_b, dx_raw, 0.0)
        # J = ½‖m_b ⊙ (x₀ − xᵇ)‖²/σ_b² + ½‖d‖²/σ_o²
        jb: Float[Array, ""] = 0.5 * einx.dot("h w, h w ->", dx, dx) / sigma_b**2
        jo: Float[Array, ""] = 0.5 * einx.dot("n, n ->", d, d) / sigma_o**2
        return jb + jo

    # xₐ = argmin J, by gradient descent on ∇J
    cost_jit: Callable = jax.jit(cost)
    x_a: Float[Array, "H W"] = minimize(cost_jit, xb)
    # the analysis trajectory, as a readable series
    analysis_map: GeoMap = background.with_data(x_a)
    traj_a: GeoImageSeries = forward(analysis_map, times, valid_b)

    # land back to NaN in the output (broadcast over time), so its validity matches xᵇ
    def restore_land(frames: Float[Array, "T H W"]) -> Float[Array, "T H W"]:
        return jnp.where(valid_b, frames, jnp.nan)

    return traj_a.map(restore_land)


# the caller binds the model, in GeoModels or a script: M(x₀) = simulate(x₀, times; H, Δt)
forward: ForwardModel = functools.partial(somax.geo.simulate, depth=4_000.0, dt=600.0)
# daily output over a 7-day window, t = 0 included: (8,)
times: Int[np.ndarray, "T"] = np.arange(0, 8 * 86_400, 86_400)
series: GeoImageSeries = vardax.geo.assimilate(
    background, obs, times, forward, sigma_b=0.05, sigma_o=0.03
)
```

vardax only knows M through `forward`, so the same assimilator runs any model the caller binds, and the dependency points down.
`simulate` turns requested output times into step counts before tracing, so a daily output with a 600 s step takes 144 steps per frame, and t = 0 returns the initial state.
Inside `cost` the series is built and unwrapped once per evaluation.
Whether that is cheap enough under `grad`, or `cost` should call the L1 loop directly, is open (OP-O8 in the open questions).

## Where each operation is written, once

| Operation | Written in | Runs on |
| --- | --- | --- |
| L1 kernel | its L1 package (radtrax, somax, …), or `_src/array.py` in geoops (observation, sensor, diagnostic geometry) or xrops (state physics) | raw arrays |
| dim-level function (reduce, stack, map_over, …) | pipekit-array | every registered carrier |
| KernelOp, observation and sensor domain (indices, radiometry, qa, normalize, spectral) | geoops | both columns |
| KernelOp, physical and state domain (atmosphere, ocean, budgets, metrics) | xrops | both columns; needs xarray installed |
| per-item geometry (reproject, rasterize, sample, windows, patches, gradient) | geoops | geoarray carriers; cubes through `xrops.lift` |
| many-item geometry (lazy regrid of a cube, mosaics of many files) | xrops | xarray carriers |
| value analysis (resample, groupby, climatology, alignment) | native xarray; xrops builds on it | xarray carriers |

The split of `KernelOp`s by domain decides who maintains them, not where they run.

### Dependency direction

```text
pipekit
   │
   ▼
pipekit-array ── [xarray] adapter ──▶ xarray
   │                                  ▲
   ▼                                  │
geoarray                              │
   │                                  │
   ▼                                  │
geoops   (never imports xarray        │
   │      or xrops)                   │
   ▼                                  │
xrops ─────────── imports ─────────────┘
   reuses geoops directly and through xrops.lift
```

- **xrops imports geoops (decided 22 Oct 2026).**
  It reuses every geoops kernel and geometry operator instead of writing its own, and reaches xarray carriers through the pipekit-array adapter.
- **geoops never imports xrops or xarray.**
  It has no `[xarray]` extra; `xrops.lift(op, dim="time")` runs a per-item geoops operator on each slice of a cube, replacing both `geoops.lift` and the earlier `xrops.ApplyGeoOp`.
- **A cube user who needs NDVI imports geoops.**
  With the slim base, that costs NumPy, SciPy, pyproj and affine, no JAX and no vector stack.

### Lifting a geometry operator over a cube

```python
# a per-item operator, written once for geoarray carriers: GeoArray["band y x"] → GeoArray["y x"]
detect: Operator = geoops.plume.DetectPlumes(threshold=tau)

# lifted over time by xrops: each slice becomes a GeoImage, runs, and is wrapped back
per_date: Operator = xrops.lift(detect, dim="time")

# lazy, one task per dask chunk: cube is DataArray["time band y x"]
masks: DataArray["time y x"] = per_date(cube)
```

Under the hood `lift` is `xr.apply_ufunc` with the spatial dims as core dims; each chunk is turned into a carrier with `GeoArray.from_xarray`, so the georeference comes from the cube's grid mapping.

## geoops

geoops holds three kinds of code: L1 kernels in its `array.py` modules, observation-domain `KernelOp`s that run on both columns, and per-item geometry operators that run on geoarray carriers.
Today it has 218 operator classes; 61 store a positional `self.axis`, and 106 kernel signatures default to `axis=-3`.

### Three operator shapes

| Shape | Base | Backends | Example |
| --- | --- | --- | --- |
| kernel operator | `KernelOp` | numpy, jax; xarray through the adapter | `NDVI(dim="band")` |
| geometry operator, array math | `Operator`, derives geometry from the georeference, then calls a primitive | numpy, jax | `Sample`, `Window`, `ShiftOrigin`, `Reproject` (pyproj in the operator, `map_coordinates` primitive) |
| geometry operator, engine-bound | `Operator`, calls GDAL or shapely; no primitive | numpy | `Rasterize`, `Vectorize` |

```python
# L1 kernel in geoops/indices/_src/array.py: "(),()->()"
def normalized_difference(
    a: Float[Array, "*s"], b: Float[Array, "*s"], eps: float
) -> Float[Array, "*s"]:
    # ND = (a − b) / (a + b + ε)
    return (a - b) / (a + b + eps)


# L2 operator: two labels of one dim, then the kernel
class NDVI(Operator):
    backends: tuple[str, ...] = ("numpy", "jax", "xarray")

    def __init__(
        self, nir: str = "B8", red: str = "B4", dim: str = "band", eps: float = 1e-10
    ) -> None:
        self.nir: str = nir
        self.red: str = red
        self.dim: str = dim
        self.eps: float = eps

    def _apply(self, scene: GeoArray["band y x"]) -> GeoArray["y x"]:
        # exact label lookups by name, any carrier
        nir_sel: dict[str, str] = {self.dim: self.nir}
        red_sel: dict[str, str] = {self.dim: self.red}
        nir: GeoArray["y x"] = pipekit_array.sel(scene, **nir_sel)
        red: GeoArray["y x"] = pipekit_array.sel(scene, **red_sel)
        # the kernel, element-wise, through KernelOp
        nd: KernelOp = KernelOp(
            normalized_difference, "(),()->()", core={}, eps=self.eps
        )
        # NDVI = (ρ_NIR − ρ_red) / (ρ_NIR + ρ_red + ε)
        return nd(nir, red)
```

```python
# L2 geometry operator, array math: pyproj in the operator, a primitive for the resampling
class Reproject(Operator):
    backends: tuple[str, ...] = ("numpy", "jax")

    def __init__(self, crs: str, resolution: float) -> None:
        self.crs: str = crs
        self.resolution: float = resolution

    def _apply(self, scene: GeoArray["band y x"]) -> GeoArray["band y x"]:
        # the target grid: transform and shape in the new CRS
        target: GridSpec = geoarray.target_grid(
            scene, crs=self.crs, resolution=self.resolution
        )
        # derived geometry: source pixel coordinates (r, c) of every target pixel, by pyproj
        rows: Float[Array, "H W"]
        cols: Float[Array, "H W"]
        rows, cols = geoarray.source_coords(scene, target)
        # L1: q(y′, x′) = ∑ wᵢ q(rᵢ, cᵢ), bilinear weights wᵢ, per band
        warped: Float[Array, "C H W"] = map_coordinates(scene.data, rows, cols, order=1)
        # rewrap on the target grid, same dims
        return GeoArray.from_grid(warped, target, dims=scene.dims)
```

### Families

| Family | Shape | Backends |
| --- | --- | --- |
| `indices`, `qa` | kernel operators | numpy, jax, xarray |
| `normalize` | kernel operators; skimage-backed ones engine-bound | numpy, jax, xarray / numpy |
| `compositing` | thin named operators over `stack`, `reduce`, `argreduce`, `take_along` | numpy, jax, xarray |
| `einx` | kernel operators over einx patterns (see OP-O2) | numpy, jax, xarray |
| `augment` | kernel operators for flips, crops, noise (PRNG key under JAX); scipy warps engine-bound | numpy, jax / numpy |
| `spectral` | kernel operators backed by scipy | numpy, xarray |
| `learn` | scikit-learn wrappers | numpy |
| `geom`, `io`, `viz` | geometry, engine-bound | numpy |
| `mask`, `measure`, `feature`, `segment`, `restore`, `radiometry` | mixed: vector and solar-geometry parts engine-bound, array parts kernel operators | per op |
| `matched_filter`, `plume` | leaving for radtrax and plumax (#482); rebuilt there as kernel operators | — |

### Rules

- **`dim=` replaces `axis=`** in every operator; kernels keep a positional axis and take it last.
- **Every operator declares `backends`.**
  Given a carrier it cannot run on, it raises `BackendError` naming the fix (`to_numpy()`, `xrops.lift`, or the right column).
- **Fitted operators** follow pipekit's `FittableTransformer`: `fit` returns a new operator with arrays filled; under JAX, `pipekit_jax.to_jax(op)` makes those arrays leaves.
- **No geoops operator reimplements a pipekit-array function or a native xarray operation.**

## xrops

xrops is the xarray column: value analysis built on native xarray operations, many-item geometry, and the physical- and state-domain `KernelOp`s, which also run on geoarray carriers.
It is the code of today's `xrtoolz` distribution, renamed per the [package layout page](./package_layout.md), with `xrcore`'s operator base folded into `xrops._src`.
It sits one tier above geoops: it imports geoops for every kernel and geometry operator it does not own, and never re-implements one.

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
    sigma0: Float[Array, "... L"],  # σ₀, kg m⁻³, L levels last
    depth: Float[Array, "... L"],  # z, m
    threshold: float,  # kg m⁻³
) -> Float[Array, "..."]:
    # numpy or jax.numpy, chosen by the input
    xp: ModuleType = array_namespace(sigma0)
    # σ₀(z_ref), z_ref the shallowest level
    ref: Float[Array, "... 1"] = sigma0[..., :1]
    # Δσ(z) = σ₀(z) − σ₀(z_ref)
    delta: Float[Array, "... L"] = sigma0 - ref
    # 𝟙[Δσ(z) > threshold]
    below: Bool[Array, "... L"] = delta > threshold
    # the first level where Δσ exceeds the threshold
    first: Int[Array, "..."] = xp.argmax(below, axis=-1)
    first_l: Int[Array, "... 1"] = first[..., None]
    # MLD = z at that level
    picked: Float[Array, "... 1"] = xp.take_along_axis(depth, first_l, axis=-1)
    mld: Float[Array, "..."] = picked[..., 0]
    # argmax of an all-False row is 0: no crossing means no MLD, so NaN
    crossed: Bool[Array, "..."] = einx.any("... [l]", below)
    return xp.where(crossed, mld, xp.nan)


# L2: the level dim named, the threshold bound
mld_op: KernelOp = KernelOp(
    mixed_layer_depth, "(l),(l)->()", core={"l": "level"}, threshold=0.03
)

# a GLORYS-like cube, lazy: (time, level, y, x) → (time, y, x)
mld_cube: DataArray["time y x"] = mld_op(sigma0_cube, depth_cube)

# one gridded profile on a geoarray carrier: GeoVolume (level, y, x) → GeoMap (y, x)
mld_map: GeoArray["y x"] = mld_op(sigma0_volume, depth_volume)
```

σ₀ is potential density anomaly (kg m⁻³), z depth (m), and the threshold 0.03 kg m⁻³ a common MLD criterion.

A profile that never crosses the threshold (a well-mixed column, or one too shallow) returns NaN rather than the shallowest depth; a caller that wants the bottom depth as a fallback fills the NaNs explicitly.

### Geometry written once at L1, wrapped per column

A gradient needs grid spacing, which is geometry.
Decided 22 Oct 2026 (OP-O7): the diagnostic kernel lives in geoops and takes spacings as arrays; each column only derives the spacings.

```python
# geoops/geom/_src/array.py — L1, spacings as arrays
def central_gradient(
    f: Float[Array, "... H W"],
    dy: Float[Array, "H W"],
    dx: Float[Array, "H W"],
) -> Float[Array, "2 ... H W"]:
    # numpy or jax.numpy, chosen by the input
    xp: ModuleType = array_namespace(f)
    # neighbours along y and x
    f_n: Float[Array, "... H W"] = xp.roll(f, -1, -2)
    f_s: Float[Array, "... H W"] = xp.roll(f, 1, -2)
    f_e: Float[Array, "... H W"] = xp.roll(f, -1, -1)
    f_w: Float[Array, "... H W"] = xp.roll(f, 1, -1)
    # ∂f/∂y ≈ (fᵢ₊₁ − fᵢ₋₁) / 2Δy
    fy: Float[Array, "... H W"] = (f_n - f_s) / (2 * dy)
    # ∂f/∂x ≈ (fⱼ₊₁ − fⱼ₋₁) / 2Δx
    fx: Float[Array, "... H W"] = (f_e - f_w) / (2 * dx)
    # (∂f/∂y, ∂f/∂x) on a new leading axis
    return einx.rearrange("... h w, ... h w -> (1 + 1) ... h w", fy, fx)


# geoops: spacing from the transform (a, e), or geodesic distances for a geographic CRS
grad_op: Operator = geoops.geom.Gradient()
grad_map: GeoArray["component y x"] = grad_op(ssh_map)  # ssh_map: GeoArray["y x"]

# xrops: spacing from coordinate values, spherical metrics on lat/lon; the same geoops kernel
grad_xr: Operator = xrops.Gradient(dims=("lat", "lon"))
grad_cube: DataArray["component time lat lon"] = grad_xr(
    ssh
)  # ssh: DataArray["time lat lon"]
```

Edges are elided here (`roll` wraps); the real kernel handles them.

| Column | Wrapper | Lives in | Spacing from |
| --- | --- | --- | --- |
| geoarray | `geoops.geom.Gradient` | geoops | the transform (`a`, `e`), or geodesic distances for a geographic CRS |
| xarray | `xrops.Gradient` (today's `xrgrad`) | xrops, importing the geoops kernel | coordinate values, with spherical metrics on lat/lon |

Two thin wrappers, one kernel.
The tiering makes this possible: xrops imports the geoops kernel, so pipekit-array keeps no domain kernels.
GeoStack and GeoNumerics still never import each other, which fixes where model-grade calculus goes:

- **Diagnostic gradients** (SSH or a plume map, for analysis): the geoops kernel above.
- **Model-grade calculus** (staggered grids, finite volumes, boundary conditions): finitevolX's kernels, raw arrays only.
  Carrier wrappers for them live in the GeoML or GeoModels package that needs them (somax, vardax), which may import both GeoNumerics and GeoStack; neither GeoStack nor finitevolX holds them.

### Families

| Family (today in xrtoolz) | Becomes |
| --- | --- |
| `atm`, `ocn`, `budgets` | state-domain kernel operators + value analysis |
| `metrics` | kernel operators over named dims; skill scores by value |
| `transforms` (Fourier, wavelet, DCT) | kernel operators over named dims |
| `interpolate` | native `interp` plus many-item geometry |
| `geo` | split: geometry (crs, mosaic, rasterize, footprint) → many-item geometry; analysis (detrend, extremes, regimes) → value analysis |
| `einx` (`xreinx`), `sklearn` (`xrsklearn`) | see OP-O2 and OP-O3 |
| `rs`, `ice` | removed (empty stubs; observation indices live in geoops and run on cubes as they are) |

## Testing

| Test | Lives in | Proves |
| --- | --- | --- |
| operator contract | geoops (today's `test_operator_contract.py`), xrops | config round trip, graph mode, declared backends, `BackendError` on the rest |
| `check_kernel_op` | pipekit-array test kit, run by both packages | each kernel operator gives equal results on a geoarray carrier, a JAX carrier, an xarray cube and raw arrays |
| lift parity | xrops | `xrops.lift(op, dim="time")` on a cube equals the op on each slice, eagerly |
| tiers | geostack `tests/test_dependency_graph.py` | geoops imports neither xrops nor xarray; xrops is tier 2 above geotoolz |
| slim base | geotoolz CI | a fresh `pip install geotoolz-ops` installs none of rasterio, geopandas, shapely, scikit-image, georeader; `import geoops` works |
| one trace | geoops, xrops | kernel operators compile once under `jit` for same-shape carriers |
| summary | both | `Sequential.summary` prints dims for every step, no `?` |
| laziness | xrops | operators on dask-backed cubes compute nothing until asked |

## Migration

```text
1 pipekit, pipekit-array: adapter, Signature, functions, KernelOp
        │
        ▼
2 geoarray: adapter (geoarray page)
        │
        ▼
3 geoops: slim base install (blocks 5)
  axis → dim, KernelOps, backends,
  gradient kernel, no [xarray] extra
        │
        ▼
4 dependency graph: xrtoolz to tier 2
        │
        ▼
5 xrops (package layout page): rename, fold xrcore,
  Signature re-export, import geoops kernels,
  xrops.lift, xrops.Gradient, state-physics KernelOps
```

Each step is a 0.X.0 bump. xrops now follows geoops instead of moving in parallel: it imports geoops's kernels, so step 5 waits for steps 3 and 4.

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| OP-O1 | Where kernel operators live | **Decided 22 Oct 2026:** observation and sensor kernels in geoops; state-domain kernels (MLD, budgets) in xrops; xrops imports geoops kernels and never re-implements one | a third, neutral package of kernel operators |
| OP-O2 | Two einx bridges (`geoops.einx`, `xreinx`) | one named-dims einx bridge in pipekit-array (`[einx]` extra); geoops keeps only its spatial-survival rule | keep both |
| OP-O3 | Two scikit-learn bridges (`geoops.learn`, `xrsklearn`) | `to_rows` / `from_rows` as dim-level functions in pipekit-array, one estimator wrapper on pipekit's `FittableTransformer` | keep both |
| OP-O4 | Lifting geometry over cubes | **Decided 22 Oct 2026:** `xrops.lift(op, dim=...)` in xrops; geoops has no `[xarray]` extra | `geoops.lift` behind `geoops[xarray]` (not taken) |
| OP-O5 | How operators use `KernelOp` | by composition: an operator keeps its own config and calls a `KernelOp` inside (as `NDVI` does) | every kernel operator subclasses `KernelOp` |
| OP-O6 | The `backends` vocabulary | `"numpy"`, `"jax"`, `"xarray"` | backends and carrier families declared separately |
| OP-O7 | Where grid calculus lives | **Decided 22 Oct 2026:** the diagnostic gradient kernel in geoops (`geoops/geom/_src/array.py`), wrapped by `geoops.geom.Gradient` and `xrops.Gradient` (today's `xrgrad`); pipekit-array keeps no domain kernels; model-grade calculus in finitevolX, wrapped by the GeoML or GeoModels package that uses it | kernel in pipekit-array (not taken) |
| OP-O8 | Does an L3 algorithm call L3 algorithms inside a differentiated cost (the assimilator calling `simulate`)? | yes, while wrap and unwrap stay cheap under `grad`; measured in the spike | the cost calls the L1 loop directly, and only the outer call returns a `GeoImageSeries` |
