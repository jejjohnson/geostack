---
title: "functax: design details"
short_title: "functax details"
description: "The reference behind the functax design: the building blocks, types, algorithms and JAX notes an implementer needs, and the survey that informed them."
date: 2026-11-05
keywords:
  - functax
  - design
  - details
  - GeoStack
---

The reference behind the [functax design](functax.md): the building blocks, types, algorithms and JAX notes an implementer needs, and the survey that informed them.

## Building blocks and how they depend on each other

The package is seven blocks in a chain; each needs only the blocks above it. Policies cut across all of them.

| # | Block | Does | Needs |
| --- | --- | --- | --- |
| 1 | Geometry | Axes, grids and point sets, with periodicity, bounds, a time epoch and a metric (Euclidean, chord, geodesic) | nothing |
| 2 | Locate | Coordinates to a cell index and a fractional offset on grids; to neighbour lists for scattered sites | geometry |
| 3 | Basis | Weights from offsets (hat, cubic, B-spline), or a kernel k(x, x′) between sites | locate; the metric, for kernels |
| 4 | Operator H | A stencil (indices, weights, a valid flag) for local bases; a dense or structured operator for kernels | locate, basis |
| 5 | Fit 𝒮 | The coefficient solver: identity, banded prefilter, tridiagonal, Cholesky, conjugate gradients, variational | basis, H, a linear-solver library |
| 6 | Plan | Geometry, H, the fit and the policies frozen together: fit once, apply many times | 1–5 |
| 7 | Outputs | Values, and on request the variance, H, Hᵀ and ∂f̂/∂x\* | the plan |

Three policies thread through blocks 2 to 6:

- **Outside:** a target off the source domain gives NaN, a clamped value, or an extrapolation.
- **Missing:** a NaN or masked sample propagates, is skipped with weights renormalised, or is filled first.
- **Boundary:** a stencil that runs off the edge mirrors, wraps, repeats the edge, or reads a constant.

The chain shows where external code enters. Only block 5 needs a solver, and only kriging and the variational fit need more than a banded one.

## Carriers as the operator's domain and codomain

An interpolator is an operator between data types, so georeferenced carriers are its natural interface. The carrier types name the task.

| Task | Domain → codomain | What it needs |
| --- | --- | --- |
| Read a field at points | grid → points | H, local stencil |
| Regrid | grid → grid | H, local or conservative |
| Grid scattered data | points → grid | a solver |
| Fill gaps | masked grid → grid | a solver with the mask |
| Resample a series | series → series | a 1-D H |
| Points to points | points → points | a kernel solve, then H |

The adjoint runs the same arrow backwards. Hᵀ takes point residuals back onto the grid, which is what data assimilation and inverse problems need.

The core should still take plain arrays plus geometry objects. A carrier adapter unpacks the geometry, calls the core, and rewraps the result. That keeps the core free of any one carrier library, and lets several carriers (a georeferenced array, xarray, a point table) share it.

## Sketch of the core types

Five types carry the design: geometry, a local basis or a kernel, a stencil that is H, a fit, and a plan. Every name here is proposed; `interp` is a placeholder for the package name.

Every sketch on this page follows the code-sketch and python-style standards: a shapes header, a unicode equation above each mathematical line, typed signatures, an aligned shape column, a status marker on every stack call, einx and lineax, descriptive names and one operation per line.

Shape letters, for the whole page. N is the number of source values; a grid with axis lengths n₁ … n\_D flattens to N = ∏ nᵢ. M is the number of targets, S the taps per target, Q the neighbours, D the spatial dims, J the coefficients, T the times, C the bands, Ny × Nx a grid and G the rows of a difference operator. Leading batch dims are `*batch` in types and … in the shape column. In types the core names its dims `source`, `target`, `tap`, `neighbour`, `dim` and `coefficient`; carriers use the stack's names (`time band y x`, `obs`). Matrices keep their letters in equations: H the read-out, K a kernel matrix, R an observation-error covariance, L a difference operator. On a regular axis, x₀ is the origin and Δ the spacing; i is a cell index and t ∈ \[0, 1) the offset inside the cell.

### Geometry

Geometry is data, not keywords: periodicity, the time epoch and the metric travel with the coordinates.

```python
# Shapes: N source values; M targets; D spatial dims

import abc
from typing import Literal

import einx
import equinox as eqx
import jax.numpy as jnp
import lineax as lx
import numpy as np
from jaxtyping import Array, Bool, Float, Int

Op = lx.AbstractLinearOperator


# proposed (interp)
class Axis(eqx.Module):
    """One coordinate axis: monotone values, optionally periodic."""

    values: Float[Array, " source"]
    # 360.0 for longitude; None when the axis is not periodic
    period: float | None = None
    # datetimes are stored as float offsets from this epoch
    epoch: np.datetime64 | None = None


# proposed (interp)
class Grid(eqx.Module):
    """A tensor-product grid; its values are (…, n₁, …, n_D)."""

    axes: tuple[Axis, ...]


# proposed (interp)
class Points(eqx.Module):
    """Scattered sites, with the metric that distances use."""

    # GeoPoints calls the 2-D case xy; a vertical coordinate stays separate
    coordinates: Float[Array, "target dim"]
    metric: Literal["euclidean", "chord", "geodesic"] = "euclidean"
```

### Locate, basis and kernel

A local basis turns fractional offsets into tap weights; a kernel compares two site sets. Locate is plain arithmetic on a regular axis and a `searchsorted` on a rectilinear one.

```python
# Shapes: M targets; S taps per target; N source values; D spatial dims


# proposed (interp)
def locate(
    axis: Axis, positions: Float[Array, " target"]
) -> tuple[Int[Array, " target"], Float[Array, " target"]]:
    """Cell index and offset of each target on an axis.

    i = ⌊(x − x₀) / Δ⌋,  t = (x − x₀) / Δ − i on a regular axis;
    a sorted search on a rectilinear one.
    """


# proposed (interp)
class Basis(eqx.Module):
    """A local basis: taps per dim, and weights from fractional offsets."""

    # 1 nearest, 2 linear, 4 cubic
    taps: eqx.AbstractVar[int]

    @abc.abstractmethod
    def weights(self, offsets: Float[Array, " target"]) -> Float[Array, "target tap"]:
        """wₛ(t) = φ(oₛ − t), with oₛ the offset of tap s from cell i."""


# proposed (interp)
class Kernel(eqx.Module):
    """A global basis: k(x, x′) between two site sets."""

    @abc.abstractmethod
    def __call__(
        self,
        targets: Float[Array, "target dim"],
        sources: Float[Array, "source dim"],
    ) -> Float[Array, "target source"]:
        """Kᵢⱼ = k(xᵢ, x′ⱼ)."""
```

### The stencil is H

A stencil stores H row by row: S flat source indices, S weights, and whether the row is inside the domain. Applying it is a gather and a contraction; its adjoint is a scatter-add.

```python
# Shapes: N source values; M targets; S taps per target; D spatial dims


# proposed (interp)
class Readout(eqx.Module):
    """c → f̂(X*): a Stencil, or a dense kernel read-out K(X*, X)."""

    # False for a target outside the domain under the "nan" policy
    valid: eqx.AbstractVar[Bool[Array, " target"]]

    @abc.abstractmethod
    def apply(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]: ...

    @abc.abstractmethod
    def apply_adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]: ...


# proposed (interp)
class Stencil(Readout):
    """A sparse read-out H: row k reads S source values at flat indices.

    f̂ₖ = ∑ₛ wₖₛ c[iₖₛ],  and its adjoint (Hᵀr)ⱼ = ∑ₖ ∑ₛ [iₖₛ = j] wₖₛ rₖ.
    """

    indices: Int[Array, "target tap"]
    weights: Float[Array, "target tap"]
    # False outside the domain; those rows have zero weights, so H and Hᵀ skip them
    valid: Bool[Array, " target"]
    n_source: int = eqx.field(static=True)

    def apply(
        self, coefficients: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        # c[iₖₛ]
        gathered = coefficients[..., self.indices]  # (…, N) → (…, M, S)
        # f̂ₖ = ∑ₛ wₖₛ c[iₖₛ]
        values = einx.dot(  # (…, M, S) → (…, M)
            "... target tap, target tap -> ... target", gathered, self.weights
        )
        return values

    def apply_adjoint(
        self, residual: Float[Array, "*batch target"]
    ) -> Float[Array, "*batch source"]:
        # wₖₛ rₖ, one contribution per tap
        contributions = einx.multiply(  # (…, M) → (…, M, S)
            "... target, target tap -> ... target tap", residual, self.weights
        )
        # 0 ∈ ℝ^(… × N), in the residual's dtype
        batch_shape = residual.shape[:-1]
        zeros_shape = (*batch_shape, self.n_source)
        zeros = jnp.zeros(zeros_shape, residual.dtype)  # → (…, N)
        # (Hᵀr)ⱼ = ∑ₖ ∑ₛ [iₖₛ = j] wₖₛ rₖ
        adjoint = zeros.at[..., self.indices].add(contributions)  # → (…, N)
        return adjoint


# proposed (interp)
def tensor_stencil(
    grid: Grid,
    targets: Points,
    bases: tuple[Basis, ...],
    *,
    policies: Policies,
) -> Stencil:
    """H = H₁ ⊗ H₂ ⊗ … ⊗ H_D, one 1-D stencil per axis; S = ∏ taps_d."""
```

One basis per axis gives mixed schemes for free: nearest in time and linear in space is one tensor stencil.

### Fits and policies

A fit maps data to coefficients. Every fit except the Hermite slopes and a neural field is linear in y.

```python
# Shapes: N source values; M targets; J coefficients; G rows of L


# proposed (interp)
class InterpError(Exception):
    """The package's base error."""


# proposed (interp)
class PolicyError(InterpError, ValueError):
    """A policy names an option that does not exist."""


# proposed (interp)
class NonlinearSchemeError(InterpError, TypeError):
    """A linear-operator view was asked of a fit that is nonlinear in y."""


OUTSIDE_CHOICES = ("nan", "clamp", "extrapolate", "raise")


# proposed (interp)
class Policies(eqx.Module):
    """What happens outside the domain, at missing samples and at the edges.

    The outside policy acts in two places. At plan time, "clamp" moves an outside
    target onto the boundary and "extrapolate" keeps the scheme's weights there, so
    both rows stay valid; "raise" raises if any target is outside. "nan" marks the
    row invalid with zero weights, and the plan writes NaN there at apply time.

    Raises:
        PolicyError: An option that is not one of the listed choices.
    """

    outside: Literal["nan", "clamp", "extrapolate", "raise"] = "nan"
    missing: Literal["propagate", "renormalise", "fill"] = "propagate"
    boundary: Literal["mirror", "wrap", "edge", "constant"] = "mirror"

    def __check_init__(self) -> None:
        # Literal types are not checked at run time; missing and boundary alike
        if self.outside not in OUTSIDE_CHOICES:
            raise PolicyError(f"outside must be one of {OUTSIDE_CHOICES}")


# proposed (interp)
class Fit(eqx.Module):
    """𝒮: data to coefficients, c = 𝒮(y)."""

    @abc.abstractmethod
    def __call__(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch coefficient"]: ...


# proposed (interp)
class Identity(Fit):
    """Local schemes: c = y."""


# proposed (interp)
class BandedPrefilter(Fit):
    """B-splines: c = B⁻¹ y, one banded solve per axis."""


# proposed (interp)
class Tridiagonal(Fit):
    """1-D cubic splines: one tridiagonal solve for the knot slopes."""


# proposed (interp)
class KernelSolve(Fit):
    """RBF and kriging weights: c = (K + λI)⁻¹ y."""

    # K + λI, an (N, N) PSD operator
    gram: Op
    solver: lx.AbstractLinearSolver

    def __call__(self, data: Float[Array, " source"]) -> Float[Array, " source"]:
        # not pinned (lineax)
        # c = (K + λI)⁻¹ y
        solution = lx.linear_solve(self.gram, data, self.solver)  # (N,) → (N,)
        coefficients = solution.value  # → (N,)
        return coefficients


# proposed (interp)
class Variational(Fit):
    """Smoothing fit on a grid: x = (HᵀR⁻¹H + λLᵀL)⁻¹ HᵀR⁻¹y."""

    # H: grid → sites, (M, N)
    observation: Op
    # R⁻¹, (M, M)
    observation_precision: Op
    # L, a difference operator, (G, N)
    roughness: Op
    # λ ≥ 0
    smoothing_weight: float
    # matrix-free CG
    solver: lx.AbstractLinearSolver

    def __call__(self, data: Float[Array, " target"]) -> Float[Array, " source"]:
        # not pinned (lineax)
        # Hᵀ
        observation_adjoint = self.observation.transpose()  # (N, M) op
        # R⁻¹H
        weighted = self.observation_precision @ self.observation  # (M, N) op
        # HᵀR⁻¹H
        data_term = observation_adjoint @ weighted  # (N, N) op
        # Lᵀ
        roughness_adjoint = self.roughness.transpose()  # (N, G) op
        # LᵀL
        roughness_gram = roughness_adjoint @ self.roughness  # (N, N) op
        # λLᵀL
        penalty = self.smoothing_weight * roughness_gram  # (N, N) op
        # A = HᵀR⁻¹H + λLᵀL
        normal = data_term + penalty  # (N, N) op
        # not pinned (lineax)
        # A is symmetric positive semi-definite, which CG needs to know
        normal = lx.TaggedLinearOperator(normal, lx.positive_semidefinite_tag)
        # R⁻¹y
        weighted_data = self.observation_precision.mv(data)  # (M,) → (M,)
        # b = HᵀR⁻¹y
        rhs = observation_adjoint.mv(weighted_data)  # (M,) → (N,)
        # not pinned (lineax)
        # x = A⁻¹ b
        solution = lx.linear_solve(normal, rhs, self.solver)  # (N,) → (N,)
        field = solution.value  # → (N,)
        return field
```

Kernel and variational fits take one field; a batch goes through `jax.vmap`. The roughness operator L is where a finite-volume or spectral differential-operator library plugs in.

### Scheme and plan

A scheme is configuration; a plan is that configuration frozen for one source and one target geometry. Building a plan may use NumPy and SciPy; applying it must trace.

```python
# Shapes: N source values; M targets; J coefficients


# proposed (interp)
class Plan(eqx.Module):
    """A fitted interpolator: fit once, apply to many fields.

    f̂(X*) = H 𝒮(y).

    Raises:
        NonlinearSchemeError: ``operator`` on a fit that is nonlinear in y.
    """

    fit: Fit
    read: Readout
    policies: Policies

    def __call__(
        self, data: Float[Array, "*batch source"]
    ) -> Float[Array, "*batch target"]:
        # c = 𝒮(y)
        coefficients = self.fit(data)  # (…, N) → (…, J)
        # f̂(X*) = H c
        values = self.read.apply(coefficients)  # (…, J) → (…, M)
        # f̂ₖ = NaN outside the domain: never an extrapolated value under "nan"
        if self.policies.outside == "nan":
            fill = jnp.asarray(jnp.nan, dtype=values.dtype)
            values = einx.where(  # (…, M) → (…, M)
                "target, ... target, -> ... target", self.read.valid, values, fill
            )
        return values

    def operator(self) -> Op:
        """H𝒮 as one (M, N) linear operator."""

    def variance(self) -> Float[Array, " target"]:
        """σ²(x*) = k(x*, x*) − k*ᵀ(K + λI)⁻¹k*; needs a kernel read-out."""


# proposed (interp)
class Scheme(eqx.Module):
    """Configuration only: builds a Plan per pair of geometries."""

    @abc.abstractmethod
    def plan(
        self,
        source: Grid | Points,
        target: Grid | Points,
        *,
        policies: Policies = Policies(),
    ) -> Plan: ...
```

## Types to adhere to: pipekit and geoarray

The package should not invent an operator, carrier, grid or dims vocabulary where pipekit and geoarray already define one, but it may not import either: pipekit is Core tier 0, a peer of functax, and geoarray is GeoStack, above it. So functax follows their conventions without subclassing their types, and each consumer (geoops, xrops, the data-assimilation libraries) wraps functax's plans in its own Operator. This section lists what they define and how each core type above maps onto it. Read from pipekit main (3b9582d) and geotoolz main (1e7308b) on 4 Nov 2026.

### What pipekit defines

pipekit owns how pieces compose, serialise and declare where they run. Its types are carrier-agnostic, but they are still an import, so functax mirrors them rather than using them.

| Type | Package | What it is | The rule it brings |
| --- | --- | --- | --- |
| `Operator`, `ConfigMixin` | pipekit | the base of every pipeline piece; `\|` composes, a graph node when called on an `Input` | implement `_apply`, never `__call__`; keyword-only constructor holding configuration only; `get_config()` derived, strict JSON; `from_state(op.state)` rebuilds an equal operator |
| `backends`, `_terminal`, `forbid_in_yaml` | pipekit | class flags on `Operator` | declare `("numpy", "jax")`, and a carrier on another backend raises `BackendError` before `_apply`; never convert silently |
| `JaxOperator` | pipekit-jax | an `Operator` that is also an `eqx.Module` | array fields are leaves; config fields `eqx.field(static=True)`; learnable fields take `converter=jnp.asarray`; compile with `eqx.filter_jit`; checked by `check_jax_operator` |
| `Signature`, dims patterns | pipekit | named dims and dtype; patterns such as `"band y x"` and `"... y x"` | the one dims vocabulary of the stack; `compute_output_signature` says what an operator does to dims |
| `KernelSpec`, `KernelOp` | pipekit-array | one raw-array kernel with a generalized-ufunc signature, such as `"(c)->()"`, run on every named carrier | core dims last, broadcast over leading dims; non-carrier inputs are keyword parameters; an output-only core dim declares its size |
| `NamedArrayAdapter`, `get_adapter` | pipekit | how carrier-agnostic code reads dims and data and rewraps a result | the package reaches carriers through adapters, never by importing a carrier type into the core |
| `DimArray` | pipekit-array | a named array with no georeference | the result type when a georeferenced dim is consumed |
| `FittableTransformer`, `StatefulOperator` | pipekit | `fit` + `transform`; carry-state operators | learned state lives in trailing-underscore attributes; `transform` is read-only |

### What geoarray defines

geoarray owns the georeferenced carriers and the grid geometry. Its carriers are JAX pytrees whose leaves are the data and the georeference.

| Type | What it is | The rule it brings |
| --- | --- | --- |
| `GeoArray` | data plus an affine transform, CRS, fill value and attrs; stored dims end in `y x` | leaves: `data` and a `(*batch, 6)` float64 transform in (a, b, c, d, e, f) order; static: dims, coords, CRS, fill, attrs. JAX carriers need x64 |
| `GeoSwath` | data on `row col` with per-pixel lat and lon as leaves | the curvilinear source or target |
| `GeoPoints` | data on `obs`; `xy` `(obs, 2)`, optional `z` with a `Vertical`, per-point `time`, and `valid` | static shapes by padding: `pad_points` pads to a fixed N with `valid=False` |
| `Grid` | transform, CRS and `(H, W)` with no data; hashable | a target grid is a value, usable as static config or a dict key |
| `PixelWindow` | a rectangle of pixels in rasterio's field order | windows, not slices, cross API boundaries |
| `geoarray.coords` | `SpectralAxis`, `VariableAxis`, `VerticalAxis`, `TimeAxis`, `PixelGrid`, `SwathGrid`, `CurvilinearGrid`, `PointAxis` | coordinates are frozen and hashable; time is `datetime64[ns]` |
| `to_pixel(transform, xy)` | map coordinates to fractional `(row, col)` | integers fall on pixel centres (area convention); runs under `jit` and `grad` |
| `RasterSource` | a lazy reader: its grid, then `read_window` / `load` | a plan can be built from a grid before any pixel is read |
| Readable types, `check_dims` | `GeoImage` (`"band y x"`), `GeoTracks`, … | annotations name a dims pattern and a geometry kind: grid, swath or points |
| Nodata model | `fill_value`, `valid_mask()` | a pixel is invalid when any band is non-finite or equals the fill |
| Errors | `GridMismatchError`, `GeoArrayTraceError`, `X64Error` | grid checks never skip under a trace; host reads under a trace raise |

### How the core types map onto them

Most core types map onto an existing type; three need a decision. The table names the change each implies for the sketch above.

| Core type | Adheres to | What changes |
| --- | --- | --- |
| `Scheme` | `pipekit.Operator`'s conventions, not the class | a scheme is configuration only: static `eqx.Module` fields that round-trip through JSON. `Linear(outside="nan")` serialises; the consumer's Operator holds those settings as its config |
| `Plan` | `JaxOperator`'s conventions, not the class | the plan is a fitted `eqx.Module` on plain arrays: `__call__(y)`; the stencil's `idx` and `w` are leaves, the geometry static. A separate plan object, because Equinox modules are immutable and `fit` cannot return `self` |
| `Grid` (regular) | `geoarray.Grid` | use its fields and transform order instead of a tuple of axes; rectilinear axes are an extension geoarray does not have |
| `Axis` (rectilinear) | the frozen, hashable style of `geoarray.coords` | keep a coordinate-values axis; datetimes follow `TimeAxis`: `datetime64[ns]`, converted to offsets before tracing |
| `Points` | `GeoPoints` | the same field names: `xy` `(obs, 2)`, separate `z`, `valid`; fixed N by `pad_points`. The metric follows the CRS: geographic means chord or geodesic |
| curvilinear source | `GeoSwath`, `CurvilinearGrid` | lat and lon are leaves; locate inverts them at fit |
| `locate` | `to_pixel` | same convention and maths for affine grids; the package adds rectilinear and curvilinear locate |
| `Stencil.apply` | `KernelSpec` | a gufunc `"(y,x)->(obs)"` with `idx` and `w` as keyword parameters. *Finding:* `KernelSpec.sizes` only maps an output-only dim to an input dim, and `obs` takes its size from a keyword |
| `Policies.missing` | the nodata model | missing means non-finite or equal to `fill_value`, as `valid_mask()` defines |
| `Policies.outside` | `GeoPoints.valid` | an outside target sets `valid=False` and gets the fill |
| batch `*B` | `KernelOp` loop dims; `map_over` | leading dims are the loop dims; a named dim maps with `map_over` |
| shape names | `Signature` patterns | the sketch's jaxtyping names should become the stack's: `y x` for grids, `obs` for points, `row col` for swaths |
| output dims | `compute_output_signature` | grid → points turns `"... y x"` into `"... obs"` |
| NumPy-only schemes | `backends = ("numpy",)` | Clough–Tocher and natural neighbour refuse JAX carriers with `BackendError` |

*Finding:* a `JaxOperator` holding a `Grid` field may not give a JSON `state`: `jsonable` takes builtins and arrays, and `Grid` serialises through its own `get_config`. Keeping the plan carrier-free sidesteps this in functax, since the plan holds no `Grid`; the question moves to the consumer's Operator, which should hold the transform, CRS and shape as plain fields.

### The grid-to-points plan, adhering

The sketch below is the grid → points plan rewritten on these types, split by owner. functax keeps the scheme and the fitted plan, on plain arrays and pixel coordinates, with no carrier and no pipekit type; geoops wraps them in an Operator on carriers. Symbols: T is time, Ny × Nx the source grid with N = Ny·Nx, M the points on `obs` and S the taps per point; A is the 2 × 2 part of the transform and (c, f) its origin.

```python
# Shapes: T times; Ny × Nx source grid; N = Ny·Nx; M points (obs); S = 4 taps

import einx
import equinox as eqx
from jaxtyping import Array, Float


# --- functax: the scheme and the plan, on plain arrays ---


# proposed (functax)
class Linear(eqx.Module):
    """A scheme: configuration only, so its config is strict JSON."""

    outside: str = eqx.field(static=True, default="nan")
    boundary: str = eqx.field(static=True, default="mirror")

    def plan(
        self, shape: tuple[int, int], pixel_rowcol: Float[Array, "obs 2"]
    ) -> "GridToPoints":
        # proposed (functax)
        # Hₖₛ: 4 taps per point, bilinear weights
        stencil = tensor_stencil_at(  # → Stencil (M, S)
            shape, pixel_rowcol, scheme=self
        )
        return GridToPoints(stencil=stencil)


# proposed (functax)
class GridToPoints(eqx.Module):
    """A fitted plan: the grid read at fixed points, f̂(X*) = H y."""

    stencil: Stencil

    def __call__(self, field: Float[Array, "*batch y x"]) -> Float[Array, "*batch obs"]:
        # y ∈ ℝ^(T × N)
        flat_field = einx.rearrange(  # (T, Ny, Nx) → (T, N)
            "... y x -> ... (y x)", field
        )
        # f̂(X*) = H y
        values = self.stencil.apply(flat_field)  # (T, N) → (T, M)
        return values
```

```python
import functax as fx
import geoarray as ga
import pipekit


# --- geoops: the Operator on carriers, wrapping functax ---


# proposed (geoops)
class SampleAtPoints(pipekit.Operator):
    """A grid read at points: unwrap, run functax's plan, rewrap as points."""

    backends = ("numpy", "jax")

    def __init__(self, *, outside: str = "nan", boundary: str = "mirror") -> None:
        self.outside = outside
        self.boundary = boundary

    def _apply(self, field: ga.GeoArray, points: ga.GeoPoints) -> ga.GeoPoints:
        # today (geoarray)
        # (row, col) = A⁻¹ (x − c, y − f) − ½
        pixel_rowcol = ga.to_pixel(field.transform, points.xy)  # (M, 2) → (M, 2)
        # proposed (functax)
        scheme = fx.interp.Linear(outside=self.outside, boundary=self.boundary)
        # proposed (functax)
        plan = scheme.plan(field.shape[-2:], pixel_rowcol)  # → plan (M, S)
        # proposed (functax)
        # f̂(X*) = H y, on the bare array: .data keeps the backend, so a JAX
        # carrier stays a JAX array (np.asarray would fail on a tracer)
        values = plan(field.data)  # (T, Ny, Nx) → (T, M)
        # today (geoarray)
        return ga.GeoPoints(  # → (T, M) points
            values,
            xy=points.xy,
            valid=plan.stencil.valid,
            dims="time obs",
            crs=field.crs,
        )
```

A pipeline that samples the same points every step builds the plan once: geoops can cache it on the grid and the points (pipekit's `Memoize`), or a model holds `GridToPoints` directly, since it is a plain `eqx.Module`.

The gap under the block: `tensor_stencil_at`, `Stencil` and both classes are proposed; the plan does not yet check that a field is on its grid; and a field with dims other than `time y x` needs its dims carried through rather than fixed as `"time obs"`.

## Algorithms: patterns, catalogue and JAX strategies

Every algorithm in scope answers three questions, and a handful of answers to each covers the whole catalogue. This section names those patterns, then lists the schemes by task, then gives the strategies that make each pattern traceable.

### Three questions: find, weigh, solve

The distinctions you name (slicing, basis, solving, neighbours) are real, but they are not rival families. They answer different questions, and most schemes combine one answer to each.

```text
   sources X, data y          targets X*
            │                      │
            │            ① find    which sources touch each target?
            │                      index · search · invert · neighbours
            │                      triangulate · overlap · all
            │                      │
            │            ② weigh   with what weights?
            │                      indicator · polynomial · distance
            │                      barycentric · area · kernel · data-dependent
            │                      │
            ▼                      ▼
   ③ solve  c = 𝒮(y)        H, M × J, from ① and ②
            none · banded · dense · iterative
            local · optimise · spectral
            │                      │
            └──────────┬───────────┘
                       ▼
                 f̂(X*) = H c
```

- **Find** decides the sparsity of H: which entries are non-zero. On a grid it is arithmetic; on scattered data it is a search.
- **Weigh** fills those entries. It depends only on geometry, except for data-dependent weights.
- **Solve** turns data into coefficients. It is the identity for most local schemes and the whole cost for global ones.

Find and weigh depend on the targets; solve depends only on the sources. That is why a plan can solve once and read many target sets, or locate once and read many fields.

### The patterns

Eleven patterns cover the catalogue. Each row is one combination of the three answers.

| # | Pattern | Find | Weigh | Solve | Examples |
| --- | --- | --- | --- | --- | --- |
| 1 | Index and blend | index or search | indicator or polynomial | none | nearest, linear, cubic convolution, Lanczos |
| 2 | Prefiltered basis | index or search | polynomial (B-spline) | banded | B-splines, cubic splines |
| 3 | Data-dependent local | index or search | data-dependent | slopes from y | PCHIP, Akima, Steffen |
| 4 | Neighbourhood | neighbours | distance or kernel | none, or local | inverse distance, local RBF, moving-window kriging |
| 5 | Global kernel | all | kernel | dense or iterative | RBF, thin-plate spline, kriging, GP |
| 6 | Triangulation | triangulate | barycentric or cubic patch | none, or gradient fit | Delaunay linear, Clough–Tocher, natural neighbour |
| 7 | Aggregation | index each source | indicator | none | binning, histograms, block means |
| 8 | Overlap | overlap | area | none | first- and second-order conservative remapping |
| 9 | Iterative fill | index (grid) | difference stencil | iterative | Laplace fill, diffusion fill, variational analysis |
| 10 | Spectral | all (transform) | Fourier or harmonic | diagonal in the transform | zero-padding, sinc, spherical harmonics |
| 11 | Learned | index or all | network | optimise | neural fields, hash-grid encodings |

The patterns nest. A B-spline is pattern 1 run on prefiltered coefficients. Moving-window kriging is pattern 4 with a small solve per target. A hash-grid neural field runs pattern 1 inside its encoding.

Each pattern below closes with a JAX sketch and its supporting libraries. This table is the summary; "oracle" marks a library used only to test against.

| # | Pattern | JAX core | Stack libraries | Outside the stack |
| --- | --- | --- | --- | --- |
| 1 | Index and blend | gather, `einx.dot` | geoarray `to_pixel`; geoops `bilinear_stencil`; gaussx `cubic_interpolation_weights` | `jax.scipy.ndimage.map_coordinates` (orders 0–1); SciPy, interpax (oracles) |
| 2 | Prefiltered basis | `lax.scan`, `tridiagonal_solve`, FFT | spectraldiffx (periodic solves) | interpax; SciPy splines (oracle) |
| 3 | Data-dependent local | `jnp.where` with guards | — | interpax; SciPy PCHIP and Akima (oracles) |
| 4 | Neighbourhood | fixed-k gather; chunked brute force | kernellib `tapered_operator`; gaussx `localization_matrix` | SciPy `cKDTree` (fit) |
| 5 | Global kernel | lineax Cholesky or CG | gaussx solvers, Nyström, GP prediction, SKI; kernellib kernels and variograms | SciPy `RBFInterpolator` (oracle) |
| 6 | Triangulation | 3-tap stencil | gaussx `fem_projector` | SciPy `Delaunay` (fit); SciPy linear and Clough–Tocher (oracles) |
| 7 | Aggregation | `jax.ops.segment_sum` | plumax `bin_positions`; geoops binning | SciPy `binned_statistic_2d` (oracle) |
| 8 | Overlap | padded stencil or `segment_sum` | xrtoolz `overlap_weights_1d` (reference); gaussx `SparseOperator` | Shapely (fit, polygons); xESMF (oracle) |
| 9 | Iterative fill | lineax CG, matrix-free | finitevolX Laplacians, CG, preconditioners; gaussx SPDE precisions; vardax OI | DIVAnd (oracle) |
| 10 | Spectral | `jnp.fft` | spectraldiffx transforms; geonnax `band_limited_resize`, `nonuniform_irfft` | s2fft, jax-finufft (to check) |
| 11 | Learned | Equinox, optax | geonnax layers and bases; pipekit-train | — |

#### 1. Index and blend

The target's cell comes from arithmetic, its taps from fixed offsets, and its weights from a polynomial in the fractional offset t. It is the workhorse: grid → points, grid → grid and every tensor-product scheme.

- **Fit:** locate each target, then compute S = tapsᵈ weights. Cost O(M S).
- **Apply:** a gather and a contraction, O(M S) per field.
- **Boundary:** mirror, wrap and edge are index remaps before the gather; constant is a mask.
- **JAX:** fully traceable. Gradients reach x\* through t, and the cell index is piecewise constant.

The 1-D linear case shows every ingredient. Symbols: x₀ is the axis origin, Δ its spacing, N the number of samples, u the target position in index units, i its cell and t its offset in the cell.

```python
# Shapes: N source samples; M targets; S = 2 taps

# x − x₀
shifted = target_positions - origin  # (M,) → (M,)
# u = (x − x₀) / Δ
index_position = shifted / spacing  # (M,) → (M,)
# ⌊u⌋
lower = jnp.floor(index_position)  # (M,) → (M,)
# i = ⌊u⌋, as an integer index
cell = lower.astype(jnp.int32)  # (M,) → (M,)
# t = u − i
offset = index_position - lower  # (M,) → (M,)
# N − 1, the last valid index
last_index = n_source - 1  # → ()
# u ≥ 0
after_start = index_position >= 0  # (M,) → (M,)
# u ≤ N − 1
before_end = index_position <= last_index  # (M,) → (M,)
# inside ⇔ 0 ≤ u ≤ N − 1
inside = after_start & before_end  # (M,) → (M,)
# taps i, i + 1
raw_taps = einx.add("target, tap -> target tap", cell, jnp.arange(2))  # → (M, 2)
# proposed (interp)
# j ↦ reflect(j) into [0, N − 1]
taps = mirror(raw_taps, n_source)  # (M, 2) → (M, 2)
# 1 − t
lower_weight = 1 - offset  # (M,) → (M,)
# w = (1 − t, t)
weights = einx.rearrange(  # (M,) → (M, 2)
    "target, target -> target (1 + 1)", lower_weight, offset
)
```

The explicit `inside` matters: JAX clamps out-of-bounds gathers silently, so an unmasked target off the grid reads the edge value.

**Supporting libraries.** geoarray's `to_pixel` already does the locate step for affine grids, and geoops' `bilinear_stencil` is this pattern's seed. gaussx's `cubic_interpolation_weights` gives Keys cubic weights in the same `(idx, w)` layout. `jax.scipy.ndimage.map_coordinates` traces but stops at order 1. SciPy's `map_coordinates`, `RegularGridInterpolator` and interpax are the test oracles.

#### 2. Prefiltered basis

A B-spline of order k does not pass through the samples, so the coefficients come from a solve first. After that, the read-out is pattern 1 with (k + 1) taps per axis.

- **Fit:** a banded solve per axis, separable on a grid. Cubic splines need a tridiagonal solve.
- **Apply:** pattern 1 on the coefficients.
- **Boundary:** the boundary condition (natural, clamped, not-a-knot, periodic) changes the banded system, not the read-out.
- **JAX:** a recursive causal and anticausal filter in `lax.scan` (Unser's poles), a tridiagonal solve (`lax.linalg.tridiagonal_solve`), or an FFT solve when periodic. All are linear, so the transpose comes free.

The cubic B-spline prefilter on one axis, as two recursive passes. Symbols: z is the filter's pole, c⁺ and c⁻ the causal and anticausal passes, and λ the gain.

```python
# Shapes: N samples along one axis

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

Scalar = Float[Array, ""]

# √3
root_three = jnp.sqrt(3.0)  # → ()
# z = √3 − 2, the cubic B-spline pole
pole = root_three - 2.0  # → ()


def causal_step(previous: Scalar, sample: Scalar) -> tuple[Scalar, Scalar]:
    # z c⁺ₖ₋₁
    carried = pole * previous
    # c⁺ₖ = yₖ + z c⁺ₖ₋₁
    current = sample + carried
    return current, current


def anticausal_step(following: Scalar, causal_value: Scalar) -> tuple[Scalar, Scalar]:
    # c⁻ₖ₊₁ − c⁺ₖ
    difference = following - causal_value
    # c⁻ₖ = z (c⁻ₖ₊₁ − c⁺ₖ)
    current = pole * difference
    return current, current


# proposed (interp)
# c⁺₋₁ from the mirror boundary condition
causal_start = mirror_causal_start(samples, pole)  # (N,) → ()
# c⁺ₖ = yₖ + z c⁺ₖ₋₁, left to right
_, causal = jax.lax.scan(causal_step, causal_start, samples)  # (N,) → (N,)
# proposed (interp)
# c⁻_N from the mirror boundary condition
anticausal_start = mirror_anticausal_start(causal, pole)  # (N,) → ()
# c⁻ₖ = z (c⁻ₖ₊₁ − c⁺ₖ), right to left
_, anticausal = jax.lax.scan(  # (N,) → (N,)
    anticausal_step, anticausal_start, causal, reverse=True
)
# c = λ c⁻,  λ = (1 − z)(1 − z⁻¹) = 6
coefficients = 6.0 * anticausal  # (N,) → (N,)
```

Both passes are linear in y, so `jax.linear_transpose` gives the prefilter's adjoint without extra code. Higher orders add one pole, and one pair of passes, per extra degree of two.

**Supporting libraries.** `lax.linalg.tridiagonal_solve` covers 1-D cubic splines with any end condition, and spectraldiffx's FFT solvers cover periodic axes. interpax has JAX splines on grids up to 3-D. SciPy's `spline_filter` and `make_interp_spline` are the oracles; @unser1999splines gives the poles and the boundary sums.

#### 3. Data-dependent local

Monotone Hermite schemes choose each slope from neighbouring differences of y. They avoid overshoot, which matters for concentrations and probabilities.

- **Fit:** slopes from y, elementwise, with branches for sign changes.
- **Apply:** a 4-tap Hermite cubic, with weights that depend on the slopes.
- **JAX:** traceable with `jnp.where`; `grad` works. There is no fixed Hᵀ, because the map is nonlinear in y. Guard divisions in untaken branches (below).

Interior PCHIP slopes on a uniform axis. Symbols: h is the spacing, δₖ the secant slope of interval k, and mₖ the slope at knot k.

```python
# Shapes: N samples on a uniform axis; N − 1 intervals; N − 2 interior knots

import jax.numpy as jnp

# Δyₖ = yₖ₊₁ − yₖ
rises = jnp.diff(values)  # (N,) → (N − 1,)
# δₖ = Δyₖ / h
secants = rises / spacing  # (N − 1,) → (N − 1,)
# δₖ₋₁ and δₖ around each interior knot
left = secants[:-1]  # → (N − 2,)
right = secants[1:]  # → (N − 2,)
# δₖ₋₁ δₖ
product = left * right  # (N − 2,) → (N − 2,)
# δₖ₋₁ δₖ > 0: no sign change, so a monotone slope exists
same_sign = product > 0  # (N − 2,) → (N − 2,)
# δₖ₋₁ + δₖ
total = left + right  # (N − 2,) → (N − 2,)
# 1 where the branch is not taken, so the division below stays finite
safe_total = jnp.where(same_sign, total, 1.0)  # (N − 2,) → (N − 2,)
# 2 δₖ₋₁ δₖ
doubled = 2.0 * product  # (N − 2,) → (N − 2,)
# 2 δₖ₋₁ δₖ / (δₖ₋₁ + δₖ), the harmonic mean
harmonic = doubled / safe_total  # (N − 2,) → (N − 2,)
# mₖ = harmonic mean, or 0 at a sign change (Fritsch–Carlson)
slopes = jnp.where(same_sign, harmonic, 0.0)  # (N − 2,) → (N − 2,)
```

The end slopes use one-sided formulas, left out here. Non-uniform axes weight the harmonic mean by the interval widths.

**Supporting libraries.** No stack library has a monotone scheme today; plumax's `lookup_nb` is the first consumer. interpax has monotonic and Akima options in JAX. SciPy's `PchipInterpolator` and `Akima1DInterpolator` are the oracles.

#### 4. Neighbourhood

On scattered data the support comes from a search, not arithmetic. Once the k nearest sources are known, the result is a stencil like pattern 1.

- **Fit:** a k-d tree query, O(M log N), or brute force, O(M N).
- **Apply:** gather and weigh, O(M k). Moving-window kriging adds a k × k solve per target, batched.
- **JAX:** the tree runs at fit on the host, and only fixed-k index arrays enter the trace. When the targets themselves are traced, use brute force in chunks (`lax.map`) or a bucket grid with fixed capacity. The neighbour set is piecewise constant in x\*, so gradients only see the weights.

The fit for inverse distance, as a sketch. Symbols: dᵢⱼ is the distance from target i to its j-th nearest source, iᵢⱼ that source's index, p the power, ε a floor that avoids division by zero, and Q the neighbour count.

```python
# Shapes: N scattered sources; M targets; Q neighbours

import einx
import numpy as np
import scipy.spatial

# not pinned (scipy)
tree = scipy.spatial.cKDTree(source_xy)  # (N, 2)
# not pinned (scipy)
# dᵢⱼ, iᵢⱼ: the Q nearest sources of target i, on the host
distances, indices = tree.query(target_xy, k=n_neighbours)  # → (M, Q), (M, Q)
# max(dᵢⱼ, ε)
floored = np.maximum(distances, floor)  # (M, Q) → (M, Q)
# vᵢⱼ = max(dᵢⱼ, ε)⁻ᵖ
inverse_distance = floored**-power  # (M, Q) → (M, Q)
# ∑ⱼ vᵢⱼ
total = einx.sum("target [neighbour]", inverse_distance)  # (M, Q) → (M,)
# wᵢⱼ = vᵢⱼ / ∑ⱼ vᵢⱼ
weights = einx.divide(  # (M, Q) → (M, Q)
    "target neighbour, target -> target neighbour", inverse_distance, total
)
# with a search radius, cKDTree reports a missing neighbour as dᵢⱼ = ∞
finite = np.isfinite(distances)  # (M, Q) → (M, Q)
valid = einx.all("target [neighbour]", finite)  # (M, Q) → (M,)
# proposed (interp)
stencil = Stencil(indices=indices, weights=weights, valid=valid, n_source=n_sources)
```

**Supporting libraries.** SciPy's `cKDTree` does the search at fit, as geoops' `GapFillIDW` and `RasterToPointCloud` already do. kernellib's `tapered_operator` builds compact kernels over a fixed neighbour count, and gaussx's `localization_matrix` and `gaspari_cohn` give localising weights. xrtoolz's `idw_to_grid` and `idw_to_points` are the oracles.

#### 5. Global kernel

Every source touches every target through a kernel k(x, x′). The solve dominates, and it is where the uncertainty comes from.

- **Fit:** (K + λI) c = y. Dense Cholesky is O(N³); kriging with a drift adds a saddle-point block.
- **Apply:** K₍\*₎ c, O(M N); the variance needs one more solve per target.
- **JAX:** Cholesky for small N, in float64 with jitter. For large N: CG with a matrix-free, chunked kernel matvec; compact-support kernels for sparsity; inducing points (Nyström); or structured algebra, such as Kronecker products on grids. Differentiating through the solve uses the implicit function theorem, which lineax provides.

An RBF fit and read-out with a dense Cholesky. Symbols: K is the kernel matrix on the sources, λ the nugget, and K\* the cross-kernel from targets to sources.

```python
# Shapes: N sources; M targets; D = 2 dims

import einx
import jax.numpy as jnp
import lineax as lx

# proposed (interp)
# Kᵢⱼ = k(xᵢ, xⱼ)
gram = kernel(source_xy, source_xy)  # (N, 2) → (N, N)
# I
identity = jnp.eye(n_sources, dtype=gram.dtype)  # → (N, N)
# λI
loading = nugget * identity  # (N, N) → (N, N)
# K + λI
loaded = gram + loading  # (N, N) → (N, N)
# not pinned (lineax)
gram_operator = lx.MatrixLinearOperator(loaded, lx.positive_semidefinite_tag)
# not pinned (lineax)
# c = (K + λI)⁻¹ y
solution = lx.linear_solve(gram_operator, values, lx.Cholesky())  # (N,) → (N,)
coefficients = solution.value  # → (N,)
# proposed (interp)
# K*ᵢⱼ = k(x*ᵢ, xⱼ)
cross = kernel(target_xy, source_xy)  # (M, 2) → (M, N)
# f̂(X*) = K* c
predictions = einx.dot(  # (M, N) → (M,)
    "target source, source -> target", cross, coefficients
)
```

Past a few thousand sources, the dense Gram matrix gives way to a matrix-free operator and CG, with the same two calls around it.

**Supporting libraries.** gaussx carries most of this pattern: `linear_solve` with CG and BBMM strategies, `randomized_nystrom`, `predict_mean` and `predict_variance`, `kronecker_posterior_predictive` on grids, and `InterpolatedOperator` for SKI with a stencil's arrays. kernellib supplies the kernel side: `to_operator`, `to_cross_operator`, `nystrom_operator`, `KRR`, `falkon_solve` and `fit_variogram`. kernellib sits a tier above functax (Core tier 1), so functax takes a kernel callable or a lineax K, and the kernellib adapter lives in kernellib. SciPy's `RBFInterpolator` is the oracle.

#### 6. Triangulation

A Delaunay triangulation gives every target a containing triangle. Barycentric weights then give a 3-tap stencil.

- **Fit:** triangulation and point location, on the host.
- **Apply:** linear on triangles is pattern 1 with 3 taps, so it traces. Clough–Tocher needs gradient estimates at the vertices, another solve, then a cubic patch per triangle.
- **JAX:** linear only. Natural neighbour needs Voronoi areas that change with the target, so it stays NumPy.

Linear interpolation on a Delaunay triangulation, fitted on the host. Symbols: τₘ is target m's triangle, Tₘ⁻¹ and rₘ its inverse affine map and reference vertex, and b the barycentric weights.

```python
# Shapes: N sources; M targets; S = 3 taps (the triangle's vertices)

import einx
import scipy.spatial

# not pinned (scipy)
triangulation = scipy.spatial.Delaunay(source_xy)  # (N, 2)
# not pinned (scipy)
# τₘ: the triangle that contains target m; −1 outside the hull
simplex = triangulation.find_simplex(target_xy)  # (M, 2) → (M,)
# Tₘ⁻¹ and rₘ for each target's triangle, as scipy stores them
affine = triangulation.transform[simplex]  # → (M, 3, 2)
inverse = affine[:, :2, :]  # → (M, 2, 2)
reference = affine[:, 2, :]  # → (M, 2)
# x − rₘ
shifted = target_xy - reference  # (M, 2) → (M, 2)
# (b₁, b₂) = Tₘ⁻¹ (x − rₘ)
partial = einx.dot(  # (M, 2, 2) → (M, 2)
    "target i j, target j -> target i", inverse, shifted
)
# b₁ + b₂
partial_sum = einx.sum("target [i]", partial)  # (M, 2) → (M,)
# b₃ = 1 − b₁ − b₂
last = 1.0 - partial_sum  # (M,) → (M,)
# w = (b₁, b₂, b₃)
weights = einx.rearrange(  # → (M, 3)
    "target i, target -> target (i + 1)", partial, last
)
# the vertex indices of each target's triangle
indices = triangulation.simplices[simplex]  # → (M, 3)
# inside the convex hull
valid = simplex >= 0  # (M,) → (M,)
# outside: simplex = −1 picked the last triangle, so zero its weights and point
# its taps at vertex 0; the plan writes the outside fill there
weights = einx.where(  # (M, 3) → (M, 3)
    "target, target tap, -> target tap", valid, weights, 0.0
)
indices = einx.where(  # (M, 3) → (M, 3)
    "target, target tap, -> target tap", valid, indices, 0
)
# proposed (interp)
stencil = Stencil(indices=indices, weights=weights, valid=valid, n_source=n_sources)
```

Everything above runs once on the host; the stencil it returns applies under `jit` like any other.

**Supporting libraries.** gaussx's `fem_projector` builds the same barycentric H on planar and spherical meshes, and geonnax's `icosphere` gives spherical meshes to project onto. SciPy's `Delaunay` does the fit; its `LinearNDInterpolator` and `CloughTocher2DInterpolator` are the oracles.

#### 7. Aggregation

Aggregation reverses the direction: many sources fall into each target cell. It is the adjoint of nearest sampling, normalised by the count.

```{math}
\bar y = \mathrm{diag}(H^\top \mathbf{1})^{-1} H^\top y
```

Here H is the nearest-neighbour stencil from the targets' grid to the source points, 1 a vector of ones, and ȳ the per-cell mean.

- **JAX:** `jax.ops.segment_sum` for sums, counts and means. Medians are not segment reductions; they need a sort by cell.

Mean binning as an adjoint. Symbols: jₙ is sample n's cell, nⱼ the count in cell j, and ȳⱼ its mean.

```python
# Shapes: N samples; M cells

import jax
import jax.numpy as jnp

# proposed (interp)
# jₙ: the cell that holds sample n, a one-tap nearest stencil
cell = nearest_cell(grid, sample_xy)  # (N, 2) → (N,)
# ∑ yₙ over each cell: Hᵀy
sums = jax.ops.segment_sum(sample_values, cell, num_segments=n_cells)  # → (M,)
# 1 per sample
ones = jnp.ones_like(sample_values)  # (N,) → (N,)
# nⱼ = Hᵀ1, the samples per cell
counts = jax.ops.segment_sum(ones, cell, num_segments=n_cells)  # → (M,)
# nⱼ > 0
occupied = counts > 0  # (M,) → (M,)
# 1 in empty cells, so the division below stays finite
safe_counts = jnp.where(occupied, counts, 1.0)  # (M,) → (M,)
# ȳⱼ = Hᵀy / nⱼ
ratio = sums / safe_counts  # (M,) → (M,)
# NaN in empty cells
means = jnp.where(occupied, ratio, jnp.nan)  # (M,) → (M,)
```

Swapping the one-tap stencil for a linear one gives cloud-in-cell gridding, with gradients in the sample positions.

**Supporting libraries.** plumax's `bin_positions` does the nearest version today, and geoops has `SpectralBinning` and `PointCloudToRaster(method="binned_stat")`. SciPy's `binned_statistic_2d` is the oracle.

#### 8. Overlap

Conservative remapping weighs each source cell by its overlap area with the target cell. It preserves integrals, which fluxes and emissions need.

- **Fit:** polygon clipping between source and target cells, on the host; geodesic on the sphere.
- **Apply:** a sparse matvec. Taps vary per target, so pad to the largest count with `valid`, or use `segment_sum` over a flat list.
- **JAX:** apply only. Second order adds a gradient reconstruction, which is another stencil.

First-order conservative weights on one axis; a 2-D grid takes the tensor product. Symbols: \[eⱼ, eⱼ₊₁\] is source cell cⱼ, \[Eᵢ, Eᵢ₊₁\] target cell Cᵢ, and wᵢⱼ the fraction of Cᵢ that cⱼ covers.

```python
# Shapes: N source cells; M target cells; S most overlaps per target

import einx
import numpy as np

# ℓᵢⱼ = max(eⱼ, Eᵢ), the overlap's lower edge
lower = einx.maximum(  # → (M, N)
    "source, target -> target source", source_lower, target_lower
)
# uᵢⱼ = min(eⱼ₊₁, Eᵢ₊₁), the overlap's upper edge
upper = einx.minimum(  # → (M, N)
    "source, target -> target source", source_upper, target_upper
)
# uᵢⱼ − ℓᵢⱼ
length = upper - lower  # (M, N) → (M, N)
# |cⱼ ∩ Cᵢ| = max(0, uᵢⱼ − ℓᵢⱼ)
overlap = np.maximum(length, 0.0)  # (M, N) → (M, N)
# |Cᵢ| = Eᵢ₊₁ − Eᵢ
width = target_upper - target_lower  # (M,) → (M,)
# wᵢⱼ = |cⱼ ∩ Cᵢ| / |Cᵢ|
weights = einx.divide(  # (M, N) → (M, N)
    "target source, target -> target source", overlap, width
)
# proposed (interp)
# keep each row's non-zero entries, padded to S: a stencil
stencil = Stencil.from_dense(weights, max_taps=max_overlaps)  # → (M, S)
```

The dense (M, N) matrix exists only on one axis and only at fit; the stencil keeps a few taps per row.

**Supporting libraries.** xrtoolz's `overlap_weights_1d` and `regrid_conservative` are the reference for this sketch. Shapely clips polygons for curvilinear and spherical cells at fit. gaussx's `SparseOperator` is the export target for fixed plans, and xESMF is the oracle.

#### 9. Iterative fill

Gaps on a grid are filled by solving an equation in the holes, with the known pixels as boundary values. Laplace fill and variational analysis both have this shape.

```{math}
(P^\top R^{-1} P + \lambda L^\top L) x = P^\top R^{-1} y
```

Here P selects the observed pixels, R is their error covariance, L a difference operator and λ its weight. With λ large and R small, this tends to the Laplace fill.

- **JAX:** Gauss–Seidel is sequential, so use Jacobi sweeps, or better CG with matrix-free P and L. Fix the iteration cap, or use a `lax.while_loop` with a tolerance. Gradients go through the solve implicitly.

Variational gap filling with matrix-free CG. Symbols: P selects the observed pixels, R = σ²I is their error covariance, L a Laplacian and λ its weight.

```python
# Shapes: Ny × Nx grid

import jax
import jax.numpy as jnp
import lineax as lx
from jaxtyping import Array, Float


def normal_matvec(field: Float[Array, "y x"]) -> Float[Array, "y x"]:
    # P x: the observed pixels, zero elsewhere
    observed = jnp.where(observed_mask, field, 0.0)  # (Ny, Nx) → (Ny, Nx)
    # PᵀR⁻¹P x, with R = σ² I
    data_term = observed / noise_variance  # (Ny, Nx) → (Ny, Nx)
    # not pinned (finitevolX)
    # L x, a five-point Laplacian with mirror edges
    curvature = laplacian(field)  # (Ny, Nx) → (Ny, Nx)
    # not pinned (finitevolX)
    # LᵀL x, since L is symmetric
    smoothness = laplacian(curvature)  # (Ny, Nx) → (Ny, Nx)
    # λLᵀL x
    penalty = smoothing_weight * smoothness  # (Ny, Nx) → (Ny, Nx)
    # A x = (PᵀR⁻¹P + λLᵀL) x
    normal = data_term + penalty  # (Ny, Nx) → (Ny, Nx)
    return normal


structure = jax.ShapeDtypeStruct(observations.shape, observations.dtype)
# not pinned (lineax)
# A, matrix-free and symmetric positive semi-definite
normal_operator = lx.FunctionLinearOperator(
    normal_matvec, structure, lx.positive_semidefinite_tag
)
# P y: the observations, zero in the gaps
masked = jnp.where(observed_mask, observations, 0.0)  # (Ny, Nx) → (Ny, Nx)
# b = PᵀR⁻¹y
rhs = masked / noise_variance  # (Ny, Nx) → (Ny, Nx)
# not pinned (lineax)
solver = lx.CG(rtol=tolerance, atol=tolerance, max_steps=max_iterations)
# not pinned (lineax)
# x = A⁻¹ b
solution = lx.linear_solve(normal_operator, rhs, solver)  # (Ny, Nx) → (Ny, Nx)
filled = solution.value  # → (Ny, Nx)
```

lineax differentiates the solve implicitly, so gradients with respect to the observations or λ cost one more solve, not a pass through every iteration.

**Supporting libraries.** finitevolX supplies L (`Difference2D.laplacian`, `masked_laplacian`), `solve_cg` and spectral, Nyström and multigrid preconditioners. gaussx's `spde_precision_grid` gives a Matérn-type LᵀL, and its `MaskedOperator` handles the selection P. spectraldiffx's Helmholtz solvers fit the periodic case directly, and vardax's `OptimalInterpolation` is the same solve in observation space. geoops' `GapFillLaplacian`, xrtoolz's `fillnan_laplacian` and DIVAnd are the oracles.

#### 10. Spectral

A band-limited field is interpolated exactly in its transform basis: zero-pad the spectrum and invert. On the sphere, the basis is spherical harmonics.

- **JAX:** FFTs trace and transpose. The pattern assumes periodicity or band limits, and rings near discontinuities (Gibbs).

Periodic upsampling by zero-padding. Symbols: F is the discrete Fourier transform, ŷ the spectrum of y, and r the refinement factor.

```python
# Shapes: N periodic samples; M = r·N targets for a refinement factor r

import jax.numpy as jnp

# ŷ = F y
spectrum = jnp.fft.rfft(values)  # (N,) → (N/2 + 1,)
# M/2 + 1, the modes of the longer signal
n_modes = n_targets // 2 + 1  # → ()
# the zero modes to append
n_padding = n_modes - spectrum.shape[0]  # → ()
# ŷ padded with zeros above the Nyquist frequency of y
padded = jnp.pad(spectrum, (0, n_padding))  # → (M/2 + 1,)
# F⁻¹ ŷ on M points
upsampled = jnp.fft.irfft(padded, n=n_targets)  # (M/2 + 1,) → (M,)
# M / N, because the inverse transform divides by M, not N
scale = n_targets / n_sources  # → ()
# f̂ = (M / N) F⁻¹ ŷ
interpolated = scale * upsampled  # (M,) → (M,)
```

For even N the Nyquist mode must be halved before padding, or the result picks up a small oscillation. Scattered targets need a non-uniform transform instead of padding.

**Supporting libraries.** spectraldiffx has the transforms (`FourierGrid`, `ChebyshevTransform1D/2D`, `SphericalHarmonicTransform`) but no evaluation at arbitrary points. geonnax's `band_limited_resize` is this sketch in 2-D, and its `nonuniform_irfft` evaluates a series at scattered points. Outside the stack, s2fft (spherical harmonics) and jax-finufft (non-uniform FFTs) are JAX options, still to check.

#### 11. Learned

A neural field fits a network to the samples by optimisation. Its encodings, such as hash grids, are pattern 1 inside the network.

- **JAX:** native; the fit is an optimiser loop. There is no H, only `grad`, so it sits outside the linear-operator API.

A neural-field fit with optax. Symbols: θ are the network's parameters, and ℒ the mean squared misfit.

```python
# Shapes: N samples; D = 2 dims

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

# user
field = FourierFeatureMLP(in_size=2, width=256, depth=4, key=model_key)
# not pinned (optax)
optimiser = optax.adam(learning_rate)
parameters = eqx.filter(field, eqx.is_array)
# not pinned (optax)
optimiser_state = optimiser.init(parameters)


def loss(model: eqx.Module) -> Float[Array, ""]:
    # f̂ evaluated at every sample
    batched = jax.vmap(model)
    # f̂(xₙ)
    predictions = batched(source_xy)  # (N, 2) → (N,)
    # rₙ = f̂(xₙ) − yₙ
    residual = predictions - values  # (N,) → (N,)
    # rₙ²
    squared = residual**2  # (N,) → (N,)
    # ℒ = ½ mean rₙ²
    half_mean = 0.5 * jnp.mean(squared)  # → ()
    return half_mean


@eqx.filter_jit
def step(
    model: eqx.Module, state: optax.OptState
) -> tuple[eqx.Module, optax.OptState, Float[Array, ""]]:
    loss_and_gradient = eqx.filter_value_and_grad(loss)
    # ℒ and ∇ℒ
    value, gradients = loss_and_gradient(model)
    # not pinned (optax)
    updates, state = optimiser.update(gradients, state)
    # θ ← θ + Δθ
    model = eqx.apply_updates(model, updates)
    return model, state, value
```

The fitted network is the read-out: there is no H to transpose, so this pattern offers values and gradients only.

**Supporting libraries.** geonnax has the layers and the Fourier and RBF bases that encodings need, and pipekit-train runs the loop with callbacks. Equinox and optax are the only outside dependencies.

### JAX strategies across patterns

Nine strategies cover every pattern above. Most follow from one rule: decide shapes at fit, never at apply.

1. **Split fit and apply.** Trees, triangulations and polygon clipping run on the host at fit; apply only gathers, contracts and solves.
2. **Fix every size.** k neighbours, S taps, the most overlaps and the iteration cap are static; pad and mark padding in `valid`.
3. **Mask, never drop.** Invalid targets keep their slot with a fill, so shapes stay static and `vmap` works.
4. **Compute `inside` explicitly.** JAX clamps out-of-bounds indices without an error.
5. **Guard untaken branches.** `jnp.where` differentiates both branches, so a NaN in the unused one poisons the gradient. Use the double `where`:

```python
# Shapes: M targets; N sources

# r > 0
positive = distance > 0  # (M, N) → (M, N)
# 1 where r = 0, so the untaken branch below stays finite
safe_distance = jnp.where(positive, distance, 1.0)  # (M, N) → (M, N)
# log r
log_distance = jnp.log(safe_distance)  # (M, N) → (M, N)
# r²
squared = safe_distance**2  # (M, N) → (M, N)
# r² log r
thin_plate = squared * log_distance  # (M, N) → (M, N)
# φ(r) = r² log r,  φ(0) = 0
kernel_value = jnp.where(positive, thin_plate, 0.0)  # (M, N) → (M, N)
```

6. **Adjoints by scatter-add.** `.at[].add` and `segment_sum` give Hᵀ; `jax.linear_transpose` checks them, and `custom_vjp` can route gradients through the hand-written adjoint.
7. **Implicit solves.** Differentiate a fit through the solve (lineax), not through its iterations.
8. **Chunk dense work.** Never form M × N when a matvec will do; map over target chunks with `lax.map`.
9. **Two plan modes.** Fixed targets precompute the stencil as leaves; traced targets (optimising sensor positions, say) locate inside the trace.

Precision is the caller's choice: weights and coefficients keep the dtype of the coordinates and the data, and nothing casts silently. Transforms are float64 by geoarray's rule, and kernel solves usually need float64 too.

### Catalogue by task

| Task | Schemes | Fit |
| --- | --- | --- |
| 1-D series | linear; cubic spline (natural, clamped, not-a-knot, periodic); PCHIP; Akima and modified Akima; Steffen | identity; tridiagonal; nonlinear slopes |
| Grid → points | nearest; linear to quadrilinear; cubic convolution; B-splines of order 0–5; Lanczos windowed sinc | identity; banded prefilter for B-splines |
| Mixed axes | a basis per axis, such as nearest in time and linear in space | as the per-axis bases |
| Grid → grid | the above at target cell centres; first- and second-order conservative remapping | identity; overlap weights computed at fit |
| Scattered → anywhere | inverse distance; RBF (thin-plate, multiquadric, Gaussian, compact Wendland); linear on a Delaunay triangulation; natural neighbour; Clough–Tocher | identity; kernel solve; triangulation at fit |
| Scattered with noise | smoothing splines; simple, ordinary and universal kriging; GP regression | kernel solve with a nugget and drift |
| Gridding by statistics | binning (mean, count, sum); 2-D histograms; LOESS | segment sums; local weighted fits |
| Gap filling | Gauss–Seidel (Laplace) fill; variational fill with a roughness penalty; EOF reconstruction | iterative or CG solve |
| Field estimation | optimal interpolation; DIVA-style variational analysis | the variational normal equations |
| Sphere | any scattered scheme with a chord or geodesic metric; periodic longitude on grids | as the base scheme |

Three entries come from studying pyinterp rather than SciPy: mixed per-axis schemes, binning, and Laplace and LOESS gap fills. All three fit the factorisation without new machinery.

## JAX compatibility, block by block

Nearly every scheme can apply under `jax.jit` and `jax.grad`, because the hard parts happen at fit. The rule: fit can do anything, apply must be traceable.

| Block | Traceable? | How |
| --- | --- | --- |
| Geometry | yes | datetimes become float offsets from the epoch before tracing; shapes are static |
| Locate, regular or rectilinear | yes | arithmetic, or `jnp.searchsorted` |
| Locate, curvilinear | apply only | invert the cell mapping at fit (Newton steps or a tree), then trace the stencil |
| Neighbours for scattered data | apply only | build a k-d tree at fit (SciPy); pass fixed-k neighbour arrays into the trace |
| Stencil apply and adjoint | yes | a gather and a contraction; the adjoint is `.at[].add`; `jax.linear_transpose` gives Hᵀ for free |
| B-spline prefilter | yes | a recursive filter in `lax.scan`, an FFT solve, or a banded solve |
| 1-D cubic spline | yes | a tridiagonal solve (`lax.linalg.tridiagonal_solve`) |
| PCHIP, Akima, Steffen | yes | slopes are elementwise with `where`; `grad` works, but there is no linear transpose |
| Kernel solve | yes | Cholesky for small N, CG for large; structured covariances help here |
| Variational fit | yes | CG with matrix-free H and L |
| Delaunay linear | apply only | triangulate at fit with SciPy; the read-out is a 3-tap barycentric stencil |
| Natural neighbour, Clough–Tocher | no | NumPy-only extras |
| Binning | mostly | `segment_sum` for mean, count and sum; medians stay NumPy |
| Missing data | yes | mask and renormalise weights; never drop samples, which would change shapes |

Gradients with respect to the target sites x\* also come free for smooth bases. Nearest has a zero gradient, and linear has a piecewise-constant one, which matters for optimisers.

## Design principles

Ten principles follow from the factorisation; each rules out a common failure of existing libraries.

1. **Plans.** Like an FFT plan: locate, triangulate and factorise once, then apply to many fields.
2. **Operators are first class.** A plan exposes H and Hᵀ, so interpolation composes inside solvers and inverse problems.
3. **Geometry is explicit.** Periodicity, bounds, the time epoch and the metric live on geometry objects, not in keyword arguments.
4. **Policies are explicit.** Outside, missing and boundary behaviour are named; the default outside the domain is NaN.
5. **Batching over leading dims.** Interpolated dims come last; everything before them is a batch.
6. **Static shapes.** Invalid targets are flagged, never dropped, so plans trace and vmap.
7. **Variance is optional.** Schemes with a probabilistic reading offer it; nothing pays for it by default.
8. **Fit can do anything; apply must trace.** NumPy, SciPy and trees are fine at fit time.
9. **Arrays at the core, carriers at the edge.** Adapters translate carriers to geometry and back.
10. **Oracles in the tests.** Each scheme is checked against SciPy, an analytic case or a reference library, and each adjoint by the dot-product test ⟨Hx, r⟩ = ⟨x, Hᵀr⟩.

## External dependencies and licences

The core needs an array library and little else; every heavier dependency is optional or test-only. Licences are as I understand them and need checking before release.

| Dependency | Used for | Status | Licence |
| --- | --- | --- | --- |
| JAX, NumPy | the array core | required | Apache-2.0, BSD-3-Clause |
| Equinox | modules as pytrees | required | Apache-2.0 |
| lineax | linear operators and solves | required | Apache-2.0 |
| einx, jaxtyping | contractions; shape annotations | required | MIT |
| SciPy | k-d trees and triangulation at fit; test oracle | optional | BSD-3-Clause |
| A structured-covariance library (gaussx) | kriging and GP solves with structure | optional | own |
| A differential-operator library (finitevolX, spectraldiffx) | the roughness operator L in variational fits | optional | own |
| pyinterp | test oracle for geo axes, binning and fills | tests only | BSD-3-Clause |
| xESMF | test oracle for conservative remapping | tests only | MIT |
| interpax | test oracle for JAX splines | tests only | MIT |

The optional libraries only ever supply operators or solvers to block 5. That keeps the dependency edges pointing into the fit, never into the core.

## Where it stands against existing libraries

No library I know treats interpolation as representation plus solver plus operator, with adjoints, in one differentiable design. That is a hypothesis from a partial survey, not a settled fact.

| Library | Strength | What it lacks here |
| --- | --- | --- |
| `scipy.interpolate` | the broadest catalogue; reusable interpolator objects | NumPy only; no adjoint; no gradients |
| `jax.scipy.ndimage.map_coordinates` | traceable grid sampling | orders 0 and 1 only; no plans |
| interpax | JAX splines on 1-D to 3-D grids, differentiable | grids only; no scattered data, solvers or exposed operators |
| pyinterp | geo axes, periodicity, time axes, binning, gap fills, trees | C++ and NumPy; no gradients or adjoints |
| xESMF | conservative remapping; weights reused like plans | NumPy and ESMF; not differentiable |
| GP libraries (GPJax, tinygp, GPyTorch) | kriging as GP regression, with variance | no grid stencils; no interpolation API |
| Verde | scattered gridding with a scikit-learn style API | NumPy; no adjoint |
| DIVAnd | variational analysis at scale | Julia; not JAX |

The gap, if it holds, is the combination: plans, operators with adjoints, explicit geometry, and JAX tracing across local and global schemes.

## Survey notes

The call-site inventory and findings behind the integration section of the [functax design](functax.md), surveyed from the code on 4 Nov 2026 (geotoolz main 1e7308b, the GeoStack submodule pins for the rest).

### geotoolz call sites

| Site | Today | Engine, backends | With the package |
| --- | --- | --- | --- |
| `geoops.geom.interp_at`, `InterpAt(method=, along=)` | bilinear or nearest at `GeoPoints`; outside → NaN with `valid=False`; edge clamp; nodata propagates | hand-rolled on `to_pixel`; NumPy and JAX, `grad` in positions | the body becomes a plan's `apply`; `apply_adjoint` adds the Hᵀ that does not exist today |
| `Reproject`, `ReprojectLike`, `ResampleLike`, `Mosaic`, `RasterToRasterLike` | grid → grid in another CRS, `resampling="bilinear"` | GDAL; NumPy only | a JAX engine: target centres mapped to the source CRS at fit, then a stencil; `average` becomes conservative |
| `Resize`, `Resample` | same-CRS rescale with anti-aliasing | georeader and skimage; NumPy only | separable tensor stencils, conservative when shrinking |
| `GeostationaryParallaxCorrect`, `BowtieCorrection`, `PhaseAlign` | sample at computed source coordinates | a private NumPy bilinear, 1-D per scan, `scipy.ndimage.shift` | the same plan path as `interp_at`, with traced targets |
| `geopatcher.PointDomain(interp=)` | a third bilinear, corner convention, NaN outside | NumPy | call `interp_at`; it disagrees with it on the pixel convention today |
| `RasterToPoints`, `RasterToPointCloud(method="idw")` | raster → points | xarray `.interp`, KD-tree | grid → points and neighbourhood plans |
| `PointsToRaster`, `PointCloudToRaster(method="idw")` | points → raster; `PointsToRaster(method="idw")` raises `NotImplementedError` | KD-tree | an inverse-distance plan: the natural first target |
| `GapFillIDW`, `GapFillNearest`, `GapFillLaplacian`, `GapFillInpaintBiharmonic` | fill invalid pixels | cKDTree, distance transform, Jacobi; NumPy only | patterns 4 and 9: valid pixels as sources, invalid ones as targets |
| `ApplySRF`, `ContinuumRemoval`, `match_cdf` | 1-D along `band` or a CDF | georeader nearest-band SRF, `np.interp`, `xp.interp` | a 1-D linear plan along `band`; the SRF becomes one sparse H with a transpose |
| `ReprojectingRasterField` | warp each chip with a 3-pixel margin; chips differ slightly from one full warp | GDAL per chip | one plan on the destination grid, sliced per chip: bit-identical chips |
| swath readers (`GeoSwath`) | nothing resamples a swath; `BowtieCorrection` defers "to a swath-to-grid resampler" | — | a new swath → grid operator on a neighbourhood or curvilinear plan |

Three I/O paths should stay on GDAL: `geocatalog.load_raster`, `geocloud.read_reproject*` and `write_cog` overviews. They warp while reading files and gain nothing from tracing.

### What the survey adds to the design

Thirteen requirements come out of the call sites; none is in the sketch yet.

1. **Descending axes**, such as wavelength from wavenumber.
2. **Batched source axes**: each column with its own heights.
3. **Cheap rebuilds inside `scan`** for moving targets (particles, departure points).
4. **Deposition as a first-class adjoint**, nearest and cloud-in-cell.
5. **A `raise` outside policy**, hand-written at four sites today.
6. **Masked sources**: renormalise over valid ones, or fall back to the nearest valid (coasts).
7. **Divergence-preserving velocity interpolation** on C-grids.
8. **The three fill conventions** of geotoolz.
9. **A sparse `linearize`** for filterax and pipekit-cycle.
10. **Export to gaussx** `SparseOperator` and `InterpolatedOperator`.
11. **No silent casts**: weights keep the coordinates' dtype; vardax casts to float32 today.
12. **Warp plans**: target centres mapped through pyproj at fit.
13. **Plans sliceable by window**, for geopatcher chips.

### Duplicates to consolidate

| Kernel | Copies today |
| --- | --- |
| Bilinear weights | geoops `bilinear_stencil` and `interp_at`; geoops `_sample_bilinear`; geopatcher `PointDomain`; vardax `InterpObs`; finitevolX `_prolongate`; plumax `_bilinear` |
| Cubic weights | gaussx `cubic_interpolation_weights` |
| Hat-basis adjoint | radtrax `_hat_weights` |
| RBF and Wendland bases | geonnax, vardax, kernellib |
| Fourier resize and point evaluation | geonnax `band_limited_resize`, `nonuniform_irfft` |
| 1-D linear lookups | plumax, radtrax and geoops calls to `np.interp` |

The bilinear row alone has six copies with at least two pixel conventions; that is the strongest argument for the package.
