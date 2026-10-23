---
title: "pipekit: named arrays, one Signature, kernels written once"
subtitle: "Design · the named-array adapter, one Signature, KernelSpec and KernelOp"
short_title: pipekit kernels
description: "pipekit-array defines the named-array adapter, a single Signature, the dim-level functions and KernelSpec/KernelOp, so a kernel written once runs on raw arrays, geoarray carriers and xarray."
date: 2026-10-23
keywords:
  - pipekit
  - pipekit-array
  - Signature
  - KernelSpec
  - KernelOp
  - JaxOperator
  - named arrays
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

This page defines the machinery; the [package tiers](#design-tiers) say where kernels live (pipekit-array keeps no domain kernels) and [the three levels](#design-levels) say what a kernel may take.
From the [case studies](./case_studies.md), decided 23 Oct 2026: readable types are dim signatures checked with `Signature.matches` on any carrier (CS-F1); the geometry kind they also check is geoarray's, not pipekit's.

## Decision

pipekit gains the pieces every array-shaped library in the stack keeps reimplementing: a named-array protocol and the dim-level functions over it (in pipekit-array), one `Signature` for dims, and a way to write a shapes-only kernel once and run it on raw arrays, geoarray carriers and xarray carriers alike.
Domain libraries then add only what is theirs: geometry (geoarray, geoops) and value-based analysis (xrops).

```text
pipekit (core)
  Operator · Sequential · Graph · Signature (the only one)
        │
        ▼
pipekit-array
  array_namespace          (exists)
  NamedArray protocol      new: dims, data, isel, rewrap
  DimArray                 new: the plain named carrier (no georeference)
  dim-level functions      new: isel, reduce, stack, map_over, …
  KernelSpec · KernelOp    new: one kernel, every carrier
  adapters: raw arrays, xarray ([xarray] extra)
        │
        ▼
carrier libraries register an adapter
  geoarray: GeoArray · GeoSwath · GeoPoints (+ grid checks)
        │
        ▼
operator libraries
  geoops (geometry on geoarray carriers)
  xrops  (value analysis on xarray carriers)
  L1 kernels lifted with KernelOp, usable in both
```

Three outcomes:

1. **No duplicated dim machinery.**
   One protocol and one `Signature` replace the three dim vocabularies in use today.
2. **Kernels written once.**
   A kernel declares its core dims with a NumPy-style generalized-ufunc signature; `KernelOp` runs it on any registered carrier, eagerly, under `jit`, or lazily through `xr.apply_ufunc`.
3. **pipekit stays generic.**
   Nothing in pipekit knows about transforms, CRS or geometry; carrier libraries plug in a compatibility check instead.

## Today

What exists on `main` (pipekit, xrtoolz, geotoolz, 21 Oct 2026):

| Piece | Where | What it does |
| --- | --- | --- |
| `Operator` | `pipekit/_base/operator.py` | `_apply` for eager calls, a graph `Node` when given nodes, `\|` composition, JSON `state` / `from_state`, `compute_output_signature` (default passthrough) |
| `Signature` | `pipekit/signature.py` | named dims as `((name, size), …)` plus dtype; `Sequential.summary` prints it; carriers without names (raw NumPy, `GeoTensor`) opt out by returning `None` |
| `Predictor`, `FittableTransformer` | `pipekit/protocols.py` | the fit / transform contract, written once |
| `array_namespace`, `StackAlong`, `ConcatenateAlong`, `MeanScalar` | pipekit-array | Array API dispatch and the first operators, all positional (`axis=0`) |
| `JaxOperator` | pipekit-jax | an `Operator` that is also an `eqx.Module`: array fields are leaves, config fields static; used by finitevolX, spectraldiffx, somax |

Duplicates this design removes:

- **Three dim vocabularies.**
  `pipekit.Signature` (tuple of pairs), `xrcore.Signature` (a separate dict-keyed class in xrtoolz-core), and geoarray's dims (`DIMS[-ndim:]`, soon stored).
- **Two operator bases for labeled data.**
  `xrcore.Operator` (DataTree dispatch) and geoops operators that subclass `pipekit.Operator` directly with their own carrier checks.
- **Positional axes everywhere.**
  106 geoops kernel signatures default to `axis=-3`; pipekit-array combinators take `axis=`; xrtoolz wraps xarray by name.
  Each package translates between names and positions on its own.

## The named-array protocol

A carrier is anything with named dims and a raw array inside. pipekit-array reaches it through an **adapter** registered per carrier type, not through methods on the carrier: `xr.DataArray` cannot grow new methods, and an adapter keeps every carrier library free of pipekit imports in its core.

```python
class NamedArrayAdapter(Protocol):
    # ("band", "y", "x")
    def dims(self, x: Any) -> tuple[str, ...]: ...

    # the raw array: NumPy, JAX or dask
    def data(self, x: Any) -> Any: ...

    # static labels of one dim, or None (band names, level values)
    def labels(self, x: Any, dim: str) -> Any | None: ...

    # select by position on named dims: the adapter slices data and every
    # coordinate, label and geometry leaf on those dims together (band labels,
    # point positions, transform origins, per-patch transform rows)
    def isel(self, x: Any, indexers: dict[str, int | slice | Any]) -> Any: ...

    # a carrier of the same type around new data whose kept dims are unchanged
    # in length and order: untouched dims keep their metadata, dropped dims lose
    # it, new dims get none; never used for selection
    def rewrap(self, x: Any, data: Any, dims: tuple[str, ...]) -> Any: ...

    # raise if a and b cannot be combined element by element
    # (geoarray: same grid; xarray: exactly equal coordinates)
    def check_compatible(self, a: Any, b: Any) -> None: ...

    # named dims and dtype for summaries and checks
    def signature(self, x: Any) -> Signature: ...


# registration in pipekit core's registry, looked up along the carrier's MRO
xarray_adapter: NamedArrayAdapter = XarrayAdapter()
pipekit.register_adapter(xr.DataArray, xarray_adapter)
```

| Carrier | Adapter lives in | Notes |
| --- | --- | --- |
| `DimArray` | pipekit-array | the reference adapter; `check_compatible` compares dims, sizes and labels; what a geo carrier becomes when a kernel consumes its spatial dims |
| `xr.DataArray` | pipekit-array `[xarray]` extra | `check_compatible` compares coordinates exactly; `rewrap` keeps coords and attrs of surviving dims |
| `GeoArray`, `GeoSwath`, `GeoPoints` | geoarray | `check_compatible` compares grids, transforms, swath lat/lon or point positions and raises `GridMismatchError`; `rewrap` keeps the georeference while the spatial dims survive, and returns a `DimArray` when a kernel consumes one (below) |
| raw `np.ndarray`, `jax.Array` | none | not named; L1 kernels take them directly |

Why an adapter rather than a `typing.Protocol` on the carrier: it covers `DataArray` without an accessor, it lets geoarray carriers stay plain pytrees, and a new carrier (a coordax `Field`, if a model ever wants one) is one registration, not a change to pipekit.

## One Signature

`pipekit.Signature` is the only description of dims in the stack.
`xrcore.Signature` is deleted (re-exported for one release), geoarray carriers produce it through their adapter, and dims checks and readable types are written in terms of it.

Two additions to today's class:

```python
@dataclass(frozen=True)
class Signature:
    # (("band", 2), ("y", 600), ("x", 600)); a size may be None
    dims: tuple[tuple[str, int | None], ...]
    dtype: str | None = None

    # new: does this signature fit a dims pattern?
    # "band y x" exact; "... y x" any leading dims
    def matches(self, pattern: str) -> bool: ...

    # new: a sizeless signature from a pattern, for annotations
    @classmethod
    def from_pattern(cls, pattern: str) -> "Signature": ...
```

What it is used for:

| Use | How |
| --- | --- |
| `Sequential.summary` on any named carrier | `Operator.compute_output_signature` defaults to the adapter's `signature(x)` for registered carriers, instead of returning `None` |
| dims checks at call time | `check_dims` (pipekit-array) compares `adapter.signature(x).matches(pattern)` against the function's annotations |
| readable types | geoarray's `GeoImage` is a name, the kind `grid` and `Signature.from_pattern("band y x")`, matched on any carrier whose adapter reports a fitting signature; the type name and the kind are geoarray's, the dims rule is pipekit's |
| kernel output dims | `KernelOp` computes its output signature from its `KernelSpec` without running |

The pattern syntax is the one already used in geoarray's types (`"time band y x"`) and close to jaxtyping's, so one string means the same thing in a type annotation, a check and a summary.

## Dim-level functions

pipekit-array implements each of these once, over the adapter, as a function and as an `Operator` (for pipelines and config):

| Function | Operator | Does | Uses from the adapter |
| --- | --- | --- | --- |
| `isel` | `ISel(dim=..., index=...)` | position lookup by dim name | `isel` (only the adapter knows which coordinates and geometry to slice) |
| `sel` | `Sel(dim=..., label=...)` | exact label lookup; no ranges, no nearest | `labels` |
| `transpose` | `Transpose(order=...)` | reorder dims by name | `dims`, `rewrap` |
| `reduce` | `Reduce(fn, dim=...)` | reduce over a named dim | `dims`, `rewrap` |
| `argreduce` | `ArgReduce(fn, dim=...)` | index of the reduction along a dim | `dims`, `rewrap` |
| `take_along` | `TakeAlong(dim=...)` | gather along a dim with an index carrier | `dims`, `check_compatible` |
| `map_over` | `MapOver(fn, dim=...)` | apply a per-item function along a dim | `dims`, `rewrap` |
| `stack` | `StackAlong(dim=...)` | new dim from many carriers | `check_compatible` |
| `concat` | `ConcatenateAlong(dim=...)` | join along an existing dim | `check_compatible` |
| `combine` | `Combine(fn)` | element-wise function of several carriers | `check_compatible` |

Existing `StackAlong(axis=)` and `ConcatenateAlong(axis=)` keep `axis=` for raw arrays and gain `dim=` for named carriers.

**Fast paths.**
An adapter may override a function with its library's native equivalent when the semantics are identical: the xarray adapter routes `reduce` to `DataArray.reduce` and `map_over` to `apply_ufunc`, so a dask-backed cube stays lazy; the geoarray adapter routes `map_over` to `jax.vmap` over the data and the transform rows together.
The shared test kit proves each override matches the generic path.

**The rule for joining this list.**
A function joins only if it never reads coordinate values or geometry and behaves identically on every registered carrier, with a test that proves it.
`resample`, `groupby`, `rolling` and anything by value stay native xarray; reprojection, sampling at points and windowing a transform stay in geoarray and geoops.

(design-kernelop)=
## Kernels written once: KernelSpec and KernelOp

A shapes-only kernel declares which axes it consumes and produces with a **generalized-ufunc signature**, the convention NumPy (`np.vectorize(signature=...)`), JAX (`jnp.vectorize(signature=...)`) and xarray (`apply_ufunc` core dims) already share.
`KernelOp` wraps the kernel once and runs it on every registered carrier.
No column-specific operator is written by hand.

```text
one L1 kernel + "(c)->()"
        │
        ▼
KernelOp(kernel, "(c)->()", core={"c": "band"})
        │ dispatch on the carrier's adapter
        ├─ raw array      → call the kernel
        ├─ geoarray, NumPy → move band last, call, rewrap
        ├─ geoarray, JAX   → same, traced under jit
        └─ xarray          → apply_ufunc(input_core_dims=[["band"]],
                             dask="parallelized"), exact join
```

**What a kernel may take.**
A kernel is an L1 primitive and `KernelOp` the L2 operator generated around it ([the three levels](#design-levels)).
When the only geometry is a dim name, `KernelOp` needs nothing more; when the kernel needs derived geometry, `geometry={name: fn(carrier) -> arrays}` computes it per call.
Those keyword arrays are passed whole, never mapped, so they suit geometry with no carrier dims (a grid bundle, a time step).
Geometry that varies per pixel (view angles, an air-mass factor) has the carrier's dims and must be mapped with it, so it is computed as a carrier and passed as a positional input with its own core dims.

**When a kernel consumes a spatial dim.**
A geo carrier's dims must end in its spatial dims (`y x`, `row col`, `obs`).
Per-column statistics consume `row` and keep `col`, so the result, `(col, band)`, is no longer a `GeoSwath`.
In that case the adapter's `rewrap` returns a `DimArray` from pipekit-array: dims, data and the labels of the surviving dims, registered as a JAX pytree, with no georeference.
`KernelOp` combines carriers of different kinds by dim name and size (a `GeoSwath["band row col"]` with a `DimArray["col band"]` maps both over `col`); grid checks apply only between carriers of the same kind.
`DimArray` lives in pipekit-array (decided 23 Oct 2026, PK-P7) and stays deliberately small:

- **It holds** a raw array (NumPy, JAX or dask), its dims and the static labels of each dim; it is a JAX pytree whose only leaf is the data, and its adapter is the reference one every other adapter is tested against.
- **It does not hold** a georeference, coordinates beyond labels, or attrs.
- **It does not do** alignment, arithmetic dunders or selection by value; those stay xarray's, and a `DimArray` crosses to xarray with `to_xarray()` when an analysis needs them.

### The spec

```python
@dataclass(frozen=True)
class KernelSpec:
    # gufunc signature over abstract core dims: "(c)->()", "(),()->()", "(l),(l),(l),()->()"
    signature: str
    # backends the kernel is written for
    backends: tuple[str, ...] = ("numpy", "jax")
    # True if the kernel only handles core dims and must be vectorized over the rest
    vectorize: bool = False
```

Kernel conventions, which make this work:

1. **Consumed axes last.**
   `apply_ufunc` moves core dims to the end before calling, and gufuncs expect them there.
2. **Broadcast over leading axes** (`"... C"` in, `"..."` out).
   Otherwise set `vectorize=True` and pay for `np.vectorize` / `jnp.vectorize`.
3. **Non-carrier inputs are keyword parameters** (a background mean, a covariance, a threshold); they are not mapped over.

### The operator

```python
class KernelOp(Operator):
    def __init__(
        self,
        kernel: Callable[..., Array],
        spec: KernelSpec | str,
        core: dict[str, str],
        geometry: dict[str, Callable[[Any], Array]] | None = None,
        **params: Any,
    ) -> None:
        # the L1 function
        self.kernel: Callable[..., Array] = kernel
        # the gufunc signature and backends
        self.spec: KernelSpec = (
            spec if isinstance(spec, KernelSpec) else KernelSpec(spec)
        )
        # abstract core dim → carrier dim, e.g. {"c": "band"}
        self.core: dict[str, str] = core
        # derived geometry: kernel argument → function of the carrier returning arrays,
        # e.g. {"grid": cgrid_from_carrier}; empty when the only geometry is a dim name
        self.geometry: dict[str, Callable[[Any], Array]] = geometry or {}
        # keyword parameters passed to every call
        self.params: dict[str, Any] = params

    def _apply(self, *carriers: Any) -> Any:
        # derived geometry from the first carrier, as arrays
        derived: dict[str, Array] = {
            name: fn(carriers[0]) for name, fn in self.geometry.items()
        }
        # every keyword input to the kernel
        kwargs: dict[str, Any] = self.params | derived
        # one dispatch for every carrier family, through its adapter
        return run_kernel(self.kernel, self.spec, self.core, carriers, kwargs)

    def compute_output_signature(self, sig: Signature) -> Signature:
        # input dims minus consumed core dims, plus produced core dims; no execution
        return apply_gufunc_signature(sig, self.spec, self.core)
```

### Two kernels, every carrier

```python
# L1: element-wise, "(),()->()"
def normalized_difference(
    a: Float[Array, "*s"], b: Float[Array, "*s"], eps: float
) -> Float[Array, "*s"]:
    # a − b and a + b + ε
    num: Float[Array, "*s"] = a - b
    den: Float[Array, "*s"] = a + b + eps
    # ND = (a − b) / (a + b + ε)
    return num / den


# L1: consumes the band axis, "(c)->()"
def matched_filter_score(
    x: Float[Array, "... C"],
    mu: Float[Array, "C"],
    sigma: Float[Array, "C C"],
    t: Float[Array, "C"],
) -> Float[Array, "..."]:
    # r = x − μ
    r: Float[Array, "... C"] = einx.subtract("... c, c", x, mu)
    # w = Σ⁻¹t, solved, never inverted
    cov: lx.MatrixLinearOperator = lx.MatrixLinearOperator(
        sigma, lx.positive_semidefinite_tag
    )
    w: Float[Array, "C"] = lx.linear_solve(cov, t).value
    # rᵀw per pixel, and tᵀw
    num: Float[Array, "..."] = einx.dot("... c, c -> ...", r, w)
    norm: Float[Array, ""] = einx.dot("c, c ->", t, w)
    # α = (x − μ)ᵀΣ⁻¹t / (tᵀΣ⁻¹t)
    return num / norm


# wrapped once
nd: KernelOp = KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-10)
score: KernelOp = KernelOp(
    matched_filter_score, "(c)->()", core={"c": "band"}, mu=mu, sigma=sigma, t=t
)

# a geoarray carrier, eager; scene: GeoArray["band y x"]
alpha_map: GeoArray["y x"] = score(scene)

# a lazy xarray cube, one task per chunk; cube: DataArray["time band y x"]
alpha_cube: DataArray["time y x"] = score(cube)

# under jit on a JAX carrier; μ, Σ and t become leaves (next section)
jscore: JaxOperator = pipekit_jax.to_jax(score)
score_jit: Callable = eqx.filter_jit(jscore)
alpha_jax: GeoArray["y x"] = score_jit(jscene)
```

In the matched filter, x is a pixel spectrum, μ the background mean, Σ the background covariance and t the CH₄ target; C is the number of bands.

### Inside the xarray path

"Unwrap and rewrap" for an xarray input is one `apply_ufunc` call, made inside `KernelOp` by pipekit-array's `[xarray]` adapter:

```python
# inside KernelOp, for xarray inputs (pipekit-array[xarray])
def _call_xarray(
    kernel: Callable,
    spec: KernelSpec,
    core: dict[str, str],
    kwargs: dict[str, Any],  # KernelOp's params | derived geometry, from _apply
    *xs: DataArray["..."],
) -> DataArray["..."]:
    # "(c)->()" with core = {"c": "band"}: band moves last and is consumed
    in_dims: list[list[str]] = [[core[d] for d in s] for s in spec.inputs]  # [["band"]]
    out_dims: list[list[str]] = [[core[d] for d in s] for s in spec.outputs]  # [[]]
    # coordinates must be equal; never inner-joined
    adapter.check_compatible(*xs)
    # apply_ufunc unwraps .data, calls kernel(*blocks, **kwargs) per chunk, and
    # rewraps coords and attrs; kwargs (eps; μ, Σ, t; a grid) are passed unmapped
    return xr.apply_ufunc(
        kernel,
        *xs,
        input_core_dims=in_dims,
        output_core_dims=out_dims,
        kwargs=kwargs,
        vectorize=spec.vectorize,
        dask="parallelized",
        join="exact",
    )
```

This is what scikit-learn's `set_output` does not do: dask stays lazy, and a 10⁻⁷ m coordinate offset raises instead of returning shape `(1, 4, 0)`.

## Fitted operators and JAX

A `KernelOp` whose parameters are plain config (an `eps`, a dim name) is a plain `Operator`; inside `jit` its parameters are constants.
One that holds arrays (a background mean, a covariance, learned weights) needs those arrays as pytree leaves under `jit` and `grad`. pipekit-jax already solves that with `JaxOperator`, so the JAX shell is generated, not written:

The conversion lives in pipekit-jax, as `pipekit_jax.to_jax(op)`, not on `KernelOp`: pipekit-jax already depends on pipekit-array (through pipekit-cycle's `[array]` extra), so an edge from pipekit-array to pipekit-jax, even behind an extra, would be a workspace cycle.

```python
# NumPy and xarray: a plain Operator
score: KernelOp = KernelOp(
    matched_filter_score, "(c)->()", core={"c": "band"}, mu=mu, sigma=sigma, t=t
)

# JAX: the same kernel and spec as a JaxOperator (an eqx.Module);
# array parameters are leaves, spec and dim names are static
jscore: JaxOperator = pipekit_jax.to_jax(score)

# compile with Equinox's filter, since the module holds arrays; jscene: GeoArray["band y x"]
score_jit: Callable = eqx.filter_jit(jscore)
alpha: GeoArray["y x"] = score_jit(jscene)

# gradients flow to the parameters, e.g. a learned target signature: ∂ℒ/∂t
loss_grad: Callable = eqx.filter_grad(loss)
dloss: JaxOperator = loss_grad(jscore, jscene)
```

Fitting uses pipekit's existing `FittableTransformer` protocol: `fit` returns a new, immutable operator with its arrays filled; `transform` is the `KernelOp` call.

```python
# estimate μ and Σ from the scene's background pixels, eagerly, in pre-processing
mf: MatchedFilter = MatchedFilter(target=t, dim="band")
fitted: MatchedFilter = mf.fit(scene)  # scene: GeoArray["band y x"]

# apply: a KernelOp underneath, any carrier
alpha_map: GeoArray["y x"] = fitted(scene)
```

| Operator holds | Base | Compile with |
| --- | --- | --- |
| config only | `KernelOp` (`Operator`) | `jax.jit` on a function that calls it |
| arrays (fitted or learned) | `pipekit_jax.to_jax(op)` → `JaxOperator`, in pipekit-jax | `eqx.filter_jit`; learnable fields use `converter=jnp.asarray` |

## Migration

| Package | Change |
| --- | --- |
| pipekit | `Signature.matches`, `Signature.from_pattern`; the adapter interface (`NamedArrayAdapter`) and its registry (`pipekit.register_adapter`), so `compute_output_signature` finds an adapter without importing pipekit-array |
| pipekit-array | registers its adapters in pipekit's registry; `DimArray` (also the plain carrier for tests and for libraries with none of their own), the dim-level functions and operators, `KernelSpec`, `KernelOp`; `[xarray]` adapter; no JAX extra and no edge to pipekit-jax |
| pipekit-jax | `to_jax(op)`, which builds a `JaxOperator` from a `KernelOp`; an `[array]` extra declares the pipekit-array edge it already has through pipekit-cycle |
| xrtoolz | `xrcore.Signature` becomes a re-export of `pipekit.Signature` for one release, then goes |
| geoarray | registers its adapter (grid checks, an `isel` that slices coordinates and moves the transform origin, georeference-preserving `rewrap`, `map_over` over transform rows); stops opting out of `Signature` |
| geoops | the 106 `axis=-3` kernels become `KernelOp`s with a core-dim mapping; geometry operators stay hand-written |

Every change is a 0.X.0 bump.

## Tests and gates

pipekit-array ships a test kit so every carrier library proves the same things:

- `check_adapter(adapter, make_carrier)`: every dim-level function on the carrier equals the generic path, including fast-path overrides.
- `check_kernel_op(op, carriers)`: the same `KernelOp` on each carrier family gives equal results.

| Gate | Passes when |
| --- | --- |
| PK-K1 Parity | one `KernelOp` gives equal results (float tolerance) on a geoarray carrier, an xarray cube and raw arrays |
| PK-K2 Laziness | on a dask-backed cube, a `KernelOp` builds a graph and computes nothing until `.compute()` |
| PK-K3 One trace | the JAX path compiles once for same-shape carriers with different transforms |
| PK-K4 Summary | `Sequential.summary` prints dims for every step of a geoops pipeline, no `?` |
| PK-K5 Overhead | dispatch adds no more than a budget you set over calling the kernel on `.data` directly |

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| PK-P1 | Registry location | **Forced by the dependency rules (review of #60):** the interface and registry in pipekit core, the adapters and the dim-level functions in pipekit-array, which registers them. pipekit-array already depends on pipekit, so a lookup from core into pipekit-array, lazy or not, would be a workspace cycle | — |
| PK-P2 | Where a kernel's spec is declared | at the operator layer (`KernelOp(kernel, "(c)->()", …)`), so L1 packages need no pipekit import | a `@kernel("(c)->()")` decorator on the L1 function |
| PK-P3 | Core dim names in the spec | abstract letters plus a `core=` mapping, so the same kernel serves `dim="band"` and `dim="variable"` | carrier dim names written in the spec |
| PK-P4 | Home of the xarray adapter | pipekit-array `[xarray]`, shared by geoarray and xrtoolz | xrcore |
| PK-P5 | `Dataset` and `DataTree` | `DataArray` only at first; `Dataset` and `DataTree` dispatch stays in xrops (today's xrcore DataTree dispatch) | adapters for both in pipekit-array |
| PK-P7 | The carrier for outputs that drop a spatial dim (fitted per-column statistics) | **Decided 23 Oct 2026:** `pipekit_array.DimArray`: dims, data, labels of the surviving dims, a pytree, no georeference | — |
| PK-P6 | `to_jax()` dependency | **Forced by the dependency rules (review of #60):** `pipekit_jax.to_jax` lives in pipekit-jax, because pipekit-jax → pipekit-cycle → pipekit-array already exists and the reverse edge would close a cycle | — |
