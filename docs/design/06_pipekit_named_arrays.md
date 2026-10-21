---
title: "pipekit: named arrays, one Signature, kernels written once"
subtitle: "Design · the named-array adapter, one Signature, KernelSpec and KernelOp"
short_title: pipekit named arrays
description: "pipekit-array defines the named-array adapter, a single Signature, the dim-level functions and KernelSpec/KernelOp, so a kernel written once runs on raw arrays, geoarray carriers and xarray."
date: 2026-10-21
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
  NamedArray protocol      new: dims, data, rewrap
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


# registration, looked up along the carrier's MRO
pipekit.register_adapter(xr.DataArray, XarrayAdapter())
```

| Carrier | Adapter lives in | Notes |
| --- | --- | --- |
| `xr.DataArray` | pipekit-array `[xarray]` extra | `check_compatible` compares coordinates exactly; `rewrap` keeps coords and attrs of surviving dims |
| `GeoArray`, `GeoSwath`, `GeoPoints` | geoarray | `check_compatible` compares grids, transforms, swath lat/lon or point positions and raises `GridMismatchError`; `rewrap` keeps the georeference |
| `pipekit_array.Named` | pipekit-array | a two-field record (`data`, `dims`) for tests and for libraries with no carrier of their own |
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
| readable types | geoarray's `GeoImage` is a carrier type plus `Signature.from_pattern("band y x")`; the type name is geoarray's, the dims rule is pipekit's |
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
    def __init__(self, kernel, spec: KernelSpec | str, core: dict[str, str], **params) -> None:
        # the L1 function
        self.kernel = kernel
        # the gufunc signature and backends
        self.spec = spec if isinstance(spec, KernelSpec) else KernelSpec(spec)
        # abstract core dim → carrier dim, e.g. {"c": "band"}
        self.core = core
        # keyword parameters passed to every call
        self.params = params

    def _apply(self, *carriers):
        # one dispatch for every carrier family, through its adapter
        return run_kernel(self.kernel, self.spec, self.core, carriers, self.params)

    def compute_output_signature(self, sig: Signature) -> Signature:
        # input dims minus consumed core dims, plus produced core dims; no execution
        return apply_gufunc_signature(sig, self.spec, self.core)
```

### Two kernels, every carrier

```python
# L1: element-wise, "(),()->()"
def normalized_difference(a: Float[Array, "*s"], b: Float[Array, "*s"], eps: float) -> Float[Array, "*s"]:
    # (a − b) / (a + b + ε)
    return (a - b) / (a + b + eps)


# L1: consumes the band axis, "(c)->()"
def matched_filter_score(
    x: Float[Array, "... C"],
    mu: Float[Array, "C"],
    sigma_inv: Float[Array, "C C"],
    t: Float[Array, "C"],
) -> Float[Array, "..."]:
    # x − μ
    r = x - mu
    # Σ⁻¹ t
    w = sigma_inv @ t
    # α = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t)
    return (r @ w) / (t @ w)


# wrapped once
nd = KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-10)
score = KernelOp(matched_filter_score, "(c)->()", core={"c": "band"}, mu=mu, sigma_inv=sigma_inv, t=t)

# a geoarray carrier, eager: GeoImage (band, y, x) → GeoMap (y, x)
alpha_map = score(scene)

# the same op on a lazy xarray cube (time, band, y, x) → (time, y, x), one task per chunk
alpha_cube = score(cube)

# and under jit on a JAX carrier; mu, Σ⁻¹ and t become leaves (next section)
alpha_jax = eqx.filter_jit(pipekit_jax.to_jax(score))(jscene)
```

In the matched filter, x is a pixel spectrum, μ the background mean, Σ the background covariance and t the CH₄ target; C is the number of bands.

## Fitted operators and JAX

A `KernelOp` whose parameters are plain config (an `eps`, a dim name) is a plain `Operator`; inside `jit` its parameters are constants.
One that holds arrays (a background mean, a covariance, learned weights) needs those arrays as pytree leaves under `jit` and `grad`. pipekit-jax already solves that with `JaxOperator`, so the JAX shell is generated, not written.
The conversion lives in pipekit-jax, not on `KernelOp`: pipekit-jax already depends on pipekit-array (through pipekit-cycle's `[array]` extra), so an edge from pipekit-array to pipekit-jax, even behind an extra, would be a workspace cycle.

```python
# NumPy and xarray: a plain Operator
score = KernelOp(matched_filter_score, "(c)->()", core={"c": "band"}, mu=mu, sigma_inv=sigma_inv, t=t)

# JAX: the same kernel and spec as a JaxOperator (an eqx.Module);
# array parameters are leaves, spec and dim names are static
jscore = pipekit_jax.to_jax(score)

# compile with Equinox's filter, since the module holds arrays
alpha = eqx.filter_jit(jscore)(jscene)

# gradients flow to the parameters, e.g. a learned target signature
dalpha_dt = eqx.filter_grad(loss)(jscore, jscene)
```

Fitting uses pipekit's existing `FittableTransformer` protocol: `fit` returns a new, immutable operator with its arrays filled; `transform` is the `KernelOp` call.

```python
# estimate μ and Σ from the scene's background pixels, eagerly, in pre-processing
fitted = MatchedFilter(target=t, dim="band").fit(scene)

# apply: a KernelOp underneath, any carrier
alpha_map = fitted(scene)
```

| Operator holds | Base | Compile with |
| --- | --- | --- |
| config only | `KernelOp` (`Operator`) | `jax.jit` on a function that calls it |
| arrays (fitted or learned) | `pipekit_jax.to_jax(op)` → `JaxOperator`, in pipekit-jax | `eqx.filter_jit`; learnable fields use `converter=jnp.asarray` |

## Migration

| Package | Change |
| --- | --- |
| pipekit | `Signature.matches`, `Signature.from_pattern`; the adapter interface (`NamedArrayAdapter`) and its registry, so `compute_output_signature` finds an adapter without importing pipekit-array |
| pipekit-array | registers its adapters in pipekit's registry; `Named`, the dim-level functions and operators, `KernelSpec`, `KernelOp`; `[xarray]` adapter; no JAX extra and no edge to pipekit-jax |
| pipekit-jax | `to_jax(op)`, which builds a `JaxOperator` from a `KernelOp`; an `[array]` extra declares the pipekit-array edge it already has through pipekit-cycle |
| xrtoolz | `xrcore.Signature` becomes a re-export of `pipekit.Signature` for one release, then goes |
| geoarray | registers its adapter (grid checks, georeference-preserving `rewrap`, `map_over` over transform rows); stops opting out of `Signature` |
| geoops | the 106 `axis=-3` kernels become `KernelOp`s with a core-dim mapping; geometry operators stay hand-written |

Every change is a 0.X.0 bump.

## Tests and gates

pipekit-array ships a test kit so every carrier library proves the same things:

- `check_adapter(adapter, make_carrier)`: every dim-level function on the carrier equals the generic path, including fast-path overrides.
- `check_kernel_op(op, carriers)`: the same `KernelOp` on each carrier family gives equal results.

| Gate | Passes when |
| --- | --- |
| K1 Parity | one `KernelOp` gives equal results (float tolerance) on a geoarray carrier, an xarray cube and raw arrays |
| K2 Laziness | on a dask-backed cube, a `KernelOp` builds a graph and computes nothing until `.compute()` |
| K3 One trace | the JAX path compiles once for same-shape carriers with different transforms |
| K4 Summary | `Sequential.summary` prints dims for every step of a geoops pipeline, no `?` |
| K5 Overhead | dispatch adds no more than a budget you set over calling the kernel on `.data` directly |

## Open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| P1 | Registry location | **Forced by the dependency rules (review of #60):** the interface and registry in pipekit core, the adapters and the dim-level functions in pipekit-array, which registers them. pipekit-array already depends on pipekit, so a lookup from core into pipekit-array, lazy or not, would be a workspace cycle | — |
| P2 | Where a kernel's spec is declared | at the operator layer (`KernelOp(kernel, "(c)->()", …)`), so L1 packages need no pipekit import | a `@kernel("(c)->()")` decorator on the L1 function |
| P3 | Core dim names in the spec | abstract letters plus a `core=` mapping, so the same kernel serves `dim="band"` and `dim="variable"` | carrier dim names written in the spec |
| P4 | Home of the xarray adapter | pipekit-array `[xarray]`, shared by geoarray and xrtoolz | xrcore |
| P5 | `Dataset` and `DataTree` | `DataArray` only at first; `Dataset` and `DataTree` dispatch stays in xrops (today's xrcore DataTree dispatch) | adapters for both in pipekit-array |
| P6 | `to_jax()` dependency | **Forced by the dependency rules (review of #60):** `pipekit_jax.to_jax` lives in pipekit-jax, because pipekit-jax → pipekit-cycle → pipekit-array already exists and the reverse edge would close a cycle | — |
