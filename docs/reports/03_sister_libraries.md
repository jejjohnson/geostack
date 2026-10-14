---
title: Sister libraries on pipekit
subtitle: Report 3 · Arrays, GeoTensor and xarray, checked against what geotoolz, xrtoolz and pipekit-array ship
short_title: 3 · Sister libraries
description: The three carrier libraries on top of pipekit (pipekit-array for Array API arrays, geotoolz for GeoTensor, xrtoolz for xarray), as proposed and as shipped, with where each symbol lives and what is still missing.
keywords:
  - pipekit-array
  - geotoolz
  - xrtoolz
  - Array API
  - GeoTensor
  - xarray
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

pipekit is carrier-agnostic, so the real work happens in libraries that pick
a carrier. This report proposed three: **pipekit-array** for any Array API
array, **geotoolz** for `GeoTensor`, and **xrtoolz** for xarray. All three
now exist. They are not layered the way the proposal drew them. geotoolz and
xrtoolz are large and depend on pipekit directly. pipekit-array has shipped
its first phase, four names, and nothing uses it yet. Each part below records
what was proposed, what shipped, and the gap. The gaps become open questions
and follow-ups at the end.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Shipped surface; originally a surface proposal for the libraries layered on pipekit |
| **Reading time** | about 25 minutes |
| **Audience** | anyone scoping the library ecosystem on top of pipekit, or deciding where a new operator belongs |
| **Companions** | [Report 1](01_toolz_primer.md) (background); [Report 2](02_pipekit_surface.md) (pipekit core); [Report 9](09_use_cases.md) (use cases revisited); forthcoming ones are listed under [References](#r3-references) |
| **Checked against** | pipekit and pipekit-array 0.0.2 (`ef3c17f`); geotoolz 0.10.2 (`088475e`); xrtoolz 0.0.5 (`308c7f2`): the revisions pinned in `references/`. pipekit-array and xrtoolz examples are from real runs; geotoolz claims come from reading its source, since its raster stack (rasterio) is not installed here |
:::

:::{important} How to read the differences
As in [Report 2](02_pipekit_surface.md), each part says what the proposal
planned and what shipped. A **Differs from the plan** box marks each gap, and
each gap is carried to [](#r3-open-questions) or [](#r3-todos). The proposal
called the xarray library `xr_toolz`; it shipped as `xrtoolz`, and this page
uses the shipped name.
:::

(r3-layers)=
## The layers above pipekit

The proposal drew a strict stack: geotoolz and xrtoolz on top, delegating
their array maths down to pipekit-array, which sits on pipekit core.

```{code-block} text
:caption: The layering as proposed.

geotoolz (GeoTensor)     xrtoolz (xarray)     others …
          │                    │                 │
          └─────────┬──────────┴─────────────────┘
                    ▼
            pipekit-array        ← duck-typed arrays (NumPy, JAX, CuPy, …)
                    │
                    ▼
               pipekit           ← carrier-agnostic framework
```

What shipped has the same three carriers and different wiring
({numref}`fig-r3-layers`). geotoolz sits directly on pipekit core. xrtoolz
sits on a small package of its own, `xrcore`, which subclasses pipekit's
`Operator`. pipekit-array also sits on pipekit core, but neither domain
library imports it. A fourth sibling, pipekit-jax, has shipped too.

```{figure} ../images/reports/r3_layers.svg
:label: fig-r3-layers
:alt: Dependency diagram. The geotoolz family depends directly on pipekit core. The xrtoolz family depends on xrcore, which depends on pipekit core. pipekit-array and pipekit-jax depend on pipekit core; pipekit-jax also needs pipekit-cycle. Dashed arrows from geotoolz and xrtoolz to pipekit-array mark planned links that do not exist.
:width: 100%

**The shipped dependencies. The dashed links to pipekit-array were planned and are not wired.**
```

:::{table} The proposed layers, and what each became.
:label: tbl-r3-layers

| Layer | Proposed | Shipped | Source lines |
|---|---|---|---|
| Array | `pipekit-array`: 12 Array API operators, about 400 lines | `pipekit-array` 0.0.2: phase A of four, 4 names | 281 |
| Geo | `geotoolz`: GeoTensor operators, with framework code reduced to a shim | `geotoolz` 0.10.2 plus four packages: `geocatalog`, `geocloud`, `geopatcher`, `geoproducts` | 35,044 (+ 49,425 in the four) |
| xarray | `xr_toolz`: xarray operators, with framework code reduced to a shim | `xrtoolz` 0.0.5 plus five packages: `xrcore`, `xreinx`, `xrgrad`, `xrreader`, `xrsklearn` | 31,729 (+ 16,588 in the five) |
| JAX | `pipekit-jax`: deferred until a project needs it | `pipekit-jax` 0.0.2: model wrapping, not tracing ([](#r3-other)) | 493 |
:::

The proposal's sizes were estimates. The shipped line counts include
docstrings and exclude tests.

(r3-array)=
## Part 1 · pipekit-array: arrays through the Array API

### What the Array API gives us

The [Python Array API standard](https://data-apis.org/array-api/) specifies
a common array interface. A conforming array implements
`__array_namespace__()`, which returns a module-like namespace with `mean`,
`sum`, `where`, `reshape` and the rest. One function then works on any
conforming backend:

```{code-block} python
:caption: The duck-array idea, formalised.
:linenos:

from pipekit_array import array_namespace

def channel_mean(x):                          # x: (..., C)
    xp = array_namespace(x)                   # numpy, jax.numpy, cupy, …
    return xp.mean(x, axis=-1, keepdims=True)  # x̄ = (1/C) ∑ᶜ xᶜ  →  (..., 1)
```

The proposal's conformance table was out of date in three places. It is
corrected below from the [NumPy](https://numpy.org/doc/2.0/reference/array_api.html)
and [array-api-compat](https://data-apis.org/array-api-compat/) documentation:

:::{table} Array API support by backend, corrected.
:label: tbl-r3-conformance

| Library | How it conforms | The proposal said |
|---|---|---|
| NumPy ≥ 2.0 | the main `numpy` namespace (standard 2022.12, working toward 2023.12) | "via `numpy.array_api`": that experimental module was **removed** in NumPy 2.0 |
| JAX | `jax.numpy` itself, from JAX 0.4.32 (`jax.experimental.array_api` before) | "since JAX 0.4.20" |
| CuPy | through `array-api-compat` | full |
| PyTorch | through `array-api-compat`; tensors do not expose `__array_namespace__` themselves | "via `torch.array_api`": there is no such module |
| Dask | through `array-api-compat`; lazy | partial |
| sparse | native support | partial |
:::

pipekit-array's dispatch handles that wrinkle correctly. `array_namespace`
tries `array-api-compat` first, which is shipped as the optional `[compat]`
extra and needed for PyTorch, then falls back to `x.__array_namespace__()`.
Either way it fails with the same message when no input conforms or the
inputs mix backends.

### What ships

pipekit-array is planned in four phases (its `docs/design/boundaries.md`,
§4). Phase A has shipped:

```{code-block} python
:caption: pipekit-array 0.0.2, from a real run on NumPy.
:linenos:

import numpy as np
from pipekit import Signature
from pipekit_array import ConcatenateAlong, MeanScalar, StackAlong, array_namespace

x = np.arange(24.0).reshape(2, 3, 4)          # (band, y, x)
array_namespace(x).__name__                   # → 'numpy'
MeanScalar()(x)                               # → 11.5            full mean, scalar
MeanScalar(axis=(-2, -1))(x)                  # → [ 5.5 17.5]     per band, (band,)
StackAlong(axis=0)([x[0], x[1]]).shape        # → (2, 3, 4)
ConcatenateAlong(axis=0)([x, x]).shape        # → (4, 3, 4)

pipe = StackAlong(axis=0) | MeanScalar(axis=(-2, -1))
pipe([x[0], x[1]])                            # → [ 5.5 17.5]

sig = Signature(dims=(("band", 2), ("y", 3), ("x", 4)), dtype="float64")
MeanScalar(axis=(-2, -1)).compute_output_signature(sig)
# → Signature((band=2) float64)               the reduced dims are dropped
```

`MeanScalar` is the first operator on *pipekit's* `Signature` that does real
shape inference: it drops the reduced dimensions, which is the payoff
[Report 2](02_pipekit_surface.md#r2-group-i) said core leaves to the sister
libraries. xrtoolz's operators (in `geo`, `interpolate`, `metrics` and
xreinx) infer shapes too, but on xrcore's separate `Signature`
([](#r3-xrtoolz)).

:::{table} The proposed pipekit-array catalogue, by phase, at 0.0.2.
:label: tbl-r3-array-catalogue

| Operator | Purpose | Phase | Status |
|---|---|---|---|
| `array_namespace` | Array API dispatch | A | **ships** |
| `MeanScalar` | reduce with `xp.mean`; the proposal's `field=` argument shipped as `axis=` and `keepdims=` | A | **ships** |
| `StackAlong(axis)` | stack a list of arrays along a new axis | A | **ships** |
| `ConcatenateAlong(axis)` | concatenate along an existing axis | A | **ships** |
| `Subsample(stride)` | stride-decimate the last two axes | B | planned |
| `ApplyToBands(inner, axis=0)` | split, apply, stack over an axis | B | planned |
| `BatchedMap(op, batch_size)` | split along axis 0, apply, concatenate | B | planned |
| `Histogram(bins)` | capture distributions per tap site | B | planned |
| `Diff(reference, atol)` | raise on drift from a stored reference | C | planned |
| `AssertValueRange(min, max, on_fail)` | raise or warn when out of range | C | planned |
| `AssertNoNaN()` | raise on any NaN | C | planned |
| `AssertValidFraction(min_valid)` | raise when too few values are valid | C | planned |
| `ModelOp(model, method, batch_size)` | framework-agnostic inference | D | planned |
:::

:::{warning} Differs from the plan
- **Two planned names are already taken.** pipekit core shipped its own
  `BatchedMap` and `Histogram` ([Report 2](02_pipekit_surface.md#r2-group-j)).
  Phase B as written would give the stack two different objects with the
  same name, which the "one object, one name" rule forbids. Either the
  array versions get distinct names, or they extend the core classes. See
  [](#r3-open-questions).
- **Nothing depends on pipekit-array yet.** No geotoolz or xrtoolz package
  imports it. The migration in [](#r3-array-migration) has not started.
:::

### The trade-offs

These held up as written, and they are why pipekit-array is worth its own
package.

**What works:**
- NumPy code is bog-standard.
- JAX array maths works, and so do `jax.jit`, `jax.vmap` and `jax.grad`
  through a pipeline of JAX operators: a tracer is not a `Node`, so dispatch
  runs `_apply` ([Report 5](05_pipekit_jax.md#r5-stage-1)). Gradients with
  respect to parameters held by operators do not reach them
  ([](#r3-other)).
- CuPy arrays stay on the GPU through the pipeline.
- PyTorch keeps gradient tracking, so a pipeline can be differentiated inside
  an autograd context.
- The same operator runs on NumPy or JAX unchanged; the input's namespace
  picks the backend. Inputs that mix backends in one call are rejected.

**What doesn't:**
- **NaN handling** differs between backends at the edges, so operators that
  use NaN as a fill value need per-backend tests.
- **In-place operations** break on JAX, whose arrays are immutable.
  pipekit-array operators are therefore pure.
- **Operations missing from the standard**, such as some FFTs and advanced
  linear algebra, need backend-specific fallbacks. This is pipekit-array's
  own open question Q1.
- **Dask is lazy**: a pipeline needs an explicit `.compute()` at the end
  (open question Q5).

### Dependency surface

Shipped as proposed: `dependencies = ["pipekit"]`, with one extra per backend
(`numpy`, `jax`, `torch`, `cupy`, `dask`), plus a `compat` extra the proposal
did not have (`array-api-compat`, needed for PyTorch). With no backend
installed, `array_namespace` raises a `TypeError` naming the extras to
install, rather than an `ImportError`.

(r3-array-migration)=
### Migrating the NumPy operators

The proposal moved the NumPy-flavoured operators out of geotoolz and xrtoolz
into pipekit-array, then re-exported them with carrier-specific defaults:

```{code-block} python
:caption: The proposed pattern: one implementation, re-exported with GeoTensor defaults.
:linenos:

# geotoolz/qc.py (proposed)
from pipekit_array.qc import AssertValueRange as _AssertValueRange

class AssertValueRange(_AssertValueRange):
    """Read GeoTensor's fill_value as the default bounds when none are given."""
```

None of this has happened, because the pipekit-array side (phase C) does not
exist yet. Neither does the geotoolz side: geotoolz has no value-range, NaN
or valid-fraction assertions at all. Its `assert_*` functions in
`_src/contract.py` are test-contract helpers, not operators.

(r3-geotoolz)=
## Part 2 · geotoolz: `GeoTensor` operators

### Scope

The proposal made geotoolz a thinner library focused on remote sensing over
`georeader.GeoTensor`, with its old framework code (`geotoolz.core`) reduced
to a shim that re-exports pipekit. The shipped geotoolz has **no `core` at
all**: it depends on pipekit (`pipekit>=0.0.2`) and its top-level `__init__`
re-exports 16 pipekit names directly. `gz.Operator` is `pipekit.Operator`.

:::{table} The pipekit names geotoolz re-exports at 0.10.2, all the same objects.
:label: tbl-r3-gz-reexports

| Re-exported | Not re-exported |
|---|---|
| `Operator`, `Carrier`, `Sequential`, `Graph`, `Input`, `Node`, `Identity`, `Const`, `Lambda`, `Sink`, `Tap`, `Snapshot`, `ShapeTrace`, `Branch`, `Switch`, `Fanout` | `Cache`, `Memoize`, `Try`, `Coalesce`, `Retry`, `Profile`, `Histogram`, `Quarantine`, the `Assert*` classes, the parallel maps, `pipe`, `compose`, `juxt`, `complement` |
:::

geotoolz defines no class with the same name as a pipekit public name. It
deliberately keeps `augment.Compose`, `io.SinkOperator` and
`io.SourceOperator` out of its top level, so they cannot be confused with
pipekit's `compose` and `Sink`. Its operators accept either a `GeoTensor` or
a plain `np.ndarray` and return the same kind, so the carrier is wider than
the proposal assumed.

:::{warning} Differs from the plan
geotoolz imports three private helpers from pipekit: `require_operator`,
`callable_name` and `nested_config` from `pipekit._base.operator`. geopatcher
imports `nested_config` too. pipekit can rename these without notice. Either
they become public, or geotoolz stops using them ([](#r3-todos)).
:::

### Layout

:::{table} geotoolz's subpackages at 0.10.2, against the proposed layout.
:label: tbl-r3-gz-layout

| Subpackage | What it holds | In the proposal |
|---|---|---|
| `io` | `ReadBounds`, `WriteCOG` (the `[cloud]` extra) | as `io` |
| `geom` | `BowtieCorrection`, `GeostationaryParallaxCorrect`, `Tile`, `Stitch`, `SlidingWindow`, co-registration | as `geom` |
| `radiometry` | `RadianceToReflectance`, `BTFromRadiance`, `DOS1`, `SimpleAtmosphericCorrection` | as `radiometry` |
| `indices` | `NDVI`, `NDWI`, `EVI`, …, with generic band-name resolution (`DEFAULT_BAND_KEYS`) | as `indices` |
| `spectral` | spectral operators | as `spectral` |
| `matched_filter` | `MatchedFilter`, `ColumnEnhancement` and 31 more | as `matched_filter` |
| `qa` | QA-bit decoding and cloud masks: `DecodeBitmask`, `MaskClouds`, `S2SCL`, `MODISStateQA`, `mask_from_qa_bits`, `mask_from_scl` | the proposal's `cloud`; its `qa` meant assertions |
| `mask` | `ApplyMask`, `PolygonMask`, `BBoxMask`, `CountryMask`, `LandMask` | as `mask` |
| `viz` | `TrueColor`, `FalseColor`, `StretchToUint8`, `ApplyColormap`, `ApplyDiscreteColormap` | as `viz` |
| `compositing` | `MedianComposite`, `MaxNDVIComposite`, `CloudFreeComposite`, `BAPComposite`, `MinCloud`, … | as `compositing` |
| `augment` | `RandomFlip`, `RandomRotate90`, … | as `augment` |
| `normalize` | stateful scalers with `fit` (`StandardScaler`, `RobustScaler`, `MinMaxScaler`), `Normalize`, `PerBandStats` | as `normalize` |
| `restore` | despeckling (`Lee`, `RefinedLee`, `Frost`) and gap-filling (IDW, Laplacian, biharmonic, nearest) | as `restore` |
| `plume` | `PlumeMask`, `PlumeFootprint`, `IMEEstimate`, `CrossSectionalFlux`, `WindAdvectionCone`, `SBMP`, … | as `plume` |
| `learn` | `ModelOp`, `SklearnOp` | the proposal's `pipekit_array.inference` |
| `einx`, `feature`, `measure`, `segment` | einx notation over carriers; edges, blobs, Hough, HOG; region properties and contours; segmentation | **new** |
| `patch_ops` | bridge to geopatcher: `GridSampler`, `ApplyToChips`, `MergePatches`, `StratifiedSample`, `BalancedSampler`, `TriangularWindow` | the proposal's `patch` |
:::

Three proposed subpackages moved into packages of their own, and two did not
ship:

- **`patch`** became **geopatcher** (`geotoolz-patcher`). It depends on pipekit
  only through an optional `[pipekit]` extra, and geotoolz bridges to it
  through `patch_ops`. `ExtractPatches` and `StitchPatches` do not exist
  under those names.
- **`catalog`** became **geocatalog** (`geotoolz-catalog`), where `GeoCatalog`
  is a Protocol. `CatalogPipeline` does not exist anywhere.
- **`presets`** and **`readers`** moved downstream into **geoproducts**
  ([](#r3-readers)).
- **geocloud** (`geotoolz-cloud`) is not the proposal's `cloud` (masks). It is
  cloud-storage I/O: stores, filesystems, COG and HDF access, caching,
  credentials.
- **`core`** does not exist, and no `qa` assertions shipped (see below).

### What geotoolz owns, and what has not shipped

:::{table} The proposal's named geotoolz symbols that did not ship under that name.
:label: tbl-r3-gz-missing

| Proposed | At 0.10.2 |
|---|---|
| `TOAToBOA` | not shipped; the nearest is `SimpleAtmosphericCorrection` |
| `DarkObjectSubtraction` | shipped as `DOS1` |
| `ACE`, `LinearUnmixing` | not shipped |
| `MaskFromQABits`, `MaskFromSCL` | functions `mask_from_qa_bits`, `mask_from_scl`; operators `DecodeBitmask`, `S2SCL` |
| `Colormap` | `ApplyColormap`, `ApplyDiscreteColormap` |
| `AssertCRSEquals`, `AssertResolutionWithin`, `AssertSchema` | not shipped |
| `ExtractPatches`, `StitchPatches` | not shipped; geopatcher's `GridSampler` / `MergePatches`, and `geom.Tile` / `Stitch` |
| `GeoCatalog`, `CatalogPipeline` | `GeoCatalog` is a Protocol in geocatalog; `CatalogPipeline` not shipped |
| super-resolution (in `restore`) | not shipped |
| point-source attribution (in `plume`) | no dedicated operator |
| CH4 / NH3 / N2O matched filtering | `matched_filter` is gas-agnostic: `ColumnEnhancement(gas="CH4", sensor="EMIT")` records the gas as metadata |
:::

The ownership argument itself held: geotoolz keeps everything that needs
`GeoTensor`'s `crs`, `transform` and `bounds`, plus the sensor knowledge and
the remote-sensing domain operators. Two of the proposal's reasons for
keeping things there did not turn into code: geo-specific QC
(`AssertCRSEquals`, `AssertResolutionWithin`) and per-sensor calibration
tables in geotoolz.

(r3-readers)=
### Sensor readers

The proposal put eight sensor modules in `geotoolz.readers`, each bundling a
reader, sensor-specific operators and zero-argument presets. There is no
`geotoolz.readers`. Readers and presets live in **geoproducts**, downstream of
geotoolz (its `[operators]` extra pulls geotoolz in), and two of the eight
have shipped:

:::{table} The eight planned sensor modules at geotoolz 0.10.2.
:label: tbl-r3-sensors

| Planned | Status | Where |
|---|---|---|
| GOES-R ABI | **ships**: `Reader`, `L2Reader`, `Calibration`, zero-arg presets | `geoproducts.goes` |
| Himawari AHI | **ships**: the same shape | `geoproducts.himawari` |
| MODIS L1B + L2 | no reader; `qa.MODISStateQA` and `geom.BowtieCorrection` exist | geotoolz |
| VIIRS SDR + EDR | no reader | — |
| MSG SEVIRI | no reader | — |
| MTG-FCI | no reader | — |
| TROPOMI L2 | no reader in geotoolz; xrreader has `open_tropomi_ch4_l2` | xrtoolz-reader |
| Sentinel-3 OLCI / SLSTR | no reader | — |
:::

The GOES and Himawari presets are NDVI, TrueColor, NaturalColor,
DayCloudPhase, FireTemperature, MaskClouds, ParallaxCorrect, a synthetic or
hybrid green band, and a `Recipe`. geoproducts also has a toy sensor for
tests and a Carbon Mapper client. georeader keeps its own readers for
Sentinel-2 SAFE, EMIT, EnMAP, PRISMA, PROBA-V, SPOT-VGT and Earth Engine.

### `ModelOp` and `SklearnOp`

Both live in `geotoolz.learn`, not in pipekit-array.

- **`ModelOp`** wraps any callable or object with `.predict`. It calls
  `np.asarray` on the input, supports `batch_size`, and returns the model's
  output without re-wrapping it as a `GeoTensor`.
- **`SklearnOp`** flattens pixels into samples, runs a scikit-learn
  estimator, and restores the grid. It has `fit`, `partial_fit`,
  `transform`, `predict` and `save_state`.

Both set `forbid_in_yaml = True`, because both hold a model object.

(r3-xrtoolz)=
## Part 3 · xrtoolz: xarray operators

### Scope

The proposal made `xr_toolz` a focused xarray library, with its framework
code reduced to a shim. It shipped as **xrtoolz**, and the framework layer
became a small package of its own, **xrcore** (`xrtoolz-core`, 418 lines).
xrcore's `Operator` subclasses pipekit's and adds one dispatch mode:

```{code-block} python
:caption: xrcore's Operator, from a real run.
:linenos:

import pipekit as pk
import xarray as xr
import xrtoolz

class Double(xrtoolz.Operator):
    def __init__(self, var: str = "u") -> None:
        self.var = var

    def _apply(self, ds: xr.Dataset) -> xr.Dataset:
        return (2 * ds[self.var]).rename(f"{self.var}2").to_dataset()

xrtoolz.Operator is pk.Operator            # → False: a subclass,
issubclass(xrtoolz.Operator, pk.Operator)  # → True   so it composes with pipekit
xrtoolz.Sequential is pk.Sequential        # → True   re-exported unchanged

ds = xr.Dataset({"u": ("x", [1.0, 2.0]), "v": ("x", [3.0, 4.0])})
tree = xr.DataTree.from_dict({"/a": ds, "/b": ds * 10})
Double("u")(tree)["b"].to_dataset()["u2"].values   # → [20. 40.]  every node
```

A `Node` argument still builds a graph and plain data still runs `_apply`. A
`DataTree` argument runs `_apply_tree`, which by default maps `_apply` over
every node with `xr.map_over_datasets`. That is the `xr.DataTree` support the
proposal listed, built in at the base class.

:::{warning} Differs from the plan
- **xrtoolz has its own `Signature`.** xrcore's `Signature` is a separate
  frozen dataclass, not pipekit's and not a subclass of it.
  `scripts/capabilities.py` allow-lists the duplicate name. A pipekit
  `Sequential.summary` and an xrtoolz operator therefore speak different
  signature types.
- **`xrtoolz.Operator` is not pipekit's `Operator`.** Code that checks
  `type(op) is pipekit.Operator` behaves differently; `isinstance` checks
  are unaffected.
:::

xrtoolz re-exports `Graph`, `Input`, `Node`, `Sequential` and `Tap` from
pipekit, all the same objects. It defines no `Tap`, `Sequential`, `Graph`,
`Identity`, `Fanout` or `Lambda` of its own.

### Layout

The proposal had sixteen subpackages. xrtoolz folds six of them into one
`geo` subpackage, and moves data access out to xrreader:

:::{table} xrtoolz's subpackages at 0.0.5, against the proposed layout.
:label: tbl-r3-xr-layout

| Subpackage | What it holds | The proposal's modules |
|---|---|---|
| `geo` | validation, CRS, subsetting, masks, climatology, regions, wavelets, extremes, vector/raster | `validation`, `crs`, `subset`, `masks`, `detrend` |
| `interpolate` | regridding, gap-filling, smoothing, resampling, binning | `interpolate` |
| `transforms` | Fourier, wavelet and DCT transforms; encoders; decompositions | `transforms` |
| `metrics` | `RMSE`, `PSDScore`, `R2Score`, … | `metrics` |
| `ocn` | `OkuboWeiss`, `RelativeVorticity`, `KineticEnergy`, … | `kinematics`, `ocn` |
| `atm` | atmospheric quantities, including `atm.gas.ch4` | `atm` |
| `budgets` | budget calculations | **new** |
| `inference` | `ModelOp`, `SklearnModelOp`, `JaxModelOp` | `inference` |
| `viz` | `plot_*` functions; matplotlib `*Panel` operators in `viz.validation` | `viz` |
| `combinators` | `Augment`, `ApplyToEach` | `core` |
| `rs`, `ice` | empty: a docstring only | `rs` |
| `einx`, `calc` | deprecated aliases for xreinx and xrgrad | — |
| (moved to xrreader) | `CMEMSSource`, `CDSSource`, `AemetSource`, `CATALOG` | `data` |
:::

:::{table} The proposal's named xrtoolz symbols, at 0.0.5.
:label: tbl-r3-xr-symbols

| Proposed | At 0.0.5 |
|---|---|
| `ValidateCoords`, `RenameCoords`, `SubsetBBox`, `SubsetTime` | **ship**, in `geo` |
| `AddLandMask`, `AddOceanMask`, `AddCountryMask` | **ship**, in `geo`, backed by regionmask |
| `CalculateClimatology`, `RemoveClimatology` | **ship**, in `geo` |
| `RMSE`, `PSDScore` | **ship**, in `metrics` |
| `OkuboWeiss`, `RelativeVorticity`, `KineticEnergy` | **ship**, in `ocn` |
| `SortCoords` | not shipped |
| `AssignCRS`, `Reproject`, `GetCRS` | functions `assign_crs`, `reproject`, `get_crs`, plus a `ReprojectMatch` operator (rioxarray) |
| `SubsetWhere` | the function `subset_where` |
| `ComputeAnomaly` | the function `calculate_anomaly` |
| `Regrid`, `GapFill`, `Smooth`, `Resample` | `RegridConservative`, `RegridLike`, `Coarsen`, `Refine`; the `FillNaN*` family; `GaussianSmooth`, `MovingAverage`, `LowpassFilter`; `ResampleTime` |
| one-hot and cyclical encoders | cyclical ships (`CyclicalEncode`, `EncodeTimeCyclical`); one-hot does not |
| PSD and wavelet decompositions | `PowerSpectrum`, `CWT`; the functions `pca`, `eof`, `ica`, `nmf`, `kmeans` |
| `NashSutcliffe` | not shipped; the nearest is `R2Score` |
:::

### The xarray combinators

The proposal kept two combinators in xrtoolz because they merge with xarray,
and both shipped:

```{code-block} python
:caption: Augment and ApplyToEach at xrtoolz 0.0.5, from a real run.
:linenos:

from xrtoolz import ApplyToEach, Augment

sorted(Augment(Double("u"))(ds).data_vars)    # → ['u', 'u2', 'v']

each = ApplyToEach(Double(), kwarg="var", values=["u", "v"])
sorted(each(ds).data_vars)                    # → ['u', 'u2', 'v', 'v2']
# Double(var="u") runs, merges; then Double(var="v") runs, merges
```

- **`Augment(inner)`** runs `inner(ds)` and merges the result back in with
  `ds.merge(result, compat="no_conflicts")`: the "compute derived variables
  and append them" pattern behind chains like
  `Augment(RelativeVorticity()) | Augment(KineticEnergy()) | Augment(OkuboWeiss())`.
- **`ApplyToEach(prototype, *, kwarg, values)`** rebuilds the prototype once
  per value (`cls(**{**config, kwarg: v})`) and merges every result. It
  rejects composite prototypes at construction. Each instance runs on the
  *accumulated* dataset, not on the original input, so a later value can see
  an earlier one's output.

The proposal asked for a docstring note that this merge differs from
pipekit's framework combinators. The note exists on pipekit's side
(`combine.py`, which still calls the library `xr_toolz`), not in xrtoolz.
pipekit's `Tap` replaced xrtoolz's, as planned.

### Inference

`xrtoolz.inference` ships `ModelOp`, `SklearnModelOp` and `JaxModelOp`, all
xrcore operators. The proposal's `sample_dim` argument became `feature_dim`:
every dimension except `feature_dim` is treated as a sample dimension, and
`output_name` and `output_dim` name the result. `JaxModelOp` imports JAX
lazily and has a `jit` flag. xrsklearn separately ships `SklearnOp` and
`XarrayEstimator`.

:::{warning} Differs from the plan
There are now two classes named `JaxModelOp`: `xrtoolz.inference.JaxModelOp`
(a JAX callable over xarray, by `feature_dim`) and `pipekit_jax.JaxModelOp`
(an `eqx.Module` with weight round-trip). They are unrelated. The
capability-index clash check cannot see this, because the index stops at
each library's top-level module ([](#r3-todos)).
:::

### The five sibling packages

None of these were in the proposal.

- **xrcore** (`xrtoolz-core`): the `Operator` subclass and `Signature` above.
- **xreinx** (`xrtoolz-einx`): einx patterns whose axes are dimension names.
  It has the functions `einsum`, `rearrange`, `reduce`, `repeat`, `matmul`
  and `pack_dataset`, and an operator for each (`Einsum`, `Rearrange`,
  `Reduce`, …). Its `Reduce` is a different class from `xrtoolz.geo.Reduce`.
- **xrgrad** (`xrtoolz-grad`): finite-difference calculus on xarray through
  finitediffx (JAX). It has `partial`, `gradient`, `divergence`, `curl` and
  `laplacian`, with `geometry=` set to `"cartesian"`, `"rectilinear"` or
  `"spherical"`. It is plain functions, with no operators and no pipekit
  dependency.
- **xrreader** (`xrtoolz-reader`): data access, with no pipekit dependency.
  It has the sources `CMEMSSource`, `CDSSource`, `AemetSource` and
  `LocalL2Source`; request types (`BBox`, `TimeRange`, `DepthRange`,
  `PressureLevels`, `Station`); a CF variable registry with
  `validate_dataset`; and Level-2 openers for TROPOMI, EMIT and GHGSat CH₄.
- **xrsklearn** (`xrtoolz-sklearn`): `XarrayEstimator`, which stacks,
  delegates to scikit-learn and unstacks for a `DataArray`, `Dataset` or
  `DataTree`; `.sklearn` accessors; and `SklearnOp`.

The array maths was meant to delegate to pipekit-array (the reduction inside
`RMSE`, for example). No xrtoolz package imports pipekit-array. `RMSE`
reduces through NumPy kernels in `metrics/_src/_pixel_kernels.py`.

(r3-where)=
## Part 4 · Where each symbol lives

The proposal flagged this as the most common confusion. The table below
shows where each name lives at the pinned revisions.

:::{table} Where each symbol lives at pipekit 0.0.2, geotoolz 0.10.2 and xrtoolz 0.0.5.
:label: tbl-r3-where

| Symbol | Defined in | Re-exported from | The proposal said |
|---|---|---|---|
| `Operator` | `pipekit._base.operator` | `pipekit`, `geotoolz`; subclassed by `xrcore.Operator` (re-exported by `xrtoolz`) | re-exported from both `core`s |
| `Sequential`, `Graph`, `Input`, `Node` | `pipekit._base` | `pipekit`, `geotoolz`, `xrtoolz` | as shipped |
| `Fanout`, `Identity`, `Const`, `Lambda`, `Sink` | `pipekit` | `pipekit`, `geotoolz` | both libraries |
| `Tap` | `pipekit.observe` | `geotoolz`, `xrtoolz` | as shipped |
| `Branch`, `Switch` | `pipekit.control` | `geotoolz` | both libraries |
| `Try`, `Coalesce`, `Retry` | `pipekit.control` | `pipekit` only | both libraries |
| `Snapshot`, `ShapeTrace` | `pipekit.observe` | `geotoolz` | both libraries |
| `Profile`, `Cache`, `Quarantine`, `AssertShape`, `AssertDType`, the parallel maps, `pipe`, `compose`, `juxt`, `complement` | `pipekit` | `pipekit` only | both libraries |
| `Signature` | `pipekit.signature`; a separate one in `xrcore` | `pipekit`; `xrtoolz` (xrcore's) | one, in `pipekit.signature` |
| `Histogram` | `pipekit.observe` | `pipekit` only | `pipekit_array.observe` |
| `BatchedMap` | `pipekit.parallel` | `pipekit` only | `pipekit_array.parallel` |
| `ApplyToBands`, `Subsample`, `Diff`, `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction` | not shipped | — | `pipekit_array` |
| `ModelOp` | `geotoolz.learn`; a separate one in `xrtoolz.inference` | each library | `pipekit_array.inference` |
| `Augment`, `ApplyToEach` | `xrtoolz.combinators` | `xrtoolz` | as shipped |
| `GeoCatalog` | `geocatalog` (a Protocol) | `geocatalog` | `geotoolz.catalog` |
| `CatalogPipeline` | not shipped | — | `geotoolz.catalog` |
| patch extraction and stitching | `geopatcher` | `geotoolz.patch_ops` | `geotoolz.patch` |
| sensor readers and presets | `geoproducts.goes`, `geoproducts.himawari` | `geoproducts` | `geotoolz.readers` |
| reprojection, CRS | `xrtoolz.geo` (functions, plus `ReprojectMatch`) | `xrtoolz.geo` | `xr_toolz.crs` |
| climatology, anomalies | `xrtoolz.geo` | `xrtoolz.geo` | `xr_toolz.detrend` |
:::

(r3-other)=
## Part 5 · Other libraries that could fit on top

The proposal asked, for each candidate, whether it deserves a library of its
own.

- **pipekit-array, for NumPy, JAX, CuPy and PyTorch: yes, and it exists.** The
  Array API is the right abstraction, and one library covers all four
  backends.
- **pipekit-numba: no, and none exists.** Numba is a compiler, not a carrier.
  A numba-jitted operator is still NumPy-flavoured at the carrier level; at
  most, a few pipekit-array operators could get `@njit` inner kernels behind
  a `[fast]` extra. That extra does not exist.
- **A JAX-traceable library: still deferred, and pipekit-jax is not it.**
  Differentiating *through* a pipeline with respect to its operators'
  parameters means operators that are PyTrees with static metadata and a
  pure `_apply`; dispatch already lets tracers through. The shipped **pipekit-jax** does
  something else:
  - `JaxModelOp` wraps an `eqx.Module` as an operator, with
    `serialize_weights`, `with_weights` and `from_registry`, so trained
    weights round-trip through a pipekit-experiment registry;
  - `DiffraxForwardModel` is a pipekit-cycle forward model on a diffrax
    solve;
  - `to_diffrax_adjoint` and `truncated_scan` interpret pipekit-cycle's
    adjoint specs.

  A pipeline of JAX operators can already be jitted, vmapped and
  differentiated with respect to its inputs; what is missing is
  differentiating with respect to the operators' own parameters, because no
  shipped operator is a PyTree ([Report 5](05_pipekit_jax.md#r5-stage-2)).
- **pipekit-dask: out of scope, and none exists.** Distributing work is the
  orchestrator's job. Operators that pickle can go inside `dask.bag.map` or
  `dask.delayed` as they are.
- **pipekit-cuda: covered by pipekit-array with CuPy.** That is the
  `[cupy]` extra.
- **pandas, polars, dicts: deferred, and none exists.** `Carrier = Any`, so a
  DataFrame operator is just an `Operator` subclass. The methane, MARS and
  atmospheric-chemistry work has not needed a DataFrame library.

(r3-criteria)=
## Part 6 · How to decide on a new sister library

The proposal's three questions, applied to the shipped code:

1. **Is there a clearly typed carrier?** If not, the operators belong in an
   existing library or in core.
2. **Are there at least eight operators that would consume it natively?** If
   not, contribute them to the closest existing library.
3. **Does the carrier already have a wide-enough standard?** The Array API
   does for arrays, and `xarray.Dataset` is its own de facto standard. If the
   carrier has no coherent standard, you would be inventing one alongside the
   operators.

:::{table} The criteria, with the shipped evidence.
:label: tbl-r3-criteria

| Library | Typed carrier | ≥ 8 native operators | A standard | Verdict |
|---|---|---|---|---|
| pipekit-array | any Array API array | 3 operators and the dispatcher; 12 planned | the Array API | passes on plan; **not yet on code** |
| geotoolz | `GeoTensor` (and plain `np.ndarray`) | hundreds: 272 top-level names | `GeoTensor` in this stack | passes |
| xrtoolz | `xr.DataArray`, `Dataset`, `DataTree` | hundreds, across a dozen subpackages | xarray | passes |
| JAX-traceable | JAX PyTrees | none yet | JAX | deferred until a project drives it |
| numba, dask, pandas | no new carrier, orchestration, no shared standard | — | — | do not pass |
:::

Two points in the shipped code bear on question 2. The domain libraries grew
packages of their own (geopatcher, geocatalog, xrreader, xrgrad), each with a
clear job and its own dependencies, and that turned out to be how they stay
focused. pipekit-array exists ahead of its operators. That is fine for a
library planned in phases, but until phase C lands, the code does not justify
the package.

(r3-summary)=
## Summary

:::{table} The ecosystem as proposed, and at the pinned revisions.
:label: tbl-r3-summary

| Layer | Library | What it owns | Proposed status | At the pinned revisions |
|---|---|---|---|---|
| Core | `pipekit` | the framework: operators, composition, observers, control, QC, parallelism | new | 0.0.2, 52 names ([Report 2](02_pipekit_surface.md)) |
| Array | `pipekit-array` | Array API operators | new | 0.0.2, phase A: 4 names; no dependents |
| Geo | `geotoolz` (+ 4) | GeoTensor operators, patches, catalogs, cloud I/O, sensor products | existing, refactored | 0.10.2; the framework moved to pipekit; readers in geoproducts (2 of 8 sensors) |
| xarray | `xrtoolz` (+ 5) | xarray operators, readers, calculus, einx, sklearn | existing, refactored | 0.0.5; the framework moved to pipekit, plus xrcore |
| JAX | `pipekit-jax` | the proposal: JAX-traceable operators | deferred | 0.0.2, but model wrapping and diffrax, not tracing |
| Out of scope | `pipekit-dask`, `pipekit-numba`, `pipekit-pandas` | — | not planned | not built |
:::

The proposal's takeaway was two new packages (pipekit, pipekit-array) and a
refactor of two existing ones. The refactor happened: neither domain library
carries framework code any more. The ecosystem grew beyond the proposal:
pipekit's workspace has seven packages, geotoolz's five and xrtoolz's six.
The piece that has not landed is the middle layer the proposal was built
around, a shared array library the domain libraries delegate to.

(r3-open-questions)=
## Open questions

Decisions the shipped code raises:

1. **Is pipekit-array still the middle layer?** Both domain libraries grew
   their own array code instead. Either pipekit-array's phases B to D land
   and geotoolz and xrtoolz migrate onto them, or pipekit-array becomes an
   optional toolkit beside them and the layering in this report is retired.
2. **What do pipekit-array's `BatchedMap` and `Histogram` become?** Core
   already ships both names. Options: give the array versions distinct names
   (`BatchedArrayMap`, say), make them subclasses that add axis-0 splitting,
   or drop them and extend the core classes.
3. **One `Signature` or two?** xrcore's `Signature` and pipekit's do the same
   job in different types, and xrtoolz's operators already infer shapes on
   xrcore's. Merging them would let `Sequential.summary` show xarray
   operators' shapes alongside array ones.
4. **One `JaxModelOp` name or two?** `xrtoolz.inference.JaxModelOp` and
   `pipekit_jax.JaxModelOp` are unrelated classes. Should one be renamed, or
   should the xrtoolz one wrap the pipekit-jax one?
5. **Where do the numeric QC asserts go?** `AssertValueRange`, `AssertNoNaN`
   and `AssertValidFraction` were promised to pipekit-array, and pipekit's
   `qc.py` still says they live there. They exist nowhere. Should they wait
   for phase C, or land in geotoolz and xrtoolz first?
6. **Does geotoolz own geo QC?** `AssertCRSEquals`, `AssertResolutionWithin`
   and `AssertSchema` were part of the case for geotoolz and did not ship.
7. **Which sensors come next in geoproducts?** Two of eight shipped (GOES,
   Himawari). xrreader already opens TROPOMI CH₄, so the TROPOMI reader may
   belong on the xarray side.

(r3-todos)=
## Upcoming work

:::{table} Follow-ups from checking the proposal against the pinned revisions.
:label: tbl-r3-todos

| Follow-up | Where | Why |
|---|---|---|
| Make `require_operator`, `callable_name` and `nested_config` public, or stop importing them | pipekit `_base/operator.py`; geotoolz, geopatcher | geotoolz depends on private API |
| Index each library's public subpackages in the capability index | geostack `scripts/capabilities.py` (`UPSTREAM`) | only top-level modules are indexed: the xrtoolz subpackages (`geo`, `metrics`, `ocn`, …) are missing, geocloud has no rows, and the clash check misses `JaxModelOp` and `Reduce` |
| Ship pipekit-array phases B to D, resolving the `BatchedMap` / `Histogram` names first | pipekit-array | 9 of the 13 planned names are missing |
| Fix pipekit's `qc.py` docstring that says the numeric asserts live in pipekit-array | pipekit `qc.py` | they don't exist yet |
| Rename `xr_toolz` to `xrtoolz` in pipekit's `combine.py` docstring, and add the merge-semantics note to xrtoolz's `Augment` / `ApplyToEach` | pipekit, xrtoolz | stale name; a note planned on the xrtoolz side |
| Build `CatalogPipeline` on `ProcessMap` / `ThreadMap` | geocatalog | Reports [2](02_pipekit_surface.md#r2-group-j) and 3 both assume it; use cases 2 and 12 need it |
| Fill or remove the empty `xrtoolz.rs` and `xrtoolz.ice` subpackages | xrtoolz | they ship as docstrings only |
| Add `SortCoords`, a one-hot encoder and a Nash–Sutcliffe score, or drop them from the plan | xrtoolz | named in the proposal, not shipped |
:::

(r3-references)=
## References

<!-- TODO(ref): placeholders. When a report page lands, point its links in this
file at the page (e.g. [Report 4](04_use_cases.md)) and delete its entry
below. When citations land, add them to references.bib, cite with [@key] in the
text, and replace the citations placeholder. -->

:::{note} Placeholders
:class: simple
The citations are still being written; the entry below stands in for them.
The companion reports are published: [Report 1](01_toolz_primer.md),
[Report 2](02_pipekit_surface.md) and [Report 9](09_use_cases.md).
:::

### Citations

*To be added:* the Python Array API standard, array-api-compat, NumPy's Array
API compatibility notes, xarray, and georeader, as entries in
`references.bib`. Until then, the [Array API standard](https://data-apis.org/array-api/),
[array-api-compat](https://data-apis.org/array-api-compat/) and
[NumPy's compatibility notes](https://numpy.org/doc/2.0/reference/array_api.html)
are linked inline.
