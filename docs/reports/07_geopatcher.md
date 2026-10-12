---
title: geopatcher
subtitle: Report 7 · The four-axis patcher, as scoped and as built
short_title: 7 · geopatcher
description: geopatcher, the locality layer that splits a geospatial field into patches, runs an operator per patch, and stitches the outputs back. Its design and figures come from the geopatcher repo's README and docs, its API from the code at geotoolz-patcher 0.10.2. The scoping report's claims are checked against both.
keywords:
  - geopatcher
  - patching
  - sliding window
  - overlap-add
  - Field protocol
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

geopatcher splits a geospatial field into local patches, runs an operator on
each one, and stitches the outputs back into a global result. It does this
along **four independently composable axes**: geometry, sampler, window and
aggregation. This report scoped it as a standalone package extracted from
geotoolz, with a framework-free core and an optional pipekit bridge. It has
been built, and it is now the largest package in the ecosystem. Most of the
design held: the four axes, the `Field` and `Domain` Protocols, the
extras-gated adapters, and the pipekit bridge in `integrations/`. Three
things did not: the dependency-light core, the class names (the axis
prefixes are gone), and the geotoolz migration. Each is carried to the open
questions at the end.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Built; originally a scoping proposal ("committed") for the patcher split |
| **Reading time** | about 20 minutes |
| **Audience** | anyone patching data for inference or training, or deciding how geotoolz and xrtoolz should use the patcher |
| **Companions** | [Report 6](06_geocatalog.md) (geocatalog, the other side of the `Domain` bridge); [Report 2](02_pipekit_surface.md) (pipekit core); earlier reports in the series are listed under [References](#r7-references) |
| **Design and figures** | the [geopatcher repo](https://github.com/jejjohnson/geopatcher)'s README and docs (concepts, patching guide, design decisions, recipes) and its four figures |
| **Code checked against** | `geotoolz-patcher` 0.10.2, import name `geopatcher`, in the geotoolz monorepo at `088475e` (the revision pinned in `references/`). Examples are from real runs |
:::

:::{important} Which geopatcher this page describes
Like geocatalog ([Report 6](06_geocatalog.md)), geopatcher has two copies:

- **The geopatcher repo** is where the package was developed, up to 0.0.6
  (May to July 2026). It is now **archived**. Its README, docs and figures
  are the design of record, and this page follows them.
- **On 13 July 2026, geopatcher merged into the geotoolz monorepo**
  ([geotoolz#97](https://github.com/jejjohnson/geotoolz/issues/97)) as the
  `geotoolz-patcher` distribution. Since then it has gone through
  fourteen releases to 0.10.2, ten of them with breaking changes. This is the copy geostack
  pins, so this page takes its **API and behaviour** from it.

The README and docs use the names from before 0.9.0, such as
`gp.SpatialRectangular` and `gp.SpatialOverlapAdd`. 0.9.0 dropped the
`Spatial` and `Temporal` prefixes, with no aliases. The examples here use the
current names (`geopatcher.spatial.geometry.Rectangular`). Where the scoping
report, the repo and the code disagree, a **Differs** box says so.
:::

(r7-stack)=
## Where geopatcher sits

The repo's README calls geopatcher the **locality layer** between catalogs
and operators. It answers a single question: "what slice of the data does my
operator see at once, and how do local outputs become a global field?"

```{code-block} text
:caption: The ecosystem as scoped.

                georeader  (GeoTensor, GeoData, readers)
                    ▲
       ┌────────────┼────────────┐
  geocatalog     pipekit     geopatcher ──┐
       ▲            ▲            ▲        │ optional [pipekit] extra
       │            └──── geopatcher.integrations.pipekit
       │                         ▲          (GridSampler, ApplyToChips, Stitch)
       └────────────┬────────────┘
              geotoolz · xrtoolz
```

The scoping report made three observations about this picture:

1. **The core depends on neither pipekit nor geocatalog.** This holds.
   geopatcher never imports geocatalog. The bridge runs the other way:
   geocatalog's `CatalogDomain` satisfies geopatcher's `Domain` Protocol.
   pipekit appears only in the `integrations.pipekit` module, behind the
   `[pipekit]` extra.
2. **The pipekit integration is a thin sliver.** This holds: one 154-line
   module.
3. **geopatcher is a peer of geocatalog and pipekit.** This holds as well.

:::{warning} Differs from the plan
The scoping report said the core does not depend on **georeader** either,
importing it only in an extras-gated raster module. At 0.10.2, georeader,
rasterio and pyproj are base dependencies, and `import geopatcher` loads
georeader and rasterio ([](#r7-deps)). As with geocatalog
([Report 6](06_geocatalog.md#r6-migration)), the bottom of the picture is not
wired: geotoolz consumes only the pipekit bridge, and xrtoolz does not
import geopatcher at all ([](#r7-migration)).
:::

(r7-axes)=
## The four-axis framework

### The four axes

Each axis is a strategy object, a small dataclass with a single job, that
the patcher composes at construction. The repo's rule is that anything you
can swap without touching the other three is on its own axis.

```{figure} ../images/reports/r7_four_axes.png
:label: fig-r7-axes
:alt: Four panels. Geometry, the shape of the neighbourhood, shown as an 8 by 8 rectangle and a disk of radius 5. Sampler, where anchors go, shown as regular-stride, jittered and random points. Window, per-cell weights, shown as boxcar, Hann, Tukey and Gaussian curves. Aggregation, local outputs to a global field, shown as boxes for OverlapAdd, Mean, WeightedSum, InvVarWMean, HardVote and ApproxQuantile.
:width: 90%

**The four orthogonal axes.** From the geopatcher repo.
```

:::{table} The four axes and their shipped classes at 0.10.2, spatial and temporal.
:label: tbl-r7-axes

| Axis | The question it answers | Spatial (`geopatcher.spatial.*`) | Temporal (`geopatcher.temporal.*`) |
|---|---|---|---|
| Geometry | what shape is the neighbourhood? | `Rectangular`, `SphericalCap`, `KNNGraph`, `RadiusGraph`, `PolygonIntersection` | `FixedLookback`, `LookbackHorizon`, `MultiScale`, `PhaseWindow`, `StencilGeometry` |
| Sampler | where do the anchors go? | `RegularStride`, `JitteredStride`, `Random`, `PoissonDisk`, `Explicit`, `ExplicitCoords`, `AlongTrack` | `RegularStride`, `Random`, `Explicit`, `StencilSampler` |
| Window | how is each cell weighted? | `Boxcar`, `Hann`, `Tukey`, `Gaussian`, `Custom` | `CausalBoxcar`, `ExponentialDecay`, `TaperedTukey`, `Periodic` |
| Aggregation | how do local outputs become global? | 21 classes: `OverlapAdd`, `Mean`, `Sum`, `Max`, `Min`, `Median`, `Mode`, `Variance`, `MeanStd`, `MinMax`, `WeightedSum`, `InvVarWeightedMean`, `HardVote`, `SoftVote`, `ByIndex`, `Learned`, `ApproxQuantile`, `ApproxCardinality`, `ApproxMode`, `StreamingHistogram`, `Reservoir` | `Mean`, `HierarchicalCombine`, `Fold`, `Forecast` |
:::

That is 38 spatial and 17 temporal classes, plus `Stencil`, `TimeStencil`
and `Closed` for coordinate-space temporal stencils. The scoping report
counted "60+"; the families, not the count, are what matters. The spatial
aggregation family is where the research-grade work sits. Overlap-add with
feathered windows, inverse-variance weighting, streaming histograms and
approximate quantiles are the merge strategies that real sliding-window
inference needs.

:::{warning} Differs from the plan
- **The prefixes are gone.** 0.9.0 renamed every `SpatialX` and `TemporalX`
  to `X` in its axis module, with no aliases: `SpatialRectangular` is now
  `geopatcher.spatial.geometry.Rectangular`, and `SpatialOverlapAdd` is
  `geopatcher.spatial.aggregation.OverlapAdd`. The axis classes are not
  re-exported at the root, so `gp.SpatialHann` no longer resolves.
- **Several temporal classes were on the wrong axis in the report.**
  `Forecast` and `Fold` are aggregations, not geometries. `Periodic` is a
  window, not a sampler. `FixedLookback` is a geometry, not a window.
  `SpatialReservoir` is an aggregation, not a sampler.
- **Two temporal names were folded into others.** `TemporalEventTriggered`
  became `temporal.sampler.Explicit(times=…)`, and `TemporalCausalRolling`
  became `temporal.sampler.RegularStride`.
:::

### The patch lifecycle

A run touches each axis in turn. The sampler places anchors, the geometry
turns each anchor into an index window, the field reads the window, the
operator transforms the patch, and the aggregation merges the outputs, using
the window's weights:

```{figure} ../images/reports/r7_patch_lifecycle.png
:label: fig-r7-lifecycle
:alt: Five panels from left to right. A 32 by 32 raster field; the same field split into 8 by 8 anchors; one 8 by 8 patch; the patch after a per-patch operator; and the merged global output. Arrows between them are labelled patcher.split, iterate, operator and patcher.merge.
:width: 100%

**Field → split → patches → operator → merge → output.** From the geopatcher repo.
```

Each `Patch` carries:
- `data`: the payload read from the field, usually an array or a `GeoTensor`;
- `anchor`: where the sampler put it, `(row, col)` for a raster;
- `indices`: the geometry's index window, such as a `rasterio.windows.Window`
  or a list of neighbour ids;
- `weights`: the window's per-cell weights;
- `masked`: set when an error policy replaced the data.

`split` is an iterator by design (the repo's ADR-001), so a field larger
than memory streams patch by patch, and `list(…)` gives the eager case.

### Overlap-add, the canonical merge

The aggregation that matters most for inference is overlap-add. Let patch
$p$ cover the cells $\Omega_p$, carry window weights $w_p(x)$, and produce
operator output $\hat y_p(x)$ for each cell $x \in \Omega_p$. The stitched
field is the weight-normalised sum over every patch that covers the cell:

$$
y(x) = \frac{\sum_{p : x \in \Omega_p} w_p(x) \hat{y}_p(x)}{\sum_{p : x \in \Omega_p} w_p(x)} .
$$ (r7-eq-ola)

A feathered window such as Hann tapers each patch towards its edges, so
seams between patches disappear. The denominator in {eq}`r7-eq-ola` makes
the reconstruction exact wherever patches overlap: if every $\hat y_p$ is the
identity, then $y(x)$ is the input.

```{figure} ../images/reports/r7_overlap_add.png
:label: fig-r7-ola
:alt: Three stacked plots against position. First, three overlapping Hann windows centred at 0, 24 and 48. Second, their accumulated weights, the denominator, which is close to one across the interior. Third, the normalised reconstruction, flat at one on the interior.
:width: 90%

**Overlap-add with feathered (Hann) windows**: the weights, their sum (the denominator in {eq}`r7-eq-ola`), and the flat normalised reconstruction. From the geopatcher repo.
```

```{code-block} python
:caption: The README quickstart in current names at geopatcher 0.10.2, from a real run.
:linenos:

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor

import geopatcher as gp
from geopatcher import spatial

# 1. Wrap any array as a Field.
ramp = np.linspace(0, 1, 64)
arr = np.outer(ramp, ramp).astype(np.float32)                 # (64, 64)
field = gp.RasterField(
    GeoTensor(values=arr, transform=rasterio.Affine.identity(), crs="EPSG:32630")
)

# 2. Compose the four axes.
patcher = gp.SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(16, 16)),     # Ωₚ: 16×16 cells
    sampler=spatial.sampler.RegularStride(step=(8, 8)),       # 50 % overlap
    window=spatial.window.Hann(),                             # wₚ(x): feathered
    aggregation=spatial.aggregation.OverlapAdd(),             # y = Σ wₚŷₚ / Σ wₚ
)

# 3. Split → operate per patch → merge.
out = [p.with_data(np.asarray(p.data) * 2.0)                  # ŷₚ = 2 · patch
       for p in patcher.split(field)]                         # 49 patches, 16×16
stitched = patcher.merge(out, field.domain)                   # → (64, 64) float64
np.abs(stitched - 2 * arr).max()                              # → 4.4e-16
patcher.merge_to_field(out, field)                            # → GeoTensor (64, 64)
```

With no overlap (stride 16, a boxcar window), both `OverlapAdd` and `Mean`
reconstruct the field exactly (`np.array_equal` is `True`). The Hann case is
exact to round-off even at the outer edge, because the outermost Hann
weights are small but not zero. One detail to know: `merge` returns
`float64` even for a `float32` field.

(r7-boundary)=
### Boundary policies

When an anchor sits close enough to the edge that its neighbourhood would
overflow the field, the geometry's `boundary` argument decides what happens:

```{figure} ../images/reports/r7_boundary_modes.png
:label: fig-r7-boundary
:alt: Four panels. A 6 by 6 patch anchored at (8, 8) overflows the field. Under drop, the default, that anchor is never emitted. Under pad, the patch is the full 6 by 6, with nodata in the overflow. Under shrink, the patch is cropped to the field and its weights crop to match.
:width: 100%

**What happens at the edge of the field.** From the geopatcher repo.
```

:::{table} Boundary policies on a 64×64 field with 16×16 patches at stride 10, from a real run.
:label: tbl-r7-boundary

| `boundary=` | Behaviour | Patches | Uncovered cells |
|---|---|---|---|
| `"drop"` (default) | overflowing anchors are never emitted | 25 | 960, left as NaN |
| `"pad"` | edge anchors emitted; the read is padded to full size with nodata | 36 | 0 |
| `"shrink"` | edge anchors emitted; the patch and its weights are cropped | 36 (sizes 14×14 to 16×16) | 0 |
| `"reflect"` | edge anchors emitted; the overflow is mirrored | 36 | 0 |
| `"raise"` | `ValueError` on the first overflowing window | — | — |
:::

The default deserves a warning of its own. With `"drop"`, a stride that does
not divide the field leaves a strip along the far edges that no patch
covers, and `merge` returns NaN there. The repo's concepts page still lists
`"reflect"` as a planned follow-up; it landed in the standalone repo's
last commits in July, just before the merge.

(r7-protocols)=
## The `Field` and `Domain` Protocols

The whole framework rests on two small Protocols, and the scoping report was
right to call them the key abstraction:

```{code-block} python
:caption: The Protocols at geopatcher 0.10.2, simplified from `_src/protocols.py`.
:linenos:

@runtime_checkable
class Domain(Protocol):
    """Metadata only: bounds and CRS, no I/O."""
    @property
    def crs(self) -> Any: ...
    @property
    def bounds(self) -> Any: ...

@runtime_checkable
class Field(Protocol):
    """A substrate the patcher reads patches out of."""
    @property
    def domain(self) -> Domain: ...
    def select(self, indexer: Any) -> Any: ...       # read one patch's data
    def with_data(self, array: Any) -> Any: ...      # rebuild around new data
```

A field may also provide `placeholder(indexer, data)`, which is duck-typed
and optional and is used by `on_error="mask"`. `AsyncField` has the same
shape with an awaitable `select`. Because the patcher only talks to these
Protocols, it does not care what carrier sits underneath: you wrap the data
once, at the boundary, and the four-axis machinery does the rest.

:::{table} The Field adapters at 0.10.2, against the scoping report.
:label: tbl-r7-fields

| Adapter | Wraps | Where, and the extra it needs | In the report |
|---|---|---|---|
| `RasterField` | georeader `GeoData` (a reader or a `GeoTensor`) | root; base install | as planned (`[raster]`) |
| `AsyncRasterField` | georeader `AsyncGeoData` | `geopatcher.fields`; base | as planned |
| `ReprojectingRasterField` | a raster read into another CRS | `geopatcher.fields`; base | **new** |
| `XarrayField` | `xarray.DataArray` | `geopatcher.fields`; `[grid]` | as planned |
| `RioXarrayField` | a rioxarray `DataArray` | `geopatcher.fields`; `[xarray-raster]` | as planned |
| `XvecField` | an xvec point cube | `geopatcher.fields`; `[point]` | as planned |
| `GeoPandasField` | `geopandas.GeoDataFrame` | `geopatcher.fields`; `[vector]` | as planned |
| `DaskField` | a dask-backed array | `geopatcher.fields`; `[dask]` | **new** |
| `CogField` | a cloud-optimised GeoTIFF, through geocloud | `geopatcher.fields`; `[cog]` | **new** |
| `MatchedField` | several co-registered fields | `geopatcher.matched` | **new** |
:::

The extras-gated adapters import lazily: `geopatcher.fields.XarrayField`
resolves on first access, and a missing extra produces a message naming the
install command. The domains (`RasterDomain`, `GridDomain`, `VectorDomain`,
`PointDomain`) live in `geopatcher.fields`; `RasterDomain` is georeader's
`GeoDataBase`.

(r7-surface)=
## Public surface

:::{table} geopatcher's public namespaces at 0.10.2.
:label: tbl-r7-surface

| Namespace | Holds |
|---|---|
| `geopatcher` (root) | `Field`, `AsyncField`, `Domain`; `Patch`, `TemporalPatch`, `SpatioTemporalPatch`; `SpatialPatcher`, `AsyncSpatialPatcher`, `TemporalPatcher`, `SpatioTemporalPatcher`; `RasterField` |
| `geopatcher.spatial` | `geometry`, `sampler`, `window`, `aggregation`, each with its classes (see {numref}`tbl-r7-axes`) |
| `geopatcher.temporal` | the same four modules, plus `stencils` (`Stencil`, `TimeStencil`, `Closed`, helpers) |
| `geopatcher.fields` | the adapters and the four domains |
| `geopatcher.matched` | `MatchedField`, `MatchedPatch` and the matched patchers |
| `geopatcher.run` | `parallel_map`, `prefetch_iterable`, `batch_split`, `unbatch`, `stack_patches`, `PatchCache`, `IndexedPatchView`, `to_delayed`, `to_dask_bag` |
| `geopatcher.observe` | `PatcherHook`, `PatchJournal`, `PatchErrorRecord`, `OnErrorPolicy`, strict mode |
| `geopatcher.config` | `{"class", "config"}` envelopes: `axis_envelope`, `from_config`, `config_from_fields` |
| `geopatcher.integrations.pipekit` | `GridSampler`, `ApplyToChips`, `MergePatches` ([](#r7-pipekit)) |
:::

A patcher does more than `split` and `merge` now. `SpatialPatcher` also has:
- `merge_to_field` (returns a `GeoTensor`) and `merge_to_xarray`;
- `patch_at` for random access to one anchor;
- `reduce` and `two_pass`, the codified helpers for operators that need
  global context;
- `to_delayed` and `to_dask_bag` for dask;
- error handling: `on_error="raise" | "skip" | "mask" | "retry"`, with
  `max_retries` and `retry_on`, and failures recorded in `patcher.errors`.

`SpatioTemporalPatcher(spatial, temporal, coupling="product" | "coupled")`
composes the spatial and temporal patchers.

(r7-beyond)=
### What shipped beyond the scope

:::{table} Shipped features the scoping report did not cover. Most are documented in the repo's recipes and design decisions.
:label: tbl-r7-beyond

| Feature | Where | What it does |
|---|---|---|
| Parallel runners | `run.parallel_map(patcher, field, operator, n_workers=8, backend="thread")` | independent per-patch jobs, on threads or processes (forkserver) |
| Streaming overlap-add | `spatial.aggregation.OverlapAdd(streaming=True, target_path=…, chunks=…)` | the accumulator on disk as zarr (or a COG), for outputs over a terabyte (ADR-002) |
| Journal and resume | `observe.PatchJournal`, `split(journal=…)` | record finished anchors; resume a killed run |
| Error policies | `on_error=`, `patcher.errors` | skip, mask or retry failing patches instead of aborting |
| Hooks | `observe.PatcherHook` | callbacks at split start, per patch, on error and at the end |
| Read cache | `run.PatchCache`, `split(cache=…)` | an LRU cache of reads, bounded by bytes, across runs |
| Random access | `run.IndexedPatchView` | a `Sequence` view over a patcher, for training data loaders (ADR-005) |
| Async prefetch | `split(prefetch=…, max_in_flight=…, max_in_flight_bytes=…)` | overlap remote reads with compute, with backpressure |
| Mixed-CRS patching | `spatial.sampler.ExplicitCoords`, `AlongTrack`, `fields.ReprojectingRasterField` | anchors in lon/lat over UTM imagery; along a satellite ground track |
| Matched fields | `geopatcher.matched` | patch co-registered fields together (ADR-003) |
| Temporal stencils | `temporal.stencils` | coordinate-space time windows such as `"-9h"` (ADR-004) |
| Config envelopes | `geopatcher.config` | serialise and rebuild patchers from `{"class", "config"}` |
| Dask and JAX | `run.to_dask_bag`, `[dask]`, `[jax]` | distributed and accelerator back-ends |
:::

(r7-pipekit)=
## The pipekit integration

The scoping report isolated the only pipekit-coupled code in geotoolz's
patcher (`_src/ops.py`, three operator wrappers) in
`geopatcher.integrations.pipekit`, behind a `[pipekit]` extra. That is what
shipped, with one rename:

:::{table} The pipekit bridge, scoped and shipped.
:label: tbl-r7-bridge

| Scoped | Shipped at 0.10.2 | What it does |
|---|---|---|
| `GridSampler(patcher)` | `GridSampler(patcher=…)` | yield patches from a field, as an operator |
| `ApplyToChips(op)` | `ApplyToChips(operator=…)` | apply an operator to every patch in an iterable |
| `Stitch(aggregation)` | `MergePatches(aggregation=…, domain=…)` (renamed in 0.10.0) | merge patches back into a field |
:::

The three compose into a sliding-window inference pipeline:

```{code-block} python
:caption: Sliding-window inference as a pipekit pipeline (a sketch; needs the [pipekit] extra).
:linenos:

import pipekit as pk
from geopatcher.integrations.pipekit import ApplyToChips, GridSampler, MergePatches

pipeline = pk.Sequential([
    GridSampler(patcher=patcher),                          # Field → Iterator[Patch]
    ApplyToChips(operator=model_op),                       # ŷₚ = model(patch)
    MergePatches(aggregation=spatial.aggregation.OverlapAdd(),
                 domain=field.domain),                     # y = Σ wₚŷₚ / Σ wₚ
])
result = pipeline(field)
```

The scoping report's three reasons for keeping the bridge in geopatcher
rather than geotoolz held:
- **It is about the patcher**: the wrappers take a patcher and an
  aggregation.
- **xrtoolz could use it too.** It doesn't yet.
- **The dependency stays opt-in.** Without `[pipekit]`, geopatcher installs
  without pipekit.

geotoolz's `patch_ops` re-exports the same three objects.

:::{warning} Differs from the plan
- **The bridge imports pipekit's private API.** It uses
  `nested_config` from `pipekit._base.operator`, the same private helper
  geotoolz imports ([Report 3](03_sister_libraries.md#r3-geotoolz)), and a
  private geopatcher helper (`_src.spatial.patcher.on_domain_grid`).
- **The install note is stale.** The extra is `pipekit>=0.0.2`, and both the
  README and the module's import error still say pipekit is not on PyPI.
- **There is only one integration.** The report's plural `integrations/`
  anticipated `pipekit_jax` and `dask` siblings. Dask and JAX arrived
  through `geopatcher.run` and the `[dask]` / `[jax]` extras instead.
- **`GridSampler` and `MergePatches` are `forbid_in_yaml`**, so a pipeline
  that contains them cannot round-trip through pipekit's loaders
  ([Report 2](02_pipekit_surface.md#r2-group-l)).
:::

(r7-deps)=
## Dependencies and extras

The scoping report's core was **numpy and scipy only**: no field adapters,
no pipekit. A user would wrap their own data in a `Field` and bring their own
adapter.

:::{table} The base install, scoped, as the README describes it, and shipped.
:label: tbl-r7-base

| Scoped | README (0.0.6) | Shipped at 0.10.2 |
|---|---|---|
| numpy, scipy | "base — RasterField only" | numpy, scipy, **georeader-spaceml**, **rasterio**, **pyproj**, typing-extensions |
:::

:::{table} The optional extras, scoped and shipped.
:label: tbl-r7-extras

| Scoped | Shipped |
|---|---|
| `raster = [georeader]` | none: georeader is in the base |
| `xarray-raster = [xarray, rioxarray]` | `xarray-raster = [rioxarray, xarray]` |
| `grid = [xarray]` | `grid = [xarray]` |
| `point = [xvec, xarray]` | `point = [xvec>=0.4]` |
| `vector = [geopandas, shapely]` | `vector = [geopandas, shapely]` |
| `pipekit = [pipekit]` | `pipekit = [pipekit>=0.0.2]` |
| `all` | `patch-full` (everything but pipekit) |
| — | `streaming = [zarr>=3.0]`, `dask`, `jax`, `cog = [geotoolz-cloud[cog]]` |
:::

:::{warning} Differs from the plan
The framework-free core does not exist. `import geopatcher` loads georeader
and rasterio, because the domains, the raster field and the matched fields
import them at module level. pyproj is imported lazily. The README's "base —
RasterField only" was already slightly off at 0.0.6, whose base included
georeader, scipy and pyproj. Someone patching an xarray grid or a
GeoDataFrame installs the raster stack anyway. See
[](#r7-open-questions).
:::

**No geocatalog dependency.** This holds exactly. You can patch your own
data without geocatalog, and find files with geocatalog without geopatcher.
When you want both, `geocatalog.patch.field_for` hands a catalog to
geopatcher as a `RasterField`, and `CatalogDomain` satisfies `Domain`
([Report 6](06_geocatalog.md#r6-bridge)). Neither package imports the other
at module level.

(r7-migration)=
## Migration from geotoolz

The scoping report planned to:
- move `geotoolz.patch` into geopatcher, except `_src/ops.py`, which goes to
  `integrations/pipekit`;
- re-export everything from geotoolz (`gz.SpatialPatcher`, `import
  geopatcher as patch`) for two minor versions;
- leave a small geotoolz `patch_ops` module with GeoTensor-aware operators
  (`ReadFromCatalogAsPatches`, `GeoTensorChipFilter`).

:::{table} The migration steps at the pinned revisions.
:label: tbl-r7-migration

| Step | Status |
|---|---|
| Patcher code leaves geotoolz for its own package | **done**: geopatcher, now `geotoolz-patcher` in the same monorepo |
| `_src/ops.py` becomes `geopatcher.integrations.pipekit` | **done** |
| geotoolz re-exports the patcher (`gz.SpatialPatcher`, `gz.patch`) | not done; geotoolz has no patcher names at its root |
| A geotoolz `patch_ops` module | **done, differently**: it re-exports `GridSampler`, `ApplyToChips` and `MergePatches`, and adds `TriangularWindow`, `StratifiedSample` and `BalancedSampler` (its `[patch]` extra) |
| `ReadFromCatalogAsPatches`, `GeoTensorChipFilter` | not built |
| xrtoolz adopts geopatcher | not done; xrtoolz mentions only the older `xrpatcher`, in docstrings |
:::

The scoping report's cross-library examples still work in current names:

```{code-block} python
:caption: Patching an xarray grid with no geotoolz in sight (a sketch; needs the [grid] extra).
:linenos:

import xarray as xr
import geopatcher as gp
from geopatcher import spatial
from geopatcher.fields import XarrayField

field = XarrayField(xr.open_dataset("reanalysis.nc")["sst"])
patcher = gp.SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(32, 32)),
    sampler=spatial.sampler.RegularStride(step=(16, 16)),
    window=spatial.window.Boxcar(),
    aggregation=spatial.aggregation.Mean(),
)
for patch in patcher.split(field):     # patch.data is an xr.DataArray
    ...
```

The scoping report's JAX example also holds as a pattern: the patcher
iterates in Python, the patches are stacked, and a jitted, vmapped function
processes them all at once. `run.batch_split` and `run.stack_patches` now
give that stacking step a name. A pipeline built from pipekit operators
still cannot be jitted ([Report 3](03_sister_libraries.md#r3-other)).

(r7-monorepo)=
## Monorepo, release and timing

The scoping report picked a single "pipekit-ecosystem" monorepo for the
patcher. Its reasons were especially strong here: six field adapters across
five ecosystems, 60-odd classes to test across all of them, and a
cross-package bridge. As with geocatalog, the patcher landed in the
**geotoolz** monorepo instead, beside geocatalog and geocloud, with
per-package releases (`geotoolz-patcher-v0.10.2`). There are 60 test modules
(about 19,000 lines), including a pipekit integration test.

:::{table} geopatcher's history.
:label: tbl-r7-history

| When | What |
|---|---|
| May – July 2026 | standalone geopatcher repo, 0.0.1–0.0.6: the four axes, the fields, async, streaming overlap-add, journals, error policies, boundary modes; its last commits add `reflect`, CRS-aware patching and `PatchCache` |
| 13 July 2026 | geopatcher and geocatalog merge into the geotoolz monorepo (`geotoolz-patcher` 0.1.0); the standalone repo is later archived |
| 2 – 8 October 2026 | 0.3.0 to 0.10.0, with breaking changes: object storage split into geocloud; organised by task (`run`, `observe`); prefixes dropped (0.9.0); `Stitch` renamed `MergePatches` (0.10.0) |
| 12 October 2026 | 0.10.2, the pinned revision |
:::

The estimate was a week and a half of focused work, in one coordinated v1.0
with the pipekit and geocatalog extractions. It was gradual instead: one
merge in July, then a series of breaking releases, each absorbed per
package. What the split was meant to unblock did happen. Aggregations,
adapters (`CogField`, `DaskField`), samplers (`AlongTrack`) and the run
machinery all shipped on the patcher's own release cycle.

(r7-tradeoffs)=
## Trade-offs

The scoping report's list, with how each turned out.

**What got better:**
1. **The patcher gets its own billing**: in part. The repo and the import
   name are `geopatcher`, but the distribution is `geotoolz-patcher`, and the
   standalone repo that a search would find is archived
   ([](#r7-open-questions)).
2. **The `Field` / `Domain` Protocols get visibility**: yes. They are the
   root's first exports, and geocatalog's bridge depends on them.
3. **The aggregations get their own roadmap**: yes. The family grew from
   about fifteen to 21 classes.
4. **The async story is clean**: yes. `AsyncSpatialPatcher`,
   `AsyncRasterField`, prefetch and backpressure are first-class.

**What got worse, and the mitigations:**
1. **The test surface is big**: 60 modules, in one monorepo CI.
2. **More extras to coordinate**: each adapter stays behind its own extra,
   as planned.
3. **More documentation**: the docs are tutorial-grade (concepts, a
   patching guide, recipes, ADRs, a notebook), but they live in the archived
   repo, with pre-0.9.0 names.
4. **People forget `[pipekit]`**: the import error says what to install,
   though its PyPI note is out of date.

(r7-open-questions)=
## Open questions

1. **Where do the docs live now?** The README, concepts, recipes, ADRs and
   figures are in an archived repo, and they use names removed in 0.9.0.
   Either they move into the geotoolz monorepo, or the archived repo is
   unarchived as the docs site. This mirrors geocatalog's question 1
   ([Report 6](06_geocatalog.md#r6-open-questions)).
2. **`geopatcher` or `geotoolz-patcher` on PyPI?** The README's badge and
   install lines say `geopatcher`; the monorepo publishes `geotoolz-patcher`.
3. **Should the core be georeader-free?** The scoped core was numpy plus
   scipy, so that xarray, vector and point users would not pull in the
   raster stack. Moving the raster domain and fields behind a `[raster]`
   extra would restore that, at the cost of one more extra for the common
   case.
4. **Should the axis classes be reachable from the root again?** With the
   prefixes gone, `Rectangular`, `Mean` and `RegularStride` exist on both the
   spatial and the temporal side, so the root cannot re-export them flatly.
   Module-qualified imports (`spatial.geometry.Rectangular`) are the price of
   that.
5. **Should `drop` stay the default boundary?** It silently leaves NaN
   strips when the stride does not divide the field (960 cells in the
   example above). `"pad"` or `"reflect"`, or at least a warning on
   uncovered cells, would make the default safe.
6. **Should the pipekit bridge stop using private API?** Making
   `nested_config` public in pipekit fixes this for both geopatcher and
   geotoolz.
7. **Should `GridSampler` and `MergePatches` be serialisable?** They are
   flagged `forbid_in_yaml`, so a sliding-window pipeline cannot be saved
   and reloaded, which matters for the deployment use cases.
8. **What do geotoolz and xrtoolz actually need from the patcher?** The
   scoped GeoTensor-aware operators (`ReadFromCatalogAsPatches`,
   `GeoTensorChipFilter`) were never built. xrtoolz has no patcher
   integration, and still refers to the older xrpatcher.

(r7-todos)=
## Upcoming work

:::{table} Follow-ups from checking the scoping report against the repo and the code.
:label: tbl-r7-todos

| Follow-up | Where | Why |
|---|---|---|
| Update the README and docs to the 0.9.0 names and `MergePatches` | geopatcher docs | `gp.SpatialRectangular`, `gp.SpatialOverlapAdd` and `Stitch` no longer exist |
| Mark `"reflect"` as shipped in the concepts page | geopatcher docs | it still says "planned follow-up" |
| Fix "base — RasterField only" in the README | geopatcher README | the base includes georeader, rasterio, pyproj and scipy |
| Drop the "pipekit isn't on PyPI" note | README; `integrations/pipekit.py` import error | the extra is `pipekit>=0.0.2` |
| Make `nested_config` public, and use it from geopatcher and geotoolz | pipekit, geopatcher, geotoolz | private API across packages ([Report 3](03_sister_libraries.md#r3-todos)) |
| Warn when `boundary="drop"` leaves cells uncovered | `spatial.geometry.Rectangular` or `merge` | silent NaN strips |
| Return the field's dtype from `merge`, or document `float64` | `SpatialPatcher.merge` | a `float32` field merges to `float64` |
| Index geopatcher's namespaces in the capability index | geostack `scripts/capabilities.py` | only 11 root names are listed today |
:::

(r7-references)=
## References

<!-- TODO(ref): placeholders. When a report page lands, point its links in this
file at the page (e.g. [Report 4](04_use_cases.md)) and delete its entry
below. When citations land, add them to references.bib, cite with [@key] in the
text, and replace the citations placeholder. -->

:::{note} Placeholders
:class: simple
Reports 4 and 5 are still being written. The entry below stands in for them
so the links in this page already resolve. Reports
[1](01_toolz_primer.md), [2](02_pipekit_surface.md),
[3](03_sister_libraries.md) and [6](06_geocatalog.md) are published.
:::

### Forthcoming reports

(r7-ref-report-4)=
Reports 4 and 5
: *Forthcoming.* The rest of the pipekit stack, including the use-case
  revisit and pipekit-jax.

### Sources

- The [geopatcher repo](https://github.com/jejjohnson/geopatcher) (archived):
  the README, concepts, the patching guide, design decisions (ADR-001 to
  ADR-006) and recipes. The four figures on this page are its own.
- `geotoolz-patcher` 0.10.2 in the geotoolz monorepo, and its changelog.

*Citations to be added:* overlap-add reconstruction, and Hann and Tukey
windows, as entries in `references.bib`.
