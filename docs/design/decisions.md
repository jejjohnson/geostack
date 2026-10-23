---
title: "Design decisions and open questions"
subtitle: "Design · one register, one ID per decision, linked to the page that argues it"
short_title: Decisions
description: "Every decision and open question across the GeoStack design pages, with a unique page-prefixed ID, its status and a link to the page that sets out the options."
date: 2026-10-23
keywords:
  - decisions
  - open questions
  - design
  - register
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

Each design page argues its own decisions, with the options and the recommendation; this page only lists them, so one ID means one question everywhere.
An ID is the page code, a hyphen and the page's own label: `OP-O8` is open question O8 on the operators page.
When a status changes, change it on the page first, then here.

| Code | Page |
| --- | --- |
| VI | [Vision](./vision.md) |
| CS | [Case studies](./case_studies.md) |
| GA | [geoarray](./geoarray.md) |
| LL | [Labeled lane](./labeled_lane.md) |
| PK | [pipekit kernels](./pipekit_kernels.md) |
| OP | [Operators](./operators.md) |
| PL | [Package layout](./package_layout.md) |
| SL | [Levels across the stack](./stack_levels.md) |
| ES | [esmax](./esmax.md) |
| NF | [Neural fields](./neural_fields.md) |

Statuses: **decided** (with the date), **forced** (follows from the dependency rules, no real choice), **open** (not yet decided; the page gives the recommendation and the alternative), **superseded** (replaced; the page says by what), **deferred**.
Testing gates are not listed here; each page keeps its own.

## Decide first

These change a principle or block other work.

| ID | Question | Why first |
| --- | --- | --- |
| [CS-F1](./case_studies.md) | A readable type becomes a dim signature that any carrier can match | changes vision principle 5 and what `GeoImageSeries` means |
| [CS-F2](./case_studies.md) | One L3 algorithm may take carriers from two families | changes vision principle 4 |
| [CS-F11](./case_studies.md) | Packaged pipelines are versioned and record provenance | a new principle; most of the missing pipelines page |
| [OP-O8](./operators.md) | May an L3 algorithm call another L3 algorithm inside a differentiated cost | decides how esmax and vardax compose |
| [GA-Q2](./geoarray.md) / [LL-F1](./labeled_lane.md) | The CRS encoding on the xarray side | every converter writes it |
| [GA-Q5](./geoarray.md) | Do geoops kernel operators accept a `DataArray` directly | the lane rule LL-H1 waits on it |

## Vision (VI)

| ID | Question | Status |
| --- | --- | --- |
| VI-V1 | Where packaged pipelines live: each in the package that owns its domain (`plumax.pipelines`, `esmax_ocean.pipelines`) | open |
| VI-V2 | Which run modes are first-class at first: eager and compiled, lazy through `apply_ufunc` | open |
| VI-V3 | Which pipelines to package first: methane plume detection, SLA mapping | open |
| VI-V4 | How a pipeline runs concurrently: the pipekit runner schedules synchronous operators; I/O stages are concurrent | open |
| VI-V5 | Whether pipeline checks can be switched off: entry checks always on; per-stage dim checks may be turned off | open |

## Case studies (CS)

Findings from running thirteen workflows through the vision; each proposes an edit.

| ID | Finding | Status |
| --- | --- | --- |
| CS-F1 | Readable types stop at `GeoArray`, so lazy inputs have none | open (decide first) |
| CS-F2 | One L3 algorithm takes carriers from two families | open (decide first) |
| CS-F3 | Fixed shapes before `jit`: padding is part of the named crossing (`to_jax(pad_to=...)`) | open |
| CS-F4 | The runner fans out I/O and carries state between cycles | open; for the pipelines page |
| CS-F5 | Derived geometry is more than spacing: time and viewing geometry, unit vectors | applied to [the three levels](#design-levels) |
| CS-F6 | Results can be records: a `Detections` type on `GeoPoints["variable obs"]` | open |
| CS-F7 | xrops wraps the native xarray operations pipelines need (`Climatology`, `Anomaly`, `Resample`) | open |
| CS-F8 | Operators declare `backends`; eager-only ones are listed | open (see OP-O6) |
| CS-F9 | A pipeline slot says whether it expects a fitted operator; `replace` checks | open |
| CS-F10 | `geoarray.sample` held equal to `xr.interp`; `GeoPoints` to and from data frames | open |
| CS-F11 | Packaged pipelines are versioned, record provenance and re-run idempotently | open (decide first) |
| CS-F12 | Carrier-free geometry helpers, and one documented path L1 → `KernelOp` → pipeline | open |
| CS-F13 | Runners own splits and hold back truth | open; for the pipelines page |
| CS-F14 | `pipelines.register(...)` promotes a graph to a pipeline | open; for the pipelines page |
| CS-F15 | A dim convention for distribution summaries (`moment` or `stat`) beside `member` | open |
| CS-F16 | Every L3 algorithm exposes its L1 loop; seeds come from the run | open |
| CS-F17 | Ahead-of-time compile for apps; the slim geoops base is required | partly decided (the slim base, with [the tiers](#design-tiers)) |

## geoarray (GA)

| ID | Decision or question | Status |
| --- | --- | --- |
| GA-A1 | Compose a frozen dataclass; do not subclass `ndarray` | open |
| GA-A2 | Float64 transform leaf; `to_jax` requires x64 | open |
| GA-A3 | `FrozenAttrs` keeps whole-carrier metadata; per-band metadata lives on a `SpectralAxis` | open (narrowed 21 Oct 2026) |
| GA-A4 | No fill value by default | open |
| GA-B1 | `map` and `combine` instead of arithmetic dunders | open |
| GA-B2 | Per-band attrs follow a band selection | superseded 21 Oct 2026 by coordinates |
| GA-C1 | Eager `np.asarray` copies to host and raises under transforms; mixed NumPy and JAX inputs refuse | decided 17 Oct 2026, partly superseded 21 Oct 2026 |
| GA-D1 | One coordinated breaking release from `GeoTensor` to `GeoArray` | decided 17 Oct 2026 |
| GA-E1 | JAX kernels in `geoops.functional` | superseded 22 Oct 2026 by the kernel model on [operators](./operators.md) |
| GA-T1 | `GeoImageSeries` and `GeoTimeSeries` as the series names | decided 21 Oct 2026 |
| GA-T2 | `band` and `variable` are separate dims | decided 21 Oct 2026 |
| GA-T3 | `level` is allowed on observations | decided 21 Oct 2026 |
| GA-T4 | coordax is an optional converter, not the JAX lane's foundation | decided 21 Oct 2026 |
| GA-Q1 | What esmax and vardax take at their boundary | open |
| GA-Q2 | The georeference in xarray: CF `grid_mapping` plus `GeoTransform` (same as LL-F1) | open (decide first) |
| GA-Q3 | Shared dim-level functions live in pipekit-array | decided 21 Oct 2026 |
| GA-Q4 | What happens to geoops compositing | open |
| GA-Q5 | Do geoops operators accept a `DataArray` | open (decide first) |
| GA-Q6 | `map_over` on xarray | open |
| GA-Q7 | Dim-name conventions | open |
| GA-Q8 | The name of the curvilinear carrier | open |
| GA-Q9 | `check_dims` on `pipekit.Signature` | decided 21 Oct 2026 |
| GA-Q10 | Edit geotoolz issues #511–#534 in place | open |

## Labeled lane (LL)

| ID | Decision or question | Status |
| --- | --- | --- |
| LL-F1 | CRS as a `spatial_ref` coordinate with `crs_wkt` and `GeoTransform` | open (same question as GA-Q2) |
| LL-F2 | Per-band metadata: `band` holds names; `wavelength` and FWHM are non-index coordinates | open |
| LL-F3 | Multi-resolution products as a `DataTree`, one node per grid | open |
| LL-G1 | Where a coordax grid coordinate lives | deferred until esmax or vardax asks |
| LL-G2 | Accept coordax's static grid; batching and gradients stay on carriers | decided 21 Oct 2026 (with GA-T4) |
| LL-H1 | Operators on labeled data: geometry operators through `xrops.lift`; kernel operators on `DataArray` | lifting decided 22 Oct 2026 (OP-O4); the rest waits on GA-Q5 |
| LL-H2 | geoarray owns the converters, behind extras | decided |

## pipekit kernels (PK)

| ID | Decision or question | Status |
| --- | --- | --- |
| PK-P1 | Adapter interface and registry in pipekit core; adapters and dim-level functions in pipekit-array | forced |
| PK-P2 | A kernel's spec is declared at the operator layer, not by a decorator | open |
| PK-P3 | Abstract core-dim letters plus a `core=` mapping | open |
| PK-P4 | The xarray adapter lives in pipekit-array `[xarray]` | open |
| PK-P5 | `DataArray` only at first; `Dataset` and `DataTree` dispatch stays in xrops | open |
| PK-P6 | `pipekit_jax.to_jax` lives in pipekit-jax | forced |
| PK-P7 | The carrier for outputs that drop a spatial dim: `pipekit_array.DimArray` (data, dims, labels; no georeference, alignment or arithmetic) | decided 23 Oct 2026 |

## Operators (OP)

| ID | Decision or question | Status |
| --- | --- | --- |
| OP-O1 | Observation and sensor kernels in geoops; state-domain kernels (MLD, budgets) in xrops, which never re-implements a geoops kernel | decided 22 Oct 2026 |
| OP-O2 | One named-dims einx bridge in pipekit-array `[einx]` | open |
| OP-O3 | `to_rows` / `from_rows` plus one scikit-learn estimator wrapper | open |
| OP-O4 | Lifting is `xrops.lift(op, dim=...)`; geoops has no `[xarray]` extra | decided 22 Oct 2026 |
| OP-O5 | Operators use `KernelOp` by composition, not subclassing | open |
| OP-O6 | The `backends` vocabulary: `"numpy"`, `"jax"`, `"xarray"` | open |
| OP-O7 | The diagnostic gradient kernel lives in geoops; model-grade calculus stays in finitevolX | decided 22 Oct 2026 |
| OP-O8 | Whether an L3 algorithm calls another L3 algorithm inside a differentiated cost | open (decide first) |

## Package layout (PL)

| ID | Decision or question | Status |
| --- | --- | --- |
| PL-Q1 | The xarray bridges | superseded by OP-O2, OP-O3 and OP-O7 |
| PL-Q2 | The xarray `Operator` base in `xrops._src` | open (its `Signature` part decided with GA-Q9) |
| PL-Q3 | The reader's name: `xrproducts` | open |
| PL-Q4 | How to split `xrtoolz.geo` | open |
| PL-Q5 | How long the shims live | open |
| PL-Q6 | State-physics families stay in xrops | decided 22 Oct 2026 (OP-O1) |
| PL-Q7 | geoproducts or xrproducts, by what one read returns | open |
| PL-Q8 | `load_states` stays in geocatalog | open |
| PL-Q9 | Points: a CF DSG `Dataset` on the labeled side, `GeoPoints` on the array side | `GeoPoints` decided; DSG open |

## Levels across the stack (SL)

| ID | Decision or question | Status |
| --- | --- | --- |
| SL-O1 | Ocean and atmosphere diagnostics: option D (xrops L1 and L2; model closure to esmax; products to xrproducts) | open |
| SL-O2 | Carrier code behind `[geo]` and `[xarray]` extras (`esmax-core[geo]`) | open |
| SL-O3 | CH₄ column physics: L1 in radtrax, xarray operators behind `radtrax[xarray]` | open |
| SL-O4 | A `plumax.runs` module per model for L3 runs | open |
| SL-O5 | Fix filterax and optax_bayes missing from the capability index before classifying them | open |
| SL-O6 | What `xtremax.simulations` is for | open |
| SL-O7 | somax's future is the esmax workspace | open (see ES-N1) |

## esmax (ES)

| ID | Decision or question | Status |
| --- | --- | --- |
| ES-N1 | The umbrella name: `esmax` | open |
| ES-N2 | One distribution per component, like pyrox | open |
| ES-N3 | Land and ice start only with real code (phase P4) | open |
| ES-N4 | `les_fvm` stays in plumax | open |
| ES-N5 | Synthetic climates are not in esmax | open |

## Neural fields (NF)

| ID | Decision or question | Status |
| --- | --- | --- |
| NF-D1 | Deterministic neural-field parts in geonnax, probabilistic in pyrox-nn, assimilation in vardax; case 9, case 11 and methane stage 3 are the acceptance tests | open (proposed; Wave 6) |
| NF-O1 | Averaging kernels: field on pressure levels or one collapsed weight per pixel | open |
| NF-O2 | pyrox `Parameterized` registry under `eqx.tree_at` rebuilds | open |
| NF-O3 | geonnax field names pyrox-nn swaps are added to, never renamed | open |
| NF-O4 | Error bars for neural-field 4D-Var: ensemble spread from the start | open |
| NF-O5 | SWOT wave-correlated noise in case 9 | open |
| NF-O6 | Ownership of field derivatives (`geonnax.diffops` vs `xrgrad` vs spectraldiffx) | open |
| NF-O7 | Whether compression is worth building | open (research, geostack#86) |
