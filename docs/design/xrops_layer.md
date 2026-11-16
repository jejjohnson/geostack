---
title: "xrops as the xarray layer"
subtitle: "Design · kernels written once on plain arrays, run on DataArray through KernelOp, and xrops keeping only the work that needs labels"
short_title: xrops layer
description: "xrops stops holding array maths: duplicates of geoops and functax go, every remaining kernel is split into an array module and run through pipekit-array's KernelOp, and the kernels move down to gridax, diagnostix, geophysix, tracerax and functax as those libraries come to exist, leaving xrops with labels, CF, datasets, laziness and plots."
date: 2026-11-12
keywords:
  - xrops
  - xrtoolz
  - KernelOp
  - diagnostix
  - gridax
  - geoops
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

xarray runs array maths through `apply_ufunc`, so array maths does not need to be written in xarray. We propose that xrops keeps only the work that needs labels and runs every kernel it uses through pipekit-array's `KernelOp`, with the kernels written once on plain arrays in the lowest library that may hold them.

- **XR-D1. Duplicates go first.** Where xrops and geoops (or functax) implement the same maths, xrops calls the lower copy and deletes its own.
- **XR-D2. Kernels split out inside xrops.** Every family that holds array maths gets an `_src/array.py` with no xarray import, and its operators run those kernels through `KernelOp` instead of a hand-written `apply_ufunc` wrapper per module.
- **XR-D3. Kernels move down when their library exists.** Once split, each array module moves to its home on the [stack layer map](./stack_layer_map.md): gridax, diagnostix, geophysix, tracerax or functax. xrops keeps the operator with the same name.
- **XR-D4. xrops keeps the label work.** Climatologies and anomalies, resampling by time, CF attributes and units, `Dataset` and `DataTree` handling, dask halos, CRS handling and plotting stay in xrops, because none of them has a plain-array form.

This page answers PL-Q12 on [package layout](./package_layout.md). If it is decided, it supersedes OP-O1 (state-domain kernels live in xrops) and the xrops-L1 part of SL-O1, and it narrows PL-Q6: the `atm`, `ocn` and `budgets` families stay in xrops as operators on `Dataset`, while their kernels move to diagnostix. It assumes nothing about [one workspace](#design-one-workspace) (PL-F), though every step is one PR instead of two if PL-F lands first.

## What xrops holds today

We read the `main` branch of xrtoolz on 12 Nov 2026. `xrtoolz-ops` has about 35,900 lines in 188 files, of which about 10,100 are docstrings. The split between kinds of code below is an estimate from sampling each family's files and counting their `apply_ufunc` calls and imports, not a measurement.

| Family | Lines | Kernel | Delegated | Needs labels |
| --- | --- | --- | --- | --- |
| `metrics` | 5,979 | 40% | 20% | 40% |
| `interpolate` | 5,722 | 40% | 15% | 45% |
| `viz` | 4,730 | 0% | 5% | 95% |
| `transforms` | 3,607 | 25% | 35% | 40% |
| `grad` | 2,571 | 20% | 15% | 65% |
| `geom` | 2,479 | 5% | 55% | 40% |
| `ocn` | 2,309 | 30% | 0% | 70% |
| `atm` | 1,682 | 25% | 0% | 75% |
| `wavelets` | 1,446 | 60% | 15% | 25% |
| `budgets` | 947 | 20% | 0% | 80% |
| the rest (`learn`, `cf`, `mask`, `anomaly`, `extremes`, `inference`, `utils`, the base) | 4,400 | about 5% | about 20% | about 75% |
| **All** | **35,900** | **about 25%** | **about 20%** | **about 55%** |

"Kernel" means array maths that would run on NumPy or JAX given the positions of its dims; "delegated" means calls to xrft, xskillscore, regionmask, pywt, scikit-image, scikit-learn, functax, rioxarray or pipekit-array; "needs labels" means work that reads coordinates by value, groups by time, sets CF metadata, handles several variables, plans dask chunks or draws.

Four findings shape the proposal.

1. **About a quarter of xrops is kernel maths, and little of it is written as a kernel.** `metrics` has separate NumPy kernels (`_pixel_kernels.py`, `_segmented_psd_kernels.py`) and `interpolate` has `_smooth_kernels.py`, but most families write the maths inline in functions that take a `DataArray`. `ocn`, `atm` and `budgets` are written as labelled arithmetic, for example `u = -(grad.GRAVITY / f_field) * deta_dy` in `ocn/_src/kinematics.py`, followed by CF attributes.
2. **There is no shared way to run a kernel on a `DataArray`.** Each module writes its own `apply_ufunc` wrapper (`_apply_pixel_kernel` in `metrics/_src/pixel.py`, `_apply_kernel_to_dataarray` in `interpolate/_src/smooth.py`). `xrops.lift` exists but is narrow: it runs a geoops geometry operator item by item along one dim, through `GeoArray.from_xarray` and `to_xarray`, and nothing inside xrops uses it.
3. **Only `grad` calls geoops, and it duplicates part of what it calls.** `grad/_src/cartesian.py` imports `geoops.geom.difference` and `difference_offsets`, but its `_difference_adaptive` and `_select_by_finiteness` are the same algorithm as `geoops.geom._src.array.adaptive_difference`, which was ported from xrtoolz's design.
4. **Most of xrops needs labels and stays.** `viz` alone is 4,730 lines, `anomaly` is all `groupby("time.dayofyear")` and rolling windows, and most of `grad` is dask `map_overlap` halo planning and coordinate spacing rather than stencils.

## Duplicates

These are the overlaps between xrops and geoops that the audit found. Only the first was compared line by line; the rest are candidates whose maths has to be compared before one copy is deleted.

| Maths | xrops | geoops or functax | Status |
| --- | --- | --- | --- |
| adaptive finite difference (the first finite of central, forward and backward, at falling accuracy) | `grad/_src/cartesian.py`: `_difference_adaptive`, `_select_by_finiteness` | `geoops.geom._src.array.adaptive_difference` | the same algorithm |
| gap fill (Laplace, biharmonic, IDW) | `interpolate/_src/gap_fill.py` and its own bridge `_functax.py`, plus a NumPy red–black SOR for `boundary="wrap"` | `geoops.restore._src.array` (`gap_fill_laplacian`, `gap_fill_biharmonic`, `gap_fill_idw`) over the same functax plans | two bridges to one library |
| morphology (remove small objects, opening, clean mask) | `transforms/_src/morphology.py`, on scikit-image | `geoops.mask._src.array`, on `scipy.ndimage` | same maths, two backends |
| points to a grid (binning, IDW) | `interpolate/_src/binning.py`, `points_to_grid.py` | `geoops.geom._src.coregister.array` (`points_to_raster_binned`, `points_to_raster_idw`) | candidate |
| divergence, curl, Laplacian | `grad/operators.py`, with the spherical metric `1/(R cos φ)` | `geoops.geom` (`divergence_2d`, `curl_2d`, `laplacian_2d`) | candidate; the spherical metric is xrops' own |
| Gaussian smoothing | `interpolate`'s `gaussian_smooth` (`gaussian_filter1d`) | `geoops.restore`'s `gaussian_denoise` | candidate |

## How a kernel runs on a DataArray

pipekit-array's `KernelOp` already does what each xrops module does by hand. A kernel declares the dims it consumes and produces as a generalized-ufunc signature, such as `"(c)->()"`, and `KernelOp` runs it on raw arrays as is, on named carriers by moving the core dims last by name, and on a `DataArray` through the xarray adapter's own path, one `apply_ufunc` that stays lazy on dask. geoops' index operators are built this way (`geoops._src.kernels.band_kernel` over `indices/_src/array.py`), and GA-Q5 already lets them take a `DataArray` directly.

The kernel conventions decide what fits.

- **Fits:** pointwise and reduction kernels over named dims (metrics, scores, spectral indices, column conversions, wind speed, the equation of state); kernels with non-carrier parameters, passed as keywords; geometry that varies per pixel, such as latitude in `1/(R cos φ)`, passed as a positional input with the carrier's dims.
- **Does not fit, and stays in xrops:** grouping and resampling by coordinate value; stencils that need halos across dask chunks, where xrops keeps its `map_overlap` planning and calls the stencil kernel inside it; work across several variables of a `Dataset`, where the operator picks the variables by name and calls one kernel per output.

An operator in xrops then has three parts, as rule 7 on the package layout page and [the three levels](#design-levels) already require: the kernel (in xrops' `_src/array.py` for now, in a lower library later), a `KernelOp` that runs it, and the xarray work around it (picking variables, coordinates in, CF attributes out).

## Where the kernels go

The homes come from each library's "Takes over" line on its design page and from the [stack layer map](./stack_layer_map.md).

| Kernels in xrops today | Home | What stays in xrops |
| --- | --- | --- |
| the partials and `grid_metrics_from_coords` of `grad`; the vertical remap kernels behind `remap_axis` | [gridax](./libraries/gridax.md) | the `grad` operators, dask halos, coordinate spacing |
| the maths of `ocn`, `atm` (vertical and wind) and `budgets`; climatology and anomaly maths; the spectra and decompositions of `transforms` and `wavelets`; the scores in `metrics` | [diagnostix](./libraries/diagnostix.md) | the operators on `Dataset`, variable names, CF attributes, the `groupby` and rolling logic of `anomaly` |
| Earth's radius, gravity, Ω, Coriolis, seawater density, column-unit conversions | [geophysix](./libraries/geophysix.md) | nothing |
| gap fill, IDW, RBF and conservative regridding | [functax](./libraries/functax.md) | the `interpolate` operators |
| the empty Lagrangian stub in `metrics/_src/lagrangian.py` | [tracerax](./libraries/tracerax.md) | DataArray views of trajectories |
| morphology and point binning | geoops' copies (see the duplicates) | the operators |
| CF, CRS handling in `geom`, regionmask masks, `learn`, `inference`, `viz` | stays | all of it |

None of these libraries can take code yet. gridax, geophysix and tracerax have no repository. The `diagnostix` repository was created on 9 Nov 2026 and still holds only the package template (`src/mypackage`). This is why XR-D2 comes before XR-D3: splitting each family's kernels into an array module inside xrops is useful on its own, because it removes the per-module wrappers and makes each kernel testable on NumPy and JAX. It also makes the later move a plain file move with no change to the operators.

## Stages

Each stage is a set of issues on xrtoolz (or geotoolz, after PL-F) and lands family by family.

1. **Remove the duplicates (XR-D1).** Start with the adaptive difference, which is confirmed. For each candidate in the duplicates table, compare the maths on the fixtures of both packages, keep the lower copy, and make xrops call it.
2. **One kernel path (XR-D2).** xrops gets a `kernel_op` helper on pipekit-array's `KernelOp`, as geoops has `band_kernel`, and the per-module `apply_ufunc` wrappers go. Then each family in the table above, in order of kernel share (`wavelets`, `metrics`, `interpolate`, `ocn`, `transforms`, `atm`, `grad`, `budgets`), gets its `_src/array.py`, with labelled arithmetic rewritten as functions of arrays and dim arguments.
3. **Move down (XR-D3).** As each target library is adopted and released, the matching array modules move there with their tests, and xrops imports them. geophysix's constants go first, because every other library uses them.
4. **What remains (XR-D4).** After stage 3, xrops holds operators, label work and plots. Its family names still match geoops' where the job is the same (gate D1 on the package layout page).

## Gates

| Gate | Passes when |
| --- | --- |
| X1 No duplicates | each row of the duplicates table is resolved, and a test asserts that xrops' adaptive difference is geoops' function |
| X2 One kernel path | no xrops module outside its kernel helper calls `xr.apply_ufunc` on a kernel of its own |
| X3 Array modules | every `_src/array.py` in xrops imports neither xarray nor xrops, checked by an import scan |
| X4 Every backend | `check_kernel_op` passes for each kernel on NumPy, on JAX and on a dask-backed `DataArray` |
| X5 Same results | each moved operator matches its output from before the move on the existing fixtures, within float tolerance |
| X6 Still lazy | an operator given a dask-backed `DataArray` returns a dask-backed result without computing |
| X7 Same public API | xrops' `__all__` lists the same names before and after each stage |

## Risks

- **Rewriting labelled arithmetic costs effort.** `ocn`, `atm` and `budgets` are short formulas in xarray broadcast style, and as kernels they need dim arguments and explicit inputs. The gain is that the same kernel runs on NumPy, JAX and dask; X5 holds the results fixed while they move.
- **Speed.** One `apply_ufunc` per kernel can be slower than fused xarray arithmetic on small fields. We measure it on the existing benchmarks before stage 2 reaches `ocn`, and a family whose operators get measurably slower keeps its arithmetic form.
- **The target libraries may change shape.** gridax, diagnostix, geophysix and tracerax are designs, and their APIs can change before they take code. Stage 2 keeps every kernel inside xrops until its home exists, so a change there moves a file, not an operator.

## Open questions

| ID | Question | Recommended | Alternative | Status |
| --- | --- | --- | --- | --- |
| XR-Q1 | Does xrops expose an operator for a kernel that needs no label work? | only when it adds label work (CF attributes, variable names, coordinates); otherwise callers run the lower library's `KernelOp` on a `DataArray` directly, as GA-Q5 allows for geoops | an xrops operator for every kernel, so that every xarray user finds everything in xrops | open |
| XR-Q2 | Morphology: which backend survives | `scipy.ndimage`, as in geoops, which has no scikit-image in its slim base | scikit-image, which xrops uses today | open |
| XR-Q3 | Where the spherical metric `1/(R cos φ)` lives | gridax, with the other grid metrics | xrops, as label work on latitude | open |
| XR-Q4 | Does `xrops.lift` stay | yes, for geoops geometry operators, which `KernelOp` cannot express | fold it into the kernel helper | open |
