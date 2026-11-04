---
title: "Operator coupling: which dims an operator mixes"
subtitle: "Design · a coupling class per operator, measured by perturbation and declared on the class"
short_title: Operator coupling
description: "Classifies every geoops operator by the dims along which its output depends on other input indices (pointwise, spectral, spatial, temporal and their combinations), with the effect on each dim, the spatial reach and the fit step, measured by a perturbation probe over all 280 operator classes."
date: 2026-11-04
keywords:
  - geoops
  - operators
  - coupling
  - classification
  - patching
  - cheat sheet
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

## Summary

Every geoops operator gets a **coupling class**: the set of dims along which some output depends on two or more input values that differ along that dim.
An operator that couples nothing is pointwise, one that couples `band` is spectral, `space` spatial, `time` temporal, and the combinations follow.
With the class come the effect on each dim (keep, select, reduce, regrid, …), the spatial reach in pixels, and whether a fitted state couples more than the apply step does.

A perturbation probe measures all of this today, without touching any operator: it ran on all 280 operator classes in geotoolz `d2f53f4` (4 Nov 2026) and classified 244 of them.
It reproduces two of the hand-kept lists in geoops's contract test exactly, which is the first sign that those lists are coupling tags written by hand.

Each operator declares its class as a class attribute, and the contract test checks that the declared class contains what the probe measures (decided 4 Nov 2026, OC-Q1 to OC-Q3; built in jejjohnson/geotoolz#614).
The class then feeds four things that today have no common vocabulary: the docs and the cheat sheet, the contract test's special-case lists, the patcher (how much overlap a tile needs), and `xrops.lift` (which operators are safe frame by frame).

What does not change: the namespaces (`indices`, `geom`, …), the [operators](./operators.md) grid of needs × carrier family, and the operator contract.

## Where this sits

The [operators](./operators.md) page places an operation by what it needs to know (geometry, dim names, shapes) and by carrier family (geoarray, xarray).
Coupling is a third, independent axis: where data flows inside the operation.
`NDVI` and `GaussianDenoise` need the same thing (dim names), but one mixes bands and the other mixes neighbouring pixels.

Two words are already taken, so this page avoids them.
**Family** means a carrier family on the operators page, and a namespace (`indices`, `qa`, …) in its "Families" tables and in geoops's `AGENTS.md`.
**Signature** means a `KernelOp`'s core-dims string, such as `"(),()->()"` or `"(c)->(c)"`.

The `KernelOp` signature does not give the coupling class on its own.
`NDVI`'s kernel is element-wise, `"(),()->()"`, yet the operator reads two bands by label and returns one, so it couples `band`.
The class is a property of the whole operator: label selection, kernel and rewrap together.

## Running example: screening a Sentinel-2 scene for a methane plume

A Sentinel-2 L2A tile covers 110 × 110 km: 5 490 × 5 490 pixels at 20 m in the SWIR, 12 bands stored as scaled integers, plus the scene classification layer (SCL).
The screening chain masks clouds from the SCL, converts to reflectance, runs a matched filter for the CH₄ column enhancement, thresholds it, labels connected plume candidates and tabulates them.
The question coupling answers is practical: which steps can run tile by tile on a patcher, with how much overlap, and which ones give a different answer per tile?

| Step | Operator | Declared class | Probe measured | Per tile |
| --- | --- | --- | --- | --- |
| 1 | `S2SCL` | pointwise, band select | reads no values (the toy SCL band holds random 12-bit integers, almost none of them SCL classes) | any tile |
| 2 | `DNToReflectance` | pointwise | pointwise | any tile |
| 3 | `ColumnEnhancement` | spatio-spectro-temporal: band reduce; the background is fitted on every call, over every pixel and frame | spatio-spectro-temporal, global | differs per tile: the background is fitted on the tile |
| 4 | `Threshold` (Otsu, the default) | spatial, global | spatial, global | differs per tile |
| 5 | `LabelConnectedComponents` | spatial, reach unbounded | pointwise (a lower bound) | needs tiles larger than the largest plume |
| 6 | `RegionProps` | conversion: raster → table | conversion | after stitching |

Steps 1 and 2 are safe anywhere.
Steps 3 and 4 are global: the probe measured that a change in a far corner moves the centre output.
Running them per tile is sometimes what you want (a local background), but it should be a choice the pipeline states, not an accident of tiling.
Step 5 is the instructive one: the probe measured it as pointwise, because flipping one isolated pixel only changes that pixel's label, yet joining two components can relabel pixels arbitrarily far away.
The probe gives a lower bound; the declared class must be at least as wide.

## Definition

Write the input as $x$ with index $j = (t, c, z, s)$: time $t$, band or variable $c$, vertical level $z$ and pixel $s = (y, x)$, any of the first three possibly absent.
An operator $T$ maps $x$ to an output with index $o$.
Output $o$ depends on input $j$ when changing $x_j$ changes $T(x)_o$; for a differentiable operator, when $\partial T(x)_o / \partial x_j \neq 0$ somewhere.

$T$ **couples** a dim $d$ when some output depends on two or more inputs that differ only along $d$: a filter mixes neighbouring pixels, a band ratio mixes bands, a composite mixes dates.
Re-indexing alone (selecting or reordering bands, a crop, a flip) moves values without mixing them, so it is an effect on $d$, not a coupling.

The coupling set $\mathcal{K}(T)$ is the set of coupled dims, with $y$ and $x$ counted together as `space`.

| $\mathcal{K}(T)$ | Class | Examples |
| --- | --- | --- |
| $\emptyset$ | pointwise | `DNToReflectance`, `ApplyMask`, `SelectBands`, `StandardScaler` (fitted over `space` and `time`) |
| `band` | spectral | `NDVI`, `BandMath`, `ContinuumRemoval`, `ApplySRF` |
| `space` | spatial | `GaussianDenoise`, `Gradient2D`, `Reproject`, `PercentileClip` |
| `time` | temporal | `MedianComposite`, `BlendMatched` |
| `band`, `time` | spectro-temporal | `MaxNDVIComposite`, the `Pixelwise*` estimators (fitted over `space`) |
| `level`, `space` | volumetric | `Gradient3D`, `Laplacian3D` |
| `band`, `space` | spatio-spectral | `SLIC`, `Felzenszwalb` |
| `space`, `time` | spatio-temporal | `HistogramStretch`, `ZeroOne` (statistics over every pixel and frame) |
| `band`, `space`, `time` | spatio-spectro-temporal | `DenoisePCA`, `ColumnEnhancement`, `PixelwiseIPCA` |

Four more facts complete the class.

- **Effect on each dim.** One of keep, select (fewer, each output from one input), reduce, mix (same size, coupled), expand, split, join, or regrid (the transform or CRS changes).
- **Spatial reach** $r$: the largest pixel distance, $\max |o_s - j_s|_\infty$, over which an output depends on an input; $r = \infty$ (stored as `None`) for a global operator.
  Neighbouring tiles overlapping by at least $2r$ put every pixel at least $r$ from the edge of a tile that holds it.
- **Fit and apply.** A fitted operator has two coupling sets, with $\mathcal{K}_{\mathrm{apply}} \subseteq \mathcal{K}_{\mathrm{fit}}$.
  `StandardScaler` applies pointwise, $x' = (x - \mu_c) / \sigma_c$, but fits $\mu_c$ and $\sigma_c$ over every pixel.
- **Carrier change.** Some operators leave the raster: conversion (to a table, geometry or scalar: `RegionProps`, `Vectorize`), fan-out (to a list of rasters: `Tile`, `SplitBands`), source (`ReadWindow`) and sink (`WriteCOG`).
  A few read only the input's grid, never its values (`LandMask`, `Rasterize`): one result then serves every frame of a stack.

## Measuring it

`geoops.testing.measure_coupling` runs an operator on a sample and perturbs it; `tests/test_operator_coupling.py` (and `scripts/operator_coupling.py` for a report) does so for every operator the contract test can build, on the first toy input it accepts.

1. **Local probe.** Bump the centre pixel of the sample, one band at a time.
   The outputs that move give the band dependence at that pixel and the spatial reach.
2. **Non-local probe.** Bump, darken and brighten a 3 × 3 block in a far corner.
   If the centre output moves, the operator reads beyond any local neighbourhood.
3. **Apply-only probe.** Repeat the non-local probe through an operator already called once, so a state fitted on first call stays frozen.
   Global in the second probe but not here means "fit over space".
4. **Time probe.** Perturb frame 0 of a 2-frame stack and see whether frame 1's output moves; for the N-ary reducers, perturb each input frame in turn.

The test reuses the contract test's tables (`CTOR_KWARGS`, `MULTI_INPUTS`, `GRAPH_PRIMARY`, `NARY_INPUTS`) rather than a second set of fixtures.

The probe measures a **lower bound** on $\mathcal{K}$, for two reasons.
A rank or topology statistic can ignore one changed pixel (`LabelConnectedComponents`, the morphology masks).
And a toy input can make an operator the identity: `ReprojectLike` with a `like` grid equal to the input grid.

## First pass: what the probe found

Read geotoolz `main` at `d2f53f4` on 4 Nov 2026; 280 operator classes, 244 probed.

| Measured class | Operators |
| --- | --- |
| spatial | 51 |
| pointwise | 50 (about ten are lower-bound artefacts, listed below) |
| reads no values | 40 (grid-only operators, and toy inputs too simple to move the output) |
| conversion | 26 |
| spectral | 25 |
| spatio-spectral | 13 |
| spatio-temporal | 8 |
| spatio-spectro-temporal | 7 |
| temporal | 6 |
| fan-out | 6 |
| pointwise, time select | 4 (`Mosaic`, `MinCloudComposite`, `BAPComposite`, `CombineMasks`) |
| volumetric | 4 |
| join, rearrange | 2 + 2 (`StackBands`, `StackMatched`; `CHWtoHWC`, `HWCtoCHW`) |
| not probed | 36 (11 readers, the `SourceOperator` and `SinkOperator` bases, 6 point and vector conversions, 3 optional ML masks, 14 needing inputs the toy set lacks) |

The findings that matter:

- **The contract test already keeps coupling tags by hand.**
  `POOLED_STATISTICS` (operators that fit once over a whole stack) is reproduced 7 of 7 by the time probe, and `TIME_INVARIANT` (masks that depend only on the grid) 10 of 10 by "reads no values".
  `STACK_AS_FEATURES` is reproduced 6 of 12; the other six are scikit-learn estimators without a fixed `random_state`, so their output moves between identical calls and the probe cannot read it.
- **Four operators pool over time but are not in `POOLED_STATISTICS`.**
  `StandardScaler`, `MinMaxScaler` and `RobustScaler` with `fit_on_call=True`, and `ReplaceOutliers`.
  Either the time-stack contract test does not reach them with those settings, or the list is incomplete; worth checking.
- **Global operators are common.**
  `PercentileClip`, `DOS1`, `StretchToUint8`, `CLAHE`, `DestripeColumn`, `MomentMatching`, `NLMeans`, `BilateralDenoise`, `Threshold` and `ColumnEnhancement` all move at the centre when a far corner changes.
  None of them says so today, and each gives a different answer per tile.
- **Known lower-bound artefacts.**
  Geometry operators that are the identity on the toy grid (`ReprojectLike`, `ResampleLike`, `PadTo`, `PhaseAlign`, `BowtieCorrection`, `GeostationaryParallaxCorrect`), and topology operators that a single isolated pixel does not reveal (`LabelConnectedComponents`, `CloseMask`, `RemoveSmallHoles`, `MergeNearbyInstances`) measure as pointwise.
  These need a hand-declared class and, later, better probe inputs.

The full per-operator table is the script's output (`--out DIR` writes `coupling.json` and `coupling.md`).
After the declarations landed, the derived lists contain the hand lists, the four pooling gaps are in the derived `POOLED_STATISTICS`, and with the estimators seeded the probe confirms 11 of 12 `STACK_AS_FEATURES` members (`PixelwiseGMM`'s toy labels do not move).

## The tag

A frozen record on each operator class, read by the docs generator, the contract test, the patcher and, later, `xrops.lift`.

```python
# geoops/_src/coupling.py; public as geoops.carrier.Coupling
@dataclasses.dataclass(frozen=True, init=False)
class Coupling:
    # 𝒦_apply: the dims an output reads across, from {"band", "level", "space", "time"}
    couples: frozenset[str]
    # the effect on each dim the operator changes: "select", "reduce", "regrid", …
    effects: tuple[tuple[str, str], ...]
    # r: the spatial reach in pixels at the default settings; None for a global operator
    reach: int | None
    # 𝒦_fit ∖ 𝒦_apply: the further dims a fitted state is estimated over
    fit: frozenset[str]
    # "transform", "fan-out", "conversion", "source" or "sink"
    kind: str
    # False when the output depends only on the input's grid
    reads_values: bool


# declared on the class, beside `backends`
class NDVI(Operator):
    backends: ClassVar[tuple[str, ...]] = ("numpy", "jax", "dask")
    coupling: ClassVar[Coupling] = Coupling(couples={"band"}, effects={"band": "reduce"})


class StandardScaler(Operator):
    # x′ = (x − μ_c) / σ_c pointwise; μ_c, σ_c fitted over every pixel and frame
    coupling: ClassVar[Coupling] = Coupling(fit={"space", "time"})


class GaussianDenoise(Operator):
    coupling: ClassVar[Coupling] = Coupling(couples={"space"}, reach=4)

    def spatial_reach(self) -> int:
        # r = ⌈4σ⌉ pixels, scipy's default truncation
        return int(np.ceil(4 * self.sigma))
```

The class name (spectral, spatio-temporal, …) is derived from `couples`, never stored, so it cannot disagree with the record.
A reach that depends on a constructor argument (σ, `window`, `iterations`) comes from `spatial_reach()`; `geoops.carrier.reach_of(op)` reads either.

The alternative was a YAML table in geoops keyed by class name.
It needs no change to any operator, but nothing then stops a new operator from shipping without a class, and the contract test has to keep the table and the code in step.

## What reads the tag

| Reader | Uses | Replaces |
| --- | --- | --- |
| cheat sheet, diagrams | cards and legends coloured by class | nothing |
| docs | `docs/operators/api/coupling.md`, generated beside `backends.md` | nothing; today operators are grouped only by namespace |
| geoops contract test | `time` in `couples ∪ fit` selects the pooled operators; `reads_values=False` the grid-only ones | `POOLED_STATISTICS`, `TIME_INVARIANT`, `STACK_AS_FEATURES` |
| geopatcher (`ApplyToChips`) | warns (`PatchReachWarning`) when neighbouring patches overlap by less than $2r$, or for a global operator | nothing; the overlap is implicit in a sampler's size and step today |
| `xrops.lift(op, dim="time")` | safe when `time` ∉ `couples ∪ fit`; otherwise refuse or fit once on the whole cube | nothing; lifting assumes per-frame operators |

## Decisions and open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| OC-Q1 | The name | **Decided 4 Nov 2026:** coupling (and "coupling class") | "family" (taken twice, see above); "scope"; "footprint" (taken by `geoarray.footprint`) |
| OC-Q2 | Where the class lives | **Decided 4 Nov 2026:** a `coupling` class attribute on each operator, a frozen `Coupling` record in `geoops/_src/coupling.py`, public in `geoops.carrier` | a YAML table in geoops keyed by class name |
| OC-Q3 | The dim vocabulary | **Decided 4 Nov 2026:** geoarray's dims, with `y x` as one `space` | `y` and `x` as separate dims |
| OC-Q4 | Measured against declared | measured ⊆ declared for every operator, and equality where the probe reads the maths exactly: `indices`, `spectral`, `radiometry` (built that way in jejjohnson/geotoolz#616; `qa` and `normalize` are not exact on toy inputs) | equality everywhere, with an exceptions list |
| OC-Q5 | The axis of the N-ary reducers | `time` for compositing; a `source` axis when the inputs are different sensors (`BlendMatched`, `MatchedField`). Built with `time` for every N-ary reducer for now | `time` for every stack |
| OC-Q6 | Where the probe lives | `geoops.testing.measure_coupling` / `assert_coupling`, so downstream packages can check their own operators; `scripts/operator_coupling.py` reports (built in jejjohnson/geotoolz#615) | keep it a script |

## Testing gates

| Gate | Passes when |
| --- | --- |
| OC-G1 Hand lists | the derived lists contain the hand lists; the probe confirms `TIME_INVARIANT` 11/12 and `STACK_AS_FEATURES` 11/12, and finds no pooling operator outside `POOLED_STATISTICS` (holds) |
| OC-G2 Every operator declares | every public operator class has a `coupling`; the measured $\mathcal{K}$ is a subset of the declared one on every probed operator (holds) |
| OC-G3 Exact where the probe is exact | for `indices`, `spectral` and `radiometry`, measured equals declared (holds) |
| OC-G4 Tiles with enough overlap | a declared-local operator run per tile with overlap ≥ $2r$ and zero window weight on each tile's outer $r$ pixels, stitched with `OverlapAdd`, equals the whole-scene result to 1e-12 in float64 (holds for a 3 × 3 mean in the patcher's tests) |
| OC-G5 Lifting | `xrops.lift(op, dim="time")` equals the per-frame loop bit for bit for every operator with `time` ∉ `couples ∪ fit_couples`, and refuses the rest |

## Status

Built as one theme in geotoolz, jejjohnson/geotoolz#614, as stacked PRs:

1. the probe script (jejjohnson/geotoolz#613);
2. `Coupling`, `measure_coupling`, `assert_coupling` (#615);
3. declarations on `indices`, `spectral`, `radiometry`, `normalize`, `qa` (#616);
4. declarations on every other operator, required by the contract test (#617);
5. the contract test's time-stack lists derived from the declarations (#618);
6. the generated `docs/operators/api/coupling.md` (#619);
7. `PatchReachWarning` in `ApplyToChips` (#620).

Still open: OC-Q5 (a `source` axis) and OC-G5, which waits for `xrops.lift`.
