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

Every geoops operator gets a **coupling class**: the set of dims along which an output depends on input values at other indices.
An operator that couples nothing is pointwise, one that couples `band` is spectral, `space` spatial, `time` temporal, and the combinations follow.
With the class come the effect on each dim (keep, select, reduce, regrid, …), the spatial reach in pixels, and whether a fitted state couples more than the apply step does.

A perturbation probe measures all of this today, without touching any operator: it ran on all 280 operator classes in geotoolz `d2f53f4` (4 Nov 2026) and classified 244 of them.
It reproduces two of the hand-kept lists in geoops's contract test exactly, which is the first sign that those lists are coupling tags written by hand.

Recommended: each operator declares its class as a class attribute, and the contract test checks that the declared class contains what the probe measures.
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
| 3 | `ColumnEnhancement` | spatio-spectral: band reduce, background fitted over the scene | spatio-spectral, global; pooled over a time stack | differs per tile: the background is fitted on the tile |
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

$T$ **couples** a dim $d$ when either

- $d$ is kept and some output depends on an input at a different index along $d$: $o_d \neq j_d$; or
- $d$ is reduced and some output depends on two or more inputs that differ only along $d$.

The coupling set $\mathcal{K}(T)$ is the set of coupled dims, with $y$ and $x$ counted together as `space`.

| $\mathcal{K}(T)$ | Class | Examples |
| --- | --- | --- |
| $\emptyset$ | pointwise | `DNToReflectance`, `ApplyMask`, `Normalize` |
| `band` | spectral | `NDVI`, `BandMath`, `ContinuumRemoval`, `ApplySRF` |
| `space` | spatial | `GaussianDenoise`, `Gradient2D`, `Reproject`, `PercentileClip` |
| `time` | temporal | `MedianComposite`, `BlendMatched` |
| `level`, `space` | volumetric | `Gradient3D`, `Laplacian3D` |
| `band`, `space` | spatio-spectral | `SLIC`, `PixelwiseKMeans` |
| `space`, `time` | spatio-temporal | `StandardScaler` on a stack, `HistogramStretch` |
| `band`, `space`, `time` | spatio-spectro-temporal | `PixelwisePCA`, `MNF`, `DenoisePCA` |

Four more facts complete the class.

- **Effect on each dim.** One of keep, select (fewer, each output from one input), reduce, mix (same size, coupled), expand, split, join, or regrid (the transform or CRS changes).
- **Spatial reach** $r$: the largest pixel distance, $\max |o_s - j_s|_\infty$, over which an output depends on an input; $r = \infty$ for a global operator.
  It is the overlap a tile needs on each side for its interior to equal the whole-scene result.
- **Fit and apply.** A fitted operator has two coupling sets, with $\mathcal{K}_{\mathrm{apply}} \subseteq \mathcal{K}_{\mathrm{fit}}$.
  `StandardScaler` applies pointwise, $x' = (x - \mu_c) / \sigma_c$, but fits $\mu_c$ and $\sigma_c$ over every pixel.
- **Carrier change.** Some operators leave the raster: conversion (to a table, geometry or scalar: `RegionProps`, `Vectorize`), fan-out (to a list of rasters: `Tile`, `SplitBands`), source (`ReadWindow`) and sink (`WriteCOG`).

## Measuring it

`scripts/operator_coupling.py` in geotoolz runs every operator the contract test can build, on the first toy input it accepts, and perturbs that input.

1. **Local probe.** Bump one pixel at the centre of a 16 × 16 scene, one band at a time.
   The outputs that move give the band dependence at that pixel and the spatial reach.
2. **Non-local probe.** Bump, darken and brighten a 3 × 3 block in a far corner.
   If the centre output moves, the operator reads beyond any local neighbourhood.
3. **Apply-only probe.** Repeat the non-local probe through an operator already called once, so a state fitted on first call stays frozen.
   Global in the second probe but not here means "fit over space".
4. **Time probe.** Perturb frame 0 of a 2-frame stack and see whether frame 1's output moves; for the N-ary reducers, perturb each input frame in turn.

It reuses the contract test's tables (`CTOR_KWARGS`, `MULTI_INPUTS`, `GRAPH_PRIMARY`, `NARY_INPUTS`) rather than a second set of fixtures.

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

## The tag

Recommended: a frozen record on each operator class, read by the docs generator, the contract test, the patcher and `xrops.lift`.

```python
# proposed (OC-Q2): geoops/_src/coupling.py
@dataclasses.dataclass(frozen=True)
class Coupling:
    # 𝒦_apply: the dims an output reads across, from {"band", "level", "space", "time"}
    couples: frozenset[str]
    # the effect on each dim the operator changes: "select", "reduce", "regrid", …
    effects: Mapping[str, str]
    # r: the spatial reach in pixels; None for a global operator
    reach: int | None = 0
    # 𝒦_fit ⊇ 𝒦_apply: the dims a fitted state is estimated over
    fit_couples: frozenset[str] = frozenset()


# proposed (OC-Q2): declared on the class, beside `backends`
class NDVI(Operator):
    backends: tuple[str, ...] = ("numpy", "jax")
    coupling: Coupling = Coupling(
        couples=frozenset({"band"}),
        effects={"band": "reduce"},
    )


class StandardScaler(Operator):
    # x′ = (x − μ_c) / σ_c: pointwise once fitted, μ_c and σ_c fitted over every pixel
    coupling: Coupling = Coupling(
        couples=frozenset(),
        effects={},
        fit_couples=frozenset({"space"}),
    )


class GaussianDenoise(Operator):
    # reach 4σ in pixels (scipy's default truncation) for the default σ = 1
    coupling: Coupling = Coupling(couples=frozenset({"space"}), effects={}, reach=4)
```

The class name (spectral, spatio-temporal, …) is derived from `couples`, never stored, so it cannot disagree with the record.
A reach that depends on a constructor argument (σ, `window`) is a property computed from it, not a constant.

Alternative: a YAML table in geoops keyed by class name.
It needs no change to any operator, but nothing then stops a new operator from shipping without a class, and the contract test has to keep the table and the code in step.

## What reads the tag

| Reader | Uses | Replaces |
| --- | --- | --- |
| docs, capability index, cheat sheet | a Class column; cards and diagram legends coloured by class | nothing; today operators are grouped only by namespace |
| geoops contract test | `fit_couples` ∋ `time` selects the pooled operators; `effects` empty and no values read selects the grid-only ones | `POOLED_STATISTICS`, `TIME_INVARIANT`, `STACK_AS_FEATURES` |
| geopatcher (`ApplyToChips`) | a tile overlap of at least `reach` on each side, or a warning for a global operator | nothing; the overlap is implicit in a sampler's size and step today |
| `xrops.lift(op, dim="time")` | safe when `time` ∉ `couples ∪ fit_couples`; otherwise refuse or fit once on the whole cube | nothing; lifting assumes per-frame operators |

## Decisions and open questions

| ID | Question | Recommended | Alternative |
| --- | --- | --- | --- |
| OC-Q1 | The name | **coupling** (and "coupling class") | "family" (taken twice, see above); "scope"; "footprint" (taken by `geoarray.footprint`) |
| OC-Q2 | Where the class lives | a `coupling` class attribute on each operator, a frozen `Coupling` record in `geoops/_src/coupling.py` | a YAML table in geoops keyed by class name |
| OC-Q3 | The dim vocabulary | geoarray's dims, with `y x` as one `space` and an optional axis note for the 1-D calculus (`Gradient1D`) | `y` and `x` as separate dims |
| OC-Q4 | Measured against declared | the contract test requires measured ⊆ declared for every operator, and equality for the kernel operators (`indices`, `qa`, `normalize`) | equality everywhere, with an exceptions list |
| OC-Q5 | The axis of the N-ary reducers | `time` for compositing; a `source` axis when the inputs are different sensors (`BlendMatched`, `MatchedField`) | `time` for every stack |
| OC-Q6 | Where the probe lives | `scripts/operator_coupling.py` now; `geoops.testing.coupling_of(op, x)` once the tag exists, so downstream packages can check their own operators | keep it a script |

## Testing gates

| Gate | Passes when |
| --- | --- |
| OC-G1 Hand lists | the probe reproduces `POOLED_STATISTICS` 7/7 and `TIME_INVARIANT` 10/10 (holds today), and `STACK_AS_FEATURES` 12/12 once the estimators are seeded |
| OC-G2 Every operator declares | every public operator class has a `coupling`; the measured $\mathcal{K}$ is a subset of the declared one on every probed operator |
| OC-G3 Kernel operators exact | for `indices`, `qa` and `normalize`, measured equals declared |
| OC-G4 Tiles with enough overlap | a declared-local operator run per tile with overlap ≥ `reach`, stitched, equals the whole-scene result to 1e-12 in float64 away from the scene edge |
| OC-G5 Lifting | `xrops.lift(op, dim="time")` equals the per-frame loop bit for bit for every operator with `time` ∉ `couples ∪ fit_couples`, and refuses the rest |

## Next steps

1. **Decide OC-Q1 to OC-Q3.** They fix the vocabulary everything else uses.
2. **Land the probe** (geotoolz PR with `scripts/operator_coupling.py`), then fix the four `POOLED_STATISTICS` gaps and seed the estimators.
3. **Declare the classes**, one namespace per PR, starting with the kernel operators, where the probe is exact; derive the contract test's lists from the tags.
4. **Use them**: a Class column in the capability index, colours on the cheat sheet and the diagrams, and the overlap check in `ApplyToChips`.
