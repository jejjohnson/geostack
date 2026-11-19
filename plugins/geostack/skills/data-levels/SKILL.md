---
name: data-levels
description: The L0–L5 data lifecycle of the GeoStack, on its two axes. The science axis orders data by maturity (raw telemetry, calibrated radiances, retrieved variables, analysis, reanalysis and forecast, derived products); the engineering axis says which package computes, reads, indexes, transforms, patches and stores each level, and where the carrier switches from a raster carrier to xarray. Also the in-situ stream that joins by matchup, and the three layers a field passes through on its way into a model. Use to place a dataset, a product or an operation on a level, to choose the carrier and the package for a job, to name a task or a benchmark as a level transition, or whenever a page, an issue or another skill says "L2" or "L3 to L4".
---

# The data levels

Draft 1 · 19 Nov 2026 · from the overview's second axis (`docs/overview.md`, "The L0–L5 data lifecycle", @ `854276a`). Where this file and that page disagree, the page wins.

Data in the stack has a level, from L0 to L5, which says how far it has come from the instrument. The level is read on two axes. The science axis says what the data is. The engineering axis says which tool handles it. Together they answer the two questions asked of any dataset or operation: what is this, and where does the code for it live.

The levels are one of the stack's three organising axes. The other two are the `modeling-cycle` (the research loop) and the `six-step-cycle` (the path from a simple model to a fast one), both in this plugin. They are orthogonal: every operation sits somewhere on all three.

## 1. The six levels

| Level | Name | What it holds | Carrier |
|---|---|---|---|
| L0 | Unstructured observations | raw instrument output: telemetry, counts, packets | raster carrier |
| L1 | Structured observations | calibrated, georeferenced radiances | raster carrier |
| L2 | Retrieved observations | per-pixel geophysical variables, with gaps where retrieval fails | raster carrier |
| L3 | Analysis | a regular, gap-filled grid: the "world right now" | xarray |
| L4 | Reanalysis and forecast | dynamical model output: data-constrained, fused or projected | xarray |
| L5 | Derived products | extremes, indicators and impacts computed from L4 | xarray |

The raster carrier is the array with its geotransform and CRS: georeader's `GeoTensor` in the overview's table, and `GeoArray` (with `GeoSwath` and `GeoPoints`) in the geoarray design, which replaces it between geotoolz packages. Check the capability index for which one the pinned code uses before writing either name.

## 2. The science axis: what the data is

Read left to right, L0 to L5.

- **One stream until retrieval.** The same L1 radiances feed every geoscience.
- **Retrieval is where it fans out** into ocean, atmosphere, land and ice variables. Each domain then carries its own analysis and its own reanalysis or forecast.
- **L5 holds what is derived from those states**: extremes (marine heatwaves, droughts, storms) and impact indicators.

Each arrow between two levels is a step with a name, and ML can enter at every one:

| Transition | Step | Running example in the overview |
|---|---|---|
| L0 → L1 | read and calibrate | detector counts to radiance |
| L1 → L2 | retrieve | methane from shortwave infrared |
| L2 → L3 | grid and gap-fill | sea level from altimeter tracks |
| L3 → L4 | model and assimilate | the ocean state from sea-level maps |
| L4 → L5 | derive | heatwave return levels |

## 3. The engineering axis: which tool handles it

Read top to bottom, from computing a level down to storing it.

| Job | L0 to L2 (observations) | L3 to L5 (states and derived products) |
|---|---|---|
| Compute the step into the level | radtrax retrieves (L1 → L2) | GeoML grids and gap-fills (L2 → L3: pyrox, kernellib, vardax's optimal interpolation); GeoModels with data assimilation (L3 → L4: somax and plumax with filterax and vardax); xtremax derives extremes (L4 → L5) |
| Compose | pipekit, whatever the carrier | pipekit |
| Read | georeader, into the raster carrier | xrreader, into xarray |
| Index | geocatalog, as `GeoSlice` rows | `geocatalog.states`, as state rows in the same catalogs |
| Transform | geotoolz | xrtoolz, which also verifies |
| Patch | geopatcher | geopatcher |
| Store | geocloud, as COG | geocloud, as Zarr or NetCDF |

The carrier switches between L2 and L3, and the tools split with it. That line is the most useful fact on the axis: an operation on gappy per-pixel data belongs with the observation tools, and one on a regular grid in time belongs with the state tools.

## 4. Two streams

The lifecycle is not one chain. In-situ networks (in every domain) have no L0 or L1. They join the satellite chain in two places:

- **Between L2 and L3, by matchup**: pairing observations from different sources that coincide in space and time. Matchup feeds L3, and it yields the pairs that validate and bias-correct the satellite stream.
- **At L4, directly, by assimilation.**

The misfits that matchup produces are what drive the loops of the `modeling-cycle`.

## 5. Three layers on the way into a model

What is done to a field before a model sees it is a third reading of the same data:

| Layer | What it is | Who owns the step into it |
|---|---|---|
| Scientific data | observations on the raster carrier, states on xarray | the readers |
| ML-ready data | cached, CF-compliant tensors with provenance and content-addressed splits | GeoStack: harmonisation and patching (geopatcher) |
| Embeddings | latent representations; a decoder maps them back to fields | GeoML: geonnax encoders |

The middle layer is the handoff that makes training reproducible.

## 6. Placing something on the levels

Ask in this order.

1. **Is it data or an operation?** Data has one level. An operation has a transition (L1 → L2) or stays within a level (a regrid of an L3 field is L3 → L3).
2. **For data: how far is it from the instrument?** Counts and packets are L0; radiances with geolocation are L1; a geophysical variable per pixel, with gaps, is L2; a gap-free grid is L3; anything a dynamical model produced is L4; anything computed from L4 is L5.
3. **In-situ?** It has no L0 or L1. Say "in-situ" and where it joins (matchup or assimilation), not a level.
4. **Which side of the carrier line?** L0 to L2 uses the observation tools, L3 to L5 the state tools (§3).
5. **Which package computes the step?** The compute row of §3, then the capability index for the name.

Name a task by its transition: "L1 → L2 retrieval", "L2 → L3 mapping". A benchmark's task is a transition and what it does (gap filling, forecast, fusion), and both go into its contract.

## 7. What is not settled

Say these when they matter, and do not paper over them.

- **"L" is used for code as well, on one older page.** `docs/design/stack_levels.md` calls code that works on arrays L1, on carriers L2 and on whole runs L3. The layer-map page proposes giving "L" back to the data levels and describing code by its kind and its data instead. In this skill and in new text, L means a data level; for code, name the kind (function, Module, Operator) and the rung of `role-ladder`.
- **L2 has two names.** The overview's L2 is retrieved observations with gaps, and gap filling is L2 → L3. pipekit-evaluate's `LEVEL_NAMES` call L2 "gap-filled observations", so the pages that follow that code (the benchmarking report, the benchmarks gallery) label gap filling L1 → L2. The overview records that the two are not reconciled. Say which convention a page follows.
- **L5 is open.** Which products belong there is not decided; the overview draws it dashed.

## Checklist

- [ ] Every dataset named has one level, and every operation a transition or a level it stays in.
- [ ] In-situ data is called in-situ, with where it joins.
- [ ] The carrier and the tools match the side of the L2 and L3 line the data is on.
- [ ] A task or benchmark is named by its transition.
- [ ] "L" means a data level; code is described by kind and rung.
- [ ] Where gap filling is given a transition, the page says which L2 convention it follows.
