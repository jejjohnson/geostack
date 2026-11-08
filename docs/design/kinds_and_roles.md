---
title: "Kinds and roles: three kinds of function, three kinds of data"
subtitle: "Design · working page · how code in the stack is packaged, what data it sees, the role each name plays in its library, and the rules that place any name in the stack"
short_title: Kinds and roles
description: "How code in the GeoStack is packaged (function, Module, Operator), what data it sees (array, named array, carrier), the seven-rung role ladder from structures to recipes, who works where, one vocabulary per axis, and the four rules that place any name in the stack."
date: 2026-11-06
keywords:
  - kinds
  - carriers
  - role ladder
  - vocabulary
  - geoarray
  - pipekit
---

+++ {"part": "abstract"}

A working page split from the [layer map](./stack_layer_map.md), which keeps the layers, the imports and the boundaries of each library. This page holds the two axes every name in the stack sits on: how its code is packaged and what data it sees. It starts from the silent failures of geo code, defines the objects in symbols with one running example (a methane plume in an EMIT hyperspectral scene over an oil and gas field: a matched filter, a plume mask, and an emission rate from the integrated mass enhancement and the wind), climbs the role ladder inside a library, and ends with the rules that decide where a name may live and what it may call.

+++

{button}`Open the full page <./pages/kinds-and-roles.html>`

**Status.** A draft, still being worked on; the open questions are tracked in [#122](https://github.com/jejjohnson/geostack/issues/122) (the role ladder per library).

## TL;DR: two axes, one ladder, four rules

- **Why.** Seven ways geo code goes wrong without an error: axes by position, a lost georeference, mismatched grids, nodata, maths locked in one container, setup that can't be traced, and pipelines held together by scripts. The first four are about the data; the last three are about how the code is packaged.
- **The data axis.** A plain array $\vect{x} \in \mathbb{R}^{N_1 \times \ldots \times N_{N_\text{ax}}}$; a named array $\vect{x}_\text{named} = (\vect{x}, \vect{d}, \mathcal{C})$, with a name and coordinates per axis (pipekit-array's `DimArray`, xarray's `DataArray`); and a carrier $\vect{x}_\text{geo} = (\vect{x}_\text{named}, \Gamma)$, with the georeference $\Gamma = (\op{crs}, \mat{A}, x_\text{fill}, \mathcal{M})$ (geoarray's `GeoArray`).
- **The kind axis.** A function $\vect{y} = f(\vect{x}, \vect{u}; \gvect{\theta})$ of the data $\vect{x}$, what it needs to know about where the data sits $\vect{u}$ (geometry, band positions), and parameters $\gvect{\theta}$; a Module, whose setup $\gvect{\psi} = \op{setup}(\mat{X}; \gvect{\theta})$ runs once and whose call is jittable after; and an Operator $h[f, \gvect{\theta}]$, which lifts a function onto carriers with a JSON config $\alpha$, $h[f, \gvect{\theta}](\vect{x}_\text{geo}; \alpha) = \iota_\alpha(f(\pi_\alpha(\vect{x}_\text{geo}); \gvect{\theta}), \vect{x}_\text{geo})$: unwrap, compute on arrays, rewrap.
- **The role ladder.** ⓪ structure, ① primitive, ② strategy, ③ model, ④ algorithm, ⑤ step (one transformation with its settings), ⑥ recipe (steps composed into a job, in code or as data); orchestration sits above the ladder and is not a rung.
- **Who works where.** Data engineers at ⑥ and in orchestration; geoscience researchers at ⑤ and ⑥, where carriers begin; ML researchers at ② to ④; numerics and CS at ⓪ to ④. Each band treats the rung below it as its floor.
- **Four rules.** The cell (kind × data) decides the lowest layer a name may live in; the tier decides what a library may import; the kind and the data decide what a name may take and call (arrays go everywhere, carriers stop at GeoStack, configs stop at Operators); each library declares its cells.

## On the page

1. The problem: why geo code is hard to write well
2. The objects, in symbols: the data axis (plain array, named array, carrier) and the kind axis (function, Module, Operator), each with a worked example in code, and a note on the parallels with operator learning (encoder, approximator, reconstructor), Lift & Learn and Koopman
3. The running example, end to end, as four lifted functions, and the data axis in three lines
4. Inside a library: the role ladder, with steps, recipes and orchestration; then the running example up the ladder: the data, primitives, strategies and models, algorithms, one step per stage, and the recipe (retrieval, segmentation, quantification) with its orchestration
5. Who works where: bands of the ladder
6. One vocabulary per axis, and whether we need a name change
7. The four rules, the three invariants, and what enforces them today
8. geoarray, gridax and functax: geometry has no CRS below GeoStack
