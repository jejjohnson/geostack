---
title: "The GeoStack layer map: layers, boundaries and one vocabulary per axis"
subtitle: "Design · a proposed layering of every library, the imports between them, where each library's boundary sits, and the words that place any name in the stack"
short_title: Stack layer map
description: "A proposed layering of the GeoStack libraries (the foundations, Core, GeoNumerics, GeoStack as geotoolz + xrtoolz, GeoModels, GeoBench and GeoApps) with every import between them, the boundaries of each library, and one vocabulary per axis: role, kind, data, layer, tier and processing level."
date: 2026-11-05
keywords:
  - layers
  - dependency graph
  - boundaries
  - functax
  - geoarray
  - role ladder
---

+++ {"part": "abstract"}

A proposal for how the libraries of the stack layer, drawn from an import scan of every repository under `references/`. Core holds the maths and plumbing that know no geography, GeoNumerics the geophysical maths on plain arrays, GeoStack (geotoolz + xrtoolz) the carriers and real data, GeoModels the physics, and GeoBench and GeoApps the runs and applications on top. The page also fixes a vocabulary, one per axis, so that any name in the stack can be placed by its role, its kind, the data it sees, its layer, its tier and the processing level of the data it touches.

+++

{button}`Open the full page <./pages/stack-layer-map.html>`

**Where it sits.** This page proposes the layering that [Levels across the stack](./stack_levels.md) and the dependency graph (`docs/api/dependency_graph.md`) describe for today, and it renames the levels of the former: its L1 to L3 become pairs of kind and data, and "L" goes back to the processing levels. The full page is a self-contained document with two maps (one with every import, one without), cards for each library, tables and equations, which opens on its own and links back here. The open questions it raises are tracked in [#122](https://github.com/jejjohnson/geostack/issues/122) (the role ladder per library) and [geotoolz#641](https://github.com/jejjohnson/geotoolz/issues/641) (geoproducts below geoops).

## TL;DR: one test per layer, one vocabulary per axis

- **Layers, bottom up.** The third-party foundations (JAX, lineax, optimistix, diffrax, xarray, georeader, …); **Core**, maths and plumbing with no geography, where GeoML dissolves and three new libraries appear (functax for function approximation, integrix for integration of every kind, argminax for optimisation with uncertainty); **GeoNumerics**, geophysical maths on plain arrays, with six new libraries (geophysix, gridax, diagnostix, radiomax, skillax, tracerax); **GeoStack**, exactly geotoolz + xrtoolz, mirrored by role between observations on GeoArray and states on xarray; **GeoModels** (sesmax replacing somax, radtrax, plumax); **GeoBench** and **GeoApps**, planned.
- **Kernels down, wrappers up.** Functions and Modules live as low as their imports allow, on plain arrays; Operators on carriers live where the carriers are. A carrier never goes below GeoStack, and a config never reaches a function.
- **The objects in symbols.** A function $f(x;\theta)$ and an Operator that lifts it onto geographic data, $h[f,\theta](\hat{x};\alpha) = \iota_\alpha\big(f(\pi_\alpha(\hat{x});\theta), \hat{x}\big)$, where $\hat{x} = (x, d, c, g)$ is a carrier: an array with dim names, coordinates and a georeference.
- **One vocabulary per axis.** Role ⓪ structure, ① primitive, ② strategy, ③ model, ④ recipe, ⑤ step (nouns, verbs, adverbs, noun phrases, sentences, paragraphs); kind ƒ function, ▣ Module, → Operator; data ▦ array, ▤ named array, ◆ carrier; layer by name; tier T0 to T4; processing level L0 to L4, for data only.
- **Boundaries written down.** Each library has an "owns" and a "hands off" line, and three pairs get a section of their own: geoarray, gridax and functax; radiomax and radtrax; functax and geonnax.

## On the page

1. The stack at a glance, and the interactive map of every import (today and planned)
2. What we build on each foundation (extends, replaces, wraps)
3. Core, reorganised (the pipekit and pyrox workspaces, functax, integrix, argminax)
4. GeoNumerics, GeoStack and GeoModels, reorganised (cards per library, and what moves)
5. radiomax and radtrax: where the line is
6. functax and geonnax: where the line is
7. The four library layers, and the test for each
8. Three kinds of function, three kinds of data: the problem, the objects in symbols, the role ladder, one vocabulary per axis, and the rules for who works with what
9. Boundaries, library by library
10. Every library, today and proposed
11. Which foundations each library leans on
