---
title: "The GeoStack layer map: layers, imports and boundaries"
subtitle: "Design · a proposed layering of every library, the imports between them, and where each library's boundary sits"
short_title: Stack layer map
description: "A proposed layering of the GeoStack libraries (the foundations, Core, GeoNumerics, GeoStack as the geotoolz repo, GeoModels, GeoBench and GeoApps) with every import between them, the nine repos they are released from and the boundaries of each library; the kinds, the data axis, the role ladder and the vocabulary are on the kinds-and-roles page."
date: 2026-11-05
keywords:
  - layers
  - dependency graph
  - boundaries
  - functax
  - geoarray
---

+++ {"part": "abstract"}

A proposal for how the libraries of the stack layer, drawn from an import scan of every repository under `references/`. Core holds the maths and plumbing that know no geography, GeoNumerics the geophysical maths on plain arrays, GeoStack (the geotoolz repo, which now holds the xarray packages too) the carriers and real data, GeoModels the physics, and GeoBench and GeoApps the runs and applications on top. How code is packaged, what data it sees, the role ladder and the vocabulary that place any name in the stack now have their own page, [Kinds and roles](./kinds_and_roles.md).

+++

{button}`Open the full page <./pages/stack-layer-map.html>`

**Where it sits.** This page proposes the layering that [Levels across the stack](./stack_levels.md) and the dependency graph (`docs/api/dependency_graph.md`) describe for today, and it renames the levels of the former: its L1 to L3 become pairs of kind and data, and "L" goes back to the processing levels. The full page is a self-contained document with two maps (one with every import, one without), cards for each library and tables, which opens on its own and links back here. The open questions it raises are tracked in [#122](https://github.com/jejjohnson/geostack/issues/122) (the role ladder per library) and [geotoolz#641](https://github.com/jejjohnson/geotoolz/issues/641) (geoproducts below geoops).

## TL;DR: one test per layer

- **Layers, bottom up.** The third-party foundations (JAX, lineax, optimistix, diffrax, xarray, georeader, …); **Core**, maths and plumbing with no geography, where GeoML dissolves and three new libraries appear (functax for function approximation, integrix for integration of every kind, argminax for optimisation with uncertainty); **GeoNumerics**, geophysical maths on plain arrays, with five new libraries (geophysix, gridax-core, diagnostix, radiomax, tracerax) and one renamed (finitevolX becomes gridax-ops); **GeoStack**, exactly the geotoolz repo, mirrored by role between observations on GeoArray and states on xarray; **GeoModels** (sesmax replacing somax, radtrax, plumax); **GeoBench** and **GeoApps**, planned.
- **Nine repos.** The libraries are released from nine repos, each a uv workspace: geocore, pipekit, geo-inference and pyrox in Core; geogrids and geofields in GeoNumerics; geotoolz; geomodels; geobench. A repo imports only from repos below it, and a repo is split when it holds two clusters that one task rarely touches together. optax_bayes is retired into argminax. The pyrox repo would gain two packages of NumPyro distributions: pyrox-xtreme (today's xtremax) and pyrox-gauss, which takes the Gaussian distribution classes out of gaussx and imports gaussx, not pyrox.
- **gridax in two packages, tiers per repo.** gridax is a code-free umbrella over gridax-core (structured grids as data, on NumPy: geometry, staggered grids, masks, metrics) and gridax-ops (today's finitevolX: everything applied to a field, on JAX). spectraldiffx keeps its own grids and does not import gridax; vertical modes and forcing move from finitevolX to sesmax-core. meshax-core and meshax-ops sit beside them as placeholders for unstructured meshes, with no code yet. Tiers are numbered per repo: a library imports lower tiers of its own repo and anything in the repos below it.
- **Kernels down, wrappers up.** Functions and Modules live as low as their imports allow, on plain arrays; Operators on carriers live where the carriers are. A carrier never goes below GeoStack, and a config never reaches a function.
- **Kinds and roles, on their own page.** Function, Module and Operator; array, named array and carrier; the role ladder from ⓪ structures to ⑥ recipes, who works where, and the four rules are on [Kinds and roles](./kinds_and_roles.md).
- **Boundaries written down.** Each library has an "owns" and a "hands off" line, and two pairs get a section of their own: radiomax and radtrax; functax and geonnax. The third, geoarray, gridax-core and functax, moved with the rules to [Kinds and roles](./kinds_and_roles.md).

## On the page

1. The stack at a glance, and the interactive map of every import (today and planned)
2. Nine repos: which packages each one releases, and what it may import
3. What we build on each foundation (extends, replaces, wraps)
4. Core, reorganised (the geocore, pipekit, geo-inference and pyrox repos; functax, integrix, argminax)
5. GeoNumerics, GeoStack and GeoModels, reorganised (cards per library, and what moves)
6. radiomax and radtrax: where the line is
7. functax and geonnax: where the line is
8. The four library layers, and the test for each
9. Three kinds of function, three kinds of data: a pointer to [Kinds and roles](./kinds_and_roles.md)
10. Boundaries, library by library
11. Every library, today and proposed
12. Which foundations each library leans on
