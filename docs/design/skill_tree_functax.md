---
title: "functax's skill shelf: one unit per public module"
subtitle: "Design · the skill tree for a base library of several public modules"
short_title: Skill tree · functax
description: "functax's shelf of agent skills as built: three listed skills, sixteen entries and eight units, one per public module, each with a contract card. Measured on functax at the pin and on the libraries that depend on it."
date: 2026-11-19
keywords:
  - agents
  - skills
  - functax
  - role ladder
  - skill tree
---

functax is the second library with a shelf, after [gaussx](./skill_tree_gaussx.md), and the first whose public names sit in several public modules.
The shelf itself lives in functax, under `plugins/functax/` (jejjohnson/functax#154); this page records how its units were chosen and what the measurements say.
[The recipe](./skill_shelf_recipe.md) lists the files of a shelf.

## The units

functax has no single namespace: a caller imports `functax.interp`, `functax.quad` or `functax.basis.sphere`, never `functax` alone.
So the units follow the public modules, one each (decided in #196), and a dependent that imports one module receives one card.
A unit's name here is its key in functax's `shelf.toml`, which holds the title a reader sees: `interp` is "Interpolation", `quad` is "Quadrature".

:::{table} functax's units, with the public names each holds. The functions of `functax.basis.sphere` and its sibling modules count for `functax.basis`.
:label: tbl-functax-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Geometry | ⓪ | numerics, geoscience | `functax.geometry` 31 | 31 |
| Errors | ⓪ | numerics | `functax.exceptions` 7, `functax` 1 | 8 |
| Interp | ① ②, setup | numerics | `functax.interp` 70 | 70 |
| Basis | ① | numerics, ML | `functax.basis` 50 | 50 |
| Quad | ⓪ ① | numerics | `functax.quad` 13 | 13 |
| Diffops | ① | numerics | `functax.diffops` 8 | 8 |
| Project | ① ④ | numerics | `functax.project` 4 | 4 |
| Transform | setup | numerics | `functax.transform` 6 | 6 |
| **All** | | | | **190** |
<!-- shelf:units:end -->
:::

## The listed skills

Three skills are listed, because a caller asks for them without mentioning the rest.

| Listed skill | Units | Entries |
|---|---|---|
| `function-approximation-with-functax`, the front door | Geometry, Errors | 4 |
| `interpolation-with-functax` | Interp | 5 |
| `bases-quadrature-and-transforms` | Basis, Quad, Diffops, Project, Transform | 7 |

Interpolation has a skill of its own because it holds 70 of the public names and is what most dependents come for.
The five small modules share one skill: each is a few functions, and a task that fits a basis usually integrates or transforms as well.

## Who uses what

:::{table} Names of each unit that each dependent package uses.
:label: tbl-functax-uses

<!-- shelf:uses:begin -->
| Dependent package | Names used | Geometry | Errors | Interp | Basis | Quad | Diffops | Project | Transform |
|---|---|---|---|---|---|---|---|---|---|
| geonnax | 68 | 14 | 0 | 0 | 41 | 1 | 8 | 0 | 4 |
| kernellib | 15 | 2 | 0 | 0 | 12 | 1 | 0 | 0 | 0 |
| pyrox-nn | 8 | 1 | 0 | 0 | 7 | 0 | 0 | 0 | 0 |
| spectraldiffx | 7 | 1 | 0 | 0 | 2 | 2 | 0 | 0 | 2 |
| gauss-flows | 6 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| integrix | 6 | 1 | 0 | 5 | 0 | 0 | 0 | 0 | 0 |
| pyrox-gp | 6 | 0 | 0 | 0 | 6 | 0 | 0 | 0 | 0 |
| geotoolz-ops | 5 | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| pyrox-lgm | 5 | 3 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| geotoolz-array | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| plumax | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
<!-- shelf:uses:end -->
:::

:::{table} The cards each dependent receives: a unit's card goes to a package that uses three of its names, or subclasses one.
:label: tbl-functax-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
| geonnax | Geometry; Basis; Diffops; Transform | Quad (1) |
| kernellib | Basis | Geometry (2); Quad (1) |
| pyrox-nn | Basis | Geometry (1) |
| spectraldiffx | none | Geometry (1); Basis (2); Quad (2); Transform (2) |
| gauss-flows | Interp | Geometry (1) |
| integrix | Interp | Geometry (1) |
| pyrox-gp | Basis |  |
| geotoolz-ops | Geometry | Interp (2) |
| pyrox-lgm | Geometry | Interp (2) |
| geotoolz-array | Geometry |  |
| plumax | none | Geometry (1) |
<!-- shelf:cards:end -->
:::

A package below the threshold on every unit receives no card, and its block names functax's front door instead.
