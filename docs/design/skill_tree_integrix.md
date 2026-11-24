---
title: "integrix's skill shelf: integrals and marches"
subtitle: "Design · the skill tree for a library whose public modules fall into two groups"
short_title: Skill tree · integrix
description: "integrix's shelf of agent skills as built: two listed skills, ten entries and two units, integrals and marches, each with a contract card. Measured on integrix at the pin and on the libraries that depend on it."
date: 2026-11-19
keywords:
  - agents
  - skills
  - integrix
  - role ladder
  - skill tree
---

integrix is the third library with a shelf, after [gaussx](./skill_tree_gaussx.md) and [functax](./skill_tree_functax.md).
The shelf itself lives in integrix, under `plugins/integrix/` (jejjohnson/integrix#85); this page records how its units were chosen and what the measurements say.
[The recipe](./skill_shelf_recipe.md) lists the files of a shelf.

## The units

integrix computes one integral in three ways: a rule sums weights against samples, a scan builds the running integral, and a march steps an integrand that depends on the solution.
Like functax it has no single namespace, and a caller imports `integrix.rule` or `integrix.march`, never `integrix` alone.
Its six area modules are small, so one unit per module would hand a dependent several cards for one job.
The units follow the two things a caller asks for instead: an integral that does not step in time, and a march.
`integrix.adjoint` sits with the marches, because its options are arguments of `integrix.march.run`.

:::{table} integrix's units, with the public names each holds.
:label: tbl-integrix-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Integrals | ⓪ ① ② | numerics, ML | `integrix.expect` 8, `integrix.adapt` 4, `integrix.rule` 4, `integrix.exceptions` 2, `integrix` 1, `integrix.scan` 1 | 20 |
| Marches | ① ② ④ | numerics, geoscience | `integrix.march` 31, `integrix.adjoint` 5 | 36 |
| **All** | | | | **56** |
<!-- shelf:units:end -->
:::

## The listed skills

Each unit is a listed skill, because a caller asks for one without mentioning the other.

| Listed skill | Unit | Entries |
|---|---|---|
| `quadrature-and-expectations-with-integrix` | Integrals | 5 |
| `time-stepping-with-integrix` | Marches | 5 |

## Who uses what

:::{table} Names of each unit that each dependent package uses.
:label: tbl-integrix-uses

<!-- shelf:uses:begin -->
| Dependent package | Names used | Integrals | Marches |
|---|---|---|---|
| finitevolx | 8 | 0 | 8 |
| plumax | 4 | 3 | 1 |
| sesmax-core | 3 | 0 | 3 |
| pipekit-jax | 1 | 0 | 1 |
| vardax | 1 | 0 | 1 |
<!-- shelf:uses:end -->
:::

:::{table} The cards each dependent receives: a unit's card goes to a package that uses three of its names, or subclasses one.
:label: tbl-integrix-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
| finitevolx | Marches |  |
| plumax | Integrals | Marches (1) |
| sesmax-core | Marches |  |
| pipekit-jax | none | Marches (1) |
| vardax | none | Marches (1) |
<!-- shelf:cards:end -->
:::

A package below the threshold on both units receives no card, and its block names integrix's front door instead.
