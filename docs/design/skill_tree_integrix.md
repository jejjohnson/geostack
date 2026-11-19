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
<!-- shelf:uses:end -->
:::

:::{table} The cards each dependent receives: a unit's card goes to a package that uses three of its names, or subclasses one.
:label: tbl-integrix-cards

<!-- shelf:cards:begin -->
<!-- shelf:cards:end -->
:::

A package below the threshold on both units receives no card, and its block names integrix's front door instead.
