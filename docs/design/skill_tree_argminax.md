---
title: "argminax's skill shelf: a small strategy library that stands on optimistix"
subtitle: "Design · the skill tree for a library whose floor is mostly third party"
short_title: Skill tree · argminax
description: "argminax's shelf of agent skills: nine use skills over three units, one listed. argminax is the smallest shelf so far and the first where a third-party library, optimistix, is most of the floor, so this page also drafts the first substrate card. No package uses argminax at the pins yet, so its cards exist before their readers. Measured at the submodule pins by scripts/skill_shelf.py."
date: 2026-11-12
keywords:
  - agents
  - skills
  - argminax
  - optimistix
  - role ladder
  - skill tree
numbering:
  heading_1: true
  heading_2: true
  table: true
---

(st-am-decision)=
## Decision

Recommended: argminax's shelf is nine use skills and two extend skills over three units, with one listed skill, and its floor is three cards: two of gaussx's and a substrate card for optimistix, which this page drafts.
Alternative: no shelf, only one skill, because the library has 31 public names; that loses the three cards its future dependents need.
Status: open, 12 Nov 2026.

What does not change: argminax's public API and its two contributor skills, `add-solver` and `add-diagnostic`.
argminax has no use skill today.
This page follows the pattern of [gaussx's shelf](./skill_tree_gaussx.md) and says only where argminax differs.

(st-am-motivation)=
## Motivation

argminax is small, new, and extends another library's protocols, so most of what a caller must know is not in argminax.

:::{table} argminax's public names by unit, at the submodule pin. The public API lives in four public modules, `solve`, `adjoint`, `curvature` and `diagnose`; the package root exports only the exceptions.
:label: tbl-am-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Solvers and adjoints | ② ④ | ML | `argminax.solve` 11, `argminax.adjoint` 3, `argminax.exceptions` 3, `argminax` 1 | 18 |
| Curvature | ⓪ ③ | numerics, ML | `argminax.curvature` 7 | 7 |
| Diagnostics | ④ | ML | `argminax.diagnose` 6 | 6 |
| **All** | | | | **31** |
<!-- shelf:units:end -->
:::

:::{table} Which part of argminax each dependent package uses. The table is empty at these pins.
:label: tbl-am-consumers

<!-- shelf:uses:begin -->
| Dependent package | Names used | Solvers and adjoints | Curvature | Diagnostics |
|---|---|---|---|---|
<!-- shelf:uses:end -->
:::

No package in the stack imports argminax at the pins of this commit.
Its `AGENTS.md` @ `c7bd0c3` names the consumers that are to switch over: vardax, radtrax, plumax, pyrox-gp, pyrox-lgm, pyrox-nn and gaussx.
The table fills in as their pins move, and the test fails until the page is regenerated.

(st-am-shelf)=
## The shelf

:::{table} argminax's shelf. Each use skill is one file; the extend skills exist today in `.claude/skills/`.
:label: tbl-am-shelf

| Role | Skill | Covers | Module |
| --- | --- | --- | --- |
| ⓪ structure | Curvature operators | `gauss_newton` and `low_rank`, returned as gaussx operators with their PSD tag | `curvature` |
| ② strategy | Gauss–Newton with an inner CG | `GaussNewtonCG`, as a solver passed to optimistix | `solve` |
| ② strategy | Adjoints for a solve stopped early | `KStep`, `OneStep`, `to_optimistix_adjoint` | `adjoint` |
| ② strategy | Positivity and bounds | `Reparametrised` | `solve` |
| ② strategy | `add-solver` | extend | |
| ③ model | The Laplace approximation | `Laplace`, `marginal_variance`, `push_forward`, `sample_by_perturbation` | `curvature` |
| ④ algorithm | Finding the mode | `laplace_mode`, its covariance and site forms, and their results | `solve` |
| ④ algorithm | Dual-form solves | `dual_solve`, `incremental_dual_solve` | `solve` |
| ④ algorithm | Retrieval diagnostics | `averaging_kernel`, `degrees_of_freedom`, `information_content` | `diagnose` |
| ④ algorithm | Evidence and the prior precision | `evidence`, `optimize_prior_precision` | `diagnose` |
| ⓪ ④ | `add-diagnostic` | extend: a curvature operator or a diagnostic | |
:::

argminax has no ① of its own.
Its `AGENTS.md` puts the boundary in one line, "one step is gaussx's; the loop is argminax's": the verbs are `gaussx.newton_update` and gaussx's solves, and argminax iterates them.

(st-am-units)=
## Units and cards

:::{table} The cards each dependent package would receive from argminax. The table is empty at these pins.
:label: tbl-am-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
<!-- shelf:cards:end -->
:::

The three cards are written before anything reads them.
That order is right for a new library: a consumer's switch-over then brings the card into its block in the same change that brings the import.

(st-am-listed)=
## Listed skill

One skill is listed, `minimise-with-uncertainty`, at most 150 lines, over all three units.
It is asked for as "fit this by Gauss–Newton", "what is the posterior covariance at the minimum" and "how many degrees of freedom does this retrieval have".
A library this size does not split by band: everyone who uses it works at ② to ④.

(st-am-below)=
## What argminax needs from below

This is where argminax differs most: its floor is larger than its shelf.

| From | Surfaces to a caller | Card in |
| --- | --- | --- |
| optimistix | yes, everywhere: a cost is written in its `fn(y, args)` form, argminax's solvers and adjoints subclass its abstract ones, and a solve that does not converge is reported through its `RESULTS` | the substrate card below |
| gaussx | yes: every curvature is a gaussx operator, and one Newton step is `gaussx.newton_update` | gaussx's structures and primitives card and its strategies card ([](#tbl-sg-cards)) |
| lineax | yes: the PSD tag on every curvature operator | covered by gaussx's card |
| matfree | no: behind the `[matfree]` extra, imported when called | none |

The optimistix card is the first substrate card drafted for the tree, from argminax's `AGENTS.md` @ `c7bd0c3`:

```markdown
- **Provides.** Minimisation, least squares and root finding in JAX, with protocols for solvers and for adjoints.
- **Nouns.** A cost or residual function in the form `fn(y, args)`.
- **Verbs.** `optx.minimise`, `optx.least_squares`, `optx.root_find`.
- **Adverbs.** A solver and an adjoint. `optx.ImplicitAdjoint` differentiates through a converged minimum.
- **Keep.** A solve that does not converge is reported through `optx.RESULTS`; read it before using the value.
- **Never.** Read optimistix's private solve layout from outside argminax, which pins one minor version because its `KStep` does.
- **One rung down.** Open optimistix's own documentation to write a new solver; argminax's `add-solver` says how to subclass one.
```

(st-am-differs)=
## Where argminax bends the pattern

1. **The floor can be larger than the shelf.** Most of what an argminax caller must know is optimistix's and gaussx's. The listed skill is short, and its cards do the work.
2. **Substrate cards have an owner by first use.** No library of ours owns optimistix, so geostack keeps the card; the library that leans on it hardest drafts it, here argminax.
3. **Cards before readers.** With no dependent at the pins, the generated tables are empty. The shelf is still written, so each switch-over is one change.
4. **A boundary is a line on two cards.** "One step is gaussx's; the loop is argminax's" belongs on gaussx's inference card and on argminax's solver card, in the same words.
5. **A small library keeps one listed skill.** Three units, one front door, and no second band.

(st-am-gates)=
## Gates

Gates GX-0 to GX-6 of [gaussx's shelf](./skill_tree_gaussx.md#sg-gates) apply with argminax's names, and one is added.

| Gate | Passes when |
| --- | --- |
| AM-1 Substrate | every third-party name in a substrate card is importable at the version the library pins |

(st-am-risks)=
## Risks

- **The substrate card is drafted from argminax's notes, not from optimistix.** It has not been checked against the pinned optimistix version. Fallback: gate AM-1, before the card ships.
- **gaussx's Laplace step moves here.** argminax's `AGENTS.md` says gaussx's `laplace_mode` moves to argminax. Fallback: gaussx's inference unit loses names and its table regenerates; the boundary line on both cards stays.

## Relationships

- Parent: jejjohnson/geostack#150
- Related: jejjohnson/geostack#151 (the design), jejjohnson/geostack#155 (shelves for the other libraries)
- Related: [skill tree](./skill_tree.md), [gaussx's shelf](./skill_tree_gaussx.md)
