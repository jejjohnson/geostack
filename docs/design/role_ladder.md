---
title: "The role ladder: where a piece of code belongs"
subtitle: "Background · explainer · seven levels of a library, from structures to recipes, taught with one job that runs"
short_title: The role ladder
description: "A visual introduction to the role ladder of the kinds and roles page: structures, primitives, strategies, models, algorithms, steps and recipes, with a methane emission example written first as a notebook cell and then rung by rung; background for kinds and roles."
date: 2026-11-13
keywords:
  - role ladder
  - library design
  - pipekit
  - operators
  - recipes
---

+++ {"part": "abstract"}

Background for [kinds and roles](./kinds_and_roles.md), which defines the ladder for the whole stack. This page teaches it with one job, a methane emission rate from a hyperspectral scene, written as a notebook cell and then rebuilt one rung at a time, in code that runs.

+++

{button}`Open the full page <./pages/role-ladder.html>`

**Status.** Background. The code on the page is one file, run in November 2026 with NumPy, SciPy, scikit-learn, einx and pipekit-ops; its strategies and model are plain dataclasses and `Scene` stands in for `GeoArray`, so that it runs without JAX.

## TL;DR: seven rungs, one direction

- **The ladder is grammar.** ⓪ structures are nouns, ① primitives verbs, ② strategies adverbs, ③ models noun phrases, ④ algorithms sentences, ⑤ steps sentences with their settings, ⑥ recipes paragraphs.
- **Calls go down only**, so the ladder is also the order to build and test in.
- **The lower five compute on plain arrays**; geographic data and saved settings enter at ⑤.
- **Composition stops at ⑥**; running a recipe is orchestration, which is not a rung.
- **It pays three ways:** reuse (one fitted background serves two stages), swap (another estimator is a one-word change), save (the recipe as JSON, loaded back, returns the same 5.68 kg s⁻¹).

## On the page

1. The running example and the notebook cell it starts from
2. The ladder, and what is not a rung
3. Part 1: the rungs one at a time, each with its test and its code
4. Part 2: the example on the ladder, with what it bought, measured
5. Part 3: the rules that make it a ladder
6. Placing a new name, and the confusing cases
7. Who works where
8. The same ladder in libraries you know
