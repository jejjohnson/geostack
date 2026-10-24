---
title: "Learned data assimilation: what can be learned in the analysis"
subtitle: "esmax · ocean · 6 · Eleven ways to learn a piece of the analysis: the objective, its gradient, the solver, the output, or the whole inverse"
short_title: "6 · Learned data assimilation"
description: "The taxonomy page took the model apart. This page takes the assimilation apart: the objective, its gradient, the solver, the output, and whether the answer is computed per window or learned across windows. Eleven ways to learn one of those pieces, on the same Gulf Stream QG example and the same 10-day window of SSH maps."
date: 2026-10-24
keywords:
  - learned data assimilation
  - 4DVarNet
  - variational inference
  - amortized inference
  - learned optimisers
---

+++ {"part": "abstract"}

The taxonomy page took the model apart. This page takes the assimilation apart: the objective, its gradient, the solver, the output, and whether the answer is computed per window or learned across windows. Eleven ways to learn one of those pieces, on the same Gulf Stream QG example and the same 10-day window of SSH maps.

+++

{button}`Open the full page <./pages/learned-data-assimilation.html>`

**Where it sits.** Boxes 4 and 5 taken apart: what can be learned in the assimilation itself. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches; it opens on its own and links back here.

## TL;DR: One analysis operator, eleven insides

- **Every type is box 2's operator:** maps, masks, errors, background, covariates and forcing in; a state window and its uncertainty out. What changes is the inside, the training data it needs, and whether the output is a point, an ensemble or a density.
- **The training signal matters more than the network.** Simulated truth (OSSE) learns the QG model's ocean; the ELBO and held-out observations learn the real one but never see unobserved scales.
- **Mean, mode or sample.** 4D-Var returns a mode, an MSE-trained network a mean, VI a density. Benchmarks must compare the same statistic.
- **Teal types fail loudly**: the classical cost or solver is still there to check the answer. **Amber types fail silently**: an unusual window gets a confident wrong field. Scoring any output with the classical 𝒥 and its innovations turns most silent failures into loud ones.

## On the page

1. Notation
2. The master equation
3. Training signals
4. The eleven types
5. Model type × assimilation type
6. What the structure implies
7. Where each type lives
