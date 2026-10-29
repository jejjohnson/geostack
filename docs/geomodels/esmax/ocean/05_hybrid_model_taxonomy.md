---
title: "Hybrid model taxonomy: what can be learned in an ocean model"
subtitle: "esmax · ocean · 5 · Ten things that can be learned in an ocean model, from parameters to the whole rollout, and how each is glued back"
short_title: "5 · Hybrid model taxonomy"
description: "The physical model is not one object. It is a stack of pieces: parameters, discretisation, terms, a tendency, an integrator, a step map, a rollout. Every hybrid or emulator learns one of those pieces and glues it back in one of a few ways. This page lists the ten that matter, on the same Gulf Stream QG example as the state-estimation page."
date: 2026-10-24
keywords:
  - hybrid models
  - emulators
  - closures
  - neural ODE
  - quasi-geostrophic
  - taxonomy
---

+++ {"part": "abstract"}

The physical model is not one object. It is a stack of pieces (parameters, discretisation, terms, a tendency, an integrator, a step map, a rollout), and every hybrid or emulator learns one of those pieces and glues it back in one of a few ways. In this page, we list the ten that matter, on the same Gulf Stream QG example as the state-estimation page.

+++

{button}`Open the full page <./pages/hybrid-model-taxonomy.html>`

**Where it sits.** Box 3 taken apart: which layer of the model is learned, and what each choice keeps. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches, which opens on its own and links back here.

## TL;DR: One signature, ten insides

- **Every type from T1 to T8 is the same operator:** today's state, covariates, forcing and parameters go in, and tomorrow's state comes out. So any of them drops into the cycles, the DA and the rollouts of the last page unchanged. T9 and T10 change the signature.
- **Types differ in which layer is learned** (the diagram below) and in how the learned piece is glued back: added, used as a coefficient, built into a conserving form, or substituted.
- **The training target splits them into two jobs.** *Emulation* reproduces the physical model faster (box 3, where the ground truth is step 1), whereas *correction* makes the model closer to the real ocean (box 6, where the ground truth is a richer model or data).
- **The lower the learned layer, the more physics survives:** conservation, freedom in $\Delta t$ and a trustworthy adjoint. The higher the layer, the more speed and the less structure.

## On the page

1. Notation
2. The physical model, taken apart
3. Three axes
4. The ten types
5. Selection matrix
6. What the structure implies
7. Where each type lives
