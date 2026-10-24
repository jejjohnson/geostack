---
title: "Learned WENO: an advection scheme that learns where to dissipate"
subtitle: "esmax · ocean · 8 · An advection scheme that keeps WENO's stencils, convexity and masks and learns only how to weigh the candidates"
short_title: "8 · Learned WENO"
description: "The advection scheme is where a coarse ocean model loses information: it smooths fronts, dissipates eddies, and decides whether a run stays stable. This page keeps everything that makes WENO safe (flux form, fixed candidate stencils, convex weights, land masks) and learns one thing: how to weigh the candidates at each face. Same Gulf Stream QG example as the other pages."
date: 2026-10-24
keywords:
  - WENO
  - learned advection
  - hybrid models
  - attention
  - mixture of experts
  - finitevolX
---

+++ {"part": "abstract"}

The advection scheme is where a coarse ocean model loses information: it smooths fronts, dissipates eddies, and decides whether a run stays stable. This page keeps everything that makes WENO safe (flux form, fixed candidate stencils, convex weights, land masks) and learns one thing: how to weigh the candidates at each face. Same Gulf Stream QG example as the other pages.

+++

{button}`Open the full page <./pages/learned-weno.html>`

**Where it sits.** One concrete hybrid (type T2 of the taxonomy), worked through from the basis functions to training and the adjoint. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches; it opens on its own and links back here.

## TL;DR: Fixed stencils, learned weights

- **Every advection scheme is a centred part plus a dissipation.** The centred part moves information; the dissipation removes it. Fifth-order upwinding is exactly sixth-order centred plus a hyperdiffusion $|u|\Delta x^5/60$.
- **A single learned kernel cannot work.** Godunov's theorem: a linear scheme that never creates oscillations is at most first order. The weights must depend on the local field.
- **WENO already does that, by hand.** It blends three small stencils with weights set by how smooth each one sees the field. Learned WENO adds a learned correction to those weights, starts from zero (so it starts as classical WENO5), and is switched off where the field is smooth.
- **Boundaries use the same mechanism.** A candidate that reads a land cell gets weight zero. Near a coast only the safe candidates remain, so the stencil narrows on its own.
- **It is a familiar machine-learning object:** soft attention, or a mixture of experts, over a fixed bank of local polynomial experts, scored by their roughness. The parallels section works each comparison through with the same numbers.
- **Every weight is smooth in the state**, so the adjoint is the true linearisation of the forecast: important for 4D-Var.

## On the page

1. The problem
2. Notation
3. Flux form: conservation for free
4. Every scheme is a centred part plus a dissipation
5. Why a single learned kernel cannot win
6. Classical WENO5, from local basis functions
7. Learned WENO
8. Parallels
9. As operators
10. Code sketch
11. Where it goes in the QG model
12. Training
13. For data assimilation
14. What to watch
15. Test ladder
16. Where it lives
