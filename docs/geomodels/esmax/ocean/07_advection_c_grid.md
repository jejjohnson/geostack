---
title: "Advection on the C-grid: one term, three ways to compute it"
subtitle: "esmax · ocean · 7 · Flux, advective and Arakawa Jacobian forms of the advection term, in finitevolX's stencil conventions, run on one test"
short_title: "7 · Advection on the C-grid"
description: "The advection term moves a quantity along with the flow. In the continuous equations it can be written three ways that are all equal. On a grid they stop being equal: each one puts the approximation in a different place and keeps a different set of quantities exactly. This page goes from the PDE to the stencils, in finitevolX's conventions, and runs all three on the same test."
date: 2026-10-24
keywords:
  - advection
  - Arakawa C-grid
  - finite volume
  - Arakawa Jacobian
  - conservation
  - finitevolX
---

+++ {"part": "abstract"}

The advection term moves a quantity along with the flow. In the continuous equations it can be written three ways that are all equal. On a grid they stop being equal: each one puts the approximation in a different place and keeps a different set of quantities exactly. This page goes from the PDE to the stencils, in finitevolX's conventions, and runs all three on the same test.

+++

{button}`Open the full page <./pages/advection-c-grid.html>`

**Where it sits.** A refresher on the term the learned schemes replace, and what each discrete form conserves. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches; it opens on its own and links back here.

## TL;DR: Same term, different approximation

- **Flux form** $\nabla\cdot(\mathbf u q)$: move $q$ to the faces where $u, v$ live, multiply, take a difference. The difference is exact; *all* the error is in the face value of $q$. Conserves $\sum q$ for any face rule. This is where upwind and WENO live.
- **Advective form** $\mathbf u\cdot\nabla q$: take the gradient of $q$ at the centre and move $u, v$ there. The error is in the gradient and in moving the velocity. Conserves nothing by construction. This is the slot a spectral gradient fits.
- **Jacobian form** $J(\psi, q)$, for QG: combine $\psi$ and $q$ directly. Arakawa's average of three versions conserves $\sum q$, $\sum q^2$ and energy exactly, but adds no dissipation, so fronts ring unless viscosity is added.
- **Centred schemes never damp; upwind-biased ones always do.** On a sharp front that is the whole difference: the centred schemes keep the variance and ring, upwind smears, WENO5 does neither.

## On the page

1. What the advection term says
2. Where the term appears
3. Three ways to write the same term
4. The grid
5. Form 1 · Flux form (finite volume)
6. Form 2 · Advective form
7. Form 3 · Jacobian form (Arakawa)
8. All three, on the same test
9. Momentum advection, briefly
10. Back to the question
