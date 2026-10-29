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

The advection term moves a quantity along with the flow. In the continuous equations it can be written in three ways that are all equal. However, on a grid they stop being equal, as each one puts the approximation in a different place and keeps a different set of quantities exactly. In this page, we go from the PDE to the stencils, in finitevolX's conventions, and run all three on the same test.

+++

{button}`Open the full page <./pages/advection-c-grid.html>`

**Where it sits.** A refresher on the term the learned schemes replace, and what each discrete form conserves. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches, which opens on its own and links back here.

## TL;DR: Same term, different approximation

Here $q(\vect{s}, t) \in \mathbb{R}$ is the advected quantity, $\vect{v} = (u, v) \in \mathbb{R}^2$ the velocity in m s⁻¹, $\psi(\vect{s}, t) \in \mathbb{R}$ the streamfunction, $J(\psi, q) \in \mathbb{R}$ the Jacobian of the two fields, and $\nabla := \nabla_{\vect{s}}$ is the gradient in the spatial coordinates $\vect{s} \in \mathbb{R}^2$.

- **Flux form** $\nabla \cdot (\vect{v} q)$: move $q$ to the faces where $u$ and $v$ live, multiply, and take a difference. The difference is exact, so *all* of the error is in the face value of $q$. It conserves $\sum q$ for any face rule, and this is where upwind and WENO live.
- **Advective form** $\vect{v} \cdot \nabla q$: take the gradient of $q$ at the centre and move $u$ and $v$ there. The error is in the gradient and in moving the velocity, and it conserves nothing by construction. This is the slot a spectral gradient fits.
- **Jacobian form** $J(\psi, q)$, for QG: combine $\psi$ and $q$ directly. Arakawa's average of three versions conserves $\sum q$, $\sum q^2$ and energy exactly, but it adds no dissipation, so fronts ring unless viscosity is added.
- **Centred schemes never damp, whereas upwind-biased ones always do.** On a sharp front that is the whole difference: the centred schemes keep the variance and ring, upwind smears, and WENO5 does neither.

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
