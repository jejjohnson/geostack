---
title: "The six-step cycle, written in probabilities"
subtitle: "esmax · ocean · 1 · The simulate, infer, emulate, amortize and improve loop as statements about distributions, with linear and nonlinear examples under each step"
short_title: "1 · Six-step cycle"
description: "The six-step figure is usually read with functions: a model f, an operator H, an emulator f(\\cdot;\\phi). This page rewrites every box as a statement about distributions: a prior, a likelihood, a posterior, and learned densities that approximate them. A deterministic model then becomes one special case instead of the starting point. Nothing here assumes linearity; linear and nonlinear instances appear only as worked examples underneath each step, to show where closed forms come from and why they stop."
date: 2026-10-24
keywords:
  - six-step cycle
  - probabilistic model
  - posterior
  - emulator
  - amortized inference
  - simulation-based inference
---

+++ {"part": "abstract"}

The six-step figure is usually read with functions: a model $f$, an observation operator $h$ and an emulator $f(\cdot; \gvect{\theta}_\text{emu})$. In this page, we rewrite every box as a statement about distributions, which means a prior, a likelihood, a posterior, and learned densities that approximate them. So a deterministic model becomes one special case instead of the starting point. Nothing here assumes linearity. Linear and nonlinear instances only appear as worked examples underneath each step, to show where the closed forms come from and why they stop.

+++

{button}`Open the full page <./pages/six-step-cycle.html>`

**Where it sits.** Start here. Every later page is one box of this cycle, written for the ocean. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches, which opens on its own and links back here.

## TL;DR: Six boxes, six distributions

Here $\vect{x} \in \mathbb{R}^{D_x}$ is the state, $\vect{y} \in \mathbb{R}^{D_y}$ the observations and $\gvect{\theta} \in \mathbb{R}^{D_\theta}$ the physical parameters.

- **Step 1 is a joint distribution** $p(\gvect{\theta}) p(\vect{x} \mid \gvect{\theta}) p(\vect{y} \mid \vect{x})$ that you can sample. A deterministic simulator is the special case where $p(\vect{x} \mid \gvect{\theta})$ is a point mass.
- **Step 2 is the posterior** $p(\gvect{\theta}, \vect{x} \mid \vect{y})$ of that joint. It is exact by definition, and every algorithm (Kalman, 4D-Var, EnKF, particles, MCMC) is an approximation of it, which is slow but trusted.
- **Steps 3 to 5 replace densities with learned ones:** $q(\vect{x}_k \mid \vect{x}_{k-1}, \gvect{\theta}; \gvect{\theta}_\text{emu})$ for the simulator's transition (the emulator), the same posterior computed with that emulator (step 4), and $q(\gvect{\theta}, \vect{x} \mid \vect{y}; \gvect{\nu})$ for the posterior itself (amortized). Each one is trained by a KL divergence to the distribution it replaces.
- **Training and use are separate phases.** Steps 3 and 5 are fitted once, offline, on simulations. Then every route (2, 4 or 5) returns posterior draws, and the draws are *used*: estimates, forecasts, predictions of new data and probabilities of events.
- **Step 6 compares distributions**, which means each learned one against the one it approximates, and the whole model against data.
- **Linear-Gaussian is the one case where every integral is matrix algebra.** Outside of it, posteriors skew and split, their shape depends on the data, and every shortcut that assumes one Gaussian can fail.

## On the page

1. The pipeline: train once, use per window
2. Why distributions instead of functions
3. Notation
4. Step 1 · The simple model is a joint distribution
5. Step 2 · Model-based inference is the posterior of that joint
6. Step 3 · The emulator is a learned conditional density
7. Step 4 · Emulator-based inference: the same posterior, one density swapped
8. Step 5 · Amortized inference: learn the posterior as a function of the data
9. Using a posterior: estimate, forecast, predict
10. Step 6 · Improve: compare distributions, then change the structure
11. Linear versus nonlinear, side by side
12. Bridge to ocean state estimation
