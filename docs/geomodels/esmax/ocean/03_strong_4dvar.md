---
title: "4D-Var: fitting a model trajectory to a window of observations"
subtitle: "esmax · ocean · 3 · Strong-constraint 4D-Var: intuition, the cost as a negative log posterior, the adjoint, and classic, incremental and reduced-basis algorithms"
short_title: "3 · 4D-Var"
description: "Four-dimensional variational assimilation (4D-Var) asks one question: which initial state, run forward through the model, best explains every observation in a time window, without straying far from what we believed before? This page is the background for the learned data assimilation page: the intuition, the mathematics, how the gradient is computed, three classic algorithms, and how it maps onto an ocean problem."
date: 2026-10-24
keywords:
  - 4D-Var
  - strong constraint
  - adjoint
  - incremental 4D-Var
  - reduced basis
  - diffrax
  - checkpointing
  - vardax
---

+++ {"part": "abstract"}

Four-dimensional variational assimilation (4D-Var) asks one question: which initial state, run forward through the model, best explains every observation in a time window without straying far from what we believed before? This page is the background for the learned data assimilation page, and it covers the intuition, the mathematics, how the gradient is computed, three classic algorithms, and how all of it maps onto an ocean problem.

+++

{button}`Open the full page <./pages/strong-4dvar.html>`

**Where it sits.** Background for box 2 when the model is trusted: the cost, its gradient, and how to minimise it. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches, which opens on its own and links back here.

## TL;DR: One cost, one gradient, three ways to minimise

- **4D-Var is the MAP estimate** of the initial state under a Gaussian background, Gaussian observation errors and a model with no error (strong constraint). So the cost $\mathcal{J}$ is minus the log posterior.
- **The control is the initial state** $\vect{x}_0 \in \mathbb{R}^{D_x}$, and every later state $\vect{x}_k = \vect{x}(t_k)$ is the model run from it. The observations at all times pull on $\vect{x}_0$ through the model.
- **The gradient is one adjoint run**, which is the backward sweep of the tangent-linear model. In JAX it is reverse-mode differentiation of the forward run, and checkpointing trades memory for recomputation.
- **Classic** minimises the full nonlinear cost with a quasi-Newton method. **Incremental** relinearises a few times and solves quadratic sub-problems with CG. **Reduced-basis** optimises a few coefficients of a basis instead of every grid value.
- **On Lorenz-96** all three reach the same analysis error (0.20, from a background error of 1.00), with very different costs.

## On the page

1. Intuition
2. Where it comes from
3. The mathematics
4. Computing the gradient
5. Algorithm 1 · Classic 4D-Var
6. Algorithm 2 · Incremental 4D-Var
7. Algorithm 3 · Reduced-basis 4D-Var
8. The three side by side
9. Application: the Gulf Stream window
10. Notes on vardax, from reading the code
