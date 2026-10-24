---
title: "Weak-constraint 4D-Var: when the model is allowed to be wrong"
subtitle: "esmax · ocean · 4 · Weak-constraint 4D-Var: model error as a control, the forcing, state and reduced formulations, and how to choose Q"
short_title: "4 · Weak-constraint 4D-Var"
description: "Strong-constraint 4D-Var treats the model as exact and corrects only the initial state. Weak-constraint 4D-Var admits that the model makes errors, and estimates them alongside the state. This page follows the 4D-Var primer: the intuition, the probability, the mathematics, how the gradient changes, three formulations, and how it maps onto the ocean problem."
date: 2026-10-24
keywords:
  - 4D-Var
  - weak constraint
  - model error
  - time-parallel
  - bias estimation
  - Lorenz-96
  - vardax
---

+++ {"part": "abstract"}

Strong-constraint 4D-Var treats the model as exact and corrects only the initial state. Weak-constraint 4D-Var admits that the model makes errors, and estimates them alongside the state. This page follows the 4D-Var primer: the intuition, the probability, the mathematics, how the gradient changes, three formulations, and how it maps onto the ocean problem.

+++

{button}`Open the full page <./pages/weak-4dvar.html>`

**Where it sits.** When the model is wrong: estimating the state and the model error together. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches; it opens on its own and links back here.

## TL;DR: Same cost, plus a price for every model error

- **The transition gets noise:** $z_k = \mathcal M_k(z_{k-1}) + \eta_k$ with $\eta_k\sim\mathcal N(0, \mathbf Q)$. The cost gains $\tfrac12\sum_k\|\eta_k\|^2_{\mathbf Q^{-1}}$. Strong constraint is the limit $\mathbf Q\to0$.
- **The controls grow** from one state to a whole trajectory: $N$ values become $N(T+1)$. The analysis can now fit observations the model alone could not reach.
- **The gradient is still one adjoint sweep** in the forcing formulation; in the state formulation it becomes $T$ independent pieces, parallel in time.
- **$\mathbf Q$ is the hard part.** Too small and the model error leaks into the initial state; too large and the analysis chases observation noise. A basis for the model error turns $\mathbf Q$ into a modelling choice.
- **On Lorenz-96 with a biased model,** strong constraint leaves a window error of 1.66; weak constraint cuts it to 0.25, and a one-number bias estimate to 0.12 while recovering the bias itself (0.99, true 1.0).

## On the page

1. Intuition
2. Where it comes from
3. The mathematics
4. The gradient
5. Formulation 1 · Model errors as controls (forcing)
6. Formulation 2 · States as controls (time-parallel)
7. Formulation 3 · Reduced model error
8. The three side by side
9. Application: the Gulf Stream window
10. Notes on vardax
