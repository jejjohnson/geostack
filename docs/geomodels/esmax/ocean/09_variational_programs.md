---
title: "Variational data assimilation, written as probabilistic programs"
subtitle: "esmax · ocean · 9 · 3D-Var, strong, incremental and weak-constraint 4D-Var, each full and reduced, written as numpyro models with gaussx, kernellib and geonnax priors"
short_title: "9 · Variational DA as programs"
description: "3D-Var, strong-constraint 4D-Var and weak-constraint 4D-Var, each in a full and a reduced form, are one idea at six sizes, and incremental 4D-Var solves the strong-constraint pair a second way: a Gaussian prior on whatever is uncertain, a model that maps it to observations, and the most probable value given the data. This page writes each as a numpyro model whose covariances are gaussx operators (circulant, Kronecker, low-rank) built from kernellib kernels, with reduced bases from kernellib feature maps and geonnax eigenfunctions. The cost function is then derived, not hand-written; the analysis is the model's MAP; the Laplace approximation is its uncertainty."
date: 2026-10-25
keywords:
  - 3D-Var
  - 4D-Var
  - incremental 4D-Var
  - weak constraint
  - reduced order
  - numpyro
  - gaussx
  - kernellib
  - Laplace approximation
---

+++ {"part": "abstract"}

3D-Var, strong-constraint 4D-Var and weak-constraint 4D-Var, each in a full and a reduced form, are one idea at six sizes, and incremental 4D-Var solves the strong-constraint pair a second way. The idea is a Gaussian prior on whatever is uncertain, a model that maps it to the observations, and the most probable value given the data. In this page, we write each one as a numpyro model whose covariances are gaussx operators (circulant, Kronecker, low-rank) built from kernellib kernels, with reduced bases from kernellib feature maps and geonnax eigenfunctions. So the cost function is derived and not hand-written, the analysis is the model's MAP, and the Laplace approximation is its uncertainty.

+++

{button}`Open the full page <./pages/variational-programs.html>`

**Where it sits.** A companion to pages 3 and 4: the same estimators as probabilistic programs, with structured covariances and their Laplace spread. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches, which opens on its own and links back here.

## TL;DR: Six priors, one likelihood, two ways to solve

- **A variational method is a choice of control and prior.** 3D-Var controls one state, strong 4D-Var the initial state, and weak 4D-Var the initial state and every model error. Each one has a reduced form which only keeps a few coefficients of the correction (of the state, of the initial state, or of the model error).
- **Each is a short numpyro model.** The latent sites are the controls, the rollout is a deterministic site, and the observations are one masked Gaussian site. numpyro's potential energy is the 4D-Var cost $\mathcal{J}$ plus a constant.
- **The covariances are structured gaussx operators**, and their square roots keep the structure: `gx.circulant` for a stationary prior state covariance $\gmat{\Sigma}_x \in \mathbb{R}^{D_x \times D_x}$ (FFT), `gx.Kronecker` for a space-time model-error covariance $\gmat{\Sigma}_\eta \in \mathbb{R}^{D_x \times D_x}$, and `gx.low_rank_plus_diag` for a reduced $\gmat{\Sigma}_x$ (Woodbury).
- **Sample whitened, not in physical units.** The control-variable transform is a reparameterisation. On 3D-Var it cuts the optimiser from 1,952 to 27 iterations per analysis, for the same analysis (they differ by at most 0.0014).
- **On Lorenz-96 with a biased model**, the window error goes from 0.64 (strong) to 0.27 (weak) to 0.12 (one bias coefficient, recovered as 1.00 ± 0.04 with a truth of 1.0). Long windows have many minima, so we solve them quasi-statically.

## On the page

1. Intuition: one ladder
2. The program behind every method
3. The machinery: priors as structured objects
4. 1 · 3D-Var
5. 2 · Reduced 3D-Var
6. 3 · Strong-constraint 4D-Var
7. 4 · Reduced strong-constraint 4D-Var
8. 5 · Incremental 4D-Var
9. 6 · Reduced incremental 4D-Var
10. 7 · Weak-constraint 4D-Var
11. 8 · Reduced weak-constraint 4D-Var
12. Inference: MAP, quasi-static windows, Laplace
13. The eight side by side
14. At ocean scale
15. Notes from building this page
