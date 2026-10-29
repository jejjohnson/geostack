---
title: "Ocean state estimation, from the generative story to each box"
subtitle: "esmax · ocean · 2 · The Gulf Stream SSH-mapping problem written once, then read box by box as operators with inputs, outputs and shapes"
short_title: "2 · Ocean state estimation"
description: "The traditional formulation of estimating the ocean state from sea-level maps, written once, then read box by box. Steps 1 and 2 are the classical problem. Steps 3 to 6 are what that problem turns into when parts of it are learned."
date: 2026-10-24
keywords:
  - ocean state estimation
  - sea surface height
  - quasi-geostrophic
  - data assimilation
  - operators
---

+++ {"part": "abstract"}

This is the traditional formulation of estimating the ocean state from sea-level maps, which we write once and then read box by box. Steps 1 and 2 are the classical problem, whereas steps 3 to 6 are what that problem turns into when parts of it are learned.

+++

{button}`Open the full page <./pages/ocean-state-estimation.html>`

**Where it sits.** The running example: the data, the QG model, and what each of the six boxes takes in and returns. The full page is a self-contained document with operator boxes (inputs, outputs, shapes and units), figures and code sketches, which opens on its own and links back here.

## TL;DR: Every box is an operator

- **Forward boxes (1, 3)** map a sea-level *field* and two physical parameters to the next field. Box 1 chains that map into a *cube* of states and a cube of synthetic maps.
- **Inverse boxes (2, 4, 5)** all have one signature: a cube of daily SSH maps with their masks and errors goes in, and a cube of states plus the parameters and an uncertainty comes out. They only differ in what is inside and in what each query costs.
- **Box 6** takes many windows of those outputs and returns diagnostics and one decision, which is either to rerun the inner loop or to change the model.

## On the page

1. The problem and the data
2. Notation
3. The mathematical objects
4. Simple model: the generative story
5. Model-based inference: the classical ways
6. Emulator: replace 𝓕 by a network
7. Emulator-based inference: same cost, new model
8. Amortized inference: learn the inverse
9. Improve: the misfit decides which loop runs
10. The six boxes, read from the formulation
