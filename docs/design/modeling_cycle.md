---
title: "The modeling cycle and the six-step cycle"
subtitle: "Background · tutorial · how a model improves through two feedback loops, and how a trusted slow method becomes a fast one without losing its referee"
short_title: The modeling cycle
description: "A visual tutorial on the modeling cycle (compare, the inner loop of estimation, the outer loop of refinement) and the six-step cycle (simple model, model-based inference, emulator, emulator-based inference, amortized inference, improve), run on one methane plume whose first model is wrong in a way tuning cannot fix."
date: 2026-11-14
keywords:
  - modeling cycle
  - six-step cycle
  - calibration
  - emulator
  - amortized inference
  - simulation-based inference
---

+++ {"part": "abstract"}

Background for axes 1 and 3 of the [overview](../overview.md). A methane plume is observed under a wind that blows 12° off the forecast. The first model trusts the forecast; the page follows compare, the inner loop and the outer loop until the misfit is noise, then takes the repaired model through the six steps and measures what each fast step costs and how far it strays from its referee. For the ocean-model version of the six steps see [the six-step cycle](../geomodels/esmax/ocean/01_six_step_cycle.md).

+++

{button}`Open the full page <./pages/modeling-cycle.html>`

**Status.** Background. The example is one file of NumPy, SciPy and scikit-learn, run in November 2026 on a synthetic scene; the sampler is a random-walk Metropolis, the emulator and the amortized map are small networks, and no GeoStack package is used.

## TL;DR: two loops, and a ladder of references

- **Compare decides.** A misfit the noise explains runs the fast inner loop (estimate state and parameters); a misfit with a pattern runs the slow outer loop (change the model or the data).
- **The inner loop cannot repair a wrong model:** it fits model 1 with a spread coefficient of 0.58 ± 0.01 where the truth is 0.30, and the misfit still has a pattern (χ² 2.41).
- **One structural change fixes it:** estimating the wind direction gives χ² 0.95 and recovers all three parameters.
- **Then make it fast, in steps:** 65 s per scene with the simulator, 3.0 s with an emulator, 0.42 ms with an amortized map.
- **Every fast step has a slower one as its referee,** and the improve step is shown working on the emulator and on the amortized map.
- **Fast is not free:** the amortized map is less sharp than the sampler, and says so with wider error bars that were checked.

## On the page

1. The running example: one plume, two models
2. Part 1, the modeling cycle: the two tracks, compare, the inner loop, the outer loop
3. Part 2, the six-step cycle: each step with its equation, code, cost and referee
4. Step 6 as a table of who is scored against whom
5. How the two cycles nest, and what a change of model resets
6. Where each step lives in the stack and outside it
