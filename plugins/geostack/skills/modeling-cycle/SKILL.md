---
name: modeling-cycle
description: The modeling cycle of the GeoStack, the research loop that every package serves. Two tracks (modelling turns a conceptual model into equations, code and predictions; observing turns a sensor's measurements into observations) meet where predictions are compared with observations, and the kind of misfit decides which of two loops runs. A random misfit, one the error model explains, runs the fast inner loop of state estimation and parameter estimation; a systematic misfit runs the slow outer loop that changes the model's structure or the observing system. Use when a model and the data disagree and the next step must be chosen, when planning how a model is to be improved, when explaining where data assimilation, calibration and model development sit relative to each other, or when deciding whether added complexity is earned.
---

# The modeling cycle

Draft 1 · 19 Nov 2026 · from the overview's first axis (`docs/overview.md`, "The modeling cycle", @ `854276a`). Where this file and that page disagree, the page wins.

The modeling cycle is the research loop: model, observe, compare, refine. Every package in the stack exists to make one or more of its steps composable. It is the first of the stack's three organising axes; the other two are `data-levels` and `six-step-cycle`, both in this plugin.

## 1. Two tracks

Both start from the same real-world system.

| Track | Runs | Ends in |
|---|---|---|
| Modelling | a conceptual model, then equations and code | predictions |
| Observing | a sensor, then measurements | observations |

The tracks meet at **compare**: the predictions against the observations. What is left over is the misfit.

## 2. Two loops

The kind of misfit decides which loop runs.

| Misfit | Loop | What changes | What stays |
|---|---|---|---|
| Random: one the error model explains | inner, fast | the state $`\mathbf{x} \in \mathbb{R}^{D_x}`$, by **state estimation** (data assimilation); the parameters $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$, by **parameter estimation** (calibration) | the model's structure; the same model runs again |
| Systematic: one it does not | outer, slow | the model's structure (back to the conceptual model), or the observing system (more or better data, better processing) | nothing is fixed |

The inner loop updates the predictions. The outer loop returns to the start of a track.

## 3. Start simple

Add complexity only when validation demands it. A change to the model's structure is an outer-loop move, so it needs a systematic misfit to justify it: a misfit that estimation did not remove.

## 4. Choosing the next step

When a model and the data disagree, ask in this order. The overview gives the two tracks and the two loops; the order of questions below is this skill's reading of them, so treat steps 2 and 4 as prompts for a judgement and not as rules.

1. **Is there an error model?** State what the observation error and the model error are assumed to be, with their covariances $`\boldsymbol{\Sigma}_\varepsilon`$ and $`\boldsymbol{\Sigma}_\eta`$. Without one, no misfit can be called random or systematic.
2. **Does the error model explain the misfit?** Compare the residuals with what it predicts. For a Gaussian error model the innovation statistic $`\chi^2 = \bar{\mathbf{r}}^\top \hat{\boldsymbol{\Sigma}}_y^{-1} \bar{\mathbf{r}}`$ has expectation $`D_y`$, and filterax's diagnostics compute it. Residuals of the expected size with no pattern in space, time or regime are random.
3. **Random: run the inner loop.** Estimate the state when the misfit changes from window to window, and the parameters when it is constant over time and sample. The same model runs again.
4. **Systematic: decide which track is at fault before changing either.** A bias that follows the instrument, the retrieval or the viewing geometry points at the observing track. One that follows a regime of the dynamics points at the model.
5. **Outer loop on the model: make the smallest structural change** that addresses the pattern, and keep the previous model as the reference it is scored against.

State each judgement on the page, with its evidence.

## 5. Where the cycle meets the other axes

- **The inner loop is where data assimilation lives.** On `data-levels` it is the compute step into L4 (model and assimilate). On `six-step-cycle` it is step 2, model-based inference, the oracle.
- **The outer loop is step 6 of `six-step-cycle`**: improve, by changing the structure.
- **The misfits come from matchup**, the operation that pairs satellite and in-situ observations (`data-levels` §4).

## Checklist

- [ ] The error model is stated before a misfit is called random or systematic.
- [ ] The evidence for that call is given: the residual statistic, and the pattern or its absence.
- [ ] A random misfit leads to estimation with the same model; a systematic one to a change of structure or of the observing system.
- [ ] Each structural change is the smallest that addresses the pattern, and is scored against the model before it.
