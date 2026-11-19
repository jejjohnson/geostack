---
name: six-step-cycle
description: The six-step cycle of the GeoStack, the path from a simple model to amortized inference. Step 1 is a simple model that can be simulated, step 2 classical model-based inference, step 3 an emulator of the model, step 4 the same inference with the emulator, step 5 an amortized map from observations to a posterior, and step 6 improves any of them. Each step has a reference it must reproduce (the emulator against the simulator, both fast inversions against the classical method, which is the oracle), a place in the stack, and a cost per query that falls from hours to milliseconds. Use to plan the route from a research model to a fast one, to decide what a new method is validated against and trained on, to order the work and the tests of a library or a project, to place a method on the cycle, or to read each step as a statement about distributions.
---

# The six-step cycle

Draft 2 · 19 Nov 2026 · from the overview's third axis (`docs/overview.md`, "The six-step cycle"), the report that ran it on the pinned code (`docs/reports/17_cycle.md`) and the esmax page that writes it in probabilities (`docs/geomodels/esmax/ocean/01_six_step_cycle.md`), all @ `854276a`. The rule for skipping steps 3 and 4 and the check of an emulator's derivative are from plumax's design pages (`docs/design/01_tier1_gaussian.md`, `04_rtm_stack.md`). Where this file and those pages disagree, the pages win, and the report wins over the overview.

The six-step cycle is the journey from research to production. It is the third of the stack's three organising axes; the other two are `data-levels` and `modeling-cycle`, both in this plugin.

## 1. The setting

Let $`\boldsymbol{\theta} \in \mathbb{R}^{D_\theta}`$ be the quantity we want (an emission rate, a model state), $f$ the physical forward model, $h$ the observation operator, $`\mathbf{y} \in \mathbb{R}^{D_y}`$ the observations and $`\boldsymbol{\varepsilon} \in \mathbb{R}^{D_y}`$ their error. In the cycle $`\boldsymbol{\theta}`$ stands for whatever is inferred, the state included.

```math
\mathbf{y} = h(f(\boldsymbol{\theta})) + \boldsymbol{\varepsilon}
```

```math
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}_\varepsilon)
```

## 2. The six steps

| Step | What it is | Trained on | Reference it must reproduce | Lives in |
|---|---|---|---|---|
| 1 · Simple model | $f$ and $h$, which can be run forward to draw pairs $`(\boldsymbol{\theta}, \mathbf{y})`$ | nothing | — | GeoModels (somax, plumax, radtrax) |
| 2 · Model-based inference | the posterior $`p(\boldsymbol{\theta} \mid \mathbf{y}) \propto p(\mathbf{y} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})`$ by a classical method | nothing | none of the other steps; it is the **oracle** for the steps after it | `pipekit-cycle.DACycle` with classical analysis steps (filterax, vardax, pyrox) |
| 3 · Emulator | $`\hat{f}(\cdot; \boldsymbol{\theta}_\text{emu})`$, fitted to $f$ | step-1 simulations | step 1, the simulator | `pipekit-train.SimulationDataset`, registered in `pipekit-experiment.ModelRegistry` |
| 4 · Emulator-based inference | step 2 with $`\hat{f}`$ in place of $f$ | nothing new | step 2, the oracle | `pipekit-cycle.DACycle` with `NeuralForward` |
| 5 · Amortized inference | a conditional density $`q(\boldsymbol{\theta} \mid \mathbf{y}; \boldsymbol{\nu})`$ that maps observations straight to an approximate posterior | step-1 simulations | step 2, the oracle | pipekit-train with gauss_flows or pyrox |
| 6 · Improve | rerun, compare row by row, change one step | — | the step before | `pipekit-evaluate.Benchmark`, with lineage in pipekit-experiment |

The emulator and the amortized map are fitted by

```math
\boldsymbol{\theta}_\text{emu}^\star = \mathrm{argmin}_{\boldsymbol{\theta}_\text{emu}} \ \mathbb{E}_{\boldsymbol{\theta} \sim p(\boldsymbol{\theta})} \left[ \Vert f(\boldsymbol{\theta}) - \hat{f}(\boldsymbol{\theta}; \boldsymbol{\theta}_\text{emu}) \Vert_2^2 \right]
```

```math
\boldsymbol{\nu}^\star = \mathrm{argmin}_{\boldsymbol{\nu}} \ \mathbb{E}_{(\boldsymbol{\theta}, \mathbf{y}) \sim p(\boldsymbol{\theta}, \mathbf{y})} \left[ -\log q(\boldsymbol{\theta} \mid \mathbf{y}; \boldsymbol{\nu}) \right]
```

Moving right, the cost of a query falls from hours to milliseconds, because the training cost is paid once, up front.

## 3. The reference rule

Each learned step is scored against the earlier step that computes the same thing.

- **The emulator against the simulator** (step 3 against step 1), on the inputs of the forward model it emulates, not on the observations.
- **Both fast inversions against the oracle** (steps 4 and 5 against step 2).
- **The amortized map is trained on simulations**, not on step 4's outputs.

This is the rule that pipekit-evaluate's benchmark ladder ships, and the report found it to be the consistent one; an older wording in the overview differs. It matters because a small emulator error is not a small posterior error: in the report's run, an emulator within 3.2 % of the simulator moved the posterior by up to 1.9 standard deviations.

**What each comparison measures.**

- **The emulator** is scored on held-out inputs of the simulator. When the inversion of step 4 uses gradients, its derivative is scored against the simulator's too, since a forward pass can be accurate while its Jacobian is not.
- **A fast inversion** is scored on the same cases as the oracle, scene by scene, on its mean and on its spread. Keep a fixed set of cases with the oracle's posteriors stored, and score every later method on it.
- **The truth and the oracle are two references, and both go on the row.** Against a known truth, a method is checked for calibration: that it knows how wrong it is on average over the prior. That is necessary and not enough, since a posterior equal to the prior passes it. Against the oracle, it is checked for reproducing the posterior of each case. In the report's run the amortized map was calibrated against the truth and 53 % too wide against the oracle where the signal was small.
- **The cost of a query is measured** on each route, since cost is what the later steps are for.

## 4. Two commitments

- **Step 2 is the oracle.** Classical methods are not legacy code waiting to be replaced by neural ones. They generate the references that validate steps 3 to 5, so the classical path stays alive. It is the oracle because it is the most trusted reference there is, not because it is exact: it is exact only in tractable cases, such as a linear-Gaussian model solved by a Kalman filter or a GP, and a finite-ensemble EnKF or 4D-Var approximates it, slowly.
- **Benchmarks formalise understanding.** Writing a benchmark contract (carriers, references, metrics, splits, baselines, expected failure modes) is formalising what is understood about the problem. Every transition between steps is a benchmarkable contract.

**The oracle has a strength.** In a physics-first cycle step 1 is a known forward model and step 2 a classical inversion of it: an oracle in the strong sense. In a data-first cycle step 1 is a labelled dataset and step 2 a classical supervised baseline: an oracle in the weak sense, since matching it does not make a method right if the labels are wrong. Say which one a result was scored against.

## 5. The same six steps as distributions

Read with functions, the cycle starts from a deterministic model. Read with distributions, that model is one special case. Here $`\mathbf{x}`$ is the state and $`\boldsymbol{\theta}`$ the physical parameters.

| Step | As a distribution |
|---|---|
| 1 | a joint $`p(\boldsymbol{\theta}) p(\mathbf{x} \mid \boldsymbol{\theta}) p(\mathbf{y} \mid \mathbf{x})`$ that can be sampled; a deterministic simulator is the case where $`p(\mathbf{x} \mid \boldsymbol{\theta})`$ is a point mass |
| 2 | the posterior $`p(\boldsymbol{\theta}, \mathbf{x} \mid \mathbf{y})`$ of that joint; every algorithm (Kalman, 4D-Var, EnKF, particles, MCMC) approximates it |
| 3 | a learned transition $`q(\mathbf{x}_k \mid \mathbf{x}_{k-1}, \boldsymbol{\theta}; \boldsymbol{\theta}_\text{emu})`$ |
| 4 | the same posterior, with that one density swapped |
| 5 | a learned posterior $`q(\boldsymbol{\theta}, \mathbf{x} \mid \mathbf{y}; \boldsymbol{\nu})`$ |
| 6 | a comparison of distributions: each learned one against the one it approximates, and the whole model against data |

Each learned density is trained by a KL divergence to the distribution it replaces. Training and use are separate phases: steps 3 and 5 are fitted once, offline, on simulations, and then every route (2, 4 or 5) returns posterior draws that are used for estimates, forecasts, predictions of new data and probabilities of events. Linear-Gaussian is the one case where every integral is matrix algebra; outside it, posteriors skew and split, and a shortcut that assumes one Gaussian can fail.

## 6. Using the cycle

**To place a method**, ask what it replaces. A model that is run forward is step 1. A method that inverts a model it is given is step 2, or step 4 when that model is learned. A network trained to imitate a model is step 3. A network that maps observations to an answer is step 5.

**To plan a project or a library**, first read what the library ships and which step each piece is, then go in order and do not skip the oracle.

1. Write the simple model and check that it simulates.
2. Invert it classically on a case small enough to trust. This is the reference for everything after.
3. Only then train the emulator, and score it against the simulator.
4. Rerun the inversion with the emulator, and score the posterior against step 2.
5. Train the amortized map on simulations, and score it against step 2 and against the truth.
6. Improve one step at a time. Changing a step makes what was built on it stale, so rerun those and no others.

**Steps 3 and 4 may be skipped; step 2 may not.** The emulator exists to make an expensive forward model cheap. Where the forward model is already cheap and the cost is the sampler, as in plumax's Gaussian plume, the route goes from step 2 to step 5, which is trained on simulations and does not need step 4. Time the classical route first to see where the cost is, say that the steps were skipped and what would bring them back, and state what the skipped steps would have been trained on and scored against, so the route is still complete when a costlier model replaces the simple one.

**To review a claimed result**, ask which step it is, what it was trained on, which reference it was scored against, and how strong that reference is. The usual faults are these.

- The emulator was scored against observations, which mixes its error with the model's and the instrument's.
- The amortized map was trained on the outputs of step 4, which makes it a distillation of that inversion, errors included.
- A learned step was compared only with the step that taught it, so an error the two share cannot show.
- Only the mean was compared, and the spread was never tested.
- The classical method no longer runs, so nothing can be scored again after a change.

**In a library**, each step is a gate. vardax writes the cycle as a chapter of its mathematical reference and wires the gates into its tests.

## 7. Where the cycle meets the other axes

- **With `data-levels`:** the five transitions between levels crossed with steps 1 to 5 give a grid in which every cell is a concrete method and a benchmarkable artifact. The whole lifecycle is a nested set of six-step cycles.
- **With `modeling-cycle`:** step 2 is its inner loop, and changing a model's structure at step 6 is its outer loop.

## Checklist

- [ ] Each method named is on one step, and the step is stated.
- [ ] A learned step says what it was trained on: simulations from step 1.
- [ ] Each step after the second names its reference: the simulator for the emulator, the oracle for both fast inversions.
- [ ] The oracle's strength is stated: a known forward model, or a labelled dataset.
- [ ] The classical method exists and runs before anything learned is trusted, and still runs afterwards.
- [ ] A fast inversion is scored on mean and spread, against the oracle and against the truth, and its cost per query is measured.
- [ ] A skipped step is named, with the reason and what would bring it back.
- [ ] After a step changes, what depends on it is rerun.
