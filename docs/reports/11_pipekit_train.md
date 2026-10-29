---
title: pipekit-train
subtitle: Report 11 · Training pipelines for emulators and amortized inference, as shipped and as still needed
short_title: 11 · pipekit-train
description: pipekit-train, the stack's thin training layer. The scoping report is checked against pipekit-train 0.0.2, which takes precedence; the package is less complete than pipekit-cycle, so every idea in the scoping report that has not shipped is kept as a proposal with a gate.
keywords:
  - pipekit-train
  - emulators
  - amortized inference
  - simulation-based inference
  - Equinox
  - NumPyro
  - BlackJAX
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

pipekit-train turns a dataset, a model and a loss into a trained pipekit `Operator` without anyone writing a fourth training loop, because it hands the loop over to an existing tool. Version 0.0.2 ships the core of the scoping report (datasets, losses, `TrainingLoop`, callbacks, sweeps, a training artifact) and more than it asked for, namely three Bayesian backends (NumPyro SVI, NumPyro MCMC, BlackJAX) and a windowed Zarr dataset for training on stored model states. However, it is also further from finished than pipekit-cycle. Equinox is the only gradient backend that runs, `CatalogDataset` does not exist, and the running example hits eight defects, which go from a trajectory path that crashes to a cache that serves the wrong simulator's data. **Every unshipped idea from the scoping report stays on the table** as a proposal with a gate. An emulator of [Report 10](10_pipekit_cycle.md#r10-running)'s transport model trains to $3.8 \times 10^{-7}$ of the true operator and drops into a cycle unchanged.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Partly built. Originally a scoping proposal ("committed"); pipekit-train 0.0.2 is the design of record where it has decided, and the scoping report's unshipped ideas are kept as proposals |
| **Reading time** | about 35 minutes |
| **Audience** | anyone training an emulator, a retrieval network or a posterior estimator with the stack, or extending the backends |
| **Companions** | [Report 10](10_pipekit_cycle.md) (the forward models and cycles an emulator replaces); [Report 13](13_state_catalog.md) (the stored states a dataset reads); [Report 9](09_use_cases.md) (use cases); [Report 12](12_pipekit_experiment.md) (pipekit-experiment); [Report 17](17_cycle.md#r17-running) (the five-band sensor); [Report 16](16_geo_data_stack.md#r16-ownership) (the stack rules and the one model-operator family) |
| **Code checked against** | `pipekit-train` 0.0.2 at `ef3c17f`, with pipekit, pipekit-cycle, pipekit-experiment and pipekit-jax from the same revision and geopatcher from geotoolz `088475e`. JAX on CPU with Equinox, Optax, Grain, Orbax, NumPyro and BlackJAX; torch, Lightning and Keras were not installed (their adapters are scaffolds that never import them). Read on 13 October 2026 |
| **Running example** | every number below is from one run of `notebooks/reports/report11_pipekit_train.py`, which seeds every draw and asserts each number the page shows (wall times are printed, not asserted) |
| **Design docs** | the package's `docs/design/` (vision, architecture, boundaries, 14 ADRs, API pages, examples) were read alongside the code |
:::

:::{important} How this page treats the scoping report
pipekit-train's own design docs already took most of the scoping report on board, as its vision cites "Report 11" as a source and ADRs D1 to D14 record what was decided. Where the code or an ADR decided differently (steps rather than epochs, the artifact's field names, the extras), the code stands, as everywhere in the series ([conventions](../overview.md#sec-conventions)). Where the scoping report asked for something that has not shipped (`CatalogDataset`, the Lightning and Keras backends, the parquet and tfrecord caches, a model that round-trips through the registry), it becomes a workstream below rather than an open question, because the package's own docs still plan it.
:::

(r11-summary)=
## Summary

**Decision.** pipekit-train stays a thin layer over existing training tools, as both the scoping report and ADR D1 say, and we keep its 0.0.2 design: step-based loops (D10), `run()` returning `(model_op, TrainingArtifact)` (D6), Equinox with Grain and Orbax (D8), and the Bayesian backends (D13, D14). Five workstreams finish it.

- **A. Datasets.** `CatalogDataset` is missing (A1). `SimulationDataset` crashes on its `cycle=` path and hides prior errors (A2), and its content hash ignores the simulator, so a cache can serve stale data (A3). `CachedDataset` reads one zarr item at 8.2 ms and only supports zarr (A4). `XarrayWindowDataset` imports a geopatcher module that no longer exists (A5).
- **B. The trained model's handoff.** The Equinox backend returns an `EquinoxModelOp` which cannot be stored with its weights or reloaded from `LocalModelRegistry`, whereas wrapping it in pipekit-jax's `JaxModelOp` works (B1). We also fill the artifact's model hash without requiring a registry (B2).
- **C. Loop components.** `Checkpoint` ignores `save_dir` and does not save the final step (C1), `HyperSweep` ranks trials on the last training batch instead of on validation (C2), `ValidationStep` is never used by the loop (C3), and NumPyro SVI runs an un-jitted update at 76 ms per step (C4).
- **D. Amortized inference.** The data path and `NLL` work, but every user writes their own distribution head and dataset flip. So we ship a recipe and a Gaussian head, and we decide where conditional density estimators live (D1).
- **E. The other two backends.** We build the Lightning (E1) and Keras 3 (E2) adapters that ADR D5 schedules for v0.2 and v0.3.

**End state.** The scoping report's three training shapes all run from pipekit-train: 1) supervised training from a catalog, 2) emulator training from a pipekit-cycle forward model, and 3) amortized inference. Each one produces an operator that drops into an inference pipeline or a cycle, reloads from the registry with its weights, and carries an artifact whose hashes identify its data and its weights.

(r11-ownership)=
## Ownership rules

pipekit-train owns the glue between the data, a model and a training tool, whereas the tools own the training maths and the model libraries own the models ([](#fig-r11-ownership)).

:::{figure} ../images/reports/r11_ownership.svg
:label: fig-r11-ownership
:alt: Data sources (pipekit-cycle, Zarr stores, and geocatalog as planned) feed pipekit-train's datasets. Its loop runs on Equinox, NumPyro SVI, NumPyro MCMC or BlackJAX; Lightning and Keras are scaffolds. The loop returns a trained operator for pipekit-cycle and a TrainingArtifact for pipekit-experiment.

pipekit-train 0.0.2: its layers, its backends and its two handoffs. Dashed
boxes and chips are planned or scaffold only.
:::

**pipekit-train** owns the datasets, losses, the loop, callbacks, sweeps, writers and the adapters to training tools. Its core only imports `pipekit`. JAX, torch and Keras arrive through extras, and `pipekit_experiment` is only imported by `LogToExperiment` and the artifact. It never reimplements an optimizer step or a sampler, it never defines its own artifact type, and it has no model-definition DSL (see its `boundaries.md`).

The neighbours (the training tools, pipekit-cycle, pipekit-experiment and the model libraries) follow [the stack rules](16_geo_data_stack.md#r16-ownership).

:::{warning} Differs from the plan
The scoping report's extras were `lightning`, `equinox`, `keras`, `array`, `jax`, `catalog`, `cycle` and `experiment`. 0.0.2 has `equinox`, `lightning`, `keras`, `numpyro`, `blackjax`, `cycle`, `experiment`, `cache`, `s3`, `gcs` and `xreader`. There is no `array` or `jax` extra because trained models are Equinox operators rather than pipekit-array `ModelOp`s (pipekit-array has no `ModelOp`). There is no `catalog` extra either, as its `pyproject.toml` says it "lands once geocatalog publishes to PyPI". The Equinox extra also pulls in Grain, Orbax and jaxtyping.
:::

(r11-surface)=
## What ships

[](#tbl-r11-surface) lists the public surface by layer.

:::{table} pipekit-train 0.0.2's public surface. Rows in italics go beyond the scoping report.
:label: tbl-r11-surface

| Layer | Names | Module | State |
|---|---|---|---|
| Datasets | `TrainingDataset`, `SimulationDataset`, `CachedDataset` | `dataset` | shipped; `CachedDataset` is zarr only |
| *Datasets* | *`IterableDataset(source, content_hash)`, `XarrayWindowDataset` (lazy), `hash_config`* | *`dataset`, `xarray_window`* | *shipped* |
| Datasets | `CatalogDataset` | | **not built** |
| Losses | `Loss` (Protocol), `MSE`, `NLL`, `KL`, `Composite` | `loss` | shipped; duck-typed, return `(scalar, metrics)` |
| Loop | `TrainingLoop`, `TrainerCarryState`, `ValidationStep` | `loop` | shipped; `ValidationStep` standalone |
| Components | `Callback`, `EarlyStopping`, `Checkpoint`, `LogToExperiment` | `callbacks` | shipped |
| *Components* | *`MetricWriter`, `JSONLWriter`; `HyperSweep`, `ParameterGrid`, `SweepResult`, `SweepCarryState`* | *`writer`, `sweep`* | *shipped; sweeps are sequential* |
| Backends | `equinox` | `adapters.equinox` | shipped: `EquinoxModelOp`, `TrainTask`, `TrainState`, `train_step`, `save_state`, `restore_state`, `ShardingSpec` |
| Backends | `lightning`, `keras` | `adapters.lightning`, `adapters.keras` | scaffolds: `run()` raises `NotImplementedError` ("v0.2", "v0.3") |
| *Backends* | *`numpyro-svi`, `numpyro-mcmc`, `blackjax`* | *`adapters.numpyro_svi`, `.numpyro_mcmc`, `.blackjax`, `.bayes`* | *shipped: `NumpyroTask`, `NumpyroPredictiveOp`, `BlackjaxTask` (NUTS only)* |
:::

The package's tests pass: 228 tests in 76 s with geopatcher shimmed (see stage 8), plus the BlackJAX (9 of 9) and NumPyro suites, which are marked slow.

(r11-running)=
## The running example: an emulator, a retrieval and a stored archive

The stages reuse the two problems of [Report 10](10_pipekit_cycle.md), so that we can check a trained model against the exact operator it replaces. In the six-step cycle, stage 2 is step 3 of a forecast (and stage 3 runs that emulator forward), and stages 6 and 7 are steps 5 and 2 of a retrieval. [Report 17](17_cycle.md#r17-running) runs all six steps on the same retrieval.

**The transport model** is the upwind advection of [Report 10](10_pipekit_cycle.md#r10-running) on a periodic 40-cell transect, $\vect{x}_{t+1} = \mat{F} \vect{x}_t$ with $\mat{F} = 0.6 \mat{I} + 0.4 \mat{S}$. Here $\vect{x}_t \in \mathbb{R}^{D_x}$ is the state at step $t$ with $D_x = 40$, $\mat{F} \in \mathbb{R}^{D_x \times D_x}$ is the linear transport model, $\mat{S} \in \mathbb{R}^{D_x \times D_x}$ is the periodic one-cell shift and $\mat{I}$ is the identity (the code keeps the array name `M` for $\mat{F}$). An **emulator** is a learned $\hat{\mat{F}} \in \mathbb{R}^{D_x \times D_x}$, and the target is $\hat{\mat{F}} = \mat{F}$.

**The retrieval** uses the five-band sensor of [Report 17](17_cycle.md#r17-running) unchanged, $y_b = e^{-k_b \alpha} + \varepsilon_b$ for the bands $b = 1, \ldots, 5$, with the absorption coefficients $\vect{k} = (0.002, \ldots, 0.010) \in \mathbb{R}^5$, the noise standard deviation $\sigma_\varepsilon = 10^{-3}$ and the column enhancement $\alpha \in [0, 300]$, a scalar. We stack the bands into the spectrum $\vect{y} \in \mathbb{R}^5$ and write the noise-free spectrum as $h(\alpha) \in \mathbb{R}^5$. An **amortized** retrieval learns $p(\alpha \mid \vect{y})$ once, for every $\vect{y}$, whereas a per-scene retrieval ([Report 10](10_pipekit_cycle.md#r10-stage-5)'s Gauss–Newton, or MCMC) solves for one $\vect{y}$ at a time.

**The archive.** Fifty hourly states of the transect, written to a Zarr store, stand in for the stored model states of [Report 13](13_state_catalog.md#r13-running).

(r11-stage-1)=
### Stage 1 · Training data from a forward model (pipekit-cycle → pipekit-train)

**TL;DR.** We turn the transport model of [Report 10](10_pipekit_cycle.md#r10-running) into 2,000 input-output pairs with fixed seeds and with train, validation and test splits.

**Problem.** A `SimulationDataset` draws each input $\vect{x}_n \in \mathbb{R}^{D_x}$ from a prior $p(\vect{x})$ with the seed $s_0 + o + n$, where $s_0$ is the base seed, $o$ is the split's offset and $n$ indexes the samples, and it returns the pair $(\vect{x}_n, \mat{F} \vect{x}_n)$. The splits are 80, 10 and 10 % of the seed range, so they never share a draw.

```{code-block} python
:caption: An emulator's training data is a SimulationDataset over a pipekit-cycle forward model.
:linenos:

# Shapes: D_x = 40 (the code's N); N = 2,000 pairs; train 1,600 · val 200 · test 200
import pipekit_cycle as pc
import pipekit_train as pt

fwd = pc.CallableForward(lambda x, dt: M @ x, dt=1.0)    # F x: (D_x,) → (D_x,)    today
class WhitePrior(pk.Operator):                           # x ~ 𝒩(0, I)    user
    def _apply(self, seed=0):
        return np.random.default_rng(seed).normal(size=N)

train = pt.SimulationDataset(fwd, WhitePrior(), n_samples=2000, seed=0)  # today
val = train.with_split("val")                          # disjoint seeds   today
len(train), len(val)                                   # → 1600, 200      today
train.content_hash()                                   # sha256 of the recipe

pt.SimulationDataset(fwd, WhitePrior(), 2000, cycle=pc.Cycle(fwd, 24))[0]
# → TypeError: Cycle._apply() missing 1 required positional argument: 'state'
```

The one-step path works. However, the trajectory path, `cycle=`, calls `self.cycle(params)` without a carry state, so every pipekit-cycle cycle raises (`dataset.py:284`). A second defect sits just above it. The prior is called as `prior(seed=...)` inside `try … except TypeError`, and any `TypeError` raised *inside* the prior silently falls back to `prior()` with no seed, which means every sample comes out identical. Workstream A2 fixes both.

:::{warning} Differs from the plan
The scoping report's `SimulationDataset` had four arguments. The shipped one adds `obs_op` (an observation operator applied after the forward model, for SBI), `seed` and `split`, and it is indexable (`__len__`, `__getitem__`), so Grain can shuffle it without materialising it.
:::

(r11-stage-2)=
### Stage 2 · Train the emulator (pipekit-train → Equinox)

**TL;DR.** We fit a linear emulator $\hat{\mat{F}}$ by least squares on the pairs and see how the choice of prior decides what it learns.

**Problem.** We take $N$ pairs $(\vect{x}_n, \mat{F} \vect{x}_n)$ and a linear model $\hat{\mat{F}} = \mat{W}$, whose parameters are the entries of the weight matrix $\mat{W} \in \mathbb{R}^{D_x \times D_x}$, so $\gvect{\theta} = \op{vec}(\mat{W}) \in \mathbb{R}^{D_x^2}$. The mean-squared error {eq}`r11-eq-mse`,

$$
\mathcal{L}_\text{MSE}(\gvect{\theta}) = \frac{1}{N D_x} \sum_{n=1}^{N} \Vert \mat{W} \vect{x}_n - \mat{F} \vect{x}_n \Vert_2^2 ,
$$ (r11-eq-mse)

is zero at $\mat{W} = \mat{F}$, and the minimiser is only unique when the inputs span $\mathbb{R}^{D_x}$, that is when the stacked inputs $\mat{X} \in \mathbb{R}^{N \times D_x}$ (one $\vect{x}_n^\top$ per row) have full rank. The loop minimises it with Adam at a step size $\gamma = 10^{-2}$ on batches $\mathcal{D}^{(i)}$ of $N_\text{batch} = 64$ pairs, where $i$ counts the optimiser iterations.

```{code-block} python
:caption: The emulator trained with the Equinox backend, early stopping and validation, from a real run.
:linenos:

# Shapes: W (D_x, D_x); batch 64; 25 steps per epoch
import equinox as eqx
from pipekit_train.adapters.equinox import EquinoxModelOp

loop = pt.TrainingLoop(
    model_op=EquinoxModelOp(eqx.nn.Linear(N, N, use_bias=False, key=key)),
    dataset=train, val_dataset=val, loss=pt.MSE(),
    optimizer_config={"name": "adam", "lr": 1e-2},
    max_steps=6000, steps_per_epoch=25, batch_size=64, eval_every_n_steps=50,
    callbacks=(pt.EarlyStopping(metric="mse", patience=5),),
    seed=0,
)                                                                       # today
trained, artifact = loop.run()            # (EquinoxModelOp, TrainingArtifact)
# white prior:  stops at step 500;  val MSE 5.3e-14;  max |W − F| = 3.8e-7
# smooth prior: stops at step 2200; val MSE 3.4e-5;   max |W − F| = 0.343
```

:::{table} The same loop on two priors. The smooth prior draws three Gaussian bumps per sample.
:label: tbl-r11-priors

| Prior | Stopped at | Validation MSE | $\max \vert \mat{W} - \mat{F} \vert$ | $\sigma_{\min} / \sigma_{\max}$ of $\mat{X}$ | Wall time |
|---|---|---|---|---|---|
| white, $\mathcal{N}(\vect{0}, \mat{I})$ | step 500 (20 epochs) | $5.3 \times 10^{-14}$ | $3.8 \times 10^{-7}$ | well conditioned | 12.3 s |
| smooth bumps | step 2,200 (88 epochs) | $3.4 \times 10^{-5}$ | $0.343$ | $1.4 \times 10^{-5}$ | 45.4 s |
:::

Both runs fit their data ([](#tbl-r11-priors)), where $\sigma_{\min} / \sigma_{\max}$ is the ratio of the smallest to the largest singular value of $\mat{X}$. However, the inputs of the smooth prior nearly lie in a low-dimensional subspace, so $\mat{W}$ is only fixed on that subspace and is arbitrary off it. In other words, a low training loss is not a correct emulator. This is a property of the data and not of the code, and it is the reason a `SimulationDataset`'s prior is part of its hash.

:::{warning} Differs from the plan
The scoping report's `TrainingLoop` took `n_epochs`, and its `run()` returned the trained model alone. 0.0.2 counts steps instead (`max_steps`, with `steps_per_epoch` for epoch callbacks), as ADR D10 decided, and `run()` returns `(trained_model_op, TrainingArtifact)` (D6). The loop also takes `task`, `metric_writer`, `eval_every_n_steps`, `log_every_n_steps`, `checkpoint_dir` and `sharding` (D12), and its backend list has six entries. `TrainerCarryState` holds `(model, opt_state, step, epoch, metrics, rng)`, but the returned state has `opt_state` and `rng` set to `None`, so resuming goes through Orbax checkpoints and not through this state.
:::

(r11-stage-3)=
### Stage 3 · Drop the emulator into a cycle (pipekit-train → pipekit-cycle)

**TL;DR.** We replace the forward model of a six-hour forecast with the emulator, and we test it on states that the prior did and did not cover.

**Problem.** The emulated forecast is $\hat{\vect{x}}_6 = \mat{W}^6 \vect{x}_0$, where $\vect{x}_0 \in \mathbb{R}^{D_x}$ is the initial state. Its error $\Vert \mat{W}^6 \vect{x}_0 - \mat{F}^6 \vect{x}_0 \Vert_\infty$ is only small when $\vect{x}_0$ lies where $\mat{W}$ was fixed.

```{code-block} python
:caption: The trained operator is a forward model with no glue.
:linenos:

# Shapes: x₀ (D_x,)
emulator = pc.NeuralForward(trained, dt=1.0)                       # today
x6, _ = pc.Cycle(emulator, n_steps=6)(x0, None)                    # today
# white-prior W:  max |x̂₆ − F⁶ x₀| = 1.1e-6 (smooth x₀), 1.3e-6 (white x₀)
# smooth-prior W: max |x̂₆ − F⁶ x₀| = 0.017 (smooth x₀), 0.533 (white x₀)
```

The handoff that the scoping report promised works as written, since `NeuralForward` wraps the trained `EquinoxModelOp` directly and the cycles of [Report 10](10_pipekit_cycle.md#r10-stage-4) accept it. The smooth-prior emulator is accurate on smooth states but wrong by half the signal on white ones. So an emulator is only as general as its prior.

(r11-stage-4)=
### Stage 4 · Callbacks, checkpoints and the artifact (pipekit-train)

**TL;DR.** We watch the smooth-prior run through its callbacks and see what it records about itself.

**Problem.** A training run must be stoppable, resumable and traceable, which asks for an early-stopping rule, periodic checkpoints, a metric log, a tracker, and an artifact that names the data and the code.

```{code-block} python
:caption: The stage-2 run with every component attached, from a real run.
:linenos:

# Shapes: as stage 2
loop = pt.TrainingLoop(
    ..., callbacks=(
        pt.EarlyStopping(metric="mse", patience=5, min_delta=1e-9),
        pt.Checkpoint(every_n_steps=250, keep_last=2, save_dir="unused",
                      registry=px.LocalModelRegistry("registry")),
        pt.LogToExperiment(tracker, run_name="emu"),
    ),
    metric_writer=pt.JSONLWriter("metrics.jsonl"),
    checkpoint_dir="ckpt",
)                                                               # today
trained, art = loop.run()
# early stop at step 2200; ckpt/ holds 1750 and 2000 (not 2200)
# metrics.jsonl: 44 lines; tracker: start, step and 44 eval/* events, end
art.dataset_hash, art.model_registry_uri          # set               today
art.backend_info["final_step"], art.deps_lock     # 2200, uv pip freeze
loop.run()                                        # resumes at step 2000
```

Early stopping, the metric log, the tracker and auto-resume all work. However, `Checkpoint` has two defects. It writes to the loop's `checkpoint_dir` and ignores its own `save_dir` (the adapter only reads `every_n_steps` and `keep_last`). In addition, the final step is only saved if it is a multiple of `every_n_steps`, so a run that stops early loses its last 200 steps on resume. Workstream C1 fixes both.

:::{warning} Differs from the plan
The scoping report's training artifact had the fields `training_pipeline`, `dataset_hash`, `training_run_id`, `trained_model_uri`, `trained_model_hash`, `backend`, `backend_version`, `hardware` and `duration_seconds`. pipekit-experiment's `TrainingArtifact` has `training_pipeline_yaml`, `dataset_hash`, `trained_model_hash`, `tracker_run_id`, `model_registry_uri`, `backend_info` (a dict with the backend, versions, devices, `total_seconds`, sharding and final metrics), `deps_lock` and `metadata`. The scoping report's `Checkpoint` counted epochs, whereas the shipped one counts steps (`every_n_steps`).
:::

(r11-stage-5)=
### Stage 5 · Store and reload the emulator (pipekit-train → pipekit-experiment)

**TL;DR.** We put the trained emulator in the registry and load it back with its weights.

**Problem.** A registered model must reload to the same outputs, so after a store and a load, $\max \vert \mat{W}_\text{loaded} - \mat{F} \vert$ should equal the trained $3.8 \times 10^{-7}$.

```{code-block} python
:caption: The registry round trip, as it fails today and as it works through pipekit-jax.
:linenos:

# Shapes: W (D_x, D_x)
reg = px.LocalModelRegistry("registry")
# Today, through Checkpoint(registry=reg):
reg.load_weights(art.trained_model_hash)          # → None: no weights.bin
reg.load(art.trained_model_hash)
# → RuntimeError: from_state cannot reconstruct EquinoxModelOp … forbid_in_yaml

# What works today: re-wrap as pipekit-jax's JaxModelOp
j = JaxModelOp(trained.module)                                      # today
h = reg.store(j, weights=j.serialize_weights())                     # today
loaded = skeleton.from_registry(reg, h)   # skeleton: a JaxModelOp   today
# → max |W_loaded − F| = 3.8e-7; a 6-step NeuralForward cycle agrees too
```

The loop always returns an `EquinoxModelOp`, even when it was given a `JaxModelOp`, and `EquinoxModelOp` has no `serialize_weights`. So `Checkpoint`'s `registry.store` writes the operator's config and metadata but no weights. The design docs (ADR D7) say that the Equinox adapter returns a `JaxModelOp`, which the code does not do. Workstream B1 closes the gap.

(r11-stage-6)=
### Stage 6 · An amortized retrieval (pipekit-train → Equinox)

**TL;DR.** We train a network that returns a mean and an uncertainty for the column enhancement for any measured spectrum, and we check its calibration.

**Problem.** For $\alpha \sim \mathcal{U}(0, 300)$ and the spectra $\vect{y} \in \mathbb{R}^5$ it produces, a network $g_\text{enc}(\vect{y}; \gvect{\nu}) \in \mathbb{R}^2$ outputs a Gaussian $q(\alpha \mid \vect{y}; \gvect{\nu}) = \mathcal{N}(\alpha; \mu_\alpha(\vect{y}; \gvect{\nu}), \sigma_\alpha^2(\vect{y}; \gvect{\nu}))$, with the mean $\mu_\alpha \in \mathbb{R}$ and the standard deviation $\sigma_\alpha \in \mathbb{R}_{>0}$. Its weights $\gvect{\nu} \in \mathbb{R}^{D_\nu}$ are variational parameters (they parameterise an approximate posterior), and they are trained on the negative log-likelihood {eq}`r11-eq-nll`,

$$
\mathcal{L}_\text{NLL}(\gvect{\nu}) = \mathbb{E}_{\alpha, \vect{y}} \left[ \frac{1}{2} \log(2 \pi \sigma_\alpha^2(\vect{y}; \gvect{\nu})) + \frac{(\alpha - \mu_\alpha(\vect{y}; \gvect{\nu}))^2}{2 \sigma_\alpha^2(\vect{y}; \gvect{\nu})} \right] .
$$ (r11-eq-nll)

A calibrated $q$ has $\mathbb{E}[r^2] = 1$ for the standardised residual $r = (\alpha - \mu_\alpha) / \sigma_\alpha \in \mathbb{R}$, and about 95 % of its $\pm 2 \sigma_\alpha$ intervals cover the truth. No unbiased estimator can beat the Cramér–Rao bound $\sigma_\mathrm{CR} = \sigma_\varepsilon / \Vert \vect{J}_h(\alpha) \Vert_2$, where $\vect{J}_h(\alpha) = \partial h / \partial \alpha \in \mathbb{R}^5$ is the Jacobian of the noise-free spectrum, with entries $[\vect{J}_h]_b = -k_b e^{-k_b \alpha}$. An RMSE averages squared errors, so its floor is the root-mean-square bound over the prior, $(\mathbb{E}_\alpha[\sigma_\mathrm{CR}^2])^{1/2} = 0.257$, whereas the mean bound, $\mathbb{E}_\alpha[\sigma_\mathrm{CR}] = 0.228$, is the comparator for a mean posterior width. On the 200 test draws below, the two are 0.253 and 0.227.

```{code-block} python
:caption: The amortized retrieval: the simulator is the dataset, the loss is NLL, and two pieces of glue are user code.
:linenos:

# Shapes: params = (α, ε₁…ε₅): (6,); y (5,); network 5 → 64 → 64 → 64 → 2
sim = pt.SimulationDataset(pc.CallableForward(beer_lambert, dt=1.0),
                           AlphaNoisePrior(), n_samples=20_000)        # today
train = Flip(sim)            # (params, y) → (y, α): a TrainingDataset subclass  user

class GaussianHead(eqx.Module):                                        # user
    mean: jax.Array
    log_var: jax.Array
    # log q(α | y) = −½ (log 2π + log σ_α² + (α − μ_α)² / σ_α²)
    def log_prob(self, a): ...

loop = pt.TrainingLoop(model_op=EquinoxModelOp(Net(key)),  # y → GaussianHead
                       dataset=train, val_dataset=train.with_split("val"),
                       loss=pt.NLL(), optimizer_config={"name": "adam",
                       "lr": 2e-3}, max_steps=8000, batch_size=256)   # today
q, _ = loop.run()                                          # 425 s on CPU
# 200 test draws: RMSE 0.303 (RMS Cramér–Rao floor 0.253); bias −0.15
# ±2σ_α coverage 99.5 %; E[r²] = 0.43: σ_α is about 1.5× too wide
```

The network is useful and honest in shape, as its $\sigma_\alpha$ grows from 0.33 for $\alpha < 100$ to 0.74 for $\alpha > 200$ just as the bound does. However, it is under-confident, since $\mathbb{E}[r^2] = 0.43$ means its intervals are about 1.5 times too wide. `NLL` only needs `predicted.log_prob(target)`, which is the right seam, but there is no Gaussian head to plug into it, and the noise has to be drawn inside the prior so that it is seeded. Workstream D1 turns this into a recipe.

(r11-stage-7)=
### Stage 7 · The same retrieval with the Bayesian backends (pipekit-train → NumPyro, BlackJAX)

**TL;DR.** We solve one retrieval exactly with MCMC, as the reference that the amortized network should match.

**Problem.** For one measured spectrum $\vect{y} \in \mathbb{R}^5$ (with the true $\alpha = 120$), the posterior is {eq}`r11-eq-posterior`,

$$
p(\alpha \mid \vect{y}) \propto \mathbf{1}[0 \le \alpha \le 300] \prod_{b=1}^{5} \mathcal{N}(y_b; e^{-k_b \alpha}, \sigma_\varepsilon^2) ,
$$ (r11-eq-posterior)

which is nearly Gaussian here, with the Laplace standard deviation $\sigma_\varepsilon / \Vert \vect{J}_h(\alpha) \Vert_2 = 0.168$.

```{code-block} python
:caption: One NumPyro model, three backends, from a real run.
:linenos:

# Shapes: x = k (5, 1); y (5,); α scalar
def model(x, y=None):                                            # user
    a = numpyro.sample("alpha", dist.Uniform(0.0, 300.0))
    numpyro.sample("obs", dist.Normal(jnp.exp(-x[..., 0] * a), 1e-3), obs=y)

data = pt.IterableDataset(source=list(zip(k[:, None], y)),
                          content_hash="beer-lambert-y3")            # today
task = NumpyroTask(model, num_warmup=1000, num_samples=4000)         # today
for backend, t in [("numpyro-mcmc", task),
                   ("blackjax", BlackjaxTask(numpyro_task=task)),
                   ("numpyro-svi", task)]:
    op, art = pt.TrainingLoop(model_op=Dummy(), dataset=data, task=t,
                              backend=backend, max_steps=4000).run()  # today
# numpyro-mcmc: α = 120.143 ± 0.167, 0 divergences, 5.6 s
# blackjax:     α = 120.152 ± 0.168, 0 divergences, 10.4 s
# numpyro-svi:  α ≈ 119.96 ± 0.65 after 4,000 steps, 305 s
```

Both samplers agree with each other and with the Laplace width to the third decimal, so they are the oracle that we can score the amortized network of stage 6 against (its mean $\sigma_\alpha$ over the prior, 0.46, is close to three times this width). However, SVI, which minimises $\mathcal{L}_\text{ELBO} = -\mathrm{ELBO}$ (the negative evidence lower bound), is far from converged after 4,000 steps, and it is slow. This is because the adapter calls `svi.update` in a Python loop without `jax.jit`, at 76 ms per step on five data points. Workstream C4 jits the step. The Bayesian backends are not in the scoping report. They come from ADRs D13 and D14 ("inference is training too"), where NUTS is the oracle rung of the benchmark ladder.

(r11-stage-8)=
### Stage 8 · Train on stored states, and cache what you simulate (pipekit-train)

**TL;DR.** We read input and target windows from a Zarr archive of model states, and we cache the simulated pairs so that later runs skip the simulator.

**Problem.** For an archive of hourly states $\vect{x}_1, \ldots, \vect{x}_{50} \in \mathbb{R}^{D_x}$, a window dataset with an input stencil of the last three hours and a target one hour ahead yields the pairs $((\vect{x}_{t-2}, \vect{x}_{t-1}, \vect{x}_t), \vect{x}_{t+1})$ for every valid origin $t$. A cache keyed by the dataset's content hash must return exactly what the source would.

```{code-block} python
:caption: XarrayWindowDataset over an archive of states, and CachedDataset over a simulation.
:linenos:

# Shapes: archive (T = 50, D_x = 40); input (1, 3, D_x); target (1, 1, D_x)
stencils = {"input": TimeStencil("-2h", "0h", "1h", closed="both"),
            "target": TimeStencil("1h", "1h", "0h", closed="both")}
ds = pt.XarrayWindowDataset(source={"input": store, "target": store},
                            stencils=stencils, store_version="adv-v1")  # today
# → ImportError unless geopatcher.time is shimmed: geopatcher ships
#   geopatcher.temporal.stencils
len(ds), len(ds.with_split("val")), len(ds.with_split("test"))  # → 35, 1, 3
x, y = ds[0]                       # y = F x[0, −1] exactly            today

cached = pt.CachedDataset(sim_M, cache_dir="cache")                    # today
list(cached)                       # 400 items: 3.70 s, then 0.68 s
pt.CachedDataset(sim_2M, cache_dir="cache")[0]   # simulator now 2F …
# → returns the cached F pairs: the hash does not include the simulator
```

The window dataset is the training reader for the archives of [Report 13](13_state_catalog.md#r13-ws-b). Its `content_hash` covers `store_version`, the stencils and the origins, and a linear last-step model trained on it reached an MSE of $3.5 \times 10^{-5}$. However, two defects stand out. It imports `geopatcher.time`, which the pinned geopatcher no longer has (`xarray_window.py:35`), and its error message wrongly says that geopatcher is not installed (A5). In addition, the 80/10/10 contiguous split with a guard band left one validation window out of 47 origins, which is the expected cost of guard bands on a short archive. The cache is correct when the source is, but `SimulationDataset.content_hash` hashes the forward model's class name, `dt` and state signature and not its function, so swapping $\mat{F}$ for $2\mat{F}$ serves stale data (A3). It is also slow, because each `__getitem__` reopens the zarr store, at 8.2 ms against 0.05 ms to recompute this simulator (A4).

(r11-names)=
## The scoping report's names, checked

[](#tbl-r11-names) checks the stack names that the scoping report's examples use.

:::{table} The stack names the scoping report's examples use, at the pinned revisions.
:label: tbl-r11-names

| Name in the report | At the pinned revisions |
|---|---|
| `pt.TrainingLoop`, `pt.SimulationDataset`, `pt.CachedDataset`, `pt.MSE`, `pt.NLL`, `pt.EarlyStopping`, `pt.Checkpoint`, `pt.LogToExperiment` | exist; signatures differ as the boxes above say |
| `pt.CatalogDataset`, `dataset.with_split("val")` on it | `CatalogDataset` does not exist; `with_split` exists on every `TrainingDataset` |
| `pa.ModelOp(UNet(...))` | does not exist: pipekit-array has no `ModelOp`. Trained models are `EquinoxModelOp`; pipekit-jax has `JaxModelOp` |
| `loop.run()` returning the model | returns `(model_op, TrainingArtifact)` |
| `optimizer_config={"name": "adam", "lr": 1e-3}`, `n_epochs`, `backend="lightning"` | the optimizer dict works (adam, adamw, sgd, rmsprop); `n_epochs` is `max_steps`; `"lightning"` raises `NotImplementedError` |
| `gz.radiometry.ToFloat32`, `gz.radiometry.PercentileClip` | exist in geotoolz |
| `gz.cloud.LoadLabel` | does not exist: geotoolz has no `cloud` module |
| `gp.SpatialRegularStride((256, 256))` | geopatcher's sampler is `RegularStride` |
| `ChemistryForward`, `PlumeForward`, `ColumnObs` (plumax adapters), `pyrox.adapters.pipekit_train.ConditionalNormalizingFlow` | not checked: plumax and pyrox are not pinned here |
| `AtmosphericPrior`, `SourcePrior`, `EmulatorNet`, `UNet` | user code in the examples |
:::

(r11-audit)=
## Current-state audit

Of the twenty pieces we checked, ten ship and work, eight ship with defects, and two are missing ([](#tbl-r11-audit)), and [](#fig-r11-readiness) counts the page's calls by marker. The audit read `references/geostack/pipekit` at `ef3c17f` on 13 October 2026.

:::{figure} ../images/reports/r11_readiness.svg
:label: fig-r11-readiness
:alt: A grid with one row per stage of the running example and per workstream, and one square per stack call in that row's sketches, coloured by its marker. Of 32 calls, 24 run today, 3 are proposed by this report's workstreams and 5 are code the reader writes.

What this page's sketches call: of 32 stack calls, 24 run today, 3 are
proposed by this report's workstreams and 5 are code the reader writes
([how to read it](../overview.md#sec-conventions)).
:::

:::{table} pipekit-train 0.0.2, piece by piece.
:label: tbl-r11-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `TrainingDataset(seed, split)`, `with_split`, `content_hash` | `dataset.py:46` | every dataset | **Healthy.** Default hash is the config |
| `SimulationDataset(forward_model, prior, n_samples, cycle, obs_op, seed, split)` | `dataset.py:185` | stage 1, 6 | **Split ownership.** One-step path works; `cycle=` raises; prior `TypeError`s are swallowed; the hash omits the simulator |
| `IterableDataset(source, content_hash)` | `dataset.py:113` | Bayesian backends | **Healthy.** Hash is user-supplied |
| `CachedDataset(source, cache_dir, format="zarr")` | `dataset.py:345` | stage 8 | **Split ownership.** Correct but 8.2 ms per item; parquet and tfrecord raise |
| `CatalogDataset` | | the scoping report, `datasets.md`, `examples/direct.md` | **Missing.** |
| `XarrayWindowDataset` | `xarray_window.py:134` | stage 8 | **Split ownership.** Works with a shim; imports `geopatcher.time` |
| `Loss`, `MSE`, `NLL`, `KL`, `Composite` | `loss.py` | every gradient backend | **Healthy.** `NLL` needs `.log_prob`; `KL` needs `.kl_divergence` |
| `TrainingLoop`, `TrainerCarryState` | `loop.py:213`, `:34` | all stages | **Healthy.** Step-based; returns `(op, artifact)` |
| `ValidationStep(model_op, dataset, metrics)` | `loop.py:91` | | **Split ownership.** Standalone; the loop evaluates with the task's loss instead |
| `EarlyStopping(metric, patience, mode, min_delta)` | `callbacks.py:80` | stage 2, 4 | **Healthy.** |
| `Checkpoint(every_n_steps, keep_last, save_dir, registry)` | `callbacks.py:175` | stage 4 | **Split ownership.** `save_dir` ignored; final step not saved; registry gets no weights |
| `LogToExperiment(tracker, run_name, config)`, `JSONLWriter` | `callbacks.py:280`, `writer.py:35` | stage 4 | **Healthy.** |
| `HyperSweep`, `ParameterGrid` | `sweep.py:179`, `:39` | | **Split ownership.** Sequential; ranks on `final_metrics`, the last training batch |
| Equinox adapter, `EquinoxModelOp` | `adapters/equinox.py:66`, `run` at `:728` | stage 2–6 | **Healthy** as a trainer; **split ownership** at the registry (stage 5) |
| `ShardingSpec` | `adapters/equinox.py:113` | `scripts/multihost_sharding_smoke.py` | **Healthy.** Not exercised here |
| NumPyro MCMC, BlackJAX NUTS | `adapters/numpyro_mcmc.py`, `blackjax.py` | stage 7 | **Healthy.** BlackJAX samplers other than NUTS raise; MCMC's `final_metrics` is empty, so a sweep over it fails |
| NumPyro SVI | `adapters/numpyro_svi.py` | stage 7 | **Split ownership.** Un-jitted update, full batch only |
| Lightning, Keras adapters | `adapters/lightning.py`, `keras.py` | | **Missing.** Scaffolds (ADR D5) |
| `TrainingArtifact` handoff | `loop.py:404` | stage 4 | **Healthy.** `trained_model_hash` and `model_registry_uri` set only with a registry |
| docs | `README.md`, `docs/design/` | | **Split ownership.** Stale in 14 places (below) |
:::

There are also smaller findings from the design docs:

- **The README says the package is "planning only"** and that "the source directory currently contains no operators", and the `__init__` docstring still says "v0.0 ships the carrier-agnostic Protocols".
- **The ADRs are ahead of their status lines.** D13 and D14 say "implementation planned", `blackjax_adapter.md` says "Design only — no code yet", and `adapters.md` heads BlackJAX "(planned)", yet all three backends ship. `decisions.md` says "Eleven ADRs total" and lists 14.
- **Hyperparameter search contradicts itself**, as it is out of scope in `vision.md`, a non-goal in `boundaries.md` §12, and resolved and shipped in `boundaries.md` Q5.
- **Checkpoints are described three ways**: a Composite Orbax handler in `adapters.md`, `PyGrainCheckpointHandler` in D8, and the JSON side-car that the code actually writes (`data_iter.json`).
- **`IterableDataset` and the sweep classes are undocumented** in the API index, and the layout in `architecture.md` omits `xarray_window.py`, `_xreader.py` and the Bayesian adapters.
- **Seed ranges overlap across datasets.** The splits within one `SimulationDataset` are disjoint, but two datasets with `seed=0` and `seed=1` share all but one sample seed.

(r11-ws-a)=
## Workstream A · Datasets

**Decision.** Each of the three training shapes gets a dataset that is correct, hashed by everything that determines its contents, and fast enough to feed Grain ([](#tbl-r11-move-a)).

:::{table} Workstream A's moves.
:label: tbl-r11-move-a

| Symbol | Destination | Reason |
|---|---|---|
| `CatalogDataset(catalog, preprocess, target_op, sampler=None)` | `pipekit_train.dataset`, behind a `[catalog]` extra on geocatalog | the scoping report's direct-supervised shape; `datasets.md` and `examples/direct.md` already specify it |
| state-aware cycle call, error-transparent prior call | `SimulationDataset.__getitem__` | `cycle=` raises today; a `TypeError` inside the prior is swallowed |
| simulator identity in `content_hash` | `SimulationDataset.content_hash` | swapping $\mat{F}$ for $2\mat{F}$ keeps the hash and serves a stale cache |
| one open store per worker; parquet and tfrecord | `CachedDataset` | 8.2 ms per item today; the two formats raise |
| `geopatcher.temporal.stencils` | `xarray_window.py:35` | the module it imports was renamed |
:::

### A1 · `CatalogDataset`

1. **Read from a catalog.** It iterates the rows that a `GeoSlice` query returns, or the patches that a geopatcher `Sampler` (such as `RegularStride`) cuts from them, through geocatalog's `CatalogDomain` bridge.
2. **Reuse the inference preprocess.** `preprocess` is the same operator that the inference pipeline runs, so training and serving cannot drift apart.
3. **Hash the recipe.** `content_hash` covers the catalog's own hash (a `CatalogBundle` or the GeoParquet file's checksum), the query, the sampler config, the preprocess config, the split and the seed.
4. **Accept states too.** With the state profile of [Report 13](13_state_catalog.md#r13-ws-b), the same class reads a state catalog through `load_states`, so there is no separate `StateCatalog` type to support.

```{code-block} python
:caption: The scoping report's direct-supervised example, as it would read with A1.
:linenos:

# Shapes: chips (N_batch = 4, 256, 256) → labels (2, 256, 256)
dataset = pt.CatalogDataset(                                     # proposed (A1)
    catalog=gc.open_catalog("s3://imeo/labeled-scenes.parquet"),        # today
    preprocess=gz.radiometry.ToFloat32() | gz.radiometry.PercentileClip(
        p_min=2, p_max=98),                                             # today
    target_op=load_label,                                    # arguments sketched
    sampler=gp.RegularStride((256, 256)),                    # arguments sketched
)
```

### A2 · A correct simulation path

`SimulationDataset` calls `cycle(params, state0)` with a state that the user supplies (`cycle_state=`). It checks once, by signature, that the prior accepts `seed`, rather than catching `TypeError` on every call, and it documents that the observation noise is drawn inside the prior so that it is seeded.

### A3 · Hash the simulator

`content_hash` includes the forward model's `get_config()` and, for a `CallableForward`, a user-supplied `version` string, since a Python function cannot be hashed reliably. Without either, it warns that the cache cannot tell two simulators apart.

### A4 · A cache that keeps up

`CachedDataset` opens the zarr group once per process and reads in chunks. Parquet lands for tabular pairs, whereas tfrecord stays deferred unless a Lightning or Keras user needs it.

### A5 · The window dataset's import

We import `TimeStencil` from `geopatcher.temporal.stencils` (or from the geopatcher root, if it re-exports it), and we make the error message say which module is missing.

(r11-ws-b)=
## Workstream B · The trained model's handoff

**Decision.** Every trained operator can be stored with its weights and reloaded to the same outputs, and the artifact always names it ([](#tbl-r11-move-b)). B1 is absorbed by [Report 16 B1](16_geo_data_stack.md#r16-ws-b), which makes `JaxModelOp` the one Equinox wrapper and aliases `EquinoxModelOp` to it.

:::{table} Workstream B's moves.
:label: tbl-r11-move-b

| Symbol | Destination | Reason |
|---|---|---|
| return the input's operator type, or `JaxModelOp` | `adapters.equinox.run` | ADR D7 says the adapter returns a `JaxModelOp`; it returns `EquinoxModelOp`, which has no `serialize_weights` |
| `serialize_weights`, `from_registry` | `EquinoxModelOp`, if it stays | so `Checkpoint(registry=...)` writes `weights.bin` |
| `trained_model_hash` without a registry | `TrainingLoop.run` | a content hash of the weights needs no registry |
:::

```{code-block} python
:caption: Stage 5, before and after workstream B.
:linenos:

# Shapes: W (D_x, D_x)
# Today:
trained, art = loop.run()                       # EquinoxModelOp      today
j = JaxModelOp(trained.module)                  # manual re-wrap      user
h = reg.store(j, weights=j.serialize_weights())                       # today

# After:
trained, art = loop.run()                       # JaxModelOp     proposed (B1)
loaded = reg.load(art.trained_model_hash)       # with weights   proposed (B1, B2)
```

(r11-ws-c)=
## Workstream C · Loop components

**Decision.** Checkpoints, sweeps, validation and the SVI backend do what their names promise ([](#tbl-r11-move-c)).

:::{table} Workstream C's moves.
:label: tbl-r11-move-c

| Symbol | Destination | Reason |
|---|---|---|
| honour `save_dir`; save the final step | `Checkpoint`, `adapters/equinox.py:800-816` | stage 4: files go to `checkpoint_dir`; step 2,200 was not saved |
| `select_metric` read from validation | `HyperSweep` | trials are ranked on `final_metrics`, the last training batch |
| `ValidationStep` as the loop's evaluator | `TrainingLoop(validation=...)` | the class exists and the loop never calls it |
| `jax.jit(svi.update)` | `adapters/numpyro_svi.py` | 76 ms per step on five points; NumPyro's own `svi.run` jits |
:::

(r11-ws-d)=
## Workstream D · Amortized inference

**Decision.** pipekit-train documents one recipe for amortized inference and ships the smallest piece that every user rewrote, whereas the density estimators stay in the model libraries ([](#tbl-r11-move-d)).

:::{table} Workstream D's moves.
:label: tbl-r11-move-d

| Symbol | Destination | Reason |
|---|---|---|
| `invert=True` (yield `(output, params)`) | `SimulationDataset` | stage 6's `Flip` subclass is boilerplate every SBI user writes |
| `GaussianHead(mean, log_var)` with `log_prob` | a model library (pyrox), or `pipekit_train.recipes` if none takes it | `NLL` needs `.log_prob`; a model-definition helper in pipekit-train core would breach `boundaries.md` |
| the SBI recipe page | pipekit-train docs | simulator, flipped dataset, head, `NLL`, calibration check against an MCMC oracle (stages 6 and 7) |
:::

(r11-ws-e)=
## Workstream E · The Lightning and Keras backends

**Decision.** We build the two backends that ADR D5 already scheduled, on the same `run(loop) -> (op, backend_info)` seam as Equinox ([](#tbl-r11-move-e)).

:::{table} Workstream E's moves.
:label: tbl-r11-move-e

| Symbol | Destination | Reason |
|---|---|---|
| `adapters.lightning.run` with `_build_lightning_module` | pipekit-train, v0.2 | the scoping report's default backend; a scaffold today |
| `adapters.keras.run` | pipekit-train, v0.3 | multi-backend Keras 3; a scaffold today |
| a `ModelOp` for torch and Keras models | pipekit-array or the adapters | there is no pipekit-array `ModelOp` to return |
:::

(r11-gates)=
## Testing gates

Each workstream only merges when its gates pass, and the existing gates are checks that the code passes today ([](#tbl-r11-gates)).

:::{table} Gates for the proposals and for what ships.
:label: tbl-r11-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Catalog parity | A1 | a `CatalogDataset` over a 10-scene catalog with `RegularStride((256, 256))` yields the same chips, to 0, as running the sampler and `preprocess` by hand; its hash changes when any of query, sampler, preprocess, split or seed changes |
| A2 Trajectories | A2 | `SimulationDataset(cycle=Cycle(fwd, 24))[i]` equals 24 manual steps to 0; a prior that raises `TypeError` internally raises it |
| A3 Simulator hash | A3 | swapping $\mat{F}$ for $2\mat{F}$ changes the hash, and `CachedDataset` misses (today it serves the $\mat{F}$ cache) |
| A4 Cache speed | A4 | reading 16,000 cached items takes at most 2× the time to simulate them for stage 1's model (today about 160×) |
| A5 Window import | A5 | `XarrayWindowDataset` imports with the pinned geopatcher and no shim; stage 8's pairs satisfy $\vect{y} = \mat{F} \vect{x}_{-1}$ to 1e-5 |
| B1 Registry round trip | B1 | `Checkpoint(registry=reg)` writes weights, and `reg.load(hash)` reloads to $\max \vert \mat{W} - \mat{F} \vert = 3.8 \times 10^{-7}$, the trained value |
| B2 Model hash | B2 | `art.trained_model_hash` is set without a registry and changes when any weight changes |
| C1 Final checkpoint | C1 | an early stop at step 2,200 leaves `ckpt/2200` in `save_dir`, and a resume starts there |
| C2 Sweep selection | C2 | on a two-trial sweep whose training and validation rankings disagree, `HyperSweep` returns the validation winner |
| C3 Validation | C3 | the loop's `eval/*` metrics equal `ValidationStep`'s on the same model to 1e-12 |
| C4 SVI speed | C4 | stage 7's 4,000 SVI steps run in under 10 s (today 305 s) with the same final ELBO to 1e-6 |
| D1 Amortized recipe | D1 | the recipe reproduces stage 6 with `invert=True` and no `Flip`; $\mathbb{E}[r^2]$ is reported, and coverage of $\pm 2 \sigma_\alpha$ is in [0.90, 0.99] after the recipe's calibration step |
| E1, E2 Backends | E1, E2 | stage 2's white-prior emulator, as a torch or Keras linear layer, reaches $\max \vert \mat{W} - \mat{F} \vert < 10^{-5}$ through `backend="lightning"` and `"keras"` |
| T1 Emulator (existing) | — | stage 2: white-prior $\max \vert \mat{W} - \mat{F} \vert < 10^{-5}$ and stage 3's six-step cycle within $10^{-5}$. Passes today: $3.8 \times 10^{-7}$ and $1.3 \times 10^{-6}$ |
| T2 MCMC oracle (existing) | — | stage 7: NumPyro and BlackJAX NUTS posterior standard deviations within 5 % of the Laplace width. Passes today: 0.167 and 0.168 against 0.168 |
| T3 Window pairs (existing) | — | stage 8's targets equal $\mat{F}$ applied to the last input to 1e-5. Passes today, with the shim |
:::

(r11-risks)=
## Sequencing and risks

The fixes come first, because the running example hit them. The missing features follow, and the two new backends come last, as ADR D5 already orders them ([](#fig-r11-roadmap)).

:::{figure} ../images/reports/r11_roadmap.svg
:label: fig-r11-roadmap
:alt: Phase 1: A2, A3 and A5 dataset fixes, B1 the registry round trip and B2 the model hash, and C1 to C4 loop fixes (checkpoints, sweeps, ValidationStep in the loop, the SVI step), side by side, then their gates. Phase 2: A1 CatalogDataset, A4 cache speed and formats, D1 the amortized recipe, then their gates. Phase 3: E1 Lightning and E2 Keras, then their gates.

Fixes first, then the missing features, then the backends.
:::

- **CatalogDataset waits on packaging.** The `[catalog]` extra is held back until geocatalog is on PyPI. Fallback: we ship `CatalogDataset` without an extra and import geocatalog lazily with an install hint, as `XarrayWindowDataset` does for geopatcher.
- **A simulator hash cannot see inside a function.** Hashing the code of a `CallableForward` is fragile across Python versions. Fallback: a required `version` string for callables, with a warning when it is absent.
- **Changing the returned operator type breaks callers.** Scripts that use `trained.module` on an `EquinoxModelOp` must keep working. Fallback: we keep `EquinoxModelOp` and give it `serialize_weights` and `from_registry` instead of returning `JaxModelOp`.
- **Two more backends to maintain.** Lightning and Keras double the test matrix. Fallback: we mark both as experimental and gate them only on the emulator gate E1/E2, as the scoping report itself suggested.
- **Under-confident posteriors look safe.** The intervals of stage 6 cover 99.5 %, so they would pass a naive coverage check. Fallback: the recipe reports $\mathbb{E}[r^2]$ and compares against an MCMC oracle on a few scenes, as stage 7 does.

(r11-open-questions)=
## Open questions

1. **Where do distribution heads and density estimators live?** The design docs name pyrox for `ConditionalNormalizingFlow`, but it is not pinned here. Is a `GaussianHead` a pyrox object, or a small `pipekit_train.recipes` helper?
2. **Should a trained model's prior travel with it?** The emulator of stage 3 is only accurate inside its training prior. Should the artifact record the prior's hash, and should `NeuralForward` warn on inputs outside it?
3. **Seeds across datasets.** Two `SimulationDataset`s with seeds 0 and 1 share all but one sample seed. Should the seeds be hashed into the sample seed rather than added?
4. **One validation path or two?** The Equinox loop evaluates with the task's loss, whereas `ValidationStep` evaluates with arbitrary metrics. C3 makes the loop use `ValidationStep`, so is the task-loss path still needed?
5. **Sweeps in parallel.** `HyperSweep` runs the trials one after another, and parallel trials and Optuna or Ray Tune samplers are deferred. Is `pipekit.parallel.ProcessMap` enough for the first parallel version?
6. **Online training.** `OnlineTrainer` is deferred to v0.3 (boundaries Q6), and the scoping report ruled online training out. Is it still wanted?

(r11-todos)=
## Upcoming work

[](#tbl-r11-todos) lists the follow-ups.

:::{table} Follow-ups from checking the scoping report and the design docs against the code.
:label: tbl-r11-todos

| Follow-up | Where | Why |
|---|---|---|
| Rewrite the README and the `__init__` docstring for 0.0.2 | pipekit-train | they say "planning only" and "v0.0" |
| Update ADR D13, D14 and the adapter pages to "implemented" | `docs/design/decisions.md`, `adapters.md`, `blackjax_adapter.md` | the backends ship |
| Settle hyperparameter search's status in one place | `vision.md`, `boundaries.md` | out of scope in one, shipped in another |
| Describe the checkpoint format the code writes | D8, `adapters.md`, `primitives.md` | three descriptions, one implementation |
| Add `IterableDataset` and the sweep classes to the API index | `docs/design/api/README.md` | exported but undocumented |
| Give `numpyro-mcmc` final metrics | `adapters/numpyro_mcmc.py` | `final_metrics` is `{}`, so a sweep over it fails |
| Run the slow NumPyro test in CI with a timeout | pipekit-train tests | `test_same_task_runs_under_both_backends` ran past 500 s here |
:::

(r11-references)=
## References

### Sources

- The scoping report, "Report 11 — `pipekit-train`: training pipelines for emulators and amortized inference" (supplied by the author).
- pipekit-train 0.0.2 in `references/geostack/pipekit/packages/pipekit-train`, with its source, tests, README, CHANGELOG and `docs/design/` (vision, architecture, boundaries, decisions D1 to D14, API pages and examples).
- The running example's scripts and outputs, described in [](#r11-running).
