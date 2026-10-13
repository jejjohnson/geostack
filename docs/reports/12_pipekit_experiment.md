---
title: pipekit-experiment
subtitle: Report 12 · Experiment tracking, the model registry and reproducibility artifacts, as shipped and as still needed
short_title: 12 · pipekit-experiment
description: pipekit-experiment, the boundary between pipekit and external orchestration tools. The scoping report is checked against pipekit-experiment 0.0.2; its registry works, but it has no tracker, no MLflow or W&B code, and several round trips break, so every unshipped idea from the scoping report is kept as a gated proposal.
keywords:
  - pipekit-experiment
  - model registry
  - content addressing
  - experiment tracking
  - reproducibility
  - MLflow
  - Hydra
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

pipekit-experiment is where a pipekit model leaves the stack: it is logged
to a tracker, stored in a registry, promoted to production and pinned in a
reproducibility artifact. The scoping report asked for a thin adapter layer
over MLflow, W&B, DVC, Hydra and Metaflow, with a content-addressed registry
at its centre. Version 0.0.2 ships the registry, the two protocols, the run
records, the training and inference artifacts, and adapters for Hydra, DVC
and Metaflow. It does not ship a single tracker: there is no MLflow or W&B
code, although their extras exist. Following one emulator from training to
an inference artifact breaks in six places: `load` cannot rebuild a JAX
model, a model stored without weights hashes on its class name alone, tags
move without a trace, an artifact pins a tag instead of a hash, the training
artifact loses its tracker run id, and nested pipelines do not survive the
Hydra round trip. **The code takes precedence where it has decided; every
idea from the scoping report that has not shipped stays as a proposal with a
gate.**

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Partly built. Originally a scoping proposal ("committed"); pipekit-experiment 0.0.2 is the design of record where it has decided, and the scoping report's unshipped ideas are kept as proposals |
| **Reading time** | about 30 minutes |
| **Audience** | anyone storing, promoting or reloading trained models, logging runs to a tracker, or writing an adapter for an orchestration tool |
| **Companions** | [Report 11](11_pipekit_train.md) (the training loop that fills artifacts and stores models); [Report 10](10_pipekit_cycle.md) (the cycle a reloaded emulator drops into); [Report 9](09_use_cases.md) (use cases 5 and 9: the registry and the regulatory artifact); [Report 13](13_state_catalog.md) (the producer hash) |
| **Code checked against** | `pipekit-experiment` 0.0.2 at `ef3c17f`, with pipekit, pipekit-train and pipekit-jax from the same revision; hydra-core 1.3.7, hydra-zen 0.17, DVC and Metaflow 2.19 installed for the adapters. Its 52 tests pass. Read on 13 October 2026 |
| **Running example** | every output below is from one run of `notebooks/reports/report12_pipekit_experiment.py`, which builds its registries, DVC repository and Metaflow flows in a temporary directory and asserts each output the page shows |
:::

:::{important} How this page treats the scoping report
pipekit-experiment has no design docs of its own, so its code and README are
the only record of what was decided. Where they decided (the registry's
hash, the fsspec backend, write-once tags, JSON artifacts), a **Differs** box
says so and the code stands. Where the scoping report asked for something
that does not exist (trackers, the MLflow registry, `rerun`, `gc`, tag
history), it becomes a workstream rather than an open question.
:::

(r12-summary)=
## Summary

**Decision.** pipekit-experiment stays the thin boundary package the
scoping report and its README describe: protocols, a content-addressed
registry, JSON artifacts and one adapter per tool, with nothing from those
tools imported by pipekit pipelines. The registry's design stands. Five
workstreams finish the package.

- **A. The registry.** `load` cannot rebuild a model whose weights live in
  `weights.bin` (A1). A model stored without weights hashes on its class
  name, so two architectures collide (A2). Tags can move silently through
  `store(name=...)` and leave no history (A3). There is no `gc` or delete
  (A4); the fsspec backend needs parent directories and fails on an empty
  `list` (A5); and the scoping report's `MLflowModelRegistry` does not exist
  (A6).
- **B. Trackers.** There is no `ExperimentTracker` implementation at all.
  Add a file tracker for tests and offline runs (B1), then MLflow (B2) and
  W&B (B3), and fix the training artifact's `tracker_run_id`, which is
  `None` for the shipped `Run` and `repr(handle)` for anything else (B4).
- **C. Artifacts.** `TrainingArtifact.rerun` does not exist (C1);
  `InferenceArtifact` accepts a tag where it should pin a hash (C2); and the
  training artifact's `model_registry_uri` omits the registry it points at
  (C3).
- **D. Orchestration adapters.** Nested pipelines do not survive
  `HydraConfigLoader.to_yaml` and `from_yaml` (D1, Report 9's proposal A);
  `MetaflowStepAdapter.as_step` builds a step Metaflow rejects (D2); the DVC
  pointer is parsed line by line (D3).
- **E. Run records.** `Run`, `RunMetrics` and `RunArtifacts` exist, but
  nothing creates them, `RunMetrics` has no `to_dataframe`, and
  `RunArtifacts.attach` takes only a URI string (E1).

**End state.** A trained model is stored with its weights under a hash that
identifies its architecture and weights, reloaded by hash or by tag in one
call, promoted with a recorded tag move, and pinned by hash in an inference
artifact. Its training run is logged to any tracker that satisfies the
protocol, and its training artifact names the run, the data, the registry
and enough of the loop to retrain.

(r12-ownership)=
## Ownership rules

pipekit-experiment owns the records a run leaves behind and the registry
models live in; the external tools own scheduling, dashboards and storage.

:::{figure} ../images/reports/r12_ownership.svg
:label: fig-r12-ownership
:alt: External tools at the top: Hydra, DVC and Metaflow have adapters, MLflow and W&B do not. pipekit-experiment holds the protocols, run records, the local and fsspec registries, the training and inference artifacts and three adapters; there is no tracker implementation. pipekit-train logs and stores into it; inference pipelines and pipekit-cycle load from it.

pipekit-experiment 0.0.2 between pipekit and the tools it adapts. Dashed
chips are missing.
:::

1. **pipekit-experiment** owns the `ExperimentTracker` and `ModelRegistry`
   protocols, the run records, the registries, the artifacts and the tool
   adapters. Its core depends only on `pipekit`; each tool arrives through
   its own extra.
2. **pipekit-train** owns the callbacks that talk to it (`LogToExperiment`,
   `Checkpoint(registry=...)`) and fills `TrainingArtifact`. Today it also
   defines a fallback, `_LocalArtifact` (`loop.py:174`), which
   `_build_artifact` returns when pipekit-experiment is not installed
   (`loop.py:427-433`): the same fields, but no `save`, `load` or
   `reload_model`. The rule is that this fallback is the only second type,
   is never persisted, and is documented as such; removing it (making
   `[experiment]` a hard dependency of `run()`) is [an open question](#r12-open-questions).
3. **Model operators** (pipekit-jax's `JaxModelOp`, pipekit-train's
   `EquinoxModelOp`) own their own weight format: `serialize_weights` and
   `from_registry`. The registry stores bytes and never interprets them.
4. **External tools** (MLflow, W&B, DVC, Hydra, Metaflow) own scheduling,
   dashboards, data storage and configuration. No pipekit pipeline imports
   one.
5. **Serving** (FastAPI, LitServe) owns reloading a model on a worker; it
   reads the registry by tag or hash and nothing else.

:::{table} The dependency tiers around pipekit-experiment.
:label: tbl-r12-tiers

| Tier | Packages | May import | Role |
|---|---|---|---|
| External | MLflow, W&B, DVC, Hydra, Metaflow, serving | anything | scheduling, dashboards, storage |
| Boundary | pipekit-experiment | pipekit; its extras: hydra-core, hydra-zen, dvc, metaflow, fsspec, s3fs; mlflow and wandb (unused) | protocols, registry, artifacts, adapters |
| Users | pipekit-train, pipekit-cycle, inference scripts | pipekit-experiment | log, store, load |
| Framework | pipekit, pipekit-jax | | operators, JAX operators |
:::

:::{warning} Differs from the plan
The scoping report's extras were `mlflow`, `wandb`, `dvc`, `hydra`,
`metaflow`, `s3`, `mlflow-registry`, `train` and `all`. 0.0.2 has `mlflow`,
`wandb`, `dvc`, `hydra`, `metaflow` and `s3`. The `mlflow` and `wandb` extras
install packages that no code in pipekit-experiment uses, because neither
adapter exists. The README lists them as planned for "v0.0.2", the version
the package already has.
:::

(r12-surface)=
## What ships

:::{table} pipekit-experiment 0.0.2's public surface. Rows in italics are in the scoping report but not in the code.
:label: tbl-r12-surface

| Group | Names | Module | State |
|---|---|---|---|
| Protocols | `ExperimentTracker`, `ModelRegistry` | `protocols` | shipped; `runtime_checkable`, method names only |
| Run records | `Run`, `RunMetrics`, `RunArtifacts`, `RunStatus` | `run` | shipped; nothing creates them |
| Registries | `LocalModelRegistry(root)`, `S3ModelRegistry(uri, storage_options)` | `registry` | shipped |
| *Registries* | *`MLflowModelRegistry`* | | *not built* |
| Artifacts | `TrainingArtifact`, `InferenceArtifact` | `artifacts` | shipped; JSON `save`/`load`, `reload_model(s)` |
| Adapters | `HydraConfigLoader`, `DVCDatasetVersioning`, `MetaflowStepAdapter` | `adapters.hydra`, `.dvc`, `.metaflow` | shipped |
| *Adapters* | *`MLflowTracker`, `WandbTracker`* | | *not built* |
:::

(r12-running)=
## The running example: one emulator from training to an artifact

The stages follow the linear emulator of [Report 11](11_pipekit_train.md),
a learned $W \in \mathbb{R}^{40 \times 40}$ that replaces Report 10's
transport operator $M$, through everything pipekit-experiment does to a
model after training. For readable outputs some stages use a two-parameter
operator `Scale(factor, offset)`, $x \mapsto \text{factor} \cdot x + \text{offset}$,
alongside it.

(r12-stage-1)=
### Stage 1 · Store the emulator under its content hash (pipekit-jax → pipekit-experiment)

**TL;DR.** Store the trained emulator and check that its address changes
exactly when the model does.

**Problem.** The registry addresses a model by

$$
h = \mathrm{SHA256}\left( \mathrm{json}(\text{state}) \Vert \texttt{0x00} \Vert w \right) ,
$$ (r12-eq-hash)

where state is the operator's class and config and $w$ its serialised
weights (empty if none). Two stores of the same model must give the same
$h$; changing the config or a weight must change it.

```{code-block} python
:caption: Storing a JaxModelOp with its weights, from a real run.
:linenos:

# Shapes: W (40, 40), b (40,) float32 → weights.bin 6,816 bytes
import pipekit_experiment as px
from pipekit_jax import JaxModelOp

reg = px.LocalModelRegistry("registry")                                # today
op = JaxModelOp(eqx.nn.Linear(40, 40, key=key))      # W and b      today
h = reg.store(op, name="lin40", weights=op.serialize_weights())       # today
# registry/<h>/operator.json  {"class": "JaxModelOp",
#                              "config": {"module_class": "Linear"}, …}
# registry/<h>/weights.bin    6,816 bytes
# registry/<h>/metadata.json  {"tags": {}}
# registry/_tags/lin40        <h>
# same op twice → same h; one config value or one weight changed → new h
reg.store(JaxModelOp(eqx.nn.Linear(40, 40, key=k1))) == \
    reg.store(JaxModelOp(eqx.nn.Linear(3, 7, key=k2)))   # → True: no weights
```

The address is exactly as the code defines it, and it behaves. Writes are
atomic on the local backend (a `.tmp` file, then `os.replace`), and refs are
checked as a single path segment, so `reg.load("../x")` raises. One case
fails: without weights, a `JaxModelOp`'s state is just
`{"module_class": "Linear"}`, so a 40 × 40 and a 3 × 7 layer share a hash,
and the second store is merged into the first one's entry.

:::{warning} Differs from the plan
The scoping report hashed `hash(training_config, dataset_content, weights)`.
The code hashes the operator's state and its weights, nothing else: the
address identifies the **model**, and its provenance lives in the
`TrainingArtifact`. This page keeps the code's choice, because a model
retrained from different data to the same weights is the same model, and
records the link from model to training in workstream C. The layout differs
too: `operator.json` rather than `operator.yaml`, `weights.bin` rather than
`model.bin`, `_tags/<name>` without `.txt`, and no `preview.json`.
:::

(r12-stage-2)=
### Stage 2 · Load it back (pipekit-experiment → pipekit-cycle)

**TL;DR.** Reload the emulator by hash and by name, with its weights, and
drop it into a cycle.

**Problem.** A round trip must be exact:
$\max \lvert W_{\text{loaded}} - W \rvert = 0$.

```{code-block} python
:caption: The registry's load path, and the skeleton path that works.
:linenos:

# Shapes: W (40, 40)
reg.load(h)
# → RuntimeError: from_state cannot reconstruct JaxModelOp: it is marked
#   forbid_in_yaml … Rebuild it in code instead.                      today
skeleton = JaxModelOp(eqx.nn.Linear(40, 40, use_bias=False, key=key0))
loaded = skeleton.from_registry(reg, "lin40")             # by tag     today
# → max |W_loaded − W| = 0.0;  a (20, 40) skeleton raises a shape error
emulator = pc.NeuralForward(loaded, dt=1.0)               # Report 10  today
```

`ModelRegistry.load(ref)` rebuilds an operator with `Operator.from_state`
and never touches `weights.bin`. That works for operators whose config is
plain values (`Scale`), and fails for every model with weights, because
`JaxModelOp` and `EquinoxModelOp` are `forbid_in_yaml`. The path that works
is the model's own: build a skeleton of the right shape, then
`from_registry(reg, ref)` reads the weights into it. Workstream A1 makes
`load` do this.

:::{warning} Differs from the plan
The scoping report's `load` was
`Operator.from_state(config).with_weights(weights)`. There is no
`with_weights` in pipekit; weights come back through
`ModelRegistry.load_weights(ref)` and the model class's `from_registry`.
:::

(r12-stage-3)=
### Stage 3 · Promote it (pipekit-experiment)

**TL;DR.** Mark the emulator as production, then promote a candidate, and
see what the registry remembers.

**Problem.** Deployments resolve the tag `production` to a hash. A
promotion must be deliberate, must not be silent, and must leave a record
of what `production` meant before.

```{code-block} python
:caption: Tag-based promotion, from a real run.
:linenos:

reg.tag(h3, "production")                                            # today
reg.tag(h4, "candidate")                                             # today
reg.tag(h4, "production")
# → FileExistsError: Tag 'production' already exists. Pass force=True …
reg.tag(h4, "production", force=True)       # moves the tag             today
reg.resolve_tag("production")               # → h4; h3 is mentioned nowhere
reg.store(op5, name="production")           # moves it again, no force   today
reg.list(tags={"v": "1"})                   # → []: the stored tag is the int 1
```

Write-once tags are the code's default, as the scoping report asked, and
`force=True` is the explicit override. Two things undo that. `store(...,
name=...)` moves a tag without `force`, so a script can repoint
`production` by storing a model under that name. And no tag move is
recorded: after a promotion, nothing says what `production` used to be.
`list(tags=...)` compares with `==`, so a string `"1"` does not match a
stored integer `1`. Workstream A3 fixes the first two.

(r12-stage-4)=
### Stage 4 · Pin it in an inference artifact (pipekit-experiment)

**TL;DR.** Freeze an inference pipeline that uses the emulator, so it can be
rebuilt with exactly this model later.

**Problem.** An inference artifact maps each model in a pipeline to the
hash it used: $\{\text{operator path} \mapsto h\}$. Rebuilding must load the
same $h$, whatever the tags say by then.

```{code-block} python
:caption: An InferenceArtifact, and what it accepts as a pin.
:linenos:

art = px.InferenceArtifact(pipeline_yaml=..., backend_info={}, deps_lock=...,
                           pinned_model_hashes={"steps.1": "production"})  # today
art.save("forecast.inference.json"); px.InferenceArtifact.load(...)   # equal
art.reload_models(reg)
# → {"steps.1": Scale(factor=3.0)}: "production" resolved to whatever it is now
# with a JaxModelOp pinned → RuntimeError (from_state, as in stage 2)
```

`pinned_model_hashes` takes any ref, and `reload_models` resolves tags, so an
artifact pinned to `"production"` drifts the next time the tag moves: the
opposite of a pin. It also inherits stage 2's failure for models with
weights. Workstream C2 accepts only hashes.

:::{warning} Differs from the plan
The scoping report made `InferenceArtifact` a subclass of
`pipekit.repro.Artifact`, the regulatory snapshot of use case 9. There is no
`pipekit.repro` module and no `Artifact` class in pipekit core;
`InferenceArtifact` and `TrainingArtifact` are dataclasses over a shared JSON
base (`_JsonArtifact`). [Report 9](09_use_cases.md#r9-case-9)'s use case 9
builds its artifact from these.
:::

(r12-stage-5)=
### Stage 5 · Log the training run, and read its artifact (pipekit-train → pipekit-experiment)

**TL;DR.** Train the emulator with two trackers attached, and check what the
training artifact records.

**Problem.** One training run should reach several trackers with identical
metric streams, and its artifact should name the tracker run, the data, the
stored model and the registry it lives in.

```{code-block} python
:caption: Two trackers on one loop, and the artifact the loop returns, from a real run.
:linenos:

# Shapes: 200 pairs; 50 steps
class MemTracker:          # satisfies px.ExperimentTracker: 4 methods     user
    def start_run(self, name, config): ...
    def log_metrics(self, run, metrics, step): ...
    def log_artifact(self, run, path, name): ...
    def end_run(self, run, status): ...

a, b = MemTracker(), MemTracker()
loop = pt.TrainingLoop(..., callbacks=(pt.LogToExperiment(a),
                                       pt.LogToExperiment(b),
                                       pt.Checkpoint(registry=reg)))      # today
trained, art = loop.run()                       # art: px.TrainingArtifact
# a and b: identical mse streams, 50 points, 3.28 → 0.0029; both "completed"
art.tracker_run_id        # → None
art.model_registry_uri    # → "registry://d120fe9b…": no registry root
reg.load_weights(art.trained_model_hash)        # → None: no weights stored
pk.loads(art.training_pipeline_yaml)
# → RuntimeError: from_state cannot reconstruct TrainingLoop … forbid_in_yaml
```

Stacking trackers works as the scoping report promised: each
`LogToExperiment` gets the same stream. But there is no tracker to stack;
every one is user code. The artifact the loop fills has four problems, and
only one is pipekit-experiment's own:

- **`tracker_run_id` is never the run id.** `_build_artifact`
  (`loop.py:414-419`) keeps a callback's `run` only if it has
  `__getitem__`, and then stores `repr(run)`. The shipped `Run` is a
  dataclass without `__getitem__`, so the field is `None`, as here; a
  tracker that returns a mapping or a string gets its whole `repr`
  (`"{'name': 'emu', 'id': 'run-0'}"` in Report 11's run), never the id
  alone (B4).
- **`model_registry_uri` is `registry://<hash>`.** It does not say which
  registry, and the registries give `Checkpoint` nothing to say it with:
  `LocalModelRegistry` exposes `root`, `S3ModelRegistry` exposes `uri`, and
  only the private `_where()` is common to both (C3).
- **The weights are not stored.** The Equinox adapter re-wraps the trained
  module as `EquinoxModelOp`, which has no `serialize_weights`; this is
  [Report 11](11_pipekit_train.md#r11-ws-b)'s workstream B1.
- **The pipeline YAML cannot be loaded.** `TrainingLoop` is
  `forbid_in_yaml`, so the artifact cannot retrain (C1).

:::{warning} Differs from the plan
The scoping report's `TrainingArtifact` had `save`, `load`, `rerun` and
`reload_model`. The shipped one has `save`, `load` (JSON) and
`reload_model(registry)`, which calls `registry.load` and so fails as in
stage 2. There is no `rerun`.
:::

(r12-stage-6)=
### Stage 6 · Hand the pipeline to orchestration tools (pipekit-experiment → Hydra, DVC, Metaflow)

**TL;DR.** Express the pipeline as Hydra YAML, version its data with DVC
and run it as a Metaflow step.

**Problem.** Each adapter must round-trip what it is given: a pipeline to
YAML and back, a file to a DVC hash, a pipeline to a runnable step.

```{code-block} python
:caption: The three adapters that ship, from real runs.
:linenos:

from pipekit_experiment.adapters.hydra import HydraConfigLoader as HL
HL.from_yaml(HL.to_yaml(Scale(2.0, 1.0)))(3.0)          # → 7.0          today
seq = pk.Sequential([Scale(2.0, 1.0), Scale(0.5, 0.0)])
HL.from_yaml(HL.to_yaml(seq))
# → InstantiationException: Sequential.operators[0] must be an Operator,
#   got DictConfig  (to_yaml wrote {class: Scale, config: …}, no _target_)

dvc = DVCDatasetVersioning(repo="dvcrepo")
dvc.add("data/train.csv"); dvc.hash("data/train.csv")   # → e5ebd4c0…    today
dvc.push("data/train.csv")      # → CalledProcessError: no remote

MetaflowStepAdapter.as_step(seq, name="scaled", inputs=["x"])
# → Metaflow: "Step scaled is missing a self.next() transition"
MetaflowStepAdapter.run(seq, self, name="scaled", inputs=["x"])  # inside a
# hand-written @step → 3.5                                              today
```

Flat operators round-trip through Hydra; nested ones do not, because
`to_yaml` writes pipekit's debug config instead of `_target_` entries, which
is Report 9's proposal A seen from this side (D1). The DVC adapter works and
shells out to the `dvc` CLI; its `hash` reads the first `md5:` or `hash:`
line of the pointer file, so a pointer with `hash: md5` first would return
the string `"md5"` (D3). `as_step` builds a step without `self.next`, which
Metaflow refuses in any position but the end, while `run` inside a
hand-written step works (D2).

The fsspec registry was checked the same way: `S3ModelRegistry("memory://")`
gives the same hashes and layout as the local one, but `list()` on an empty
registry raises `FileNotFoundError`, a `file://` registry fails on its first
store unless `storage_options={"auto_mkdir": True}`, and its writes are not
atomic (A5). Real S3 was not tested.

(r12-names)=
## The scoping report's names, checked

:::{table} The stack names the scoping report's examples use, at the pinned revisions.
:label: tbl-r12-names

| Name in the report | At the pinned revisions |
|---|---|
| `pe.S3ModelRegistry(bucket=..., prefix=...)` | `S3ModelRegistry(uri, storage_options=None)`, over any fsspec URI |
| `pe.adapters.mlflow.MLflowTracker`, `pe.adapters.wandb.WandbTracker`, `MLflowModelRegistry` | do not exist |
| `pe.adapters.hydra.HydraConfigLoader.from_hydra_cfg` | exists, with `to_hydra_cfg`, `to_yaml` and `from_yaml` |
| `pe.adapters.metaflow.MetaflowStepAdapter.run(loop)` | `run(op, namespace, *, name, inputs)` and `as_step(op, *, name, inputs, decorate=True)` |
| `pe.TrainingArtifact(...).save`, `.load`, `.rerun()`, `.reload_model()` | `save`, `load` and `reload_model(registry)` exist; `rerun` does not |
| `pk.repro.Artifact` | does not exist |
| `registry.store(op, name=..., tags=...)`, `registry.load("methane_emulator_v3")`, `registry.tag(h, "production")` | exist; `store` also takes `weights=`; `load` fails for models with weights; `tag` raises unless `force=True` |
| `registry.gc()` | does not exist |
| `loop.to_yaml()`, `pt.Checkpoint(every_n_epochs=10)`, `n_epochs=100` | `TrainingLoop` has no `to_yaml` and is `forbid_in_yaml`; `Checkpoint` and the loop count steps ([Report 11](11_pipekit_train.md)) |
| `pe` as the import alias | these reports use `px`, because `pe` is pipekit-evaluate elsewhere |
:::

(r12-audit)=
## Current-state audit

Of sixteen pieces checked, eight ship and work, six ship with defects, and
two are missing. The audit read `references/pipekit` at `ef3c17f` on
13 October 2026.

:::{table} pipekit-experiment 0.0.2, piece by piece.
:label: tbl-r12-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| `ExperimentTracker` | `protocols.py:24` | `LogToExperiment` | **Healthy.** Matches the plan; the runtime check sees names only |
| `ModelRegistry` | `protocols.py:50` | `Checkpoint`, artifacts | **Healthy.** Adds `weights=`, `load_weights`, `force=` |
| tracker implementations | | | **Missing.** None; the `mlflow` and `wandb` extras are empty |
| `Run`, `RunMetrics`, `RunArtifacts` | `run.py:21`, `:60`, `:83` | | **Split ownership.** Nothing creates them; no `to_dataframe`; `attach` takes a string |
| content hash | `registry.py:46` | every store | **Split ownership.** Exact with weights; class-name only without them |
| `LocalModelRegistry(root)` | `registry.py:245` | stages 1–5 | **Healthy.** Atomic writes, path-safe refs |
| `S3ModelRegistry(uri, storage_options)` | `registry.py:297` | stage 6 | **Split ownership.** Works on `memory://`; empty `list` raises; `file://` needs `auto_mkdir`; writes not atomic |
| `load(ref)` | `registry.py` (`_resolve` at `:235`) | `reload_model(s)` | **Split ownership.** `from_state` only; fails for models with weights |
| `tag`, `resolve_tag` | `registry.py:200`, `:223` | stage 3 | **Split ownership.** Write-once with `force`; `store(name=)` bypasses it; no history |
| `gc`, delete, untag, `MLflowModelRegistry` | | | **Missing.** |
| `TrainingArtifact` | `artifacts.py:48` | `TrainingLoop.run` | **Healthy** as a record; no `rerun` (C1) |
| `InferenceArtifact` | `artifacts.py:88` | use case 9 | **Split ownership.** Pins accept tags |
| `HydraConfigLoader` | `adapters/hydra.py` | Report 9 | **Healthy** for flat operators; nested fails (D1) |
| `DVCDatasetVersioning` | `adapters/dvc.py` | | **Healthy.** CLI wrapper; line-scanned pointer hash |
| `MetaflowStepAdapter` | `adapters/metaflow.py:328`, `:359` | | **Healthy** for `run`; `as_step` builds an invalid step |
| docs | `README.md`, `CHANGELOG.md` | | **Healthy** with two stale claims: "shipped (v0.0.1)", and promotion "atomic" on every backend |
:::

Smaller findings:

- **Concurrent writers can race.** `store` reads and rewrites
  `metadata.json` with no lock, and `tag` checks that a tag is free before
  writing it, so two processes can both pass the check.
- **Re-storing merges tags.** Storing the same model again with new tags
  merges them into `metadata.json` rather than replacing them.
- **`b""` and `None` weights hash the same**, which is the intended reading
  of "no weights".
- **`LogToExperiment` never calls `log_artifact`**, so a tracker never sees
  the stored model or the artifact file.
- **A directory's DVC hash carries a `.dir` suffix** (`1df8f132….dir`),
  which a caller comparing against a file hash must expect.

(r12-ws-a)=
## Workstream A · The registry

**Decision.** The registry's content hash and storage layout stand; what
changes is that a stored model comes back whole, its address cannot collide,
and every tag move is deliberate and recorded.

:::{table} Workstream A's moves.
:label: tbl-r12-move-a

| Symbol | Destination | Reason |
|---|---|---|
| weight rebinding in `load(ref, like=None)` | `_ModelRegistryBase.load` | with `like` (a skeleton), call `like.from_registry(self, ref)`; without it, keep today's `Operator.from_state` path, which already rebuilds serializable operators such as `Scale`; only when that path cannot rebuild the stored model (it is `forbid_in_yaml`, or `weights.bin` exists and the class has no way to rebind it) raise a message that names `like=` instead of `from_state`'s |
| architecture in the hash for weightless stores | `JaxModelOp.state`, `EquinoxModelOp.state`, or a `store` check | `Linear(40, 40)` and `Linear(3, 7)` share a hash without weights |
| `store(name=...)` honours write-once; `_tags/_history.jsonl` | `registry.py` | a tag moves only with `tag(..., force=True)`, and every move appends (time, tag, old, new) |
| `gc(keep=...)`, `delete(hash)`, `untag(name)` | `registry.py` | the scoping report's operational policies (§7.2) |
| `auto_mkdir`, empty `list`, write-then-rename where the backend supports it | `S3ModelRegistry` | stage 6's three failures |
| `MLflowModelRegistry` | `adapters/mlflow.py` | names map to MLflow model names, hashes to versions, as the scoping report proposed |
:::

```{code-block} python
:caption: Stages 2 and 3, before and after workstream A.
:linenos:

# Shapes: W (40, 40)
# Today:
loaded = skeleton.from_registry(reg, "lin40")            # works       today
reg.load("lin40")                                        # RuntimeError
reg.store(op5, name="production")                        # silent move today

# After:
loaded = reg.load("lin40", like=skeleton)                # proposed (A1)
reg.store(op5, name="production")        # → FileExistsError  proposed (A3)
reg.tag(h5, "production", force=True)    # → appends to _history  proposed (A3)
reg.tag_history("production")            # → [(t₁, None, h3), (t₂, h3, h4), …]
```

(r12-ws-b)=
## Workstream B · Trackers

**Decision.** Ship trackers, starting with one that needs no service, so
that `LogToExperiment` has something to log to out of the box.

:::{table} Workstream B's moves.
:label: tbl-r12-move-b

| Symbol | Destination | Reason |
|---|---|---|
| `JSONLTracker(root)` | `pipekit_experiment.trackers` | an offline, testable tracker: one directory per run with `config.json`, `metrics.jsonl`, `artifacts.json`; it creates `Run` and `RunMetrics` |
| `MLflowTracker(tracking_uri, experiment_name)` | `adapters/mlflow.py` | the scoping report's first adapter; the extra already exists |
| `WandbTracker(project, entity=None)` | `adapters/wandb.py` | the second adapter; the extra already exists |
| `_run_id(handle)`, used for `tracker_run_id` | pipekit-train `loop.py:414-419` | normalize each return type the `ExperimentTracker` protocol allows: a `Run` gives `run.run_id`; a mapping gives `handle["run_id"]` (or `"id"`); a `str` is the id itself; anything else gives `None` with a warning. Today the check is `hasattr(run, "__getitem__")` and the value `repr(run)`, so a `Run` gives `None` and a mapping or string gives its `repr` |
| `log_artifact` on train end | pipekit-train `LogToExperiment` | trackers never see the stored model or the artifact |
:::

(r12-ws-c)=
## Workstream C · Artifacts

**Decision.** An artifact pins by hash, names its registry, and keeps
enough of its training loop to run it again.

:::{table} Workstream C's moves.
:label: tbl-r12-move-c

| Symbol | Destination | Reason |
|---|---|---|
| `TrainingArtifact.rerun(build_loop)` | `artifacts.py` | `TrainingLoop` is `forbid_in_yaml`, so rerun needs a builder: a script entry point or a Hydra config (D1) that rebuilds the loop, checked against `dataset_hash` |
| hash-only pins | `InferenceArtifact.__post_init__`, `reload_models` | stage 4: a tag drifts |
| `ModelRegistry.location` | `protocols.py` and both backends in `registry.py` | a public URI for the registry: `LocalModelRegistry` returns `self.root.resolve().as_uri()` (`file:///…`), `S3ModelRegistry` its `uri`; both already compute it privately as `_where()` (`:293`, `:366`). `LocalModelRegistry` has `root` and no `uri`, so the address cannot be built from `uri` |
| `model_registry_uri = f"{registry.location.rstrip('/')}/{hash}"` | pipekit-train `callbacks.py:266` | `registry://<hash>` does not say which registry |
| `training_artifact_hash` in the model's metadata | `Checkpoint` → `store(tags=...)` | links the content-addressed model back to its provenance (stage 1's box) |
:::

(r12-ws-d)=
## Workstream D · Orchestration adapters

**Decision.** Each adapter round-trips what it is given; the three that ship
get their gaps closed before more are added.

:::{table} Workstream D's moves.
:label: tbl-r12-move-d

| Symbol | Destination | Reason |
|---|---|---|
| nested `_target_` emission in `to_yaml` | `adapters/hydra.py`, with pipekit core's `get_config` | Report 9's proposal A; nested pipelines fail today |
| `self.next(...)` in generated steps | `MetaflowStepAdapter.as_step(..., next=...)` | Metaflow rejects the generated step |
| YAML parse of `.dvc` pointers | `DVCDatasetVersioning.hash` | the line scan depends on key order |
:::

(r12-ws-e)=
## Workstream E · Run records

**Decision.** The run records are produced by the trackers (B1) and given
the conveniences the scoping report specified.

:::{table} Workstream E's moves.
:label: tbl-r12-move-e

| Symbol | Destination | Reason |
|---|---|---|
| `RunMetrics.to_dataframe()` | `run.py` | the scoping report's analysis path; pandas as an optional import |
| `RunArtifacts.attach(name, source: str \| Path \| Operator)` | `run.py` | a `Path` is stored as a URI; an `Operator` is stored in a registry and attached by hash |
:::

(r12-gates)=
## Testing gates

Each workstream merges only when its gates pass; the existing gates are
checks the code passes today.

:::{table} Gates for the proposals and for what ships.
:label: tbl-r12-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Load with weights | A1 | `reg.load("lin40", like=skeleton)` returns weights equal to the stored ones to 0; `reg.load(h_scale)` still rebuilds `Scale(2.0, 1.0)` with no `like`, and `InferenceArtifact.reload_models` and `TrainingArtifact.reload_model` keep working for it; without `like`, loading a `JaxModelOp` raises an error that names `like=` |
| A2 No collision | A2 | weightless stores of `Linear(40, 40)` and `Linear(3, 7)` get different hashes |
| A3 Recorded promotion | A3 | `store(name="production")` raises when the tag exists; after two forced moves, `tag_history` returns three entries in order |
| A4 Garbage collection | A4 | `gc(keep={"production"})` removes every untagged hash and nothing a tag points to |
| A5 fsspec parity | A5 | on `memory://` and `file://` with default options, empty `list()` returns `[]` and stage 1's store and load give the local registry's hashes |
| A6 MLflow registry | A6 | against a local MLflow file store, store, tag and `load(like=)` round-trip stage 1's emulator to 0 |
| B1 File tracker | B1 | two `JSONLTracker`s on stage 5's loop write identical 50-line metric files, and `isinstance(t, ExperimentTracker)` holds |
| B2, B3 Service trackers | B2, B3 | against a local MLflow file store (B2) and W&B offline mode (B3), stage 5's 50 metric points are retrievable by run id |
| B4 Run id | B4 | `art.tracker_run_id` equals the id for each return type: a `Run` (`run.run_id`), a mapping (`["run_id"]`), a `str`; an unknown handle gives `None` and one warning |
| C1 Retrain | C1 | `art.rerun(build_loop)` on Report 11's white-prior emulator gives weights within 1e-6 of the stored ones on CPU, and the same `dataset_hash` |
| C2 Hash pins | C2 | an `InferenceArtifact` with a tag in `pinned_model_hashes` raises at construction |
| C3 Registry URI | C3 | `art.model_registry_uri` equals `f"{reg.location}/{hash}"` for both `LocalModelRegistry` (a `file://` URI) and `S3ModelRegistry("memory://…")` |
| D1 Nested Hydra | D1 | `from_yaml(to_yaml(seq))(x) == seq(x)` for stage 6's `Sequential`, and the configs are equal |
| D2 Metaflow step | D2 | a two-step flow built with `as_step(..., next="end")` runs and returns 3.5 |
| D3 DVC pointer | D3 | `hash` returns the md5 for pointers with either key order |
| E1 Run records | E1 | `RunMetrics.to_dataframe()` has one row per (name, step) with B1's values |
| R1 Content address (existing) | — | stage 1: same model twice gives the same hash; one changed config value or weight gives a new one. Passes today |
| R2 Skeleton reload (existing) | — | stage 2: `from_registry` reloads to 0 by hash and by tag. Passes today |
| R3 Stacked trackers (existing) | — | stage 5: two `LogToExperiment` callbacks receive identical streams. Passes today |
:::

(r12-risks)=
## Sequencing and risks

The registry and artifact fixes come first, because every model that goes
through the registry today comes back without its weights or loses its
pin; trackers come second, round trips last.

:::{figure} ../images/reports/r12_roadmap.svg
:label: fig-r12-roadmap
:alt: Phase 1: A1, A2 and A3 registry fixes and B4, C2 and C3 artifact fixes, then their gates. Phase 2: B1, B2 and B3 trackers, and A4, A5 and A6 registry additions, then their gates. Phase 3: C1 retraining, D1 to D3 adapter round trips, and E1 run records, then their gates.

Correctness first, then trackers, then round trips.
:::

- **Write-once `store(name=)` breaks scripts.** Scripts that store under
  `production` to promote would start raising. Fallback: warn for one
  release, then raise, and document `tag(..., force=True)` as the
  promotion call.
- **Two writers on a shared registry.** Neither tags nor metadata are
  locked, and S3 has no rename. Fallback: conditional writes where the
  backend has them (S3 `If-None-Match`), a lock file elsewhere, and a
  documented single-writer rule until then.
- **Retraining needs more than a config.** A `TrainingLoop` holds Python
  objects (datasets, callbacks) that YAML cannot carry. Fallback: `rerun`
  takes a builder function or a Hydra config rather than parsing
  `training_pipeline_yaml`, and checks the rebuilt `dataset_hash`.
- **Tracker APIs churn.** MLflow and W&B change often. Fallback: pin
  versions in the extras and run B2 and B3 against local, offline modes in
  CI.
- **`like=` asks the caller for the architecture.** A user loading by tag
  may not know the shapes. Fallback: store the module's constructor
  arguments in `operator.json` when the model class can provide them, so
  `load` can build the skeleton itself.

(r12-open-questions)=
## Open questions

1. **Should the registry hash include provenance?** The code identifies the
   model by state and weights; the scoping report wanted training config and
   data too. This page keeps the code's choice and links provenance through
   metadata (C). Is a second, provenance-inclusive hash needed for
   regulatory use?
2. **Where does a `like` skeleton come from at serving time?** A server that
   loads `production` needs the architecture. Is it in the deployment
   config, in `operator.json`, or in a model card stored beside the weights?
3. **Does the stack need `pipekit.repro.Artifact`?** Two JSON dataclasses
   cover training and inference. Is a core base class still wanted for
   other snapshots (use case 9's regulatory artifact)?
4. **Do tags carry stages?** The scoping report used `candidate` and
   `production` as plain tags. Should stages be a first-class field with
   allowed transitions, as MLflow has?
5. **Should pipekit-train keep `_LocalArtifact`?** It lets `run()` work
   without pipekit-experiment, at the cost of a second artifact type that
   cannot be saved or reloaded. The alternative is to require the
   `[experiment]` extra for `run()` and return only `TrainingArtifact`.
6. **A preview with each model.** The scoping report's `preview.json` held
   example inputs and outputs for sanity checks. Is it worth adding, and
   does it belong in the registry or the model card?

(r12-todos)=
## Upcoming work

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r12-todos

| Follow-up | Where | Why |
|---|---|---|
| Update the README's version table and the "atomic" promotion claim | pipekit-experiment `README.md` | it says "shipped (v0.0.1)" and promotion is atomic only locally |
| Drop or fill the `mlflow` and `wandb` extras | `pyproject.toml` | they install packages nothing uses |
| Add a real-DVC and a real-Metaflow test | pipekit-experiment tests | today's tests mock both tools |
| Add a `JaxModelOp` registry round-trip test | pipekit-experiment tests | no test reloads a model with weights |
| Compare tag values as strings in `list(tags=...)`, or document typed equality | `registry.py` | `{"v": "1"}` does not match a stored `1` |
:::

(r12-references)=
## References

<!-- TODO(ref): placeholders. When a report page lands, point its links in this
file at the page and delete its entry below. -->

:::{note} Companion reports
:class: simple
Reports [9](09_use_cases.md), [10](10_pipekit_cycle.md),
[11](11_pipekit_train.md), [13](13_state_catalog.md) and
[14](14_pipekit_evaluate.md) (pipekit-evaluate) are published.
:::

### Sources

- The scoping report, "Report 12 — `pipekit-experiment`: experiment
  tracking and model registry" (supplied by the author).
- pipekit-experiment 0.0.2 in `references/pipekit/packages/pipekit-experiment`:
  its source, tests, README, CHANGELOG and `pyproject.toml`.
- The running example's scripts and outputs, described in
  [](#r12-running).
