---
title: "pipekit's skill shelf: operators, recipes and cycles, with no skill to start from"
subtitle: "Design · the skill tree for the library at the recipe rung"
short_title: Skill tree · pipekit
description: "pipekit's shelf of agent skills: 21 use skills over nine units across seven packages, two of them listed. pipekit is the first shelf at the recipe rung and at orchestration, the first written with no existing skill, and the one that shows a dependent can need a whole contract while using a single name. Measured at the submodule pins by scripts/skill_shelf.py."
date: 2026-11-12
keywords:
  - agents
  - skills
  - pipekit
  - recipes
  - role ladder
  - skill tree
numbering:
  heading_1: true
  heading_2: true
  table: true
---

(st-pk-decision)=
## Decision

Recommended: pipekit's shelf is 21 use skills over nine units; two skills are listed, `compose-and-run-pipelines` for the core package and `cycles-training-and-evaluation` for the packages built on it; and a dependent that subclasses `Operator` receives the operator card whatever its name count.
Alternative: one listed skill per package, seven in all, which matches how pipekit is installed and lists five skills most sessions never use.
Status: open, 12 Nov 2026.

What does not change: pipekit's packages and their public API.
pipekit has no use skill and no contributor skill today, so everything on this page is new.
This page follows the pattern of [gaussx's shelf](./skill_tree_gaussx.md) and says only where pipekit differs.

At the pin (`f2b3522`) the core package still imports as `pipekit`.
Its rename to `pipekit_ops` (jejjohnson/pipekit#232) changes the module names in the script's configuration, not the units.

(st-pk-motivation)=
## Motivation

pipekit is seven packages with seven different jobs, and nine packages in the stack build on it.

:::{table} pipekit's public names by unit, at the submodule pin, over its seven packages.
:label: tbl-pk-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Operators and composition | ⑤ ⑥ | geoscience, data | `pipekit.adapters` 9, `pipekit._base` 7, `pipekit.compose` 5, `pipekit.blocks` 4, `pipekit.protocols` 2, `pipekit.signature` 2, `pipekit.state` 2, `pipekit.testing` 2, `pipekit.combine` 1, `pipekit.imports` 1 | 35 |
| Control and execution | ⑥ | data | `pipekit.qc` 6, `pipekit.control` 5, `pipekit.observe` 5, `pipekit.parallel` 5, `pipekit.cache` 2 | 23 |
| Recipes as data | ⑥ | data | `pipekit.serial` 6, `pipekit.hashing` 4 | 10 |
| Named arrays | ⓪ ① ⑤ | geoscience, ML | `pipekit_array.dims` 12, `pipekit_array.dim_operators` 8, `pipekit_array.kernel` 6, `pipekit_array.einx` 5, `pipekit_array.testing` 5, `pipekit_array.batching` 4, `pipekit_array.rows` 3, `pipekit_array.xarray` 3, `pipekit_array.combinators` 2, `pipekit_array.dimarray` 2, `pipekit_array._namespace` 1, `pipekit_array.estimator` 1, `pipekit_array.reductions` 1 | 53 |
| Cycles | ⑥ | ML, geoscience | `pipekit_cycle.protocols` 14, `pipekit_cycle.adjoints` 6, `pipekit_cycle.cycle` 4, `pipekit_cycle.obs` 4, `pipekit_cycle.da` 3, `pipekit_cycle.forward` 3, `pipekit_cycle.state` 3, `pipekit_cycle.bfn` 1 | 38 |
| JAX bridge | ⑤ ⑥ | ML | `pipekit_jax.cycle` 3, `pipekit_jax` 2, `pipekit_jax.adjoints` 2, `pipekit_jax.diffrax_model` 1, `pipekit_jax.model_op` 1, `pipekit_jax.operator` 1 | 10 |
| Training | ⑥ | ML | `pipekit_train.loss` 5, `pipekit_train.callbacks` 4, `pipekit_train.dataset` 4, `pipekit_train.sweep` 4, `pipekit_train.loop` 3, `pipekit_train.writer` 2 | 22 |
| Evaluation | ⑥ | ML | `pipekit_evaluate.benchmark` 21, `pipekit_evaluate.taxonomy` 6, `pipekit_evaluate.report` 3, `pipekit_evaluate` 1, `pipekit_evaluate.scorers` 1 | 32 |
| Experiments and registries | orchestration | data | `pipekit_experiment.registry` 3, `pipekit_experiment.run` 3, `pipekit_experiment.artifacts` 2, `pipekit_experiment.protocols` 2 | 10 |
| **All** | | | | **233** |
<!-- shelf:units:end -->
:::

:::{table} Which part of pipekit each dependent package uses.
:label: tbl-pk-consumers

<!-- shelf:uses:begin -->
| Dependent package | Names used | Operators and composition | Control and execution | Recipes as data | Named arrays | Cycles | JAX bridge | Training | Evaluation | Experiments and registries |
|---|---|---|---|---|---|---|---|---|---|---|
| geotoolz-ops | 26 | 14 | 5 | 2 | 5 | 0 | 0 | 0 | 0 | 0 |
| geotoolz-array | 18 | 4 | 0 | 0 | 14 | 0 | 0 | 0 | 0 | 0 |
| xrtoolz-ops | 7 | 3 | 0 | 0 | 3 | 0 | 0 | 0 | 1 | 0 |
| vardax | 6 | 1 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | 0 |
| sesmax-core | 5 | 2 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 0 |
| geotoolz-patcher | 4 | 2 | 0 | 0 | 1 | 0 | 0 | 1 | 0 | 0 |
| somax | 3 | 2 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| geotoolz-products | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| plumax | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
<!-- shelf:uses:end -->
:::

The dependents split cleanly by package.
geotoolz-ops and geotoolz-array take operators and named arrays, vardax takes the cycle protocols, and sesmax-core takes the JAX bridge.
Training and evaluation are each touched by one package, for a single name: they are used from scripts.

(st-pk-shelf)=
## The shelf

Most of pipekit sits at ⑥, so its shelf is where the recipe and orchestration files of `role-ladder` are used most.

:::{table} pipekit's shelf. Each use skill is one file. pipekit has no contributor skills, so the extend rows are proposals.
:label: tbl-pk-shelf

| Role | Skill | Covers | Package |
| --- | --- | --- | --- |
| ⓪ structure | Named arrays | `DimArray`, its adapter, `Padded` batches, row layouts | `pipekit_array` |
| ⓪ structure | Signatures and adapters | `Signature`, `Carrier`, the named-array adapter registry | `pipekit` |
| ① primitive | Functions on dims | `sel`, `isel`, `reduce`, `concat`, `stack`, `transpose`, `map_over`, `take_along` | `pipekit_array` |
| ② strategy | Adjoints for a cycle | direct, backsolve, recursive-checkpoint, truncated and implicit adjoints | `pipekit_cycle`, `pipekit_jax` |
| ② strategy | Losses | `MSE`, `NLL`, `KL`, `Composite` | `pipekit_train` |
| ③ model | The cycle protocols | `ForwardModel`, `ObservationOperator`, `AnalysisStep` and the other contracts a model implements | `pipekit_cycle` |
| ⑤ step | Writing an Operator | `Operator`, its config, its signature, state, `Predictor` and `FittableTransformer` | `pipekit` |
| ⑤ step | Building blocks | `Identity`, `Const`, `Lambda`, `Sink`, `compose`, `pipe`, `juxt` | `pipekit` |
| ⑤ step | Operators on named arrays | `Sel`, `Reduce`, `MapOver`, `Transpose`, `KernelOp` | `pipekit_array` |
| ⑤ step | A JAX model as a step | `JaxOperator`, `JaxModelOp`, `JaxKernelOp`, `to_jax` | `pipekit_jax` |
| ⑤ step | *add-operator*, *add-adapter* | extend, proposed | |
| ⑥ recipe | Composing | `Sequential`, `Graph`, `Input`, `Node`, `Fanout` | `pipekit` |
| ⑥ recipe | Control flow | `Branch`, `Switch`, `Try`, `Retry`, `Coalesce` | `pipekit` |
| ⑥ recipe | Parallel maps and caching | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap`, `Cache`, `Memoize` | `pipekit` |
| ⑥ recipe | Checks and observation | the `Assert*` checks, `Quarantine`, `Tap`, `Snapshot`, `Profile`, `ShapeTrace` | `pipekit` |
| ⑥ recipe | Recipes as data | `dumps`, `loads`, `loads_sandboxed`, `register`, `pipeline_hash` | `pipekit` |
| ⑥ recipe | Cycles | `Cycle`, `WindowedCycle`, `EnsembleCycle`, `Recurrence` | `pipekit_cycle` |
| ⑥ recipe | Data-assimilation cycles | `DACycle`, `EnsembleDACycle`, `SmootherCycle`, the observation and forward wrappers | `pipekit_cycle` |
| ⑥ recipe | Scan cycles in JAX | `ScanCycle`, `EnsembleScanCycle`, `Rollout`, `DiffraxForwardModel` | `pipekit_jax` |
| ⑥ recipe | Training loops | `TrainingLoop`, datasets, callbacks, `HyperSweep` | `pipekit_train` |
| ⑥ recipe | Benchmarks and evaluation | `Benchmark`, `Scorer`, `Oracle`, `evaluate`, reports | `pipekit_evaluate` |
| ⑥ recipe | *add-cycle-template*, *add-scorer* | extend, proposed | |
| orchestration | Experiments and registries | `Run`, its artifacts and metrics, local and S3 model registries, trackers | `pipekit_experiment` |
:::

That is two structures, one primitive, two strategies, one model, four steps, ten recipes and one orchestration skill.

(st-pk-units)=
## Units and cards

:::{table} The cards each dependent package would receive from pipekit at a threshold of three names.
:label: tbl-pk-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
| geotoolz-ops | Operators and composition; Control and execution; Named arrays | Recipes as data (2) |
| geotoolz-array | Operators and composition; Named arrays |  |
| xrtoolz-ops | Operators and composition; Named arrays | Evaluation (1) |
| vardax | Cycles | Operators and composition (1) |
| sesmax-core | JAX bridge | Operators and composition (2) |
| geotoolz-patcher | none | Operators and composition (2); Named arrays (1); Training (1) |
| somax | none | Operators and composition (2); Cycles (1) |
| geotoolz-products | none | Operators and composition (1) |
| plumax | none | Operators and composition (1) |
<!-- shelf:cards:end -->
:::

Four packages receive no card, and that is wrong for all four.
geotoolz-patcher, somax, geotoolz-products and plumax each take one or two names from the operator unit, and the name is `Operator`: geotoolz-patcher, somax and plumax subclass it, and geotoolz-products imports it.
A package that subclasses a base class or implements a protocol needs the whole contract, however few names it imports.
So the threshold counts calls, and a subclass always earns the card.
The script does not detect subclassing yet; it is a follow-up below.

(st-pk-listed)=
## Listed skills

:::{table} The listed use skills. Line budgets are proposals.
:label: tbl-pk-skills

| Skill | Band | Units | Asked for as |
| --- | --- | --- | --- |
| `compose-and-run-pipelines`, the front door, at most 150 lines | geoscience, data | operators and composition; control and execution; recipes as data; named arrays | "chain these steps", "retry and cache this", "save the pipeline as JSON" |
| `cycles-training-and-evaluation`, at most 150 lines | ML | cycles; the JAX bridge; training; evaluation; experiments and registries | "run a data-assimilation cycle", "train this model", "benchmark these estimators" |
:::

(st-pk-below)=
## What pipekit needs from below

pipekit is at tier 0 of GeoStack and imports no other library of the stack.
Its floor is third party, and only for the packages built on the core.

| From | Surfaces to a caller | Card in |
| --- | --- | --- |
| equinox, jax | yes, in pipekit-jax: a `JaxOperator` is also a Module | the JAX bridge |
| diffrax | yes, in pipekit-jax: `DiffraxForwardModel` and the adjoint conversion | the JAX bridge; adjoints for a cycle |
| xarray | yes, in pipekit-array, as a named array with no CRS | named arrays |

(st-pk-differs)=
## Where pipekit bends the pattern

1. **A contract is not a name count.** Subclassing `Operator` is one import and the whole contract. The card rule needs "subclasses or implements" beside "uses at least three names".
2. **The shelf is mostly ⑥.** Ten of 21 entries are recipes. `role-ladder`'s recipe and orchestration files carry the general rules, and pipekit's entries say only what each combinator or template does.
3. **Nothing exists yet.** With no skill and no contributor skill to start from, the skeleton came from the packages and their subpackages alone, and the four extend skills are proposals.
4. **Units are packages.** Each package above the core is one unit, because each has one job and its own dependents.

(st-pk-gates)=
## Gates

Gates GX-0 to GX-6 of [gaussx's shelf](./skill_tree_gaussx.md#sg-gates) apply with pipekit's names, and one is added.

| Gate | Passes when |
| --- | --- |
| PK-1 Contracts | every package that subclasses `pipekit.Operator` or implements a `pipekit_cycle` protocol receives that unit's card |

(st-pk-risks)=
## Risks

- **The rename lands mid-way.** The pin predates `pipekit_ops`. Fallback: the units are named by job, so only the module patterns in `scripts/skill_shelf.py` change, and the test fails until they do.
- **Two listed skills are too few for seven packages.** Someone installing only pipekit-train gets a front door about the core. Fallback: list `cycles-training-and-evaluation` from the plugin of each package it covers.

(st-pk-followups)=
## Follow-up reports

- [ ] `feat(scripts): give a dependent that subclasses a unit's base class that unit's card` — gate PK-1

## Relationships

- Parent: jejjohnson/geostack#150
- Related: jejjohnson/geostack#151 (the design), jejjohnson/geostack#155 (pipekit is first in its order)
- Related: jejjohnson/pipekit#232 (the rename to pipekit-ops)
- Related: [skill tree](./skill_tree.md), [gaussx's shelf](./skill_tree_gaussx.md), [geotoolz's shelf](./skill_tree_geotoolz.md)
