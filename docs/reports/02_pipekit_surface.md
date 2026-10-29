---
title: What pipekit ships
subtitle: Report 2 · pipekit's surface, now documented in pipekit; the planning stays here
short_title: 2 · pipekit surface
description: The tour of pipekit's shipped surface moved to pipekit's own docs. This page points there and keeps the planning (the deferred operators, the open questions and the upcoming work).
keywords:
  - pipekit
  - operators
  - composition
  - parallelism
  - serialisation
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

This report began as the proposal for pipekit's first public surface and became a record of what pipekit 0.0.2 ships. The record now lives with the code, as pipekit's [tour of the core](https://jejjohnson.github.io/pipekit/core-tour/), which goes through every module group by group, with examples checked against a real run and a list of today's limits. However, some of the content is GeoStack planning rather than pipekit documentation, so we keep it here: where the deferred operators went, the open questions and the upcoming work.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | The surface moved to [pipekit's docs](https://jejjohnson.github.io/pipekit/core-tour/); the planning stays here |
| **Audience** | anyone deciding what belongs in pipekit core, or what to build next |
| **Companions** | [Report 1](01_toolz_primer.md) (background, now in pipekit's docs); [Report 3](03_sister_libraries.md) (sister libraries); [Report 9](09_use_cases.md) (use cases, distilled into pipekit's [patterns](https://jejjohnson.github.io/pipekit/patterns/)); [Report 10](10_pipekit_cycle.md) (cycle); [Report 11](11_pipekit_train.md) (train); [Report 12](12_pipekit_experiment.md) (experiment); [Report 13](13_state_catalog.md) (state catalog) |
| **Checked against** | pipekit 0.0.2 (`ef3c17f`, the revision pinned in `references/geostack/pipekit`), and pipekit main when the tour moved |
:::

(r2-layout)=
## Where each group went

The proposal organised pipekit by **semantic group**, one module each, and that organisation shipped ({numref}`fig-r2-groups`). For each group below we link its section in the tour and keep the one difference from the plan that matters to a user, whereas [the full inventory](#r2-inventory) has the symbols and line counts.

```{figure} ../images/reports/r2_groups.svg
:label: fig-r2-groups
:alt: Fifteen cards, one per pipekit module. Composition, building blocks, combination, caching and quality control shipped as planned. Foundations, control flow, observers, shape inference, parallelism, serialisation and state differ from the plan. Protocols and hashing are new. Group K is reserved and ships nothing.
:width: 100%

**pipekit 0.0.2 by group, coloured by how each matches the plan.**
```

(r2-group-a)=
## Group A · Foundations

This group is now [Foundations](https://jejjohnson.github.io/pipekit/core-tour/#foundations) in the tour, and it holds `Operator`, `ConfigMixin`, `Sequential`, `Graph`, `Input` and `Node`. It differs from the plan in that there is no `Graph.summary()` and no signatures through `Graph`.

(r2-group-b)=
## Group B · Composition

This group is now [Composition](https://jejjohnson.github.io/pipekit/core-tour/#composition) in the tour, and it holds `pipe`, `compose`, `compose_left`, `juxt` and `complement`. All of them shipped as planned.

(r2-group-c)=
## Group C · Building blocks

This group is now [Building blocks](https://jejjohnson.github.io/pipekit/core-tour/#building-blocks) in the tour, and it holds `Identity`, `Const`, `Lambda` and `Sink`. All of them shipped as planned.

(r2-group-d)=
## Group D · Control flow

This group is now [Control flow](https://jejjohnson.github.io/pipekit/core-tour/#control-flow) in the tour, and it holds `Branch`, `Switch`, `Try`, `Coalesce` and `Retry`. It differs from the plan in that `Try` and `Retry` are not flagged `forbid_in_yaml`.

(r2-group-e)=
## Group E · Observers

This group is now [Observers](https://jejjohnson.github.io/pipekit/core-tour/#observers) in the tour, and it holds `Tap`, `Snapshot`, `ShapeTrace`, `Profile` and `Histogram`. It differs from the plan in that `ShapeTrace` has no hook for an extra line.

(r2-group-f)=
## Group F · Combination

This group is now [Combination](https://jejjohnson.github.io/pipekit/core-tour/#combination) in the tour, and it holds `Fanout`. The group shipped as planned, with nothing to add.

(r2-group-g)=
## Group G · Caching

This group is now [Caching](https://jejjohnson.github.io/pipekit/core-tour/#caching) in the tour, and it holds `Cache` and `Memoize`. Both operators shipped as planned, with no differences worth noting.

(r2-group-h)=
## Group H · Quality control

This group is now [Quality control](https://jejjohnson.github.io/pipekit/core-tour/#quality-control) in the tour, and it holds `Quarantine`, four `Assert*` operators and `QCError`. It shipped as planned, plus `QCError`.

(r2-group-i)=
## Group I · Shape inference

This group is now [Shape inference](https://jejjohnson.github.io/pipekit/core-tour/#shape-inference) in the tour, and it holds `Signature` and `compute_output_signature`. It differs from the plan in that an unknown dimension prints `?` instead of raising, and inference stops at `Sequential`.

(r2-group-j)=
## Group J · Parallelism

This group is now [Parallelism](https://jejjohnson.github.io/pipekit/core-tour/#parallelism) in the tour, and it holds `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap` and `check_pickleable`. It differs from the plan in that `BatchedMap` batches lists, not array axis 0.

(r2-pickle)=
### The pickleability discipline

Every operator in a process-pool pipeline must pickle, whereas closures do not. We describe the discipline (`forbid_in_yaml`, `check_pickleable`) in the tour's [Parallelism](https://jejjohnson.github.io/pipekit/core-tour/#parallelism) section.

(r2-group-k)=
## Group K · Where the deferred operators went

The proposal used Group K for operators that are carrier-specific and so do not belong in core. pipekit 0.0.2 reuses the letter (its docstring reserves Group K for a disk-cache backend), but nothing ships under it. We still track the deferred list, because most of it has not landed anywhere, and {numref}`tbl-r2-deferred` checks each operator against the capability index of every upstream library.

:::{table} The proposal's deferred operators, and where each one is at pipekit 0.0.2, geotoolz 0.10.2 and xrtoolz 0.0.5.
:label: tbl-r2-deferred

| Operator | Why not core | Planned home | Status |
|---|---|---|---|
| `ModelOp` | NumPy-specific (`np.asarray`, `np.concatenate`) | pipekit-array, or geotoolz / xr_toolz | in **geotoolz**; pipekit-jax adds `JaxModelOp` |
| `ApplyToBands` | NumPy (`np.take`, `np.stack`) | pipekit-array | not shipped |
| `Subsample` | NumPy slicing | pipekit-array | not shipped |
| `Diff` | NumPy reductions | pipekit-array | not shipped |
| `AssertValueRange`, `AssertNoNaN`, `AssertValidFraction` | reductions on the array | pipekit-array, or geotoolz / xr_toolz | not shipped |
| `Mode`, `ModeGated`, `pipeline_mode` | global mode state (`contextvars`) | deferred: open design questions | not shipped, as planned |
| `Provenance`, `Watermark` | carrier-specific metadata | geotoolz / xr_toolz | not shipped |
| `Augment` | `xr.merge` | xr_toolz | in **xrtoolz**, as planned |
| `ApplyToEach` | `xr.merge` | xr_toolz | in **xrtoolz**, as planned |
| `Spy`, `Hook` | cross-cutting hooks: open design questions | deferred; the hook stub is reserved | not shipped, as planned |
:::

pipekit-array exists (0.0.2), but it ships only four names (`array_namespace`, `ConcatenateAlong`, `StackAlong`, `MeanScalar`), none of them from this list. However, it is the planned home for most of the list, which makes it the biggest gap between plan and code (see [Report 3](03_sister_libraries.md)).

(r2-group-l)=
## Group L · Serialisation

This group is now [Serialisation](https://jejjohnson.github.io/pipekit/core-tour/#serialisation) in the tour, and it holds `dumps`, `loads`, `loads_sandboxed`, `register`, `unregister` and `allowed`. It differs from the plan in that core has no YAML or Hydra extras and nested pipelines do not round-trip.

(r2-group-m)=
## Group M · State

This group is now [State](https://jejjohnson.github.io/pipekit/core-tour/#state) in the tour, and it holds `StatefulOperator` and `CarryState`. It differs from the plan in that `Graph` does not thread state.

(r2-group-n)=
## Group N · Protocols, new

This group is now [Protocols](https://jejjohnson.github.io/pipekit/core-tour/#protocols) in the tour, and it holds `Predictor` and `FittableTransformer`. These are structural seams that the plan did not have.

(r2-hashing)=
## Shared hashing (`hashing.py`), new

This module holds `stable_json`, `stable_repr` and `sha256_hex`, which pipekit-experiment's registry shares, and pipekit's [patterns](https://jejjohnson.github.io/pipekit/patterns/) page uses them for orchestrator task keys.

(r2-inventory)=
## The full inventory

[](#tbl-r2-inventory) is the canonical list of what each group ships, with the proposed and shipped line counts side by side.

:::{table} The v0.1 inventory as proposed, against pipekit 0.0.2. The proposal's line counts are estimates; the shipped counts include docstrings.
:label: tbl-r2-inventory

| Group | Module | Shipped symbols | Proposed lines | Shipped lines | Matches the plan |
|---|---|---|---|---|---|
| A · Foundations | `_base/` | `Operator`, `ConfigMixin`, `Carrier`, `Sequential`, `Input`, `Node`, `Graph` | ~600 | 837 | differs: no `Graph.summary` or signatures |
| B · Composition | `compose.py` | `pipe`, `compose`, `compose_left`, `complement`, `juxt` | ~60 | 104 | yes |
| C · Building blocks | `blocks.py` | `Identity`, `Const`, `Lambda`, `Sink` | ~120 | 145 | yes |
| D · Control flow | `control.py` | `Branch`, `Switch`, `Try`, `Coalesce`, `Retry` | ~200 | 373 | differs: `Try`, `Retry` unflagged |
| E · Observers | `observe.py` | `Tap`, `Snapshot`, `ShapeTrace`, `Profile`, `Histogram` | ~250 | 348 | differs: no `ShapeTrace` hook |
| F · Combination | `combine.py` | `Fanout` | ~80 | 64 | yes |
| G · Caching | `cache.py` | `Cache`, `Memoize` | ~80 | 120 | yes |
| H · Quality control | `qc.py` | `Quarantine`, four `Assert*`, `QCError` | ~130 | 228 | yes, plus `QCError` |
| I · Shape inference | `signature.py` | `Signature`, `compute_output_signature` | ~150 | 105 | differs: `?` not raise |
| J · Parallelism | `parallel.py` | `ThreadMap`, `ProcessMap`, `AsyncMap`, `BatchedMap`, `check_pickleable` | ~250 | 286 | differs: list batching |
| K | — | nothing (reserved for a disk cache) | — | — | repurposed |
| L · Serialisation | `serial.py` | `dumps`, `loads`, `loads_sandboxed`, `register`, `unregister`, `allowed` | ~80 | 144 | differs: no extras |
| M · State | `state.py` | `StatefulOperator`, `CarryState` | ~150 | 123 | differs: no `Graph` support |
| N · Protocols | `protocols.py` | `Predictor`, `FittableTransformer` | — | 61 | new |
| Hashing | `hashing.py` | internal | — | 84 | new |
| **Total** | | **52 public names** | **~2,150** | **3,164** | |
:::

(r2-not-here)=
## What is intentionally not here

All of the proposal's exclusions held at 0.0.2:

- **Distributed execution** (Dask, Ray, Beam, Flink) is the orchestrator's job.
- **Automatic differentiation and JAX `jit`** as pipekit features live in a separate package, pipekit-jax. They do not clash with dual-mode `__call__`, as this page first said, because a tracer is not a `Node`, so a pipeline of JAX operators already runs under `jit`, `vmap` and `grad` with respect to its inputs ([Report 5](05_pipekit_jax.md#r5-stage-1)).
- **Type-checking operator chains at construction** is not done: `Sequential([op_a, op_b])` does not check that `op_a`'s output type matches `op_b`'s input. `Signature` checks shape, and type checking is a stretch goal.
- **Carrier-specific operators** are covered in [](#r2-group-k) and [Report 3](03_sister_libraries.md).
- **A GUI or pipeline builder** is a possibility, but not in scope.
- **Workflow scheduling** (Airflow, Prefect, Dagster) belongs in adapters in extras or downstream libraries, and pipekit-experiment's DVC, Hydra and Metaflow adapters are the first of these.

(r2-open-questions)=
## Open questions

The proposal carried seven scoping questions, each with a lean ([](#tbl-r2-questions)). The shipped code settles six of them and partly settles question 4, whereas its leans on questions 3 and 5 were overturned.

:::{table} The proposal's scoping questions, and how pipekit 0.0.2 answered them.
:label: tbl-r2-questions

| # | Question | The proposal's lean | At 0.0.2 |
|---|---|---|---|
| 1 | Name: `pipekit`, or `opkit`, `composable`, `pipekit-core`? | `pipekit` | **settled**: `pipekit` (the PyPI distribution name is revisited in [Report 16 §F](16_geo_data_stack.md#r16-ws-f)) |
| 2 | Is `ConfigMixin` always on? | yes, with `__config_mixin_auto__` and `__config_exclude__` | **settled** as leaned |
| 3 | How does `Signature` meet carriers without named dims? | `None` opts out; `summary` raises a clean error | **settled differently**: `None` opts out, `summary` prints `?` |
| 4 | How is YAML round-trip enforced? | the YAML loader rejects flagged operators; an optional `strict=True` dumper | **partly settled**: pipekit-experiment's `HydraConfigLoader.from_yaml` loads YAML, nested pipelines included; nothing dumps a nested pipeline, and `from_state` already rejects flagged operators |
| 5 | Hydra-zen built in, or an extra? | an extra, `pipekit[hydra]` | **settled differently**: an extra on pipekit-experiment, `pipekit-experiment[hydra]` (hydra-core and hydra-zen), not on pipekit |
| 6 | Which of `Coalesce`, `Retry`, `Mode`, `Provenance` to promote? | ship `Coalesce` and `Retry`; defer `Mode` and `Provenance` | **settled** as leaned |
| 7 | `AsyncMap` only, or `AsyncSequential` too? | `AsyncMap` only | **settled** as leaned |
:::

The differences recorded above raise six new questions that need a decision rather than a fix:

1. **Should `Graph` thread state?** Today `Sequential` does and `Graph` does not, so either `Graph` learns to or the docs say that stateful operators are `Sequential`-only. DA cycles with several inputs would want the former.
2. **Should `CarryState` be frozen?** If yes, it becomes a frozen dataclass and the hash is safe. If not, it should stop defining `__hash__`, since states routinely hold arrays.
3. **What dumps a nested pipeline?** Flat operators round-trip through `dumps` / `loads`, whereas `Sequential`, `Try`, `Cache` and the rest do not. Loading is answered by pipekit-experiment's `HydraConfigLoader`, which builds nested pipelines from hand-written YAML, so the open half is the other direction, a dumper that writes a built pipeline as nested `{"class", "config"}` payloads or `_target_` YAML.
4. **Should `AsyncMap.__call__` work inside a running loop?** At the moment a server handler must know to call `.run`, and an alternative is to detect a running loop and fail with a message that names `.run`.
5. **Should shape inference be wired through `Graph` and `Fanout`?** Only `Sequential` threads signatures today, so a `Graph` reports its input signature as its output.
6. **Should a `Graph`'s `state` describe the whole graph?** Today it records the inputs and the operators that produce the outputs, not the steps upstream of them, so two graphs that differ in an upstream step have equal `state` and hash the same. Anything that keys work by a pipeline's hash (the registry, orchestrator tasks, use cases 9, 12 and 15) needs the whole graph, which is also what a `Graph` dumper (question 3) would write.

(r2-todos)=
## Upcoming work

These are changes whose shape is already clear ([](#tbl-r2-todos)), and each is a candidate issue on pipekit or a sister library.

:::{table} Follow-ups from checking the proposal against pipekit 0.0.2.
:label: tbl-r2-todos

| Follow-up | Where | Why |
|---|---|---|
| Fix the `compute_output_signature` docstring: `summary` shows `?`, it does not raise | pipekit `_base/operator.py` | the docstring contradicts the behaviour |
| Fix the `dumps` docstring: a flagged operator fails with `RuntimeError`, not `TypeError` | pipekit `serial.py` | the same |
| Add `Graph.summary()` and signature propagation through `Graph` | pipekit `_base/graph.py` | proposed and not shipped |
| Make `Graph.state` record every node, not only the output operators | pipekit `_base/graph.py` | two graphs differing upstream hash the same (question 6) |
| Override `compute_output_signature` in `Fanout` and the parallel maps, returning `None` | pipekit `combine.py`, `parallel.py` | today they report a passthrough signature for a dict or a list |
| Route `ProcessMap(on_error="log_and_continue")` through `logging` | pipekit `parallel.py` | deployments cannot capture or silence `stderr` prints |
| Add a `ShapeTrace` hook for an extra line (the geotoolz CRS line) | pipekit `observe.py` | proposed and not shipped |
| Ship axis-0 batching, `ApplyToBands`, `Subsample`, `Diff` and the numeric asserts | pipekit-array | assigned there by the proposal; nothing has shipped |
| Decide where `Provenance` and `Watermark` live, and build them | geotoolz, xrtoolz | assigned there by the proposal; not shipped |
| Build `CatalogPipeline` on `ProcessMap` / `ThreadMap` | a sister library (geotoolz-catalog) | use cases 2 and 12 assume it |
:::

(r2-references)=
## References

### Citations

*To be added:* `toolz`, the Python Array API standard, and the use-case gallery, as entries in `references.bib`.
