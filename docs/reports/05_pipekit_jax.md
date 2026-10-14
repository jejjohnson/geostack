---
title: pipekit under JAX
subtitle: Report 5 · jit, vmap and grad through pipekit pipelines, what pipekit-jax ships, and what an operator-as-PyTree layer would add
short_title: 5 · pipekit under JAX
description: The scoping report on a JAX-traceable pipekit, checked against pipekit 0.0.2 and pipekit-jax 0.0.2. Its correction holds and goes further, since jit, vmap and grad with respect to inputs already run through plain pipekit pipelines. Gradients with respect to operator parameters do not, Cache returns wrong values under jit, and the shipped pipekit-jax is a model wrapper and a diffrax forward model rather than the planned operator layer.
keywords:
  - pipekit-jax
  - JAX
  - Equinox
  - jit
  - vmap
  - autodiff
  - differentiable retrieval
  - diffrax
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

The scoping report asked when pipekit would need a JAX-traceable twin, and
corrected an earlier claim that `jax.jit` cannot pass through pipekit's
dual-mode `__call__`. The correction is right, and the code goes further than
the report expected: a plain pipekit `Sequential`, `Graph` or `Fanout` of
JAX operators already runs under `jit`, `vmap` and `grad` with respect to its
inputs, with no new package. On the five-band methane retrieval of Reports
[10](10_pipekit_cycle.md#r10-running), [11](11_pipekit_train.md) and
[17](17_cycle.md#r17-running), the jitted pipeline matches the eager one exactly, `vmap`
retrieves 500 scenes in one call, and `grad` gives each scene an error bar
that is calibrated against the truth. Three things do not work. Gradients
with respect to **parameters held by operators** never reach them, because a
pipekit operator is one opaque leaf to JAX. Seven of pipekit's control,
observation and caching operators misbehave under tracing, and one of them,
`Cache`, **returns a wrong value without an error**. And pipekit-jax 0.0.2
is not the operator layer the report planned: it ships a model wrapper with
a weight round trip, a diffrax forward model and two adjoint helpers.
**The report's operator layer stays as gated proposals, and the core gets
trace guards first.**

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Partly built. Originally a future-direction analysis ("not for v0.1, not even v0.2"); pipekit-jax 0.0.2 shipped since, with a different scope. The report's correction and its "build on pipekit" recommendation stand; its operator layer is kept as proposals |
| **Reading time** | about 25 minutes |
| **Audience** | anyone who will jit, vmap or differentiate through a pipekit pipeline, and anyone deciding what a JAX layer over pipekit should contain |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit's surface, whose JAX exclusion this report corrects); [Report 3](03_sister_libraries.md) (the sister libraries); [Report 11](11_pipekit_train.md) (training the wrapped module); [Report 10](10_pipekit_cycle.md) (forward models and cycles); [Report 16](16_geo_data_stack.md) (one model-operator family); [Report 17](17_cycle.md) (the same sensor) |
| **Code checked against** | pipekit 0.0.2 at `ef3c17f` (pipekit, pipekit-cycle, pipekit-experiment, pipekit-jax), with JAX 0.11.2, Equinox 0.13.8, Optax 0.2.8 and diffrax 0.7.2. pipekit-jax's 36 tests pass. Read on 14 October 2026 |
| **Running example** | every output is from one run of `notebooks/reports/report05_pipekit_jax.py`, which draws the scenes from fixed seeds in float64 and asserts each number the page shows |
| **Sources** | the scoping report [@geostack_r5_draft]; JAX [@jax2018github], Equinox [@kidger2021equinox] and diffrax [@kidger2021neural] |
:::

:::{important} What changed from the scoping report
- **The correction stands, and goes further:** a plain `Sequential` traces under `jit`, `vmap` and `grad` ([stage 1](#r5-stage-1)).
- **The recommendation stands:** build on pipekit, do not fork it ([](#r5-summary)).
- **The shipped package differs:** `JaxModelOp` wraps an `eqx.Module`; no planned operator exists ([](#r5-ownership)).
- **Gradients over operator parameters** need PyTree operators ([stage 2](#r5-stage-2), [stage 4](#r5-stage-4)).
- **`Cache` returns a wrong value under jit**, not merely a redundant one ([stage 3](#r5-stage-3)).
- **Reports [2](02_pipekit_surface.md#r2-not-here) and [3](03_sister_libraries.md#r3-array)** repeated the corrected claim; they now link here ([](#r5-audit)).

A **Differs** box carries each argument ([conventions](../overview.md#sec-conventions)).
:::

(r5-summary)=
## Summary

**Decision.** JAX support stays in two places. **pipekit core** stays
JAX-free and carrier-agnostic, but refuses or flags the operators that are
wrong under tracing. **pipekit-jax** holds everything that imports JAX or
Equinox: today's model wrapper, forward model and adjoint helpers, and, when
a project needs gradients through operator parameters, the operator-as-PyTree
layer the scoping report designed. Five workstreams.

- **A. Trace guards in pipekit core.** `Cache` keys every tracer of one shape
  to the same entry, so a pipeline that caches twice returns the first
  result for the second call (A1). `Snapshot`, `Sink`, `Profile` and `Tap`
  run once, at trace time, on a tracer; `Snapshot` leaks it (A2).
- **B. Operators as PyTrees in pipekit-jax.** A `JaxOperator(Operator,
  eqx.Module)` mixin and a `JaxSequential` whose operators are a tuple field
  let gradients reach parameters and let `eqx.tree_at` replace a step (B1).
  Their configs hold arrays, which breaks `state` as JSON; the config needs
  an array policy (B2).
- **C. JAX-native control and observation.** A `CondBranch` and `CondSwitch`
  on `jax.lax.cond` and `jax.lax.switch` (C1), and a `DebugTap` on
  `jax.debug.callback`, which runs on every call rather than once per trace
  (C2).
- **D. Transformation wrappers.** `VMapOver`, `Checkpoint` and
  `ScanSequential` wait for a project that needs them; `truncated_scan`
  already covers the cycle layer's scans.
- **E. Docs.** One page on build time against trace time (E1), and fixes to
  pipekit-jax's README and install notes (E2).

**End state.** A pipeline of JAX operators runs under `jit`, `vmap` and
`grad` as it does today. An operator that cannot be traced says so instead
of returning a wrong value, a pipeline whose parameters are learned is built
from PyTree operators, and its trained weights round-trip through the
registry the way `JaxModelOp`'s do.

(r5-ownership)=
## Ownership rules

pipekit core owns composition and the operators that misbehave under
tracing; pipekit-jax owns everything that imports JAX
([](#fig-r5-ownership)).

:::{figure} ../images/reports/r5_ownership.svg
:label: fig-r5-ownership
:alt: At the top, pipekit pipelines of JAX operators (Sequential, Graph, Fanout) run under jit, vmap and grad over inputs today. Below, pipekit core holds Cache, Snapshot, Sink, Profile, Tap, Branch and Switch, which are wrong or trace-time only under jit, and would add trace guards. pipekit-jax 0.0.2 ships JaxModelOp, DiffraxForwardModel, to_diffrax_adjoint and truncated_scan, and would add JaxOperator, JaxSequential, CondBranch and DebugTap. pipekit-train and pipekit-cycle use it; JAX, Equinox, Optax and diffrax are external.

JAX support between pipekit core and pipekit-jax. Dashed chips are
proposed.
:::

1. **pipekit core** owns `Operator`, `Sequential`, `Graph`, `Fanout` and the
   control, observation and caching operators. It never imports JAX. Its
   guards (A) detect a tracer through `sys.modules`, only when JAX is
   already imported.
2. **pipekit-jax** owns every class that subclasses or wraps an
   `eqx.Module`, every operator built on a `jax.lax` primitive, and the
   JAX-side adjoint helpers. It depends on pipekit, pipekit-cycle (which
   [Report 16](16_geo_data_stack.md#r16-ws-b)'s B moves to the `[diffrax]`
   extra), Equinox and JAX.

Neighbours (pipekit-train, pipekit-cycle, the algorithm libraries) follow
[the stack rules](16_geo_data_stack.md#r16-ownership); for this page the
one that matters is that pipekit-train differentiates the module a
`JaxModelOp` wraps, never a pipekit `Sequential` (stage 2).

:::{warning} Differs from the plan
The scoping report's map put pipekit-jax beside geotoolz and xrtoolz at
v0.3 or later, with a `jax_geotoolz` domain layer under it. pipekit-jax
0.0.2 exists now, is a workspace member of pipekit, and has no domain layer
under it; nothing named `jax_geotoolz` exists. Its README describes it as
"JAX / Equinox carrier integration", the successor to pipekit-train's
`EquinoxModelOp`, not the operator framework the report described.
:::

(r5-surface)=
## What ships

[](#tbl-r5-surface) lists what pipekit-jax 0.0.2 exports beside the scoping
report's names; the italic rows were planned and never built.

:::{table} pipekit-jax 0.0.2's public surface. Rows in italics are in the scoping report but not in the code.
:label: tbl-r5-surface

| Group | Names | Module | State |
|---|---|---|---|
| Model wrapper | `JaxModelOp(module)`, `serialize_weights`, `with_weights`, `from_registry` | `model_op` | shipped; an `Operator` holding an `eqx.Module`, `forbid_in_yaml` |
| Forward model | `DiffraxForwardModel(vector_field, solver, dt0, dt, adjoint, …)`, `step`, `with_adjoint`, `as_dynamics` | `diffrax_model` | shipped behind `[diffrax]`; an `eqx.Module` satisfying `ForwardModel` and `SupportsAdjoint` |
| Adjoints | `to_diffrax_adjoint(spec)`, `truncated_scan(f, init, xs, *, k)` | `adjoints` | shipped |
| *Operators* | *`JaxOperator`, `JaxSequential`, `JaxGraph`* | | *not built* |
| *Control* | *`JaxBranch`, `JaxSwitch`* | | *not built* |
| *Observation* | *`JaxDebugPrint`* | | *not built* |
| *Transformations* | *`VMapOver`, `ScanSequential`, `Checkpoint`* | | *not built* |
:::

(r5-running)=
## The running example: one retrieval under jit, vmap and grad

The sensor is the five-band sensor of
[Report 17](17_cycle.md#r17-running), unchanged: $y_b = e^{-k_b \alpha} +
\varepsilon_b$ with $k = (0.002, \dots, 0.010)$, $\sigma = 10^{-3}$ and
$\alpha \sim \mathcal{U}(0, 300)$. The retrieval
inverts each band, $z_b = -\log y_b / k_b$, and averages the five with
inverse-variance weights. To first order
$\operatorname{var} z_b = \sigma^2 / (k_b y_b)^2$, which gives
{eq}`r5-eq-retrieval`:

$$
\hat\alpha = \frac{\sum_b w_b z_b}{\sum_b w_b} , \qquad
w_b = \left(k_b e^{-k_b z_b}\right)^2 .
$$ (r5-eq-retrieval)

The test set is 500 scenes; stage 2 adds 2,000 labelled scenes, as from
controlled releases.

(r5-stage-1)=
### Stage 1 · jit, vmap and grad through a pipeline (pipekit → JAX)

**TL;DR.** Run the retrieval as a plain pipekit `Sequential` under the three
transformations, and use the gradient as each scene's error bar.

**Problem.** A pipeline traces if every `_apply` is pure JAX and dispatch
does not intercept the tracer. `Operator.__call__` builds a graph node only
when an argument is a `Node` (`operator.py:160`), and a JAX tracer is not
one. The noise propagates through the retrieval as

$$
\sigma_{\hat\alpha} \approx \sigma \, \lVert \nabla_y \hat\alpha \rVert ,
$$ (r5-eq-sigma)

so if `grad` is right ({eq}`r5-eq-sigma`), the errors divided by $\sigma_{\hat\alpha}$ have an RMS
of 1.

```{code-block} python
:caption: The retrieval as a pipekit Sequential, under jit, vmap and grad, from a real run.
:linenos:

# Shapes: n = 500 scenes, B = 5 bands
# z_b = −log y_b / k_b
invert = Lambda(lambda y: -jnp.log(y) / K)              # (B,) → (B,)   today
# α̂ = Σ_b w_b z_b / Σ_b w_b,  w_b = (k_b e^{−k_b z_b})²
combine = Lambda(weighted_mean)                         # (B,) → ()     today
retrieval = Sequential([invert, combine])                               # today
jax.jit(retrieval)(y[0])          # → 215.44 (α = 215.29); = eager      today
jax.make_jaxpr(retrieval)(y[0])   # → one jaxpr, 12 equations           today
jax.vmap(retrieval)(y)            # (n, B) → (n,): RMSE 0.262           today
# σ_α̂ = σ ‖∇_y α̂‖
s = SIG * jnp.linalg.norm(jax.vmap(jax.grad(retrieval))(y), axis=1)  # today
# RMS of σ_α̂ 0.258; RMS of (α̂ − α) / σ_α̂ = 0.97
```

All three transformations pass through the pipeline unchanged. The jitted
output agrees with the eager one to 10⁻¹², because it is the same
arithmetic in one jaxpr. The gradient is right too: the linearised error bar
has an RMS of 0.258 against an actual RMSE of 0.262, and the standardised
errors have an RMS of 0.97, within the 0.03 that 500 scenes allow. None of
this needed pipekit-jax; it needs only operators whose `_apply` is pure JAX.
`Graph` with a diamond (one node feeding two) and `Fanout` behave the same
under `jit` and `vmap` (stage 3).

:::{warning} Differs from the plan
The scoping report expected `jit(seq)` to work but proposed a
`JaxSequential` with a tuple of operators "so it can be vmapped across".
`vmap` over the **data** needs nothing: `jax.vmap(retrieval)` works on the
plain `Sequential`. A tuple field matters only for vmapping or
differentiating over the **operators' parameters** (stages 2 and 4).
:::

(r5-stage-2)=
### Stage 2 · Gradients with respect to an operator's parameters (pipekit-jax → Optax)

**TL;DR.** Learn the absorption coefficients from labelled scenes, and see
that the gradient cannot reach them through the pipeline.

**Problem.** The coefficients are known only to 10 %, as from an old line
list. With 2,000 scenes whose $\alpha$ is known, fit $\log k$ by minimising

$$
L(\log k) = \frac{1}{n} \sum_i \left(\hat\alpha_{\log k}(y_i) - \alpha_i\right)^2 ,
$$ (r5-eq-loss)

where the inversion step holds $\log k$ as an `eqx.Module` wrapped in a
`JaxModelOp`. The gradient of {eq}`r5-eq-loss`,
$\partial L / \partial \log k$, must flow through the pipeline into the
operator.

```{code-block} python
:caption: Two ways to differentiate a pipeline's parameters, from a real run.
:linenos:

# Shapes: n = 2,000 labelled scenes, B = 5, log k (5,)
class BandInversion(eqx.Module):                                       # user
    log_k: jax.Array                    # z_b = −log y_b / exp(log k_b)
pipe = Sequential([JaxModelOp(BandInversion(log_k0)),
                   Lambda(weighted_mean)])                              # today
jax.tree.leaves(pipe)            # → [Sequential(...)]: one opaque leaf today
# ∂L/∂ pipe
eqx.filter_grad(loss)(pipe, y, a)          # → None: nothing to diff   today
# rebuild the pipeline inside the loss; differentiate the module instead
def loss_m(m, y, a):                                                    # user
    return loss(Sequential([JaxModelOp(m), Lambda(weighted_mean)]), y, a)
# log k ← Adam(log k, ∂L/∂ log k), 400 steps, learning rate 10⁻²
eqx.filter_value_and_grad(loss_m)(m, y, a)                              # today
# test RMSE: 3.45 at the start → 0.260 (true k: 0.262); k still ≤ 8.7 % off
```

The plain pipeline is opaque to JAX. `Sequential` and `JaxModelOp` are
ordinary Python objects, so `jax.tree.leaves` sees the whole pipeline as one
leaf, and Equinox's filtered gradient finds no array to differentiate. The
workaround is the one pipekit-train already uses: differentiate the module,
and rebuild the pipeline inside the loss. Then 400 Adam steps bring the test
RMSE from 3.45 to 0.260, as good as the true coefficients (0.262).

The coefficients themselves stay up to 8.7 % off. The labels constrain the
retrieval, not each band's absorption, and several sets of five $k_b$ give
the same $\hat\alpha$. A loss on $\hat\alpha$ is the right objective for the
retrieval; recovering the spectroscopy needs a loss on $y$.

:::{warning} Differs from the plan
The scoping report's headline use case (7.1, "differentiable retrievals")
assumed that `filter_grad` through a pipeline reaches the target spectrum,
the cloud threshold and the other parameters inside its operators. It does
not, for any shipped operator, because none of them is a PyTree. The report
treated PyTree registration as a detail of `JaxSequential`; it is the
feature, and it is workstream B.
:::

(r5-stage-3)=
### Stage 3 · Control, observation and caching operators under jit (pipekit)

**TL;DR.** Put each of pipekit's side-effecting and branching operators in a
jitted pipeline and record what it does.

**Problem.** Under `jit` an operator's `_apply` runs once, on a tracer, and
only the JAX operations it records survive into the compiled function. An
operator whose job is a side effect, a Python decision or a lookup keyed on
its input's value cannot do that job. [](#tbl-r5-under-jit) records what each one does.

:::{table} pipekit's operators inside a jitted pipeline, called three times (twice on one input, once on another of the same shape).
:label: tbl-r5-under-jit

| Operator | Module | Under `jit` | Verdict |
|---|---|---|---|
| `Tap(fn)` | `observe.py:40` | `fn` called once, with a `DynamicJaxprTracer` | trace time only |
| `ShapeTrace()` | `observe.py:131` | prints once | trace time only; what it is for |
| `Profile().wrap(op)` | `observe.py:185` | one timing, of the trace | trace time only |
| `Snapshot().at(key)` | `observe.py:78` | captures a tracer; using it raises `UnexpectedTracerError` | leaks a tracer |
| `Sink(write_fn)` | `blocks.py:105` | one write, of a tracer | trace time only |
| `Branch(predicate, …)` | `control.py:41` | `TracerBoolConversionError` | fails loudly |
| `Switch(key, cases)` | `control.py:92` | `ConcretizationTypeError` | fails loudly |
| `Cache(inner)` | `cache.py:36` | `x ↦ 10(x + 1) + 1` gives 2 for 21; a later trace raises `UnexpectedTracerError` | **wrong value** |
| `Fanout`, `Graph` (diamond) | `combine.py:21`, `_base/graph.py:97` | equal to eager; `vmap` over 500 scenes works | traces |
:::

```{code-block} python
:caption: Cache under jit, and the JAX-native replacements, from a real run.
:linenos:

# Shapes: x (5,) = (1, …, 5)
cache = Cache(Lambda(lambda x: x + 1))                                  # today
twice = Sequential([cache, Lambda(lambda x: 10 * x), cache])            # today
twice(x)                  # eager: (21, 31, 41, 51, 61)                 today
jax.jit(twice)(x)         # → (2, 3, 4, 5, 6): the first call's output  today
# key(x) = sha256(repr(x)) for a tracer: one key per shape and dtype
# replacement for Tap: runs on every call, not once per trace
jax.debug.callback(fn)    # 3 calls → 3 callbacks                       today
# replacement for Branch: both arms traced, one selected at run time
CondBranch(pred, retrieval, last_band)                  # lax.cond     proposed (C1)
# → equals the eager Branch to 10⁻¹²
```

Four operators are honest about it: `Branch` and `Switch` raise
(`TracerBoolConversionError` is a subclass of `ConcretizationTypeError`), and
`Fanout` and a diamond `Graph` simply trace. Four run at trace time only,
which is what the scoping report predicted; `ShapeTrace` is useful that way,
`Tap`, `Profile` and `Sink` are not, and `Snapshot` keeps a tracer that
raises when touched later.

`Cache` is the dangerous one. A tracer has no `tobytes`, so
`_input_key` (`cache.py:97`) falls back to `stable_repr`, and every tracer
of one shape and dtype has the same representation. The second `cache` call
in `twice` hits the first one's entry and returns $x + 1$ in place of
$10(x + 1) + 1$, with no error. The stored tracers then poison later
traces of other functions.

:::{warning} Differs from the plan
The scoping report said `Cache` and `Memoize` are "redundant under jit and
may actively hurt" and should be dropped from pipekit-jax. The finding is
stronger: they give wrong results today, in pipekit core, for anyone who
jits a pipeline containing them. Dropping them from a future package does
not protect today's users; workstream A1 makes `Cache` refuse a tracer.
`Memoize` is an alias of `Cache` (`cache.py:94`), not a second class.
:::

(r5-stage-4)=
### Stage 4 · Operators as PyTrees, prototyped (pipekit-jax)

**TL;DR.** Prototype the scoping report's `JaxOperator` and `JaxSequential`
in a few lines, and see what the multiple inheritance gives and costs.

**Problem.** For $\partial L / \partial \theta$ to reach parameters $\theta$
held by operators, each operator must be a PyTree whose array fields are
leaves, and so must the container.

```{code-block} python
:caption: The scoping report's PyTree operators, prototyped, from a real run.
:linenos:

# Shapes: B = 5; two operators, each holding log k (5,)
class JaxOperator(Operator, eqx.Module): pass                  # proposed (B1)
# MRO: Invert → JaxOperator → Operator → ConfigMixin → Module
class JaxSequential(Sequential, eqx.Module):                   # proposed (B1)
    operators: tuple
seq = JaxSequential([Invert(log_k0), Combine(log_k0)])         # proposed (B1)
jax.tree.leaves(seq)            # → 2 leaves, one per log k     proposed (B1)
# ∂L/∂θ for both operators; the tied gradient is their sum
eqx.filter_grad(loss)(seq, y, a)       # = stage 2's, to 10⁻⁹   proposed (B1)
eqx.tree_at(lambda s: s.operators[0], seq, Invert(log_k))      # proposed (B1)
seq.summary(None)               # → still a pipekit Sequential  proposed (B1)
json.dumps(Invert(log_k0).state)       # → TypeError: an Array  proposed (B2)
```

The multiple inheritance works as the scoping report said: Equinox's
`Module` registers the class as a PyTree and pipekit's `Operator` keeps
dispatch, `summary` and `|`. A single `JaxOperator` is differentiable; put
in a plain `Sequential` it is opaque again, so the container must be a
PyTree too. With both, the gradient reaches every operator, and
`eqx.tree_at` can replace one step. Because each operator holds its own
$\log k$, the two copies are separate leaves, and their gradients sum to the
gradient of stage 2's tied parameter. Tying them again is what
`eqx.nn.Shared` is for.

One cost the report did not mention. `ConfigMixin` derives `get_config` from
the `__init__` parameters, and Equinox's generated `__init__` takes the array
fields. So the config holds a JAX array, and `state` is no longer JSON. The
array values belong in the registry's weight blob, as `JaxModelOp` already
does it, and the config should record only their shapes and dtypes (B2).

(r5-stage-5)=
### Stage 5 · What pipekit-jax ships (pipekit-jax → pipekit-experiment, pipekit-cycle)

**TL;DR.** Round-trip the trained inversion through the registry, truncate a
scan's gradient, and solve an ODE forward model with three adjoints.

**Problem.** Three shipped pieces, each with a check. (a) A reloaded model
must equal the stored one byte for byte. (b) For a plume column refilled
each hour, $c_{t+1} = 0.9\, c_t + q_t$ over $T = 24$ hours, the exact
sensitivity of the last column to the first hour's source is
{eq}`r5-eq-scan`,

$$
\frac{\partial c_T}{\partial q_t} = 0.9^{\,T - 1 - t} ,
$$ (r5-eq-scan)

and truncating to the last $k$ steps must zero every earlier term. (c) For
first-order loss $\dot c = -\lambda c$, $\lambda = 0.25\ \text{h}^{-1}$, one
hour's solve and its gradient are both $e^{-\lambda} = 0.7788$.

```{code-block} python
:caption: pipekit-jax's three pieces on the running example, from a real run.
:linenos:

# Shapes: log k (5,); q (24,); c ()
# (a) the stage 2 module, through the registry and back
h = reg.store(JaxModelOp(m), weights=JaxModelOp(m).serialize_weights())  # today
JaxModelOp(BandInversion(jnp.zeros(5))).from_registry(reg, h)           # today
# → byte-identical weights; max |k_reloaded − k̂| = 0
# (b) ∂c_T/∂q_t through the carry, kept for the last k steps only
truncated_scan(decay, 0.0, q, k=3)     # → 3 nonzero gradients    today
truncated_scan(decay, 0.0, q, k=24)    # → 24; ∂c_T/∂q_0 = 0.9²³ = 0.0886  today
# forward values identical to lax.scan
# (c) ċ = −λc over one hour
ode = DiffraxForwardModel(vector_field=lambda t, c, a: -LAM * c,
                          dt=1.0, dt0=0.01)                             # today
ode.step(100.0, 1.0)                   # → 77.88, error < 10⁻¹²     today
ode.with_adjoint(pc.DirectAdjoint())   # also Backsolve; ∂c/∂c₀ = 0.7788  today
ode.with_adjoint(pc.TruncatedAdjoint(k=3)).step(1.0, 1.0)  # → ValueError  today
```

All three work as documented. `JaxModelOp` round-trips the weights exactly
through the registry, which is how stage 4's operators should store theirs.
`truncated_scan` keeps exactly the last $k$ gradients through the carry and
leaves the forward values alone. `DiffraxForwardModel` satisfies
pipekit-cycle's `ForwardModel` and `SupportsAdjoint`, and the
checkpointed, direct and backsolve adjoints give the same gradient on this
non-stiff problem. A `TruncatedAdjoint` spec raises, by design: truncation
is a property of a rollout, which `truncated_scan` handles, not of one solve.

:::{warning} Differs from the plan
The scoping report's `ScanSequential(op, n_steps)` and `Checkpoint(op)` are
not built. Their closest shipped relatives are `truncated_scan` (a scan with
a gradient window, for cycles) and the adjoint specs, which choose
checkpointing for an ODE solve. Neither wraps a pipekit operator.
:::

(r5-names)=
## The scoping report's names, checked

[](#tbl-r5-names) checks each name the scoping report's examples use.

:::{table} The names the scoping report's examples use, at the pinned revisions.
:label: tbl-r5-names

| Name in the report | At the pinned revisions |
|---|---|
| `Operator.__call__` checks `isinstance(a, (Node, Input))` | checks `isinstance(a, Node)`; `Input` subclasses `Node` (`operator.py:160`, `_base/graph.py:77`) |
| `JaxOperator`, `JaxSequential`, `JaxGraph` | do not exist; prototyped in stage 4 |
| `JaxBranch`, `JaxSwitch`, `JaxDebugPrint` | do not exist |
| `VMapOver`, `ScanSequential`, `Checkpoint` | do not exist; `truncated_scan` is the shipped scan |
| `Cache`, `Memoize` | exist; `Memoize = Cache` |
| `BatchedMap`, `ProcessMap`, `AsyncMap`, `ThreadMap` | exist, over Python iterables (`parallel.py`); `jax.vmap` is the batch path |
| `ModelOp` in pipekit | does not exist in pipekit core; `JaxModelOp` (pipekit-jax), `EquinoxModelOp` (pipekit-train) and `xrtoolz.inference.ModelOp` do |
| `jax_geotoolz` | does not exist |
| `eqx.nn.Sequential`, `eqx.nn.Shared`, `eqx.tree_at`, `filter_jit`, `filter_grad`, `filter_vmap`, `filter_pmap` | exist in Equinox 0.13.8 |
| "`eqx.Module` … the source is ~100 lines" | the module system is a package (`equinox/_module/`), not a 100-line file |
| "raw `jit` raises `TypeError` for any non-JAX leaf" | true when the module is an argument: `jax.jit(lambda m, x: m(x))(mlp, y)` raises; `eqx.filter_jit` does not |
:::

(r5-audit)=
## Current-state audit

Of twelve pieces checked, six ship and work, three ship with defects or run
at trace time only, one returns wrong values, and two are missing
([](#tbl-r5-audit)); [](#fig-r5-readiness) counts the page's calls by
marker. The audit read `references/pipekit` at `ef3c17f` on 14 October 2026.

:::{figure} ../images/reports/r5_readiness.svg
:label: fig-r5-readiness
:alt: A grid with one row per stage of the running example and per workstream, and one square per stack call in that row's sketches, coloured by its marker. Of 41 calls, 24 run today, 15 are proposed by this report's workstreams and 2 are code the reader writes.

What this page's sketches call: of 41 stack calls, 24 run today, 15 are
proposed by this report's workstreams and 2 are code the reader writes
([how to read it](../overview.md#sec-conventions)).
:::

:::{table} JAX support in the stack, piece by piece.
:label: tbl-r5-audit

| Symbols | Defined in | Used by | Verdict |
|---|---|---|---|
| dispatch under tracing | pipekit `_base/operator.py:141` | stage 1 | **Healthy.** A tracer is not a `Node`, so `_apply` runs |
| `Sequential`, `Graph`, `Fanout` | pipekit `_base/sequential.py:41`, `_base/graph.py:97`, `combine.py:21` | stages 1, 3 | **Healthy** under `jit`, `vmap` and `grad` over inputs; opaque to `grad` over parameters (`self.operators` is a list, `sequential.py:82`) |
| `Branch`, `Switch` | pipekit `control.py:41`, `:92` | stage 3 | **Healthy** as failures: JAX raises a clear error |
| `Tap`, `ShapeTrace`, `Profile`, `Sink` | pipekit `observe.py`, `blocks.py:105` | stage 3 | **Split ownership.** Trace time only, silently |
| `Snapshot` | pipekit `observe.py:99` | stage 3 | **Split ownership.** Stores a tracer that raises later |
| `Cache`, `Memoize` | pipekit `cache.py:36`, `:94`, key at `:97` | stage 3 | **Wrong value.** Every tracer of one shape has one key |
| `JaxModelOp` | pipekit-jax `model_op.py:42` | stages 2, 5; pipekit-train, pipekit-experiment | **Healthy.** Byte-identical round trip; an `Operator`, not a PyTree |
| `DiffraxForwardModel` | pipekit-jax `diffrax_model.py:30` | stage 5 | **Healthy.** `ForwardModel` and `SupportsAdjoint`; three adjoints agree |
| `truncated_scan`, `to_diffrax_adjoint` | pipekit-jax `adjoints.py:26`, `:84` | stage 5 | **Healthy.** |
| PyTree operators (`JaxOperator`, `JaxSequential`) | | stage 4 | **Missing.** The mixin works in a prototype; its config is not JSON |
| JAX-native control and observation | | stage 3 | **Missing.** `lax.cond` and `jax.debug.callback` work as operators in a few lines |
| docs | pipekit-jax `README.md`, pipekit `docs/installation.md:132`, `pyproject.toml` | | **Split ownership.** The README links a `docs/design/` that does not exist; the install notes list `orbax-checkpoint` as a dependency, which `pyproject.toml` does not declare; `jaxtyping` is declared and never imported |
:::

Smaller findings:

- **Reports 2 and 3 said a pipeline cannot be jitted.**
  [Report 2](02_pipekit_surface.md#r2-not-here)'s exclusions ("these clash with dual-mode `__call__`") and
  [Report 3](03_sister_libraries.md#r3-array)'s
  "`jax.jit` does not compose with a pipekit pipeline" repeated the claim the
  scoping report corrected; both now link to stage 1.
- **pipekit-array's namespace dispatch** already supports `jax.numpy` arrays
  (`_namespace.py`); a pipekit-array operator in a jitted pipeline is pure
  JAX when its input is a JAX array. Not run here.
- **pipekit-jax depends on pipekit-cycle in its base install**, only for
  `diffrax_model.py`'s default adjoint spec. [Report 16](16_geo_data_stack.md#r16-ws-b)'s
  B moves it to the `[diffrax]` extra.

(r5-ws-a)=
## Workstream A · Trace guards in pipekit core

**Decision.** pipekit core stays free of JAX, but an operator that cannot do
its job on a tracer says so. Detection uses `sys.modules`: if `jax` is
already imported, `isinstance(x, jax.core.Tracer)`; otherwise nothing is
traced and nothing is checked. [](#tbl-r5-move-a) lists the moves.

:::{table} Workstream A's moves.
:label: tbl-r5-move-a

| Symbol | Change | Reason |
|---|---|---|
| `pipekit._base.tracing.is_traced(x)` | new, private | one check, no import of JAX |
| `Cache._apply` | raises `TypeError` on a traced input, naming `jit` and suggesting JAX's own compilation cache | A1: stage 3's wrong value |
| `_SnapshotAt._apply`, `Sink._apply` | raise `TypeError` on a traced input | A2: a stored or written tracer is never what the user wanted |
| `Tap._apply`, `_ProfileWrap._apply` | one `UserWarning` per instance when traced | A2: trace-time behaviour is legitimate for `Tap(print)`, not for timings |
| `ShapeTrace` | unchanged | its trace-time print is the point |
:::

```{code-block} python
:caption: Stage 3's Cache, after workstream A.
:linenos:

# Shapes: x (5,)
jax.jit(Sequential([cache, Lambda(lambda x: 10 * x), cache]))(x)
# → TypeError: Cache cannot key a traced value; inside jax.jit,
#   JAX's compilation cache already does this                proposed (A1)
jax.jit(Sequential([snap.at("z"), retrieval]))(y)
# → TypeError: Snapshot cannot store a traced value          proposed (A2)
```

(r5-ws-b)=
## Workstream B · Operators as PyTrees in pipekit-jax

**Decision.** When a project differentiates or vmaps over operator
parameters, pipekit-jax adds the scoping report's mixin. Until then the
documented pattern is stage 2's: differentiate the module, rebuild the
pipeline in the loss.

1. **B1 · `JaxOperator` and `JaxSequential`.** `JaxOperator(Operator,
   eqx.Module)` as in stage 4; `JaxSequential(Sequential, eqx.Module)` with
   `operators: tuple`, keeping `Sequential`'s `summary`, `|` and stateful
   path. Config fields are `eqx.field(static=True)`, so two pipelines with
   the same config share one compiled function.
2. **B2 · An array policy for configs.** `get_config` of a `JaxOperator`
   records each array field as `{"shape": …, "dtype": …}`, `forbid_in_yaml`
   stays false for the structure, and the values travel as the registry's
   weight blob through `serialize_weights` and `with_weights`, shared with
   `JaxModelOp`.
3. **No `JaxGraph` yet.** A `Graph` already traces as a callable (stage 3);
   a PyTree `Graph` meets the tree-not-DAG problem the scoping report
   described and waits for a use.

```{code-block} python
:caption: Stage 2's fit, after workstream B.
:linenos:

# Shapes: log k (5,) per operator
seq = pkj.JaxSequential([pkj.Invert(log_k0), pkj.Combine(log_k0)])  # proposed (B1)
g = eqx.filter_grad(loss)(seq, y, a)    # reaches both log k     proposed (B1)
reg.store(seq, weights=seq.serialize_weights())                  # proposed (B2)
json.dumps(seq.state)                   # shapes and dtypes only proposed (B2)
```

(r5-ws-c)=
## Workstream C · JAX-native control and observation

**Decision.** Branching on a traced value and observing at run time are
operators in pipekit-jax, with pipekit's names and JAX's semantics.

1. **C1 · `CondBranch(predicate, if_true, if_false)` and `CondSwitch(index,
   branches)`** on `jax.lax.cond` and `jax.lax.switch`. Both arms are traced
   and must return the same shape and dtype; the constructor cannot check
   that, so the first call does.
2. **C2 · `DebugTap(fn)`** on `jax.debug.callback`, which runs `fn` on
   concrete values on every call (three calls, three callbacks in stage 3).
   Its ordering across devices is not guaranteed; the docstring says so.

(r5-ws-d)=
## Workstream D · Transformation wrappers

**Decision.** Not now. `jax.vmap`, `jax.checkpoint` and `jax.lax.scan`
apply to a pipeline directly, as stage 1 does with `vmap`. A `VMapOver(op)`,
`Checkpoint(op)` or `ScanSequential(op, n)` operator adds a name and a
config to the transformation, which matters only when a pipeline that uses
one must round-trip through YAML. The gate is a project that needs that.

(r5-ws-e)=
## Workstream E · Docs

1. **E1 · Build time and trace time.** One page in pipekit's docs: what runs
   once at trace time (`ShapeTrace`, `Tap`, a `Lambda`'s Python), what
   survives into the compiled function, why `Cache` refuses a tracer, and
   stage 2's pattern for differentiating a module inside a pipeline.
2. **E2 · pipekit-jax's README and install notes.** Drop the link to
   `docs/design/` (or add the notes), remove `orbax-checkpoint` from the
   install notes, and drop `jaxtyping` from the dependencies or use it.

(r5-gates)=
## Testing gates

[](#tbl-r5-gates) gives each workstream's gate.

:::{table} Each workstream's gate, as a test in the package that owns it.
:label: tbl-r5-gates

| Gate | Workstream | Passes when |
|---|---|---|
| A1 Cache refuses tracers | A | stage 3's `twice` under `jax.jit` raises `TypeError` naming `jit`; eager, it still returns (21, …, 61) with one miss and one hit; pipekit core still imports with JAX absent |
| A2 Observers under tracing | A | under `jit`, `Snapshot` and `Sink` raise `TypeError`; `Tap` and `Profile` warn once each; `ShapeTrace` prints once and warns never |
| B1 PyTree pipeline | B | stage 4's `JaxSequential` gradient equals the rebuilt-module gradient to 10⁻⁹; `eqx.tree_at` replaces one operator; two pipelines with equal configs compile once under `eqx.filter_jit` |
| B2 JSON-safe config | B | `json.dumps(op.state)` works for every `JaxOperator`; a store and reload through `LocalModelRegistry` is byte-identical |
| C1 Conditional operators | C | `CondBranch` under `jit` equals the eager `Branch` to 10⁻¹² on stage 3's scene; mismatched arm shapes raise on the first call |
| C2 Run-time tap | C | `DebugTap` fires once per call under `jit` (3 calls, 3 callbacks) |
| E1, E2 Docs | E | the page builds and is linked from pipekit-jax's README; the install notes list exactly `pyproject.toml`'s dependencies |
:::

(r5-sequencing)=
## Sequencing and risks

:::{figure} ../images/reports/r5_roadmap.svg
:label: fig-r5-roadmap
:alt: Phase 1, A1 and A2 trace guards and E1 the build-time page, then gates A1, A2, E1. Phase 2, B1 and B2 PyTree operators with a JSON-safe config, and C1 and C2 conditional and run-time-tap operators, then gates B1, B2, C1, C2. Phase 3, D the transformation wrappers, when a project needs them, then gate D.

The order: an honest core first, then PyTree operators, then wrappers.
:::

[](#fig-r5-roadmap) orders the work. Phase 1 is small and protects users now: A1 is a few lines and a test, and
it removes the one silent wrong answer. Phase 2 starts when a project fits
operator parameters, as stage 2 does; until then stage 2's pattern works.
Phase 3 waits for a pipeline that must round-trip a transformation through
YAML.

- **Equinox's `Module` and `ConfigMixin` disagree about `__init__`.** The
  mixin reads `__init__`'s parameters; Equinox generates `__init__` from
  fields. Stage 4 shows they coexist, but a custom `__init__` on a
  `JaxOperator` must keep the field names, or `get_config` breaks. B1's
  test covers both.
- **Static fields and retracing.** A config field left as a leaf retraces
  on every new value, and an array marked static breaks hashing. B1's
  compile-once check catches the first; Equinox raises on the second.
- **The workaround may be enough.** If every project trains one module per
  pipeline, as pipekit-train does, B never needs to ship. That is the open
  question below.

(r5-open-questions)=
## Open questions

1. **Does any project need gradients through more than one operator?**
   Stage 2's rebuild pattern covers one learned module. Workstream B is
   worth it only for pipelines with several learned steps, or for vmapping
   over whole pipelines (the scoping report's hyperparameter sweeps, 7.6).
2. **Should `Cache` refuse tracers or pass them through?** Refusing (A1) is
   explicit; passing through to the inner operator would make a pipeline
   with a `Cache` jittable unchanged, at the cost of a cache that silently
   does nothing under `jit`.
3. **Where do the PyTree operators live if geotoolz needs them?** A JAX
   version of a GeoTensor operator (the scoping report's `jax_geotoolz`)
   would subclass `JaxOperator`. Whether that is a geotoolz extra or a new
   package is open, and nothing needs it yet.
4. **`JaxModelOp` and `JaxOperator`: one class or two?** A wrapper keeps
   any module unchanged; a mixin makes the operator itself the module.
   [Report 16](16_geo_data_stack.md#r16-ws-b) settles on `JaxModelOp` as the
   one wrapper; B1 would add the mixin beside it, not replace it.

(r5-todos)=
## Upcoming work

[](#tbl-r5-todos) lists the follow-ups.

:::{table} Follow-ups from checking the scoping report against the code.
:label: tbl-r5-todos

| Follow-up | Where | Why |
|---|---|---|
| Make `Cache` refuse a traced input | pipekit `cache.py:66` | stage 3: a wrong value under `jit`, with no error |
| Fix the README's `docs/design/` link and the install notes' dependency list | pipekit-jax `README.md:61`, pipekit `docs/installation.md:132` | the directory does not exist; `orbax-checkpoint` is not a dependency |
| Drop or use `jaxtyping` | pipekit-jax `pyproject.toml:17` | declared, never imported |
| Corrected here: the JAX exclusion in Reports [2](02_pipekit_surface.md#r2-not-here) and [3](03_sister_libraries.md#r3-array) | geostack `docs/reports/02_pipekit_surface.md`, `03_sister_libraries.md` | they said a pipeline cannot be jitted; stage 1 shows it can |
:::

(r5-references)=
## References

### Sources

- The scoping report, "Report 5 — `pipekit-jax`: a future-direction
  analysis" [@geostack_r5_draft].
- pipekit 0.0.2 and pipekit-jax 0.0.2 in `references/pipekit` at `ef3c17f`:
  source, tests, README, CHANGELOG and docs.
- JAX [@jax2018github], Equinox [@kidger2021equinox] and diffrax
  [@kidger2021neural], at the versions in the header.
- The running example's notebook, described in [](#r5-running).
