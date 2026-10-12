---
title: toolz primer
subtitle: Report 1 · Where pipekit comes from, and why its operators are classes
short_title: 1 · toolz primer
description: Background for the pipekit design. What pipekit inherits from toolz, the six things toolz cannot do, and why operators settled on a typed base class while carriers did not.
keywords:
  - pipekit
  - toolz
  - functional composition
  - operators
  - carriers
numbering:
  heading_1: true
  heading_2: true
  figure: true
  table: true
  code: true
---

+++ {"part": "abstract"}

pipekit is `toolz` with each function replaced by a typed class. This report
shows what that inheritance gives, the six things plain functions cannot
carry, and why operators settled on a concrete base class while the data
flowing through them did not. It is background: no new code lands here. The
shipped surface is [Report 2](02_pipekit_surface.md).

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Background and motivation |
| **Reading time** | about 15 minutes |
| **Audience** | anyone reviewing the pipekit design before the structural reports |
| **Companions** | [Report 2](02_pipekit_surface.md) (pipekit's shipped surface), [Report 3](03_sister_libraries.md) (sister libraries), [Report 9](09_use_cases.md) (use cases) |
| **Checked against** | toolz 1.2.0 and pipekit 0.0.2 (`ef3c17f`, the revision pinned in `references/pipekit`) |
:::

:::{important} What changed since the first draft
The first draft predates pipekit's first release. Checking it against the
pinned revision changed five things, each corrected in place below:

- `toolz.do` is only curried in `toolz.curried`; the mental-model example
  now imports it from there, since plain `do(print)` raises `TypeError`.
- A `toolz` composition *does* print its steps; the introspection gap is
  narrower than first claimed (see [Introspection](#r1-introspection)).
- `complement`, `juxt`, `Memoize`, `Try` and `Coalesce` now ship in pipekit;
  they are no longer gaps or deferred ([](#tbl-r1-mapping)).
- The `Operator` base class lives in pipekit itself; geotoolz and xrtoolz
  import it instead of each carrying their own.
- Serialisation is partial: flat operators round-trip, nested pipelines do
  not yet (see [Serialisation](#r1-serialisation)).
:::

(r1-why)=
## Why this report exists

The pipekit design makes more sense once you have seen three things: what it
inherited from `toolz`, what that inheritance cannot do, and why moving from
"any function" to a typed operator class is the right call rather than
over-engineering. This is the motivation layer. The operators themselves are
in [Report 2](02_pipekit_surface.md).

(r1-ancestor)=
## toolz as the ancestor

[`pytoolz/toolz`](https://github.com/pytoolz/toolz) is pipekit's explicit
intellectual ancestor. Looking at it directly is the fastest way to see which
idea pipekit carries forward and what it adds.

### The `toolz.functoolz` surface

:::{table} The `functoolz` primitives pipekit inherits from.
:label: tbl-r1-functoolz

| `toolz` primitive | What it does |
|---|---|
| `identity(x)` | pass the value through unchanged |
| `pipe(value, *funcs)` | apply functions left to right to a value |
| `compose(*funcs)` | right-to-left composition: `f(g(h(x)))` |
| `compose_left(*funcs)` | left-to-right composition |
| `juxt(*funcs)` | several outputs: `juxt(f, g)(x)` is `(f(x), g(x))` |
| `do(f, x)` | run `f(x)` for its side effect and return `x` |
| `complement(pred)` | negate a predicate |
| `excepts(exceptions, func, handler)` | catch the listed exceptions while applying `func` |
| `memoize(func)` | cache results by input |
| `curry` | partial application, as a class wrapper |
| `thread_first`, `thread_last` | Clojure-style threading macros |
| `flip(f, a, b)` | call `f(b, a)` |
:::

`toolz` also ships `itertoolz` (about 30 functions on iterables) and
`dicttoolz` (about 15 on dicts). pipekit only really inherits from
`functoolz`.

### The mental model

`toolz` says a pipeline is a sequence of pure functions, composed with `pipe`
or `compose`. Each function is responsible for its transform and nothing
else; metadata, types and side effects are someone else's problem.

```{code-block} python
:caption: A toolz pipeline. `do` must come from `toolz.curried` to be used without its value.
:linenos:

from toolz import pipe
from toolz.curried import do

pipe(
    "  HELLO  ",
    str.strip,
    str.lower,
    do(print),                  # side effect: prints "hello"
    lambda s: s.split("l"),
)
# → ['he', '', 'o']
```

This is functional composition over plain Python. Functions are values,
composition is an operation, and side effects are explicit (`do`). It is
elegant and minimal, and it directly inspired pipekit's `|` syntax, the
`Sequential` shape, the `Tap` / `Sink` split, and the `juxt` / `Fanout`
pair.

(r1-mapping)=
### One-to-one mappings to pipekit

Almost every pipekit primitive has a `toolz` ancestor. The mapping is tight
enough that the most accurate one-line description of pipekit is
**`toolz`, with each function replaced by a typed class**.

:::{table} toolz primitives and their pipekit counterparts at pipekit 0.0.2.
:label: tbl-r1-mapping

| `toolz` | pipekit | What pipekit's version adds |
|---|---|---|
| `pipe(value, *funcs)` | `pipe(value, *operators)`, `Sequential(operators)(value)` | operators carry config; a readable repr; reuse |
| `compose_left(*funcs)` | `compose_left(...)`, `op1 \| op2 \| op3` | `\|` flattens into one `Sequential`; class-typed steps |
| `compose(*funcs)` | `compose(...)` | the same, in mathematical order |
| `do(f, x)` | `Tap(fn)`, `Sink(write_fn)` | the closure is owned by an operator with a name; marked `forbid_in_yaml` |
| `juxt(f, g, h)` | `juxt(...)` (tuple) and `Fanout({...})` (dict) | both shapes; named outputs for downstream steps |
| `memoize` | `Cache`, with `Memoize` as an alias | config-aware hashing |
| `excepts` | `Try`, `Coalesce` | operator-shaped arms; an explicit exception list |
| `complement` | `complement` | ships as a plain function |
| `identity` | `Identity` | an operator, so it composes everywhere |
| `curry` | implicit in `__init__` | an operator's constructor is its partial application |
:::

The same pipeline in both libraries shows the difference. Both outputs below
are from real runs.

::::{tab-set}

:::{tab-item} toolz
```{code-block} python
:caption: A toolz composition prints its steps as raw function objects, in reverse order.
:linenos:

from toolz import compose_left

steps = compose_left(str.strip, str.lower, lambda s: s.split("l"))
print(steps)
# Compose(<function <lambda> at 0x7f…>, <method 'lower' of 'str' objects>,
#         <method 'strip' of 'str' objects>)
```
:::

:::{tab-item} pipekit
```{code-block} python
:caption: The same steps as pipekit operators: named, indexable, in order.
:linenos:

from pipekit import Lambda, Tap

steps = (
    Lambda(str.strip) | Lambda(str.lower) | Tap(print)
    | Lambda(lambda s: s.split("l"))
)
print(steps)
# Sequential([Lambda(name='strip'), Lambda(name='lower'),
#             Tap(name='print'), Lambda(name='<lambda>')])
steps[2]                   # → Tap(name='print')
steps("  HELLO  ")         # prints "hello"; → ['he', '', 'o']
```
:::
::::

`toolz` is the design template. pipekit is the typed, introspectable,
serialisable variant of the same idea.

(r1-gaps)=
## Where toolz stops being enough

`toolz` handles the mechanics of function composition cleanly. Six things are
beyond it, and each is why the domain libraries needed more.

:::{table} The six gaps, and where pipekit puts each one.
:label: tbl-r1-gaps

| Gap | What `toolz` gives | Where it lives in pipekit | Status at 0.0.2 |
|---|---|---|---|
| Introspection | a repr of raw function objects | named operators, `Sequential.__getitem__`, `summary()` | ships |
| Serialisation | nothing | `get_config`, `dumps` / `loads`, `from_state` | flat operators only |
| Dual-mode dispatch | eager calls only | `Operator.__call__` switches on `Node` inputs | ships |
| Control flow | Python `if` / `match` outside the pipeline | `Branch`, `Switch`, `Try`, `Coalesce` | ships |
| Shape tracking | `Any → Any` | `Signature`, `Sequential.summary()` | ships; needs signatures |
| Closure discipline | any closure, silently | the `forbid_in_yaml` class flag | ships |
:::

(r1-introspection)=
### Introspection

A `toolz` composition is not opaque: `Compose` prints its steps (the `toolz`
tab in [One-to-one mappings](#r1-mapping)). But it prints them as raw function
objects, in reverse order, with lambdas anonymous and no configuration or
per-step state to inspect. That is fine at an interactive prompt. For a
14-step preprocessing pipeline that has to be audited, you want named steps,
in order, that you can index (`steps[2]`) and summarise.

(r1-serialisation)=
### Serialisation

`toolz` pipelines are tuples of function references and closures. They do
not round-trip to YAML, JSON or a Hydra config, and they cannot be shipped to
a worker process and rebuilt. Every deployment shape in the
[use-case gallery](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/plans/geotoolz/examples/usecases.md)
(an ETL job, a LitServe or FastAPI service, a tile server, a regulatory
artifact, an orchestrator) needs serialisation in some form.

pipekit gives every operator a `get_config()` and a JSON `dumps` / `loads`.
At 0.0.2 the round-trip works for flat operators; a nested pipeline such as a
`Sequential` dumps, but `loads` refuses it and asks for a YAML or Hydra
loader that has not shipped yet:

```{code-block} python
:caption: What round-trips at pipekit 0.0.2, from a real run.
:linenos:

import pipekit as pk
from pipekit import Const, Identity, Sequential

pk.loads(pk.dumps(Const(3)))                   # → Const(value=3)
pk.loads(pk.dumps(Sequential([Identity()])))   # RuntimeError: from_state cannot
# reconstruct Sequential: config contains non-primitive values ['operators'].
```

The nested loader is an open question on the overview page, under
serialising a `Graph`.

### Dual-mode dispatch

`toolz` functions only run eagerly. You cannot pass a symbolic placeholder
through them to build a graph that runs later. Graph building (the `Input` /
`Node` API) needs the same callable to behave differently depending on
whether its argument is data or a placeholder. In pipekit that switch lives
once, in `Operator.__call__`: any `Node` argument builds a graph node, and
anything else calls `_apply`.

### Control flow

In a `toolz`-style pipeline, a conditional is a Python `if` or `match`
outside the pipeline, which splits the pipeline apart wherever one is needed.
pipekit makes conditionals first-class operators (`Branch`, `Switch`, `Try`,
`Coalesce`), composable inside a `Sequential` like any other step.

### Shape tracking

Every `toolz` step is `Any → Any`. For geoscience that hides structure worth
surfacing: xarray's named dimensions and a `GeoTensor`'s spatial metadata.
pipekit's `Signature` lets an operator declare what it accepts and returns,
and `Sequential.summary()` prints a Keras-style table that catches a shape
mismatch when the pipeline is built, not deep in a hot loop. Operators that
declare no signature show `?` in that table, so the payoff grows as more
operators declare one.

### Closure discipline

`toolz` accepts any closure. That is right for interactive work and wrong for
pipelines that must be audited: a closure cannot be inspected for what it
captures, serialised, or rebuilt on a worker. pipekit's answer is a class
flag. Operators whose constructor takes a closure declare
`forbid_in_yaml = True` (in pipekit 0.0.2: `Tap`, `Lambda`, `Sink`,
`Branch`, `Switch`, `Coalesce`, `ShapeTrace`, `Quarantine`,
`AssertCallable`), and a loader can refuse them. Operators that only nest
other operators, such as `Try`, `Retry` or `Cache`, are not flagged: a
walker finds the flagged children inside them.

:::{important} The gaps in one line
pipekit is what `toolz` becomes when the pipeline is a first-class object:
printable, configurable, summarisable, dispatchable, and bound by rules about
what it may contain.
:::

(r1-trajectory)=
## The trajectory: functions, Protocols, typed entities

The question was how pipekit moved from raw functions, through Protocols, to
typed entities. The answer differs for the steps of a pipeline (operators)
and for the data flowing through it ({term}`carriers <carrier>`), so the two
are told separately ({numref}`fig-r1-trajectories`).

```{figure} ../images/reports/r1_trajectories.svg
:label: fig-r1-trajectories
:alt: Two lanes. Operators move from toolz functions through an OperatorLike Protocol to pipekit's Operator class, which is highlighted as settled. Carriers show Any in pipekit core, Array API Protocols in pipekit-array, and GeoTensor or xarray in geotoolz and xrtoolz, all in use at once.
:width: 100%

**Operators settled on a typed class; carriers use all three stages at once.**
```

(r1-operators)=
### Operators

::::{tab-set}

:::{tab-item} 1 · Functions
The `toolz` shape: an operator is a Python function. Maximum flexibility,
zero boilerplate.

```{code-block} python
:linenos:

def my_op(x):
    return ...

pipe(value, my_op, other_op, third_op)
```

**Buys you:** any existing function adapts trivially, with no subclassing
and no dependency beyond `toolz`.

**Costs you:** no config, no readable repr, no serialisation, no dispatch,
no flags, no state, and no reuse beyond reusing the function.
:::

:::{tab-item} 2 · Protocols
The natural next thought is to write a `Protocol` for what an operator looks
like. Anything structurally compatible counts; nothing has to inherit.

```{code-block} python
:linenos:

from typing import Protocol

class OperatorLike(Protocol):
    def __call__(self, x): ...
    def get_config(self) -> dict: ...
```

**Buys you:** structural typing. Tools can recognise operator-shaped objects
without forcing inheritance, which suits third-party code that does not
import your library.

**Costs you:** a Protocol is a check, not a base class.

- **No default implementations.** Every implementer rebuilds `get_config`,
  `__repr__` and `__or__`; there is no `super()` to reuse.
- **No class-level flags.** `forbid_in_yaml` and `_terminal` describe an
  operator's contract, not its instances, and a Protocol cannot carry them
  cleanly.
- **No shared mechanism.** Dual-mode dispatch is infrastructure every
  operator needs; with a Protocol each one reimplements it.
- **No class registry.** `Operator.from_state` rebuilds an operator by
  walking `cls.__subclasses__()`; a Protocol has no subclasses to walk.

**Verdict:** a Protocol is the right tool to check that something is
operator-shaped, and the wrong tool to be the operator base.
:::

:::{tab-item} 3 · Typed base class
Where pipekit landed. A simplified view of `pipekit/_base/operator.py`:

```{code-block} python
:caption: The Operator base class, simplified from pipekit 0.0.2.
:linenos:

class Operator(ConfigMixin):
    forbid_in_yaml: ClassVar[bool] = False   # closure-holding operators set True
    _terminal: ClassVar[bool] = False        # may only end a Sequential

    def __call__(self, *args, **kwargs):     # dual-mode dispatch
        if any(isinstance(a, Node) for a in args):
            ...                              # build a graph node
        return self._apply(*args, **kwargs)  # run eagerly

    def _apply(self, *args, **kwargs):       # the one method subclasses write
        raise NotImplementedError

    def __or__(self, other) -> Sequential:   # pipe syntax, defined once
        ...

    @classmethod
    def from_state(cls, state) -> Operator:  # walks cls.__subclasses__()
        ...
```

**Buys you:**

- `__init__` carries config, `get_config()` (derived by `ConfigMixin`)
  exposes it, and `from_state` rebuilds from it.
- `__or__` is defined once, so every subclass gets pipe syntax for free.
- Dual-mode `__call__` is defined once; subclasses only write `_apply`.
- Class flags (`forbid_in_yaml`, `_terminal`) are visible to `Sequential`'s
  validation and to any loader.
- IDEs and type checkers see the whole surface.

**Costs you:**

- Subclassing is more ceremony than a function (`class NDVI(Operator)`).
- A third-party function does not adapt for free; `Lambda` is the escape
  hatch, and `ConfigMixin` keeps the boilerplate small.
- More framework code to maintain.

**Verdict:** the right design for operators. Every responsibility has a
natural home, and the cost is a one-time decision to subclass. geotoolz and
xr_toolz each arrived here independently before the base class moved into
pipekit, which is the strongest sign it is right.
:::
::::

(r1-carriers)=
### Carriers

The carrier trajectory is different, and not settled.

::::{tab-set}

:::{tab-item} 1 · Any
pipekit's core uses `Carrier = Any`. The framework does not care what flows
through; operators that need structure check it in their own signatures.

**Buys you:** maximum flexibility. Composition works the same whether the
carrier is a `GeoTensor`, an `xr.DataArray`, a NumPy array, a scalar or a
user type, and test pipelines can run on scalars.

**Costs you:** operators that do want structure (named axes, CRS, units)
duck-check it every time, and there is no shared metadata surface.
:::

:::{tab-item} 2 · Protocols
The ecosystem has converged on Protocol-shaped array descriptions through the
[Array API standard](https://data-apis.org/array-api/): `numpy.ndarray`,
`jax.Array`, `cupy.ndarray`, `torch.Tensor` and `dask.array.Array` all
expose `__array_namespace__`. Operators written against it declare the
Protocol and work across backends; pipekit-array is where they live.

**Buys you:** an operator written once runs on NumPy, JAX, CuPy and PyTorch
arrays, which is real value for cross-framework pipelines.

**Costs you:** the Array API becomes a dependency; operators call the
namespace (`xp.mean(x)`) rather than NumPy (`np.mean(x)`); and some
operations do not generalise cleanly, NaN handling for one.
:::

:::{tab-item} 3 · Concrete types
geotoolz and xrtoolz type their operators to a specific carrier: `GeoTensor`
for one, `xr.DataArray` or `xr.Dataset` for the other.

**Buys you:** rich metadata used natively. geotoolz operators read `.crs`
and `.transform` without defensive checks; xrtoolz operators use `.dims` and
`.coords` directly.

**Costs you:** each library is locked to its carrier, so
`Sequential([geotoolz_op, xrtoolz_op])` does not compose across them, and a
carrier-specific operator is written once per library.
:::
::::

(r1-lands)=
### Where pipekit lands

:::{table} The two trajectories, resolved separately.
:label: tbl-r1-lands

| Axis | Where it lands | Status |
|---|---|---|
| Operators | a typed concrete base class, `pipekit.Operator` | settled |
| Carriers, pipekit core | `Carrier = Any` | settled: the core stays carrier-agnostic |
| Carriers, multi-backend | Array API Protocols, in pipekit-array | in progress |
| Carriers, domain libraries | `GeoTensor` in geotoolz; `xr.DataArray` / `xr.Dataset` in xrtoolz | in use |
:::

It was never one arc from functions to Protocols to typed entities. It was
two trajectories that shared vocabulary. Operators started at `toolz`
functions, considered Protocols, and landed on a typed base class. Carriers
are still at all three stages at once, by design: each library uses the
stage that serves it. Telling them as one story blurs the picture.

(r1-not-overengineering)=
## Why typed operators are not over-engineering

"Just use functions" is tempting: less code, faster to write, familiar to
anyone who has used `toolz`, no subclassing.

The case for typed operators is that the six gaps in {numref}`tbl-r1-gaps`
have no function-shaped solutions. Each needs somewhere to live: `get_config`
on the operator, `forbid_in_yaml` on the class, dispatch in `__call__`,
subclass walking in `from_state`. A function is the wrong host for all of
them.

So the argument is not that classes beat functions. It is that **the
responsibilities pipekit takes on are class-shaped**. Remove them (no
introspection, no serialisation, no dispatch, no flags) and you are left with
`toolz`, which is a perfectly good answer for problems that fit it. pipekit's
problems do not.

(r1-next)=
## Where this leaves us

[Report 2](02_pipekit_surface.md) turns this background into pipekit's shipped surface:

- `toolz`-style functions for the familiar surface: `pipe`, `compose`,
  `compose_left`, `juxt`, `complement`;
- a small typed `Operator` base with the machinery for the six gaps:
  `get_config`, `from_state`, `forbid_in_yaml`, `_terminal`, dual-mode
  dispatch and the pipe operator;
- composition on top: `Sequential`, `Graph`, `Fanout`, `Branch`, `Switch`;
- observers, control flow and building blocks;
- shape inference (`Signature`) and parallelism (`ThreadMap`, `ProcessMap`,
  `AsyncMap`, `BatchedMap`).

[Report 3](03_sister_libraries.md) covers the sister libraries (pipekit-array, geotoolz, xrtoolz), and
[Report 9](09_use_cases.md) maps everything back to the use cases. The
[overview](../overview.md) places all three in the wider stack.

(r1-references)=
## References

<!-- TODO(ref): placeholders. When a report page lands, point its links in this
file at the page (e.g. [Report 2](02_pipekit_surface.md)) and delete its entry
below. When citations land, add them to references.bib, cite with [@key] in the
text, and replace the citations placeholder. -->

:::{note} Placeholders
:class: simple
The citations are still being written; the entry below stands in for them.
The companion reports are published: [Report 2](02_pipekit_surface.md),
[Report 3](03_sister_libraries.md) and [Report 9](09_use_cases.md).
:::

### Citations

*To be added:* `toolz`, the Python Array API standard, and the use-case
gallery, as entries in `references.bib`.
