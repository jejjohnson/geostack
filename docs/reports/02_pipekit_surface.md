---
title: What pipekit ships
subtitle: Report 2 · The v0.1 surface proposal, checked against what shipped in pipekit 0.0.2
short_title: 2 · pipekit surface
description: pipekit's public surface, organised into semantic groups, one module each. Each group records what was proposed, what shipped in pipekit 0.0.2, and where the two differ.
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

This report began as the proposal for pipekit's first public surface: one
module per semantic group, about 42 symbols. pipekit 0.0.2 has since
shipped, so the report now records what was built. Most groups landed as
planned. Seven differ in ways that matter to a user, two groups are new, and
most of the operators the plan handed to sister libraries do not exist yet.
Each difference is marked where it occurs, and each becomes either an open
question or a follow-up at the end.

+++

:::{note} About this report
:class: dropdown

| | |
|---|---|
| **Status** | Shipped surface; originally the v0.1 surface proposal (revised after the L0–L4 analysis) |
| **Reading time** | about 30 minutes |
| **Audience** | anyone deciding what belongs in pipekit core, or writing an operator against it |
| **Companions** | [Report 1](01_toolz_primer.md) (background); [Report 3](03_sister_libraries.md) (sister libraries); [Report 9](09_use_cases.md) (use cases); [Report 10](10_pipekit_cycle.md) (cycle); [Report 11](11_pipekit_train.md) (train); [Report 12](12_pipekit_experiment.md) (experiment); [Report 13](13_state_catalog.md) (state catalog) |
| **Checked against** | pipekit 0.0.2 (`ef3c17f`, the revision pinned in `references/geostack/pipekit`); every output on this page is from a real run |
:::

:::{important} How to read the differences
Each group says what the proposal planned and what shipped, following the
[series conventions](../overview.md#sec-conventions). The proposal named the
release v0.1; what shipped is versioned 0.0.2, although pipekit's own
docstrings still call it "the v0.1 public surface".
:::

(r2-layout)=
## Module layout

The proposal organised pipekit by **semantic group** rather than by tier:
each group is one module with one purpose, so it can be documented, reviewed
and planned on its own. That organisation shipped. pipekit's top-level
docstring lists the same groups, A to N, one module each
({numref}`fig-r2-groups`).

```{figure} ../images/reports/r2_groups.svg
:label: fig-r2-groups
:alt: Fifteen cards, one per pipekit module. Composition, building blocks, combination, caching and quality control shipped as planned. Foundations, control flow, observers, shape inference, parallelism, serialisation and state differ from the plan. Protocols and hashing are new. Group K is reserved and ships nothing.
:width: 100%

**pipekit 0.0.2 by group, coloured by how each matches the plan.**
```

The module-by-module symbols and line counts are in
[the full inventory](#r2-inventory) ([](#tbl-r2-inventory)); `_base/` splits
into `operator.py` (338 lines), `sequential.py` (243) and `graph.py` (256).
In aggregate, [](#tbl-r2-layout) compares the plan with what shipped.

:::{table} The layout, planned and shipped.
:label: tbl-r2-layout

| | Planned | Shipped at 0.0.2 |
|---|---|---|
| Modules | 12 | 16 (`serial`, `protocols` and `hashing` added) |
| Public names | about 42 | 52 in `pipekit.__all__` |
| Source | about 1,600 lines (layout); about 2,150 (inventory) | 3,164 lines, docstrings included |
| Tests | about 1,600 lines | 2,284 lines across 16 test modules |
| Third-party dependencies | none | none (`dependencies = []`) |
:::

The proposal's own layout block named the parallel module's thread-pool
operator `ParMap`; its Group J table, and the shipped code, call it
`ThreadMap`.

(r2-group-a)=
## Group A · Foundations (`_base/`)

The irreducible kernel: pure Python, no third-party dependencies, nothing
imported from NumPy, xarray or anything else. That held.

### `Operator`, the base class

The class everything else subclasses. It combines the best features of the
two parent libraries:

- **dual-mode `__call__`** (from geotoolz): eager on data, symbolic on a
  `Node` or `Input`;
- **`ConfigMixin`** (from xr_toolz): `get_config()` derived from the
  `__init__` signature through `inspect`, with `__config_mixin_auto__` and
  `__config_exclude__` as escape hatches;
- **`state` / `from_state`** (from geotoolz): a JSON-safe record, rebuilt by
  walking every transitive subclass;
- **class flags**: `forbid_in_yaml` and `_terminal`;
- **the pipe operator**: `op_a | op_b` is `Sequential([op_a, op_b])`,
  flattened;
- **`_dispatch_post_apply_hooks`**: a no-op reserved for a later hook
  surface.

All of it shipped. A simplified view of the shipped class:

```{code-block} python
:caption: `Operator` at pipekit 0.0.2, simplified from `_base/operator.py`.
:linenos:

class Operator(ConfigMixin):
    forbid_in_yaml: ClassVar[bool] = False   # holds a closure: no round-trip
    _terminal: ClassVar[bool] = False        # may only end a Sequential
    _is_stateful: ClassVar[bool] = False     # set by StatefulOperator (Group M)

    def __call__(self, *args, **kwargs):     # any Node arg → build a Node
        ...                                  # else → _apply, then the hook stub
    def _apply(self, *args, **kwargs): ...   # the one method subclasses write
    def compute_output_signature(self, sig): # default: passthrough
        return sig
    @property
    def state(self) -> dict: ...             # {"module", "class", "config"}
    @classmethod
    def from_state(cls, state) -> Operator: ...
    def __or__(self, other) -> Sequential: ...
```

Writing an operator is a constructor that stores its arguments under the
same names, plus `_apply`:

```{code-block} python
:caption: A user operator, its state record, and a JSON round-trip.
:linenos:

import pipekit as pk
from pipekit import Operator

class Scale(Operator):
    def __init__(self, factor: float) -> None:
        self.factor = factor              # same name as the argument → get_config

    def _apply(self, x):
        return x * self.factor

Scale(2.0).state
# → {'module': '__main__', 'class': 'Scale', 'config': {'factor': 2.0}}
pk.loads(pk.dumps(Scale(2.0)))            # → Scale(factor=2.0)
```

The shipped class adds one thing the proposal did not have: the
`_is_stateful` flag that Group M's `Sequential` plumbing reads.

### `Sequential`, linear composition

The workhorse: apply operators left to right. Everything proposed shipped:

- construction-time validation: every item is an `Operator`, and a
  `_terminal` operator may only come last;
- `|` flattening, so `(a | b) | c` and `a | (b | c)` are both one
  three-step `Sequential`;
- `len`, indexing, and slicing (`pipe[1:3]` is a new `Sequential`);
- `summary(input_signature)`, a Keras-style table, and `describe()`, an
  indented tree;
- `Sequential([])(x)` is the identity.

```{code-block} python
:caption: Sequential at 0.0.2, from a real run.
:linenos:

from pipekit import Sequential, Signature

pipe = Scale(0.5) | Scale(2.0) | Scale(3.0)
pipe(10.0)        # → 30.0
len(pipe)         # → 3
pipe[1:]          # → Sequential([Scale(factor=2.0), Scale(factor=3.0)])
Sequential([])(7) # → 7;  Sequential([])() raises TypeError

print(pipe.summary(Signature(dims=(("y", 4), ("x", 4)), dtype="float32")))
# Operator   Input sig                      Output sig
# -----------------------------------------------------------------------
# [0] Scale  Signature((y=4, x=4) float32)  Signature((y=4, x=4) float32)
# [1] Scale  Signature((y=4, x=4) float32)  Signature((y=4, x=4) float32)
# [2] Scale  Signature((y=4, x=4) float32)  Signature((y=4, x=4) float32)
```

### `Graph`, `Input` and `Node`, DAG composition

Symbolic graphs with several inputs and outputs. Calling operators on
`Input` or `Node` objects builds the graph; calling the `Graph` evaluates it
in topological order.

```{code-block} python
:caption: A two-input Graph, called by keyword and by position.
:linenos:

from pipekit import Graph, Input, Lambda

a, b = Input("a"), Input("b")
s = Scale(2.0)(a)                                       # Node: s = 2a
total = Lambda(lambda u, v: u + v, name="add")(s, b)    # Node: s + b

g = Graph(inputs={"a": a, "b": b}, outputs={"s": s, "sum": total})
g(a=1.0, b=10.0)     # → {'s': 2.0, 'sum': 12.0}
g(1.0, 10.0)         # → the same; positional args bind in declaration order
```

What shipped as proposed: `Input` subclasses `Node` (the xr_toolz shape),
cycles are rejected at construction, inputs can be positional or keyword, and
a single-input `Graph` composes into a `Sequential`
(`(Scale(2.0) | g1)(1.0)` returns `{'y': 6.0}`).

:::{warning} Differs from the plan
- **`Graph` has `describe()` but no `summary()`.**
- **Signatures do not propagate through a `Graph`.** The proposal had a
  `_bind` / `_execute` / `_compute_signatures` triad from xr_toolz. None of
  it shipped: `Graph` inherits the passthrough `compute_output_signature`,
  so a two-output graph reports its *input* signature as its output.
- **`Sequential.summary` does not raise on an operator that doesn't track
  shape.** It prints `?` for that step and carries on. That is arguably the
  better behaviour, but it contradicts the proposal and the docstring of
  `Operator.compute_output_signature`, which still says summary "raises a
  clean error". See [](#r2-group-i).
:::

(r2-group-b)=
## Group B · Composition convenience (`compose.py`)

Small free functions, not `Operator` subclasses, that carry the `toolz`
surface ([Report 1](01_toolz_primer.md)). They shipped exactly as proposed
([](#tbl-r2-compose)).

:::{table} Group B at 0.0.2. Outputs are from real runs.
:label: tbl-r2-compose

| Symbol | `toolz` ancestor | What it does | Example |
|---|---|---|---|
| `pipe(value, *ops)` | `toolz.pipe` | `Sequential(ops)(value)`, with less ceremony | |
| `compose(*ops)` | `toolz.compose` | right to left: `compose(f, g, h)` is `Sequential([h, g, f])` | `compose(Scale(2.0), inc)(3.0)` → `8.0` |
| `compose_left(*ops)` | `toolz.compose_left` | left to right: an alias of `Sequential` | `compose_left(Scale(2.0), inc)(3.0)` → `7.0` |
| `complement(pred)` | `toolz.complement` | negates a predicate | `complement(str.isdigit)` is named `complement(isdigit)` |
| `juxt(*ops)` | `toolz.juxt` | runs each operator on the input; returns a tuple | `juxt(Identity(), Scale(2.0))(5.0)` → `(5.0, 10.0)` |
:::

`juxt` returns a plain function, as a free function should. The consequence is
that it is not an `Operator`: it cannot sit inside a `Sequential` or be
joined with `|`. `Fanout` ([](#r2-group-f)) is the operator-shaped version.

(r2-group-c)=
## Group C · Building blocks (`blocks.py`)

Tiny operators that look trivial on their own and unlock common patterns in
combination. Shipped as proposed ([](#tbl-r2-blocks)).

:::{table} Group C at 0.0.2.
:label: tbl-r2-blocks

| Operator | Purpose | `forbid_in_yaml` |
|---|---|---|
| `Identity()` | explicit no-op; the default for `Branch.if_false` and `Switch.default` | no |
| `Const(value)` | return a fixed value, ignoring the input | no |
| `Lambda(fn, name=...)` | wrap a callable; the escape hatch | yes |
| `Sink(write_fn, name=...)` | side-effect write that returns its input; checkpointing | yes |
:::

`Sink` returns the carrier, so it is not `_terminal`. An operator that really
ends a pipeline (writes and returns `None`) subclasses with
`_terminal = True`.

(r2-group-d)=
## Group D · Control flow (`control.py`)

Conditionals and exception handling as operators *inside* the pipeline,
rather than Python `if` and `try` around it. All five proposed operators
shipped: two from geotoolz core, three promoted from the
`pipeline_idioms.ipynb` build-your-own notebook ([](#tbl-r2-control)).

:::{table} Group D at 0.0.2, with the `forbid_in_yaml` flag as shipped.
:label: tbl-r2-control

| Operator | Purpose | `forbid_in_yaml` |
|---|---|---|
| `Branch(predicate, if_true, if_false=Identity())` | binary conditional | yes |
| `Switch(key, cases, default=Identity())` | multi-way dispatch on `key(carrier)` | yes |
| `Try(primary, fallback, on)` | `toolz.excepts`: run `fallback` on a listed exception | **no** |
| `Coalesce(sources, is_ok, on=())` | the first source whose output passes `is_ok` wins | yes |
| `Retry(op, attempts, base_delay, on)` | exponential backoff: attempt *i* waits `base_delay · 2^(i−1)` | **no** |
:::

```{code-block} python
:caption: Try, Coalesce and Retry at 0.0.2, from a real run.
:linenos:

from pipekit import Coalesce, Const, Lambda, Retry, Try

def down(_):
    raise IOError("down")

Try(Lambda(down), Const(0.0), on=(IOError,))(1.0)          # → 0.0
Coalesce(
    [Lambda(down), Const(-1), Const(5)],
    is_ok=lambda r: r > 0,                                  # -1 fails the check
    on=(IOError,),                                          # down() falls through
)(0)                                                        # → 5
Retry(flaky, attempts=5, base_delay=0.0, on=(ConnectionError,))(42)
# → 42, after two failed attempts and one success
```

Three details that shipped but weren't in the proposal: `Try.on` and
`Retry.on` are required and reject an empty tuple, so neither is ever a silent
catch-all. `Coalesce` grew the same optional `on` tuple. When every source
fails, `Coalesce` raises `RuntimeError`.

:::{warning} Differs from the plan
The proposal said all five "carry `forbid_in_yaml = True`". Only the three
that hold a user callable (`Branch`, `Switch`, `Coalesce`) do. `Try` and
`Retry` hold only nested operators and exception types, which pickle, so they
are not flagged. This is deliberate and documented in `control.py`:
`check_pickleable` finds flagged operators *inside* them by walking the tree
([](#r2-pickle)). A nested-operator config is still not round-trippable
through `loads` ([](#r2-group-l)).
:::

(r2-group-e)=
## Group E · Observers (`observe.py`)

Identity-with-side-effect operators: the carrier flows through unchanged and
something useful happens on the side. They matter for notebook exploration
and debugging, and help with production logging. [](#tbl-r2-observe) lists
the five.

:::{table} Group E at 0.0.2.
:label: tbl-r2-observe

| Symbol | Purpose |
|---|---|
| `Tap(fn, name=...)` | fire a callback; return the input. `Sink` with an observation name |
| `Snapshot()` + `.at(key)` | capture intermediates by name in `snapshot.captures` |
| `ShapeTrace(printer=print, mode="every" \| "diff_only", label=None)` | log `shape` and `dtype` at each step, read with `getattr` |
| `Profile()` + `.wrap(op)` | per-step timings; `profile.report()` gives calls, total and mean per operator |
| `Histogram(bins=10, to_array=list)` + `.at(key)` | capture distributions; `to_array` adapts the carrier to numbers |
:::

`Snapshot`, `Profile` and `Histogram` use the **controller, not operator**
pattern. The class itself is not an `Operator`; `.at()` and `.wrap()` return
small operators that close over the controller's storage, so the collected
data lives on one object the user can inspect:

```{code-block} python
:caption: The controller pattern at 0.0.2, from a real run.
:linenos:

from pipekit import Histogram, Lambda, Profile, Snapshot

snap, prof, hist = Snapshot(), Profile(), Histogram(bins=2)
pipe = Lambda(lambda v: [v, 2 * v], name="pair") | snap.at("mid") | hist.at("out")
pipe(1.0)                  # → [1.0, 2.0]
snap.captures              # → {'mid': [1.0, 2.0]}
prof.wrap(Scale(3.0))(2.0) # → 6.0;  prof.report() → {'Scale': {'calls': 1.0, ...}}
```

`Histogram` was on the deferred list and moved into core with a
carrier-agnostic `to_array`, as the proposal planned. The default is `list`,
so the adapter is only needed for arrays (`to_array=lambda a: a.ravel().tolist()`).

:::{warning} Differs from the plan
The proposal said the "`crs` line" that geotoolz's trace prints would become
a subclass hook, and that `ShapeTrace` would log "shape / dtype / extra".
`ShapeTrace` ships with no hook: it logs `shape` and `dtype` only, and a
carrier with neither prints `shape=None dtype=None`. A domain library that
wants the CRS line overrides `_apply`. See [](#r2-todos).
:::

(r2-group-f)=
## Group F · Combination (`combine.py`)

One input in, many outputs out. `Fanout` is the dict-keyed operator; `juxt`
is the tuple-returning function.

```{code-block} python
:caption: Fanout at 0.0.2.
:linenos:

from pipekit import Fanout, Lambda

Fanout({"n": Lambda(len), "u": Lambda(str.upper)})("ab")   # → {'n': 2, 'u': 'AB'}
```

The proposal described `Fanout` as sugar over a single-input `Graph`. The
shipped one is a direct loop over its branches, with the same behaviour and
fewer moving parts. Combinators that merge carrier-specifically (`Augment`,
`ApplyToEach`) stay in xrtoolz because they use `xr.merge`. That is where
they shipped ([](#r2-group-k)).

(r2-group-g)=
## Group G · Caching (`cache.py`)

`Cache(inner)`, with `Memoize` as an alias (named after `toolz.memoize`),
memoises an inner operator in memory. The key is a SHA-256 of the input and
the inner operator's canonical JSON config. A disk backend is deferred, as
planned.

```{code-block} python
:caption: Cache at 0.0.2, from a real run.
:linenos:

from pipekit import Cache, Memoize

c = Cache(Scale(2.0))
c(1.0); c(1.0); c(2.0)
c.hits, c.misses        # → (1, 2)
Memoize is Cache        # → True
```

Three details that shipped but weren't in the proposal:
- The counters are public, `hits` and `misses`, rather than `_hits` and
  `_misses`.
- Array-likes are keyed by their full bytes, dtype and shape, because NumPy
  truncates the `repr` of a large array. A `repr`-based key, as proposed,
  would let two different large arrays collide.
- Bookkeeping is lock-protected, so one `Cache` is safe to share across
  `ThreadMap` workers.

The hashing primitives moved into a shared module ([](#r2-hashing)).

(r2-group-h)=
## Group H · Quality control (`qc.py`)

Assertions as pass-through operators. Every QC check turns a research-time
"I bet this won't break" into a permanent runtime guard, and CI gets a
pipeline-correctness suite for free. Shipped as proposed, plus a dedicated
exception, `QCError` (a subclass of `AssertionError`); [](#tbl-r2-qc) lists
the operators.

:::{table} Group H at 0.0.2.
:label: tbl-r2-qc

| Operator | Purpose | `forbid_in_yaml` |
|---|---|---|
| `Quarantine(check, sentinel, on_quarantine=None)` | non-raising: on failure call `on_quarantine` and return `sentinel` | yes |
| `AssertShape(expected_shape)` | raise `QCError` on a shape mismatch; `None` matches any size | no |
| `AssertDType(expected_dtype)` | raise `QCError` on a dtype mismatch | no |
| `AssertHasAttribute(name, value=...)` | raise `QCError` if the attribute is missing, or has the wrong value | no |
| `AssertCallable(predicate, message=...)` | raise `QCError` if a user predicate fails | yes |
:::

```{code-block} python
:caption: The error an assertion raises, from a real run.

AssertShape((2, 4))(x)   # x.shape == (2, 3)
# QCError: AssertShape: axis 1 expected 4, got 3 (expected=(2, 4), actual=(2, 3)).
```

The checks stay carrier-agnostic by reading `shape`, `dtype` and attributes
through `getattr`, never looking inside the array. Numeric checks that do
look inside (`AssertValueRange`, `AssertNoNaN`) were assigned to a sister
library and have not shipped anywhere ([](#r2-group-k)). The proposal's
"raise (or warn)" has no warn mode: an assertion raises, and `Quarantine` is
the non-raising alternative.

(r2-group-i)=
## Group I · Shape inference (`signature.py`)

`Signature` is an immutable record of named dimensions and a dtype. Each
dimension is a `(name, size)` pair, and a size of `None` marks it dynamic.
It has `format()`, `drop_dims()` and `replace_dims()`, as proposed, and the
free function `compute_output_signature(op, sig)` dispatches to the method.

```{code-block} python
:caption: Signature at 0.0.2.
:linenos:

from pipekit import Signature

sig = Signature(dims=(("time", 12), ("y", 256), ("x", 256)), dtype="float32")
sig.format()                   # → '(time=12, y=256, x=256) float32'
sig.drop_dims(("time",))       # → Signature((y=256, x=256) float32)
sig.replace_dims({"x": 128})   # → Signature((time=12, y=256, x=128) float32)
```

The proposal's **critical wrinkle** was that `Signature` assumes named
dimensions, which NumPy and `GeoTensor` carriers don't have. The fix shipped
as planned: `compute_output_signature` returns `Signature | None`, and `None`
means "this operator does not track shape".

:::{warning} Differs from the plan
- **`summary` shows `?` instead of raising.** The proposal (and its open
  question 3) said `Sequential.summary` raises a clear error when an operator
  returns `None`. It prints `?` and continues. The docstring on `Operator`
  still describes the raise.
- **Shape inference is plumbing only in core.** No pipekit operator except
  `Sequential` overrides `compute_output_signature`. Everything else,
  including `Graph`, `Fanout` and the four parallel maps, passes the input
  signature through unchanged, even though each returns something of a
  different shape (a dict, or a list). The payoff arrives when sister
  libraries declare signatures for their own operators.
:::

(r2-group-j)=
## Group J · Parallelism (`parallel.py`)

A group neither parent library had. It gets more space because parallelism
is the single biggest deployment concern, and its decisions ripple back into
how every operator must be written.

### The four parallelism shapes

Pipelines hit four distinct patterns, and pipekit ships one wrapper for
each ([](#tbl-r2-parallel)). Every wrapper takes an iterable and returns a
list in input order.

:::{table} Group J at 0.0.2: the four shapes and the shipped signatures.
:label: tbl-r2-parallel

| Shape | When | Constraint | Shipped operator |
|---|---|---|---|
| Thread pool | I/O-bound: S3 reads, tile downloads | the GIL must release in the I/O path | `ThreadMap(op, n_workers=8)` |
| Process pool | CPU-bound, pickleable work | every step must pickle: no closures, no captured RNGs | `ProcessMap(op, n_workers=8, on_error="raise" \| "log_and_continue")` |
| Async | I/O on one thread: FastAPI, LitServe | `async def _apply`, or a sync operator run in a thread | `AsyncMap(op, semaphore=8)` |
| Batched | GPU inference, vectorised NumPy | the inner operator takes a *list* of items | `BatchedMap(op, batch_size=8, flatten=True)` |
:::

```{code-block} python
:caption: The parallel maps at 0.0.2, from a real run.
:linenos:

from pipekit import AsyncMap, BatchedMap, Lambda, ThreadMap

ThreadMap(Scale(2.0), n_workers=2)([1, 2, 3])                 # → [2.0, 4.0, 6.0]
add_up = Lambda(lambda b: [sum(b)], name="sum")
BatchedMap(add_up, batch_size=2)([1, 2, 3, 4, 5])             # → [3, 7, 5]
# batches: [1, 2] → 3,  [3, 4] → 7,  [5] → 5
AsyncMap(Scale(2.0), semaphore=2)([1, 2])                     # → [2.0, 4.0]
```

:::{warning} Differs from the plan
- **`BatchedMap` batches a list, not axis 0.** The proposal's constraint was
  "carrier axis 0 is the batch dimension". The shipped `BatchedMap` is
  carrier-agnostic: it slices an iterable into lists, and the inner operator
  decides what to do with each list (stack it, send it to a GPU). The
  `flatten` switch is new. Splitting a single array along axis 0 is left to
  pipekit-array, which does not have it yet.
- **`ProcessMap`'s `log_and_continue` prints.** A failed item writes one line
  to `stderr` and leaves `None` in its slot in the results. It does not use
  `logging`, so a deployment cannot route or silence it.
:::

(r2-pickle)=
### The pickleability discipline

Process-pool parallelism is the most common production shape, and it imposes
a severe constraint that isn't obvious until something breaks at scale:
**every operator in the pipeline must pickle.**

- Closures in `Lambda`, `Tap`, `Sink`, `Branch.predicate`, `Switch.key` and
  `ModelOp.model` are pickling hazards.
- RNGs captured in augmentation operators silently produce correlated batches
  across workers.
- Default cloud credentials, file handles and GPU contexts do not pickle.

The response is that the `forbid_in_yaml` flag does double duty as a
pickleability warning, and `check_pickleable(op)` walks the operator tree to
find flagged operators before a pipeline fans out. The proposal called it "a
future utility"; it shipped in 0.0.2. It scans every attribute of every
operator generically, including list, dict and `Graph` node containers, so
new operator shapes are covered without being registered:

```{code-block} python
:caption: check_pickleable finds flagged operators however deeply they nest.
:linenos:

from pipekit import Branch, Cache, Identity, Tap, ThreadMap, check_pickleable

pipe = ThreadMap(Cache(Scale(2.0) | Tap(print) | Branch(bool, Identity())))
[type(op).__name__ for op in check_pickleable(pipe)]   # → ['Tap', 'Branch']
```

It is lint, not enforcement, and it **returns** the flagged operators rather
than emitting warnings; the caller decides what to do with the list.

### The async story

`AsyncMap` runs an operator over an iterable with bounded concurrency. If the
operator's `_apply` is `async def`, it is awaited directly; if not, it runs
in a worker thread through `asyncio.to_thread`. As proposed, `Sequential`
does not understand async: async operators are unwrapped at the `AsyncMap`
layer, and there is no `AsyncSequential`.

`AsyncMap.__call__` is synchronous: it drives its own loop with
`asyncio.run`. That makes it usable from plain scripts, but it fails inside a
running event loop, which is exactly where FastAPI and LitServe handlers run.
There, call the `async` method `run` instead:

```{code-block} python
:caption: AsyncMap inside a running event loop, from a real run.
:linenos:

async def handler(items):
    AsyncMap(Scale(2.0))(items)              # RuntimeError: asyncio.run() cannot be
                                             # called from a running event loop
    return await AsyncMap(Scale(2.0)).run(items)   # → [2.0, 4.0] for [1, 2]
```

An async operator awaited by `AsyncMap` goes through `_apply` directly,
skipping `__call__`. It therefore skips the post-apply hook stub, which will
matter once hooks land.

### What pipekit deliberately leaves out

pipekit's parallelism is intentionally minimal, and that shipped as planned:

- **No automatic parallelism.** `Sequential` is sequential; `Graph` evaluates
  topologically in one process.
- **No backend abstraction.** The four wrappers are standard library
  (`concurrent.futures`, `asyncio`).
- **No fault tolerance** beyond `on_error="log_and_continue"`. Retries of
  whole runs, dead-letter queues and durable state are the orchestrator's
  job.
- **No distributed execution.** Dask, Ray, Beam and Flink are downstream.
  Operators are pickleable and stateless when they need to be; scheduling
  them across a cluster is someone else's problem (see the use cases in
  [Report 9](09_use_cases.md)).

Every parallel case in the
[use-case gallery](https://github.com/jejjohnson/research_journal_v2/blob/main/notes/geotoolz/plans/geotoolz/examples/usecases.md)
wires up its own combination of these primitives: ETL with `n_workers=8`,
FastAPI with thread-pool I/O, LitServe with batched GPU inference, Prefect
orchestration. The catalog-iteration pattern those use cases repeat
(`CatalogPipeline`: a catalog, an operator, `n_workers`, `on_error`) is
domain-specific, because the catalog is a geocatalog concept, so it was meant
to live in a sister library. pipekit ships the `ProcessMap` and `ThreadMap`
it would be built on. `CatalogPipeline` itself has not shipped anywhere.

(r2-group-k)=
## Group K · Where the deferred operators went

The proposal used Group K for operators that are carrier-specific and so do
not belong in core. pipekit 0.0.2 reuses the letter: its docstring reserves
Group K for a disk-cache backend, and nothing ships under it. The deferred
list is still worth tracking, because most of it has not landed anywhere.
{numref}`tbl-r2-deferred` checks each one against the capability index of
every upstream library.

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

pipekit-array exists (0.0.2), but it ships four names (`array_namespace`,
`ConcatenateAlong`, `StackAlong`, `MeanScalar`), none of them from this list.
It is the planned home for most of the list, which makes it the biggest gap
between plan and code. [Report 3](03_sister_libraries.md) covers it.

(r2-group-l)=
## Group L · Serialisation (`serial.py`)

The proposal kept serialisation light and inline, next to
`Operator.from_state`. It shipped as its own module with six names
([](#tbl-r2-serial)):

:::{table} Group L at 0.0.2.
:label: tbl-r2-serial

| Symbol | What it does |
|---|---|
| `dumps(op)` | JSON-encode `op.state`, with sorted keys |
| `loads(s)` | rebuild through `Operator.from_state`; the defining module must already be imported |
| `register(cls)` | add a class to the sandboxed allowlist; doubles as a class decorator |
| `unregister(cls)`, `allowed()` | remove from, and list, the allowlist |
| `loads_sandboxed(s)` | `loads`, but raises `PermissionError` for anything not registered |
:::

The sandboxed loader is for user-uploaded pipelines ([use case 6](09_use_cases.md#r9-case-6)). As
proposed, it lives in core, and domain libraries register their own operators
with the allowlist.

```{code-block} python
:caption: What round-trips at 0.0.2, from a real run.
:linenos:

import pipekit as pk

pk.dumps(pk.Const(3))
# → '{"class": "Const", "config": {"value": 3}, "module": "pipekit.blocks"}'
pk.loads(pk.dumps(Scale(2.0)))                       # → Scale(factor=2.0)
pk.loads_sandboxed(pk.dumps(Scale(2.0)))             # PermissionError: not allowed
pk.register(Scale)
pk.loads_sandboxed(pk.dumps(Scale(2.0)))             # → Scale(factor=2.0)
pk.loads(pk.dumps(pk.Lambda(abs)))                   # RuntimeError: forbid_in_yaml
pk.loads(pk.dumps(pk.Try(pk.Identity(), pk.Const(0), on=(IOError,))))
# RuntimeError: config contains non-primitive values ['fallback', 'primary', ...]
```

:::{warning} Differs from the plan
- **Only flat operators round-trip.** Anything that nests an operator
  (`Sequential`, `Try`, `Cache`, `Fanout`, the parallel maps) dumps a debug
  payload that `loads` rejects. The proposal assumed heavier loaders would
  pick these up.
- **The YAML and Hydra loaders don't exist.** The proposal (and open
  question 5) put them in `pipekit[yaml]` and `pipekit[hydra]` extras.
  pipekit's `pyproject.toml` declares no optional dependencies, so neither
  extra exists. Until one does, there is no way to load a nested pipeline.
- **The registry is global.** The allowlist is one module-level dict, so
  every caller in a process shares it.
- **A stale docstring.** `dumps` says a flagged operator fails at `loads`
  with a `TypeError`; `from_state` raises `RuntimeError`.
:::

(r2-group-m)=
## Group M · State primitives (`state.py`)

Group M was added in the proposal's last revision so that the `pipekit-cycle`
sibling ([Report 10](10_pipekit_cycle.md)) could exist without pipekit itself
shipping cycle abstractions. Its two primitives shipped.

### The problem

pipekit's operators are stateless: `_apply(carrier) → carrier`, with no
memory across calls. That is right for L0 to L2 processing, where each scene
flows through on its own. It fails for L3 and L4 work:

- **data-assimilation cycles** carry state across time steps (forecast,
  analysis, forecast, …);
- **iterative inference** carries convergence state (residual, iteration
  count, learning-rate schedule);
- **cumulative aggregation** carries running state (an online mean, a
  streaming histogram);
- **recurrent models** carry a hidden state.

`Snapshot` captures values to *observe* them; it does not *thread* state
through the computation. That needs a different primitive.

### `StatefulOperator` and `CarryState`

A `StatefulOperator` implements `_apply(carrier, state) → (carrier, state)`.
A `CarryState` is the base for the state threaded through: it imposes no
fields, and gives each subclass `to_dict` / `from_dict` for checkpointing.
When a `Sequential` contains a stateful operator, it threads the state
through and returns `(carrier, state)`; stateless steps pass the state along
untouched.

Running example: a streaming mean. After $n$ values $x_1, \dots, x_n$ the
state holds the running sum and the count, and the step emits their ratio:

$$
\begin{aligned}
S_n &= \sum_{i=1}^{n} x_i = S_{n-1} + x_n , \\
\bar{x}_n &= \frac{S_n}{n} .
\end{aligned}
$$ (r2-eq-stream-mean)

The state is the pair $(S_n, n)$; the carrier out is $\bar{x}_n$
({eq}`r2-eq-stream-mean`).

```{code-block} python
:caption: A streaming mean as a stateful pipeline, from a real run.
:linenos:

from pipekit import CarryState, StatefulOperator

class MeanState(CarryState):
    def __init__(self, total: float, n: int) -> None:
        self.total = total               # Sₙ = ∑ᵢ xᵢ
        self.n = n                       # n

class StreamMean(StatefulOperator):
    def _apply(self, x, state):
        st = MeanState(state.total + x, state.n + 1)   # Sₙ = Sₙ₋₁ + xₙ;  n ← n + 1
        return st.total / st.n, st                     # x̄ₙ = Sₙ / n

pipe = Scale(1.0) | StreamMean()          # stateless step, then stateful step
x_bar, st = pipe(4.0, state=MeanState(0.0, 0))   # x̄₁ = 4.0
x_bar, st = pipe(8.0, state=st)                  # x̄₂ = 6.0
st                                               # → MeanState(total=12.0, n=2)
MeanState.from_dict(st.to_dict()) == st          # → True
pipe(4.0)         # TypeError: ... called without a `state` argument
```

### Why this lives in core

1. **`StatefulOperator` takes part in `Sequential`'s dispatch.** The logic
   that detects stateful steps and threads state lives where `Sequential`
   does. Putting the class in a sibling package would invert the dependency.
2. **Other siblings want state too.** pipekit-train's training loop threads
   optimiser, step and metrics through epochs, and amortised-inference loops
   do the same. With the base class in core, they all share it.

pipekit-cycle ships the cycle operators on top of these primitives, in a
package of its own inside the pipekit workspace.

:::{warning} Differs from the plan
- **`Graph` does not thread state.** The proposal said a `StatefulOperator`
  "composes into `Sequential` / `Graph` just like `Operator`". Only
  `Sequential` threads state. A stateful operator inside a `Graph` fails
  with `TypeError: StreamMean._apply() missing 1 required positional
  argument: 'state'`.
- **`CarryState` is not frozen.** The proposal described it as frozen and
  dataclass-shaped. It is a plain class whose fields can be reassigned, yet
  it defines `__hash__` over those fields. A mutated state therefore changes
  hash, and a state holding a list or an array is unhashable
  (`TypeError: unhashable type: 'list'`).
- **`initial_state_fn` is a convention, not plumbing.** The class attribute
  exists for pipekit-cycle's wrappers. `Sequential` never calls it, so the
  caller always supplies the first state.
:::

(r2-group-n)=
## Group N · Structural protocols (`protocols.py`), new

Not in the proposal. The model-inference wrappers in the sister libraries
(`ModelOp` and `SklearnOp` in geotoolz, `SklearnOp` in xrtoolz's sklearn
package, `JaxModelOp` in pipekit-jax) rely on the same duck-typed seams. Group N writes them down
once, as `runtime_checkable` Protocols:

- `Predictor`: anything with `predict(x)`;
- `FittableTransformer`: anything with `fit(x)` and `transform(x)`.

These are exactly the "check, not base class" use of a Protocol that
[Report 1](01_toolz_primer.md) argues for. `isinstance(model, Predictor)`
checks that the methods are present, not their signatures.

(r2-hashing)=
## Shared hashing (`hashing.py`), new

Also not in the proposal. `stable_json`, `stable_repr` and `sha256_hex` are
the content-hash primitives behind `Cache`, pipekit-train's dataset hash, and
pipekit-experiment's content-addressed registries. Before this module, each
of those carried its own copy, which could drift apart. The module is
internal: it is importable as `pipekit.hashing`, but it is not in
`pipekit.__all__`.

(r2-inventory)=
## The full inventory

[](#tbl-r2-inventory) is the canonical list of what each group ships, with
the proposed and shipped line counts side by side.

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

The proposal's exclusions all held at 0.0.2:

- **Distributed execution** (Dask, Ray, Beam, Flink): the orchestrator's
  job.
- **Automatic differentiation and JAX `jit`** as pipekit features: they
  live in a separate package, pipekit-jax. They do not clash with dual-mode
  `__call__`, as this page first said: a tracer is not a `Node`, so a
  pipeline of JAX operators already runs under `jit`, `vmap` and `grad` with
  respect to its inputs ([Report 5](05_pipekit_jax.md#r5-stage-1)).
- **Type-checking operator chains at construction**: `Sequential([op_a,
  op_b])` does not check that `op_a`'s output type matches `op_b`'s input.
  `Signature` checks shape; type checking is a stretch goal.
- **Carrier-specific operators**: see [](#r2-group-k) and
  [Report 3](03_sister_libraries.md).
- **A GUI or pipeline builder**: a possibility, not in scope.
- **Workflow scheduling** (Airflow, Prefect, Dagster): adapters belong in
  extras or downstream libraries. pipekit-experiment's DVC, Hydra and
  Metaflow adapters are the first of these.

(r2-open-questions)=
## Open questions

The proposal carried seven scoping questions, each with a lean
([](#tbl-r2-questions)). The shipped code settles five of them and leaves one
open; its lean on question 3 was overturned.

:::{table} The proposal's scoping questions, and how pipekit 0.0.2 answered them.
:label: tbl-r2-questions

| # | Question | The proposal's lean | At 0.0.2 |
|---|---|---|---|
| 1 | Name: `pipekit`, or `opkit`, `composable`, `pipekit-core`? | `pipekit` | **settled**: `pipekit` (the PyPI distribution name is revisited in [Report 16 §F](16_geo_data_stack.md#r16-ws-f)) |
| 2 | Is `ConfigMixin` always on? | yes, with `__config_mixin_auto__` and `__config_exclude__` | **settled** as leaned |
| 3 | How does `Signature` meet carriers without named dims? | `None` opts out; `summary` raises a clean error | **settled differently**: `None` opts out, `summary` prints `?` |
| 4 | How is YAML round-trip enforced? | the YAML loader rejects flagged operators; an optional `strict=True` dumper | **open**: there is no YAML loader. `from_state` already rejects flagged operators |
| 5 | Hydra-zen built in, or an extra? | an extra, `pipekit[hydra]` | **open**: no extra exists yet |
| 6 | Which of `Coalesce`, `Retry`, `Mode`, `Provenance` to promote? | ship `Coalesce` and `Retry`; defer `Mode` and `Provenance` | **settled** as leaned |
| 7 | `AsyncMap` only, or `AsyncSequential` too? | `AsyncMap` only | **settled** as leaned |
:::

The differences recorded above raise five new questions that need a decision
rather than a fix:

1. **Should `Graph` thread state?** Today `Sequential` does and `Graph`
   does not. Either `Graph` learns to, or the docs say stateful operators
   are `Sequential`-only. DA cycles with several inputs would want the
   former.
2. **Should `CarryState` be frozen?** If yes, it becomes a frozen dataclass
   and the hash is safe. If not, it should stop defining `__hash__`, since
   states routinely hold arrays.
3. **What is the nested-pipeline loader?** Flat operators round-trip;
   `Sequential`, `Try`, `Cache` and the rest do not. The choice is a JSON
   loader that recurses into nested `{"class", "config"}` payloads, or the
   YAML / Hydra extras the proposal assumed.
4. **Should `AsyncMap.__call__` work inside a running loop?** At the moment
   a server handler must know to call `.run`. An alternative is to detect a
   running loop and fail with a message that names `.run`.
5. **Should shape inference be wired through `Graph` and `Fanout`?** Only
   `Sequential` threads signatures today, so a `Graph` reports its input
   signature as its output.

(r2-todos)=
## Upcoming work

Changes whose shape is already clear ([](#tbl-r2-todos)). Each is a candidate
issue on pipekit or a sister library.

:::{table} Follow-ups from checking the proposal against pipekit 0.0.2.
:label: tbl-r2-todos

| Follow-up | Where | Why |
|---|---|---|
| Fix the `compute_output_signature` docstring: `summary` shows `?`, it does not raise | pipekit `_base/operator.py` | the docstring contradicts the behaviour |
| Fix the `dumps` docstring: a flagged operator fails with `RuntimeError`, not `TypeError` | pipekit `serial.py` | the same |
| Add `Graph.summary()` and signature propagation through `Graph` | pipekit `_base/graph.py` | proposed and not shipped |
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

*To be added:* `toolz`, the Python Array API standard, and the use-case
gallery, as entries in `references.bib`.
