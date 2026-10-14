# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Report 5 · pipekit under JAX transformations, run
#
# Regenerates every number in `docs/reports/05_pipekit_jax.md`. The running
# example is the five-band methane sensor of Reports 10, 11 and 17: a column
# enhancement α observed through transmittances y_b = exp(−k_b α) + ε_b.
#
# 1. `jax.jit`, `jax.vmap` and `jax.grad` through a pipekit `Sequential` of
#    JAX operators (stage 1);
# 2. gradients with respect to parameters held by operators: learning the
#    absorption coefficients from labelled scenes (stage 2);
# 3. pipekit's control, observation and caching operators under `jit`
#    (stage 3);
# 4. the operator-as-PyTree design the scoping report proposed, prototyped
#    (stage 4);
# 5. what pipekit-jax ships: `JaxModelOp`'s weight round trip,
#    `truncated_scan` and `DiffraxForwardModel` (stage 5).
#
# Every draw is seeded; JAX runs in float64. Each stage ends in a cell
# asserting the numbers the page shows, to the precision it shows them.
#
# Pinned revisions: pipekit 0.0.2 at `ef3c17f` (pipekit, -cycle, -experiment,
# -jax). It needs JAX 0.11, Equinox 0.13, Optax and diffrax 0.7 (the
# `pipekit-jax[diffrax]` extra); it runs in under a minute on a laptop CPU.

# %%
from __future__ import annotations

import tempfile
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pipekit_cycle as pc
from pipekit import (
    Branch,
    Cache,
    Fanout,
    Graph,
    Identity,
    Input,
    Lambda,
    Operator,
    Profile,
    Sequential,
    ShapeTrace,
    Sink,
    Snapshot,
    Switch,
    Tap,
)
from pipekit_experiment import LocalModelRegistry
from pipekit_jax import DiffraxForwardModel, JaxModelOp, truncated_scan

jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## Stage 1 · jit, vmap and grad through a pipeline
#
# The retrieval inverts each band, z_b = −log y_b / k_b, and averages the five
# with inverse-variance weights. To first order var(z_b) = σ² / (k_b y_b)², so
# w_b ∝ (k_b y_b)² and α̂ = Σ_b w_b z_b / Σ_b w_b.

# %%
K = jnp.array([0.002, 0.004, 0.006, 0.008, 0.010])
SIG, A_MAX, N_TEST, N_TRAIN = 1e-3, 300.0, 500, 2_000
TEST_SEED, TRAIN_SEED = 10_000, 20_000


def simulate(n: int, seed: int, k: jax.Array = K) -> tuple[jax.Array, jax.Array]:
    # α ~ 𝒰(0, 300), y_b = exp(−k_b α) + ε_b, ε_b ~ 𝒩(0, σ²)     () → (n,), (n, B)
    ka, ke = jax.random.split(jax.random.key(seed))
    alpha = jax.random.uniform(ka, (n,), minval=0.0, maxval=A_MAX)
    y = jnp.exp(-alpha[:, None] * k) + SIG * jax.random.normal(ke, (n, k.size))
    return alpha, jnp.clip(y, 1e-6, None)


def band_inversions(k: jax.Array) -> Operator:
    # z_b = −log y_b / k_b                                         (B,) → (B,)
    return Lambda(lambda y: -jnp.log(y) / k)


def weighted_mean(k: jax.Array) -> Operator:
    # α̂ = Σ_b w_b z_b / Σ_b w_b,  w_b = (k_b exp(−k_b z_b))²          (B,) → ()
    def apply(z):
        w = (k * jnp.exp(-k * z)) ** 2
        return (w * z).sum(-1) / w.sum(-1)

    return Lambda(apply)


retrieval = Sequential([band_inversions(K), weighted_mean(K)])
alpha_test, y_test = simulate(N_TEST, TEST_SEED)
y0 = y_test[0]

eager = retrieval(y0)
jitted = jax.jit(retrieval)(y0)
batched = jax.vmap(retrieval)(y_test)                      # (n, B) → (n,)
grad_y = jax.grad(retrieval)(y0)                           # ∂α̂/∂y_b, (B,)
n_eqns = len(jax.make_jaxpr(retrieval)(y0).jaxpr.eqns)
# σ_α ≈ σ ‖∇_y α̂‖: the gradient is the per-scene error bar
sig_lin = SIG * jnp.linalg.norm(jax.vmap(jax.grad(retrieval))(y_test), axis=1)
stage1 = {
    "jit_minus_eager": float(abs(jitted - eager)),
    "rmse": float(jnp.sqrt(jnp.mean((batched - alpha_test) ** 2))),
    "rms_sig_lin": float(jnp.sqrt(jnp.mean(sig_lin**2))),
    "z_rms": float(jnp.sqrt(jnp.mean(((batched - alpha_test) / sig_lin) ** 2))),
    "n_eqns": n_eqns,
    "grad_sign": bool((grad_y < 0).all()),
    "alpha0": float(alpha_test[0]), "alpha_hat0": float(eager),
}
stage1

# %%
assert stage1["jit_minus_eager"] < 1e-12  # one jaxpr, the same arithmetic
assert stage1["n_eqns"] == 12 and stage1["grad_sign"]  # more absorption, less y
assert round(stage1["rmse"], 3) == 0.262
assert round(stage1["rms_sig_lin"], 3) == 0.258
# the linearised error bar is calibrated: z = (α̂ − α) / σ_α has RMS 1 to the
# sampling error of 500 scenes (the RMS of 500 unit normals is 1 ± 0.03)
assert abs(stage1["z_rms"] - 1) < 3 * 0.032 and round(stage1["z_rms"], 2) == 0.97

# %% [markdown]
# ## Stage 2 · Gradients with respect to an operator's parameters
#
# The coefficients are known only to 10 % (say, from an old line list), and
# 2,000 scenes with known α are available (controlled releases). Learning k
# means differentiating the loss with respect to a parameter held by an
# operator inside the pipeline.

# %%
class BandInversion(eqx.Module):  # the learnable step, as an Equinox module
    log_k: jax.Array

    def __call__(self, y: jax.Array) -> jax.Array:
        # z_b = −log y_b / k_b, k = exp(log k)                       (B,) → (B,)
        return -jnp.log(y) / jnp.exp(self.log_k)


K_START = K * jnp.array([1.10, 0.92, 1.07, 0.95, 1.08])  # 5 to 10 % off
alpha_train, y_train = simulate(N_TRAIN, TRAIN_SEED)


def pipeline(m: BandInversion) -> Sequential:
    k = jnp.exp(m.log_k)
    return Sequential([JaxModelOp(m), weighted_mean(k)])


def loss_of_pipeline(p: Sequential, y, a) -> jax.Array:
    # L = mean (α̂ − α)²                                          (n, B) → ()
    return jnp.mean((jax.vmap(p)(y) - a) ** 2)


start = BandInversion(jnp.log(K_START))
# (a) the pipeline as the differentiated argument: it is one opaque leaf
pipe0 = pipeline(start)
leaves = jax.tree.leaves(pipe0)
grad_opaque = eqx.filter_grad(loss_of_pipeline)(pipe0, y_train, alpha_train)


# (b) the module as the differentiated argument; the pipeline is rebuilt
# inside the loss, so the gradient reaches log k through it
def loss_of_module(m: BandInversion, y, a) -> jax.Array:
    return loss_of_pipeline(pipeline(m), y, a)


opt = optax.adam(1e-2)


@eqx.filter_jit
def step(m, state):
    loss, g = eqx.filter_value_and_grad(loss_of_module)(m, y_train, alpha_train)
    updates, state = opt.update(g, state, m)
    return eqx.apply_updates(m, updates), state, loss


m, state = start, opt.init(start)
losses = []
for _ in range(400):
    m, state, loss = step(m, state)
    losses.append(float(loss))
k_hat = jnp.exp(m.log_k)


def test_rmse(k: jax.Array) -> float:
    return float(jnp.sqrt(jnp.mean((jax.vmap(pipeline(BandInversion(jnp.log(k))))(y_test)
                                    - alpha_test) ** 2)))


stage2 = {
    "n_leaves": len(leaves), "leaf_type": type(leaves[0]).__name__,
    "grad_opaque": grad_opaque,
    "k_err_start": float(jnp.max(jnp.abs(K_START / K - 1))),
    "k_err_end": float(jnp.max(jnp.abs(k_hat / K - 1))),
    "rmse_start": test_rmse(K_START), "rmse_end": test_rmse(k_hat),
    "rmse_true_k": stage1["rmse"],
}
stage2

# %%
# the whole Sequential is one leaf, so there is nothing to differentiate
assert stage2["n_leaves"] == 1 and stage2["leaf_type"] == "Sequential"
assert stage2["grad_opaque"] is None
assert round(stage2["k_err_start"], 2) == 0.10
assert round(stage2["rmse_start"], 2) == 3.45  # 13 times the true-k error
# 400 Adam steps bring the test RMSE to the true-k value (0.260 against
# 0.262) while the coefficients stay up to 9 % off: labelled α constrain the
# retrieval, not each band's k
assert round(stage2["rmse_end"], 3) == 0.260 and stage2["rmse_end"] < stage2["rmse_true_k"]
assert round(stage2["k_err_end"], 3) == 0.087
assert losses[-1] < losses[0] / 100

# %% [markdown]
# ## Stage 3 · Control, observation and caching operators under jit
#
# Each operator runs inside a jitted pipeline three times: twice on one input,
# once on another of the same shape. The question is what it does at trace
# time and what survives into the compiled function.

# %%
y1, y2 = y_test[1], y_test[2]
outcome: dict[str, str] = {}


def err(fn) -> str:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — the exception type is the finding
        return type(e).__name__
    return "ok"


# Tap: the callback sees a tracer, once
seen: list[str] = []
f = jax.jit(Sequential([Tap(lambda x: seen.append(type(x).__name__)), retrieval]))
f(y0), f(y0), f(y1)
outcome["Tap"] = f"{len(seen)} call, {seen[0]}"

# ShapeTrace: prints at trace time only (its print is the point)
n_print = []
st = ShapeTrace()
st._apply = (lambda orig: lambda x: (n_print.append(1), orig(x))[1])(st._apply)
g = jax.jit(Sequential([st, retrieval]))
g(y0), g(y1)
outcome["ShapeTrace"] = f"{len(n_print)} print"

# Profile: one timing, the trace's
prof = Profile()
h = jax.jit(Sequential([prof.wrap(band_inversions(K)), weighted_mean(K)]))
for y in (y0, y0, y1):
    h(y).block_until_ready()
outcome["Profile"] = f"{len(prof.timings['Lambda'])} timing"

# Snapshot: captures a tracer, which escapes the trace
snap = Snapshot()
jax.jit(Sequential([band_inversions(K), snap.at("z"), weighted_mean(K)]))(y0)
outcome["Snapshot"] = (f"{type(snap.captures['z']).__name__}; using it raises "
                       + err(lambda: snap.captures["z"] + 1))

# Sink: writes once, a tracer
writes: list[str] = []
s = jax.jit(Sequential([retrieval, Sink(lambda x: writes.append(type(x).__name__))]))
s(y0), s(y1), s(y2)
outcome["Sink"] = f"{len(writes)} write, {writes[0]}"

# Branch and Switch: a Python decision on a traced value
br = Branch(predicate=lambda y: y.min() < 0.5, if_true=retrieval,
            if_false=Sequential([Lambda(lambda y: -jnp.log(y[-1]) / K[-1])]))
sw = Switch(key=lambda y: int(y.min() < 0.5), cases={1: retrieval})
outcome["Branch"] = err(lambda: jax.jit(br)(y0))
outcome["Switch"] = err(lambda: jax.jit(sw)(y0))

# Cache: every tracer of one shape has the same key
cache = Cache(Lambda(lambda x: x + 1.0))
twice = Sequential([cache, Lambda(lambda x: 10.0 * x), cache])
x5 = jnp.arange(1.0, 6.0)
cache_eager = np.asarray(twice(np.asarray(x5)))
cache.clear()
cache_jit = np.asarray(jax.jit(twice)(x5))
cache2 = Cache(Lambda(lambda x: x + 1.0))
jax.jit(cache2)(x5)
outcome["Cache"] = (f"jit gives {cache_jit[0]:.0f} for {cache_eager[0]:.0f}; a later "
                    "trace raises " + err(lambda: jax.jit(lambda x: cache2(x) * 1.0)(x5)))

# Fanout and a diamond Graph: pure composition
fan = Fanout({"z": band_inversions(K), "alpha": retrieval})
outcome["Fanout"] = err(lambda: jax.jit(fan)(y0))
xin = Input("y")
z = band_inversions(K)(xin)
a_hat = weighted_mean(K)(z)
resid = Lambda(lambda z, a: z - a)(z, a_hat)   # z feeds two nodes: a diamond
diamond = Graph(inputs={"y": xin}, outputs={"alpha": a_hat, "resid": resid})
dj = jax.jit(diamond)(y0)
dv = jax.vmap(diamond)(y_test)
outcome["Graph"] = err(lambda: jax.jit(diamond)(y0))

# the JAX-native replacements
calls: list[int] = []
dbg = jax.jit(Sequential([Lambda(lambda y: (jax.debug.callback(lambda: calls.append(1)),
                                            y)[1]), retrieval]))
dbg(y0), dbg(y0), dbg(y1)
jax.effects_barrier()


class CondBranch(Operator):  # a Branch whose decision is lax.cond     user
    def __init__(self, predicate, if_true, if_false) -> None:
        self.predicate, self.if_true, self.if_false = predicate, if_true, if_false

    def _apply(self, x):
        return jax.lax.cond(self.predicate(x), self.if_true, self.if_false, x)


cb = CondBranch(lambda y: y.min() < 0.5, retrieval,
                Lambda(lambda y: -jnp.log(y[-1]) / K[-1]))
cond_vs_eager = float(abs(jax.jit(cb)(y0) - br(y0)))
stage3 = {"outcome": outcome, "debug_callbacks": len(calls),
          "cond_vs_eager": cond_vs_eager,
          "diamond_close": float(jnp.abs(dj["alpha"] - eager).max()),
          "diamond_vmap_shape": tuple(dv["resid"].shape)}
stage3

# %%
assert outcome == {
    "Tap": "1 call, DynamicJaxprTracer",
    "ShapeTrace": "1 print",
    "Profile": "1 timing",
    "Snapshot": "DynamicJaxprTracer; using it raises UnexpectedTracerError",
    "Sink": "1 write, DynamicJaxprTracer",
    "Branch": "TracerBoolConversionError",
    "Switch": "ConcretizationTypeError",
    "Cache": "jit gives 2 for 21; a later trace raises UnexpectedTracerError",
    "Fanout": "ok",
    "Graph": "ok",
}
# TracerBoolConversionError is a ConcretizationTypeError: the same cause
assert issubclass(jax.errors.TracerBoolConversionError, jax.errors.ConcretizationTypeError)
assert stage3["debug_callbacks"] == 3  # once per call, not once per trace
assert stage3["cond_vs_eager"] < 1e-12
assert stage3["diamond_close"] < 1e-12 and stage3["diamond_vmap_shape"] == (N_TEST, 5)

# %% [markdown]
# ## Stage 4 · Operators as PyTrees, prototyped
#
# The scoping report's design: `JaxOperator(Operator, eqx.Module)` and a
# `JaxSequential` whose operators are a tuple field. Both are prototyped here
# in a few lines, to see what the multiple inheritance gives and costs.

# %%
class JaxOperator(Operator, eqx.Module):  # proposed (B1)
    pass


class Invert(JaxOperator):
    log_k: jax.Array

    def _apply(self, y):
        return -jnp.log(y) / jnp.exp(self.log_k)


class Combine(JaxOperator):
    log_k: jax.Array

    def _apply(self, z):
        k = jnp.exp(self.log_k)
        w = (k * jnp.exp(-k * z)) ** 2
        return (w * z).sum(-1) / w.sum(-1)


class JaxSequential(Sequential, eqx.Module):  # proposed (B1)
    operators: tuple

    def __init__(self, operators) -> None:
        self.operators = tuple(operators)


inv, comb = Invert(jnp.log(K_START)), Combine(jnp.log(K_START))
mro = [c.__name__ for c in Invert.__mro__[:5]]
plain = Sequential([inv, comb])
jseq = JaxSequential([inv, comb])
g_op = eqx.filter_grad(lambda o, y: o(y).sum())(inv, y0)
g_plain = eqx.filter_grad(loss_of_pipeline)(plain, y_train, alpha_train)
g_jseq = eqx.filter_grad(loss_of_pipeline)(jseq, y_train, alpha_train)
g_ref = eqx.filter_grad(loss_of_module)(BandInversion(jnp.log(K_START)), y_train,
                                        alpha_train)
# tree surgery: swap the first operator for the true coefficients
swapped = eqx.tree_at(lambda s: s.operators[0], jseq, Invert(jnp.log(K)))
cfg = inv.get_config()
stage4 = {
    "mro": mro, "op_leaves": len(jax.tree.leaves(inv)),
    "op_grad_shape": tuple(g_op.log_k.shape), "plain_grad": g_plain,
    "jseq_leaves": len(jax.tree.leaves(jseq)),
    # the module of stage 2 ties one log k to both steps; as operators they are
    # two leaves, and the tied gradient is the sum of the two
    "jseq_vs_module_grad": float(jnp.abs(g_jseq.operators[0].log_k
                                         + g_jseq.operators[1].log_k
                                         - g_ref.log_k).max()),
    "swapped_changes_output": float(abs(swapped(y0) - jseq(y0))) > 1.0,
    "config_value_type": type(cfg["log_k"]).__name__,
    "state_error": err(lambda: __import__("json").dumps(inv.state)),
    "jseq_summary": err(lambda: jseq.summary(None)),
}
stage4

# %%
assert stage4["mro"] == ["Invert", "JaxOperator", "Operator", "ConfigMixin", "Module"]
assert stage4["op_leaves"] == 1 and stage4["op_grad_shape"] == (5,)
assert stage4["plain_grad"] is None   # a plain Sequential stays one opaque leaf
assert stage4["jseq_leaves"] == 2     # a JaxSequential exposes both operators
# the two parameters are separate leaves; their gradients sum to the tied one
assert stage4["jseq_vs_module_grad"] < 1e-9
assert stage4["swapped_changes_output"]
# the cost: auto-derived config now holds a JAX array, so state is not JSON
assert stage4["config_value_type"] == "ArrayImpl"
assert stage4["state_error"] == "TypeError"
assert stage4["jseq_summary"] == "ok"

# %% [markdown]
# ## Stage 5 · What pipekit-jax ships
#
# (a) `JaxModelOp`'s weight round trip through a registry; (b) `truncated_scan`,
# a scan whose carry is cut from the gradient except in the last k steps;
# (c) `DiffraxForwardModel`, a pipekit-cycle forward model on a diffrax solve,
# with its adjoint chosen by a spec.

# %%
root = Path(tempfile.mkdtemp(prefix="report05_"))
reg = LocalModelRegistry(root / "reg")
trained = JaxModelOp(m)
h = reg.store(trained, weights=trained.serialize_weights(), tags={"role": "inversion"})
template = JaxModelOp(BandInversion(jnp.zeros(5)))
reloaded = template.from_registry(reg, h)
same_bytes = reloaded.serialize_weights() == trained.serialize_weights()
reload_err = float(jnp.abs(jnp.exp(reloaded.module.log_k) - k_hat).max())


# (b) a plume column decaying and refilled each hour: c_{t+1} = a c_t + q_t
def decay(c, q):
    c_new = 0.9 * c + q
    return c_new, c_new


q = jnp.ones(24)
grads = {}
for k_trunc in (3, 24):
    def final(q, k=k_trunc):
        c_T, _ = truncated_scan(decay, jnp.array(0.0), q, k=k)
        return c_T
    grads[k_trunc] = jax.grad(final)(q)
forward_same = float(abs(truncated_scan(decay, 0.0, q, k=3)[0]
                         - jax.lax.scan(decay, 0.0, q)[0]))

# (c) first-order loss of the column, dc/dt = −λ c, over one hour
LAM = 0.25
ode = DiffraxForwardModel(vector_field=lambda t, c, args: -LAM * c, dt=1.0, dt0=0.01)
c1 = ode.step(jnp.array(100.0), 1.0)
dc1 = {type(a).__name__: float(jax.grad(lambda c0, mm=mm: mm.step(c0, 1.0))(100.0))
       for a, mm in [(ode.adjoint, ode),
                     (pc.DirectAdjoint(), ode.with_adjoint(pc.DirectAdjoint())),
                     (pc.BacksolveAdjoint(), ode.with_adjoint(pc.BacksolveAdjoint()))]}
stage5 = {
    "same_bytes": same_bytes, "reload_err": reload_err,
    "trunc_nonzero": {k: int((jnp.abs(v) > 0).sum()) for k, v in grads.items()},
    "trunc_first": float(grads[24][0]), "forward_same": forward_same,
    "ode_err": float(abs(c1 - 100.0 * np.exp(-LAM))),
    "adjoint_grads": dc1,
    "is_forward_model": isinstance(ode, pc.ForwardModel),
    "supports_adjoint": isinstance(ode, pc.SupportsAdjoint),
    "truncated_spec": err(lambda: ode.with_adjoint(pc.TruncatedAdjoint(k=3))
                          .step(jnp.array(1.0), 1.0)),
}
stage5

# %%
assert stage5["same_bytes"] and stage5["reload_err"] == 0.0
# k = 3 keeps the last three steps' gradients through the carry; k = T is a
# plain scan, where ∂c_T/∂q_t = 0.9^(T−1−t) reaches back to the first hour
assert stage5["trunc_nonzero"] == {3: 3, 24: 24}
assert round(stage5["trunc_first"], 4) == round(0.9**23, 4) == 0.0886
assert stage5["forward_same"] == 0.0  # truncation changes gradients only
assert stage5["ode_err"] < 1e-12  # Tsit5 at dt0 = 0.01 against exp(−λ)
assert all(abs(v - np.exp(-LAM)) < 1e-6 for v in stage5["adjoint_grads"].values())
assert sorted(stage5["adjoint_grads"]) == ["BacksolveAdjoint", "DirectAdjoint",
                                           "RecursiveCheckpointAdjoint"]
assert stage5["is_forward_model"] and stage5["supports_adjoint"]
assert stage5["truncated_spec"] == "ValueError"  # truncation is a rollout's choice


# %% [markdown]
# ## The readiness figure
#
# The audit's readiness figure counts the markers on this page's own sketches
# (`today`, `proposed (…)`, `user`); `docs/images/reports/make_figures.py`
# draws it. These are the counts its caption quotes.

# %%
import importlib.util
import sys
from collections import Counter

root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
spec = importlib.util.spec_from_file_location(
    "report_figures", root / "docs/images/reports/make_figures.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["report_figures"] = rf
spec.loader.exec_module(rf)
calls = rf.marked_calls(next((root / "docs/reports").glob("05_*.md")))
ready = Counter(c.status for c in calls)
ready

# %%
assert len(calls) == 41
assert ready == Counter({'here': 15, 'today': 24, 'user': 2})
