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
# # Report 11 · pipekit-train, run
#
# Regenerates every number in `docs/reports/11_pipekit_train.md`:
#
# 1. an emulator of the advection operator M = 0.6 I + 0.4 S learned from a
#    `SimulationDataset`, on a white and a smooth prior, then used as a
#    `NeuralForward` in a `Cycle` (stages 1 to 4);
# 2. the callbacks, checkpointing and resume of that loop (stage 5);
# 3. an amortized Beer–Lambert retrieval trained with `NLL` (stage 6);
# 4. the same retrieval's posterior from three Bayesian backends (stage 7);
# 5. the dataset hashes, `CachedDataset` and `XarrayWindowDataset` (stage 8);
# 6. a `JaxModelOp` through the Equinox loop and the registry.
#
# Every draw is seeded; outputs go to a temporary directory. Each part ends in
# a cell asserting the values the page shows, to the precision it shows them.
# Wall times are printed, not asserted: they depend on the machine.
#
# Pinned revision: pipekit 0.0.2 at `ef3c17f` (pipekit-train, -cycle,
# -experiment, -jax). It needs JAX, Equinox, Optax, NumPyro, BlackJAX, xarray,
# zarr and geopatcher (geotoolz `088475e`). The whole notebook takes about fifteen
# minutes on a laptop CPU, most of it the stage-6 network.

# %%
from __future__ import annotations

import importlib
import sys
import tempfile
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pipekit_cycle as pc
import xarray as xr
from pipekit import Operator
from pipekit_experiment import LocalModelRegistry
from pipekit_jax import JaxModelOp
from pipekit_train import (
    MSE,
    NLL,
    CachedDataset,
    Checkpoint,
    EarlyStopping,
    IterableDataset,
    JSONLWriter,
    LogToExperiment,
    SimulationDataset,
    TrainingDataset,
    TrainingLoop,
)
from pipekit_train.adapters.bayes import NumpyroTask
from pipekit_train.adapters.blackjax import BlackjaxTask
from pipekit_train.adapters.equinox import EquinoxModelOp

root = Path(tempfile.mkdtemp(prefix="report11_"))
N = 40
S = np.roll(np.eye(N), 1, axis=0)
M = 0.6 * np.eye(N) + 0.4 * S


class SmoothPrior(Operator):
    """x is a sum of periodic Gaussian bumps with random centre, width, amplitude."""

    def __init__(self, n: int = N, n_bumps: int = 3) -> None:
        self.n = n
        self.n_bumps = n_bumps

    def _apply(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        i = np.arange(self.n)
        x = np.zeros(self.n)
        for _ in range(self.n_bumps):
            c = rng.uniform(0, self.n)
            w = rng.uniform(1.5, 6.0)
            a = rng.normal()
            d = np.minimum(np.abs(i - c), self.n - np.abs(i - c))
            x += a * np.exp(-0.5 * (d / w) ** 2)
        return x.astype(np.float32)


class WhitePrior(Operator):
    def _apply(self, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).normal(size=N).astype(np.float32)


# %% [markdown]
# ## Part 1 · The emulator on two priors (stages 1 to 5)

# %%
class MemTracker:
    """An in-memory ExperimentTracker: start_run, log_metrics, end_run."""

    def __init__(self) -> None:
        self.events: list[tuple] = []

    def start_run(self, name, config):
        self.events.append(("start", name, sorted(config)))
        return {"name": name, "id": "run-0"}

    def log_metrics(self, run, metrics, step):
        self.events.append(("log", step, dict(metrics)))

    def end_run(self, run, status):
        self.events.append(("end", status))


class EvalRecorder:
    def __init__(self) -> None:
        self.evals: list[tuple] = []

    def on_eval_end(self, loop, state, m):
        self.evals.append((state.step, dict(m)))


fwd = pc.CallableForward(lambda x, dt: (M @ x).astype(np.float32), dt=1.0)
true_fwd = pc.CallableForward(lambda x, dt: M @ x, dt=1.0)


def emulate(name: str, prior: Operator) -> dict:
    out = root / f"emu_{name}"
    train = SimulationDataset(fwd, prior, n_samples=2000, seed=0, split="train")
    val = train.with_split("val")
    model = eqx.nn.Linear(N, N, use_bias=False, key=jax.random.key(0))
    tracker, rec = MemTracker(), EvalRecorder()
    reg = LocalModelRegistry(str(out / "registry"))
    es = EarlyStopping(metric="mse", patience=5, mode="min", min_delta=1e-9)
    ckpt = Checkpoint(every_n_steps=250, keep_last=2, save_dir="unused",
                      registry=reg)
    loop = TrainingLoop(
        model_op=EquinoxModelOp(model), dataset=train, val_dataset=val,
        loss=MSE(), optimizer_config={"name": "adam", "lr": 1e-2},
        max_steps=6000, steps_per_epoch=25, batch_size=64,
        eval_every_n_steps=50, log_every_n_steps=50,
        callbacks=(rec, es, ckpt, LogToExperiment(tracker, run_name="emu")),
        metric_writer=JSONLWriter(str(out / "metrics.jsonl")),
        checkpoint_dir=str(out / "ckpt"), seed=0,
    )
    t0 = time.time()
    trained, art = loop.run()
    r = {"wall_s": time.time() - t0, "len": (len(train), len(val))}
    W = np.asarray(trained.module.weight)
    X = np.stack([train[i][0] for i in range(len(train))])
    sv = np.linalg.svd(X, compute_uv=False)
    r |= {
        "final_step": art.backend_info["final_step"],
        "val_mse": rec.evals[-1][1]["mse"],
        "W_minus_M_max": float(np.abs(W - M).max()),
        "sv_ratio": float(sv[-1] / sv[0]),
        "ckpt_dirs": sorted(p.name for p in (out / "ckpt").iterdir()),
        "jsonl_lines": sum(1 for _ in open(out / "metrics.jsonl")),
        "tracker_first": tracker.events[0][:2],
        "tracker_last": tracker.events[-1],
        "eval_logged": sum(1 for e in tracker.events if e[0] == "log"
                           and any(k.startswith("eval/") for k in e[2])),
        "dataset_hash": art.dataset_hash,
        "model_registry_uri": art.model_registry_uri,
        "registry_weights": reg.load_weights(art.trained_model_hash),
    }
    try:
        reg.load(art.trained_model_hash)
        r["registry_load"] = "ok"
    except RuntimeError as e:
        r["registry_load"] = f"RuntimeError: {e}"
    # stage 4: the trained operator as a forward model in a 6-step Cycle
    nf = pc.NeuralForward(model_op=trained, dt=1.0)
    for x0name, x0 in {"smooth": SmoothPrior()(seed=12345),
                       "white": np.random.default_rng(7).normal(size=N)}.items():
        xn, _ = pc.Cycle(step_op=nf, n_steps=6)(jnp.asarray(x0, jnp.float32), None)
        xt, _ = pc.Cycle(step_op=true_fwd, n_steps=6)(np.float64(x0), None)
        r[f"cycle6_{x0name}"] = float(np.abs(np.asarray(xn) - xt).max())
    # stage 5: rerun the same loop on the same checkpoint_dir resumes
    es.reset()
    _, art2 = loop.run()
    r["resume_final_step"] = art2.backend_info["final_step"]
    return r


emu = {"white": emulate("white", WhitePrior()),
       "smooth": emulate("smooth", SmoothPrior())}
{k: {kk: v[kk] for kk in ("final_step", "val_mse", "W_minus_M_max", "wall_s")}
 for k, v in emu.items()}

# %%
# stage 1: Cycle is a StatefulOperator, so a dataset step cannot call it bare
try:
    pc.Cycle(true_fwd, n_steps=2)(np.ones(N))
    cycle_bare = ""
except TypeError as e:
    cycle_bare = str(e)
cycle_bare

# %%
w, s = emu["white"], emu["smooth"]
assert w["len"] == (1600, 200)
assert "missing 1 required positional argument: 'state'" in cycle_bare
assert w["final_step"] == 500 and s["final_step"] == 2200
assert f"{w['val_mse']:.1e}" == "5.3e-14" and f"{s['val_mse']:.1e}" == "3.4e-05"
assert f"{w['W_minus_M_max']:.1e}" == "3.8e-07"
assert round(s["W_minus_M_max"], 3) == 0.343
assert f"{s['sv_ratio']:.1e}" == "1.4e-05" and w["sv_ratio"] > 0.5
assert f"{w['cycle6_smooth']:.1e}" == "1.1e-06"
assert f"{w['cycle6_white']:.1e}" == "1.3e-06"
assert round(s["cycle6_smooth"], 3) == 0.017
assert round(s["cycle6_white"], 3) == 0.533
assert s["ckpt_dirs"] == ["1750", "2000"]  # the final step 2200 is not saved
assert s["jsonl_lines"] == 44 and s["eval_logged"] == 44
assert s["tracker_first"] == ("start", "emu")
assert s["tracker_last"] == ("end", "completed")
assert s["dataset_hash"] and s["model_registry_uri"]
assert s["registry_weights"] is None
assert s["registry_load"].startswith("RuntimeError: from_state cannot reconstruct "
                                     "EquinoxModelOp")
assert "forbid_in_yaml" in s["registry_load"]
assert s["resume_final_step"] == 2600  # resumed at 2000, stopped 600 steps on

# %% [markdown]
# ## Part 2 · Amortized Beer–Lambert retrieval with `NLL` (stage 6)
#
# θ = (α, ε₁…ε₅) with α ~ U(0, 300); y_b = exp(−k_b α) + σ ε_b, σ = 1e-3.
# The network maps y to a Gaussian head q(α | y).

# %%
K = np.array([0.002, 0.004, 0.006, 0.008, 0.010], np.float32)
SIG = 1e-3


class AlphaNoisePrior(Operator):
    """params = [alpha, eps_1..eps_5]; the noise is drawn jointly, so it is seeded."""

    def _apply(self, seed: int = 0) -> np.ndarray:
        r = np.random.default_rng(seed)
        theta = np.concatenate([[r.uniform(0, 300)], r.normal(size=5)])
        return theta.astype(np.float32)


def beer_lambert(p, dt):
    return (np.exp(-K * p[0]) + SIG * p[1:]).astype(np.float32)


class Flip(TrainingDataset):
    """(params, y) → (y, alpha): the retrieval direction."""

    __config_mixin_auto__ = False

    def __init__(self, base) -> None:
        super().__init__(seed=base.seed, split=base.split)
        self.base = base

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i):
        p, y = self.base[i]
        return y, p[:1]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def with_split(self, split):
        return Flip(self.base.with_split(split))

    def content_hash(self) -> str:
        return "flip-" + self.base.content_hash()


class GaussianHead(eqx.Module):
    mean: jax.Array
    log_var: jax.Array

    def log_prob(self, t):
        return -0.5 * (jnp.log(2 * jnp.pi) + self.log_var
                       + (t - self.mean) ** 2 / jnp.exp(self.log_var))


class Net(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, key) -> None:
        self.mlp = eqx.nn.MLP(5, 2, 64, 3, activation=jax.nn.gelu, key=key)

    def __call__(self, y):
        o = self.mlp((y - 0.55) / 0.3)  # rough input standardisation
        return GaussianHead(mean=150.0 + 150.0 * o[:1], log_var=2.0 * o[1:] + 2.0)


sim = SimulationDataset(pc.CallableForward(beer_lambert, dt=1.0),
                        AlphaNoisePrior(), n_samples=20_000, seed=0)
train, test = Flip(sim), Flip(sim.with_split("test"))
loop = TrainingLoop(
    model_op=EquinoxModelOp(Net(jax.random.key(0))), dataset=train,
    val_dataset=train.with_split("val"), loss=NLL(),
    optimizer_config={"name": "adam", "lr": 2e-3}, max_steps=8000,
    batch_size=256, eval_every_n_steps=1000, log_every_n_steps=1000, seed=0,
)
t0 = time.time()
q, _ = loop.run()
retrieval_wall = time.time() - t0

Y = np.stack([test[i][0] for i in range(200)])
A = np.stack([test[i][1] for i in range(200)])[:, 0]
head = jax.vmap(q.module)(jnp.asarray(Y))
mu = np.asarray(head.mean)[:, 0]
sd = np.sqrt(np.exp(np.asarray(head.log_var)))[:, 0]
err = mu - A


def sigma_cr(alpha: np.ndarray) -> np.ndarray:
    """Cramér–Rao bound σ / ‖J(α)‖ with J_b = −k_b exp(−k_b α)."""
    J = K[None].astype(np.float64) * np.exp(-K[None] * alpha[:, None])
    return SIG / np.sqrt((J**2).sum(1))


grid = np.linspace(0.0, 300.0, 300_001)  # the U(0, 300) prior, by quadrature
cr_prior_mean = float(sigma_cr(grid).mean())
cr_prior_rms = float(np.sqrt((sigma_cr(grid) ** 2).mean()))  # the RMSE floor
cr_test_rms = float(np.sqrt((sigma_cr(A) ** 2).mean()))
retrieval = {
    "rmse": float(np.sqrt(np.mean(err**2))), "bias": float(err.mean()),
    "coverage_2sigma": float(np.mean(np.abs(err) <= 2 * sd)),
    "mean_z2": float(np.mean((err / sd) ** 2)), "mean_sigma": float(sd.mean()),
    "sigma_low": float(sd[A < 100].mean()), "sigma_high": float(sd[A > 200].mean()),
    "cr_prior_mean": cr_prior_mean, "cr_prior_rms": cr_prior_rms,
    "cr_test_rms": cr_test_rms, "wall_s": retrieval_wall,
}
retrieval

# %%
r = retrieval
assert round(r["rmse"], 3) == 0.303 and round(r["bias"], 2) == -0.15
assert round(r["coverage_2sigma"], 3) == 0.995 and round(r["mean_z2"], 2) == 0.43
assert round(r["sigma_low"], 2) == 0.33 and round(r["sigma_high"], 2) == 0.74
assert round(r["mean_sigma"], 2) == 0.46
assert round(cr_prior_mean, 3) == 0.228 and round(cr_prior_rms, 3) == 0.257
cr_test_mean = float(sigma_cr(A).mean())
assert round(cr_test_rms, 3) == 0.253 and round(cr_test_mean, 3) == 0.227

# %% [markdown]
# ## Part 3 · The same posterior from three Bayesian backends (stage 7)

# %%
k = K
y = (np.exp(-k * 120.0) + np.random.default_rng(3).normal(0, SIG, 5)).astype(
    np.float32)
data = IterableDataset(source=list(zip(k[:, None], y, strict=True)),
                       content_hash="beer-lambert-y3")


def model(x, y=None):
    a = numpyro.sample("alpha", dist.Uniform(0.0, 300.0))
    numpyro.sample("obs", dist.Normal(jnp.exp(-x[..., 0] * a), SIG), obs=y)


class Dummy(Operator):
    def _apply(self, x):
        return x


laplace_sd = float(SIG / np.linalg.norm(-k * np.exp(-k * 120.0)))
task = NumpyroTask(model, num_warmup=1000, num_samples=4000)
bayes = {}
for backend, t in [("numpyro-mcmc", task),
                   ("blackjax", BlackjaxTask(numpyro_task=task)),
                   ("numpyro-svi", task)]:
    t0 = time.time()
    op, art = TrainingLoop(model_op=Dummy(), dataset=data, task=t,
                           backend=backend, max_steps=4000, seed=0,
                           optimizer_config={"name": "adam", "lr": 1e-2}).run()
    post = op.posterior
    if backend == "numpyro-svi":
        bayes[backend] = {kk: np.asarray(v).tolist() for kk, v in post.items()}
    else:
        a = np.asarray(post["alpha"])
        bayes[backend] = {"mean": float(a.mean()), "sd": float(a.std()),
                          "divergences": art.backend_info.get("num_divergences")}
    bayes[backend]["wall_s"] = time.time() - t0
laplace_sd, bayes

# %%
assert round(laplace_sd, 3) == 0.168
mc, bj = bayes["numpyro-mcmc"], bayes["blackjax"]
assert (round(mc["mean"], 3), round(mc["sd"], 3)) == (120.143, 0.167)
assert (round(bj["mean"], 3), round(bj["sd"], 3)) == (120.152, 0.168)
assert mc["divergences"] in (0, None) and bj["divergences"] in (0, None)

# %% [markdown]
# ## Part 4 · Hashes, the cache and the window dataset (stage 8)

# %%
f1 = pc.CallableForward(lambda x, dt: M @ x, dt=1.0)
f2 = pc.CallableForward(lambda x, dt: 2 * M @ x, dt=1.0)  # different physics
f3 = pc.CallableForward(lambda x, dt: M @ x, dt=2.0)


def h(**kw) -> str:
    base = dict(forward_model=f1, prior=SmoothPrior(), n_samples=2000, seed=0)
    return SimulationDataset(**(base | kw)).content_hash()


ref = h()
hash_changes = {
    "new objects": h(prior=SmoothPrior()) != ref,
    "n_samples": h(n_samples=1000) != ref,
    "seed": h(seed=1) != ref,
    "split": h(split="val") != ref,
    "prior": h(prior=SmoothPrior(n_bumps=5)) != ref,
    "forward function": h(forward_model=f2) != ref,
    "forward dt": h(forward_model=f3) != ref,
}

cache_dir = root / "cache"
src = SimulationDataset(f1, SmoothPrior(), n_samples=500, seed=0)
cached = CachedDataset(src, str(cache_dir))
t0 = time.time()
first = [x for x, _ in cached]
t_first = time.time() - t0
t0 = time.time()
second = [x for x, _ in cached]
t_second = time.time() - t0
stale = next(iter(CachedDataset(
    SimulationDataset(f2, SmoothPrior(), n_samples=500, seed=0),
    str(cache_dir))))[1]
serves_stale = bool(np.allclose(stale, M @ src[0][0]))
t0 = time.time()
_ = [cached[i] for i in range(50)]
ms_cached = (time.time() - t0) / 50 * 1e3
t0 = time.time()
_ = [src[i] for i in range(50)]
ms_direct = (time.time() - t0) / 50 * 1e3
hash_changes, len(first), (t_first, t_second), serves_stale, (ms_cached, ms_direct)

# %% [markdown]
# `XarrayWindowDataset` imports `geopatcher.time`, which the pinned geopatcher
# ships as `geopatcher.temporal.stencils` (finding A5). The next cell records
# the import error, then aliases the module so the rest of the stage runs.

# %%
try:
    importlib.import_module("geopatcher.time")
    window_import = "ok"
except ImportError as e:
    window_import = f"ImportError: {e}"
    import geopatcher.temporal.stencils as _stencils

    sys.modules["geopatcher.time"] = _stencils
from geopatcher.time import TimeStencil  # noqa: E402
try:  # Report 16's D2 moves the dataset; fall back to the pinned location
    from geopatcher.integrations.train import XarrayWindowDataset
except ModuleNotFoundError as err:
    if err.name != "geopatcher.integrations.train":
        raise
    from pipekit_train import XarrayWindowDataset

T = 50
x = SmoothPrior()(seed=3).astype(np.float64)
traj = np.empty((T, N))
for t in range(T):
    traj[t] = x
    x = M @ x
times = np.datetime64("2020-01-01T00", "ns") + np.arange(T) * np.timedelta64(1, "h")
xr.Dataset({"q": (("time", "x"), traj.astype(np.float32))},
           coords={"time": times, "x": np.arange(N)}).to_zarr(
    root / "adv.zarr", mode="w")
lazy = xr.open_zarr(root / "adv.zarr", chunks=None)
stencils = {"input": TimeStencil("-2h", "0h", "1h", closed="both"),
            "target": TimeStencil("1h", "1h", "0h", closed="both")}
win = XarrayWindowDataset(source={"input": lazy, "target": lazy},
                          stencils=stencils, store_version="adv-v1")
win_lens = tuple(len(win.with_split(s)) for s in ("train", "val", "test"))
xi, yi = win[0]
target_exact = bool(np.allclose(yi[0, 0], M @ xi[0, -1], atol=1e-5))


class LastStep(eqx.Module):
    lin: eqx.nn.Linear

    def __call__(self, x):  # (1, 3, N) → (1, 1, N)
        return self.lin(x[0, -1])[None, None]


_, wart = TrainingLoop(
    model_op=EquinoxModelOp(LastStep(eqx.nn.Linear(N, N, use_bias=False,
                                                   key=jax.random.key(0)))),
    dataset=win, val_dataset=win.with_split("val"), loss=MSE(),
    optimizer_config={"name": "adam", "lr": 1e-2}, max_steps=300, batch_size=8,
    eval_every_n_steps=100, seed=0,
).run()
win_mse = wart.backend_info["final_metrics"]["mse"]
win.close()
window_import, len(win._all_origins), win_lens, target_exact, win_mse

# %%
assert all(v for kk, v in hash_changes.items()
           if kk not in ("new objects", "forward function"))
assert not hash_changes["new objects"]
assert not hash_changes["forward function"]  # the simulator is not hashed (A3)
assert len(first) == 400 and np.allclose(np.stack(first), np.stack(second))
assert serves_stale
assert window_import.startswith("ImportError")
assert len(win._all_origins) == 47 and win_lens == (35, 1, 3) and target_exact
assert f"{win_mse:.1e}" == "3.5e-05"

# %% [markdown]
# ## Part 5 · `JaxModelOp` through the Equinox loop, and the registry

# %%
ds8 = SimulationDataset(fwd, WhitePrior(), 2000)
reg8 = LocalModelRegistry(str(root / "registry8"))
skel = JaxModelOp(eqx.nn.Linear(N, N, use_bias=False, key=jax.random.key(0)))
trained8, art8 = TrainingLoop(skel, ds8, loss=MSE(),
                              optimizer_config={"name": "adam", "lr": 1e-2},
                              max_steps=500, batch_size=64,
                              callbacks=(Checkpoint(registry=reg8),), seed=0).run()
out_type = type(trained8).__name__
weights_after_ckpt = reg8.load_weights(art8.trained_model_hash) is not None
j = JaxModelOp(trained8.module)
h8 = reg8.store(j, weights=j.serialize_weights())
re8 = skel.from_registry(reg8, h8)
reload_err = float(np.abs(np.asarray(re8.module.weight) - M).max())
xn, _ = pc.Cycle(pc.NeuralForward(re8, dt=1.0), n_steps=6)(np.ones(N, np.float32),
                                                           None)
reload_cycle = float(np.abs(np.asarray(xn) - np.linalg.matrix_power(M, 6)
                            @ np.ones(N)).max())
out_type, weights_after_ckpt, reload_err, reload_cycle

# %%
assert out_type == "EquinoxModelOp"  # a JaxModelOp goes in, an Equinox one out
assert not weights_after_ckpt
assert f"{reload_err:.1e}" == "3.8e-07" and reload_cycle < 1e-5
