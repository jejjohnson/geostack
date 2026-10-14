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
# # Report 17 · The six-step cycle on one retrieval, run
#
# Regenerates every number in `docs/reports/17_cycle.md`: the six steps of the
# cycle on one row of the cycle matrix, the L1 → L2 retrieval of a column
# enhancement from five-band transmittances (the sensor of Reports 10 and 11).
#
# 1. the simple model as a ladder `Task` (stage 1);
# 2. the oracle: the exact posterior by quadrature, and a Laplace step below it
#    (stage 2);
# 3. two polynomial emulators of the forward model (stage 3);
# 4. the oracle's inversion rerun with each emulator (stage 4);
# 5. an amortized Gaussian head, scored against the truth and the oracle
#    (stage 5);
# 6. the shipped runner, and the lineage the registry can and cannot record
#    (stage 6).
#
# Every draw is seeded. Each stage ends in a cell asserting the numbers the
# page shows, to the precision it shows them. Where a bound depends on a
# random draw, the comment says where the bound comes from.
#
# Pinned revisions: pipekit 0.0.2 at `ef3c17f` (pipekit, -cycle, -train,
# -experiment, -evaluate). It needs NumPy, SciPy and those packages; it runs in
# about thirty seconds on a laptop CPU.

# %%
from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pipekit_cycle as pc
import xarray as xr
from pipekit import Lambda, Operator, StatefulOperator
from pipekit_evaluate.benchmark import (
    DEFAULT_STAGED,
    LEVEL_NAMES,
    RUNG_NAMES,
    Benchmark,
    Cell,
    Estimate,
    Estimator,
    ReferenceSource,
    Task,
)
from pipekit_experiment import LocalModelRegistry
from pipekit_train import SimulationDataset
from scipy.stats import norm
from xrtoolz.metrics import wasserstein_1

# %% [markdown]
# ## Stage 1 · The simple model, as a ladder task
#
# A sensor measures transmittances y_b = exp(−k_b α) + ε_b in B = 5 bands, with
# k = (0.002, …, 0.010), a column enhancement α ~ 𝒰(0, 300) and noise
# ε_b ~ 𝒩(0, σ²), σ = 10⁻³. The test set is n = 500 scenes.

# %%
K = np.array([0.002, 0.004, 0.006, 0.008, 0.010])
SIG, A_MAX, N_TEST, N_TRAIN, SEED = 1e-3, 300.0, 500, 2_000, 0
# The test scenes use their own seeds, outside the SimulationDataset's 0 … 1,999
# and the amortized head's stream, so no training draw is a test draw
TEST_SEED = 10_000
G_ORACLE = 30_001  # quadrature nodes on [0, A_MAX]: a step of 0.01


def f(alpha: np.ndarray) -> np.ndarray:
    # f_b(α) = exp(−k_b α)                                     (n,) → (n, B)
    return np.exp(-np.multiply.outer(np.atleast_1d(alpha), K))


forward = pc.CallableForward(lambda a, dt: f(a)[0], dt=1.0, name="beer_lambert")


class UniformPrior:  # the ladder's Prior protocol                         user
    def sample(self, n: int, *, key: Any = None) -> np.ndarray:
        return np.random.default_rng(key).uniform(0, A_MAX, n)

    def log_prob(self, theta: Any) -> Any:
        return np.where((theta >= 0) & (theta <= A_MAX), -np.log(A_MAX), -np.inf)


alpha_test = UniformPrior().sample(N_TEST, key=TEST_SEED)
noise = np.random.default_rng(TEST_SEED + 1).standard_normal((N_TEST, K.size))
y_test = f(alpha_test) + SIG * noise


@dataclass
class RetrievalTask:  # the ladder's Task protocol                         user
    name: str = "beer_lambert_5band"
    domain: str = "methane"

    def prior(self) -> UniformPrior:
        return UniformPrior()

    def simulator(self):
        return forward

    def observation_operator(self):
        return pc.IdentityObs()

    def datasets(self) -> dict[str, np.ndarray]:
        # θ for rung 2, f(θ) for rung 3, y for every estimator
        return {"theta": alpha_test, "state": f(alpha_test), "obs": y_test}


task = RetrievalTask()
stage1 = {
    "is_task": isinstance(task, Task),
    "rungs": dict(RUNG_NAMES),
    "staged": {r: s.value for r, s in DEFAULT_STAGED.items()},
    "L2": LEVEL_NAMES["L2"],
    "cell": Cell("L2", 4, 0).label,
}
stage1

# %%
assert stage1["is_task"]
assert stage1["rungs"] == {1: "simple_model", 2: "model_based_inference",
                           3: "emulator", 4: "emulator_based_inference",
                           5: "amortized_predictor", 6: "improve"}
# rung 3 against the simulator, rungs 4 and 5 against the rung-2 oracle
assert stage1["staged"] == {1: "none", 2: "known_theta", 3: "simulator_output",
                            4: "oracle_posterior", 5: "oracle_posterior",
                            6: "best_validated_below"}
assert stage1["L2"] == "gap_filled_obs"  # the overview's L2 is retrieved obs
assert stage1["cell"] == "L2/r4/c0"  # a cell names one level, not a transition

# %% [markdown]
# ## Stage 2 · The oracle, and a cheaper step below it
#
# With a flat prior on [0, 300], p(α | y) ∝ 𝟙[0 ≤ α ≤ 300] ∏_b 𝒩(y_b; f_b(α), σ²).
# The oracle evaluates it on G = 30,001 nodes (exact to quadrature, a step of
# 0.01 against a smallest posterior standard deviation of 0.068). Below
# it on the complexity axis, a Gauss–Newton fit with its Laplace width
# σ / ‖J(α̂)‖, run as a `pipekit_cycle.Recurrence` (Report 10, stage 5).

# %%
GRID = np.linspace(0.0, A_MAX, G_ORACLE)


@dataclass
class GridEstimate:  # an Estimate: a posterior on the grid             user
    w: np.ndarray  # (n, G) normalised weights

    def point(self) -> np.ndarray:
        return self.w @ GRID

    def sd(self) -> np.ndarray:
        m = self.point()
        return np.sqrt((self.w * (GRID[None] - m[:, None]) ** 2).sum(1))

    def cdf(self) -> np.ndarray:
        return np.cumsum(self.w, 1)

    def sample(self, k: int, *, key: Any = None) -> np.ndarray:
        u = np.random.default_rng(key).random((len(self.w), k))
        c = self.cdf()
        return np.stack([GRID[np.searchsorted(ci, ui)] for ci, ui in zip(c, u,
                                                                          strict=True)])

    def log_prob(self, theta: Any) -> Any:
        raise NotImplementedError


@dataclass
class GaussEstimate:  # an Estimate: mean and standard deviation          user
    mean: np.ndarray
    std: np.ndarray

    def point(self) -> np.ndarray:
        return self.mean

    def sd(self) -> np.ndarray:
        return self.std

    def cdf(self) -> np.ndarray:
        return norm.cdf(GRID[None], self.mean[:, None], self.std[:, None])

    def sample(self, k: int, *, key: Any = None) -> np.ndarray:
        z = np.random.default_rng(key).standard_normal((len(self.mean), k))
        return self.mean[:, None] + self.std[:, None] * z

    def log_prob(self, theta: Any) -> Any:
        return norm.logpdf(theta, self.mean, self.std)


def grid_posterior(y: np.ndarray, fwd) -> GridEstimate:
    # log p(α_g | y) = −½ Σ_b (y_b − f_b(α_g))² / σ² + c        (n, B) → (n, G)
    F = fwd(GRID)                                                 # (G, B)
    ll = -0.5 * ((y[:, None, :] - F[None]) ** 2).sum(-1) / SIG**2
    w = np.exp(ll - ll.max(1, keepdims=True))
    return GridEstimate(w / w.sum(1, keepdims=True))


class GaussNewtonStep(StatefulOperator):  # one Gauss–Newton update        user
    def __init__(self, y: np.ndarray) -> None:
        self.y = y

    def _apply(self, a, s):
        # α ← α + Jᵀ(y − f(α)) / JᵀJ,  J_b = −k_b exp(−k_b α)       (n,) → (n,)
        J = -K * f(a)
        step = (J * (self.y - f(a))).sum(1) / (J * J).sum(1)
        a_new = np.clip(a + step, 0.0, A_MAX)
        return a_new, pc.IterationState(iter_count=s.iter_count + 1,
                                        residual_norm=float(np.abs(a_new - a).max()))


def laplace(y: np.ndarray) -> GaussEstimate:
    a0 = np.clip(-np.log(np.clip(y[:, 0], 1e-6, None)) / K[0], 0, A_MAX)
    gn = pc.Recurrence(GaussNewtonStep(y),
                       condition_op=Lambda(lambda s: s.residual_norm < 1e-8),
                       max_iters=50)                                    # today
    a_hat, _ = gn(a0, pc.IterationState())
    # σ_L = σ / ‖J(α̂)‖
    return GaussEstimate(a_hat, SIG / np.linalg.norm(K * f(a_hat), axis=1))


def coverage95(est, truth: np.ndarray) -> float:
    c = est.cdf()
    lo = GRID[[np.searchsorted(ci, 0.025) for ci in c]]
    hi = GRID[[min(np.searchsorted(ci, 0.975), G_ORACLE - 1) for ci in c]]
    return float(np.mean((truth >= lo) & (truth <= hi)))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


oracle = grid_posterior(y_test, f)
lap = laplace(y_test)
m_or, s_or = oracle.point(), oracle.sd()
cr = SIG / np.linalg.norm(K * f(alpha_test), axis=1)  # Cramér–Rao, σ / ‖J(α)‖
stage2 = {
    "oracle_rmse": rmse(m_or, alpha_test), "cr_rms": float(np.sqrt(np.mean(cr**2))),
    "oracle_cov95": coverage95(oracle, alpha_test),
    "laplace_rmse": rmse(lap.point(), alpha_test),
    "laplace_cov95": coverage95(lap, alpha_test),
    "sd_min": float(s_or.min()),
    "laplace_vs_oracle_mean": float(np.max(np.abs(lap.point() - m_or) / s_or)),
}
{k: round(v, 3) for k, v in stage2.items()}

# %%
assert round(stage2["oracle_rmse"], 3) == 0.236
# σ / ‖J(α)‖ is a local Fisher scale, not a bound on a prior-averaged posterior
# mean: the prior and the boundary at α = 0 let the oracle go below it
assert round(stage2["cr_rms"], 3) == 0.247 and stage2["oracle_rmse"] < stage2["cr_rms"]
# 95 % coverage on n = 500: standard error √(0.95 · 0.05 / 500) = 0.0097, so
# 0.962 is 1.2 standard errors above nominal
assert round(stage2["oracle_cov95"], 3) == 0.962
assert abs(stage2["oracle_cov95"] - 0.95) < 3 * np.sqrt(0.95 * 0.05 / N_TEST)
assert round(stage2["laplace_rmse"], 3) == 0.237
assert round(stage2["laplace_cov95"], 3) == 0.960
assert round(stage2["sd_min"], 3) == 0.068  # six nodes per smallest width
assert stage2["sd_min"] > 5 * (GRID[1] - GRID[0])  # the grid resolves every posterior
assert round(stage2["laplace_vs_oracle_mean"], 2) == 0.25  # at worst a quarter width

# %% [markdown]
# ## Stage 3 · Two emulators of the forward model
#
# An emulator f̂ of degree d fits each band by a polynomial in α / 300, by least
# squares on 2,000 draws from a `SimulationDataset`. Scored against the
# simulator's outputs on the test scenes (the reference rung 3 calls for), in
# units of the noise σ.

# %%
class PriorDraw(Operator):  # the SimulationDataset prior: seeded α       user
    def _apply(self, seed: int = 0) -> np.ndarray:
        return np.array([np.random.default_rng(seed).uniform(0, A_MAX)])


sim = SimulationDataset(forward, PriorDraw(), n_samples=N_TRAIN, seed=SEED)  # today
pairs = [sim[i] for i in range(len(sim))]
a_tr = np.array([p[0][0] for p in pairs])
f_tr = np.stack([p[1] for p in pairs])                               # (1,600, B)


def fit_emulator(d: int) -> np.ndarray:
    # C = argmin ‖V C − F‖²,  V = [(α/300)^d … 1]           (1,600, d+1) → (d+1, B)
    V = np.vander(a_tr / A_MAX, d + 1)
    return np.linalg.lstsq(V, f_tr, rcond=None)[0]


def emulator(C: np.ndarray):
    d = len(C) - 1
    return lambda a: np.vander(np.atleast_1d(a) / A_MAX, d + 1) @ C


coef = {d: fit_emulator(d) for d in (4, 6)}
emu = {d: emulator(C) for d, C in coef.items()}
dense = np.linspace(0, A_MAX, 3_001)
stage3 = {d: {"test_max_over_sigma": float(np.abs(emu[d](alpha_test) - f(alpha_test)).max()
                                           / SIG),
              "dense_max_over_sigma": float(np.abs(emu[d](dense) - f(dense)).max() / SIG),
              "rel_max": float((np.abs(emu[d](dense) - f(dense)) / f(dense)).max())}
          for d in coef}
n_train = len(sim)
n_train, {d: {k: round(v, 4) for k, v in s.items()} for d, s in stage3.items()}

# %%
assert n_train == 1_600  # the train split: 80 % of 2,000 seeds
assert np.intersect1d(a_tr, alpha_test).size == 0  # held out from the emulator
assert round(stage3[4]["dense_max_over_sigma"], 1) == 2.2
assert round(stage3[6]["dense_max_over_sigma"], 2) == 0.03
assert round(stage3[4]["rel_max"], 3) == 0.032  # 3.2 % of the transmittance, worst case

# %% [markdown]
# ## Stage 4 · The oracle's inversion, rerun with each emulator
#
# The same quadrature with f̂ in place of f, scored against the oracle's
# posterior: the shift of the mean in units of the oracle's standard deviation,
# the ratio of the widths, and the 1-Wasserstein distance W₁ = ∫ |F̂ − F| dα
# between the two posterior distribution functions.

# %%
def versus_oracle(est) -> dict[str, float]:
    shift = (est.point() - m_or) / s_or
    w1 = np.abs(est.cdf() - oracle.cdf()).sum(1) * (GRID[1] - GRID[0])
    return {"shift_rms": float(np.sqrt(np.mean(shift**2))),
            "shift_max": float(np.abs(shift).max()),
            "sd_ratio": float(np.mean(est.sd() / s_or)),
            "w1_over_sd": float(np.mean(w1 / s_or))}


fast = {d: grid_posterior(y_test, emu[d]) for d in coef}
stage4 = {d: versus_oracle(fast[d]) for d in coef}


def w1_samples(a, b, n: int = 2_000) -> float:
    # xrtoolz's W₁ takes samples only: draw n from each posterior     (500, n) → (500,)
    da = xr.DataArray(a.sample(n, key=1), dims=("scene", "member"))
    db = xr.DataArray(b.sample(n, key=2), dims=("scene", "member"))
    return float((wasserstein_1(da, db) / s_or).mean())                     # today


w1_xrtoolz = {"emulator_4": w1_samples(fast[4], oracle),
              "oracle_vs_itself": w1_samples(oracle, oracle)}
stage4_truth = {d: coverage95(fast[d], alpha_test) for d in coef}
{d: {k: round(v, 3) for k, v in s.items()} for d, s in stage4.items()}, stage4_truth, w1_xrtoolz

# %%
assert {k: round(v, 2) for k, v in stage4[4].items()} == {
    "shift_rms": 0.44, "shift_max": 1.94, "sd_ratio": 1.0, "w1_over_sd": 0.37}
assert round(stage4[6]["shift_max"], 2) == 0.02
# xrtoolz's sample-based W₁ agrees with the exact one to its sampling floor:
# two 2,000-draw samples of the same posterior are already 0.04 σ apart
assert round(w1_xrtoolz["emulator_4"], 2) == 0.37
assert round(w1_xrtoolz["oracle_vs_itself"], 2) == 0.04
assert stage4[6]["w1_over_sd"] < w1_xrtoolz["oracle_vs_itself"]  # below the floor
assert round(stage4[6]["w1_over_sd"], 3) == 0.004
# against θ the shift barely shows: 1.4 points of coverage, 1.4 standard errors
assert round(stage4_truth[4], 3) == 0.948
assert round(stage4_truth[6], 3) == 0.962  # the degree-6 emulator is the oracle

# %% [markdown]
# ## Stage 5 · An amortized Gaussian head
#
# q(α | y) = 𝒩(μ(y), s(y)²), fitted once on 20,000 simulated pairs: μ by least
# squares on features φ(y) (the per-band inversions z_b = −log y_b / k_b and a
# cubic in their mean), and log s² by least squares on log (α − μ)², shifted by
# the mean of log χ²₁ (−1.27). Scored against the truth and against the oracle.

# %%
def features(y: np.ndarray) -> np.ndarray:
    # φ(y) = [1, z₁…z₅, z̄², z̄³] / 300,  z_b = −log y_b / k_b         (n, B) → (n, 8)
    z = -np.log(np.clip(y, 1e-4, None)) / K / A_MAX
    zbar = z.mean(1)
    return np.column_stack([np.ones(len(y)), z, zbar**2, zbar**3])


N_AMORT = 20_000
r5 = np.random.default_rng(SEED + 5)
a_am = r5.uniform(0, A_MAX, N_AMORT)
y_am = f(a_am) + SIG * r5.standard_normal((N_AMORT, K.size))
Phi = features(y_am)
beta_mu = np.linalg.lstsq(Phi, a_am, rcond=None)[0]               # (8,)
beta_s = np.linalg.lstsq(Phi, np.log((a_am - Phi @ beta_mu) ** 2), rcond=None)[0]
LOG_CHI2_MEAN = -1.2704  # 𝔼[log χ²₁] = −γ − log 2


def amortized(y: np.ndarray) -> GaussEstimate:
    P = features(y)
    return GaussEstimate(P @ beta_mu, np.exp(0.5 * (P @ beta_s - LOG_CHI2_MEAN)))


q = amortized(y_test)
z = (alpha_test - q.point()) / q.sd()
stage5 = {"rmse": rmse(q.point(), alpha_test), "ez2": float(np.mean(z**2)),
          "cov95": coverage95(q, alpha_test), **versus_oracle(q)}
low = alpha_test < 50
stage5["sd_ratio_low"] = float(np.mean(q.sd()[low] / s_or[low]))
stage5["shift_rms_low"] = float(np.sqrt(np.mean(((q.point() - m_or) / s_or)[low] ** 2)))
stage5["n_low"] = int(low.sum())
{k: round(v, 3) for k, v in stage5.items()}

# %%
assert np.intersect1d(a_am, alpha_test).size == 0  # held out from the head
assert round(stage5["rmse"], 3) == 0.255
# against the truth it passes: 𝔼[z²] = 0.90 is 1.6 standard errors below 1
# (√(2 / 500) = 0.063 for Gaussian z) and coverage 0.954 (SE 0.0097)
assert round(stage5["ez2"], 2) == 0.90 and round(stage5["cov95"], 3) == 0.954
assert abs(stage5["ez2"] - 1) < 2 * np.sqrt(2 / N_TEST)
# against the oracle it does not: 16 % too wide, 53 % for α < 50, and its mean
# is 1.0 oracle standard deviation off there (3.6 at worst)
assert round(stage5["sd_ratio"], 2) == 1.16 and round(stage5["sd_ratio_low"], 2) == 1.53
assert round(stage5["shift_rms_low"], 2) == 1.04 and stage5["n_low"] == 83
assert round(stage5["shift_max"], 1) == 3.6 and round(stage5["w1_over_sd"], 2) == 0.38

# %% [markdown]
# ## Stage 6 · The shipped runner, and the lineage
#
# **(a) The runner.** The estimators of stages 2 to 5 as ladder estimators at
# level L2, scored by `Benchmark` with the default staged rule.

# %%
@dataclass
class Rung:  # a ladder Estimator around a function of y                 user
    rung: int
    complexity: int
    fn: Any

    def fit(self, task: Task) -> Rung:
        return self

    def __call__(self, obs: Any) -> Estimate:
        return self.fn(obs)


@dataclass
class EmulatorRung:  # rung 3: a forward emulator is θ → y, not y → θ        user
    rung: int
    complexity: int
    fhat: Any
    theta: Any = None

    def fit(self, task: Task) -> EmulatorRung:
        # the runner calls every estimator on obs; rung 3 needs θ, so it has
        # to read it from the task here
        self.theta = task.datasets()["theta"]
        return self

    def __call__(self, obs: Any) -> Estimate:
        return GaussEstimate(self.fhat(self.theta).ravel(), np.zeros(1))


@dataclass
class Score:  # a ladder Scorer                                        user
    name: str = "s"

    def __call__(self, prediction: Estimate, reference: Any) -> dict[str, float]:
        if hasattr(reference, "point"):  # an oracle posterior
            return {k: round(v, 3) for k, v in versus_oracle(prediction).items()
                    if k in ("shift_max", "sd_ratio")}
        ref = np.asarray(reference).ravel()
        e = np.abs(np.asarray(prediction.point()).ravel() - ref)
        unit = SIG if ref.size == N_TEST * K.size else 1.0  # spectra in σ, α as is
        return {"max_err": round(float(e.max() / unit), 2),
                "rmse": round(float(np.sqrt(np.mean((e / unit) ** 2))), 3)}


def run(d: int):
    ests = [Rung(2, 0, laplace), Rung(2, 1, lambda y: grid_posterior(y, f)),
            EmulatorRung(3, 0, emu[d]), Rung(4, 0, lambda y: grid_posterior(y, emu[d])),
            Rung(5, 0, amortized)]
    assert all(isinstance(e, Estimator) for e in ests)
    rep = Benchmark(task=task, estimators=ests, scorers=[Score()], level="L2").run()
    return {r.cell.label: (r.reference.value, dict(r.scores)) for r in rep.results}


cells = run(4)
cells

# %%
assert cells == {
    "L2/r2/c0": ("known_theta", {"s.max_err": 1.12, "s.rmse": 0.237}),
    "L2/r2/c1": ("known_theta", {"s.max_err": 1.12, "s.rmse": 0.236}),
    "L2/r3/c0": ("simulator_output", {"s.max_err": 2.01, "s.rmse": 0.277}),
    "L2/r4/c0": ("oracle_posterior", {"s.shift_max": 1.941, "s.sd_ratio": 1.001}),
    "L2/r5/c0": ("oracle_posterior", {"s.shift_max": 3.631, "s.sd_ratio": 1.157}),
}

# %% [markdown]
# **(b) The lineage.** Each artifact goes into a `LocalModelRegistry` as a small
# operator whose configuration names the hashes it was built from. Tags are
# not part of the hash: a second `store` of the same emulator with another
# oracle tag overwrites the first, and a tag filter matches whole values only.

# %%
root = Path(tempfile.mkdtemp(prefix="report17_"))
(root / "r16ops.py").write_text(
    "from pipekit import Operator\n\n\n"
    "class Artifact(Operator):\n"
    "    def __init__(self, kind: str = '', config: dict | None = None,\n"
    "                 parents: list | None = None) -> None:\n"
    "        self.kind = kind\n"
    "        self.config = config or {}\n"
    "        self.parents = parents or []\n\n"
    "    def _apply(self, x):\n"
    "        return x\n"
)
sys.path.insert(0, str(root))
from r16ops import Artifact  # noqa: E402

reg = LocalModelRegistry(root / "reg")
h1 = reg.store(Artifact("simulator", {"k": K.tolist(), "sigma": SIG}),
               tags={"rung": 1})
h2 = reg.store(Artifact("grid_posterior", {"nodes": G_ORACLE}, [h1]),
               tags={"rung": 2, "role": "oracle"})


def emulator_artifact(d: int) -> Artifact:
    return Artifact("poly_emulator", {"degree": d, "coef": coef[d].round(12).tolist()},
                    [h1])


def store_cycle(d: int) -> dict[int, str]:
    h3 = reg.store(emulator_artifact(d), tags={"rung": 3, "oracle": h1})
    h4 = reg.store(Artifact("grid_posterior", {"nodes": G_ORACLE}, [h3]),
                   tags={"rung": 4, "oracle": h2})
    h5 = reg.store(Artifact("gaussian_head", {"beta_mu": beta_mu.round(9).tolist(),
                                              "beta_s": beta_s.round(9).tolist()},
                            [h1]), tags={"rung": 5, "oracle": h2})
    return {3: h3, 4: h4, 5: h5}


v1 = store_cycle(4)
v2 = store_cycle(6)  # step 6: improve the emulator
changed = {r: v1[r] != v2[r] for r in (3, 4, 5)}


def parents(h: str) -> list[str]:
    return json.loads((reg.root / h / "operator.json").read_text())["config"]["parents"]


def dependents(h: str) -> list[str]:  # the reverse lineage query          user
    return sorted(x for x in reg.list() if h in parents(x))


stale = dependents(v1[3])
by_rung = {r: len(reg.list(tags={"rung": r})) for r in (1, 2, 3, 4, 5)}
# tags are metadata, not content: re-store the v2 emulator with another oracle
h3_again = reg.store(emulator_artifact(6), tags={"oracle": "not-the-simulator"})
tag_now = json.loads((reg.root / v2[3] / "metadata.json").read_text())["tags"]["oracle"]
by_parent_tag = reg.list(tags={"parents": [h1]})
stage6 = {"changed": changed, "stale_is_v1_rung4": stale == [v1[4]],
          "by_rung": by_rung, "same_hash": h3_again == v2[3], "tag_now": tag_now,
          "by_parent_tag": by_parent_tag}
stage6

# %%
assert cells == run(4)  # rerunning the runner is deterministic
v2_cells = run(6)
assert v2_cells["L2/r3/c0"] == ("simulator_output", {"s.max_err": 0.03, "s.rmse": 0.003})
assert v2_cells["L2/r4/c0"] == ("oracle_posterior", {"s.shift_max": 0.022,
                                                     "s.sd_ratio": 1.0})
assert v2_cells["L2/r5/c0"] == cells["L2/r5/c0"]  # rung 5 did not use the emulator
# a new emulator changes rungs 3 and 4 only: rung 5 was trained on the simulator
assert changed == {3: True, 4: True, 5: False}
assert stage6["stale_is_v1_rung4"]  # only the old rung-4 artifact depends on it
assert by_rung == {1: 1, 2: 1, 3: 2, 4: 2, 5: 1}
assert stage6["same_hash"] and tag_now == "not-the-simulator"  # lineage overwritten
assert by_parent_tag == []  # parents are config, not tags: the tag filter sees none
