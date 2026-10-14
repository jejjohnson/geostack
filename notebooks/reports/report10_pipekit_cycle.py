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
# # Report 10 · pipekit-cycle, run
#
# Regenerates every number in `docs/reports/10_pipekit_cycle.md`: a methane
# enhancement advected along a 40-cell transect, observed every fourth cell
# for ten hourly cycles, then forecast, filtered (Kalman and EnKF), re-run
# with swapped forward models, and a Gauss–Newton retrieval. The last cell
# asserts the values the page shows, to the precision it shows them.
#
# Pinned revision: pipekit and pipekit-cycle 0.0.2 at `ef3c17f`. Every random
# draw comes from a generator seeded in this notebook.
#
# > Jupytext percent-format script; `pixi run notebooks-to-ipynb` makes a
# > notebook of it. It needs only NumPy, pipekit and pipekit-cycle.

# %%
from __future__ import annotations

import numpy as np
import pipekit as pk
import pipekit_cycle as pc
from pipekit import Lambda, StatefulOperator

# Shapes: N = 40 cells, p = 10 observed cells, T = 10 hourly cycles
N, T, c = 40, 10, 0.4
M = (1 - c) * np.eye(N) + c * np.roll(np.eye(N), 1, axis=0)  # upwind advection
H = np.eye(N)[::4]
p = H.shape[0]
Q, R = 0.01 * np.eye(N), 0.1 * np.eye(p)
rng = np.random.default_rng(0)
xt = 50 * np.exp(-0.5 * ((np.arange(N) - 8) / 3) ** 2)  # ppb, plume at cell 8
truth, y = [], []
for _ in range(T):
    xt = M @ xt + rng.multivariate_normal(np.zeros(N), Q)
    truth.append(xt)
    y.append(H @ xt + rng.normal(0, np.sqrt(0.1), p))
x0, P0 = np.zeros(N), 100 * np.eye(N)
obs_source = Lambda(lambda i: y[i])


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(a**2)))


# %% [markdown]
# ## Stage 1 · Forecast only: `Cycle`

# %%
class Advect:  # ForwardModel on the mean alone
    dt = 1.0
    state_signature = None

    def step(self, x, dt):
        return M @ x


fc = pc.Cycle(Advect(), n_steps=6, save_history=True, history_stride=2)
out = fc(truth[0], pc.DAState())
peaks = [int(np.argmax(h[0])) for h in fc.history]
type(out).__name__, len(fc.history), peaks, out[1].t


# %% [markdown]
# ## Stage 2 · Kalman filter as a `DACycle`, against a hand-written filter

# %%
class KFForward:  # carrier = (mean, covariance)
    dt = 1.0
    state_signature = None

    def step(self, carrier, dt):
        m, P = carrier
        return M @ m, M @ P @ M.T + Q


# KalmanAnalysis and EnKF below are user-written AnalysisSteps on purpose: the
# report audits the protocols a user implements, and the algorithm library that
# owns these numerics (filterax) is not pinned yet (Report 10, upcoming work).
class KalmanAnalysis:
    def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
        m, P = forecast
        Hm = obs_op.linearize(m).H
        K = P @ Hm.T @ np.linalg.inv(Hm @ P @ Hm.T + obs_err_cov)
        return m + K @ (obs - obs_op(m)), (np.eye(N) - K @ Hm) @ P


da = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                obs_source=obs_source, n_steps=T)
(mA, PA), sA = da((x0, P0), pc.DAState(obs_err_cov=R))
m, P = x0, P0
for k in range(T):
    m, P = M @ m, M @ P @ M.T + Q
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    m, P = m + K @ (y[k] - H @ m), (np.eye(N) - K @ H) @ P
kf_rmse, prior_rmse = rms(mA - truth[-1]), rms(truth[-1])
sd = float(np.sqrt(np.diag(PA)).mean())
(np.abs(mA - m).max(), np.abs(PA - P).max(), kf_rmse, prior_rmse,
 sA.t, sA.cycle_count, len(da.history))


# %% [markdown]
# ## Stage 3 · Stochastic EnKF as an `EnsembleDACycle`, ten seeds per size

# %%
def enkf_run(Ne: int, seed: int) -> float:
    erng = np.random.default_rng(seed)

    class Member:
        dt = 1.0
        state_signature = None

        def step(self, x, dt):
            return M @ x + erng.multivariate_normal(np.zeros(N), Q)

    class EnKF:
        def __call__(self, forecast, obs, *, obs_op, obs_err_cov):
            X = np.stack(forecast, axis=1)
            A = X - X.mean(1, keepdims=True)
            Pf = A @ A.T / (Ne - 1)
            K = Pf @ H.T @ np.linalg.inv(H @ Pf @ H.T + obs_err_cov)
            eps = erng.multivariate_normal(np.zeros(p), obs_err_cov, Ne).T
            HX = np.stack([obs_op(x) for x in forecast], axis=1)
            Xa = X + K @ (obs[:, None] + eps - HX)
            return [Xa[:, j] for j in range(Ne)]

    eda = pc.EnsembleDACycle(Member(), pc.LinearObs(H), EnKF(),
                             obs_source=obs_source, n_steps=T, n_members=Ne)
    members = [erng.multivariate_normal(x0, P0) for _ in range(Ne)]
    mem, _ = eda(members, pc.DAState(obs_err_cov=R))
    return rms(np.mean(mem, 0) - mA)


errs = {Ne: np.array([enkf_run(Ne, s) for s in range(10)])
        for Ne in (50, 200, 800)}
enkf = {Ne: float(np.median(e)) for Ne, e in errs.items()}

# The rate check (gate K2). The claim is error ∝ N_e^(−1/2) from 200 to 800
# members, i.e. a log-log slope of −0.5. Ten seeds per size give a noisy
# slope, so the check is against the slope's own sampling distribution: a
# seeded bootstrap over the seeds (resample each size's 10 errors with
# replacement, recompute the slope of the medians, 4,000 times) gives a 99 %
# interval, and −0.5 must lie inside it.
def slope(e200: np.ndarray, e800: np.ndarray) -> float:
    return float(np.log(np.median(e800) / np.median(e200)) / np.log(4))


boot_rng = np.random.default_rng(1)
boot = np.array([slope(boot_rng.choice(errs[200], 10),
                       boot_rng.choice(errs[800], 10)) for _ in range(4000)])
rate_ci = tuple(np.quantile(boot, [0.005, 0.995]))
enkf, slope(errs[200], errs[800]), rate_ci


# %% [markdown]
# ## Stage 4 · Swap the forward model

# %%
emulator = pc.CallableForward(lambda cr, dt: KFForward().step(cr, dt), dt=1.0)
neural = pc.NeuralForward(Lambda(lambda cr: KFForward().step(cr, 1.0)), dt=1.0)
swaps = {}
for name, fwd in {"CallableForward": emulator, "NeuralForward": neural}.items():
    cyc = pc.DACycle(fwd, pc.LinearObs(H), KalmanAnalysis(),
                     obs_source=obs_source, n_steps=T)
    (mS, _), _ = cyc((x0, P0), pc.DAState(obs_err_cov=R))
    swaps[name] = float(np.abs(mS - mA).max())
swaps

# %% [markdown]
# ## Stage 5 · Gauss–Newton retrieval as a `Recurrence`
#
# Five bands, y_b = exp(−k_b α) plus noise with σ = 1e-4; retrieve α.

# %%
kb = np.array([0.002, 0.004, 0.006, 0.008, 0.010])
alpha_true = 120.0
yb = np.exp(-kb * alpha_true) + np.random.default_rng(3).normal(0, 1e-4, 5)
f = pc.CallableObs(lambda a: np.exp(-kb * a),
                   linearize_fn=lambda a: -kb * np.exp(-kb * a), name="beer_lambert")


class GaussNewton(StatefulOperator):
    def _apply(self, a, s):
        J = f.linearize(a)
        step = float(J @ (yb - f(a)) / (J @ J))
        return a + step, pc.IterationState(iter_count=s.iter_count + 1,
                                           residual_norm=abs(step))


gn = pc.Recurrence(GaussNewton(),
                   condition_op=Lambda(lambda s: s.residual_norm < 1e-8),
                   max_iters=50)
a_hat, _ = gn(0.0, pc.IterationState())
grad = float(f.linearize(a_hat) @ (yb - f(a_hat)))
a_hat, gn.iters, gn.converged, grad


# %% [markdown]
# ## The two gaps
#
# (a) A stateful `Sequential` threads state but is not accepted as a cycle
# step. (b) `obs_source` is indexed from zero on every call, so a restarted
# cycle reads the first window again.

# %%
class Count(StatefulOperator):
    def _apply(self, x, s):
        return x, pc.DAState(t=s.t, cycle_count=s.cycle_count + 1)


seq = pk.Sequential([Lambda(lambda x: M @ x), Count()])
seq_count = seq(x0, pc.DAState())[1].cycle_count
try:
    pc.Cycle(seq, n_steps=3)
    seq_error = ""
except TypeError as e:
    seq_error = str(e)

seen: list[int] = []
da5 = pc.DACycle(KFForward(), pc.LinearObs(H), KalmanAnalysis(),
                 obs_source=Lambda(lambda i: (seen.append(i), y[i])[1]),
                 n_steps=5)
c1, s1 = da5((x0, P0), pc.DAState(obs_err_cov=R))
(m2, _), s2 = da5(c1, s1)
restart_err = float(np.abs(m2 - mA).max())
seq_count, seq_error, seen, restart_err, s2.cycle_count

# %%
assert type(out).__name__ == "tuple" and len(out) == 2
assert len(fc.history) == 3 and peaks == [9, 10, 10] and out[1].t == 0.0
assert np.abs(mA - m).max() == 0.0 and np.abs(PA - P).max() == 0.0
assert round(kf_rmse, 3) == 0.125 and round(prior_rmse, 1) == 17.1
assert (sA.t, sA.cycle_count, len(da.history)) == (10.0, 10, 10)
assert round(sd, 3) == 0.193
# the medians the page prints (a seeded rerun reproduces them exactly) ...
assert {k: round(v, 3) for k, v in enkf.items()} == {50: 0.367, 200: 0.063,
                                                     800: 0.027}
# ... and the behaviour they illustrate, bounded by its sampling distribution
assert rate_ci[0] <= -0.5 <= rate_ci[1]
assert round(slope(errs[200], errs[800]), 2) == -0.61
assert (round(rate_ci[0], 2), round(rate_ci[1], 2)) == (-0.96, -0.38)
assert swaps == {"CallableForward": 0.0, "NeuralForward": 0.0}
assert round(a_hat, 3) == 120.014 and gn.iters == 6 and gn.converged
assert abs(grad) < 1e-18
assert seq_count == 1
assert seq_error == ("step_op must be a StatefulOperator or a ForwardModel, "
                     "got Sequential.")
assert seen == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
assert round(restart_err, 1) == 15.4 and s2.cycle_count == 10


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
from pathlib import Path

root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
spec = importlib.util.spec_from_file_location(
    "report_figures", root / "docs/images/reports/make_figures.py")
rf = importlib.util.module_from_spec(spec)
sys.modules["report_figures"] = rf
spec.loader.exec_module(rf)
calls = rf.marked_calls(next((root / "docs/reports").glob("10_*.md")))
ready = Counter(c.status for c in calls)
ready

# %%
assert len(calls) == 27
assert ready == Counter({'here': 2, 'today': 20, 'user': 5})
