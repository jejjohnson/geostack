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
# # Report 15 · Benchmarking, run
#
# Regenerates every number in `docs/reports/15_benchmarking.md`: one sea
# surface height (SSH) mapping benchmark, built on a synthetic field with known
# space and time correlation scales and three simulated altimeters.
#
# 1. the contract as data, and its content hash (stage 1);
# 2. the same mapping method scored under six splits (stage 2);
# 3. the baselines the contract must pin (stage 3);
# 4. a deterministic and an ensemble entry under CRPS (stage 4);
# 5. the same methods against three references (stage 5);
# 6. the benchmark runner pipekit-evaluate ships (stage 6).
#
# Every draw is seeded; nothing is written to disk. Each stage ends in a cell
# asserting the numbers the page shows, to the precision it shows them.
#
# Pinned revisions: pipekit 0.0.2 at `ef3c17f` (pipekit, -train, -evaluate),
# xrtoolz 0.0.5 at `308c7f2`, geopatcher at geotoolz `088475e`. It needs NumPy,
# SciPy, pandas, xarray, xskillscore and those libraries; about 75 seconds on a
# laptop CPU, most of it the optimal interpolation of stage 3.

# %%
from __future__ import annotations

import dataclasses
import importlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pipekit.hashing import sha256_hex, stable_json
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from xrtoolz.interpolate import fillnan_climatology, fillnan_temporal
from xrtoolz.metrics import crps_ensemble, spread_skill_ratio

# %% [markdown]
# ## The field and the altimeters
#
# η(t, y, x) is an SSH anomaly in metres on a doubly periodic 800 km square
# (32 × 32 cells of Δ = 25 km), daily from 1 Jan 2023 to 30 Dec 2025 (T = 1,095
# days, so 2025 has 364): a
# seasonal cycle of amplitude a = 4 cm plus a mesoscale anomaly with standard
# deviation σ = 8 cm and a Gaussian covariance of length L = 100 km and time
# scale τ = 30 days. Three altimeters s1, s2, s3 each fly one diagonal ground
# track across the domain every third day, with noise σₒ = 2 cm.

# %%
NX = NY = 32
DX = 25.0  # km
T = 3 * 365
L, TAU = 100.0, 30.0  # km, days
SIG, AMP, SIG_O = 0.08, 0.04, 0.02  # m
rng = np.random.default_rng(0)

# C(r, Δt) = σ² exp(−r² / 2L² − Δt² / 2τ²)  ⇔  spectrum ∝ exp(−L²|k|²/2 − τ²ω²/2)
kx = 2 * np.pi * np.fft.fftfreq(NX, d=DX)
kt = 2 * np.pi * np.fft.fftfreq(T, d=1.0)
KT, KY, KX = np.meshgrid(kt, kx, kx, indexing="ij")
spectrum = np.exp(-0.5 * L**2 * (KX**2 + KY**2) - 0.5 * TAU**2 * KT**2)
anom = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((T, NY, NX)))
                            * np.sqrt(spectrum)))
anom *= SIG / anom.std()
times = pd.date_range("2023-01-01", periods=T, freq="D")
season = AMP * np.cos(2 * np.pi * (times.dayofyear.values - 60) / 365.25)
eta = anom + season[:, None, None]  # (T, NY, NX)
x = np.arange(NX) * DX
y = np.arange(NY) * DX
truth = xr.DataArray(eta, dims=("time", "y", "x"),
                     coords={"time": times, "y": y, "x": x}, name="ssh")

# satellite s sees cell (i, j) on day t when (j + dₛ i − vₛ t) mod 32 = 0, every 3 days
SATS = {"s1": (1, 3), "s2": (-1, 5), "s3": (1, 11)}  # track direction, drift
tt, ii, jj = np.meshgrid(np.arange(T), np.arange(NY), np.arange(NX), indexing="ij")
rows = []
for name, (d, v) in SATS.items():
    t_, i_, j_ = np.nonzero(((jj + d * ii - v * tt) % NX == 0) & (tt % 3 == 0))
    rows.append(pd.DataFrame({"t": t_, "i": i_, "j": j_, "sat": name}))
obs = pd.concat(rows, ignore_index=True)
ti, yi, xi, sat = obs.t.values, obs.i.values, obs.j.values, obs.sat.values
yobs = eta[ti, yi, xi] + SIG_O * rng.standard_normal(ti.size)


def lag_corr(d: int) -> float:
    return float(np.corrcoef(anom[:-d].ravel(), anom[d:].ravel())[0, 1])


field = {"n_obs": ti.size, "per_sat": int((sat == "s1").sum()),
         "anom_std": float(anom.std()), "rho3": lag_corr(3), "rho30": lag_corr(30)}
field

# %%
assert field["n_obs"] == 35_040 and field["per_sat"] == 11_680
assert round(field["anom_std"], 3) == 0.08
assert round(field["rho3"], 3) == 0.995  # exp(−9 / 1,800) = 0.995
assert round(field["rho30"], 2) == 0.58  # exp(−½) = 0.61; one 3-year sample

# %% [markdown]
# ## Stage 1 · The contract as data, and its hash

# %%
contract = {
    "name": "ssh-mapping-synthetic", "version": "1.0.0",
    "task": "gap_filling", "variant": "deterministic",
    "input": {"carrier": "points", "vars": ["ssh"], "units": "m",
              "sources": ["s2", "s3"]},
    "output": {"carrier": "grid", "dims": ["time", "y", "x"],
               "shape": [364, 32, 32], "units": "m"},
    "splits": {"kind": "leave_one_platform_out", "platform": "s1",
               "period": ["2025-01-01", "2025-12-30"]},
    "references": {"truth": "osse", "reanalysis": "smoothed-50km",
                   "observations": "s1"},
    "metrics": [{"name": "rmse", "dims": ["time", "y", "x"]}],
    "baselines": [{"name": "linear_in_time"},
                  {"name": "oi", "length_km": 100, "time_days": 30,
                   "window_days": 15}],
}


def contract_hash(c: dict) -> str:
    # h = SHA-256 of the key-sorted JSON
    return sha256_hex(stable_json(c))


h0 = contract_hash(contract)
reordered = dict(reversed(list(contract.items())))
edited = {**contract, "splits": {**contract["splits"], "platform": "s2"}}
report_fields = [f.name for f in dataclasses.fields(
    importlib.import_module("pipekit_evaluate").BenchmarkReport)]
h0[:12], contract_hash(reordered) == h0, contract_hash(edited)[:12], report_fields

# %%
assert contract_hash(reordered) == h0 and contract_hash(edited) != h0
assert len(h0) == 64 and h0[:12] == "0335f1032c69"
assert report_fields == ["task", "domain", "axis", "results"]

# %% [markdown]
# ## Stage 2 · One method, six splits
#
# The method is inverse-distance weighting of the k = 10 nearest observations
# in scaled coordinates (x/L, y/L, t/τ). It is fitted on the training
# observations and scored on the held-out ones, against the noisy observation
# and against the true field at the same place and time.

# %%
coords = np.c_[x[xi], y[yi], ti.astype(float)]  # (n_obs, 3): km, km, day
scale = np.array([1 / L, 1 / L, 1 / TAU])
box = np.array([NX * DX / L, NY * DX / L, 1e9])  # periodic in x and y


def knn(train: np.ndarray, test: np.ndarray, k: int = 10) -> np.ndarray:
    """user: xrtoolz.interpolate.idw_to_points weighs in (lon, lat) only; this
    needs scaled (x, y, t) on a periodic domain."""
    # ŷ = Σ wᵢ yᵢ / Σ wᵢ,  wᵢ = 1 / dᵢ over the k nearest training obs
    tree = cKDTree(coords[train] * scale, boxsize=box)
    d, j = tree.query(coords[test] * scale, k=k)
    w = 1.0 / np.maximum(d, 1e-6)
    return (yobs[train][j] * w).sum(1) / w.sum(1)


def score(train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    pred = knn(train, test)
    e_obs = np.sqrt(np.mean((pred - yobs[test]) ** 2))
    e_true = np.sqrt(np.mean((pred - eta[ti[test], yi[test], xi[test]]) ** 2))
    return float(e_obs), float(e_true)


def blocks(km: float, days: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ids = ((ti // days) * 1000 + y[yi] // km) * 1000 + x[xi] // km
    u = np.unique(ids)
    test_blocks = u[np.random.default_rng(seed).permutation(u.size)[: u.size // 5]]
    is_test = np.isin(ids, test_blocks)
    return np.nonzero(~is_test)[0], np.nonzero(is_test)[0]


r = np.random.default_rng(1).random(ti.size)
splits = {
    "random 80/20": (np.nonzero(r >= 0.2)[0], np.nonzero(r < 0.2)[0]),
    "withhold s1": (np.nonzero(sat != "s1")[0], np.nonzero(sat == "s1")[0]),
    "withhold s2": (np.nonzero(sat != "s2")[0], np.nonzero(sat == "s2")[0]),
    "blocks 50 km × 7 d": blocks(50, 7),
    "blocks 200 km × 60 d": blocks(200, 60),
    "last 20 % of time": (np.nonzero(ti < 0.8 * T)[0], np.nonzero(ti >= 0.8 * T)[0]),
}
s2 = {k: score(*v) for k, v in splits.items()}
{k: tuple(round(e, 4) for e in v) for k, v in s2.items()}

# %% [markdown]
# The training loader's own split: `XarrayWindowDataset` splits its origins
# contiguously 80/10/10 and drops a guard band as wide as the stencil's reach
# (finding A5 of Report 11: the module imports `geopatcher.time`, which the
# pinned geopatcher ships as `geopatcher.temporal.stencils`).

# %%
try:
    importlib.import_module("geopatcher.time")
except ImportError:
    import geopatcher.temporal.stencils as _stencils

    sys.modules["geopatcher.time"] = _stencils
from geopatcher.time import TimeStencil  # noqa: E402
from pipekit_train import XarrayWindowDataset  # noqa: E402

cell = truth.isel(y=[0], x=[0]).squeeze("y").drop_vars("y").to_dataset()
stencils = {"input": TimeStencil("-2D", "0D", "1D", closed="both"),
            "target": TimeStencil("1D", "1D", "0D", closed="both")}
win = {g: XarrayWindowDataset(source={"input": cell, "target": cell},
                              stencils=stencils, store_version="v1", split_guard=g)
       for g in (None, 60)}
guard = {g: (w._split_guard, [len(w.with_split(s)) for s in ("train", "val", "test")])
         for g, w in win.items()}
# the guard drops g origins on each side of a boundary: targets 2g + 1 days apart
rho_guard = float(np.exp(-0.5 * (2 * guard[None][0] + 1) ** 2 / TAU**2))
guard, rho_guard

# %%
r2 = {k: (round(a, 4), round(b, 4)) for k, (a, b) in s2.items()}
assert r2 == {
    "random 80/20": (0.0262, 0.0168), "withhold s1": (0.0336, 0.027),
    "withhold s2": (0.0273, 0.0188), "blocks 50 km × 7 d": (0.0267, 0.018),
    "blocks 200 km × 60 d": (0.043, 0.0383), "last 20 % of time": (0.1056, 0.1036),
}
assert guard[None] == (2, [871, 105, 108])
assert guard[60] == (60, [813, 0, 50])  # the validation split is empty
assert round(rho_guard, 3) == 0.986

# %% [markdown]
# ## Stage 3 · The baselines a contract must pin
#
# The input is the 2025 maps gridded from s2 and s3 (NaN where neither flew);
# each baseline fills them, and is scored against the true 2025 field and
# against the withheld s1 observations.

# %%
test_days = np.nonzero(times.year == 2025)[0]
inp = sat != "s1"
grid = np.full((T, NY, NX), np.nan)
grid[ti[inp], yi[inp], xi[inp]] = yobs[inp]
G = truth.copy(data=grid)
truth25 = truth.isel(time=test_days)
held = (sat == "s1") & np.isin(ti, test_days)


def rmse_truth(p: xr.DataArray) -> float:
    return float(np.sqrt(((p.isel(time=test_days) - truth25) ** 2).mean()))


def rmse_s1(p: xr.DataArray) -> float:
    v = p.values[ti[held], yi[held], xi[held]]
    return float(np.sqrt(np.mean((v - yobs[held]) ** 2)))


carry = pd.DataFrame(grid.reshape(T, -1)).ffill().fillna(0.0).to_numpy()
base = {
    "carry forward": G.copy(data=carry.reshape(T, NY, NX)),
    "nearest in time": fillnan_temporal(G, method="nearest").fillna(0.0),
    "linear in time": fillnan_temporal(G, method="linear").fillna(0.0),
    "monthly climatology": fillnan_climatology(G, group="month",
                                               residual="zero").fillna(0.0),
}


def oi(length: float, window: int = 15) -> tuple[xr.DataArray, xr.DataArray]:
    """Space-time optimal interpolation, one solve per day over ±window days.

    user: the stack has no OI; workstream C1 proposes it in xrtoolz.interpolate.
    """
    mean, var = np.zeros((T, NY, NX)), np.zeros((T, NY, NX))
    gx, gy = (a.ravel() for a in np.meshgrid(x, y))
    P = NX * DX
    o_t, o_y, o_x, o_v = ti[inp], y[yi[inp]], x[xi[inp]], yobs[inp]

    def d2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dd = np.abs(a[:, None] - b[None, :])
        return np.minimum(dd, P - dd) ** 2

    for day in test_days:
        s = np.abs(o_t - day) <= window
        ot, oy, ox, ov = o_t[s], o_y[s], o_x[s], o_v[s]
        # C_oo = σ² exp(−r²/2L² − Δt²/2τ²) + σₒ² I               (M, M)
        Coo = SIG**2 * np.exp(-0.5 * (d2(oy, oy) + d2(ox, ox)) / length**2
                              - 0.5 * (ot[:, None] - ot[None, :]) ** 2 / TAU**2)
        Coo += SIG_O**2 * np.eye(ot.size)
        # C_go between the N grid cells and the M obs                (N, M)
        Cgo = SIG**2 * np.exp(-0.5 * (d2(gy, oy) + d2(gx, ox)) / length**2
                              - 0.5 * (day - ot[None, :]) ** 2 / TAU**2)
        mu = ov.mean()
        # x̂ = μ + C_go C_oo⁻¹ (y − μ);  σ̂² = σ² − diag(C_go C_oo⁻¹ C_og)
        mean[day] = (mu + Cgo @ np.linalg.solve(Coo, ov - mu)).reshape(NY, NX)
        var[day] = (SIG**2 - np.einsum("ij,ji->i", Cgo,
                                       np.linalg.solve(Coo, Cgo.T))).reshape(NY, NX)
    return truth.copy(data=mean), truth.copy(data=var)


ois = {length: oi(length) for length in (50.0, 100.0)}


def min_eig(length: float, day: int = 800, window: int = 15) -> float:
    """Smallest eigenvalue of C_oo on one day: is the kernel valid on the torus?"""
    P = NX * DX
    s = (np.abs(ti - day) <= window) & inp
    oy, ox, ot = y[yi[s]], x[xi[s]], ti[s]
    dy = np.minimum(np.abs(oy[:, None] - oy), P - np.abs(oy[:, None] - oy))
    dx_ = np.minimum(np.abs(ox[:, None] - ox), P - np.abs(ox[:, None] - ox))
    C = SIG**2 * np.exp(-0.5 * (dy**2 + dx_**2) / length**2
                        - 0.5 * (ot[:, None] - ot[None, :]) ** 2 / TAU**2)
    return float(np.linalg.eigvalsh(C + SIG_O**2 * np.eye(ot.size)).min())


eig = {length: min_eig(length) for length in (100.0, 150.0)}
base["OI, L = 50 km"] = ois[50.0][0]
base["OI, L = 100 km"] = ois[100.0][0]
s3 = {k: (rmse_truth(p), rmse_s1(p)) for k, p in base.items()}
oi100 = s3["OI, L = 100 km"][0]
gain = {k: 1 - oi100 / s3[k][0]
        for k in ("carry forward", "nearest in time", "linear in time")}
{k: tuple(round(e, 4) for e in v) for k, v in s3.items()}, gain, eig

# %%
r3 = {k: (round(a, 4), round(b, 4)) for k, (a, b) in s3.items()}
assert r3 == {
    "carry forward": (0.0796, 0.0741), "nearest in time": (0.0576, 0.058),
    "linear in time": (0.0497, 0.0518), "monthly climatology": (0.084, 0.0792),
    "OI, L = 50 km": (0.0267, 0.0322), "OI, L = 100 km": (0.0178, 0.0259),
}
assert round(eig[100.0], 6) == 0.000391
assert round(eig[150.0], 4) == -0.0014  # not positive definite
assert {k: round(v, 2) for k, v in gain.items()} == {
    "carry forward": 0.78, "nearest in time": 0.69, "linear in time": 0.64}

# %% [markdown]
# ## Stage 4 · A deterministic and an ensemble entry under CRPS

# %%
m25 = ois[100.0][0].isel(time=test_days)
s25 = np.sqrt(np.maximum(ois[100.0][1].isel(time=test_days), 0.0))
# xᵢ = x̂ + σ̂ εᵢ, εᵢ ~ 𝒩(0, 1), N_e = 20 members (marginals only)
eps = np.random.default_rng(5).standard_normal((20, *m25.shape))
ens = m25 + s25 * xr.DataArray(eps, dims=("member", *m25.dims))
dims = ("time", "y", "x")
crps_ens = float(crps_ensemble(ens, truth25, dims=dims))
try:
    crps_ensemble(m25.expand_dims(member=1), truth25, dims=dims)
    one_member = "accepted"
except ValueError as e:
    one_member = str(e)
crps_dup = float(crps_ensemble(xr.concat([m25, m25], "member"), truth25, dims=dims))
mae = float(abs(m25 - truth25).mean())
ssr = float(spread_skill_ratio(ens, truth25, dims=dims))
crps_ens, one_member, crps_dup, mae, ssr

# %%
assert round(crps_ens, 4) == 0.0094 and round(ssr, 2) == 0.98
assert one_member == "Ensemble/sample dim 'member' has size 1; need at least 2."
assert round(crps_dup, 4) == round(mae, 4) == 0.0126  # CRPS of a point = MAE

# %% [markdown]
# ## Stage 5 · Three references
#
# The reanalysis stand-in is the true field smoothed with a Gaussian of 50 km
# (2 cells); "learned the reanalysis" is the L = 100 km OI map smoothed the
# same way, as a model trained against that product would be.

# %%
rean = truth.copy(data=gaussian_filter(eta, sigma=(0, 2, 2), mode="wrap"))
rean25 = rean.isel(time=test_days)
smooth = ois[100.0][0].copy(data=gaussian_filter(ois[100.0][0].values,
                                                 sigma=(0, 2, 2), mode="wrap"))
entries = {"OI, L = 100 km": ois[100.0][0], "OI, L = 50 km": ois[50.0][0],
           "learned the reanalysis": smooth}
tracks = {k: (float(np.sqrt(((p.isel(time=test_days) - rean25) ** 2).mean())),
              rmse_truth(p), rmse_s1(p)) for k, p in entries.items()}
{k: tuple(round(e, 4) for e in v) for k, v in tracks.items()}

# %%
r5 = {k: tuple(round(e, 4) for e in v) for k, v in tracks.items()}
assert r5["OI, L = 100 km"] == (0.0223, 0.0178, 0.0259)
assert r5["OI, L = 50 km"] == (0.0283, 0.0267, 0.0322)
assert r5["learned the reanalysis"] == (0.0122, 0.0266, 0.0325)
best = {i: min(r5, key=lambda k: r5[k][i]) for i in range(3)}
assert best == {0: "learned the reanalysis", 1: "OI, L = 100 km",
                2: "OI, L = 100 km"}

# %% [markdown]
# ## Stage 6 · The runner pipekit-evaluate ships

# %%
from pipekit_evaluate import Benchmark, Estimate, Estimator, Task  # noqa: E402


@dataclass
class Field:  # user: an Estimate over a map
    mean: xr.DataArray

    def point(self) -> xr.DataArray:
        return self.mean

    def sample(self, n: int, *, key: Any = None) -> None:
        return None

    def log_prob(self, theta: Any) -> Any:
        raise NotImplementedError


@dataclass
class MappingTask:  # user: the Task protocol over this benchmark
    name: str = "ssh-mapping-synthetic"
    domain: str = "ocean"

    def prior(self) -> Any:
        return None

    def simulator(self) -> Any:
        return None

    def observation_operator(self) -> Any:
        return None

    def datasets(self) -> dict[str, Any]:
        return {"obs": G, "theta": truth25}


@dataclass
class Baseline:  # user: a precomputed map as an Estimator
    label: str
    rung: int
    complexity: int

    def fit(self, task: Task) -> Baseline:
        return self

    def __call__(self, obs: Any) -> Estimate:
        return Field(base[self.label].isel(time=test_days))


@dataclass
class RMSEScorer:  # user
    name: str = "rmse"

    def __call__(self, prediction: Estimate, reference: Any) -> dict[str, float]:
        return {"truth": float(np.sqrt(((prediction.point() - reference) ** 2).mean()))}


ests = [Baseline("linear in time", 2, 0), Baseline("OI, L = 50 km", 2, 1),
        Baseline("OI, L = 100 km", 2, 2)]
protocols = (isinstance(MappingTask(), Task), all(isinstance(e, Estimator) for e in ests))
rep = Benchmark(task=MappingTask(), estimators=ests, scorers=[RMSEScorer()]).run()
cells = {r.cell.label: (r.reference.value, round(r.scores["rmse.truth"], 4))
         for r in rep.results}
try:
    Benchmark(task=MappingTask(), estimators=[*ests, Baseline("nearest in time", 2, 0)],
              scorers=[RMSEScorer()]).run()
    duplicate = ""
except ValueError as e:
    duplicate = str(e)
protocols, cells, duplicate

# %%
assert protocols == (True, True)
assert cells == {"L4/r2/c0": ("known_theta", 0.0497),
                 "L4/r2/c1": ("known_theta", 0.0267),
                 "L4/r2/c2": ("known_theta", 0.0178)}
assert duplicate.startswith("duplicate benchmark cells (rung, complexity): [(2, 0)]")
