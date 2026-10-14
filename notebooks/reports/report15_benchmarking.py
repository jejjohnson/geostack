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
import inspect
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from pipekit.hashing import sha256_hex, stable_json
import scipy
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from xrtoolz.interpolate import fillnan_climatology, fillnan_temporal
from xrtoolz.metrics import crps_ensemble, mae, rmse, spread_skill_ratio

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
# Space is periodic; time is not (the FFT is padded by a year and cropped).

# %%
# Every setting of the synthetic data in one mapping: the contract hashes it
GEN = {"nx": 32, "dx_km": 25.0, "days": 3 * 365, "start": "2023-01-01",
       "length_km": 100.0, "time_scale_days": 30.0, "sigma_m": 0.08,
       "seasonal_amplitude_m": 0.04, "obs_noise_m": 0.02, "revisit_days": 3,
       "satellites": {"s1": [1, 3], "s2": [-1, 5], "s3": [1, 11]},
       "time_padding_days": 365, "seed": 0}
NX = NY = GEN["nx"]
DX = GEN["dx_km"]
T = GEN["days"]
L, TAU = GEN["length_km"], GEN["time_scale_days"]
SIG, AMP, SIG_O = GEN["sigma_m"], GEN["seasonal_amplitude_m"], GEN["obs_noise_m"]
rng = np.random.default_rng(GEN["seed"])

# C(r, Δt) = σ² exp(−r² / 2L² − Δt² / 2τ²)  ⇔  spectrum ∝ exp(−L²|k|²/2 − τ²ω²/2)
# Space is periodic by design. Time is not: the FFT runs over T + pad days and
# the first T are kept, so the wrapped lag between the two ends is ≥ pad = 12τ
# and exp(−pad²/2τ²) ≈ 0; 30 Dec 2025 is uncorrelated with 1 Jan 2023.
TP = T + GEN["time_padding_days"]
kx = 2 * np.pi * np.fft.fftfreq(NX, d=DX)
kt = 2 * np.pi * np.fft.fftfreq(TP, d=1.0)
KT, KY, KX = np.meshgrid(kt, kx, kx, indexing="ij")
spectrum = np.exp(-0.5 * L**2 * (KX**2 + KY**2) - 0.5 * TAU**2 * KT**2)
anom = np.real(np.fft.ifftn(np.fft.fftn(rng.standard_normal((TP, NY, NX)))
                            * np.sqrt(spectrum)))[:T]  # (T, NY, NX)
anom *= SIG / anom.std()
times = pd.date_range(GEN["start"], periods=T, freq="D")
season = AMP * np.cos(2 * np.pi * (times.dayofyear.values - 60) / 365.25)
eta = anom + season[:, None, None]  # (T, NY, NX)
x = np.arange(NX) * DX
y = np.arange(NY) * DX
truth = xr.DataArray(eta, dims=("time", "y", "x"),
                     coords={"time": times, "y": y, "x": x}, name="ssh")

# satellite s sees cell (i, j) on day t when (j + dₛ i − vₛ t) mod 32 = 0, every 3 days
SATS = {k: tuple(v) for k, v in GEN["satellites"].items()}  # direction, drift
tt, ii, jj = np.meshgrid(np.arange(T), np.arange(NY), np.arange(NX), indexing="ij")
rows = []
for name, (d, v) in SATS.items():
    on = ((jj + d * ii - v * tt) % NX == 0) & (tt % GEN["revisit_days"] == 0)
    t_, i_, j_ = np.nonzero(on)
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
assert round(field["rho30"], 2) == 0.57  # exp(−½) = 0.61; one 3-year sample

# %% [markdown]
# ## Stage 1 · The contract as data, and its hash

# %%
# The OI baseline's code, defined here so that the contract can hash its source:
# editing oi or the kernel changes the contract. It reads the data (grid,
# test_days) at call time, in stage 3.
P = NX * DX


def k_min_image(a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    """exp(−d²/2L²) with d the shortest distance on the circle: not a valid
    covariance on a periodic domain."""
    dd = np.abs(a[:, None] - b[None, :])
    return np.exp(-0.5 * np.minimum(dd, P - dd) ** 2 / length**2)


def k_wrapped(a: np.ndarray, b: np.ndarray, length: float,
              images: tuple[int, int] = (-3, 3)) -> np.ndarray:
    """Σₙ exp(−(d + nP)²/2L²): the Gaussian summed over periodic images, a valid
    covariance on the circle (the periodization of a positive-definite function)."""
    d = a[:, None] - b[None, :]
    lo, hi = images
    return sum(np.exp(-0.5 * (d + n * P) ** 2 / length**2) for n in range(lo, hi + 1))


def oi(b: dict) -> tuple[xr.DataArray, xr.DataArray]:
    """Space-time optimal interpolation, one solve per day over ±window days,
    with every setting read from the contract's baseline entry `b`.

    user: the stack has no OI; workstream C1 proposes it in xrtoolz.interpolate.
    """
    assert b["space_kernel"] == "gaussian_periodic_images"
    assert b["time_kernel"] == "gaussian" and b["mean"] == "window_obs_mean"
    length, tau, window = float(b["length_km"]), b["time_days"], b["window_days"]
    sig, sig_o, imgs = b["prior_sigma_m"], b["noise_m"], tuple(b["images"])

    def k(a: np.ndarray, c: np.ndarray) -> np.ndarray:
        return k_wrapped(a, c, length, imgs)

    mean, var = np.zeros((364, NY, NX)), np.zeros((364, NY, NX))
    gx, gy = (a.ravel() for a in np.meshgrid(x, y))
    # the declared input: the non-NaN cells of the sparse grid G, as every
    # other baseline sees it (coincident s2/s3 samples already averaged)
    c_t, c_i, c_j = np.nonzero(~np.isnan(grid))
    o_t, o_y, o_x, o_v = c_t, y[c_i], x[c_j], grid[c_t, c_i, c_j]
    for n, day in enumerate(test_days):
        s = np.abs(o_t - day) <= window
        ot, oy, ox, ov = o_t[s], o_y[s], o_x[s], o_v[s]
        # C_oo = σ² k(y) k(x) exp(−Δt²/2τ²) + σₒ² I                (M, M)
        Coo = (sig**2 * k(oy, oy) * k(ox, ox)
               * np.exp(-0.5 * (ot[:, None] - ot[None, :]) ** 2 / tau**2))
        Coo += sig_o**2 * np.eye(ot.size)
        # C_go between the N grid cells and the M obs                (N, M)
        Cgo = (sig**2 * k(gy, oy) * k(gx, ox)
               * np.exp(-0.5 * (day - ot[None, :]) ** 2 / tau**2))
        mu = ov.mean()
        # x̂ = μ + C_go C_oo⁻¹ (y − μ);  σ̂² = σ² k(0)² − diag(C_go C_oo⁻¹ C_og)
        mean[n] = (mu + Cgo @ np.linalg.solve(Coo, ov - mu)).reshape(NY, NX)
        prior = sig**2 * k(np.zeros(1), np.zeros(1))[0, 0] ** 2
        var[n] = (prior - np.einsum("ij,ji->i", Cgo,
                                    np.linalg.solve(Coo, Cgo.T))).reshape(NY, NX)
    return truth25.copy(data=mean), truth25.copy(data=var)


def source_hash(*fns) -> str:
    # h = SHA-256 of the functions' source: an immutable identity for code
    # that lives in this notebook until workstream C1 moves it upstream
    return sha256_hex("\n".join(inspect.getsource(fn) for fn in fns))


# %%
# Each baseline with every setting that changes its score: the method, its
# arguments, the fill for cells with no observation before or after, and the
# code that implements it. Stage 3 reads its settings from here.
CODE = {"xrtoolz": "308c7f2", "scipy": scipy.__version__,
        "oi": "notebook source sha256:" + source_hash(oi, k_wrapped)}
EDGE = {"edge_fill_m": 0.0}  # cells with no observation on the needed side
BASELINES = [
    {"name": "persistence", "method": "carry_forward", "reference": "last_obs",
     "code": "pandas.DataFrame.ffill", **EDGE},
    {"name": "nearest_in_time", "method": "nearest",
     "code": f"xrtoolz.interpolate.fillnan_temporal@{CODE['xrtoolz']}", **EDGE},
    {"name": "linear_in_time", "method": "linear",
     "code": f"xrtoolz.interpolate.fillnan_temporal@{CODE['xrtoolz']}", **EDGE},
    {"name": "monthly_climatology", "group": "month", "residual": "zero",
     "code": f"xrtoolz.interpolate.fillnan_climatology@{CODE['xrtoolz']}", **EDGE},
    *({"name": f"oi_{ln}km", "length_km": ln, "time_days": 30, "window_days": 15,
       "prior_sigma_m": 0.08, "noise_m": 0.02, "mean": "window_obs_mean",
       "space_kernel": "gaussian_periodic_images", "images": [-3, 3],
       "time_kernel": "gaussian", "code": CODE["oi"]} for ln in (50, 100)),
]
BL = {b["name"]: b for b in BASELINES}
# every reference derived from the generator, with the derivation's settings
REFS = {
    "truth": {"from": "data.generator", "period": ["2025-01-01", "2025-12-30"]},
    "reanalysis": {"from": "truth", "op": "scipy.ndimage.gaussian_filter",
                   "sigma_cells": [0, 2, 2], "mode": "wrap",
                   "code": f"scipy@{CODE['scipy']}"},
    "observations": {"from": "data.generator", "platform": "s1",
                     "period": ["2025-01-01", "2025-12-30"]},
}
# grids are scored cell by cell; the s1 track is matched up first: the map is
# sampled at each observation's (time, y, x) cell and reduced over the obs
SCORING = {
    "truth": {"carrier": "grid", "dims": ["time", "y", "x"]},
    "reanalysis": {"carrier": "grid", "dims": ["time", "y", "x"]},
    "observations": {"carrier": "points", "matchup": "cell_of_obs",
                     "match_on": ["time", "y", "x"], "dims": ["obs"]},
}

contract = {
    "name": "ssh-mapping-synthetic", "version": "1.0.0",
    "task": "L1->L2", "operation": "gap filling", "variant": "deterministic",
    "input": {"carrier": "sparse grid", "from": "along-track obs binned to the grid",
              "aggregate": "mean",  # coincident s2/s3 samples in one cell
              "vars": ["ssh"], "units": "m", "sources": ["s2", "s3"]},
    "output": {"carrier": "grid", "dims": ["time", "y", "x"],
               "shape": [364, 32, 32], "units": "m"},
    "splits": {"kind": "leave_one_platform_out", "platform": "s1",
               "period": ["2025-01-01", "2025-12-30"]},
    "references": REFS,
    "data": {"generator": GEN},  # labels alone do not identify the arrays
    "metrics": [{"name": "rmse", "code": f"xrtoolz.metrics.rmse@{CODE['xrtoolz']}"}],
    "scoring": SCORING,  # how each track is matched and reduced
    "baselines": BASELINES,  # every setting that changes a score
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
assert len(h0) == 64 and h0[:12] == "4f0a8b663178"
# the generator is part of the contract: another seed is another benchmark
reseeded = {**contract, "data": {"generator": {**GEN, "seed": 1}}}
assert contract_hash(reseeded) != h0
# so is every baseline setting that changes a score: the kernel, the edge fill
rekerneled = {**contract, "baselines": [
    {**b, "space_kernel": "gaussian_min_image"} if b["name"].startswith("oi") else b
    for b in BASELINES]}
refilled = {**contract, "baselines": [
    {**b, "edge_fill_m": None} if "edge_fill_m" in b else b for b in BASELINES]}
assert len({h0, contract_hash(rekerneled), contract_hash(refilled)}) == 3
# the OI's code is content-addressed: its source hash is in the contract
assert all(b["code"] == CODE["oi"] for b in BASELINES if b["name"].startswith("oi"))
assert CODE["oi"].startswith("notebook source sha256:")
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
# the contract's period: every split is drawn inside 2025, on both sides
in25 = times.year.values[ti] == 2025
d25 = ti[in25].min()  # first day of 2025


def obs_da(v: np.ndarray) -> xr.DataArray:
    return xr.DataArray(v, dims="obs")


def knn(train: np.ndarray, test: np.ndarray, k: int = 10) -> np.ndarray:
    """user: xrtoolz.interpolate.idw_to_points weighs in (lon, lat) only; this
    needs scaled (x, y, t) on a periodic domain."""
    # ŷ = Σ wᵢ yᵢ / Σ wᵢ,  wᵢ = 1 / dᵢ over the k nearest training obs
    tree = cKDTree(coords[train] * scale, boxsize=box)
    d, j = tree.query(coords[test] * scale, k=k)
    w = 1.0 / np.maximum(d, 1e-6)
    return (yobs[train][j] * w).sum(1) / w.sum(1)


def score(train: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    pred = obs_da(knn(train, test))
    e_obs = rmse(pred, obs_da(yobs[test]), dim="obs")
    e_true = rmse(pred, obs_da(eta[ti[test], yi[test], xi[test]]), dim="obs")
    return float(e_obs), float(e_true)


def where(mask: np.ndarray) -> np.ndarray:
    return np.nonzero(mask & in25)[0]


def blocks(km: float, days: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ids = (((ti - d25) // days) * 1000 + y[yi] // km) * 1000 + x[xi] // km
    u = np.unique(ids[in25])
    test_blocks = u[np.random.default_rng(seed).permutation(u.size)[: u.size // 5]]
    is_test = np.isin(ids, test_blocks)
    return where(~is_test), where(is_test)


r = np.random.default_rng(1).random(ti.size)
cut = d25 + int(0.8 * 364)  # day 291 of 2025: the test set is the last 73 days
splits = {
    "random 80/20": (where(r >= 0.2), where(r < 0.2)),
    "withhold s1": (where(sat != "s1"), where(sat == "s1")),
    "withhold s2": (where(sat != "s2"), where(sat == "s2")),
    "blocks 50 km × 7 d": blocks(50, 7),
    "blocks 200 km × 60 d": blocks(200, 60),
    "last 20 % of 2025": (where(ti < cut), where(ti >= cut)),
}
n25 = int(in25.sum())
s2 = {k: score(*v) for k, v in splits.items()}
n25, {k: tuple(round(e, 4) for e in v) for k, v in s2.items()}

# %% [markdown]
# The training loader's own split: `XarrayWindowDataset` splits its origins
# contiguously 80/10/10 and drops a guard band as wide as the stencil's reach
# (finding A5 of Report 11: the module imports `geopatcher.time`, which the
# pinned geopatcher ships as `geopatcher.temporal.stencils`).

# %%
# Report 11's A5: pipekit-train imports geopatcher.time, which the pinned
# geopatcher ships as geopatcher.temporal.stencils. Alias it only when that
# exact module is absent, so any other import error still surfaces. The fix
# belongs upstream in pipekit-train; this cell goes when it lands.
try:
    importlib.import_module("geopatcher.time")
except ModuleNotFoundError as err:
    if err.name != "geopatcher.time":
        raise
    import geopatcher.temporal.stencils as _stencils

    sys.modules["geopatcher.time"] = _stencils
from geopatcher.time import TimeStencil  # noqa: E402
try:  # Report 16's D2 moves the dataset; fall back to the pinned location
    from geopatcher.integrations.train import XarrayWindowDataset
except ModuleNotFoundError as err:
    if err.name != "geopatcher.integrations.train":
        raise
    from pipekit_train import XarrayWindowDataset

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
assert n25 == 11_616  # observations in 2025, the contract's period
assert d25 + 364 - cut == 73  # the causal test set: the last 73 days of 2025
assert r2 == {
    "random 80/20": (0.0262, 0.0173), "withhold s1": (0.0332, 0.027),
    "withhold s2": (0.0275, 0.0192), "blocks 50 km × 7 d": (0.0261, 0.0168),
    "blocks 200 km × 60 d": (0.0458, 0.0418), "last 20 % of 2025": (0.0865, 0.0839),
}
# against the error in use (withheld s1): random optimistic, big blocks pessimistic
use = s2["withhold s1"][1]
assert round(1 - s2["random 80/20"][1] / use, 2) == 0.36
assert round(s2["blocks 200 km × 60 d"][1] / use - 1, 2) == 0.55
assert round(max(v[1] for v in s2.values()) / min(v[1] for v in s2.values())) == 5
assert all(np.all(times.year.values[ti[i]] == 2025)
           for tr_te in splits.values() for i in tr_te)
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
# bin to the grid, averaging coincident s2/s3 samples (the contract's "aggregate")
assert contract["input"]["aggregate"] == "mean"
cells = pd.DataFrame({"t": ti[inp], "i": yi[inp], "j": xi[inp], "v": yobs[inp]}
                     ).groupby(["t", "i", "j"]).v
binned, n_per_cell = cells.mean(), cells.size()
grid[tuple(np.array(binned.index.to_list()).T)] = binned.to_numpy()
# cells of 2025 where s2 and s3 coincide: one averaged value each
n_shared25 = int(((n_per_cell > 1)
                  & (n_per_cell.index.get_level_values("t") >= test_days[0])).sum())
G = truth.copy(data=grid)  # the L1 sparse grid, all three years
assert n_shared25 == 242  # where s2 and s3 coincide; OI and the rest see one value
truth25 = truth.isel(time=test_days)  # (364, 32, 32), the contract's output
held = (sat == "s1") & in25
# withheld s1 observations, indexed in the 2025 maps' own time coordinate
held_t = xr.DataArray(times[ti[held]], dims="obs")
held_y = xr.DataArray(y[yi[held]], dims="obs")
held_x = xr.DataArray(x[xi[held]], dims="obs")
grid_dims = contract["scoring"]["truth"]["dims"]
obs_dims = contract["scoring"]["observations"]["dims"]


def rmse_truth(p: xr.DataArray) -> float:
    return float(rmse(p, truth25, dim=grid_dims))


def rmse_s1(p: xr.DataArray) -> float:
    v = p.sel(time=held_t, y=held_y, x=held_x)
    return float(rmse(v, obs_da(yobs[held]), dim=obs_dims))


def in2025(p: xr.DataArray) -> xr.DataArray:
    return p.isel(time=test_days)


def edge(p: xr.DataArray, b: dict) -> xr.DataArray:
    return p.fillna(b["edge_fill_m"])


carry = pd.DataFrame(grid.reshape(T, -1)).ffill().to_numpy()
mc = BL["monthly_climatology"]
base = {
    "persistence (carry forward)": in2025(edge(
        G.copy(data=carry.reshape(T, NY, NX)), BL["persistence"])),
    "nearest in time": in2025(edge(fillnan_temporal(
        G, method=BL["nearest_in_time"]["method"]), BL["nearest_in_time"])),
    "linear in time": in2025(edge(fillnan_temporal(
        G, method=BL["linear_in_time"]["method"]), BL["linear_in_time"])),
    "monthly climatology": in2025(edge(fillnan_climatology(
        G, group=mc["group"], residual=mc["residual"]), mc)),
}
ois = {ln: oi(BL[f"oi_{ln}km"]) for ln in (50, 100)}


def min_eig(kernel, length: float, day: int = 800, window: int = 15) -> float:
    """Smallest eigenvalue of the prior covariance on one day, before the noise
    σₒ² I is added: is the kernel a valid covariance on the torus?"""
    c_t, c_i, c_j = np.nonzero(~np.isnan(grid))
    s = np.abs(c_t - day) <= window
    oy, ox, ot = y[c_i[s]], x[c_j[s]], c_t[s]
    C = (SIG**2 * kernel(oy, oy, length) * kernel(ox, ox, length)
         * np.exp(-0.5 * (ot[:, None] - ot[None, :]) ** 2 / TAU**2))
    return float(np.linalg.eigvalsh(C).min())


eig = {(kern.__name__, length): min_eig(kern, length)
       for kern in (k_min_image, k_wrapped) for length in (100.0, 150.0)}
base["OI, L = 50 km"] = ois[50][0]
base["OI, L = 100 km"] = ois[100][0]
s3 = {k: (rmse_truth(p), rmse_s1(p)) for k, p in base.items()}
oi100 = s3["OI, L = 100 km"][0]
gain = {k: 1 - oi100 / s3[k][0]
        for k in ("persistence (carry forward)", "nearest in time", "linear in time")}
{k: tuple(round(e, 4) for e in v) for k, v in s3.items()}, gain, eig

# %%
r3 = {k: (round(a, 4), round(b, 4)) for k, (a, b) in s3.items()}
assert r3 == {
    "persistence (carry forward)": (0.0798, 0.0749), "nearest in time": (0.0559, 0.0572),
    "linear in time": (0.0477, 0.0502), "monthly climatology": (0.0835, 0.0787),
    "OI, L = 50 km": (0.0266, 0.032), "OI, L = 100 km": (0.0181, 0.0262),
}
# the minimum-image Gaussian is not a covariance on the torus, even at L = 100 km;
# the wrapped one is (smallest eigenvalue zero to rounding, relative to σ² = 0.0064)
assert f"{eig[('k_min_image', 100.0)]:.1e}" == "-9.1e-06"
assert f"{eig[('k_min_image', 150.0)]:.1e}" == "-1.7e-03"
# semidefinite to rounding, against a scale-aware tolerance (gate C1), not a
# Cholesky: the prior is only semidefinite
assert all(eig[("k_wrapped", ln)] >= -1e-10 * SIG**2 for ln in (100.0, 150.0))
assert {k: round(v, 2) for k, v in gain.items()} == {
    "persistence (carry forward)": 0.77, "nearest in time": 0.68,
    "linear in time": 0.62}
assert round(s3["OI, L = 50 km"][0] / oi100 - 1, 2) == 0.47  # half the right L

# %% [markdown]
# ## Stage 4 · A deterministic and an ensemble entry under CRPS

# %%
m25 = ois[100][0]
s25 = np.sqrt(np.maximum(ois[100][1], 0.0))
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
mae_ = float(mae(m25, truth25, dim=grid_dims))
ssr = float(spread_skill_ratio(ens, truth25, dims=dims))
crps_ens, one_member, crps_dup, mae_, ssr

# %%
assert round(crps_ens, 4) == 0.0096 and round(ssr, 2) == 0.96
assert one_member == "Ensemble/sample dim 'member' has size 1; need at least 2."
assert abs(crps_dup - mae_) < 1e-12  # CRPS of a point forecast is its MAE
assert round(mae_, 4) == 0.0129

# %% [markdown]
# ## Stage 5 · Three references
#
# The reanalysis stand-in is the true field smoothed with a Gaussian of 50 km
# (2 cells); "learned the reanalysis" is the L = 100 km OI map smoothed the
# same way, as a model trained against that product would be.

# %%
ra = contract["references"]["reanalysis"]
rean = truth.copy(data=gaussian_filter(eta, sigma=ra["sigma_cells"], mode=ra["mode"]))
rean25 = rean.isel(time=test_days)
smooth = ois[100][0].copy(data=gaussian_filter(ois[100][0].values,
                                                 sigma=ra["sigma_cells"], mode=ra["mode"]))
# every entry is a contract-shaped (364, 32, 32) map of 2025
entries = {"OI, L = 100 km": ois[100][0], "OI, L = 50 km": ois[50][0],
           "learned the reanalysis": smooth}
assert all(p.shape == (364, NY, NX) for p in entries.values())
tracks = {k: (float(rmse(p, rean25, dim=grid_dims)), rmse_truth(p), rmse_s1(p))
          for k, p in entries.items()}
{k: tuple(round(e, 4) for e in v) for k, v in tracks.items()}

# %%
r5 = {k: tuple(round(e, 4) for e in v) for k, v in tracks.items()}
assert r5["OI, L = 100 km"] == (0.0227, 0.0181, 0.0262)
assert r5["OI, L = 50 km"] == (0.0282, 0.0266, 0.032)
assert r5["learned the reanalysis"] == (0.0124, 0.0266, 0.0327)
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
        return Field(base[self.label])


@dataclass
class RMSEScorer:  # user
    name: str = "rmse"

    def __call__(self, prediction: Estimate, reference: Any) -> dict[str, float]:
        return {"truth": float(rmse(prediction.point(), reference, dim=grid_dims))}


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
assert cells == {"L4/r2/c0": ("known_theta", 0.0477),
                 "L4/r2/c1": ("known_theta", 0.0266),
                 "L4/r2/c2": ("known_theta", 0.0181)}
assert duplicate.startswith("duplicate benchmark cells (rung, complexity): [(2, 0)]")
