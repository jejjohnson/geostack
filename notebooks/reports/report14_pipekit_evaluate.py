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
# # Report 14 · pipekit-evaluate, run
#
# Regenerates every number in `docs/reports/14_pipekit_evaluate.md`: the
# double penalty of a shifted plume (stage 1), a scale-blind blur (stage 2),
# plume detection (stage 3) and a linear-Gaussian benchmark ladder on the
# shipped `pipekit_evaluate` API (stage 4). Every field is synthetic and every
# draw seeded. Each part ends in a cell asserting the numbers the page shows,
# to the precision it shows them.
#
# Pinned revisions: pipekit-evaluate 0.0.2 at pipekit `ef3c17f`, xrtoolz 0.0.5
# at `308c7f2`. It needs NumPy, SciPy, xarray, xrft, xskillscore, xrtoolz,
# pipekit-cycle and pipekit-evaluate. Cells marked `user` are code the report's
# workstreams move upstream (FSS, the spectrum gap, the glue of stage 4).

# %%
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.ndimage import gaussian_filter, uniform_filter
from xrtoolz.metrics import (
    RMSE,
    SSIM,
    EvaluateByRegion,
    FrequencyBandSkill,
    PhaseShiftError,
    centroid_displacement,
    instance_f1_at_iou,
    mask_iou_matrix,
    match_instances,
    psd_score,
    resolved_scale,
)
from xrtoolz.metrics.object import CriticalSuccessIndex, ProbabilityOfDetection
from xrtoolz.transforms import fit_spectral_slope, isotropic_power_spectrum
from xrtoolz.transforms import power_spectrum

PSD_KW = dict(window=None, detrend=None, window_correction=False)


def fss(p: np.ndarray, r: np.ndarray, thr: float, win: int) -> float:
    """Fractions Skill Score (Roberts and Lean 2008), periodic. user"""
    fp = uniform_filter((p >= thr).astype(float), win, mode="wrap")
    fr = uniform_filter((r >= thr).astype(float), win, mode="wrap")
    ref = np.mean(fp**2) + np.mean(fr**2)
    return float(1 - np.mean((fp - fr) ** 2) / ref) if ref > 0 else np.nan


def spectrum_gap(p: xr.DataArray, r: xr.DataArray, dims, iso=False) -> float:
    """Relative spectral-energy difference over k ≠ 0: blind to phase. user"""
    Pp = power_spectrum(p, dims, isotropic=iso, **PSD_KW)
    Pr = power_spectrum(r, dims, isotropic=iso, **PSD_KW)
    k = Pr[[d for d in Pr.dims if d.startswith("freq")][-1]]
    pos = abs(k) > 0
    return float(abs(Pp - Pr).where(pos).sum() / Pr.where(pos).sum())


# %% [markdown]
# ## Stage 1 · The double penalty: a plume shifted 6 cells against a flat field

# %%
N = 40
x = np.arange(N)


def plume1d(c: float) -> np.ndarray:
    d = (x - c + N / 2) % N - N / 2
    return 50 * np.exp(-0.5 * (d / 3) ** 2)


co1 = {"x": x.astype(float)}
truth = xr.DataArray(plume1d(8), dims="x", coords=co1, name="ch4")
A = xr.DataArray(plume1d(14), dims="x", coords=co1, name="ch4")
B = xr.full_like(truth, float(truth.mean()))

M = 64
yy, xx = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")


def plume2d(cy: float, cx: float) -> np.ndarray:
    dy, dx = (yy - cy + M / 2) % M - M / 2, (xx - cx + M / 2) % M - M / 2
    return 50 * np.exp(-0.5 * (dx**2 + dy**2) / 3**2)


co2 = {"y": np.arange(M, dtype=float), "x": np.arange(M, dtype=float)}
truth2 = xr.DataArray(plume2d(32, 8), dims=("y", "x"), coords=co2, name="ch4")
A2 = xr.DataArray(plume2d(32, 14), dims=("y", "x"), coords=co2, name="ch4")
B2 = xr.full_like(truth2, float(truth2.mean()))

s1 = {}
for nm, P, P2 in (("A", A, A2), ("B", B, B2)):
    ps = PhaseShiftError("ch4", "x", periodic=True)(P.to_dataset(), truth.to_dataset())
    sc = psd_score(P, truth, psd_dims=["x"], **PSD_KW)["score"]
    s1[nm] = {
        "rmse1": float(RMSE("ch4", dims="x")(P.to_dataset(), truth.to_dataset())),
        "shift": float(ps["shift_x"]), "residual": float(ps["residual_rmse"]),
        "psd_score": float(sc.where(sc.freq_x > 0).mean()),
        "fss1": fss(P.values, truth.values, 10, 9),
        "gap1": spectrum_gap(P, truth, "x"),
        "ssim2": float(SSIM("ch4", ["y", "x"])(P2.to_dataset(), truth2.to_dataset())),
        "rmse2": float(RMSE("ch4", dims=["y", "x"])(P2.to_dataset(),
                                                   truth2.to_dataset())),
        "fss2": fss(P2.values, truth2.values, 10, 9),
    }
s1

# %%
a, b = s1["A"], s1["B"]
assert (round(a["rmse1"], 2), round(b["rmse1"], 2)) == (20.50, 15.62)
assert (a["shift"], round(a["residual"], 6)) == (6.0, 0.0)
assert (b["shift"], round(b["residual"], 2)) == (0.0, 15.62)
assert (round(a["psd_score"], 3), round(b["psd_score"], 3)) == (-1.105, 0.0)
assert (round(a["fss1"], 3), b["fss1"]) == (0.625, 0.0)
assert (round(a["gap1"], 3), round(b["gap1"], 3)) == (0.0, 1.0)
assert (round(a["ssim2"], 3), round(b["ssim2"], 3)) == (0.885, 0.336)
assert (round(a["rmse2"], 3), round(b["rmse2"], 3)) == (4.671, 4.096)
assert round(a["fss2"], 3) == 0.566

# %% [markdown]
# ## Stage 2 · A scale-blind blur of a k⁻³ field

# %%
rng = np.random.default_rng(0)
n = 128
K = np.hypot(*np.meshgrid(np.fft.fftfreq(n), np.fft.fftfreq(n), indexing="ij"))
amp = np.zeros_like(K)
amp[K > 0] = K[K > 0] ** -2.0  # 2-D, E(k) ∝ k⁻³ ⇒ |F̂|² ∝ k⁻⁴
f = np.real(np.fft.ifft2(amp * np.exp(2j * np.pi * rng.random((n, n)))
                         * rng.standard_normal((n, n))))
f = (f - f.mean()) / f.std()
co = {"y": ("y", np.arange(n, dtype=float), {"units": "px"}),
      "x": ("x", np.arange(n, dtype=float), {"units": "px"})}
t = xr.Dataset({"z": (("y", "x"), f)}, coords=co)
blur = xr.Dataset({"z": (("y", "x"), gaussian_filter(f, 3, mode="wrap"))}, coords=co)

rmse_blur = float(RMSE("z", dims=["y", "x"])(blur, t)) / float(t.z.std())
ssim_blur = float(SSIM("z", ["y", "x"])(blur, t))
Pt = isotropic_power_spectrum(t.z, ["y", "x"], **PSD_KW)
Pb = isotropic_power_spectrum(blur.z, ["y", "x"], **PSD_KW)
slope_t, _ = fit_spectral_slope(Pt, k_min=0.05, k_max=0.15)
slope_b, _ = fit_spectral_slope(Pb, k_min=0.05, k_max=0.15)
lam = resolved_scale(psd_score(blur.z, t.z, psd_dims=["y", "x"], isotropic=True,
                               **PSD_KW), "freq_r", level=0.5)
# psd_score(k) = 1 − (1 − H)², H = exp(−2π²σ²k²); score = 0.5 ⇒ H = 1 − 1/√2
lam_analytic = 1 / np.sqrt(np.log(1 / (1 - 1 / np.sqrt(2))) / (2 * np.pi**2 * 9))

bands = {"large (>20 px)": (0.0, 0.05), "meso (8-20 px)": (0.05, 0.125),
         "small (<8 px)": (0.125, 0.5)}
band_rmse = FrequencyBandSkill("z", ["y", "x"], bands)(blur, t).z.values
zero = blur.copy()
zero["z"] = zero["z"] * 0.0
band_std = FrequencyBandSkill("z", ["y", "x"], bands)(zero, t).z.values
west = np.zeros((n, n), bool)
west[:, : n // 2] = True
regions = {name: xr.DataArray(m, dims=("y", "x"),
                              coords={"y": co["y"][1], "x": co["x"][1]})
           for name, m in (("west", west), ("east", ~west))}
by_region = EvaluateByRegion(RMSE("z", dims=["y", "x"]), regions=regions)(blur, t)
rmse_blur, ssim_blur, slope_t, slope_b, lam, band_rmse, band_rmse / band_std

# %%
assert round(rmse_blur, 3) == 0.098 and round(ssim_blur, 3) == 0.885
assert (round(slope_t, 2), round(slope_b, 2)) == (-3.10, -9.39)
assert round(lam, 2) == 11.98 and round(lam_analytic, 2) == 12.03
assert np.allclose(band_rmse.round(3), [0.053, 0.067, 0.049])
assert np.allclose((band_rmse / band_std).round(2), [0.05, 0.60, 0.99])
assert list(by_region.region.values) == ["west", "east"]

# %% [markdown]
# ## Stage 3 · Plume detection: one hit, one miss, one false alarm

# %%
def blob(cy: float, cx: float, s: float = 3) -> np.ndarray:
    return np.exp(-0.5 * ((yy - cy) ** 2 + (xx - cx) ** 2) / s**2)


ref_field = blob(16, 16) + blob(44, 40)
pred_field = blob(16, 19) + blob(50, 10)  # plume 1 shifted 3 px; plume 2 missed
lt, nt = ndimage.label(ref_field > 0.3)
lp, npred = ndimage.label(pred_field > 0.3)


def masks(lab: np.ndarray, k: int) -> np.ndarray:
    return np.stack([lab == i for i in range(1, k + 1)])


iou = mask_iou_matrix(masks(lp, npred), masks(lt, nt))
match = match_instances(masks(lp, npred), masks(lt, nt), iou_threshold=0.3)
f1 = instance_f1_at_iou(masks(lp, npred), masks(lt, nt), iou_threshold=0.3)
cd = centroid_displacement(
    xr.Dataset({"label": (("lat", "lon"), lp)}, coords={"lat": co2["y"], "lon": co2["x"]}),
    xr.Dataset({"label": (("lat", "lon"), lt)}, coords={"lat": co2["y"], "lon": co2["x"]}))
reserved = {}
for cls in (ProbabilityOfDetection, CriticalSuccessIndex):
    try:
        cls()
        reserved[cls.__name__] = ""
    except NotImplementedError as e:
        reserved[cls.__name__] = str(e)
iou.round(3), {k: match[k] for k in ("tp", "fp", "fn")}, f1["f1"], cd.distance.values

# %%
assert (nt, npred) == (2, 2)
assert np.allclose(iou.round(3), [[0.438, 0.0], [0.0, 0.0]])
assert (match["tp"], match["fp"], match["fn"]) == (1, 1, 1) and f1["f1"] == 0.5
assert round(float(cd.distance[1]), 1) == 30.6  # object 2 paired with object 2
assert all("name reservation; implementation lands with V5" in m
           for m in reserved.values())

# %% [markdown]
# ## Stage 4 · A linear-Gaussian benchmark ladder on the shipped API

# %%
from pipekit_cycle.protocols import ForwardModel, ObservationOperator  # noqa: E402
from pipekit_evaluate import (  # noqa: E402
    Axis,
    Benchmark,
    Estimate,
    Estimator,
    Prior,
    ReferenceRule,
    ReferenceSource,
    Scorer,
    Task,
)

NS, STRIDE, SIG_OBS, ELL, SEED = 40, 4, 0.3, 3.0, 0
idx = np.arange(NS)
Bc = np.exp(-0.5 * (idx[:, None] - idx[None, :]) ** 2 / ELL**2) + 1e-6 * np.eye(NS)
H = np.eye(NS)[::STRIDE]  # (10, 40)
R = SIG_OBS**2 * np.eye(H.shape[0])


@dataclass
class GaussPrior:  # user
    cov: np.ndarray

    def sample(self, k: int, *, key: Any = None) -> np.ndarray:
        rng = np.random.default_rng(key)
        return rng.multivariate_normal(np.zeros(len(self.cov)), self.cov, size=k)

    def log_prob(self, theta: Any) -> Any:
        raise NotImplementedError


class IdentityForward:  # user: a static problem, θ is the state
    dt = 1.0
    state_signature = None

    def step(self, state: Any, dt: float) -> Any:
        return state


@dataclass
class LinearObs:  # user
    H: np.ndarray

    def __call__(self, state: Any) -> Any:
        return self.H @ state

    def linearize(self, state: Any) -> Any:
        return self.H


@dataclass
class GaussEstimate:  # user
    mean: np.ndarray
    cov: np.ndarray | None = None

    def point(self) -> np.ndarray:
        return self.mean

    def sample(self, k: int, *, key: Any = None) -> np.ndarray | None:
        if self.cov is None:
            return None
        return np.random.default_rng(key).multivariate_normal(self.mean, self.cov, k)

    def log_prob(self, theta: Any) -> Any:
        raise NotImplementedError


@dataclass
class LGTask:  # user
    name: str = "lg40_every4"
    domain: str = "toy"

    def __post_init__(self) -> None:
        rng = np.random.default_rng(SEED)
        xt = rng.multivariate_normal(np.zeros(NS), Bc)
        y = H @ xt + SIG_OBS * rng.standard_normal(H.shape[0])
        self._data = {"theta": xt, "state": xt, "obs": y}

    def prior(self) -> Prior:
        return GaussPrior(Bc)

    def simulator(self) -> ForwardModel:
        return IdentityForward()

    def observation_operator(self) -> ObservationOperator:
        return LinearObs(H)

    def datasets(self) -> dict[str, Any]:
        return self._data


def kalman(Bm: np.ndarray, Rm: np.ndarray, y: np.ndarray) -> GaussEstimate:
    # K = B Hᵀ (H B Hᵀ + R)⁻¹;  x̂ = K y;  P = (I − K H) B
    K_ = Bm @ H.T @ np.linalg.inv(H @ Bm @ H.T + Rm)
    return GaussEstimate(K_ @ y, (np.eye(NS) - K_ @ H) @ Bm)


@dataclass
class ExactOI:  # rung 2, complexity 1: the oracle
    rung: int = 2
    complexity: int = 1

    def fit(self, task: Task) -> ExactOI:
        return self

    def __call__(self, obs: Any) -> Estimate:
        return kalman(Bc, R, obs)


@dataclass
class ZeroPrior:  # rung 2, complexity 0: the prior mean
    rung: int = 2
    complexity: int = 0

    def fit(self, task: Task) -> ZeroPrior:
        return self

    def __call__(self, obs: Any) -> Estimate:
        return GaussEstimate(np.zeros(NS), Bc)


@dataclass
class WrongRidge:  # rung 5: a linear gain with B = I and noise 10× too small
    rung: int = 5
    complexity: int = 0

    def fit(self, task: Task) -> WrongRidge:
        return self

    def __call__(self, obs: Any) -> Estimate:
        return GaussEstimate(kalman(np.eye(NS), 0.01 * R, obs).mean)


def _ref_mean(reference: Any) -> np.ndarray:
    return reference.point() if isinstance(reference, Estimate) else reference


@dataclass
class RMSEScorer:  # user
    name: str = "rmse"

    def __call__(self, prediction: Estimate, reference: Any) -> dict[str, float]:
        d = prediction.point() - _ref_mean(reference)
        return {"field": float(np.sqrt(np.mean(d**2)))}


@dataclass
class GaussCoverage:  # user: 90 % marginal coverage
    name: str = "cov90"

    def __call__(self, prediction: Estimate, reference: Any) -> dict[str, float]:
        cov = getattr(prediction, "cov", None)
        if cov is None:
            return {}
        z = np.abs(prediction.point() - _ref_mean(reference)) / np.sqrt(np.diag(cov))
        return {"frac": float(np.mean(z < 1.6449))}


task = LGTask()
ests = [ExactOI(), ZeroPrior(), WrongRidge()]
scorers = [RMSEScorer(), GaussCoverage()]
protocols = ([isinstance(e, Estimator) for e in ests], isinstance(task, Task),
             [isinstance(s, Scorer) for s in scorers])
reports = {axis: Benchmark(task=task, estimators=ests, scorers=scorers,
                           axis=axis).run() for axis in (Axis.STAGED, Axis.COMPLEXITY)}
staged = {r.cell.label: (r.cell.rung_name, r.reference.value, r.scores)
          for r in reports[Axis.STAGED].results}
complexity = {r.cell.label: (r.reference.value, r.scores, r.note)
              for r in reports[Axis.COMPLEXITY].results}
rule = ReferenceRule(staged={2: ReferenceSource.KNOWN_THETA,
                             5: ReferenceSource.KNOWN_THETA})
by_rule = Benchmark(task=task, estimators=ests, scorers=scorers,
                    reference_rule=rule).run().to_dict()
staged, complexity, by_rule

# %%
assert protocols == ([True] * 3, True, [True] * 2)
assert staged["L4/r2/c1"][:2] == ("model_based_inference", "known_theta")
assert staged["L4/r5/c0"][:2] == ("amortized_predictor", "oracle_posterior")
assert {k: {m: round(v, 3) for m, v in s.items()} for k, (_, _, s) in staged.items()} \
    == {"L4/r2/c1": {"rmse.field": 0.406, "cov90.frac": 0.825},
        "L4/r2/c0": {"rmse.field": 0.569, "cov90.frac": 0.975},
        "L4/r5/c0": {"rmse.field": 0.597}}
assert complexity["L4/r2/c1"][0] == "simpler_step"
assert round(complexity["L4/r2/c1"][1]["rmse.field"], 3) == 0.735
assert complexity["L4/r2/c0"][1] == {} and complexity["L4/r5/c0"][1] == {}
assert "generative rung: not scored against a reference" in complexity["L4/r2/c0"][2]
assert round(by_rule["L4/r5/c0"]["rmse.field"], 3) == 0.490


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
calls = rf.marked_calls(next((root / "docs/reports").glob("14_*.md")))
ready = Counter(c.status for c in calls)
ready

# %%
assert len(calls) == 27
assert ready == Counter({'here': 6, 'today': 13, 'user': 8})
