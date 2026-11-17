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
# # Background · the modeling cycle and the six-step cycle on one methane plume
#
# Regenerates every measured number on
# `docs/design/pages/modeling-cycle.html` (stub: `docs/design/modeling_cycle.md`).
# A Gaussian plume simulator and a scene whose wind is 12° off the forecast: the inner
# loop with model 1, compare, the refined model 2, then an emulator, emulator-based
# inference and an amortized map, each scored against the step before.
#
# Needs: NumPy, SciPy, scikit-learn, matplotlib.
# It illustrates a docs page and is not a stage of the DVC pipeline; run it by hand:
#
#     python notebooks/background/modeling_cycle.py
#
# It writes `docs/images/background_modeling_cycle/results.json`, the file the page's
# numbers were taken from. Seeded; timings depend on the machine and change from run to run; counts, sizes and errors do not. About five minutes.

# %%
"""The modeling cycle and the six-step cycle on one methane plume.

A Gaussian plume simulator, an observed scene from a "real world" the first
model gets wrong, the inner loop (estimate), the compare step, the outer loop
(refine), then steps 1 to 5 of the six-step cycle on the refined model.
NumPy, SciPy and scikit-learn only.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np
import scipy.ndimage
import sklearn.decomposition
import sklearn.neural_network
import sklearn.preprocessing

# Shapes: H × W = 32 × 32 pixels of 50 m; K = 48 wind samples; D = 3 parameters
SIDE, PIXEL_M, WIND_M_S, NOISE = 32, 50.0, 4.0, 0.25          # NOISE in g m⁻²
SOURCE = (5.5, 16.0)                                          # pixel x, y
MEANDER = np.deg2rad(np.linspace(-6.0, 6.0, 48))              # wind wobble within a scene
SUB = (np.arange(3) + 0.5) / 3 - 0.5                          # 3 × 3 sub-pixel samples
PRIOR_LOW = np.array([0.2, 0.10, np.deg2rad(-30.0)])          # Q kg s⁻¹, a, φ rad
PRIOR_HIGH = np.array([6.0, 0.60, np.deg2rad(30.0)])


# === simulator ===
def plume(rate, spread, direction):
    """f(θ): column mass of a time-averaged Gaussian plume, in g m⁻².

    Ω(x, y) = Q / (U √(2π) σ_y(x)) · exp(−y² / 2σ_y(x)²),  σ_y(x) = a x^0.9
    with (x, y) the downwind and crosswind distance from the source.
    """
    col, row = np.meshgrid(np.arange(SIDE), np.arange(SIDE))
    east = (col[..., None, None] + SUB[:, None] - SOURCE[0]) * PIXEL_M      # (H, W, 3, 1) m
    north = (row[..., None, None] + SUB[None, :] - SOURCE[1]) * PIXEL_M     # (H, W, 1, 3) m
    angle = direction + MEANDER[:, None, None, None, None]                  # (K, 1, 1, 1, 1)
    downwind = east * np.cos(angle) + north * np.sin(angle)                 # (K, H, W, 3, 3) m
    crosswind = -east * np.sin(angle) + north * np.cos(angle)
    sigma = spread * np.maximum(downwind, 1.0) ** 0.9                       # σ_y, m
    column = rate / (WIND_M_S * np.sqrt(2 * np.pi) * sigma) * np.exp(-(crosswind**2) / (2 * sigma**2))
    column = np.where(downwind > 0, column, 0.0)
    return 1e3 * column.mean(axis=(0, 3, 4))                                # (H, W) g m⁻²


def sensor(field):
    """h: the instrument's blur, one pixel wide."""
    return scipy.ndimage.gaussian_filter(field, 1.0)


def simulate_v2(theta):
    """Model 2: y = h(f(Q, a, φ)); the wind direction is a parameter."""
    return sensor(plume(theta[0], theta[1], theta[2]))


def simulate_v1(theta):
    """Model 1: y = h(f(Q, a)); the wind direction is taken from the forecast, 0°."""
    return sensor(plume(theta[0], theta[1], 0.0))


# === inference ===
def log_posterior(theta, scene, forward, low, high):
    """log p(θ | y) up to a constant: flat prior in a box, Gaussian noise."""
    if np.any(theta < low) or np.any(theta > high):
        return -np.inf
    residual = scene - forward(theta)
    return -0.5 * np.sum(residual**2) / NOISE**2


def metropolis(scene, forward, start, low, high, steps, rng):
    """Random-walk Metropolis; the proposal adapts to the chain during burn-in."""
    dims = len(start)
    theta, value = np.array(start, float), log_posterior(start, scene, forward, low, high)
    step = np.diag(((high - low) * 0.02) ** 2)
    chain, calls = [], 1
    for index in range(steps):
        if index in (300, 700, 1200):
            step = 2.38**2 / dims * np.cov(np.array(chain[index // 2:]).T) + 1e-12 * np.eye(dims)
        proposal = rng.multivariate_normal(theta, step)
        proposed = log_posterior(proposal, scene, forward, low, high)
        calls += 1
        if np.log(rng.uniform()) < proposed - value:
            theta, value = proposal, proposed
        chain.append(theta.copy())
    return np.array(chain[1500:]), calls


# === compare ===
def compare(scene, prediction):
    """Is the misfit random (the error model explains it) or systematic?"""
    residual = (scene - prediction) / NOISE
    # χ²: mean squared residual in units of the noise; 1 if the model is right
    chi2 = float(np.mean(residual**2))
    # ρ₁: correlation of each residual with its right-hand neighbour; 0 if random
    neighbour = float(np.mean(residual[:, :-1] * residual[:, 1:]) / np.mean(residual**2))
    count = residual.size
    random = chi2 < 1 + 3 * np.sqrt(2 / count) and abs(neighbour) < 3 / np.sqrt(count)
    return {"chi2": round(chi2, 3), "neighbour": round(neighbour, 3), "verdict": "random" if random else "systematic",
            "chi2_limit": round(1 + 3 * np.sqrt(2 / count), 3), "neighbour_limit": round(3 / np.sqrt(count), 3)}


# === emulator ===
class Emulator:
    """f̂(θ; θ_emu): θ → principal components of the field → field."""

    def __init__(self, components=40, seed=0):
        self.scale_in = sklearn.preprocessing.StandardScaler()
        self.basis = sklearn.decomposition.PCA(components)
        self.scale_out = sklearn.preprocessing.StandardScaler()
        self.network = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=(128, 128), max_iter=3000, tol=1e-6, random_state=seed
        )

    def fit(self, thetas, fields):
        # the field is linear in Q, so emulate the field per unit rate
        unit = fields.reshape(len(fields), -1) / thetas[:, :1]
        codes = self.scale_out.fit_transform(self.basis.fit_transform(unit))
        self.network.fit(self.scale_in.fit_transform(thetas[:, 1:]), codes)
        return self

    def __call__(self, theta):
        code = self.network.predict(self.scale_in.transform(np.atleast_2d(theta)[:, 1:]))
        unit = self.basis.inverse_transform(self.scale_out.inverse_transform(code))
        return (theta[0] * unit).reshape(SIDE, SIDE)


# === amortized ===
class Amortized:
    """q(θ | y; ν): scene → a Gaussian over θ, in one forward pass."""

    def __init__(self, hidden=(128, 128), seed=0):
        self.basis = sklearn.decomposition.PCA(60)
        self.scale = sklearn.preprocessing.StandardScaler()
        self.scale_theta = sklearn.preprocessing.StandardScaler()
        options = dict(max_iter=600, tol=1e-6, early_stopping=True, random_state=seed)
        self.mean = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=hidden, **options)
        self.spread = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(64, 64), **options)

    def features(self, scenes):
        return self.basis.transform(scenes.reshape(len(scenes), -1))

    def fit(self, thetas, scenes):
        half = len(thetas) // 2
        self.basis.fit(scenes.reshape(len(scenes), -1))
        features = self.scale.fit_transform(self.features(scenes))
        targets = self.scale_theta.fit_transform(thetas)
        # ν for the mean, on the first half
        self.mean.fit(features[:half], targets[:half])
        # ν for the spread, on the second half's squared errors: log e² ≈ log σ² − 1.27
        error = targets[half:] - self.mean.predict(features[half:])
        self.spread.fit(features[half:], np.log(error**2 + 1e-9) + 1.27)
        return self

    def __call__(self, scene):
        features = self.scale.transform(self.features(scene[None]))
        mean = self.scale_theta.inverse_transform(self.mean.predict(features))[0]
        std = np.sqrt(np.exp(self.spread.predict(features)[0])) * self.scale_theta.scale_
        return mean, std


# === run ===
def summary(chain):
    return {"mean": [round(float(v), 4) for v in chain.mean(axis=0)], "std": [round(float(v), 4) for v in chain.std(axis=0)]}


def timed(function, *args, repeat=1):
    start = time.perf_counter()
    for _ in range(repeat):
        out = function(*args)
    return out, (time.perf_counter() - start) / repeat


# %%
# Where the recorded output goes. Set BACKGROUND_OUT to write somewhere else.
import os as _os
import pathlib as _pathlib

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_modeling_cycle"))
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    R: dict = {}
    truth = np.array([2.0, 0.30, np.deg2rad(12.0)])
    scene = simulate_v2(truth) + NOISE * rng.normal(size=(SIDE, SIDE))          # the real world
    _, R["simulate_ms"] = timed(simulate_v2, truth, repeat=20)
    R["simulate_ms"] = round(R["simulate_ms"] * 1e3, 2)

    # ---- the modeling cycle: inner loop with model 1, compare, refine, inner loop with model 2
    (chain1, calls1), seconds1 = timed(metropolis, scene, simulate_v1, [1.0, 0.3], PRIOR_LOW[:2], PRIOR_HIGH[:2], 4000, rng)
    fit1 = simulate_v1(chain1.mean(axis=0))
    R["v1"] = {**summary(chain1), "compare": compare(scene, fit1), "seconds": round(seconds1, 1), "calls": calls1}
    R["v1_start"] = compare(scene, simulate_v1(np.array([1.0, 0.3])))
    (chain2, calls2), seconds2 = timed(metropolis, scene, simulate_v2, [1.0, 0.3, 0.0], PRIOR_LOW, PRIOR_HIGH, 6000, rng)
    fit2 = simulate_v2(chain2.mean(axis=0))
    R["v2"] = {**summary(chain2), "compare": compare(scene, fit2), "seconds": round(seconds2, 1), "calls": calls2}
    R["truth"] = [float(v) for v in truth]

    # ---- step 1: simulate (θ, y) pairs
    def draw(count):
        thetas = rng.uniform(PRIOR_LOW, PRIOR_HIGH, size=(count, 3))
        return thetas, np.stack([simulate_v2(t) for t in thetas])
    (thetas, fields), seconds_pairs = timed(draw, 4000)
    test_thetas, test_fields = draw(300)
    R["pairs"] = {"count": 4000, "seconds": round(seconds_pairs, 1)}

    # ---- step 3: emulator, and its improve step: error against step 1 as training grows
    R["emulator_curve"] = []
    for count in (250, 1000, 4000):
        emulator, seconds_fit = timed(lambda: Emulator().fit(thetas[:count], fields[:count]))
        error = np.array([emulator(t) - f for t, f in zip(test_thetas, test_fields)])
        R["emulator_curve"].append({"pairs": count, "rmse_over_noise": round(float(np.sqrt(np.mean(error**2)) / NOISE), 3),
                                    "fit_seconds": round(seconds_fit, 1)})
    _, emulate = timed(emulator, truth, repeat=200)
    R["emulate_ms"] = round(emulate * 1e3, 3)

    # ---- step 4: the same inference, with the emulator in place of the simulator
    (chain4, calls4), seconds4 = timed(metropolis, scene, emulator, [1.0, 0.3, 0.0], PRIOR_LOW, PRIOR_HIGH, 6000, rng)
    R["step4"] = {**summary(chain4), "seconds": round(seconds4, 2), "calls": calls4}

    # ---- step 5: amortized inference, trained on noisy scenes from the same pairs,
    # ---- and its improve step: score it against step 4, then train a larger one
    references = []
    for theta in test_thetas[:20]:
        fresh = simulate_v2(theta) + NOISE * rng.normal(size=(SIDE, SIDE))
        chain, _ = metropolis(fresh, emulator, theta, PRIOR_LOW, PRIOR_HIGH, 4000, rng)
        references.append((fresh, chain.mean(axis=0), chain.std(axis=0)))
    R["step5_curve"] = []
    for label, copies, hidden in (("first", 4, (128, 128)), ("improved", 12, (256, 256))):
        order = rng.permutation(len(thetas) * copies)
        noisy = np.concatenate([fields + NOISE * rng.normal(size=fields.shape) for _ in range(copies)])[order]
        targets = np.tile(thetas, (copies, 1))[order]
        amortized, seconds_amortized = timed(lambda: Amortized(hidden).fit(targets, noisy))
        shifts = [np.abs(amortized(fresh)[0] - mean) / std for fresh, mean, std in references]
        ratios = [amortized(fresh)[1] / std for fresh, mean, std in references]
        inside = []
        for theta, field in zip(test_thetas, test_fields):
            mean, std = amortized(field + NOISE * rng.normal(size=(SIDE, SIDE)))
            inside.append(np.abs(theta - mean) < 1.645 * std)
        R["step5_curve"].append({
            "label": label, "scenes": len(noisy), "fit_seconds": round(seconds_amortized, 1),
            "mean_shift_in_std": [round(float(v), 2) for v in np.median(shifts, axis=0)],
            "std_ratio": [round(float(v), 2) for v in np.median(ratios, axis=0)],
            "coverage_90": [round(float(v), 3) for v in np.mean(inside, axis=0)],
        })
    (mean5, std5), query5 = timed(amortized, scene, repeat=200)
    R["step5"] = {"mean": [round(float(v), 4) for v in mean5], "std": [round(float(v), 4) for v in std5],
                  "query_ms": round(query5 * 1e3, 3)}
    R["step4_vs_step2"] = {"mean_shift_in_std": [round(float(abs(a - b) / s), 2) for a, b, s in zip(chain4.mean(0), chain2.mean(0), chain2.std(0))],
                           "std_ratio": [round(float(a / b), 2) for a, b in zip(chain4.std(0), chain2.std(0))]}
    R["step5_vs_step2"] = {"mean_shift_in_std": [round(float(abs(a - b) / s), 2) for a, b, s in zip(mean5, chain2.mean(0), chain2.std(0))],
                           "std_ratio": [round(float(a / b), 2) for a, b in zip(std5, chain2.std(0))]}
    R["lines"] = sum(1 for _ in open(__file__))
    print(json.dumps(R, indent=1))
    (OUT / "results.json").write_text(json.dumps(R, indent=1) + "\n")

    # the scene, the two fits and their misfits
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def picture(name, field, cmap, low, high):
        figure = plt.figure(figsize=(2.0, 2.0), dpi=110)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.set_axis_off()
        axes.imshow(field, origin="lower", cmap=cmap, vmin=low, vmax=high, interpolation="nearest")
        figure.savefig(OUT / f"{name}.png", transparent=True)
        plt.close(figure)

    top = float(scene.max())
    picture("scene", scene, "magma", 0, top)
    picture("fit1", fit1, "magma", 0, top)
    picture("fit2", fit2, "magma", 0, top)
    picture("res1", (scene - fit1) / NOISE, "RdBu_r", -4, 4)
    picture("res2", (scene - fit2) / NOISE, "RdBu_r", -4, 4)
