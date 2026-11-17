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
# # Background · the role ladder: one methane job, rung by rung
#
# Regenerates every measured number on
# `docs/design/pages/role-ladder.html` (stub: `docs/design/role_ladder.md`).
# The page's code in one file: structures, primitives, strategies, a model, algorithms,
# steps and a recipe, run on a synthetic scene; then the estimator is swapped and the
# recipe is saved to JSON and loaded back.
#
# Needs: NumPy, SciPy, scikit-learn, einx, jaxtyping, pipekit-ops.
# It illustrates a docs page and is not a stage of the DVC pipeline; run it by hand:
#
#     python notebooks/background/role_ladder.py
#
# It writes `docs/images/background_role_ladder/results.json`, the file the page's
# numbers were taken from. Deterministic. A few seconds.

# %%
"""The methane running example, built up the role ladder, small enough to run.

NumPy, SciPy and scikit-learn for rungs 0 to 4; pipekit-ops for rungs 5 and 6.
"""

from __future__ import annotations

import dataclasses
import json

import einx
import numpy as np
import pipekit_ops
import scipy.ndimage
import sklearn.covariance
from jaxtyping import Bool, Float

# === structures ===
# Shapes: B = 40 bands; R × C = 96 × 96 pixels; P = R C pixels


# user
@dataclasses.dataclass(frozen=True)
class Scene:
    """A small carrier: the numbers, the pixel size and the attributes."""

    data: np.ndarray
    pixel_m: float
    attrs: dict


# === primitives ===
# user
def target_signature(
    mean: Float[np.ndarray, " band"],
    absorption: Float[np.ndarray, " band"],
    air_mass: float,
) -> Float[np.ndarray, " band"]:
    """Radiance change per ppm m of methane: t = −M (k ⊙ μ)."""
    # t = −M (k ⊙ μ)
    target = -air_mass * absorption * mean                    # (B,), (B,) → (B,)
    return target


# user
def pixel_area(pixel_m: float) -> float:
    """Area of one square pixel in m²: a = Δs²."""
    # a = Δs²
    area = pixel_m * pixel_m                                  # → ()
    return area


# user
def large_components(
    mask: Bool[np.ndarray, "y x"], min_pixels: int
) -> Bool[np.ndarray, "y x"]:
    """Keep connected pieces of at least N_min pixels: b_n = c_n 1[|K(n)| ≥ N_min]."""
    # K(n): the connected piece that holds pixel n
    labels, count = scipy.ndimage.label(mask)                 # (R, C) → (R, C)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)  # (R, C) → (K + 1,)
    keep = sizes >= min_pixels                                # (K + 1,) → (K + 1,)
    # label 0 is the background, never a plume
    keep[0] = False
    # b_n = c_n 1[|K(n)| ≥ N_min]
    plume = keep[labels]                                      # (R, C) → (R, C)
    return plume


# === strategies ===
# user
@dataclasses.dataclass(frozen=True)
class LedoitWolf:
    """How to estimate Σ: shrink the sample covariance towards a scaled identity."""

    def __call__(
        self, spectra: Float[np.ndarray, "pixel band"]
    ) -> Float[np.ndarray, "band band"]:
        estimator = sklearn.covariance.LedoitWolf()
        fitted = estimator.fit(spectra)
        return fitted.covariance_                             # (P, B) → (B, B)


# user
@dataclasses.dataclass(frozen=True)
class SampleCovariance:
    """How to estimate Σ: the plain sample covariance."""

    def __call__(
        self, spectra: Float[np.ndarray, "pixel band"]
    ) -> Float[np.ndarray, "band band"]:
        return np.cov(spectra, rowvar=False)                  # (P, B) → (B, B)


# user
@dataclasses.dataclass(frozen=True)
class SigmaThreshold:
    """How to pick candidate pixels: above k noise levels."""

    k: float = 3.0

    def __call__(
        self, values: Float[np.ndarray, "y x"], noise: float
    ) -> Bool[np.ndarray, "y x"]:
        # c_n = 1[y_n > k σ_y]
        return values > self.k * noise                        # (R, C) → (R, C)


# === model ===
# user
@dataclasses.dataclass(frozen=True)
class Background:
    """What was fitted: the scene's mean spectrum μ and covariance Σ."""

    mean: np.ndarray
    covariance: np.ndarray

    @classmethod
    def fit(
        cls,
        spectra: Float[np.ndarray, "pixel band"],
        estimator: LedoitWolf | SampleCovariance = LedoitWolf(),
    ) -> Background:
        # μ_b = (1 / P) ∑_n x_nb
        mean = einx.mean("[pixel] band", spectra)             # (P, B) → (B,)
        # Σ, by the chosen strategy
        covariance = estimator(spectra)                       # (P, B) → (B, B)
        return cls(mean, covariance)

    def solve(self, vector: Float[np.ndarray, " band"]) -> Float[np.ndarray, " band"]:
        # s = Σ⁻¹ t
        return np.linalg.solve(self.covariance, vector)       # (B,) → (B,)

    def noise(self, target: Float[np.ndarray, " band"]) -> float:
        # σ_y = (tᵀ Σ⁻¹ t)^(−1/2)
        direction = self.solve(target)                        # (B,) → (B,)
        precision = einx.dot("band, band ->", target, direction)  # → ()
        return float(precision) ** -0.5


# === algorithms ===
# user
def matched_filter(
    radiance: Float[np.ndarray, "band y x"],
    background: Background,
    target: Float[np.ndarray, " band"],
) -> Float[np.ndarray, "y x"]:
    """Methane enhancement per pixel: y_n = sᵀ (x_n − μ) / (tᵀ s), s = Σ⁻¹ t."""
    # s = Σ⁻¹ t
    direction = background.solve(target)                      # (B,) → (B,)
    # tᵀ s
    normalisation = einx.dot("band, band ->", target, direction)  # → ()
    # r_n = x_n − μ
    residual = einx.subtract(                                 # (B, R, C) → (B, R, C)
        "band y x, band -> band y x", radiance, background.mean
    )
    # sᵀ r_n
    projection = einx.dot("band, band y x -> y x", direction, residual)  # → (R, C)
    # y_n = sᵀ r_n / (tᵀ s)
    enhancement = projection / normalisation                  # (R, C) → (R, C)
    return enhancement


# user
def segment(
    values: Float[np.ndarray, "y x"],
    noise: float,
    rule: SigmaThreshold,
    min_pixels: int,
) -> Bool[np.ndarray, "y x"]:
    """Plume mask: candidates by the rule, then only the large pieces."""
    # c_n = 1[y_n > k σ_y]
    candidates = rule(values, noise)                          # (R, C) → (R, C)
    # b_n = c_n 1[|K(n)| ≥ N_min]
    plume = large_components(candidates, min_pixels)          # (R, C) → (R, C)
    return plume


# user
def emission_rate(
    values: Float[np.ndarray, "y x"],
    plume: Bool[np.ndarray, "y x"],
    area: float,
    wind: float,
    kappa: float,
) -> float:
    """Emission rate from the plume's mass and the wind: Q = U κ a ∑ b y / L."""
    # L = (a ∑_n b_n)^(1/2)
    length = (area * np.count_nonzero(plume)) ** 0.5          # → ()
    # ∑_n b_n y_n
    total = float(np.sum(values, where=plume))                # (R, C) → ()
    # Q = U κ a ∑_n b_n y_n / L
    rate = wind * kappa * area * total / length               # → ()
    return rate


# === steps ===
ESTIMATORS = {"ledoit_wolf": LedoitWolf, "sample": SampleCovariance}


# user
class MethaneRetrieval(pipekit_ops.Operator):
    """Step: radiance scene → enhancement scene, with its noise in the attrs."""

    def __init__(self, *, estimator: str = "ledoit_wolf") -> None:
        self.estimator = estimator

    def _apply(self, scene: Scene) -> Scene:
        # --- unwrap: arrays and numbers out of the carrier ---
        spectra = einx.id("band y x -> (y x) band", scene.data)   # (B, R, C) → (P, B)
        absorption = scene.attrs["absorption"]
        air_mass = scene.attrs["air_mass"]
        # --- compute: ② strategy, ③ model, ① primitive, ④ algorithm ---
        strategy = ESTIMATORS[self.estimator]()
        background = Background.fit(spectra, strategy)
        target = target_signature(background.mean, absorption, air_mass)
        values = matched_filter(scene.data, background, target)   # → (R, C)
        # --- rewrap: a carrier again, σ_y carried for the next step ---
        attrs = {"noise_ppm_m": background.noise(target)}
        return Scene(values, scene.pixel_m, attrs)


# user
class PlumeSegmentation(pipekit_ops.Operator):
    """Step: enhancement scene → plume mask."""

    def __init__(self, *, k: float = 3.0, min_pixels: int = 20) -> None:
        self.k = k
        self.min_pixels = min_pixels

    def _apply(self, enhancement: Scene) -> Scene:
        noise = enhancement.attrs["noise_ppm_m"]
        rule = SigmaThreshold(self.k)
        plume = segment(enhancement.data, noise, rule, self.min_pixels)
        return Scene(plume, enhancement.pixel_m, {})


# user
class EmissionRate(pipekit_ops.Operator):
    """Step: enhancement and mask → emission rate in kg s⁻¹."""

    def __init__(self, *, wind_m_s: float = 3.5, kappa: float = 6.56e-7) -> None:
        self.wind_m_s = wind_m_s
        self.kappa = kappa

    def _apply(self, enhancement: Scene, plume: Scene) -> float:
        area = pixel_area(enhancement.pixel_m)
        return emission_rate(
            enhancement.data, plume.data, area, self.wind_m_s, self.kappa
        )


# === recipe ===
# user
def plume_recipe(
    *, k: float = 3.0, estimator: str = "ledoit_wolf"
) -> pipekit_ops.Graph:
    """Recipe: retrieval, segmentation and quantification, wired into one job."""
    # today (pipekit)
    scene_in = pipekit_ops.Input("scene")
    enhancement = MethaneRetrieval(estimator=estimator)(scene_in)
    plume = PlumeSegmentation(k=k)(enhancement)
    emission = EmissionRate()(enhancement, plume)
    # today (pipekit)
    return pipekit_ops.Graph(
        inputs={"scene": scene_in}, outputs={"emission": emission}
    )


# === data ===
def make_scene(seed: int = 0) -> tuple[Scene, np.ndarray, np.ndarray]:
    """A synthetic scene: correlated background plus a 3 000 ppm m disk."""
    rng = np.random.default_rng(seed)
    bands, rows, cols = 40, 96, 96
    mean = 8.0 + 2.0 * np.sin(np.linspace(0, 3, bands))
    mix = rng.normal(size=(bands, bands)) * 0.05
    correlated = rng.normal(size=(rows * cols, bands)) @ mix.T
    white = rng.normal(size=(rows * cols, bands)) * 0.02
    absorption = 2e-5 * (1 + np.cos(np.linspace(0, 9, bands)) ** 2)
    air_mass = 2.2
    yy, xx = np.mgrid[:rows, :cols]
    disk = (yy - 48) ** 2 + (xx - 48) ** 2 <= 8**2
    truth = np.where(disk, 3000.0, 0.0)
    pixels = mean + correlated + white
    radiance = einx.id("(y x) band -> band y x", pixels, y=rows, x=cols)
    target = target_signature(mean, absorption, air_mass)
    radiance = radiance + einx.multiply("band, y x -> band y x", target, truth)
    attrs = {"absorption": absorption, "air_mass": air_mass}
    return Scene(radiance, 60.0, attrs), truth, disk


# %%
# Where the recorded output goes. Set BACKGROUND_OUT to write somewhere else.
import os as _os
import pathlib as _pathlib

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_role_ladder"))
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    scene, truth, disk = make_scene()
    results: dict = {}
    results["q_true"] = emission_rate(truth, disk, pixel_area(60.0), 3.5, 6.56e-7)
    # === run ===
    # user
    recipe = plume_recipe()
    result = recipe(scene=scene)                    # → {"emission": Q}
    # swap one strategy; nothing else changes
    swapped = plume_recipe(estimator="sample")(scene=scene)
    # today (pipekit)
    # the recipe as data: class names and settings, never arrays
    recipe_json = pipekit_ops.dumps(recipe)
    # today (pipekit)
    loaded = pipekit_ops.loads(recipe_json)
    again = loaded(scene=scene)                     # the same Q
    # === end ===
    enhancement = MethaneRetrieval()(scene)
    plume = PlumeSegmentation()(enhancement)
    results.update(
        q=result["emission"], q_sample=swapped["emission"], q_loaded=again["emission"],
        noise=enhancement.attrs["noise_ppm_m"], plume_px=int(plume.data.sum()),
        disk_px=int(disk.sum()), json=json.loads(recipe_json),
        lines=sum(1 for _ in open(__file__)),
    )
    print(json.dumps({k: v for k, v in results.items() if k != "json"}, indent=1))
    (OUT / "results.json").write_text(json.dumps(results, indent=1) + "\n")
