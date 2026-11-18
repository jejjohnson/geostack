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
# # Background · the modeling cycle, example 1: methane retrieval
#
# Regenerates every measured number and panel on
# `docs/design/pages/cycle-retrieval.html` (stub: `docs/design/cycle_retrieval.md`).
# One scene: a 2 kg s⁻¹ plume over a dark and a bright surface, seen by an imaging
# spectrometer. Three retrievals of the column enhancement (the matched filter, the
# filter with its sensitivity map, the filter with saturation) are each compared with
# a per-pixel nonlinear retrieval on radtrax's forward model.
#
# Needs radtrax and plumax (the `geomodels` pixi environment). It was run against
# radtrax at f1cdea1 and plumax at 290f31f, newer than the submodule pins. Every
# setting is read from `configs/background/cycle_retrieval.yaml` and recorded in the
# output. Not a stage of the DVC pipeline; run it by hand:
#
#     pixi run -e geomodels python notebooks/background/cycle_retrieval.py
#
# The spectroscopy is radtrax's synthetic table (150 random Voigt lines, no HITRAN),
# with its cross-sections multiplied by `spectroscopy.line_strength_scale` so that the
# band-mean unit absorption is of the order of methane's at 2.3 µm. It is not a real
# methane spectrum, and the noise level that follows is this table's, not a sensor's.
#
# It writes `docs/images/background_cycle_retrieval/`: `results.json`,
# `enhancement.npz` (the maps example 2 reads) and seven panels. Seeded; timings
# depend on the machine. About three minutes.

# %%
"""Example 1 of the modeling cycle: column enhancement from a radiance cube.

The forward model, the matched filter and its sensitivity are radtrax's; the truth
plume is plumax's column plume. This file composes them and measures.
"""

from __future__ import annotations

import json
import time

import einx
import equinox as eqx
import jax

jax.config.update("jax_enable_x64", True)  # float32 shifts line centres

import jax.numpy as jnp
import lineax as lx
import numpy as np
import optimistix as optx
import plumax.gauss_plume as gp
import radtrax.retrieval.matched_filter as mf
import scipy.ndimage
from jaxtyping import Array, Float
from radtrax import ForwardModel, unit_absorption_target
from radtrax.atmosphere import AtmosphericState, ViewingGeometry, afgl_us_standard
from radtrax.instrument import BandInstrument, SpectralResponseFunction
from radtrax.solvers import BeerLambert
from radtrax.spectroscopy import TableOpacity
from radtrax.spectroscopy.synthetic import synthetic_table
from radtrax.surface import Lambertian

# %%
# Settings come from configs/background/cycle_retrieval.yaml; the recorded output goes
# to docs/images/background_cycle_retrieval (set BACKGROUND_OUT to write elsewhere).
import os as _os
import pathlib as _pathlib

from omegaconf import OmegaConf

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(
    _os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_cycle_retrieval")
)
OUT.mkdir(parents=True, exist_ok=True)
CFG = OmegaConf.load(_root / "configs/background/cycle_retrieval.yaml")

# %%
# Shapes: H, W rows and columns; N = H W pixels; B bands; G wavenumbers; L layers
# Units: [ΔX] = mol m⁻²; [y] = radiance per unit solar irradiance; [k] = m² mol⁻¹

METHANE_KG_PER_MOL = 0.01604
# 1 ppm·m of methane at 101 325 Pa and 288 K, in mol m⁻²
MOL_M2_PER_PPM_M = 1e-6 * 101_325.0 / (8.314462618 * 288.0)


# === forward ===
def build_forward_model() -> tuple[ForwardModel, AtmosphericState, ViewingGeometry]:
    """y = I(S(O(x), R, g)): opacity, Beer–Lambert solver and band instrument."""
    # today (radtrax)
    # σ(ν; p, T): 150 synthetic Voigt lines on 4230–4370 cm⁻¹, strengths scaled
    table = synthetic_table(CFG.spectroscopy.table_seed)
    scale = CFG.spectroscopy.line_strength_scale
    table["absorption_cross_section"] = table["absorption_cross_section"] * scale
    # today (radtrax)
    # τ_l(ν) = σ(ν; ln p_l, T_l) · x_l · n_l Δz_l
    opacity = TableOpacity.from_datasets({"CH4": table})
    # λ = 10⁷ / ν [nm], reversed so that it ascends
    wavelength_nm = 1e7 / np.asarray(opacity.nu_cm)[::-1]
    centres_nm = np.arange(
        CFG.sensor.band_start_nm, CFG.sensor.band_stop_nm, CFG.sensor.band_step_nm
    )
    # today (radtrax)
    # f_bg: Gaussian band responses, rows summing to 1
    response = SpectralResponseFunction(
        wavelengths_hr_nm=wavelength_nm,
        band_centers_nm=centres_nm,
        band_widths_nm=np.full(centres_nm.size, CFG.sensor.band_fwhm_nm),
        band_names=tuple(f"{centre:.0f}" for centre in centres_nm),
    )
    instrument = BandInstrument.from_srf(response, opacity.nu_cm)
    # today (radtrax)
    # x = (p_l, T_l, x_CH4,l): AFGL US Standard, 1 km layers
    atmosphere = afgl_us_standard(z_top_km=CFG.atmosphere.top_km, gases=["CH4"])
    # today (radtrax)
    # m = 1/μ₀ + 1/μ
    geometry = ViewingGeometry.from_angles(
        sza_deg=CFG.sensor.solar_zenith_deg, vza_deg=CFG.sensor.view_zenith_deg
    )
    # today (radtrax)
    # L(ν) = (μ₀ / π) A exp(−m ∑_l τ_l(ν)),  y_b = ∑_g f_bg L(ν_g)
    model = ForwardModel(opacity, BeerLambert(), instrument)
    return model, atmosphere, geometry


def band_radiance(
    model: ForwardModel,
    atmosphere: AtmosphericState,
    geometry: ViewingGeometry,
    enhancement: Float[Array, ""],
    albedo: Float[Array, ""],
) -> Float[Array, " B"]:
    """Band radiances of one pixel with ΔX more methane at the surface."""
    # today (radtrax)
    # x' = x + ΔX in the bottom layer
    plume_atmosphere = atmosphere.add_column("CH4", enhancement)
    # today (radtrax)
    # y = F(x', A)
    return model(plume_atmosphere, Lambertian(albedo), geometry)


# === scene ===
def pixel_centres() -> tuple[Float[np.ndarray, "H W"], Float[np.ndarray, "H W"]]:
    """Pixel-centre coordinates of the scene [m]."""
    axis = (np.arange(CFG.scene.pixels) + 0.5) * CFG.scene.pixel_size_m
    return np.meshgrid(axis, axis, indexing="xy")


def true_enhancement() -> Float[np.ndarray, "H W"]:
    """The plume the scene contains: plumax's column plume, in mol m⁻²."""
    east, north = pixel_centres()
    receptors = einx.rearrange("h w, h w -> (h w) (1 + 1)", east, north)
    initial_width = CFG.scene.pixel_size_m / 12.0**0.5
    # today (plumax)
    # r(s) = exp(−c² / 2w²) / (√(2π) w),  w² = w₀² + w_ref² (d / d_ref)^(2b)
    plume = gp.ColumnPlume(
        reference_distance=CFG.plume.reference_distance_m,
        initial_width=initial_width,
        upwind_scale=initial_width,
        pixel_size=CFG.scene.pixel_size_m,
    )
    geometry = gp.PlumeGeometry(
        location=jnp.asarray(list(CFG.plume.source_m)),
        direction=np.deg2rad(CFG.plume.direction_deg),
        reference_width=CFG.plume.reference_width_m,
        width_exponent=CFG.plume.width_exponent,
    )
    # q = Q / U [kg m⁻¹]
    line_density = CFG.plume.emission_rate_kg_s / CFG.plume.wind_speed_m_s
    # ΔΩ = q r [kg m⁻²]
    column_mass = line_density * plume(jnp.asarray(receptors), geometry)
    # ΔX = ΔΩ / M [mol m⁻²]
    enhancement = np.asarray(column_mass) / METHANE_KG_PER_MOL
    return einx.rearrange("(h w) -> h w", enhancement, h=CFG.scene.pixels)


def surface_albedo(rng: np.random.Generator) -> Float[np.ndarray, "H W"]:
    """A dark surface around the facility and a bright one downwind, with texture."""
    east, _ = pixel_centres()
    white = rng.standard_normal(east.shape)
    texture = scipy.ndimage.gaussian_filter(white, CFG.surface.texture_scale_px)
    texture = texture / texture.std()
    is_dark = east < CFG.surface.boundary_x_m
    base = np.where(is_dark, CFG.surface.dark_albedo, CFG.surface.bright_albedo)
    # A = A₀ (1 + s η)
    return base * (1.0 + CFG.surface.texture * texture)


@eqx.filter_jit
def render(
    model: ForwardModel,
    atmosphere: AtmosphericState,
    geometry: ViewingGeometry,
    enhancement: Float[Array, " N"],
    albedo: Float[Array, " N"],
) -> Float[Array, "N B"]:
    """The noise-free cube, one forward-model call per pixel."""

    def one_pixel(pixel: tuple[Array, Array]) -> Float[Array, " B"]:
        return band_radiance(model, atmosphere, geometry, pixel[0], pixel[1])

    return jax.lax.map(one_pixel, (enhancement, albedo), batch_size=128)


def add_noise(clean: Float[Array, "N B"], key: Array) -> Float[Array, "N B"]:
    """y = F + ε, ε ~ 𝒩(0, diag(σ²)), σ_b² = (F_b / SNR)² + σ_floor²."""
    variance = (clean / CFG.sensor.snr) ** 2 + CFG.sensor.noise_floor**2
    return clean + jnp.sqrt(variance) * jax.random.normal(key, clean.shape)


# === filter ===
def background_statistics(
    cube: Float[np.ndarray, "H W B"], plume_mask: np.ndarray | None
) -> tuple[Float[np.ndarray, " B"], lx.AbstractLinearOperator]:
    """μ_y and Σ_y of the scene, leaving the masked pixels out."""
    keep = np.ones(cube.shape[:2], bool) if plume_mask is None else ~plume_mask
    background = cube[keep][:, None, :]
    # today (radtrax)
    # μ_y = mean_n y_n
    mean = mf.estimate_mean(background)
    # today (radtrax)
    # Σ_y: Ledoit–Wolf shrinkage of the empirical covariance
    covariance = mf.estimate_cov_shrunk(background, mean)
    return np.asarray(mean), covariance


def matched_filter(
    cube: Float[np.ndarray, "H W B"],
    mean: Float[np.ndarray, " B"],
    covariance: lx.AbstractLinearOperator,
    unit_absorption: Float[Array, " B"],
) -> Float[np.ndarray, "H W"]:
    """Model 1: every pixel is the scene mean plus α t plus clutter."""
    # t = μ_y ⊙ k
    target = mean * unit_absorption
    # today (radtrax)
    # α̂_n = tᵀ Σ_y⁻¹ (y_n − μ_y) / (tᵀ Σ_y⁻¹ t)
    return np.asarray(mf.apply_image(cube, mean, covariance, target))


def sensitivity(
    cube: Float[np.ndarray, "H W B"],
    mean: Float[np.ndarray, " B"],
    covariance: lx.AbstractLinearOperator,
    unit_absorption: Float[Array, " B"],
    enhancement: Float[np.ndarray, "H W"] | None = None,
) -> Float[np.ndarray, "H W"]:
    """Score recovered per unit of true enhancement, pixel by pixel."""
    target = mean * unit_absorption
    # today (radtrax)
    # A_n(α) = gᵀ (y_n ⊙ (exp(α k) − 1)) / (α gᵀ t),  g = Σ_y⁻¹ t
    return np.asarray(
        mf.score_sensitivity(
            cube, covariance, target, unit_absorption, enhancement=enhancement
        )
    )


def plume_free_background(
    cube: Float[np.ndarray, "H W B"], unit_absorption: Float[Array, " B"]
) -> tuple[Float[np.ndarray, " B"], lx.AbstractLinearOperator, np.ndarray]:
    """μ_y and Σ_y without the plume: mask, re-estimate, repeat."""
    # user
    mask = np.zeros(cube.shape[:2], bool)
    for _ in range(CFG.retrieval.mask_iterations):
        mean, covariance = background_statistics(cube, mask)
        score = matched_filter(cube, mean, covariance, unit_absorption)
        enhancement = score / sensitivity(cube, mean, covariance, unit_absorption)
        # smooth, so that the faint tail of the plume rises above the noise
        smooth = scipy.ndimage.gaussian_filter(enhancement, CFG.retrieval.mask_smoothing_px)
        centre = np.median(smooth)
        # robust σ = 1.4826 · median |x − median x|
        spread = 1.4826 * np.median(np.abs(smooth - centre))
        mask = smooth > centre + CFG.retrieval.mask_sigma * spread
        mask = scipy.ndimage.binary_dilation(mask, iterations=CFG.retrieval.mask_dilation_px)
    mean, covariance = background_statistics(cube, mask)
    return mean, covariance, mask


def saturated_enhancement(
    score: Float[np.ndarray, "H W"],
    cube: Float[np.ndarray, "H W B"],
    mean: Float[np.ndarray, " B"],
    covariance: lx.AbstractLinearOperator,
    unit_absorption: Float[Array, " B"],
) -> Float[np.ndarray, "H W"]:
    """Model 3: solve α̂_n = α_n A_n(α_n) for α_n by fixed-point iteration."""
    # user
    # α⁰ = α̂ / A(0)
    enhancement = score / sensitivity(cube, mean, covariance, unit_absorption)
    for _ in range(CFG.retrieval.saturation_iterations):
        # α^{i+1} = α̂ / A(α^i), with A evaluated on the positive part
        positive = np.clip(enhancement, 0.0, None)
        gain = sensitivity(cube, mean, covariance, unit_absorption, positive)
        enhancement = score / gain
    return enhancement


# === referee ===
def nonlinear_retrieval(
    model: ForwardModel,
    atmosphere: AtmosphericState,
    geometry: ViewingGeometry,
    spectra: Float[Array, "N B"],
) -> tuple[Float[Array, "N 2"], Float[Array, "N 2"], Float[Array, " N"]]:
    """The reference: (ΔX, A) of every pixel by Gauss–Newton on the full model.

    θ̂_n = argmin_θ ‖(y_n − F(θ)) / σ_n‖²,  Σ_θ|y = (J_Fᵀ Σ_ε⁻¹ J_F)⁻¹.

    The packaged optimal estimation with priors and quality flags is
    proposed (radtrax#110); this is the same solve without a prior.
    """

    def whitened_residual(
        theta: Float[Array, " 2"], spectrum: Float[Array, " B"]
    ) -> Float[Array, " B"]:
        predicted = band_radiance(model, atmosphere, geometry, theta[0], theta[1])
        variance = (spectrum / CFG.sensor.snr) ** 2 + CFG.sensor.noise_floor**2
        # r = (y − F(θ)) / σ
        return (spectrum - predicted) / jnp.sqrt(variance)

    # today (optimistix)
    solver = optx.GaussNewton(rtol=1e-8, atol=1e-10, linear_solver=lx.QR())
    first_guess = jnp.array([0.0, CFG.retrieval.first_guess_albedo])

    def one_pixel(spectrum: Float[Array, " B"]) -> tuple[Array, Array, Array]:
        # today (optimistix)
        # θ̂ = argmin ½ ‖r(θ)‖²
        solution = optx.least_squares(
            whitened_residual,
            solver,
            first_guess,
            args=spectrum,
            max_steps=CFG.retrieval.max_steps,
            throw=False,
        )
        # J_r = ∂r/∂θ at θ̂
        jacobian = jax.jacfwd(whitened_residual)(solution.value, spectrum)
        # Σ_θ|y = (J_rᵀ J_r)⁻¹
        precision = einx.dot("b i, b j -> i j", jacobian, jacobian)
        covariance = jnp.linalg.inv(precision)
        std = jnp.sqrt(jnp.diagonal(covariance))
        # χ² = ‖r(θ̂)‖²
        residual = whitened_residual(solution.value, spectrum)
        chi_squared = einx.dot("b, b ->", residual, residual)
        return solution.value, std, chi_squared

    return jax.lax.map(one_pixel, spectra, batch_size=CFG.retrieval.batch_size)


# === compare ===
def slope(estimate: np.ndarray, reference: np.ndarray) -> float:
    """Least-squares slope through the origin of estimate against reference."""
    return float(np.sum(estimate * reference) / np.sum(reference * reference))


def agreement(
    estimate: np.ndarray, reference: np.ndarray, masks: dict[str, np.ndarray]
) -> dict[str, float]:
    """Slope against the reference on each mask, and the scatter on all of them."""
    everywhere = np.any(list(masks.values()), axis=0)
    out = {name: round(slope(estimate[m], reference[m]), 3) for name, m in masks.items()}
    difference = (estimate - reference)[everywhere]
    out["rms_mol_m2"] = round(float(np.sqrt(np.mean(difference**2))), 4)
    return out


def timed(function, *args):
    """Run once to compile, then time the second call."""
    function(*args)
    start = time.perf_counter()
    result = function(*args)
    return result, time.perf_counter() - start


# === figures ===
def save_panels(maps: dict[str, np.ndarray], limits: dict[str, tuple]) -> None:
    """One small PNG per map, for the page."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, field in maps.items():
        low, high, colour_map = limits[name]
        figure = plt.figure(figsize=(2.4, 2.4), dpi=110)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.imshow(field, origin="lower", vmin=low, vmax=high, cmap=colour_map)
        axes.set_axis_off()
        figure.savefig(OUT / f"{name}.png")
        plt.close(figure)


# === run ===
if __name__ == "__main__":
    results: dict = {"config": OmegaConf.to_container(CFG)}
    rng = np.random.default_rng(CFG.seed)
    key_noise = jax.random.key(CFG.seed)
    shape = (CFG.scene.pixels, CFG.scene.pixels)

    model, atmosphere, geometry = build_forward_model()
    truth = true_enhancement()
    albedo = surface_albedo(rng)
    clean = render(
        model, atmosphere, geometry, jnp.asarray(truth.ravel()), jnp.asarray(albedo.ravel())
    )
    spectra = add_noise(clean, key_noise)
    cube = np.asarray(spectra).reshape(*shape, -1)

    # today (radtrax)
    # k = ∂ ln y / ∂ΔX at ΔX = 0
    unit_absorption = unit_absorption_target(
        model, atmosphere, Lambertian(CFG.retrieval.reference_albedo), geometry
    )
    results["bands_nm"] = [float(name) for name in model.instrument.band_names]
    results["unit_absorption_m2_mol"] = np.round(np.asarray(unit_absorption), 4).tolist()

    # --- step 3: how good is the linear emulator y ≈ y₀ (1 + ΔX k)? ---
    reference_albedo = jnp.asarray(CFG.retrieval.reference_albedo)
    clear = band_radiance(model, atmosphere, geometry, jnp.asarray(0.0), reference_albedo)
    emulator_error = []
    for amount in (0.02, 0.05, 0.1, 0.2, 0.4):
        exact = band_radiance(model, atmosphere, geometry, jnp.asarray(amount), reference_albedo)
        linear = clear * (1.0 + amount * unit_absorption)
        worst = float(jnp.max(jnp.abs(linear - exact) / (exact / CFG.sensor.snr)))
        emulator_error.append({"enhancement_mol_m2": amount, "error_over_noise": round(worst, 2)})
    results["emulator_error"] = emulator_error

    # --- model 1: the matched filter, background from every pixel ---
    scene_mean, scene_covariance = background_statistics(cube, None)
    naive, seconds_filter = timed(
        matched_filter, cube, scene_mean, scene_covariance, unit_absorption
    )

    # --- model 2: the plume masked out of its own background ---
    start = time.perf_counter()
    mean, covariance, plume_mask = plume_free_background(cube, unit_absorption)
    seconds_background = time.perf_counter() - start
    score = matched_filter(cube, mean, covariance, unit_absorption)

    # --- model 3: divide by the sensitivity map; model 4: with saturation ---
    gain = sensitivity(cube, mean, covariance, unit_absorption)
    with_gain = score / gain
    start = time.perf_counter()
    with_saturation = saturated_enhancement(score, cube, mean, covariance, unit_absorption)
    seconds_saturation = time.perf_counter() - start

    # --- the referee: nonlinear retrieval of every pixel ---
    retrieve = eqx.filter_jit(nonlinear_retrieval)
    (estimate, std, chi_squared), seconds_referee = timed(
        retrieve, model, atmosphere, geometry, spectra
    )
    reference = np.asarray(estimate[:, 0]).reshape(shape)
    reference_std = np.asarray(std[:, 0]).reshape(shape)
    reference_albedo_map = np.asarray(estimate[:, 1]).reshape(shape)
    n_bands = cube.shape[-1]
    results["referee"] = {
        "seconds": round(seconds_referee, 1),
        "reduced_chi_squared": round(float(np.mean(chi_squared)) / (n_bands - 2), 3),
        "std_mol_m2": round(float(np.median(reference_std)), 4),
        "std_ppm_m": round(float(np.median(reference_std)) / MOL_M2_PER_PPM_M),
        "truth_within_two_std": round(
            float(np.mean(np.abs(reference - truth) < 2 * reference_std)), 3
        ),
    }

    # --- compare each model with the referee, by surface and by strength ---
    east, _ = pixel_centres()
    # the twin knows where the plume is, so the masks carry no selection bias;
    # on real data they would come from an independent pass
    in_plume = truth > CFG.retrieval.compare_above_mol_m2
    is_dark = east < CFG.surface.boundary_x_m
    masks = {
        "dark": in_plume & is_dark,
        "bright": in_plume & ~is_dark,
        "weak": in_plume & (truth < CFG.retrieval.strong_above_mol_m2),
        "strong": in_plume & (truth >= CFG.retrieval.strong_above_mol_m2),
    }
    results["plume_pixels"] = {name: int(mask.sum()) for name, mask in masks.items()}
    models = {
        "matched_filter": naive,
        "masked_background": score,
        "with_sensitivity": with_gain,
        "with_saturation": with_saturation,
    }
    results["against_referee"] = {
        name: agreement(field, reference, masks) for name, field in models.items()
    }
    results["against_truth"] = {
        name: agreement(field, truth, masks)
        for name, field in {**models, "referee": reference}.items()
    }
    # plume mass on the referee's mask, relative to the truth's
    results["mass_ratio"] = {
        name: round(float(field[in_plume].sum() / truth[in_plume].sum()), 3)
        for name, field in {**models, "referee": reference}.items()
    }
    results["sensitivity"] = {
        "dark_median": round(float(np.median(gain[is_dark])), 3),
        "bright_median": round(float(np.median(gain[~is_dark])), 3),
        "strongest_pixel_saturation": round(
            float(
                sensitivity(cube, mean, covariance, unit_absorption, np.clip(with_saturation, 0, None))[
                    np.unravel_index(np.argmax(truth), shape)
                ]
                / gain[np.unravel_index(np.argmax(truth), shape)]
            ),
            3,
        ),
    }
    background = ~scipy.ndimage.binary_dilation(truth > 1e-3, iterations=3)
    results["noise_mol_m2"] = {
        "matched_filter": round(float(naive[background].std()), 4),
        "with_saturation_dark": round(float(with_saturation[background & is_dark].std()), 4),
        "with_saturation_bright": round(float(with_saturation[background & ~is_dark].std()), 4),
        "referee_dark": round(float(reference[background & is_dark].std()), 4),
        "referee_bright": round(float(reference[background & ~is_dark].std()), 4),
    }
    results["offset_mol_m2"] = {
        name: {
            "dark": round(float(field[background & is_dark].mean()), 4),
            "bright": round(float(field[background & ~is_dark].mean()), 4),
        }
        for name, field in {**models, "referee": reference}.items()
    }
    results["background_mask_pixels"] = int(plume_mask.sum())
    results["seconds"] = {
        "matched_filter": round(seconds_filter, 4),
        "masked_background": round(seconds_background, 3),
        "with_saturation": round(seconds_saturation, 3),
        "referee": round(seconds_referee, 1),
    }
    results["scene"] = {
        "truth_max_mol_m2": round(float(truth.max()), 3),
        "truth_max_ppm_m": round(float(truth.max()) / MOL_M2_PER_PPM_M),
        "plume_mass_kg": round(
            float(truth.sum()) * METHANE_KG_PER_MOL * CFG.scene.pixel_size_m**2
        ),
        "mol_m2_per_ppm_m": MOL_M2_PER_PPM_M,
    }

    # the noise is a fixed fraction of the radiance, so the corrected map's is flat
    with_saturation_std = np.full(shape, with_saturation[background].std())
    np.savez_compressed(
        OUT / "enhancement.npz",
        truth=truth.astype(np.float32),
        albedo=albedo.astype(np.float32),
        matched_filter=naive.astype(np.float32),
        with_saturation=with_saturation.astype(np.float32),
        with_saturation_std=with_saturation_std.astype(np.float32),
        referee=reference.astype(np.float32),
        referee_std=reference_std.astype(np.float32),
    )
    save_panels(
        {
            "truth": truth,
            "albedo": albedo,
            "matched_filter": naive,
            "with_sensitivity": with_gain,
            "with_saturation": with_saturation,
            "referee": reference,
            "matched_filter_error": naive - reference,
        },
        {
            "truth": (0.0, 0.15, "magma"),
            "albedo": (0.0, 0.5, "gray"),
            "matched_filter": (0.0, 0.15, "magma"),
            "with_sensitivity": (0.0, 0.15, "magma"),
            "with_saturation": (0.0, 0.15, "magma"),
            "referee": (0.0, 0.15, "magma"),
            "matched_filter_error": (-0.08, 0.08, "RdBu_r"),
        },
    )
    with open(__file__) as source:
        results["lines"] = len(source.read().splitlines())
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=1))
