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
# # Background · the modeling cycle, example 2: plume quantification
#
# Regenerates every measured number and panel on
# `docs/design/pages/cycle-quantification.html` (stub:
# `docs/design/cycle_quantification.md`). It starts from the column map that
# `cycle_retrieval.py` wrote and asks for the emission rate, three ways: the integrated
# mass enhancement, a column plume along the reported wind, and a column plume whose
# direction and position are fitted. NUTS on the same posterior referees the fit.
#
# Needs plumax with its `geo`, `assim` and `inference` extras (the `geomodels` pixi
# environment). It was run against plumax at 290f31f, newer than the submodule pin.
# Every setting is read from `configs/background/cycle_quantification.yaml` and
# recorded in the output. Not a stage of the DVC pipeline; run it by hand, after
# `cycle_retrieval.py`:
#
#     pixi run -e geomodels python notebooks/background/cycle_quantification.py
#
# It writes `docs/images/background_cycle_quantification/`: `results.json` and five
# panels. Seeded; timings depend on the machine. About two minutes.

# %%
"""Example 2 of the modeling cycle: an emission rate from a column map.

The mask, the mass enhancement, the column plume and the coupled solver are
plumax's. This file composes them and measures.
"""

from __future__ import annotations

import json
import math
import time

import einx
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import plumax.coupled as cpl
import plumax.gauss_plume as gp
import rasterio
from geoarray import GeoArray
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS, init_to_value
from plumax.geo import IMEEstimate, PlumeMask
from plumax.population.catalog import event_from_posterior

# %%
# Settings come from configs/background/cycle_quantification.yaml; the recorded output
# goes to docs/images/background_cycle_quantification (set BACKGROUND_OUT to write
# elsewhere).
import os as _os
import pathlib as _pathlib

from omegaconf import OmegaConf

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(
    _os.environ.get(
        "BACKGROUND_OUT", _root / "docs/images/background_cycle_quantification"
    )
)
OUT.mkdir(parents=True, exist_ok=True)
CFG = OmegaConf.load(_root / "configs/background/cycle_quantification.yaml")

# %%
# Shapes: H, W rows and columns; N = H W pixels; D_θ controls; S posterior draws
# Units: [ΔΩ] = kg m⁻²; [q] = kg m⁻¹; [Q] = kg s⁻¹; [U] = m s⁻¹

METHANE_KG_PER_MOL = 0.01604


# === carrier ===
def pixel_receptors(n_pixels: int) -> Float[Array, "N 2"]:
    """Pixel-centre coordinates [m], south-west pixel first."""
    axis = (np.arange(n_pixels) + 0.5) * CFG.scene.pixel_size_m
    east, north = np.meshgrid(axis, axis, indexing="xy")
    receptors = einx.rearrange("h w, h w -> (h w) (1 + 1)", east, north)
    return jnp.asarray(receptors)


def as_geoarray(field: np.ndarray, *, fill_value: float | bool) -> GeoArray:
    """The map on a north-up UTM grid, the carrier plumax.geo's operators take."""
    size = CFG.scene.pixel_size_m
    west, south = CFG.scene.origin_utm_m
    north = south + field.shape[0] * size
    transform = rasterio.Affine(size, 0.0, west, 0.0, -size, north)
    # row 0 of a north-up raster is the northern edge
    north_up = field[::-1]
    return GeoArray(
        north_up,
        transform=transform,
        crs=CFG.scene.crs,
        fill_value=fill_value,
        attrs={},
        dims=("y", "x"),
        coords=(),
    )


# === mass ===
def mass_enhancement(
    column_mass: Float[np.ndarray, "H W"], noise: float, mask_sigma: float
) -> tuple[dict[str, float], np.ndarray]:
    """Model 0: Q = U · IME / L, from a thresholded mask."""
    carrier = as_geoarray(column_mass, fill_value=-9999.0)
    # today (plumax)
    # mask_n = [ΔΩ_n > k σ], small components dropped
    mask = PlumeMask(
        threshold=mask_sigma * noise, min_area_px=CFG.mass_enhancement.min_area_px
    )(carrier)
    # today (plumax)
    # IME = ∑_{n ∈ mask} ΔΩ_n a,  Q = U · IME / L
    estimate = IMEEstimate(
        wind_speed=CFG.met.wind_speed_m_s,
        length_method=CFG.mass_enhancement.length_method,
        return_uncertainty=True,
    )(carrier, mask)
    summary = {
        "mask_sigma": mask_sigma,
        "mask_pixels": int(np.asarray(mask).sum()),
        "ime_kg": round(float(estimate["ime_kg"]), 1),
        "length_m": round(float(estimate["length_m"]), 0),
        "emission_rate_kg_s": round(float(estimate["emission_rate_kg_s"]), 3),
        "emission_std_kg_s": round(float(estimate["emission_rate_uncertainty_kg_s"]), 3),
    }
    # back to south-up rows, for the panel
    return summary, np.asarray(mask)[::-1]


# === plume ===
def plume_problem(
    column_mass: Float[np.ndarray, "H W"],
    column_std: Float[np.ndarray, "H W"],
    *,
    fit_geometry: bool,
) -> tuple[cpl.CoupledProblem, dict]:
    """Models 1 and 2: a column plume, along the reported wind or fitted."""
    initial_width = CFG.scene.pixel_size_m / 12.0**0.5
    # today (plumax)
    # r(s; θ) = exp(−c² / 2w²) / (√(2π) w),  w² = w₀² + w_ref² (d / d_ref)^(2b)
    plume = gp.ColumnPlume(
        reference_distance=CFG.plume.reference_distance_m,
        initial_width=initial_width,
        upwind_scale=initial_width,
        pixel_size=CFG.scene.pixel_size_m,
    )
    source_position = jnp.asarray(list(CFG.catalogue.source_m))
    met_direction = math.radians(CFG.met.direction_deg)
    geometry = gp.PlumeGeometry(
        location=source_position,
        direction=met_direction,
        reference_width=math.exp(CFG.prior.mean.log_reference_width),
        width_exponent=CFG.prior.mean.width_exponent,
    )
    # today (plumax)
    source = gp.ColumnPlumeSource(plume=plume, geometry=geometry)
    # today (plumax)
    # Σ_ε = diag(σ_n²)
    instrument = cpl.Instrument(
        "EMIT",
        pixel_receptors(column_mass.shape[0]),
        retrieval_variance=jnp.asarray(column_std.ravel() ** 2),
    )
    prior_mean = dict(OmegaConf.to_container(CFG.prior.mean))
    prior_std = {name: CFG.prior.std[name] for name in prior_mean}
    if fit_geometry:
        # the structural change: φ and s₀ become controls
        prior_mean |= {"direction": met_direction, "location": source_position}
        prior_std |= {
            "direction": math.radians(CFG.prior.std.direction_deg),
            "location": CFG.prior.std.location_m,
        }
    # today (plumax)
    # θ ~ 𝒩(μ_θ, diag(σ_θ²))
    prior = cpl.CoupledPrior.from_std(prior_mean, prior_std)
    # today (plumax)
    # h(θ) = q r(s; θ) + Ω_bg
    forward = cpl.build_coupled_forward(source, [instrument])
    observations = {"EMIT": jnp.asarray(column_mass.ravel())}
    problem = cpl.CoupledProblem(forward=forward, prior=prior, observations=observations)
    return problem, prior_std


def summarise_fit(
    prior_std: dict,
    result: cpl.CoupledResult,
    column_std: Float[np.ndarray, "H W"],
) -> tuple[dict, Float[np.ndarray, "H W"]]:
    """The numbers the page quotes from one solve, and its residual map."""
    residual = np.asarray(result.per_instrument_residuals["EMIT"])
    residual = residual.reshape(column_std.shape)
    # χ²_red = mean_n (r_n / σ_n)²
    reduced_chi_squared = float(np.mean((residual / column_std) ** 2))
    blocks = {}
    for name, value in result.map_estimate.items():
        posterior_std = np.asarray(result.std[name])
        # data fraction = 1 − σ²_post / σ²_prior
        data_fraction = 1.0 - (posterior_std / np.asarray(prior_std[name])) ** 2
        blocks[name] = {
            "map": np.round(np.asarray(value), 5).tolist(),
            "std": np.round(posterior_std, 5).tolist(),
            "data_fraction": np.round(data_fraction, 4).tolist(),
        }
    summary = {
        "converged": bool(result.converged),
        "steps": int(result.n_steps),
        "reduced_chi_squared": round(reduced_chi_squared, 3),
        "line_density_kg_m": round(float(result.line_density), 4),
        "line_density_std_kg_m": round(float(result.line_density_std), 4),
        "blocks": blocks,
    }
    return summary, residual


def with_met_wind(result: cpl.CoupledResult) -> dict[str, float]:
    """Q = q U with the reported wind, and where its variance comes from."""
    speed, speed_std = CFG.met.wind_speed_m_s, CFG.met.wind_speed_std_m_s
    # today (plumax)
    # log Q ~ 𝒩(μ_q + μ_U, σ_q² + σ_U²),  σ_U² = log(1 + s_U² / U²)
    rate = result.with_wind(speed=speed, speed_std=speed_std)
    log_density_variance = float(np.asarray(result.std["log_line_density"]) ** 2)
    log_wind_variance = math.log(1.0 + speed_std**2 / speed**2)
    wind_share = log_wind_variance / (log_wind_variance + log_density_variance)
    return {
        "emission_rate_kg_s": round(float(rate.emission_rate), 3),
        "emission_std_kg_s": round(float(rate.emission_std), 3),
        "log_density_std": round(math.sqrt(log_density_variance), 4),
        "log_wind_std": round(math.sqrt(log_wind_variance), 4),
        "wind_share_of_variance": round(wind_share, 4),
    }


# === referee ===
def nuts_referee(
    problem: cpl.CoupledProblem, prior_std: dict, result: cpl.CoupledResult, key: Array
) -> dict[str, Float[np.ndarray, " S"]]:
    """The same posterior by NUTS, to check the Laplace approximation.

    The sampler moves in χ, with θ = θ* + L χ and L Lᵀ the Laplace covariance, so
    that the posterior it explores has unit scale in every direction. The density
    is the exact one: only the coordinates come from the approximation.
    """
    stacked = problem.stacked_observations()
    prior_mean = problem.prior.mean
    layout = problem.prior.layout
    map_flat = jnp.asarray(result.mean)
    # Σ_θ|y = L Lᵀ
    cholesky = jnp.linalg.cholesky(jnp.asarray(result.covariance))

    def unflatten(flat: Float[Array, " D"]) -> dict[str, Array]:
        control, start = {}, 0
        for name, size in layout:
            block = flat[start : start + size]
            control[name] = block.reshape(jnp.shape(prior_mean[name]))
            start += size
        return control

    def model() -> None:
        # χ: improper flat, the densities below carry the whole posterior
        flat_prior = dist.ImproperUniform(dist.constraints.real_vector, (), (map_flat.size,))
        whitened = numpyro.sample("whitened", flat_prior)
        # θ = θ* + L χ
        control = unflatten(map_flat + cholesky @ whitened)
        for name, block in control.items():
            # log p(θ_block) = log 𝒩(θ_block; μ, σ²)
            block_prior = dist.Normal(prior_mean[name], jnp.asarray(prior_std[name]))
            numpyro.factor(f"prior_{name}", jnp.sum(block_prior.log_prob(block)))
            numpyro.deterministic(name, block)
        # today (plumax)
        # h(θ) = q r(s; θ) + Ω_bg
        predicted = problem.predict_stacked(control)
        # y ~ 𝒩(h(θ), Σ_ε)
        likelihood = dist.Normal(predicted, jnp.sqrt(stacked.variance)).to_event(1)
        numpyro.sample("y", likelihood, obs=stacked.values)

    start_at = {"whitened": jnp.zeros(map_flat.size)}
    kernel = NUTS(model, init_strategy=init_to_value(values=start_at))
    sampler = MCMC(
        kernel,
        num_warmup=CFG.referee.warmup,
        num_samples=CFG.referee.samples,
        progress_bar=False,
    )
    sampler.run(key)
    return {name: np.asarray(draws) for name, draws in sampler.get_samples().items()}


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
    key_referee, key_event = jax.random.split(jax.random.key(CFG.seed))
    maps = np.load(_root / CFG.input)
    retrieval = json.loads((_root / CFG.input).with_name("results.json").read_text())
    truth_plume = retrieval["config"]["plume"]
    true_rate = truth_plume["emission_rate_kg_s"]
    true_wind = truth_plume["wind_speed_m_s"]
    results["truth"] = {
        "emission_rate_kg_s": true_rate,
        "wind_speed_m_s": true_wind,
        "line_density_kg_m": true_rate / true_wind,
        "direction_deg": truth_plume["direction_deg"],
    }

    # ΔΩ = M ΔX [kg m⁻²]
    column_mass = maps["with_saturation"].astype(float) * METHANE_KG_PER_MOL
    column_std = maps["with_saturation_std"].astype(float) * METHANE_KG_PER_MOL
    naive_mass = maps["matched_filter"].astype(float) * METHANE_KG_PER_MOL
    noise = float(np.median(column_std))
    results["noise_kg_m2"] = noise

    # --- model 0: the integrated mass enhancement, for three masks ---
    start = time.perf_counter()
    by_mask = [
        mass_enhancement(column_mass, noise, mask_sigma)
        for mask_sigma in CFG.mass_enhancement.mask_sigma
    ]
    results["mass_enhancement"] = [summary for summary, _ in by_mask]
    first_mask = by_mask[0][1]
    seconds_mass = (time.perf_counter() - start) / len(CFG.mass_enhancement.mask_sigma)
    results["mass_enhancement_naive_map"], _ = mass_enhancement(
        naive_mass, noise, CFG.mass_enhancement.mask_sigma[0]
    )

    # --- model 1: a column plume along the reported wind ---
    along_wind, along_wind_std = plume_problem(column_mass, column_std, fit_geometry=False)
    # today (plumax)
    # θ* = argmin 𝒥(θ),  Σ_θ|y = (Σ_θ⁻¹ + J_hᵀ Σ_ε⁻¹ J_h)⁻¹
    along_wind_result = cpl.solve_coupled(along_wind)
    results["along_wind"], along_wind_residual = summarise_fit(
        along_wind_std, along_wind_result, column_std
    )
    results["along_wind"] |= with_met_wind(along_wind_result)

    # --- model 2: direction and position fitted ---
    fitted, fitted_std = plume_problem(column_mass, column_std, fit_geometry=True)
    start = time.perf_counter()
    fitted_result = cpl.solve_coupled(fitted)
    seconds_fit = time.perf_counter() - start
    results["fitted"], fitted_residual = summarise_fit(fitted_std, fitted_result, column_std)
    results["fitted"] |= with_met_wind(fitted_result)
    direction = float(np.asarray(fitted_result.map_estimate["direction"]))
    results["fitted"]["direction_deg"] = round(math.degrees(direction), 2)
    direction_std = float(np.asarray(fitted_result.std["direction"]))
    results["fitted"]["direction_std_deg"] = round(math.degrees(direction_std), 2)
    # with the true wind speed, to separate the fit's error from the wind's
    perfect_wind = fitted_result.with_wind(speed=true_wind, speed_std=1e-6)
    results["fitted"]["emission_rate_with_true_wind_kg_s"] = round(
        float(perfect_wind.emission_rate), 3
    )

    # --- the same fit on example 1's first map: its bias carries through ---
    naive_std = np.full_like(naive_mass, naive_mass[maps["truth"] < 1e-3].std())
    naive_problem, naive_prior_std = plume_problem(naive_mass, naive_std, fit_geometry=True)
    naive_result = cpl.solve_coupled(naive_problem)
    results["fitted_naive_map"], _ = summarise_fit(naive_prior_std, naive_result, naive_std)

    # --- the referee: NUTS on model 2's posterior ---
    start = time.perf_counter()
    draws = nuts_referee(fitted, fitted_std, fitted_result, key_referee)
    seconds_referee = time.perf_counter() - start
    referee = {}
    for name in ("log_line_density", "direction", "log_reference_width", "width_exponent"):
        laplace_mean = float(np.asarray(fitted_result.map_estimate[name]))
        laplace_std = float(np.asarray(fitted_result.std[name]))
        referee[name] = {
            "nuts_mean": round(float(draws[name].mean()), 5),
            "nuts_std": round(float(draws[name].std()), 5),
            "laplace_mean": round(laplace_mean, 5),
            "laplace_std": round(laplace_std, 5),
            # shift in units of the NUTS standard deviation
            "shift_in_std": round(
                (laplace_mean - float(draws[name].mean())) / float(draws[name].std()), 3
            ),
            "std_ratio": round(laplace_std / float(draws[name].std()), 3),
        }
    results["referee"] = referee

    # --- what example 3 receives: one catalogue event ---
    rate = fitted_result.with_wind(
        speed=CFG.met.wind_speed_m_s, speed_std=CFG.met.wind_speed_std_m_s
    )
    # today (plumax)
    event = event_from_posterior(
        rate,
        x=CFG.catalogue.source_m[0],
        y=CFG.catalogue.source_m[1],
        time=0.0,
        tier="I",
        instrument="EMIT",
        key=key_event,
    )
    log_rate_draws = np.log(np.asarray(event.q_samples))
    results["event"] = {
        "emission_rate_kg_s": round(event.emission_rate, 3),
        "emission_std_kg_s": round(event.emission_std, 3),
        "log_rate_std": round(float(log_rate_draws.std()), 3),
        "draws": int(log_rate_draws.size),
    }
    results["seconds"] = {
        "mass_enhancement": round(seconds_mass, 4),
        "fit": round(seconds_fit, 2),
        "referee": round(seconds_referee, 1),
    }

    predicted = column_mass - fitted_residual
    scale = 0.15 * METHANE_KG_PER_MOL
    save_panels(
        {
            "observed": column_mass,
            "along_wind_residual": along_wind_residual,
            "fitted_prediction": predicted,
            "fitted_residual": fitted_residual,
            "mask": first_mask,
        },
        {
            "observed": (0.0, scale, "magma"),
            "along_wind_residual": (-0.3 * scale, 0.3 * scale, "RdBu_r"),
            "fitted_prediction": (0.0, scale, "magma"),
            "fitted_residual": (-0.3 * scale, 0.3 * scale, "RdBu_r"),
            "mask": (0, 1, "gray"),
        },
    )
    with open(__file__) as source_file:
        results["lines"] = len(source_file.read().splitlines())
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps({k: v for k, v in results.items() if k != "config"}, indent=1))
