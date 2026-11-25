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
# # Background · processing levels: sea surface temperature from counts to heatwaves
#
# Regenerates every measured number on
# `docs/design/pages/processing-levels.html` (stub: `docs/design/processing_levels.md`).
# Simulates an ocean, an atmosphere and a two-channel infrared sensor, then carries the
# counts through calibration, retrieval, gridding and gap-filling, assimilation and
# heatwave detection, scoring every level against the truth. Also draws the six panels
# of day 32.
#
# Needs: NumPy, SciPy, einx, matplotlib.
# The third-party packages are in the `background` pixi environment. Not a stage of the DVC pipeline; run it by hand:
#
#     pixi run -e background python notebooks/background/processing_levels.py
#
# It writes `docs/images/background_processing_levels/results.json`, the file the page's
# numbers were taken from. Seeded, so the numbers repeat. About two minutes.

# %%
"""One variable through six levels: sea surface temperature from counts to heatwaves.

A synthetic truth is observed by a simulated two-channel infrared sensor. Five
operators then carry the data from L0 to L5, and each level is scored against
the truth. NumPy and SciPy only.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import einx
import numpy as np
import scipy.ndimage

# Shapes: D = 60 days; S × P = 70 × 50 swath pixels; C = 2 channels
#         Y × X = 64 × 64 grid cells of 25 km
DAYS, GRID, CELL_KM = 60, 64, 25.0
LINES, PIXELS, PIXEL_KM, TILT = 70, 50, 20.0, np.deg2rad(12.0)
WAVELENGTH_UM = np.array([11.0, 12.0])
PLANCK_C1, PLANCK_C2 = 1.191042e8, 14387.77      # W m⁻² sr⁻¹ µm⁴; µm K
ABSORPTION = np.array([0.5, 0.8])                # K per g cm⁻² of water vapour
PERSISTENCE, ANOMALY_STD = 0.95, 0.8             # per day; K


# === physics ===
def planck(temperature, wavelength):
    """Spectral radiance of a black body: B_λ(T) = c₁ / (λ⁵ (exp(c₂ / λT) − 1))."""
    return PLANCK_C1 / (wavelength**5 * np.expm1(PLANCK_C2 / (wavelength * temperature)))


def inverse_planck(radiance, wavelength):
    """Brightness temperature: T_b = c₂ / (λ ln(1 + c₁ / (λ⁵ L)))."""
    return PLANCK_C2 / (wavelength * np.log1p(PLANCK_C1 / (wavelength**5 * radiance)))


# === truth ===
def smooth_noise(rng, shape, length_cells):
    field = scipy.ndimage.gaussian_filter(rng.normal(size=shape), length_cells, mode="wrap")
    return field / field.std()


def step_anomaly(anomaly, rng):
    """One day of the anomaly: advect one cell east, decay, add new variability."""
    advected = np.roll(anomaly, 1, axis=-1)
    forcing = smooth_noise(rng, anomaly.shape, 4.0)
    return PERSISTENCE * advected + np.sqrt(1 - PERSISTENCE**2) * ANOMALY_STD * forcing


def make_truth(rng):
    yy, xx = np.mgrid[:GRID, :GRID]
    climatology = 291.0 + 6.0 * yy / (GRID - 1)                    # (Y, X) K
    anomaly = ANOMALY_STD * smooth_noise(rng, (GRID, GRID), 4.0)
    anomalies, blobs = [], []
    for day in range(DAYS):
        anomaly = step_anomaly(anomaly, rng)
        # a warm blob that drifts with the flow: the heatwave to be found at L5
        ramp = np.clip(min(day - 20, 46 - day) / 5.0, 0.0, 1.0)
        centre_x = (18 + day) % GRID
        distance2 = (yy - 36) ** 2 + (np.minimum(abs(xx - centre_x), GRID - abs(xx - centre_x))) ** 2
        blobs.append(3.0 * ramp * np.exp(-distance2 / (2 * 7.0**2)))
        anomalies.append(anomaly)
    truth = climatology + np.stack(anomalies) + np.stack(blobs)    # (D, Y, X) K
    return climatology, truth


def heatwave_threshold(rng, climatology, days=2000):
    """90th percentile of the undisturbed anomaly, per cell: the 'climatology run'."""
    anomaly = ANOMALY_STD * smooth_noise(rng, (GRID, GRID), 4.0)
    samples = []
    for _ in range(days):
        anomaly = step_anomaly(anomaly, rng)
        samples.append(anomaly)
    return climatology + np.percentile(np.stack(samples), 90, axis=0)   # (Y, X) K


# === sensor ===
@dataclasses.dataclass(frozen=True)
class Calibration:
    """Detector calibration, per channel: L = gain · DN + offset."""

    gain: np.ndarray            # (C,) W m⁻² sr⁻¹ µm⁻¹ count⁻¹
    offset: np.ndarray          # (C,) W m⁻² sr⁻¹ µm⁻¹


def make_calibration():
    low, high = planck(180.0, WAVELENGTH_UM), planck(320.0, WAVELENGTH_UM)
    return Calibration(gain=(high - low) / 4095.0, offset=low)


def swath_positions(day):
    """Where each swath pixel looks, in km: a tilted, shifting raster."""
    line, pixel = np.mgrid[:LINES, :PIXELS]
    origin_x = 150.0 + (day * 290.0) % 900.0
    x = origin_x + PIXEL_KM * (pixel * np.cos(TILT) - line * np.sin(TILT))
    y = 80.0 + PIXEL_KM * (pixel * np.sin(TILT) + line * np.cos(TILT))
    return x, y                                                   # (S, P), (S, P) km


def sample_grid(field, x, y):
    """Bilinear sample of a (Y, X) field at km positions."""
    coordinates = np.stack([y / CELL_KM - 0.5, x / CELL_KM - 0.5])
    return scipy.ndimage.map_coordinates(field, coordinates, order=1, mode="nearest")


def observe(truth, rng, calibration):
    """Forward-simulate the sensor: truth → L0 counts, day by day."""
    counts, cloud_truth = [], []
    for day in range(DAYS):
        x, y = swath_positions(day)
        sst = sample_grid(truth[day], x, y)                               # (S, P) K
        vapour = 2.5 + 1.2 * smooth_noise(rng, (LINES, PIXELS), 8.0)      # g cm⁻²
        cloudy = smooth_noise(rng, (LINES, PIXELS), 3.0) > 0.25
        top = 250.0 + 5.0 * rng.normal(size=(LINES, PIXELS))
        channels = []
        for c in range(2):
            brightness = np.where(cloudy, top, sst - ABSORPTION[c] * vapour)
            brightness = brightness + 0.05 * rng.normal(size=brightness.shape)
            radiance = planck(brightness, WAVELENGTH_UM[c])
            channels.append(np.round((radiance - calibration.offset[c]) / calibration.gain[c]))
        counts.append(np.stack(channels))
        cloud_truth.append(cloudy)
    return np.stack(counts).astype(np.uint16), np.stack(cloud_truth)      # (D, C, S, P)


# === l0_to_l1 ===
def calibrate(counts, calibration):
    """L0 → L1: counts to brightness temperature, per channel."""
    # L = g · DN + o
    radiance = einx.add(                                        # (D, C, S, P) → (D, C, S, P)
        "d c s p, c -> d c s p",
        einx.multiply("d c s p, c -> d c s p", counts.astype(np.float64), calibration.gain),
        calibration.offset,
    )
    # T_b = c₂ / (λ ln(1 + c₁ / (λ⁵ L)))
    wavelength = einx.rearrange("c -> 1 c 1 1", WAVELENGTH_UM)
    brightness = inverse_planck(radiance, wavelength)           # (D, C, S, P) K
    return brightness


def geolocate():
    """L0 → L1, the other half: where every pixel of every day looks."""
    positions = [swath_positions(day) for day in range(DAYS)]
    x = np.stack([p[0] for p in positions])                     # (D, S, P) km
    y = np.stack([p[1] for p in positions])                     # (D, S, P) km
    return x, y


# === l1_to_l2 ===
def fit_split_window(brightness, reference):
    """Fit θ on matchups: SST ≈ θ₀ + θ₁ T₁₁ + θ₂ (T₁₁ − T₁₂)."""
    t11, t12 = brightness
    design = np.column_stack([np.ones_like(t11), t11, t11 - t12])         # (M, 3)
    # θ = argmin ‖Aθ − y‖²
    theta, *_ = np.linalg.lstsq(design, reference, rcond=None)         # (3,)
    residual = reference - design @ theta                              # (M,) K
    return theta, float(residual.std())


def retrieve(brightness, theta, cloud_limit=270.0):
    """L1 → L2: brightness temperatures to SST, with a cloud flag."""
    t11 = brightness[:, 0]                                      # (D, S, P) K
    t12 = brightness[:, 1]
    # cloud test: a sea surface is never this cold
    clear = t11 > cloud_limit                                   # (D, S, P) bool
    # SST = θ₀ + θ₁ T₁₁ + θ₂ (T₁₁ − T₁₂)
    sst = theta[0] + theta[1] * t11 + theta[2] * (t11 - t12)    # (D, S, P) K
    return np.where(clear, sst, np.nan), clear


# === l2_to_l3 ===
def bin_to_grid(sst, x, y):
    """L2 → L3, step 1: average the clear pixels that fall in each grid cell."""
    row = np.floor(y / CELL_KM).astype(int)
    col = np.floor(x / CELL_KM).astype(int)
    inside = np.isfinite(sst) & (row >= 0) & (row < GRID) & (col >= 0) & (col < GRID)
    index = row[inside] * GRID + col[inside]
    total = np.bincount(index, weights=sst[inside], minlength=GRID * GRID)
    count = np.bincount(index, minlength=GRID * GRID)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return einx.rearrange("(y x) -> y x", mean, y=GRID), einx.rearrange("(y x) -> y x", count, y=GRID)


def grid_covariance(length_km=150.0, std=ANOMALY_STD):
    """Background covariance between all grid cells: B_ij = σ² exp(−d²_ij / 2ℓ²)."""
    yy, xx = np.mgrid[:GRID, :GRID]
    points = np.stack([yy.ravel(), xx.ravel()], axis=-1) * CELL_KM
    dy = points[:, None, 0] - points[None, :, 0]
    dx = abs(points[:, None, 1] - points[None, :, 1])
    dx = np.minimum(dx, GRID * CELL_KM - dx)
    return std**2 * np.exp(-(dy**2 + dx**2) / (2 * length_km**2))        # (N, N) K²


def gap_fill(binned, count, climatology, covariance, obs_std):
    """L2 → L3, step 2: optimal interpolation of the anomalies onto every cell."""
    observed = np.flatnonzero(count.ravel() > 0)                # (M,)
    innovation = (binned - climatology).ravel()[observed]       # y − H x_b      (M,)
    obs_variance = obs_std**2 / count.ravel()[observed]         # R diagonal     (M,)
    cross = covariance[:, observed]                             # B Hᵀ           (N, M)
    system = covariance[np.ix_(observed, observed)] + np.diag(obs_variance)   # H B Hᵀ + R
    # x̂ = x_b + B Hᵀ (H B Hᵀ + R)⁻¹ (y − H x_b)
    weights = np.linalg.solve(system, innovation)               # (M,)
    mean = climatology.ravel() + cross @ weights                # (N,) K
    # v̂ = diag(B − B Hᵀ (H B Hᵀ + R)⁻¹ H B)
    solved = np.linalg.solve(system, einx.rearrange("n m -> m n", cross))   # (M, N)
    reduction = einx.dot("n m, m n -> n", cross, solved)                    # (N,)
    variance = np.diag(covariance) - reduction                  # (N,) K²
    return einx.rearrange("(y x) -> y x", mean, y=GRID), einx.rearrange("(y x) -> y x", variance, y=GRID)


# === l3_to_l4 ===
def assimilate(analysis_l3, variance_l3, climatology):
    """L3 → L4: a forecast-analysis cycle with a per-cell Kalman filter."""
    state = np.zeros((GRID, GRID))                              # anomaly, K
    spread = np.full((GRID, GRID), ANOMALY_STD**2)              # its variance, K²
    states, spreads = [], []
    for day in range(DAYS):
        # forecast: x_f = ρ · shift(x_a);  P_f = ρ² · shift(P_a) + (1 − ρ²) σ²
        state = PERSISTENCE * np.roll(state, 1, axis=-1)
        spread = PERSISTENCE**2 * np.roll(spread, 1, axis=-1) + (1 - PERSISTENCE**2) * ANOMALY_STD**2
        # analysis: K = P_f / (P_f + R);  x_a = x_f + K (y − x_f);  P_a = (1 − K) P_f
        gain = spread / (spread + variance_l3[day])
        state = state + gain * (analysis_l3[day] - climatology - state)
        spread = (1 - gain) * spread
        states.append(climatology + state)
        spreads.append(spread)
    return np.stack(states), np.stack(spreads)                  # (D, Y, X) K, K²


def forecast(state, climatology, days):
    """L4, forward in time: the model alone, no observations."""
    anomaly = state - climatology
    out = []
    for _ in range(days):
        anomaly = PERSISTENCE * np.roll(anomaly, 1, axis=-1)
        out.append(climatology + anomaly)
    return np.stack(out)


# === l4_to_l5 ===
def heatwave_mask(sst, threshold, min_days=5):
    """L4 → L5, step 1: above the 90th percentile for at least five days running."""
    hot = sst > threshold                                       # (D, Y, X) bool
    run = np.zeros(hot.shape, dtype=int)
    for day in range(hot.shape[0]):
        previous = run[day - 1] if day else 0
        run[day] = np.where(hot[day], previous + 1, 0)
    # a day belongs to a heatwave if its run reaches min_days at some point
    longest = run.copy()
    for day in range(hot.shape[0] - 2, -1, -1):
        longest[day] = np.where(hot[day] & hot[day + 1], np.maximum(longest[day], longest[day + 1]), longest[day])
    return hot & (longest >= min_days)


def heatwave_events(mask, sst, climatology):
    """L4 → L5, step 2: connected regions in (day, y, x) become rows of a table."""
    labels, count = scipy.ndimage.label(mask)
    events = []
    for event in range(1, count + 1):
        days, rows, cols = np.nonzero(labels == event)
        intensity = (sst - climatology)[days, rows, cols]
        area = np.bincount(days).max() * CELL_KM**2
        events.append({
            "start_day": int(days.min()), "end_day": int(days.max()),
            "duration_days": int(days.max() - days.min() + 1),
            "peak_intensity_K": round(float(intensity.max()), 2),
            "max_area_km2": int(area),
        })
    return sorted(events, key=lambda e: -e["max_area_km2"])


# === run ===
def rmse(estimate, reference, where=None):
    difference = (estimate - reference)[np.isfinite(estimate) if where is None else where]
    return round(float(np.sqrt(np.mean(difference**2))), 3)


# %%
# Where the recorded output goes. Set BACKGROUND_OUT to write somewhere else.
import os as _os
import pathlib as _pathlib

_root = _pathlib.Path.cwd()
while not (_root / "myst.yml").exists() and _root != _root.parent:
    _root = _root.parent
OUT = _pathlib.Path(_os.environ.get("BACKGROUND_OUT", _root / "docs/images/background_processing_levels"))
OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    climatology, truth = make_truth(rng)
    threshold = heatwave_threshold(np.random.default_rng(1), climatology)
    calibration = make_calibration()

    counts, cloud_truth = observe(truth, rng, calibration)              # L0
    brightness = calibrate(counts, calibration)                         # L1
    x, y = geolocate()
    truth_swath = np.stack([sample_grid(truth[d], x[d], y[d]) for d in range(DAYS)])

    # matchups: 500 clear pixels from the first ten days against "buoys"
    clear_first = np.flatnonzero(~cloud_truth[:10].ravel())
    picks = rng.choice(clear_first, 500, replace=False)
    buoys = truth_swath[:10].ravel()[picks] + 0.2 * rng.normal(size=500)
    pairs = (brightness[:10, 0].ravel()[picks], brightness[:10, 1].ravel()[picks])
    theta, retrieval_std = fit_split_window(pairs, buoys)
    sst_l2, clear = retrieve(brightness, theta)                         # L2

    covariance = grid_covariance()
    binned, counts_grid, l3, l3_var = [], [], [], []
    for d in range(DAYS):
        mean, count = bin_to_grid(sst_l2[d], x[d], y[d])
        analysis, variance = gap_fill(mean, count, climatology, covariance, retrieval_std)
        binned.append(mean); counts_grid.append(count); l3.append(analysis); l3_var.append(variance)
    binned, counts_grid, l3, l3_var = map(np.stack, (binned, counts_grid, l3, l3_var))   # L3

    l4, l4_var = assimilate(l3, l3_var, climatology)                    # L4
    l4_forecast = forecast(l4[49], climatology, 5)

    mask_l5 = heatwave_mask(l4, threshold)                              # L5
    mask_true = heatwave_mask(truth, threshold)
    events = heatwave_events(mask_l5, l4, climatology)
    events_true = heatwave_events(mask_true, truth, climatology)
    mask_l3 = heatwave_mask(l3, threshold)

    observed = counts_grid > 0
    skip = slice(5, None)                                               # let the filter spin up
    R = {
        "theta": [round(float(v), 4) for v in theta],
        "theta_physics": [0.0, 1.0, round(ABSORPTION[0] / (ABSORPTION[1] - ABSORPTION[0]), 4)],
        "retrieval_std": round(retrieval_std, 3),
        "cloud_fraction": round(float(cloud_truth.mean()), 3),
        "cloud_flag_agreement": round(float((clear == ~cloud_truth).mean()), 4),
        "coverage_l2": round(float(np.isfinite(sst_l2).mean()), 3),
        "coverage_l3_binned": round(float(observed.mean()), 3),
        "rmse_l1_t11_vs_sst": rmse(np.where(clear, brightness[:, 0], np.nan), truth_swath),
        "rmse_l2": rmse(sst_l2, truth_swath),
        "rmse_l3_binned": rmse(binned[skip], truth[skip]),
        "rmse_l3": rmse(l3[skip], truth[skip]),
        "rmse_l3_observed": rmse(l3[skip], truth[skip], observed[skip]),
        "rmse_l3_unobserved": rmse(l3[skip], truth[skip], ~observed[skip]),
        "rmse_l4": rmse(l4[skip], truth[skip]),
        "rmse_l4_observed": rmse(l4[skip], truth[skip], observed[skip]),
        "rmse_l4_unobserved": rmse(l4[skip], truth[skip], ~observed[skip]),
        "rmse_climatology": rmse(np.broadcast_to(climatology, truth.shape)[skip], truth[skip]),
        "rmse_forecast_by_day": [rmse(l4_forecast[k], truth[50 + k]) for k in range(5)],
        "rmse_persist_clim_by_day": [rmse(climatology, truth[50 + k]) for k in range(5)],
        "l3_std_mean": round(float(np.sqrt(l3_var[skip]).mean()), 3),
        "l4_std_mean": round(float(np.sqrt(l4_var[skip]).mean()), 3),
        "mhw_cell_days_true": int(mask_true.sum()), "mhw_cell_days_l4": int(mask_l5.sum()),
        "mhw_cell_days_l3": int(mask_l3.sum()),
        "mhw_hits_l4": int((mask_l5 & mask_true).sum()), "mhw_hits_l3": int((mask_l3 & mask_true).sum()),
        "events_l4": events[:3], "events_true": events_true[:3], "n_events_l4": len(events), "n_events_true": len(events_true),
        "sizes": {
            "L0": [list(counts.shape), str(counts.dtype), counts.nbytes],
            "L1": [list(brightness.shape), "float32 + 2 × float32 geolocation", brightness.size * 4 + x.size * 4 * 2],
            "L2": [list(sst_l2.shape), "float32 + uint8 flag + geolocation", sst_l2.size * 5 + x.size * 4 * 2],
            "L3": [list(l3.shape), "2 × float32 (mean, variance)", l3.size * 8],
            "L4": [list(l4.shape), "2 × float32 (mean, variance)", l4.size * 8],
            "L5": [[len(events), 5], "table rows", len(json.dumps(events))],
        },
        "example": {
            "day": 30, "dn": [int(v) for v in counts[30, :, 20, 25]],
            "gain": [float(f"{v:.5g}") for v in calibration.gain], "offset": [float(f"{v:.5g}") for v in calibration.offset],
            "tb": [round(float(v), 2) for v in brightness[30, :, 20, 25]],
            "sst_l2": None if not clear[30, 20, 25] else round(float(sst_l2[30, 20, 25]), 2),
            "truth": round(float(truth_swath[30, 20, 25]), 2), "clear": bool(clear[30, 20, 25]),
        },
        "lines": sum(1 for _ in open(__file__)),
    }
    print(json.dumps(R, indent=1))
    (OUT / "results.json").write_text(json.dumps(R, indent=1) + "\n")

    # the six panels of day 32, as the difference from climatology
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def panel(name, draw):
        figure = plt.figure(figsize=(2.2, 2.2), dpi=110)
        axes = figure.add_axes([0, 0, 1, 1])
        axes.set_axis_off()
        axes.add_patch(plt.Rectangle((0, 0), 1600, 1600, color="#8a9aa3", alpha=0.25))
        draw(axes)
        axes.set_xlim(0, 1600)
        axes.set_ylim(0, 1600)
        figure.savefig(OUT / f"{name}.png", transparent=True)
        plt.close(figure)

    def grid_panel(field, mask=None):
        def draw(axes):
            axes.imshow(field, origin="lower", extent=[0, 1600, 0, 1600], cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
            if mask is not None:
                centres = np.linspace(12.5, 1587.5, GRID)
                axes.contour(centres, centres, mask.astype(float), levels=[0.5], colors="k", linewidths=1.6)
        return draw

    def swath_panel(axes):
        day = 32
        seen = np.isfinite(sst_l2[day])
        local = sample_grid(climatology, x[day], y[day])
        axes.scatter(x[day][~seen], y[day][~seen], s=2.2, c="#ffffff", marker="s", linewidths=0, alpha=0.9)
        axes.scatter(x[day][seen], y[day][seen], s=2.2, c=(sst_l2[day] - local)[seen], cmap="RdBu_r", vmin=-2.5, vmax=2.5, marker="s", linewidths=0)

    panel("truth", grid_panel(truth[32] - climatology, mask_true[32]))
    panel("l2", swath_panel)
    panel("binned", grid_panel(np.ma.masked_invalid(binned[32] - climatology)))
    panel("l3", grid_panel(l3[32] - climatology))
    panel("l4", grid_panel(l4[32] - climatology))
    panel("l5", grid_panel(l4[32] - climatology, mask_l5[32]))
