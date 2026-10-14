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
# # radtrax · 1 · The Beer–Lambert column retrieval, run
#
# Regenerates every number and figure in
# `docs/geomodels/radtrax/01_column_retrieval.md`: a methane column
# enhancement ΔΩ retrieved from top-of-atmosphere reflectance with the
# single-layer Beer–Lambert model, on a synthetic Sentinel-2 scene and a
# synthetic EMIT-like cube.
#
# 1. the forward model, with Sentinel-2's real spectral response functions
#    (stage 1);
# 2. the effective cross-section against exact band transmittance: the curve
#    of growth (stage 2);
# 3. the multispectral proxy baseline, two ways, inverted linearly (stage 3);
# 4. a look-up table of the band-averaged observable, inverted by
#    interpolation (stage 4);
# 5. hyperspectral: continuum fitting against geotoolz's matched filter
#    (stage 5).
#
# **The methane spectrum is synthetic.** HITRAN is not reachable from this
# environment, so the absorption lines are drawn at random in the two
# methane bands near 1.66 μm and 2.3 μm and scaled to a chosen band-averaged
# cross-section. Shapes, ratios and the direction of every bias carry over to
# real spectra; the magnitudes are those of this spectrum.
#
# Pinned revisions: georeader-spaceml 2.4.1 (Sentinel-2 SRFs, document
# COPE-GSEG-EOPG-TN-15-0007 v4.0, and the Gaussian SRF helper), geotoolz
# 0.10.2 at `088475e` (the matched filter), xrtoolz 0.0.5 at `308c7f2` (the
# mass-to-mixing-ratio conversion). It needs NumPy, SciPy, xarray,
# matplotlib, pandas and openpyxl; it runs in under a minute on a laptop CPU.

# %%
from __future__ import annotations

import importlib.util
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from georeader import reflectance  # noqa: E402
from georeader.readers.S2_SAFE_reader import read_srf  # noqa: E402
from geotoolz.matched_filter import MatchedFilter  # noqa: E402
from xrtoolz.atm.gas.ch4 import column_mass_to_delta_vmr  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)  # SRF normalisation
SEED = 0
root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
FIG = root / "docs/images/radtrax_01_column_retrieval"
FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Stage 1 · The forward model
#
# R_b = ρ_b · ⟨exp(−m σ(λ) (Ω_bg + ΔΩ))⟩_b, where ⟨·⟩_b is the average over
# band b's spectral response, m = 1/cos θ_s + 1/cos θ_v the two-way air-mass
# factor and Ω_bg the background methane column. The source note's model,
# R = ρ τ_bg exp(−m σ̄ ΔΩ), is its weak-absorption limit.

# %%
rng = np.random.default_rng(SEED)
WL = np.arange(1500.0, 2500.0, 0.01)  # nm, monochromatic grid       (G,)
HWHM = 0.1  # nm: a Lorentz half-width, broadened to keep the grid coarse


def lines(n: int, lo: float, hi: float, centre: float, width: float):
    # positions uniform over the band, log-normal strengths under a Gaussian
    # envelope exp(−(λ − centre)² / 2 width²)
    pos = rng.uniform(lo, hi, n)
    return pos, rng.lognormal(0.0, 0.8, n) * np.exp(
        -0.5 * ((pos - centre) / width) ** 2
    )


def lorentz_sum(pos, s):
    # σ_mono(λ) = Σ_i s_i (HWHM/π) / ((λ − λ_i)² + HWHM²)             (G,)
    out = np.zeros_like(WL)
    for p, si in zip(pos, s, strict=True):
        sl = slice(*np.searchsorted(WL, [p - 4.0, p + 4.0]))
        out[sl] += si * (HWHM / np.pi) / ((WL[sl] - p) ** 2 + HWHM**2)
    return out


band_23 = lorentz_sum(*lines(3000, 2165.0, 2420.0, 2310.0, 70.0))  # 2.3 μm band
band_166 = lorentz_sum(*lines(600, 1620.0, 1710.0, 1666.0, 15.0))  # 1.66 μm

# Sentinel-2A's spectral response functions, at 1 nm, from georeader
srf_s2 = read_srf("S2A")  # today
f11 = np.interp(WL, srf_s2.index.values, srf_s2["B11"].values, left=0, right=0)
f12 = np.interp(WL, srf_s2.index.values, srf_s2["B12"].values, left=0, right=0)


def band_mean(x, f):
    # ⟨x⟩_b = ∫ x f dλ / ∫ f dλ
    return (x * f).sum(-1) / f.sum()


# Scale the two bands so that the band-averaged cross-sections are
# σ̄₁₂ = 0.40 m²/kg and σ̄₁₁ = 0.04 m²/kg: a modelling choice (a ratio of
# one tenth), not a measured value
SIG12, RATIO11 = 0.40, 0.10
sigma = band_23 * SIG12 / band_mean(band_23, f12)
sigma = sigma + band_166 * (RATIO11 * SIG12 - band_mean(sigma, f11)) / band_mean(
    band_166, f11
)
s11, s12 = band_mean(sigma, f11), band_mean(sigma, f12)

# Background: 1.9 ppm over a 1013 hPa column, as a mass column
# Ω_bg = X · (M_CH4 / M_air) · p_s / g
OMEGA_BG = 1.9e-6 * (16.04 / 28.96) * 101325.0 / 9.80665  # kg/m²
SZA, VZA = 35.0, 5.0
M_AMF = 1 / np.cos(np.radians(SZA)) + 1 / np.cos(np.radians(VZA))


def transmittance(m, omega, f):
    # T_b(m, Ω) = ⟨exp(−m σ Ω)⟩_b                       (…,) → (…,)
    m, omega = np.broadcast_arrays(np.asarray(m, float), np.asarray(omega, float))
    out = np.empty(m.shape)
    sel = f > 0
    sig_b, f_b = sigma[sel], f[sel]
    for i in np.ndindex(m.shape):
        out[i] = (np.exp(-m[i] * sig_b * omega[i]) * f_b).sum() / f_b.sum()
    return out


stage1 = {
    "sigma11": float(s11),
    "sigma12": float(s12),
    "omega_bg": OMEGA_BG,
    "amf": float(M_AMF),
    "b12_lo_hi": (
        int(srf_s2.index[srf_s2["B12"] > 0.5].min()),
        int(srf_s2.index[srf_s2["B12"] > 0.5].max()),
    ),
    "T12_bg": float(transmittance(M_AMF, OMEGA_BG, f12)),
}
stage1

# %%
assert (round(stage1["sigma11"], 3), round(stage1["sigma12"], 3)) == (0.040, 0.400)
assert round(stage1["omega_bg"], 4) == 0.0109  # kg/m² of background methane
assert round(stage1["amf"], 3) == 2.225  # θ_s = 35°, θ_v = 5°
assert stage1["b12_lo_hi"] == (2113, 2286)  # S2A B12 above half its peak, nm
assert round(stage1["T12_bg"], 3) == 0.990  # the background removes 1 % in B12

# %% [markdown]
# ## Stage 2 · The effective cross-section against exact band transmittance
#
# The linear model says τ_b(ΔΩ) = m σ̄_b ΔΩ. The exact band optical depth of
# the plume, above the background, is
# τ_b(ΔΩ) = −ln(T_b(m, Ω_bg + ΔΩ) / T_b(m, Ω_bg)). Strong line cores
# saturate first, so the exact curve bends below the line.

# %%
omegas = np.linspace(0.0, 1.0, 41)
tau12 = -np.log(
    transmittance(M_AMF, OMEGA_BG + omegas, f12) / transmittance(M_AMF, OMEGA_BG, f12)
)
tau11 = -np.log(
    transmittance(M_AMF, OMEGA_BG + omegas, f11) / transmittance(M_AMF, OMEGA_BG, f11)
)
lin12 = M_AMF * s12 * omegas


def deficit(omega: float) -> float:
    # 1 − τ_exact / τ_linear at one ΔΩ
    t = -np.log(
        transmittance(M_AMF, OMEGA_BG + omega, f12)
        / transmittance(M_AMF, OMEGA_BG, f12)
    )
    return float(1 - t / (M_AMF * s12 * omega))


stage2 = {
    "deficit_005": deficit(0.05),
    "deficit_030": deficit(0.30),
    "deficit_050": deficit(0.50),
    "deficit_100": deficit(1.0),
    "depth_030": float(1 - np.exp(-M_AMF * s12 * 0.30)),
}

fig, ax = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
ax.plot(omegas, lin12, "--", color="#64748b", label="linear, m σ̄₁₂ ΔΩ")
ax.plot(omegas, tau12, color="#0f766e", lw=2, label="exact, band 12")
ax.plot(omegas, tau11, color="#1d4ed8", lw=2, label="exact, band 11")
ax.axvline(0.3, color="#b45309", lw=1, ls=":")
ax.text(0.31, 0.05, "plume peak\n(stage 3)", color="#b45309", fontsize=9)
ax.set_xlabel("column enhancement ΔΩ (kg/m²)")
ax.set_ylabel("plume optical depth τ_b")
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIG / "curve_of_growth.svg")
plt.close(fig)
{k: round(v, 3) for k, v in stage2.items()}

# %%
# the linear model overestimates the optical depth, more as ΔΩ grows
assert (round(stage2["deficit_005"], 3), round(stage2["deficit_030"], 3)) == (
    0.038,
    0.151,
)
assert (round(stage2["deficit_050"], 2), round(stage2["deficit_100"], 2)) == (
    0.22,
    0.36,
)
assert round(stage2["depth_030"], 3) == 0.234  # a 23 % dip in B12 at the peak

# %% [markdown]
# ## Stage 3 · The multispectral proxy baseline
#
# A 200 × 200 scene at 20 m with two surface types and a Gaussian plume
# (peak 0.3 kg/m²). B11 serves as the proxy for B12. Two baselines: (a) the
# source note's, R₁₂ ≈ k R₁₁ + b fitted on pixels outside the suspected plume
# mask; (b) Varon et al.'s MBSP, R₁₁ ≈ c R₁₂ through the origin, fitted on the
# whole scene. Both inverted linearly with ΔΩ = τ_obs / (m σ̄).

# %%
N, PIX = 200, 20.0  # pixels, m
yy, xx = np.mgrid[0:N, 0:N] * PIX
# surface: a smooth B11 field, two cover types with different B12/B11 ratios
smooth = np.cumsum(np.cumsum(rng.normal(size=(N, N)), 0), 1)
smooth = (smooth - smooth.min()) / np.ptp(smooth)
rho11 = 0.20 + 0.20 * smooth
veg = (np.sin(xx / 900.0) + np.cos(yy / 700.0)) > 0.9  # patches of a second cover
ratio = np.where(veg, 0.80, 0.84) * (1 + 0.01 * rng.normal(size=(N, N)))
rho12 = rho11 * ratio
# the plume: an elongated Gaussian, rotated 30°, peak 0.3 kg/m²
x0, y0 = 1400.0, 2000.0
u = (xx - x0) * np.cos(np.radians(30)) + (yy - y0) * np.sin(np.radians(30))
v = -(xx - x0) * np.sin(np.radians(30)) + (yy - y0) * np.cos(np.radians(30))
d_omega = 0.3 * np.exp(-0.5 * ((u / 420.0) ** 2 + (v / 140.0) ** 2))
plume = d_omega > 0.005
NOISE = 0.0015  # reflectance units, per band

# exact band transmittances on a ΔΩ grid, interpolated per pixel (fast and
# exact to the grid's interpolation error, checked below)
grid = np.linspace(0.0, 0.4, 401)
T11_grid = transmittance(M_AMF, OMEGA_BG + grid, f11)
T12_grid = transmittance(M_AMF, OMEGA_BG + grid, f12)
R11 = rho11 * np.interp(d_omega, grid, T11_grid) + NOISE * rng.normal(size=(N, N))
R12 = rho12 * np.interp(d_omega, grid, T12_grid) + NOISE * rng.normal(size=(N, N))
interp_err = float(
    abs(
        np.interp(0.1234, grid, T12_grid) - transmittance(M_AMF, OMEGA_BG + 0.1234, f12)
    )
)

# (a) the source note: R₁₂ ≈ k R₁₁ + b on clean pixels
suspect = d_omega > 0.001  # the suspected-plume mask, given by a detector
k, b = np.polyfit(R11[~suspect], R12[~suspect], 1)
tau_a = -np.log(R12 / (k * R11 + b))
omega_a = tau_a / (M_AMF * s12)
# (b) MBSP: R₁₁ ≈ c R₁₂ through the origin, whole scene
c = float((R11 * R12).sum() / (R12 * R12).sum())
tau_b = -np.log(c * R12 / R11)
omega_b = tau_b / (M_AMF * (s12 - s11))  # B11 absorbs too: subtract it


# the local background: a ring of clean pixels around the plume, whose median
# is subtracted before integrating, as IME practice does
ring = ~suspect & (np.hypot(u / 420.0, v / 140.0) < 6.0)


def score(omega_hat):
    # plume mass (kg) above the ring's median, its ratio to the truth, the
    # RMSE over the plume and the noise over clean pixels (kg/m²)
    off = np.median(omega_hat[ring])
    mass = float((omega_hat[plume] - off).sum() * PIX**2)
    true = float(d_omega[plume].sum() * PIX**2)
    return {
        "mass": mass,
        "ratio": mass / true,
        "offset": float(off),
        "rmse": float(np.sqrt(np.mean((omega_hat[plume] - off - d_omega[plume]) ** 2))),
        "noise": float(np.std(omega_hat[~suspect])),
    }


true_mass = float(d_omega[plume].sum() * PIX**2)
omega_a_nob11 = tau_a / (M_AMF * s12)  # as the source: B11 assumed blind
stage3 = {
    "true_mass": true_mass,
    "k": float(k),
    "b": float(b),
    "c": c,
    "n_plume": int(plume.sum()),
    # the second cover, away from the plume: a false enhancement
    "patch_a": float(omega_a[veg & ~suspect].mean()),
    "patch_b": float(omega_b[veg & ~suspect].mean()),
    "a": score(omega_a),
    "b_mbsp": score(omega_b),
    "interp_err": interp_err,
    # with B11 blind the bias would vanish; with σ̄₁₁ = σ̄₁₂ / 10 the
    # first-order loss is σ̄₁₁ / σ̄₁₂ of the signal
    "b11_first_order": float(s11 / s12),
}
{
    k_: (v if not isinstance(v, dict) else {a: round(z, 4) for a, z in v.items()})
    for k_, v in stage3.items()
}

# %%
assert stage3["interp_err"] < 1e-6
assert round(stage3["true_mass"]) == 108_979  # kg inside the plume mask
assert stage3["n_plume"] == 3_777  # pixels with ΔΩ > 0.005 kg/m²
assert (round(stage3["k"], 3), round(stage3["b"], 4), round(stage3["c"], 3)) == (
    0.816,
    0.0016,
    1.220,
)
assert round(stage3["b_mbsp"]["rmse"], 4) == 0.0206
# the second cover reads as +0.04 kg/m² of methane in both, and drags the
# ring's median below zero
assert (round(stage3["patch_a"], 2), round(stage3["patch_b"], 2)) == (0.04, 0.04)
assert stage3["a"]["offset"] < 0 and stage3["b_mbsp"]["offset"] < 0
# (a) the source note's baseline and a B11 assumed blind: low by about
# σ̄₁₁ / σ̄₁₂ (B11 absorbs under the plume) plus the saturation of stage 2
assert round(stage3["a"]["ratio"], 2) == 0.80
assert round(stage3["b11_first_order"], 2) == 0.10
# (b) MBSP with σ̄₁₂ − σ̄₁₁ removes the B11 term; saturation remains
assert round(stage3["b_mbsp"]["ratio"], 2) == 0.89
# the second surface cover, not the sensor, sets the noise floor
assert (
    round(stage3["a"]["noise"], 3) == 0.029
    and round(stage3["b_mbsp"]["noise"], 3) == 0.032
)

# %% [markdown]
# ## Stage 4 · A look-up table of the band-averaged observable
#
# Tabulate the MBSP observable exactly, g(m, ΔΩ) = −ln(T₁₂/T₁₂,bg) +
# ln(T₁₁/T₁₁,bg), on a grid of air-mass factors and enhancements, and invert
# it per pixel by interpolation. This is what the source note's LUT step does
# with MODTRAN, and what Varon et al. do with their radiative transfer model.

# %%
M_GRID = np.linspace(2.0, 3.5, 7)
D_GRID = np.linspace(0.0, 0.4, 81)
LUT = np.empty((M_GRID.size, D_GRID.size))
for i, mi in enumerate(M_GRID):
    LUT[i] = -np.log(
        transmittance(mi, OMEGA_BG + D_GRID, f12) / transmittance(mi, OMEGA_BG, f12)
    ) + np.log(
        transmittance(mi, OMEGA_BG + D_GRID, f11) / transmittance(mi, OMEGA_BG, f11)
    )


def lut_invert(g, m):
    # ΔΩ = g⁻¹(τ_obs; m): interpolate the table at m, then invert in ΔΩ
    j = np.searchsorted(M_GRID, m) - 1
    w = (m - M_GRID[j]) / (M_GRID[j + 1] - M_GRID[j])
    row = (1 - w) * LUT[j] + w * LUT[j + 1]
    # noise makes g negative off the plume: extend the table as an odd function
    return np.sign(g) * np.interp(np.abs(g), row, D_GRID)


omega_lut = lut_invert(tau_b, M_AMF)
monotone = bool(np.all(np.diff(LUT, axis=1) > 0))
stage4 = {"lut": score(omega_lut), "monotone": monotone, "shape": LUT.shape}
stage4

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3), constrained_layout=True)
for a, (img, title) in zip(
    axs,
    [
        (d_omega, "truth"),
        (omega_a, "(a) k R₁₁ + b, linear"),
        (omega_lut, "(b) MBSP, look-up table"),
    ],
    strict=True,
):
    im = a.imshow(img, vmin=-0.05, vmax=0.3, cmap="magma", origin="lower")
    a.set_title(title, fontsize=10)
    a.set_xticks([]), a.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.85, label="ΔΩ (kg/m²)")
fig.savefig(FIG / "retrievals.svg")
plt.close(fig)

# %%
assert stage4["monotone"] and stage4["shape"] == (7, 81)
# the table removes most of the saturation loss; the rest is the ring's offset
assert round(stage4["lut"]["ratio"], 2) == 0.96
assert round(stage4["lut"]["rmse"], 4) == 0.0175 < stage3["b_mbsp"]["rmse"]

# %% [markdown]
# ## Stage 5 · Hyperspectral: continuum fitting against a matched filter
#
# An EMIT-like cube: 68 bands from 2000 to 2500 nm, 7.4 nm apart with an
# 8.5 nm FWHM. Surfaces have a smooth quadratic continuum, and 15 % of the
# pixels carry a carbonate absorption near 2340 nm, inside the methane band.
# (a) The source note's continuum fit: a quadratic over the 2120–2150 and
# 2430–2470 nm windows, then a least-squares column over 2200–2400 nm.
# (b) geotoolz's matched filter with a linearised target, its background
# fitted on the whole scene (b1) or with the suspected plume left out (b2).

# %%
NH = 120
centres = np.arange(2000.0, 2500.0, 7.4)
WLH = np.arange(1995.0, 2505.0, 0.01)
sigma_h = np.interp(WLH, WL, sigma, right=0.0)
srf_h = reflectance.srf(centres, np.full(centres.size, 8.5), WLH)  # today
srf_h = srf_h / srf_h.sum(0, keepdims=True)  # (G, C)
lam = centres - 2250.0
# surface reflectance: a + b λ + c λ² per pixel, plus carbonate on 15 % of
# pixels: calcite's absorption near 2340 nm sits inside the methane band
a0 = 0.25 + 0.08 * rng.random((NH, NH))
a1 = 1e-4 * rng.normal(size=(NH, NH))
a2 = -2e-7 * rng.random((NH, NH))
surf = a0[..., None] + a1[..., None] * lam + a2[..., None] * lam**2
carb = rng.random((NH, NH)) < 0.15
dip = 0.10 * np.exp(-0.5 * ((centres - 2340) / 12) ** 2)
surf = surf * (1 - carb[..., None] * dip)
yh, xh = np.mgrid[0:NH, 0:NH]
d_omega_h = 0.15 * np.exp(-0.5 * (((xh - 60) / 14.0) ** 2 + ((yh - 60) / 6.0) ** 2))
plume_h = d_omega_h > 0.005


def band_T(omega):
    # T_c(Ω) = Σ_λ f_c(λ) exp(−m σ(λ) Ω)                    (n,) → (n, C)
    return np.exp(-M_AMF * np.outer(omega, sigma_h)) @ srf_h


d_grid = np.linspace(0.0, 0.2, 201)
Tg = band_T(OMEGA_BG + d_grid)  # (201, C)
T_pix = np.stack(
    [np.interp(d_omega_h.ravel(), d_grid, Tg[:, j]) for j in range(centres.size)], -1
).reshape(NH, NH, -1)
cube = surf * T_pix + 0.002 * rng.normal(size=surf.shape)  # (H, W, C)
s_bar = sigma_h @ srf_h  # σ̄_c (C,)

# (a) continuum: quadratic over the windows, column from 2200–2400 nm
win = ((centres >= 2120) & (centres <= 2150)) | ((centres >= 2430) & (centres <= 2470))
fit = (centres >= 2200) & (centres <= 2400)
V = np.vander(lam, 3)
coef, *_ = np.linalg.lstsq(V[win], cube[..., win].reshape(-1, win.sum()).T, rcond=None)
cont = (V @ coef).T.reshape(NH, NH, -1)
tau_h = -np.log(cube / cont)
A = M_AMF * s_bar[fit]
omega_total = (tau_h[..., fit] @ A) / (A @ A)
omega_cont = omega_total - np.median(omega_total)  # remove the background column
# (b) matched filter: target t = −μ ⊙ m σ̄, unit kg/m²
mu = np.median(cube.reshape(-1, centres.size), 0)
target = -mu * M_AMF * s_bar
bands_first = np.moveaxis(cube, -1, 0)  # (C, H, W)
# (b1) the background fitted on the whole scene, plume included
mf = MatchedFilter(target=target, fit_on_call=True, axis=0)  # today
omega_mf = np.asarray(mf(bands_first))
# (b2) the background fitted with the suspected plume left out (NaN pixels
# are excluded from the fit), then applied to the whole scene
masked = bands_first.copy()
masked[:, d_omega_h > 1e-3] = np.nan
mf_bg = MatchedFilter(target=target, axis=0).fit(masked)  # today
omega_mf_bg = np.asarray(mf_bg.transform(bands_first))


def score_h(om):
    off = ~plume_h & (d_omega_h < 0.001)
    return {
        "ratio": float(om[plume_h].sum() / d_omega_h[plume_h].sum()),
        "carb_bias": float(om[carb & off].mean()),
        "bare_bias": float(om[~carb & off].mean()),
        "noise": float(om[~carb & off].std()),
    }


# xrtoolz: the plume peak as a mixing-ratio enhancement over a 500 m layer
dvmr = column_mass_to_delta_vmr(
    xr.DataArray(0.15), pressure_pa=101325.0, temperature_k=288.0, path_length_m=500.0
)  # today
stage5 = {
    "n_bands": int(centres.size),
    "n_win": int(win.sum()),
    "n_fit": int(fit.sum()),
    "cont": score_h(omega_cont),
    "mf": score_h(omega_mf),
    "mf_bg": score_h(omega_mf_bg),
    "peak_ppm_500m": float(dvmr) * 1e6,
}
stage5

fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.3), constrained_layout=True)
for a, (img, title) in zip(
    axs,
    [
        (d_omega_h, "truth"),
        (omega_cont, "(a) continuum fit"),
        (omega_mf_bg, "(b2) matched filter"),
    ],
    strict=True,
):
    im = a.imshow(img, vmin=-0.03, vmax=0.15, cmap="magma", origin="lower")
    a.set_title(title, fontsize=10)
    a.set_xticks([]), a.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.85, label="ΔΩ (kg/m²)")
fig.savefig(FIG / "hyperspectral.svg")
plt.close(fig)

# %%
assert (stage5["n_bands"], stage5["n_win"], stage5["n_fit"]) == (68, 9, 27)
# the continuum fit reads carbonate as methane: +0.0069 kg/m² over carbonate
# against −0.0029 over bare surface
assert round(stage5["cont"]["carb_bias"], 4) == 0.0069
assert round(stage5["cont"]["bare_bias"], 4) == -0.0029
assert round(stage5["cont"]["ratio"], 2) == 0.95
# the matched filter learns carbonate from the scene covariance, but a plume
# inside the background fit absorbs 28 % of the signal
assert round(stage5["mf"]["ratio"], 2) == 0.72
assert round(stage5["mf_bg"]["ratio"], 2) == 0.91
assert abs(stage5["mf_bg"]["carb_bias"]) < 5e-4
assert round(stage5["mf_bg"]["noise"], 4) == 0.0016 < stage5["cont"]["noise"]
assert round(stage5["cont"]["noise"], 4) == 0.0095

assert round(stage5["peak_ppm_500m"]) == 442  # 0.15 kg/m² over 500 m, in ppm

# %% [markdown]
# ## The provenance figure
#
# Where each term of the forward model comes from, drawn with the docs'
# shared figure helpers.

# %%
spec = importlib.util.spec_from_file_location(
    "overview_figures", root / "docs/images/overview/make_figures.py"
)
ov = importlib.util.module_from_spec(spec)
sys.modules["overview_figures"] = ov
spec.loader.exec_module(ov)

rows = [  # (source, item, operation, term, hue)
    (
        "satellite L1 product",
        "digital numbers or radiance",
        "scale, offset, π L / E cos θ_s",
        "R_b",
        ov.OBS,
    ),
    (
        "satellite metadata",
        "θ_s, θ_v per pixel or tile",
        "1/cos θ_s + 1/cos θ_v",
        "m",
        ov.OBS,
    ),
    ("agency SRF table", "f_b(λ)", "⟨·⟩_b over the band", "σ̄_b, T_b", ov.ENG),
    (
        "line list (HITRAN)",
        "σ(λ) at p, T",
        "line-by-line, then ⟨·⟩_b",
        "σ̄_b, T_b",
        ov.ENG,
    ),
    (
        "scene statistics",
        "clean pixels or windows",
        "proxy or continuum fit",
        "ρ_b τ_bg,b",
        ov.ML,
    ),
    (
        "radiative transfer model",
        "T_b over (m, ΔΩ, H₂O)",
        "tabulate once",
        "LUT",
        ov.STATE,
    ),
]
W = 760
svg = ov.Svg(
    W,
    70 + 64 * len(rows),
    "Where each term of the Beer–Lambert retrieval comes from: the L1 product"
    " gives reflectance, its metadata the air-mass factor, the SRF table and"
    " a line list the band cross-sections, scene statistics the baseline, and"
    " a radiative transfer model the look-up table.",
)
for i, h in enumerate(["source", "what it gives", "operation", "term"]):
    svg.text(
        [24, 222, 420, 640][i],
        30,
        h,
        size=12.5,
        weight=700,
        color=ov.MUTED,
        anchor="start",
    )
for r, (src, item, op, term, hue) in enumerate(rows):
    y = 46 + 64 * r
    svg.rect(24, y, 180, 46, stroke=hue[0], fill=hue[1])
    svg.text(114, y + 28, src, size=12, color=ov.INK)
    svg.text(222, y + 28, item, size=12, color=ov.INK, anchor="start")
    svg.arrow([(400, y + 23), (416, y + 23)], color=ov.LINE)
    svg.text(420, y + 28, op, size=11.5, color=ov.MUTED, anchor="start")
    svg.rect(636, y + 6, 100, 34, stroke=ov.STATE[0], fill=ov.CARD, rx=17)
    svg.text(686, y + 28, term, size=12.5, weight=700, color=ov.STATE[0])
svg.h = 46 + 64 * len(rows)
(FIG / "provenance.svg").write_text(svg.render(), encoding="utf-8")
sorted(p.name for p in FIG.iterdir())

# %%
assert sorted(p.name for p in FIG.iterdir()) == [
    "curve_of_growth.svg",
    "hyperspectral.svg",
    "provenance.svg",
    "retrievals.svg",
]
