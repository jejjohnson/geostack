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
# # radtrax · 2 · The SBMP, MBSP and MBMP family, run
#
# Regenerates every number and figure in `docs/geomodels/radtrax/02_mbmp.md`:
# the multi-band and multi-pass Sentinel-2 retrievals derived from the
# radiative transfer equation, on a synthetic two-pass scene with every
# nuisance the derivations name.
#
# 1. the two SWIR simplifications of the RTE, in numbers (stage 1);
# 2. the band model and its linearisation: α_b against σ̄_b, and plumax's
#    band LUT against the background (stage 2);
# 3. a two-pass scene: a plume, a stable mineral patch, a wetted field, a
#    patch whose spectral ratio changes, a scene-wide brightening and a
#    scene-wide water-vapour change (stage 3);
# 4. the retrievals, side by side, with geotoolz's `SBMP` operator (stage 4);
# 5. noise (stage 5).
#
# **The methane spectrum is synthetic**, drawn exactly as in radtrax · 1 (same
# seed, same lines, σ̄₁₁ = 0.04 and σ̄₁₂ = 0.40 m²/kg). Sentinel-2A's band 11
# and 12 SRFs are real, from georeader.
#
# Pinned revisions: georeader-spaceml 2.4.1, geotoolz 0.10.2 at `088475e`
# (`plume.SBMP`), plumax 0.1.2 at `28f257e` (`radtran.nb_lut`; a clone, not a
# submodule yet). Needs NumPy, SciPy, xarray, pandas, openpyxl and matplotlib.

# %%
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "radtrax_02"  # stable SVG ids across reruns
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from georeader.readers.S2_SAFE_reader import read_srf  # noqa: E402
from geotoolz.plume import SBMP  # noqa: E402
from plumax.radtran.nb_lut import NBLookup, build_nb_lut, inject_plume  # noqa: E402
from plumax.radtran.srf import SpectralResponseFunction  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)  # openpyxl, SRF sheet
warnings.filterwarnings("ignore", category=RuntimeWarning)  # SRF normalisation
SEED = 0
root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
FIG = root / "docs/images/radtrax_02_mbmp"
FIG.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Stage 1 · Why the SWIR RTE reduces to extinction
#
# Two terms of μ dI/dτ = I − J are dropped. Thermal emission: the Planck
# radiance of a 300 K surface against reflected sunlight at 2.2 μm. Rayleigh
# scattering: its optical depth, τ_R(λ) ≈ 0.008569 λ⁻⁴ (1 + 0.0113 λ⁻² +
# 0.00013 λ⁻⁴), λ in μm (Hansen and Travis, 1974).

# %%
H_PL, C_L, K_B = 6.62607015e-34, 2.99792458e8, 1.380649e-23


def planck(lam_um, t_k):
    # B_λ(T) = 2hc² / λ⁵ / (exp(hc / λ k T) − 1), W m⁻² sr⁻¹ m⁻¹
    lam = lam_um * 1e-6
    return 2 * H_PL * C_L**2 / lam**5 / np.expm1(H_PL * C_L / (lam * K_B * t_k))


def tau_rayleigh(lam_um):
    return 0.008569 * lam_um**-4 * (1 + 0.0113 * lam_um**-2 + 0.00013 * lam_um**-4)


# sunlight at the top of the atmosphere: a 5778 K disc, diluted by
# (R_sun / 1 au)²; reflected by ρ = 0.2 at θ_s = 35°
DILUTE = (6.957e8 / 1.495978707e11) ** 2
E_SUN = np.pi * planck(2.2, 5778.0) * DILUTE
L_refl = 0.2 * E_SUN * np.cos(np.radians(35.0)) / np.pi
stage1 = {
    "thermal_over_reflected_2p2": float(planck(2.2, 300.0) / L_refl),
    "tau_r_1p0": float(tau_rayleigh(1.0)),
    "tau_r_2p2": float(tau_rayleigh(2.2)),
    "rayleigh_ratio_1_to_2": float(tau_rayleigh(1.0) / tau_rayleigh(2.0)),
}
stage1

# %%
# a 300 K surface emits under 0.1 % of what it reflects at 2.2 μm
assert stage1["thermal_over_reflected_2p2"] < 1e-3
# Rayleigh: ~16× weaker at 2 μm than at 1 μm, and τ_R < 10⁻³ in band 12
assert round(stage1["rayleigh_ratio_1_to_2"]) == 16
assert stage1["tau_r_2p2"] < 1e-3

# %% [markdown]
# ## Stage 2 · The band model and its linearisation
#
# R_b = A_b T_b(Ω), T_b(Ω) = ⟨exp(−m σ(λ) Ω)⟩_b. Around the background,
# T_b(Ω_bg + ΔΩ) ≈ T_b(Ω_bg) exp(−m α_b ΔΩ) with
# α_b = −(1/m) ∂ ln T_b / ∂Ω at Ω_bg. The line spectrum is radtrax · 1's.

# %%
rng = np.random.default_rng(SEED)
WL = np.arange(1500.0, 2500.0, 0.01)  # nm                               (G,)
HWHM = 0.1


def lines(n: int, lo: float, hi: float, centre: float, width: float):
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


band_23 = lorentz_sum(*lines(3000, 2165.0, 2420.0, 2310.0, 70.0))
band_166 = lorentz_sum(*lines(600, 1620.0, 1710.0, 1666.0, 15.0))
srf_s2 = read_srf("S2A")  # today
f11 = np.interp(WL, srf_s2.index.values, srf_s2["B11"].values, left=0, right=0)
f12 = np.interp(WL, srf_s2.index.values, srf_s2["B12"].values, left=0, right=0)


def band_mean(x, f):
    # ⟨x⟩_b = ∫ x f dλ / ∫ f dλ
    return (x * f).sum(-1) / f.sum()


SIG12, RATIO11 = 0.40, 0.10  # a modelling choice, as in radtrax · 1
sigma = band_23 * SIG12 / band_mean(band_23, f12)
sigma = sigma + band_166 * (RATIO11 * SIG12 - band_mean(sigma, f11)) / band_mean(
    band_166, f11
)
s11, s12 = float(band_mean(sigma, f11)), float(band_mean(sigma, f12))
OMEGA_BG = 1.9e-6 * (16.04 / 28.96) * 101325.0 / 9.80665  # kg/m²
SZA, VZA = 35.0, 5.0
M_AMF = 1 / np.cos(np.radians(SZA)) + 1 / np.cos(np.radians(VZA))


def T_band(omega, f, m=M_AMF):
    # T_b(Ω) = ⟨exp(−m σ Ω)⟩_b, for a vector of Ω                  (n,) → (n,)
    sel = f > 0
    sig_b, f_b = sigma[sel], f[sel]
    omega = np.atleast_1d(np.asarray(omega, float))
    return np.array([(np.exp(-m * sig_b * o) * f_b).sum() / f_b.sum() for o in omega])


def alpha(f, h=1e-6):
    # α_b = −(1/m) ∂ ln T_b / ∂Ω at Ω_bg, central difference
    t = T_band([OMEGA_BG + h, OMEGA_BG - h], f)
    return float(-(np.log(t[0]) - np.log(t[1])) / (2 * h) / M_AMF)


a11, a12 = alpha(f11), alpha(f12)


def tau_exact(d_omega, f):
    # τ_b(ΔΩ) = −ln(T_b(Ω_bg + ΔΩ) / T_b(Ω_bg))
    return -np.log(T_band(OMEGA_BG + np.asarray(d_omega), f) / T_band(OMEGA_BG, f))


# plumax's band LUT (nB) drops the background: nB_b(ΔX) = ⟨exp(−m σ ΔX)⟩_b.
# Build it from the same spectrum, as a HAPI-style dataset in cm²/molecule
N_A, M_CH4 = 6.02214076e23, 0.01604  # 1/mol, kg/mol
sig_cm2 = sigma * M_CH4 / N_A * 1e4  # m²/kg → cm²/molecule              (G,)
ds = xr.Dataset(
    {"absorption_cross_section": (("wavenumber", "temperature", "pressure"),
                                  np.repeat(np.repeat(sig_cm2[:, None, None], 2, 1),
                                            2, 2))},
    coords={"wavenumber": 1e7 / WL, "temperature": [250.0, 300.0],
            "pressure": [0.5, 1.0]},
)
srf = SpectralResponseFunction(
    wavelengths_hr_nm=WL, band_centers_nm=np.array([1610.0, 2190.0]),
    band_widths_nm=np.array([90.0, 175.0]), band_names=("B11", "B12"),
    srf_type="custom", custom_srfs=np.stack([f11, f12]))               # today
D_MAX = 0.45  # kg/m², the top of every table on this page
nb = build_nb_lut(ds, srf, T_K=280.0, p_atm=0.8, amf=M_AMF,
                  max_delta_column=D_MAX / M_CH4, n_grid=181)          # today
grid = nb.delta_column * M_CH4  # mol/m² → kg/m²                        (181,)
tau_nb12 = -np.log(nb.nB[1])
tau_ex12 = tau_exact(grid, f12)
stage2 = {
    "s11": s11, "s12": s12, "a11": a11, "a12": a12,
    # τ_b at the plume peak, 0.3 kg/m², against the linear m α_b ΔΩ
    "lin_err_12_at_0p3": float(M_AMF * a12 * 0.3 / tau_exact(0.3, f12)[0] - 1),
    "dtau_mb_at_0p3": float(tau_exact(0.3, f12)[0] - tau_exact(0.3, f11)[0]),
    # plumax's nB against the background-aware ratio, band 12, at 0.3 kg/m²
    "nb_over_exact_12": float(np.interp(0.3, grid, tau_nb12)
                              / np.interp(0.3, grid, tau_ex12)),
}
{k: round(v, 4) for k, v in stage2.items()}

# %%
# α_b sits just below σ̄_b: the background has already darkened line cores
assert (round(stage2["a11"], 4), round(stage2["a12"], 4)) == (0.0398, 0.3953)
assert stage2["a12"] < stage2["s12"] and stage2["a11"] < stage2["s11"]
# the linearisation overstates band 12's optical depth by 16 % at 0.3 kg/m²
assert round(stage2["lin_err_12_at_0p3"], 2) == 0.16
# plumax's background-free nB overstates it by 1.1 % there
assert round(stage2["nb_over_exact_12"], 3) == 1.011

# %% [markdown]
# ## Stage 3 · A two-pass scene with every nuisance
#
# 200 × 200 pixels. A plume (peak 0.3 kg/m²) on the plume pass only. The
# surface has a spatially textured albedo A₁₁ and spectral ratio
# r = A₁₂ / A₁₁ (4 % texture), and a mineral patch whose ratio is 15 % low,
# on both passes. Between passes: the whole scene brightens 5 %, a field is
# wetted (both bands ×0.8), one patch's spectral ratio drops 3 %, and the
# water-vapour transmission of band 12 drops 1 % scene-wide.

# %%
NY = NX = 200
yy, xx = np.mgrid[0:NY, 0:NX].astype(float)
plume = 0.3 * np.exp(-0.5 * (((xx - 70) / 25) ** 2 + ((yy - 100) / 8) ** 2))
plume[xx < 20] *= np.exp(-(((20 - xx[xx < 20]) / 6) ** 2))  # a source at x ≈ 20
srng = np.random.default_rng(SEED + 1)


def texture(scale: float):
    # zero-mean, unit-sd smooth field
    g = gaussian_filter(srng.normal(size=(NY, NX)), scale)
    return (g - g.mean()) / g.std()


A11 = 0.25 * (1 + 0.15 * texture(6.0))
ratio = 0.80 * (1 + 0.04 * texture(3.0))
regions = {
    "mineral": (yy >= 140) & (yy < 185) & (xx >= 10) & (xx < 90),
    "wetted": (yy >= 20) & (yy < 60) & (xx >= 130) & (xx < 180),
    "ratio_change": (yy >= 140) & (yy < 180) & (xx >= 130) & (xx < 190),
}
ratio = np.where(regions["mineral"], 0.85 * ratio, ratio)
regions["plume"] = plume > 0.01
regions["clean"] = ~(
    gaussian_filter(regions["plume"].astype(float), 4) > 0.01
) & ~regions["mineral"] & ~regions["wetted"] & ~regions["ratio_change"]

# the background-aware band table, for injection and for exact inversion
nb_bg = NBLookup(
    delta_column=nb.delta_column,
    nB=np.stack([T_band(OMEGA_BG + grid, f) / T_band(OMEGA_BG, f) for f in (f11, f12)]),
    band_names=("B11", "B12"))
T_BG = np.array([T_band(OMEGA_BG, f11)[0], T_band(OMEGA_BG, f12)[0]])


def scene(a11, r, d_omega, h2o12=1.0):
    # R_b = A_b T_b(Ω_bg) · T_b(Ω_bg + ΔΩ) / T_b(Ω_bg)        → (2, NY, NX)
    clean = np.stack([a11 * T_BG[0], a11 * r * T_BG[1] * h2o12])
    return inject_plume(clean, d_omega / M_CH4, nb_bg)                  # today


ref = scene(A11, ratio, 0.0 * plume)
wet = np.where(regions["wetted"], 0.8, 1.0)
r_p = np.where(regions["ratio_change"], 0.97 * ratio, ratio)
obs = scene(1.05 * wet * A11, r_p, plume, h2o12=0.99)
stage3 = {"pixels": {k: int(v.sum()) for k, v in regions.items()},
          "plume_mass_kg": float(plume.sum() * 20.0**2)}
stage3

# %%
assert stage3["pixels"]["plume"] == 4097
assert round(stage3["plume_mass_kg"], -2) == 148700  # with 20 m pixels

# %% [markdown]
# ## Stage 4 · The retrievals
#
# Scale factors are least-squares fits through the origin. Source note (1)
# fits them over every pixel; derivation (2) over background pixels only,
# here the pixels outside a first-pass detection (ΔΩ > 0.03 kg/m², grown
# by a Gaussian of 2 pixels). Unless a row says otherwise, ΔΩ comes from the exact band
# tables: τ₁₂(ΔΩ) for one band, g(ΔΩ) = τ₁₂(ΔΩ) − τ₁₁(ΔΩ) for two.

# %%
R11p, R12p = obs
R11r, R12r = ref


def fit_c(y, x, keep=None):
    # c = argmin Σ (y − c x)² = Σ x y / Σ x², over the kept pixels
    keep = np.ones(y.shape, bool) if keep is None else keep
    return float((x[keep] * y[keep]).sum() / (x[keep] ** 2).sum())


tau12_tab = -np.log(nb_bg.nB[1])                  # τ₁₂ on the grid      (181,)
g_mb = tau12_tab + np.log(nb_bg.nB[0])            # τ₁₂ − τ₁₁            (181,)


def invert(s, table):
    # ΔΩ = table⁻¹(s), odd-extended for negative s
    return np.sign(s) * np.interp(np.abs(s), table, grid)


def mb_signals(keep):
    # ln(1 + δ) per pass, δ = c R₁₂ / R₁₁ − 1, c from R₁₁ ≈ c R₁₂
    cp, cr = fit_c(R11p, R12p, keep), fit_c(R11r, R12r, keep)
    dp, dr = cp * R12p / R11p - 1, cr * R12r / R11r - 1
    return dp, dr, cp, cr


# pass 1: c over every pixel, a first detection, then the background mask
dp_all, dr_all, cp_all, cr_all = mb_signals(None)
first = invert(-(np.log1p(dp_all) - np.log1p(dr_all)), g_mb)
bg = ~(gaussian_filter((first > 0.03).astype(float), 2) > 0.05)
dp, dr, c_p, c_r = mb_signals(bg)
c_sb = fit_c(R12r, R12p, bg)                     # SBMP: R₁₂′ ≈ c R₁₂
DA = a12 - a11
gz_score = np.squeeze(SBMP(swir1=0, swir2=1, axis=0)(obs, ref))          # today
retrievals = {
    # (1): one band, two passes, A_p = A_r, no scale factor
    "sbmp_note": invert(-np.log(R12p / R12r), tau12_tab),
    # one band, two passes, a scene-wide scale factor
    "sbmp_scaled": invert(-np.log(c_sb * R12p / R12r), tau12_tab),
    # one pass, two bands
    "mbsp": invert(-np.log1p(dp), g_mb),
    # MBMP, the exact table, c over the background
    "mbmp": invert(-(np.log1p(dp) - np.log1p(dr)), g_mb),
    # MBMP with c over every pixel, as (1) fits it
    "mbmp_c_all": invert(-(np.log1p(dp_all) - np.log1p(dr_all)), g_mb),
    # MBMP, log of the ratio of ratios, linear in m (α₁₂ − α₁₁)
    "mbmp_log_lin": -(np.log1p(dp) - np.log1p(dr)) / (M_AMF * DA),
    # (1): δ differenced, linear: ΔΩ = −(δ − δ′) / (m (α₁₂ − α₁₁))
    "mbmp_delta_lin": -(dp - dr) / (M_AMF * DA),
    # (2): δ differenced, σ₁₁ = 0, Taylor: ΔΩ = −(δ − δ′) / (m σ̄₁₂)
    "mbmp_taylor_s12": -(dp - dr) / (M_AMF * s12),
    # geotoolz: ln(R₁₁/R₁₂) − ln(R₁₁′/R₁₂′), no scale factors, exact table
    "geotoolz_sbmp": invert(gz_score, g_mb),
}


def summarise(est):
    # the offset is the mean over clean pixels; a patch's artefact is its
    # mean minus the offset
    off = float(est[regions["clean"]].mean())
    out = {"mass_frac": float(est[regions["plume"]].sum()
                              / plume[regions["plume"]].sum()),
           "offset": off,
           "clean_sd": float(est[regions["clean"]].std())}
    for k in ("mineral", "wetted", "ratio_change"):
        out[k] = float(est[regions[k]].mean()) - off
    return out


stage4 = {k: summarise(v) for k, v in retrievals.items()}
stage4["gz_minus_log_sd"] = float(np.std(gz_score + np.log1p(dp) - np.log1p(dr)))
stage4["gz_minus_log_mean"] = float(np.mean(gz_score + np.log1p(dp) - np.log1p(dr)))
stage4["ln_cp_over_cr"] = float(np.log(c_p / c_r))
stage4["c_all_over_c_bg"] = float(cp_all / c_p - 1)
stage4["bg_mask_misses"] = int((regions["plume"] & bg).sum())
{k: ({a: round(b, 4) for a, b in v.items()} if isinstance(v, dict) else v)
 for k, v in stage4.items()}

# %%
S = stage4
# the note's SBMP (no scale factor) reads the brightening and the water
# vapour as an offset, and the wetted field as 0.28 kg/m² of methane
assert round(S["sbmp_note"]["offset"], 3) == -0.045
assert round(S["sbmp_note"]["wetted"], 2) == 0.28
# a scene-wide factor removes most of the offset, not the wetted field;
# the wetted field also biases the factor, leaving −0.011 kg/m²
assert round(S["sbmp_scaled"]["offset"], 3) == -0.011
assert round(S["sbmp_scaled"]["wetted"], 2) == 0.29
# MBSP: the stable mineral patch and the 4 % ratio texture
assert round(S["mbsp"]["mineral"], 2) == 0.21 and round(S["mbsp"]["clean_sd"], 3) == 0.054
# MBMP cancels the mineral patch, the wetted field and both scene-wide
# changes, not the patch whose ratio changed (3 % → 0.039 kg/m²)
assert abs(S["mbmp"]["mineral"]) < 1e-3 and abs(S["mbmp"]["wetted"]) < 1e-3
assert abs(S["mbmp"]["offset"]) < 1e-3 and S["mbmp"]["clean_sd"] < 1e-3
assert round(S["mbmp"]["ratio_change"], 3) == 0.039
assert round(S["mbmp"]["mass_frac"], 3) == 0.988
# c over every pixel: the plume moves c by 0.31 % → an offset, 11 % of mass
assert round(S["c_all_over_c_bg"], 4) == 0.0031
assert round(S["mbmp_c_all"]["offset"], 4) == -0.0088
assert round(S["mbmp_c_all"]["mass_frac"], 2) == 0.89
# linear inversions read the bent curve of growth as less methane
assert round(S["mbmp_log_lin"]["mass_frac"], 3) == 0.907
assert round(S["mbmp_delta_lin"]["mass_frac"], 3) == 0.876
# (2)'s σ₁₁ = 0 with σ̄₁₂: 22 % low
assert round(S["mbmp_taylor_s12"]["mass_frac"], 2) == 0.78
# geotoolz's SBMP is the MBMP log signal plus ln(c_p / c_r), a constant;
# here that constant is the 1 % water-vapour change, 0.013 kg/m²
assert S["gz_minus_log_sd"] < 1e-9
assert abs(S["gz_minus_log_mean"] - S["ln_cp_over_cr"]) < 1e-9
assert abs(S["geotoolz_sbmp"]["offset"] - invert(-np.log(0.99), g_mb)) < 5e-4
assert abs(S["geotoolz_sbmp"]["mineral"]) < 1e-3 and abs(S["geotoolz_sbmp"]["wetted"]) < 1e-3
assert round(S["geotoolz_sbmp"]["mass_frac"], 2) == 1.16

# %% [markdown]
# ## Stage 5 · Noise
#
# 1 % multiplicative noise per band and pass, on a stable surface. The
# retrieval noise is the noisy retrieval minus the noise-free one, off the
# plume. Theory, to first order: SBMP √2 ε / (m α₁₂), MBSP √2 ε / (m Δα),
# MBMP 2 ε / (m Δα).

# %%
EPS = 0.01
nrng = np.random.default_rng(SEED + 2)
ref_n = ref * (1 + EPS * nrng.normal(size=ref.shape))
obs_s = scene(A11, ratio, plume)
obs_n = obs_s * (1 + EPS * nrng.normal(size=obs_s.shape))


def triple(o, r):
    cs = fit_c(r[1], o[1])
    cp, cr = fit_c(o[0], o[1]), fit_c(r[0], r[1])
    dp, dr = cp * o[1] / o[0] - 1, cr * r[1] / r[0] - 1
    return {"sbmp": -np.log(cs * o[1] / r[1]) / (M_AMF * a12),
            "mbsp": -np.log1p(dp) / (M_AMF * DA),
            "mbmp": -(np.log1p(dp) - np.log1p(dr)) / (M_AMF * DA)}


clean_est, noisy_est = triple(obs_s, ref), triple(obs_n, ref_n)
off = ~(gaussian_filter(regions["plume"].astype(float), 4) > 0.01)
stage5 = {k: float(np.std((noisy_est[k] - clean_est[k])[off])) for k in clean_est}
stage5["theory"] = {"sbmp": np.sqrt(2) * EPS / (M_AMF * a12),
                    "mbsp": np.sqrt(2) * EPS / (M_AMF * DA),
                    "mbmp": 2 * EPS / (M_AMF * DA)}
stage5

# %%
for k in ("sbmp", "mbsp", "mbmp"):
    assert abs(stage5[k] / stage5["theory"][k] - 1) < 0.03
assert (round(stage5["sbmp"], 3), round(stage5["mbsp"], 3),
        round(stage5["mbmp"], 3)) == (0.016, 0.018, 0.025)

# %% [markdown]
# ## Figure · the methods on one scene

# %%
show = [("truth", plume), ("SBMP (note)", retrievals["sbmp_note"]),
        ("SBMP (scaled)", retrievals["sbmp_scaled"]), ("MBSP", retrievals["mbsp"]),
        ("MBMP", retrievals["mbmp"]), ("geotoolz SBMP", retrievals["geotoolz_sbmp"])]
fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), constrained_layout=True)
for ax, (name, im) in zip(axes.flat, show, strict=True):
    h = ax.imshow(im, cmap="RdBu_r", vmin=-0.3, vmax=0.3)
    for key, col in (("mineral", "#a16207"), ("wetted", "#1d4ed8"),
                     ("ratio_change", "#7c3aed")):
        ax.contour(regions[key], levels=[0.5], colors=col, linewidths=1.0)
    ax.set_title(name, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
fig.colorbar(h, ax=axes, shrink=0.7, label="ΔΩ (kg/m²)")
axes[0, 0].text(12, 165, "mineral", color="#a16207", fontsize=8)
axes[0, 0].text(132, 15, "wetted", color="#1d4ed8", fontsize=8)
axes[0, 0].text(132, 195, "ratio change", color="#7c3aed", fontsize=8)
fig.savefig(FIG / "methods.svg", metadata={"Date": None})
plt.close(fig)

# %%
assert sorted(p.name for p in FIG.iterdir()) == ["methods.svg"]
