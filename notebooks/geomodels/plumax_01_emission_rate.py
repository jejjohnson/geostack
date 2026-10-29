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
# # plumax · 1 · Emission rate from first principles, run
#
# Regenerates every number and figure in
# `docs/geomodels/plumax/01_emission_rate.md`: five ways to estimate a point
# source's emission rate Q from a column map, each an integral of the same
# advection–diffusion equation.
#
# 1. the column of plumax's Gaussian plume, and its independence of the
#    release height (stage 1);
# 2. the cross-sectional flux, with geotoolz's `CrossSectionalFlux`, and the
#    effective wind under shear (stage 2);
# 3. puffs superposed into a plume, and where the two differ (stage 3);
# 4. the divergence method on a grid, and Gauss's theorem (stage 4);
# 5. the integrated mass enhancement: the flux through a threshold contour,
#    and the effective wind each length definition needs (stage 5).
#
# Everything is analytic or noise-free unless a stage says otherwise; the
# noise draw is seeded.
#
# Pinned revisions: plumax 0.1.2 at `28f257e`, geotoolz 0.10.2 at `088475e`
# and georeader-spaceml 2.4.1, all from `references/`. It needs JAX, NumPy, SciPy, scikit-image and matplotlib; it
# runs in under a minute on a laptop CPU.

# %%
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "plumax_01"  # stable SVG ids across reruns
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio.transform  # noqa: E402
from georeader.geotensor import GeoTensor  # noqa: E402
from geoops.plume import CrossSectionalFlux, IMEEstimate  # noqa: E402
from geoops.plume._src.array import plume_length  # noqa: E402
from plumax.gauss_plume import get_dispersion_params, plume_concentration  # noqa: E402
from plumax.gauss_puff.puff import puff_concentration  # noqa: E402
from scipy import integrate  # noqa: E402
from scipy.special import erf  # noqa: E402

jax.config.update("jax_enable_x64", True)
root = Path.cwd()
while not (root / "myst.yml").exists() and root != root.parent:
    root = root.parent
FIG = root / "docs/images/plumax_01_emission_rate"
FIG.mkdir(parents=True, exist_ok=True)
Q, U = 2.0, 3.0  # kg/s (7.2 t/h) and m/s along +x
D = get_dispersion_params("D")  # neutral stability, Briggs coefficients  # today

# %% [markdown]
# ## Stage 1 · The column of a Gaussian plume
#
# Ω(x, y) = ∫₀^∞ c dz. For the reflecting-ground plume the integral of the
# two images is ∫_{−∞}^{∞} exp(−(z−H)²/2σ_z²) dz = √(2π) σ_z for any H, so
# the column is Q / (√(2π) U σ_y) · exp(−y² / 2σ_y²) whatever the release
# height.

# %%
def column_numeric(x: float, y: float, h: float) -> float:
    # Ω = ∫₀^∞ c(x, y, z) dz with plumax's 3D plume
    f = lambda z: float(plume_concentration(  # noqa: E731
        jnp.array(x), jnp.array(y), jnp.array(z), 0.0, 0.0, h, U, 0.0, Q, D))
    return integrate.quad(f, 0.0, 2000.0, limit=200)[0]


def sigma_y(x):
    # σ_y(x) = a x (1 + b x)^c, Briggs, class D
    a, b, c = (float(v) for v in D[:3])
    return a * x * (1 + b * x) ** c


def column_analytic(x, y):
    # Ω(x, y) = Q / (√(2π) U σ_y) exp(−y² / 2σ_y²)              (…,) → (…,)
    s = sigma_y(x)
    return Q / (np.sqrt(2 * np.pi) * U * s) * np.exp(-0.5 * (y / s) ** 2)


pts = [(500.0, 0.0), (500.0, 40.0), (1500.0, 0.0)]
stage1 = {
    "rel_err_H0": max(abs(column_numeric(x, y, 0.0) / column_analytic(x, y) - 1)
                      for x, y in pts),
    "rel_err_H30": max(abs(column_numeric(x, y, 30.0) / column_analytic(x, y) - 1)
                       for x, y in pts),
    "sigma_y_500": sigma_y(500.0), "sigma_y_2000": sigma_y(2000.0),
    # Briggs grows like x near the source, not like √x as K-theory does
    "growth_exponent": float(np.log(sigma_y(400.0) / sigma_y(100.0)) / np.log(4.0)),
    "omega_peak_500": float(column_analytic(500.0, 0.0)),
}
stage1

# %%
assert stage1["rel_err_H0"] < 1e-6 and stage1["rel_err_H30"] < 1e-6
assert (round(stage1["sigma_y_500"], 1), round(stage1["sigma_y_2000"], 1)) == (39.0, 146.1)
assert round(stage1["growth_exponent"], 2) == 0.99  # K-theory would give 0.50
assert round(stage1["omega_peak_500"], 4) == 0.0068  # kg/m², 500 m downwind

# %% [markdown]
# The steady plume is the 2D heat equation in travel time τ = x / U started
# from the "initial condition" U c(0⁺, y, z) = Q δ(y) δ(z − H): the flux
# U ∬ c dy dz through every cross-section is Q, which is the cross-sectional
# flux in 3D.

# %%
# U ∬ c dy dz on a 1 m × 1 m (y, z) grid at x = 500 and 1500 m      (Y, Z) → ()
yg = jnp.linspace(-1500.0, 1500.0, 3001)
zg = jnp.linspace(0.0, 1500.0, 1501)
YG, ZG = jnp.meshgrid(yg, zg, indexing="ij")


def plume_flux(x: float, h: float) -> float:
    c = plume_concentration(jnp.full_like(YG, x), YG, ZG, 0.0, 0.0, h, U, 0.0, Q, D)
    return float(U * jnp.trapezoid(jnp.trapezoid(c, zg, axis=1), yg))


stage1["flux_3d"] = {x: plume_flux(x, 30.0) for x in (500.0, 1500.0)}
assert all(abs(v / Q - 1) < 1e-3 for v in stage1["flux_3d"].values())

# %% [markdown]
# ## Stage 2 · Cross-sectional flux
#
# Q = U ∫ Ω(x, y) dy at every x > 0. On a 20 m grid, geotoolz's
# `CrossSectionalFlux` sums the plume pixels on each transect; a detection
# threshold truncates the transect. Under wind shear, the wind that
# multiplies the column is the concentration-weighted one,
# U_eff(x) = ∫ u(z) c dz / ∫ c dz.

# %%
PIX = 20.0
xs = np.arange(-200.0, 3000.0, PIX)  # pixel centres on multiples of 20 m
ys = np.arange(-1000.0, 1000.0, PIX) + PIX / 2
XX, YY = np.meshgrid(xs, ys[::-1])  # rows north to south
omega = np.where(XX > 0, column_analytic(np.maximum(XX, 1.0), YY), 0.0)   # (H, W)
transform = rasterio.transform.from_origin(xs[0] - PIX / 2, ys[-1] + PIX / 2, PIX, PIX)
gt = GeoTensor(omega, transform=transform, crs="EPSG:32630", fill_value_default=np.nan)

THRESH = 0.002  # kg/m²: a detection limit
full = XX > 0
detected = omega > THRESH
csf = CrossSectionalFlux(source=(0.0, 0.0), wind_u=U, wind_v=0.0, n_transects=5,
                         transect_spacing_m=400.0)                       # today
q_full = csf(gt, full)["flux_kg_s"].to_numpy()
q_det = csf(gt, detected)["flux_kg_s"].to_numpy()


def u_profile(z):
    # u(z) = U (z / 10 m)^0.15, U at 10 m
    return U * (np.maximum(z, 0.5) / 10.0) ** 0.15


def u_eff(x: float) -> float:
    # U_eff = ∫ u c dz / ∫ c dz, ground-level source on the centreline
    c = lambda z: float(plume_concentration(  # noqa: E731
        jnp.array(x), jnp.array(0.0), jnp.array(z), 0.0, 0.0, 0.0, U, 0.0, Q, D))
    num = integrate.quad(lambda z: u_profile(z) * c(z), 0.0, 2000.0, limit=200)[0]
    return num / integrate.quad(c, 0.0, 2000.0, limit=200)[0]


stage2 = {"q_full": q_full.round(3).tolist(), "q_det": q_det.round(3).tolist(),
          "ueff_100": u_eff(100.0), "ueff_1000": u_eff(1000.0),
          "ueff_2000": u_eff(2000.0)}
stage2

# %%
# untruncated, every transect returns Q to the discretisation of a 20 m grid
assert all(abs(q - Q) / Q < 0.01 for q in q_full)
# a 0.002 kg/m² threshold truncates the far transects: Q is biased low there
assert stage2["q_det"] == [1.897, 1.616, 1.246, 0.774, 0.0]
# with shear, the column moves slower than U₁₀ near the source and faster
# once the plume is deeper than 10 m
assert (round(stage2["ueff_100"], 2), round(stage2["ueff_1000"], 2),
        round(stage2["ueff_2000"], 2)) == (2.55, 3.38, 3.62)

# %% [markdown]
# ## Stage 3 · Puffs superposed into a plume
#
# With constant K (K-theory, σᵢ² = 2Kᵢt), the column of one puff released at
# t′ is M / (2π σ_x σ_y) · exp(−(x − U(t − t′))² / 2σ_x² − y² / 2σ_y²). A
# steady release Q is the integral over t′; the steady plume drops K_x. The
# two agree when the Péclet number U x / K_x is large.

# %%
K = 10.0  # m²/s, all three directions


def puff_column(x, y, age, u):
    # Ω₁(x, y; age) = exp(−(x − u age)² / 4K age − y² / 4K age) / (4π K age)
    s2 = 2 * K * age
    return np.exp(-((x - u * age) ** 2 + y**2) / (2 * s2)) / (2 * np.pi * s2)


def superposed(x, u, horizon=3600.0, n=20_000):
    # Ω(x, 0) = ∫₀^T Q Ω₁(x, 0; a) da, release ages a ∈ (0, T]
    ages = np.linspace(horizon / n, horizon, n)
    return Q * np.trapezoid(puff_column(x, 0.0, ages, u), ages)


def kplume(x, u):
    # steady K-theory plume: Q / (√(2π) u σ_y), σ_y² = 2 K x / u
    return Q / (np.sqrt(2 * np.pi) * u * np.sqrt(2 * K * x / u))


def pe(x, u):
    return u * x / K


stage3 = {f"u{u:g}_x{x:g}": {"pe": pe(x, u),
                             "ratio": float(superposed(x, u) / kplume(x, u))}
          for u in (5.0, 1.0) for x in (50.0, 500.0)}
{k: {a: round(b, 3) for a, b in v.items()} for k, v in stage3.items()}

# %%
# Pe = 250: the plume formula is the puff sum to 0.1 %; Pe = 5: 4 % off
assert round(stage3["u5_x500"]["pe"]) == 250 and abs(stage3["u5_x500"]["ratio"] - 1) < 2e-3
assert round(stage3["u1_x50"]["pe"]) == 5 and round(stage3["u1_x50"]["ratio"], 3) == 0.958
assert round(stage3["u5_x50"]["ratio"], 3) == 0.990  # Pe = 25
assert round(stage3["u1_x500"]["ratio"], 3) == 0.995  # Pe = 50

# %% [markdown]
# The puff itself: in the frame x′ = x − U t moving with the wind, the
# advection term drops out and ∂c/∂t = K ∇²c; the product of three 1D heat
# kernels with σᵢ² = 2Kt, plus an image at z = −H for a reflecting ground,
# is plumax's `puff_concentration`. Check that it conserves mass, that it
# solves the moving-frame diffusion equation, and that its column is
# M / (2π σ_x σ_y) exp(…) for any release height.

# %%
M_PUFF, H_PUFF, T0 = 1.0, 30.0, 100.0  # kg, m, s since release


def puff(x, y, z, t):
    # c = M / ((2π)^{3/2} σ_x σ_y σ_z) e^{…} [e^{−(z−H)²/2σ_z²} + e^{−(z+H)²/2σ_z²}]
    s = jnp.sqrt(2 * K * t)
    return puff_concentration(x - U * t, y, z, 0.0, 0.0, H_PUFF, s, s, s, M_PUFF)


xg = jnp.linspace(-300.0, 300.0, 301) + U * T0  # 2 m spacing; σ = 45 m
yg3 = jnp.linspace(-300.0, 300.0, 301)
zg3 = jnp.linspace(0.0, 400.0, 201)
X3, Y3, Z3 = jnp.meshgrid(xg, yg3, zg3, indexing="ij")
c3 = puff(X3, Y3, Z3, T0)                                     # (301, 301, 201)
# M = ∭_{z ≥ 0} c dV
mass = float(jnp.trapezoid(jnp.trapezoid(jnp.trapezoid(c3, zg3), yg3), xg))
# Ω = ∫₀^∞ c dz against M / (2π σ²) e^{−r²/2σ²}, σ² = 2 K t
col = jnp.trapezoid(c3, zg3)                                         # (301, 301)
s2 = 2 * K * T0
col_an = M_PUFF / (2 * np.pi * s2) * jnp.exp(
    -((X3[..., 0] - U * T0) ** 2 + Y3[..., 0] ** 2) / (2 * s2))
# ∂c/∂t + U ∂c/∂x − K ∇²c at one point, by central differences
p0, e, dt = (U * T0 + 20.0, 15.0, 40.0), 1e-1, 1e-3
f = lambda *a: float(puff(*a))  # noqa: E731
lap = sum((f(*[p0[j] + e * (j == i) for j in range(3)], T0)
           - 2 * f(*p0, T0) + f(*[p0[j] - e * (j == i) for j in range(3)], T0)) / e**2
          for i in range(3))
dcdt = (f(*p0, T0 + dt) - f(*p0, T0 - dt)) / (2 * dt)
dcdx = (f(p0[0] + e, *p0[1:], T0) - f(p0[0] - e, *p0[1:], T0)) / (2 * e)
stage3["puff_mass"] = mass
stage3["puff_col_err"] = float(jnp.max(jnp.abs(col - col_an)) / jnp.max(col_an))
stage3["puff_pde_rel"] = abs(dcdt + U * dcdx - K * lap) / abs(dcdt)

# %%
assert abs(stage3["puff_mass"] - M_PUFF) < 1e-6
assert stage3["puff_col_err"] < 1e-6
assert stage3["puff_pde_rel"] < 1e-4

# %% [markdown]
# ## Stage 4 · The divergence method and Gauss's theorem
#
# q = ∇·F, F = u Ω. On the grid, central differences give q at every pixel;
# summed over a box around the source they give the box's emission, which
# Gauss's theorem equates to the net flux through its edges. The plume field
# here satisfies U ∂Ω/∂x = K_y ∂²Ω/∂y² away from the source, so the
# crosswind diffusive flux through the side edges is what the advective
# divergence misses.

# %%
F = U * omega                                    # (H, W), F_x; F_y = 0
div = np.gradient(F, PIX, axis=1)                # ∂F_x/∂x, kg/m²/s       (H, W)


def box(x0, x1, half):
    return (XX >= x0) & (XX <= x1) & (np.abs(YY) <= half)


def area_sum(b):
    # Q_A = ∬_A ∇·F dA
    return float(div[b].sum() * PIX**2)


def boundary_flux(x0, x1, half):
    # ∮ F·n dl = ∫ F_x(x1, y) dy − ∫ F_x(x0, y) dy  (no F_y)
    j1, j0 = np.argmin(abs(xs - x1)), np.argmin(abs(xs - x0))
    rows = np.abs(ys[::-1]) <= half
    return float((F[rows, j1].sum() - F[rows, j0].sum()) * PIX)


rng = np.random.default_rng(0)
SIG_OM = 0.003  # kg/m² of retrieval noise per pixel, one overpass
noisy = F + U * SIG_OM * rng.normal(size=F.shape)
div_noisy = np.gradient(noisy, PIX, axis=1)
# with np.gradient's central difference, the area sum over columns a..b
# telescopes to (Δx/2) (F_{b+1} + F_b − F_a − F_{a−1}) per row: four edge
# columns weighted ½, so sd = U σ_Ω Δx √n_rows, and averaging N overpasses
# divides it by √N
n_rows = int((np.abs(ys) <= 600).sum())
sd_box = U * SIG_OM * PIX * np.sqrt(n_rows)
# check by Monte Carlo: the box sum's spread over 2,000 noise draws
mc_rng = np.random.default_rng(1)
wide = box(-100, 1500, 600)
mc = np.array([np.gradient(U * SIG_OM * mc_rng.normal(size=F.shape), PIX, axis=1)[wide].sum()
               * PIX**2 for _ in range(2000)])
stage4 = {
    "box_wide": area_sum(box(-100, 1500, 600)),
    "box_narrow": area_sum(box(-100, 1500, 60)),
    "gauss_gap": abs(area_sum(box(-100, 1500, 600)) - boundary_flux(-100, 1500, 600)),
    "box_wide_noisy": float(div_noisy[box(-100, 1500, 600)].sum() * PIX**2),
    "pixel_noise": float(div_noisy[~box(-100, 3000, 900)].std()),
    "sd_box": float(sd_box), "sd_box_mc": float(mc.std()),
    "n_overpasses_10pct": int(np.ceil((sd_box / (0.1 * Q)) ** 2)),
}
stage4

# %%
# a box wide enough to hold the plume recovers Q
assert abs(stage4["box_wide"] - Q) < 1e-5
# a box narrower than the plume loses what diffuses out of its sides
assert round(stage4["box_narrow"], 2) == 0.81
# the area sum and the boundary flux are the same number, to rounding
assert stage4["gauss_gap"] < 1e-6
# one overpass at 0.003 kg/m² of noise: the box sum has sd 1.39 kg/s, 70 % of
# Q; this draw is 1.51, within 1 sd; a 10 % sd needs 49 overpasses
assert round(stage4["sd_box"], 2) == 1.39
# the formula against 2,000 noise draws: the sample sd of a sample of 2,000
# has a relative standard error of 1/√(2·1999) ≈ 1.6 %, so 5 % is 3 errors
assert abs(stage4["sd_box_mc"] / stage4["sd_box"] - 1) < 0.05
assert abs(stage4["box_wide_noisy"] - Q) < stage4["sd_box"]
assert round(stage4["box_wide_noisy"], 2) == 1.51
assert stage4["n_overpasses_10pct"] == 49

# %% [markdown]
# What the narrow box misses is the diffusive boundary term the divergence
# method drops. The Briggs column Ω = Q / (√(2π) U σ_y) e^{−y²/2σ_y²} solves
# U ∂Ω/∂x = K_eff(x) ∂²Ω/∂y² with K_eff = (U/2) dσ_y²/dx, so on a box
# [0, x₁] × [−h, h] Gauss's theorem with both terms reads
# Q = U ∫_{−h}^{h} Ω(x₁, y) dy + 2 ∫₀^{x₁} K_eff Ω(x, h) h / σ_y² dx.

# %%
H_BOX, X1 = 60.0, 1500.0


def k_eff(x):
    # K_eff = (U/2) dσ_y²/dx, by central differences
    d = 1e-3 * x
    return U / 2 * (sigma_y(x + d) ** 2 - sigma_y(x - d) ** 2) / (2 * d)


# advective: U ∫_{−h}^{h} Ω(x₁, y) dy = Q erf(h / √2 σ_y(x₁))
adv_box = float(Q * erf(H_BOX / (np.sqrt(2) * sigma_y(X1))))
# diffusive: 2 ∫₀^{x₁} K_eff Ω(x, h) h / σ_y² dx, through both side edges
dif_box = 2 * integrate.quad(
    lambda x: k_eff(x) * column_analytic(x, H_BOX) * H_BOX / sigma_y(x) ** 2,
    1e-9, X1, limit=500, points=[1.0, 10.0, 100.0])[0]
stage4.update({"narrow_adv": adv_box, "narrow_dif": dif_box})

# %%
# the grid's 0.81 is the advective flux out of the downwind edge (0.82
# exactly; the grid samples the edge at 20 m); the other 1.18 kg/s leaves
# through the side edges by crosswind diffusion, which the method drops
assert round(stage4["narrow_adv"], 2) == 0.82 and round(stage4["narrow_dif"], 2) == 1.18
assert abs(stage4["narrow_adv"] - stage4["box_narrow"]) < 0.01
assert abs(stage4["narrow_adv"] + stage4["narrow_dif"] - Q) < 1e-6

# %% [markdown]
# ## Stage 5 · The integrated mass enhancement
#
# Inside a contour where Ω equals the threshold t, ∮ U Ω n_x dl = U t ∮ n_x dl
# = 0, so the advective boundary term of the column equation vanishes
# exactly, and the source leaves through crosswind diffusion across the
# contour: Q = −∮ K_y ∂Ω/∂y n_y dl. On the K-theory plume (K_y = 10 m²/s),
# that is checked on the contour. Then the IME, τ and the effective wind
# that each definition of L needs.

# %%
KY = 10.0
YYk = YY
sig_k = np.sqrt(2 * KY * np.maximum(XX, 1.0) / U)
omega_k = np.where(XX > 0, Q / (np.sqrt(2 * np.pi) * U * sig_k)
                   * np.exp(-0.5 * (YYk / sig_k) ** 2), 0.0)
dOmega_dy = np.where(XX > 0, -YYk / sig_k**2 * omega_k, 0.0)   # analytic ∂Ω/∂y
T_K = 0.004


def s_k(x):
    return np.sqrt(2 * KY * x / U)


def half_width(x):
    # the contour Ω = t: y = ±f(x), f² = 2σ² ln(Q / (√(2π) U σ t))
    arg = Q / (np.sqrt(2 * np.pi) * U * s_k(x) * T_K)
    return np.sqrt(2 * s_k(x) ** 2 * np.log(np.maximum(arg, 1.0)))


# the contour closes where f = 0 again: Q / (√(2π) U σ) = t
x_end = (Q / (np.sqrt(2 * np.pi) * U * T_K)) ** 2 * U / (2 * KY)
# upper and lower sides together: n_x dl = −2 f′ dx, so the advective flux is
# −2 U t [f(x_end) − f(0)] = 0; and −K_y Ω_y n_y dl = 2 K_y f t / σ² dx.
# Substituting x = x_end w² removes the 1/√x singularity at the source.
advective = float(-2 * U * T_K * (half_width(x_end) - half_width(1e-12)))
diffusive = float(integrate.quad(
    lambda w: 2 * KY * half_width(x_end * w * w) * T_K / s_k(x_end * w * w) ** 2
    * 2 * x_end * w, 0.0, 1.0, limit=500)[0])

# IME on the Briggs plume with geotoolz, and the U_eff each L needs
mask = omega > THRESH
ime_kg = float(omega[mask].sum() * PIX**2)
area = float(mask.sum() * PIX**2)
L = {"sqrt_area": float(np.sqrt(area)),
     **{m: plume_length(mask, transform, method=m)                       # today
        for m in ("max_axis", "convex_hull", "skeleton")}}
u_eff_needed = {k: Q * v / ime_kg for k, v in L.items()}     # U_eff = Q L / IME
ime_op = IMEEstimate(wind_speed=U, length_method="max_axis")           # today
res = ime_op(gt, GeoTensor(mask, transform=transform, crs="EPSG:32630"))
tau = ime_kg / Q
stage5 = {"advective": advective, "diffusive": diffusive, "x_end": float(x_end),
          "ime_kg": ime_kg, "tau_s": tau, "L": L, "u_eff_needed": u_eff_needed,
          "q_ime_raw_u": res["emission_rate_kg_s"]}
{k: (round(v, 4) if isinstance(v, float) else {a: round(b, 2) for a, b in v.items()})
 for k, v in stage5.items()}

# %%
# on a contour of constant Ω the advective term is zero and diffusion carries Q
assert abs(stage5["advective"]) < 1e-6  # f(0⁺) → 0
assert abs(stage5["diffusive"] - Q) < 1e-9
assert round(stage5["x_end"]) == 663  # where the 4 g/m² contour closes
assert round(stage5["ime_kg"]) == 796 and round(stage5["tau_s"]) == 398
assert {k: round(v) for k, v in stage5["L"].items()} == {
    "sqrt_area": 465, "max_axis": 1_740, "convex_hull": 1_740, "skeleton": 1_740}
# the same plume needs U_eff = 1.17 m/s with L = √A and 4.37 m/s with the
# chord: a calibration is tied to the length it was fitted with
assert {k: round(v, 2) for k, v in stage5["u_eff_needed"].items()} == {
    "sqrt_area": 1.17, "max_axis": 4.37, "convex_hull": 4.37, "skeleton": 4.37}
# geotoolz's IME with the chord and the raw wind U underestimates Q here
assert round(stage5["q_ime_raw_u"], 2) == 1.37

# %% [markdown]
# ## The figures

# %%
fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)
im = axs[0].imshow(omega * 1e3, extent=[xs[0], xs[-1], ys[0], ys[-1]],
                   cmap="magma", vmin=0, vmax=8, origin="upper")
axs[0].contour(XX, YY, omega, levels=[THRESH], colors="#38bdf8", linewidths=1)
for d in (400, 800, 1200, 1600, 2000):
    axs[0].axvline(d, color="#94a3b8", lw=0.6, ls=":")
axs[0].set_xlim(-200, 2800), axs[0].set_ylim(-500, 500)
axs[0].set_xlabel("downwind x (m)"), axs[0].set_ylabel("crosswind y (m)")
axs[0].set_title("column Ω, threshold contour, transects", fontsize=10)
fig.colorbar(im, ax=axs[0], label="Ω (g/m²)", shrink=0.85)
d = np.arange(1, 6) * 400
axs[1].plot(d, q_full, "o-", color="#0f766e", label="full transect")
axs[1].plot(d, q_det, "s-", color="#b45309", label="above 2 g/m²")
axs[1].axhline(Q, color="#64748b", ls="--", lw=1)
axs[1].set_xlabel("transect distance (m)"), axs[1].set_ylabel("Q estimate (kg/s)")
axs[1].set_title("cross-sectional flux", fontsize=10)
axs[1].legend(frameon=False, fontsize=9)
axs[1].spines[["top", "right"]].set_visible(False)
fig.savefig(FIG / "csf.svg", metadata={"Date": None})
plt.close(fig)

spec = importlib.util.spec_from_file_location(
    "overview_figures", root / "docs/images/overview/make_figures.py")
ov = importlib.util.module_from_spec(spec)
sys.modules["overview_figures"] = ov
spec.loader.exec_module(ov)
W = 760
rows = [("advection–diffusion", "∂c/∂t + u·∇c = ∇·(K∇c) + S", ov.STATE),
        ("column equation", "∂Ω/∂t + U ∂Ω/∂x = K_y ∂²Ω/∂y² + q", ov.STATE)]
methods = [("Gaussian plume", "steady, solve"), ("Gaussian puff", "Green's function"),
           ("cross-sectional flux", "∫ dy"), ("divergence", "∇_h · F"),
           ("IME", "∬ dA, then τ_res")]
svg = ov.Svg(W, 330, "Five emission-rate methods from one equation: the plume and"
             " the puff solve the 3D advection-diffusion equation; the cross-sectional"
             " flux, the divergence method and the IME integrate its column form over"
             " a transect, pointwise and over an area.")
# the 3D equation spans the page; its column form sits over the three column
# methods only, so the plume and puff arrows do not cross it
svg.box(24, 20, W - 48, 60, rows[0][0], [rows[0][1]], hue=rows[0][2], title_size=14,
        sub_size=12.5)
svg.box(320, 112, W - 24 - 320, 60, rows[1][0], [rows[1][1]], hue=rows[1][2],
        title_size=14, sub_size=12.5)
svg.arrow([(528, 80), (528, 110)], color=ov.LINE)
svg.text(538, 100, "∫ dz", size=12, color=ov.MUTED, anchor="start")
bw = (W - 48 - 4 * 12) / 5
for k, (t, s) in enumerate(methods):
    x = 24 + k * (bw + 12)
    svg.box(x, 250, bw, 60, t, [s], hue=ov.OBS if k > 1 else ov.ML, title_size=12.5,
            sub_size=11)
    src_y = 80 if k < 2 else 172
    svg.arrow([(x + bw / 2, src_y), (x + bw / 2, 248)], color=ov.LINE)
(FIG / "derivation.svg").write_text(svg.render(), encoding="utf-8")
sorted(p.name for p in FIG.iterdir())

# %%
assert sorted(p.name for p in FIG.iterdir()) == ["csf.svg", "derivation.svg"]
