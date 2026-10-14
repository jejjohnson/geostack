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
# # radtrax · 3 · The package design, checked
#
# Regenerates every number in `docs/geomodels/radtrax/03_design.md`, the
# design and roadmap of radtrax (formerly rtmx), a differentiable radiative
# transfer package in JAX and Equinox. radtrax does not exist yet; this
# notebook runs the design's claims against the stack that does.
#
# 1. the source note's v0 sketch, as written, in Equinox 0.13 (stage 1);
# 2. a corrected v0: a layered Beer–Lambert model, checked against plumax's
#    single-slab forward model and its hand-written Jacobian (stage 2);
# 3. why layers: pressure broadening, layered against one slab (stage 3);
# 4. batching and Jacobians: `filter_vmap` outside `jacrev` (stage 4);
# 5. the v4 gradient pitfalls, the float32 wavenumber grid, and the
#    stack's block-tridiagonal solver (stage 5).
#
# The spectrum is synthetic: Voigt lines drawn at random near 4300 cm⁻¹
# (2.33 μm) and tabulated on a (p, T) grid, as HAPI would. Pinned revisions:
# plumax 0.1.2 at `28f257e`, gaussx 0.6.5 at `7fa1a38` (from references/),
# equinox 0.13.8, lineax 0.1.1, optimistix 0.1.0, optax 0.2.8.

# %%
from __future__ import annotations

import warnings

import equinox as eqx
import gaussx as gx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import optimistix as optx
import xarray as xr
from jaxtyping import Array, Float
from plumax.radtran.config import number_density_cm3
from plumax.radtran.forward import forward_nonlinear
from scipy.special import voigt_profile

jax.config.update("jax_enable_x64", True)
SEED = 0
rng = np.random.default_rng(SEED)

# %% [markdown]
# ## Stage 1 · The source note's v0 sketch, as written
#
# Three constructs of the sketch, each tried on its own: a JAX array as a
# `static=True` field, a concrete module whose field is `eqx.AbstractVar`,
# and where L-BFGS, fixed-point and Levenberg–Marquardt solvers live.

# %%
def tries(fn):
    # (raised?, warned?, message) for one construct
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            fn()
            return {"raised": False, "warned": bool(w),
                    "message": str(w[0].message)[:90] if w else ""}
        except Exception as e:  # noqa: BLE001
            return {"raised": True, "warned": bool(w), "message": f"{type(e).__name__}: {e}"[:90]}


class StaticGrid(eqx.Module):
    nu: Float[Array, " n"] = eqx.field(static=True)


class AbstractOpacity(eqx.Module):
    pass


class Model(eqx.Module):
    opacity: eqx.AbstractVar[AbstractOpacity]


stage1 = {
    "static_jax_array": tries(lambda: StaticGrid(jnp.linspace(0.0, 1.0, 5))),
    "abstractvar_unset": tries(lambda: Model()),
    "abstractvar_set": tries(lambda: Model(opacity=AbstractOpacity())),
    "optax_has_lbfgs": hasattr(optax, "lbfgs"),
    "optimistix_fixed_point": hasattr(optx, "FixedPointIteration"),
    "optimistix_lm": hasattr(optx, "LevenbergMarquardt"),
}
stage1

# %%
# a JAX array in a static field is flagged by Equinox
assert stage1["static_jax_array"]["warned"] or stage1["static_jax_array"]["raised"]
# an AbstractVar on a concrete class is just a required field: the class
# builds when it is passed and fails like any missing argument when it is not
assert stage1["abstractvar_set"] == {"raised": False, "warned": False, "message": ""}
assert stage1["abstractvar_unset"]["message"].startswith("TypeError")
# L-BFGS is in optax itself; fixed points and LM are in optimistix
assert stage1["optax_has_lbfgs"] and stage1["optimistix_fixed_point"]
assert stage1["optimistix_lm"]

# %% [markdown]
# ## Stage 2 · A corrected v0, against plumax
#
# The synthetic table σ(ν; p, T): 150 Voigt lines between 4250 and
# 4350 cm⁻¹, Lorentz half-width γ_air (p / 1 atm) (296 / T)^0.75, Doppler
# width at the line's temperature. A layered Beer–Lambert model in
# Equinox, every array a leaf, with an (ln p, T) bilinear table lookup.

# %%
# cm⁻¹; lines sit in 4252–4348, the grid runs ±20 cm⁻¹ wider so the band
# responses' tails (stage 2) are not cut off                           (G,)
NU = np.arange(4230.0, 4370.0, 0.01)
P_GRID = np.geomspace(0.03, 1.1, 10)  # atm
T_GRID = np.linspace(200.0, 310.0, 6)  # K
N_LINES = 150
line_nu = rng.uniform(4252.0, 4348.0, N_LINES)
line_s = rng.lognormal(0.0, 1.2, N_LINES) * 2e-22  # cm/molecule at 296 K
line_e = rng.uniform(0.0, 600.0, N_LINES)  # lower-state energy, cm⁻¹
GAMMA_AIR, N_AIR, C2 = 0.06, 0.75, 1.4387769  # cm⁻¹/atm, -, cm K
M_CH4 = 16.04


def sigma_table():
    # σ(ν; p, T) = Σ_i S_i(T) V(ν − ν_i; σ_D, γ_L)      → (n_p, n_T, G)
    out = np.zeros((P_GRID.size, T_GRID.size, NU.size))
    for j, t in enumerate(T_GRID):
        s_t = line_s * (296.0 / t) ** 1.5 * np.exp(-C2 * line_e * (1 / t - 1 / 296.0))
        sig_d = line_nu * np.sqrt(1.380649e-23 * t / (M_CH4 * 1.66054e-27)) / 2.998e8
        for i, p in enumerate(P_GRID):
            gam = GAMMA_AIR * p * (296.0 / t) ** N_AIR
            for k in range(N_LINES):
                sl = slice(*np.searchsorted(NU, [line_nu[k] - 5, line_nu[k] + 5]))
                out[i, j, sl] += s_t[k] * voigt_profile(NU[sl] - line_nu[k],
                                                        sig_d[k], gam)
    return out


SIGMA = sigma_table()


class Atmosphere(eqx.Module):
    p_atm: Float[Array, " L"]
    t_k: Float[Array, " L"]
    dz_cm: Float[Array, " L"]
    vmr: Float[Array, " L"]


class TableOpacity(eqx.Module):
    log_p: Float[Array, " P"]
    t_grid: Float[Array, " T"]
    sigma: Float[Array, "P T G"]

    def __call__(self, atm: Atmosphere) -> Float[Array, "L G"]:
        # σ_l = bilinear in (ln p, T);  τ_l = σ_l n_l x_l Δz_l
        def one(p, t):
            fp = jnp.clip(jnp.interp(jnp.log(p), self.log_p, jnp.arange(self.log_p.size)),
                          0, self.log_p.size - 1 - 1e-12)
            ft = jnp.clip(jnp.interp(t, self.t_grid, jnp.arange(self.t_grid.size)),
                          0, self.t_grid.size - 1 - 1e-12)
            i, j = jnp.floor(fp).astype(int), jnp.floor(ft).astype(int)
            a, b = fp - i, ft - j
            s = self.sigma
            return ((1 - a) * (1 - b) * s[i, j] + a * (1 - b) * s[i + 1, j]
                    + (1 - a) * b * s[i, j + 1] + a * b * s[i + 1, j + 1])
        sig = jax.vmap(one)(atm.p_atm, atm.t_k)                         # (L, G)
        n = atm.p_atm * 101325.0 / (1.380649e-23 * atm.t_k) * 1e-6      # cm⁻³
        return sig * (n * atm.vmr * atm.dz_cm)[:, None]


class BeerLambert(eqx.Module):
    def __call__(self, tau, albedo, mu0, mu):
        # L = (E₀ μ₀ / π) A exp(−m Σ_l τ_l),  m = 1/μ₀ + 1/μ            (G,)
        m = 1.0 / mu0 + 1.0 / mu
        return mu0 / jnp.pi * albedo * jnp.exp(-m * tau.sum(0))


class Instrument(eqx.Module):
    srf: Float[Array, "B G"]
    nu: Float[Array, " G"]
    shift: Float[Array, ""]

    def __call__(self, radiance):
        # y_b = Σ_g f_bg L(ν_g + δ)                                      (B,)
        return self.srf @ jnp.interp(self.nu + self.shift, self.nu, radiance)


class ForwardModel(eqx.Module):
    opacity: TableOpacity
    solver: BeerLambert
    instrument: Instrument

    def __call__(self, atm, albedo, mu0, mu):
        return self.instrument(self.solver(self.opacity(atm), albedo, mu0, mu))


# EMIT-like Gaussian bands: 14 cm⁻¹ apart (7.6 nm here), 8.5 nm FWHM. A width
# in wavelength is Δν = ν² Δλ / 10⁷ in wavenumber: 15.7 cm⁻¹ at 4300 cm⁻¹
centres = np.arange(4257.0, 4345.0, 14.0)                             # (B,)
FWHM_NM = 8.5
fwhm_cm = centres**2 * FWHM_NM / 1e7                                  # (B,)
SRF = np.exp(-0.5 * ((NU[None] - centres[:, None]) / (fwhm_cm[:, None] / 2.355)) ** 2)
SRF = SRF / SRF.sum(1, keepdims=True)
fm = ForwardModel(TableOpacity(jnp.log(P_GRID), jnp.asarray(T_GRID), jnp.asarray(SIGMA)),
                  BeerLambert(), Instrument(jnp.asarray(SRF), jnp.asarray(NU), jnp.array(0.0)))

# one slab on table nodes, against plumax's forward_nonlinear
P0, T0, X0, DZ0 = float(P_GRID[-2]), float(T_GRID[4]), 1.9e-6, 8.0e5  # atm, K, -, cm
MU0, MU = float(np.cos(np.radians(35.0))), 1.0
ds = xr.Dataset({"absorption_cross_section": (("wavenumber", "temperature", "pressure"),
                                              SIGMA.transpose(2, 1, 0))},
                coords={"wavenumber": NU, "temperature": T_GRID, "pressure": P_GRID})
amf = 1 / MU0 + 1 / MU
ref = forward_nonlinear(ds, NU, T_K=T0, p_atm=P0, vmr=X0, path_length_cm=DZ0,
                        amf=amf, surface_reflectance=1.0, solar_irradiance=MU0)  # today
slab = Atmosphere(jnp.array([P0]), jnp.array([T0]), jnp.array([DZ0]), jnp.array([X0]))


def hires(atm):
    return fm.solver(fm.opacity(atm), 1.0, MU0, MU)


ours = hires(slab)
# ∂L/∂x for the slab by reverse mode, against plumax's hand-written one
jac = jax.jacrev(lambda x: hires(eqx.tree_at(lambda a: a.vmr, slab, x)))(slab.vmr)[:, 0]
stage2 = {
    "n_ref_cm3": number_density_cm3(P0, T0),
    "radiance_rel": float(np.max(np.abs(np.asarray(ours) / ref.radiance - 1))),
    "jacobian_rel": float(np.max(np.abs(np.asarray(jac) - ref.jacobian))
                          / np.max(np.abs(ref.jacobian))),
    "band_depth": float(1 - np.min(np.asarray(ours)) / (MU0 / np.pi)),
}
stage2

# %%
# the layered model reduces to plumax's slab, radiance and Jacobian alike
assert stage2["radiance_rel"] < 1e-12 and stage2["jacobian_rel"] < 1e-12
assert 0.3 < stage2["band_depth"] < 0.99  # line cores are strong, not black

# %% [markdown]
# ## Stage 3 · Why layers: the same column through one slab and through 20
#
# A 20-layer atmosphere (1 km layers, scale height 8 km, 6.5 K/km lapse to
# 217 K) holds 1.9 ppm. One slab with the same column, at the surface's
# (p, T) or at the column's pressure-weighted mean, broadens every line as
# if all methane sat there.

# %%
zc = np.arange(0.5, 20.0, 1.0)  # km, layer centres                     (20,)
p_l = 1.0 * np.exp(-zc / 8.0)
t_l = np.maximum(288.0 - 6.5 * zc, 217.0)
dz = np.full(20, 1e5)
layered = Atmosphere(jnp.asarray(p_l), jnp.asarray(t_l), jnp.asarray(dz),
                     jnp.full(20, X0))
n_l = p_l * 101325 / (1.380649e-23 * t_l) * 1e-6
col = float((n_l * X0 * dz).sum())  # molecules / cm²
w = n_l * dz / (n_l * dz).sum()


def slab_at(p, t):
    n = p * 101325 / (1.380649e-23 * t) * 1e-6
    return Atmosphere(jnp.array([p]), jnp.array([t]), jnp.array([col / (n * X0)]),
                      jnp.array([X0]))


def band_tau(atm):
    # τ_b = −ln(y_b / y_b(A, no gas))                                     (B,)
    y = fm(atm, 1.0, MU0, MU)
    return -np.log(np.asarray(y) / (MU0 / np.pi))


tau_lay = band_tau(layered)
tau_surf = band_tau(slab_at(float(p_l[0]), float(t_l[0])))
tau_mean = band_tau(slab_at(float((w * p_l).sum()), float((w * t_l).sum())))
stage3 = {
    "column_molec_cm2": col,
    "surface_slab_rel": float(np.max(np.abs(tau_surf / tau_lay - 1))),
    "mean_slab_rel": float(np.max(np.abs(tau_mean / tau_lay - 1))),
}
stage3

# %%
assert round(stage3["column_molec_cm2"] / 1e19, 1) == 4.1
# a slab at the surface broadens the lines, so a band's optical depth differs
# by up to 19 % from the layered one; at the mean (p, T), by up to 3 %
assert round(stage3["surface_slab_rel"], 3) == 0.186
assert round(stage3["mean_slab_rel"], 3) == 0.031

# %% [markdown]
# ## Stage 4 · Jacobians over a swath: `filter_vmap` outside `jacrev`
#
# The K matrix for a plume in the bottom layer, the albedo and the spectral
# shift, per pixel, for 256 pixels; against finite differences on one.

# %%
def y_of(theta, atm=layered):
    # θ = (ΔX in layer 0 [ppm], albedo, shift [cm⁻¹])                    (3,) → (B,)
    a = eqx.tree_at(lambda s: s.vmr, atm, atm.vmr.at[0].add(theta[0] * 1e-6))
    f = eqx.tree_at(lambda m: m.instrument.shift, fm, theta[2])
    return f(a, theta[1], MU0, MU)


K = jax.jit(jax.jacrev(y_of))
thetas = jnp.stack([jnp.linspace(0.0, 2.0, 256), jnp.linspace(0.1, 0.4, 256),
                    jnp.linspace(-0.02, 0.02, 256)], 1)                  # (256, 3)
K_all = eqx.filter_jit(eqx.filter_vmap(jax.jacrev(y_of)))(thetas)       # (256, B, 3)
th = np.array([1.0, 0.25, 0.0037])  # off the 0.01 cm⁻¹ grid nodes
h = np.array([1e-4, 1e-6, 1e-5])
K_fd = np.stack([(np.asarray(y_of(jnp.asarray(th + h[i] * np.eye(3)[i])))
                  - np.asarray(y_of(jnp.asarray(th - h[i] * np.eye(3)[i])))) / (2 * h[i])
                 for i in range(3)], 1)
stage4 = {
    "fd_rel": float(np.max(np.abs(np.asarray(K(jnp.asarray(th))) - K_fd)
                           / np.max(np.abs(K_fd), 0))),
    "vmap_vs_loop": float(max(np.max(np.abs(np.asarray(K_all[i]) - np.asarray(K(thetas[i]))))
                              for i in (0, 128, 255))),
    "K_shape": tuple(K_all.shape),
}
stage4

# %%
assert stage4["fd_rel"] < 1e-6
assert stage4["vmap_vs_loop"] < 1e-12
assert stage4["K_shape"] == (256, 7, 3)

# %% [markdown]
# ## Stage 5 · What v2 and v4 must handle
#
# (a) Single precision on a wavenumber grid near 4300 cm⁻¹, against the
# Doppler width. (b) The discrete-ordinates "streaming multiplier"
# (e^{−τ/μ₀} − e^{−τ/μ}) / (1/μ − 1/μ₀), and its gradient, at μ → μ₀.
# (c) Eigenvector gradients at a repeated eigenvalue. (d) The stack's
# block-tridiagonal solver, for the discrete-ordinates boundary-value problem.

# %%
spacing32 = float(np.spacing(np.float32(4300.0)))
doppler_hwhm = float(4300.0 * np.sqrt(2 * np.log(2) * 1.380649e-23 * 296.0
                                      / (M_CH4 * 1.66054e-27)) / 2.998e8)
TAU, MU0_S = 0.5, 0.6


def multiplier_naive(mu):
    return (jnp.exp(-TAU / MU0_S) - jnp.exp(-TAU / mu)) / (1 / mu - 1 / MU0_S)


def multiplier_safe(mu, eps=1e-3):
    # the limit at μ = μ₀ is τ e^{−τ/μ₀}; a second-order Taylor branch near it,
    # with the double-where so the unused branch never sees 0/0
    d = 1 / mu - 1 / MU0_S
    near = jnp.abs(d) < eps
    d_safe = jnp.where(near, 1.0, d)
    exact = (jnp.exp(-TAU / MU0_S) - jnp.exp(-TAU / mu)) / d_safe
    taylor = TAU * jnp.exp(-TAU / MU0_S) * (1 - TAU * d / 2 + (TAU * d) ** 2 / 6)
    return jnp.where(near, taylor, exact)


def first_eigvec_sum(a):
    return jnp.linalg.eigh(a)[1][:, 0].sum()


A_deg = jnp.diag(jnp.array([1.0, 1.0, 2.0]))
A_split = jnp.diag(jnp.array([1.0, 1.0 + 1e-3, 2.0]))

# a symmetric block-tridiagonal system, 30 blocks of 8, through gaussx
nb, d = 30, 8
Dk = rng.normal(size=(nb, d, d))
Dk = Dk @ Dk.transpose(0, 2, 1) + 8 * np.eye(d)
Ak = 0.3 * rng.normal(size=(nb - 1, d, d))
op = gx.BlockTriDiag(jnp.asarray(Dk), jnp.asarray(Ak))                 # today
dense = np.zeros((nb * d, nb * d))
for k in range(nb):
    dense[k * d:(k + 1) * d, k * d:(k + 1) * d] = Dk[k]
for k in range(nb - 1):
    dense[(k + 1) * d:(k + 2) * d, k * d:(k + 1) * d] = Ak[k]
    dense[k * d:(k + 1) * d, (k + 1) * d:(k + 2) * d] = Ak[k].T
b = rng.normal(size=nb * d)
x_gx = np.asarray(gx.solve(op, jnp.asarray(b)))                         # today

stage5 = {
    "spacing32_cm1": spacing32,
    "doppler_hwhm_cm1": doppler_hwhm,
    "spacing_over_doppler": spacing32 / doppler_hwhm,
    "naive_at_mu0": float(multiplier_naive(MU0_S)),
    "naive_grad_at_mu0": float(jax.grad(multiplier_naive)(MU0_S)),
    "safe_at_mu0": float(multiplier_safe(MU0_S)),
    "safe_limit": TAU * np.exp(-TAU / MU0_S),
    "safe_grad_at_mu0": float(jax.grad(multiplier_safe)(MU0_S)),
    "safe_grad_fd": float((multiplier_naive(MU0_S + 1e-4) - multiplier_naive(MU0_S - 1e-4))
                          / 2e-4),
    "eig_grad_finite_degenerate": bool(np.all(np.isfinite(
        np.asarray(jax.grad(first_eigvec_sum)(A_deg))))),
    "eig_grad_max_split": float(np.max(np.abs(np.asarray(jax.grad(first_eigvec_sum)(A_split))))),
    "blocktridiag_rel": float(np.max(np.abs(x_gx - np.linalg.solve(dense, b)))
                              / np.max(np.abs(x_gx))),
}
{k: (round(v, 6) if isinstance(v, float) else v) for k, v in stage5.items()}

# %%
# float32 resolves 4300 cm⁻¹ to 4.9e-4 cm⁻¹, 7 % of the Doppler half-width
assert round(stage5["spacing32_cm1"], 7) == 0.0004883
assert round(stage5["spacing_over_doppler"], 2) == 0.07
# the naive multiplier is 0/0 at μ = μ₀, value and gradient
assert np.isnan(stage5["naive_at_mu0"]) and np.isnan(stage5["naive_grad_at_mu0"])
# the Taylor branch gives the limit and the right slope
assert abs(stage5["safe_at_mu0"] - stage5["safe_limit"]) < 1e-12
assert abs(stage5["safe_grad_at_mu0"] / stage5["safe_grad_fd"] - 1) < 1e-4
# eigenvector gradients: not finite at a repeated eigenvalue, ~1/gap near one
assert not stage5["eig_grad_finite_degenerate"]
assert stage5["eig_grad_max_split"] > 100
# gaussx's block-tridiagonal solve agrees with a dense solve
assert stage5["blocktridiag_rel"] < 1e-10
