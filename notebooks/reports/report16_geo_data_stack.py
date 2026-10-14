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
# # Report 16 · The geo data stack, run
#
# Regenerates every number in `docs/reports/16_geo_data_stack.md`: one
# synthetic methane scene goes through geocatalog, geopatcher, geotoolz and
# xrtoolz (stages 1 to 5), then the seams the realignment fixes are exercised
# as they stand today (stage 6). The last cell asserts the values the page
# shows, to the precision it shows them.
#
# Pinned revisions: pipekit and pipekit-array 0.0.2 at `ef3c17f`; geotoolz
# 0.10.2 with geocatalog 0.5.1 and geopatcher 0.10.2 at `088475e`; xrtoolz
# 0.0.5 with xrcore 0.0.2 at `308c7f2`. Every random draw comes from one
# generator seeded here.
#
# > Jupytext percent-format script; `pixi run notebooks-to-ipynb` makes a
# > notebook of it. It needs the pixi default environment (the upstream
# > libraries installed from `references/`). It writes one GeoTIFF to a
# > temporary directory and runs in a few seconds.

# %%
from __future__ import annotations

import importlib
import importlib.metadata as md
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pipekit as pk
import rasterio
import xarray as xr
import xrcore
from geocatalog import GeoSlice, build, patch
from geopatcher import SpatialPatcher, spatial
from geotoolz import matched_filter as mf
from geotoolz import plume
from geotoolz.patch_ops import ApplyToChips, GridSampler, MergePatches
from pipekit import Lambda, Sequential
from pipekit_array import MeanScalar
from rasterio.transform import from_origin
from xrtoolz.atm import WindSpeed

rng = np.random.default_rng(16)
# Stage 6's seams are checked as they stand at the pinned revisions ("today"),
# or against the realigned stack ("target", gate F5's scheduled job)
STATE = os.environ.get("REPORT16_STATE", "today")
assert STATE in ("today", "target")

# %% [markdown]
# ## The scene
#
# A 256 × 256 scene of 30 m pixels in UTM 13N, with c = 40 SWIR bands between
# 2100 and 2450 nm. The background spectra are Gaussian with mean μ and a
# rank-6-plus-diagonal covariance Σ. A methane plume with enhancement α(p) in
# ppm·m is added through the linear model x = x_bg + α t, where t is the
# unit-absorption target: the radiance change per ppm·m.

# %%
# Shapes: c = 40 bands, H = W = 256 pixels
c, H, W, res = 40, 256, 256, 30.0
x0, y0, crs = 500_000.0, 4_400_000.0, "EPSG:32613"
lam = np.linspace(2100.0, 2450.0, c)  # (c,) nm
# μ_b: a smooth radiance continuum
mu_true = 8.0 + 2.0 * np.exp(-(((lam - 2200.0) / 120.0) ** 2))  # (c,)
# Σ = A Aᵀ + 0.02² I
A = rng.normal(size=(c, 6)) * 0.15  # (c, 6)
Sigma_true = A @ A.T + np.diag(np.full(c, 0.02**2))  # (c, c)
L_chol = np.linalg.cholesky(Sigma_true)  # (c, c)
# κ_b: CH₄ absorption per ppm·m, two bands near 2300 and 2360 nm
kappa = 1.5e-5 * (
    np.exp(-(((lam - 2300.0) / 15.0) ** 2))
    + 0.6 * np.exp(-(((lam - 2360.0) / 20.0) ** 2))
)
# t_b = −μ_b κ_b
t = -mu_true * kappa  # (c,)
# x_bg = μ + L z,  z ~ 𝒩(0, I)
bg = mu_true[:, None, None] + np.einsum(
    "cd,dhw->chw", L_chol, rng.normal(size=(c, H, W))
)  # (c, H, W)

# α(p): a plume from a source at row 128, column 80, spreading east
ii, jj = np.mgrid[0:H, 0:W]
along = (jj - 80) * res  # (H, W) m
cross = (ii - 128) * res  # (H, W) m
downwind = np.clip(along, 0.0, None)
alpha_true = (
    1500.0
    * np.exp(
        -0.5 * (downwind / 900.0) ** 2
        - 0.5 * (cross / (150.0 + 0.25 * downwind)) ** 2
    )
    * (along >= -60.0)
)  # (H, W) ppm·m
# x = x_bg + α t
scene = (bg + alpha_true[None] * t[:, None, None]).astype("float32")  # (c, H, W)

tmp = Path(tempfile.mkdtemp())
path = tmp / "scene_20260715.tif"
with rasterio.open(
    path, "w", driver="GTiff", height=H, width=W, count=c, dtype="float32",
    crs=crs, transform=from_origin(x0, y0, res, res),
) as dst:
    dst.write(scene)

# σ_α = (tᵀ Σ⁻¹ t)^(−1/2): the matched filter's noise floor
sigma_theory = float(1.0 / np.sqrt(t @ np.linalg.solve(Sigma_true, t)))
round(sigma_theory, 1), round(float(alpha_true.max()), 1)

# %% [markdown]
# ## Stage 1 · Which data (geocatalog → geopatcher)

# %%
cat = build.build_raster_catalog([path], filename_regex=r"scene_(?P<date>\d{8})",
                                 date_format="%Y%m%d")
slice_ = GeoSlice(
    bounds=(x0, y0 - H * res, x0 + W * res, y0),
    interval=pd.Interval(pd.Timestamp("2026-07-15"),
                         pd.Timestamp("2026-07-15T23:59"), closed="both"),
    resolution=(res, res),
    crs=crs,
)
field = patch.field_for(cat, slice_)
n_rows, field_type, field_shape = len(cat), type(field).__name__, field.domain.shape
n_rows, field_type, field_shape

# %% [markdown]
# ## Stage 2 · Which piece, and the enhancement (geopatcher → geotoolz)
#
# 128 × 128 chips at stride 64 (half the chip, where the Hann window sums to
# one), a matched filter fitted per chip, merged by window-weighted
# overlap-add.

# %%
n, s = 128, 64
patcher = SpatialPatcher(
    geometry=spatial.geometry.Rectangular(size=(n, n)),
    sampler=spatial.sampler.RegularStride(step=(s, s)),
    window=spatial.window.Hann(),
    aggregation=spatial.aggregation.OverlapAdd(),
)
chips = GridSampler(patcher=patcher)(field)
K_axis = (W - n) // s + 1
n_chips = len(chips)

# α̂(x) = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t), μ and Σ fitted on each chip
pipe = Sequential([
    GridSampler(patcher=patcher),                                   # field → 9 chips
    ApplyToChips(operator=mf.MatchedFilter(target=t, fit_on_call=True)),
    MergePatches(aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain),
])
alpha = pipe(field)  # GeoTensor (H, W), ppm·m
a = np.squeeze(np.asarray(alpha))
bg_px = alpha_true < 1.0
sigma_hat = float(np.std(a[bg_px]))
peak_hat = float(a.max())  # the retrieved peak, against the true 1500
corr = float(np.corrcoef(a.ravel(), alpha_true.ravel())[0, 1])
same_grid = (alpha.crs.to_string() == crs) and tuple(alpha.transform)[:6] == tuple(
    from_origin(x0, y0, res, res)
)[:6]

# The identity operator through the same pipeline returns the scene unchanged
ident = Sequential([
    GridSampler(patcher=patcher),
    ApplyToChips(operator=Lambda(lambda chip: chip)),
    MergePatches(aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain),
])
recon_err = float(np.abs(np.asarray(ident(field)) - scene).max())
(n_chips, K_axis, round(sigma_hat, 1), round(sigma_hat / sigma_theory - 1, 3),
 round(corr, 2), same_grid, recon_err)

# %% [markdown]
# ## Stage 3 · Mass and mask (geotoolz)

# %%
# ΔΩ = 10⁻⁶ α̂ M_CH₄ / V_m, M_CH₄ = 0.01604 kg mol⁻¹, V_m = 0.024465 m³ mol⁻¹
k_mass = 1e-6 * 0.01604 / 0.024465
kg_m2 = plume.ColumnToMass(gas="CH4", units_in="ppm_m")(alpha)  # (H, W) kg m⁻²
kk = np.squeeze(np.asarray(kg_m2))
mass_factor_err = float(np.abs(kk - k_mass * a).max())
# σ̂ = 1.4826 · median |ΔΩ − median ΔΩ|, a robust noise estimate
sigma_kg = float(1.4826 * np.median(np.abs(kk - np.median(kk))))
mask = plume.PlumeMask(threshold=3.0 * sigma_kg, min_area_px=50)(kg_m2)
m = np.squeeze(np.asarray(mask)).astype(bool)
truth_3s = alpha_true > 3.0 * sigma_theory
n_mask, n_false = int(m.sum()), int((m & ~truth_3s).sum())
area = res * res  # m²
ime_true_all = float((alpha_true * k_mass * area).sum())
ime_true_mask = float((alpha_true * k_mass * area)[m].sum())
(round(sigma_kg / k_mass, 1), n_mask, n_false, round(ime_true_all, 1),
 round(ime_true_mask, 1))

# %% [markdown]
# ## Stage 4 · Wind (xrtoolz → xrcore)
#
# An ERA5-shaped 10 m wind field in memory: 24 hourly steps on a 5 × 5 grid,
# u₁₀ = 3 + 0.5 sin(2π h / 24) m s⁻¹ and v₁₀ = 4 m s⁻¹.

# %%
times = pd.date_range("2026-07-15", periods=24, freq="h")
lat, lon = np.linspace(39.0, 40.0, 5), np.linspace(-106.0, -105.0, 5)
hours = np.arange(24)
u10 = (3.0 + 0.5 * np.sin(2 * np.pi * hours / 24))[:, None, None] + np.zeros((24, 5, 5))
v10 = np.full((24, 5, 5), 4.0)
era5 = xr.Dataset(
    {"u10": (("time", "latitude", "longitude"), u10),
     "v10": (("time", "latitude", "longitude"), v10)},
    coords={"time": times, "latitude": lat, "longitude": lon},
)
# U₁₀ = √(u₁₀² + v₁₀²)
ws = WindSpeed(u="u10", v="v10", name="ws10")(era5)["ws10"]  # (24, 5, 5)
U10 = float(ws.interp(time=pd.Timestamp("2026-07-15T17:30"), latitude=39.5,
                      longitude=-105.5))
# U_eff = 0.33 U₁₀ + 0.45, the calibration of Varon et al. (2018)
U_eff = 0.33 * U10 + 0.45
round(U10, 3), round(U_eff, 3)

# %% [markdown]
# ## Stage 5 · Emission rate (geotoolz)

# %%
# IME = ∑_{p∈𝒫} ΔΩ_p A_p,  Q = U_eff IME / ℓ,  σ_Q = 0.5 Q
est = plume.IMEEstimate(wind_speed=U_eff)(kg_m2, mask)
ime_bias = est["ime_kg"] / ime_true_mask - 1.0
q_closed = U_eff * est["ime_kg"] / est["length_m"]

# The same mask with the true background: α̂ = (x − μ)ᵀ Σ⁻¹ t / (tᵀ Σ⁻¹ t)
# with μ and Σ given, not fitted on chips that contain the plume
pipe_true = Sequential([
    GridSampler(patcher=patcher),
    ApplyToChips(operator=mf.MatchedFilter(mean=mu_true, cov_op=Sigma_true, target=t)),
    MergePatches(aggregation=spatial.aggregation.OverlapAdd(), domain=field.domain),
])
kg_true_bg = plume.ColumnToMass(gas="CH4", units_in="ppm_m")(pipe_true(field))
est_true_bg = plume.IMEEstimate(wind_speed=U_eff)(kg_true_bg, mask)
ime_bias_true_bg = est_true_bg["ime_kg"] / ime_true_mask - 1.0
({k: round(float(v), 3) for k, v in est.items()}, round(ime_bias, 3),
 abs(q_closed - est["emission_rate_kg_s"]), round(ime_bias_true_bg, 3))

# %% [markdown]
# ## Stage 6 · The seams, as they stand

# %%
# (a) Two Signature classes
sizes = {"time": 24, "latitude": 5, "longitude": 5}
if STATE == "today":
    sig_x = xrcore.Signature(sizes, dtype="float64")
else:  # workstream A: one class, built from a mapping with from_sizes
    sig_x = xrcore.Signature.from_sizes(sizes, dtype="float64")
sig_p = pk.Signature(dims=(("time", 24), ("latitude", 5), ("longitude", 5)),
                     dtype="float64")
wind_mean = Sequential([WindSpeed(u="u10", v="v10", name="ws10"), MeanScalar(axis=0)])
same_class = xrcore.Signature is pk.Signature
try:
    summary_x = wind_mean.summary(sig_x)
    sig_error = ""
except ValueError as e:
    summary_x, sig_error = "", str(e)
summary_p = wind_mean.summary(sig_p)
# the composed pipeline's last output signature: the summary table's last cell
final_p = summary_p.rstrip().splitlines()[-1].split("  ")[-1].strip()
final_x = summary_x.rstrip().splitlines()[-1].split("  ")[-1].strip() if summary_x else ""

# (b) The training dataset's stencil import
try:
    importlib.import_module("geopatcher.time")
    time_import = ""
except ModuleNotFoundError as e:
    time_import = str(e)
try:
    if STATE == "today":
        from pipekit_train import XarrayWindowDataset  # noqa: F401
    else:  # workstream D2: the dataset lives with the stencils
        from geopatcher.integrations.train import XarrayWindowDataset  # noqa: F401
    xwd_error = ""
except ImportError as e:
    xwd_error = str(e)

# (c) What a stencil import loads
# D3's contract is "NumPy and nothing heavier": check every heavy dependency in
# the geo stack's tree, not only the two the raster code pulls in
HEAVY = ("georeader", "rasterio", "shapely", "pyproj", "geopandas", "xarray",
         "pandas", "scipy", "dask", "zarr")
probe = subprocess.run(
    [sys.executable, "-c",
     "import sys, geopatcher.temporal.stencils as _; "
     f"print(sorted(m for m in {HEAVY!r} if m in sys.modules))"],
    capture_output=True, text=True, check=True,
)
stencil_loads = probe.stdout.strip()

# (d) Training-window origins on the stage 4 cube (workstream D's example)
from geopatcher.temporal.stencils import TimeStencil, valid_origin_points  # noqa: E402

o_in = valid_origin_points(times.values, TimeStencil(start="-5h", stop="0h",
                                                     step="1h", closed="both"))
o_tg = valid_origin_points(times.values, TimeStencil(start="1h", stop="1h",
                                                     step="0h", closed="both"))
n_origins = len(np.intersect1d(o_in, o_tg))

# (e) What the base installs pull in
xr_base = [r for r in md.requires("xrtoolz") if "extra ==" not in r]
xr_reqs = sorted(r for r in xr_base if "reader" in r or "pipekit" in r)
gz_reqs = sorted(r for r in md.requires("geotoolz") if r.startswith("pipekit"))
(same_class, sig_error, final_p, time_import, xwd_error[:60], stencil_loads,
 n_origins, xr_reqs, gz_reqs)

# %%
assert (round(sigma_theory, 1), round(float(alpha_true.max()), 1)) == (103.9, 1500.0)
assert (n_rows, field_type, tuple(field_shape)) == (1, "RasterField", (40, 256, 256))
assert (n_chips, K_axis) == (9, 3) and same_grid and recon_err < 1e-5
assert round(sigma_hat, 1) == 107.2 and round(sigma_hat / sigma_theory - 1, 3) == 0.032
assert round(corr, 2) == 0.73 and round(peak_hat) == 1588
assert mass_factor_err < 1e-12
assert round(sigma_kg / k_mass, 1) == 111.2 and (n_mask, n_false) == (1246, 3)
assert (round(ime_true_all, 1), round(ime_true_mask, 1)) == (944.0, 584.6)
assert (round(U10, 3), round(U_eff, 3)) == (4.722, 2.008)
assert round(est["ime_kg"], 1) == 534.9 and round(est["length_m"], 1) == 1651.1
assert round(est["emission_rate_kg_s"], 3) == 0.651
assert round(est["emission_rate_uncertainty_kg_s"], 3) == 0.325
assert round(ime_bias, 3) == -0.085 and abs(q_closed - est["emission_rate_kg_s"]) < 1e-12
assert round(ime_bias_true_bg, 3) == 0.017  # the true μ, Σ on the same mask
# Stages 1 to 5 do not change with the realignment; stage 6's seams do.
# the composed pipeline ends on the reduced grid, whichever class built the input
assert final_p == "Signature((latitude=5, longitude=5) float64)" and n_origins == 18
if STATE == "today":
    assert not same_class and sig_error == "too many values to unpack (expected 2)"
    assert time_import == "No module named 'geopatcher.time'"
    assert xwd_error.startswith("XarrayWindowDataset requires `geopatcher`")
    assert stencil_loads == "['georeader', 'rasterio', 'shapely']"
    assert xr_reqs == ["pipekit>=0.0.1", "xrtoolz-reader>=0.0.2"]
    assert gz_reqs == ["pipekit>=0.0.2"]
else:  # gate F5: the realigned stack
    assert same_class and sig_error == "" and final_x == final_p  # A1
    assert xwd_error == ""                                      # D1, D2
    assert stencil_loads == "[]"                                # D3
    assert not any("reader" in r for r in xr_reqs)              # C1
    assert gz_reqs and all(r.startswith("pipekit-core") for r in gz_reqs)  # F1
