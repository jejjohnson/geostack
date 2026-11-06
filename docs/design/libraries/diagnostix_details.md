---
title: "diagnostix: design details"
short_title: "diagnostix details"
description: "The reference behind diagnostix: core types, the arrays callers pass in, the algorithms with their pitfalls, JAX notes, tests, dependencies, and a survey of the code diagnostix takes over, read on 5 Nov 2026."
date: 2026-11-05
keywords:
  - diagnostix
  - design
  - details
  - GeoStack
---

The reference behind diagnostix: core types, the arrays callers pass in, the algorithms with their pitfalls, JAX notes, tests, dependencies, and a survey of the code diagnostix takes over, read on 5 Nov 2026.

## Core types

Most diagnostics take arrays and a gridax geometry and return an array of the same grid. The types below are for the results that carry more than one array, and for the one input that needs structure, a control volume.

```python
# Shapes: T times; Ny × Nx cells; K wavenumber bins; M modes; E members

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int8


# proposed (diagnostix)
class ControlVolume(eqx.Module):
    """A region of a gridax grid and the signed faces that bound it."""

    inside: Bool[Array, "y x"]
    # +1 where the east face of cell (i, j) leaves the region, −1 enters, 0 not on it
    east_face_sign: Int8[Array, "y x"]
    # the same for the north face
    north_face_sign: Int8[Array, "y x"]


# proposed (diagnostix)
class Budget(eqx.Module):
    """d/dt ∫ q dV + ∮ q u·n dS = ∫ S dV, term by term, and what is left."""

    tendency: Float[Array, "time"]
    boundary_flux: Float[Array, "time"]
    source: Float[Array, "time"]
    residual: Float[Array, "time"]


# proposed (diagnostix)
class Spectrum(eqx.Module):
    """Power against wavenumber, isotropic or along one axis."""

    wavenumber: Float[Array, "k"]
    power: Float[Array, "*batch k"]
    kind: str = eqx.field(static=True)


# proposed (diagnostix)
class EOFs(eqx.Module):
    """Area-weighted EOFs of an anomaly field."""

    modes: Float[Array, "mode y x"]
    principal_components: Float[Array, "time mode"]
    explained_variance: Float[Array, "mode"]


# proposed (diagnostix)
class PSDScore(eqx.Module):
    """1 − PSD(error) / PSD(reference), and where it crosses a level."""

    wavenumber: Float[Array, "k"]
    score: Float[Array, "k"]
    resolved_wavelength: Float[Array, ""]


# proposed (diagnostix)
class SpreadSkill(eqx.Module):
    """Ensemble spread, RMSE of the mean, and their ratio."""

    spread: Float[Array, "*batch"]
    rmse: Float[Array, "*batch"]
    ratio: Float[Array, "*batch"]
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `ControlVolume` | the region and its signed boundary faces | built on the host from a boolean region; every face sign is −1, 0 or +1; a face is on the boundary exactly when one side is inside |
| `Budget` | the four terms over time | residual = tendency + boundary\_flux − source, in the same units |
| `Spectrum` | wavenumbers and power | wavenumbers ascending and positive; `kind` static |
| `EOFs` | modes, principal components, variance fractions | modes orthonormal under the area weights; fractions sum to at most 1 |
| `PSDScore` | the score per wavenumber and the resolved wavelength | score ≤ 1; the wavelength is NaN when the score never crosses the level |
| `SpreadSkill` | spread, skill and their ratio | spread includes the (E + 1) / E finite-ensemble factor |

Single-valued scores (RMSE, bias, correlation, CRPS) return plain arrays.

## Types to adhere to

| Input | Comes from | Form |
| --- | --- | --- |
| The grid | gridax | `LonLatGeometry`, `CurvilinearGeometry` or a Cartesian geometry, with metrics |
| Fields | a product, a model's output, xrops' arrays | arrays on the A-grid with trailing `(y, x)` (or `(z, y, x)`) axes and any leading batch axes |
| Weights | gridax cell areas, or the caller | `Float[Array, "y x"]`, non-negative, not necessarily normalised |
| Validity | the caller | `Bool[Array, "y x"]` named `valid`, True = use the cell; xrops converts geoops' "True means drop" masks at its boundary |
| Time | the caller | float seconds or days since an epoch, plus an integer day of year for climatologies; never `datetime` |
| Constants | geophysix | g, ρ₀, f from the grid's latitudes |

```python
# Shapes: T times; Ny × Nx cells

import diagnostix as dgx
import gridax as gdx

# proposed (gridax)
grid = gdx.cells.LonLatGeometry.from_axes(lon, lat)
# proposed (gridax)
weights = gdx.metrics.cell_area(grid)  # → (Ny, Nx)
# proposed (diagnostix)
# a boolean region, made into signed boundary faces once, on the host
box = dgx.budgets.ControlVolume.from_region(grid, region)  # (Ny, Nx) → volume
```

NaN is not a mask: a diagnostic that sees NaN in a valid cell returns NaN there, so missing data must be marked in `valid`.

## Algorithms

### Geostrophy, vorticity, strain and Okubo–Weiss

From sea-surface height η (or geopotential) to geostrophic velocity, then the velocity-gradient invariants.

```{math}
f = 2 \Omega \sin \varphi
```

```{math}
u_g = -\frac{g}{f} \frac{\partial \eta}{\partial y}
```

```{math}
v_g = \frac{g}{f} \frac{\partial \eta}{\partial x}
```

```{math}
\zeta = \frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}
```

```{math}
s_n = \frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}
```

```{math}
s_s = \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y}
```

```{math}
W = s_n^2 + s_s^2 - \zeta^2
```

1. Take the derivatives with gridax's metric operators: on a lon–lat grid, ∂/∂x = (R cos φ)⁻¹ ∂/∂λ and ∂/∂y = R⁻¹ ∂/∂φ; on a curvilinear grid, the grid's own metric terms.
2. Centred differences on the A-grid inside the domain, one-sided at the edges, and NaN wherever the stencil touches an invalid cell.
3. Mask geostrophy where |φ| is below a latitude cut (default 5°), where f → 0.
4. Eddy cores are W < −W₀ with W₀ = 0.2 σ\_W, the standard deviation of W over valid cells.

Cost: O(N) per field, one pass of stencils. Pitfalls: a stencil that reaches across a coast mixes land fill into the gradient (step 2 is the fix); spherical metric terms left out give wrong vorticity at high latitude; W₀ depends on the domain, so it is a parameter, not a constant.

### Boundary-layer height by the bulk Richardson number

```{math}
\mathrm{Ri}_b(z) = \frac{g}{\theta_{v,s}} \frac{(\theta_v(z) - \theta_{v,s}) (z - z_s)}{(u(z) - u_s)^2 + (v(z) - v_s)^2 + b u_*^2}
```

1. Virtual potential temperature on each level from T, p and specific humidity (geophysix).
2. Ri\_b on every level above the surface reference.
3. The PBL height is the lowest z where Ri\_b crosses the critical value Ri\_c (default 0.25), linearly interpolated between the two levels that bracket the crossing.
4. No crossing in the column returns NaN, never the model top.

Cost: O(N\_z) per column, vectorised over columns. JAX: find the first crossing with `jnp.argmax` on the boolean along z, then check it is a real crossing; no Python loop over levels. Pitfalls: a zero-shear column divides by zero without the u∗ term (b ≈ 100) or a floor; levels must be ordered upward, so the function checks that z is monotone.

### Budgets with a true boundary flux

```{math}
\frac{\mathrm{d}}{\mathrm{d}t} \int_V q \mathrm{d}V + \oint_{\partial V} q \mathbf{u} \cdot \mathbf{n} \mathrm{d}S = \int_V S \mathrm{d}V
```

```{math}
F(t) = \sum_{i,j} \sigma^{E}_{ij} (q u)^{E}_{ij} A^{E}_{ij} + \sigma^{N}_{ij} (q v)^{N}_{ij} A^{N}_{ij}
```

```{math}
r(t) = \frac{Q(t + \Delta t) - Q(t - \Delta t)}{2 \Delta t} + F(t) - \int_V S \mathrm{d}V
```

1. `ControlVolume.from_region` marks every cell face with one side inside the region and one outside, and gives it σ = +1 (flux leaves) or −1 (flux enters).
2. Interpolate q and the normal velocity to faces (the mean of the two neighbour cells on the A-grid; gridax's face values on a C-grid).
3. Q(t) is the volume integral of q with gridax cell volumes; the tendency is a centred difference in time.
4. The residual is reported per time, with each term, so a budget is checked term by term.

Cost: O(N) per time. Pitfalls: xrops' `boundary_flux` sums φ u A over every face of the field (its docstring says so), which is the net flux through a boundary only when the field covers exactly the region; diagnostix sums over the region's own boundary. Snapshot fields for the tendency and time-mean fluxes for F is correct; the other way round is not. The face interpolation must follow the grid convention of the model that made the fields, or the residual is mostly interpolation error.

### Climatology and anomalies

```{math}
\bar{x}(d) = \frac{1}{|Y_d|} \sum_{y \in Y_d} x(y, d)
```

```{math}
\tilde{x}(d) = a_0 + \sum_{k=1}^{K} a_k \cos\left(\frac{2 \pi k d}{365.25}\right) + b_k \sin\left(\frac{2 \pi k d}{365.25}\right)
```

1. Group by an integer day of year (or month) with `jax.ops.segment_sum` and a static number of segments (366 or 12), counting only valid samples.
2. The smoothed climatology fits the first K harmonics (default 3) to the daily means by weighted least squares, the weights being the counts.
3. The anomaly subtracts the climatology at each sample's day; adding it back is the inverse.

Cost: O(N) to group, O(K²) per pixel to fit. Pitfalls: 29 February is its own segment with a quarter of the samples, so smoothing matters most there; a climatology fitted on the evaluation period leaks into the anomaly scores, so the base period is an argument.

### Isotropic spectra and the PSD score

```{math}
\hat{\phi}(\mathbf{k}) = \sum_{m,n} w_{mn} \phi_{mn} e^{-2 \pi i (k_x x_m + k_y y_n)}
```

```{math}
P_2(\mathbf{k}) = \frac{\Delta x \Delta y}{N_x N_y U} |\hat{\phi}(\mathbf{k})|^2
```

```{math}
P(k) = \frac{\Delta k_x \Delta k_y}{\Delta k} \sum_{k \le |\mathbf{k}| < k + \Delta k} P_2(\mathbf{k})
```

Here Δk\_x = 1 / (N\_x Δx) and Δk\_y = 1 / (N\_y Δy) are the spacings of the discrete wavenumbers, so P₂ is a 2-D density (Σ P₂ Δk\_x Δk\_y is the variance) and P is a 1-D density in |k| (Σ P(k) Δk is the same variance).

```{math}
S(k) = 1 - \frac{P_{\hat{y} - y}(k)}{P_{y}(k)}
```

1. Remove the mean (or a plane), multiply by a 2-D Hann taper w, and divide by its energy U = mean(w²) so the taper does not remove variance.
2. `jnp.fft.fft2`, then |φ̂|² with the Δx Δy / (N\_x N\_y U) normalisation, giving the 2-D density P₂.
3. Bin |k| into K annuli of width Δk = min(Δk\_x, Δk\_y) with `segment_sum` (K static), and multiply each annulus sum by Δk\_x Δk\_y / Δk, the area of one wavenumber cell over the bin width. Without that factor P(k) is bin-integrated power, not a density, and its amplitude changes with the grid size and spacing.
4. The PSD score S(k) uses the error's spectrum over the reference's; the resolved wavelength is 1 / k∗ at the first k where S falls below the level (default 0.5), interpolated in log k.

Cost: O(N log N) per field. Pitfalls: the FFT needs a uniform Cartesian grid, so a lon–lat field is first put on a local tangent plane (gridax) or the domain kept small enough that cos φ barely changes; gaps cannot go through an FFT, so a field with invalid cells is filled (and the fraction reported) or refused; xrft and the numpy convention differ by the normalisation in step 2, so the oracle test fixes one; Parseval (Σ P(k) Δk = variance) is the check: on an 8 × 8 grid with unit spacing and Δk = 1/8, the factor is 1/8, and leaving it out integrates to eight times the taper-corrected mean square.

### Area-weighted EOFs

```{math}
X_w = X \operatorname{diag}(\sqrt{w})
```

```{math}
X_w = U \Sigma V^{\top}
```

```{math}
e_m = \operatorname{diag}(\sqrt{w})^{-1} V_{:,m}
```

```{math}
c_m(t) = U_{t,m} \sigma_m
```

```{math}
\lambda_m = \frac{\sigma_m^2}{\sum_{j} \sigma_j^2}
```

1. X is the anomaly matrix, time × valid cells, with the climatology removed (previous section).
2. The weights w are gridax cell areas (∝ cos φ on a regular lon–lat grid), normalised to sum to the number of cells.
3. Thin SVD of X\_w; keep M modes; modes are back-weighted so e\_mᵀ diag(w) e\_n = δ\_mn.
4. Fix the sign of each mode so its largest-magnitude loading is positive.

Cost: O(T N min(T, N)) for the thin SVD; a randomized SVD (gaussx) for large fields. Pitfalls: xrops' `eof` is sklearn PCA without weights, so polar cells count as much as tropical ones; the valid-cell selection must be static (made once on the host) to run under `jit`; mode signs are arbitrary until step 4, so comparisons across runs fail without it.

### Weighted errors, CRPS and spread–skill

```{math}
\mathrm{RMSE} = \sqrt{\frac{\sum_i w_i m_i (\hat{y}_i - y_i)^2}{\sum_i w_i m_i}}
```

```{math}
\mathrm{CRPS} = \frac{1}{E} \sum_{e} |x_e - y| - \frac{1}{2 E (E - 1)} \sum_{e, e'} |x_e - x_{e'}|
```

```{math}
\sum_{e, e'} |x_e - x_{e'}| = 2 \sum_{i=1}^{E} (2 i - E - 1) x_{(i)}
```

```{math}
\mathrm{SSR} = \sqrt{\frac{E + 1}{E}} \frac{\sqrt{\overline{\mathrm{Var}_e(x)}}}{\mathrm{RMSE}(\bar{x}, y)}
```

1. m is the validity mask (1 where valid) and w the cell areas; every score is a weighted mean over valid cells, then over time if asked.
2. CRPS uses the fair (E − 1) estimator by default; the pairwise term comes from the sorted members x₍ᵢ₎, O(E log E) instead of O(E²).
3. Spread–skill averages the ensemble variance before the square root, and includes the (E + 1)/E factor, so a reliable ensemble has SSR ≈ 1.

Cost: O(N) for pointwise scores, O(N E log E) for CRPS. Pitfalls: averaging RMSE per time step and then over time is not the RMSE of the whole period, so the reduction order is an argument; the biased CRPS penalises small ensembles; spread that is averaged after the square root is biased low.

## JAX compatibility

| Routine | `jit` | `grad` | `vmap` | Notes |
| --- | --- | --- | --- | --- |
| Geostrophy, vorticity, strain, Okubo–Weiss | yes | yes | yes | stencils on fixed shapes; NaN where invalid, so `grad` is taken over valid cells |
| Bulk-Richardson PBL height | yes | yes, away from crossings | over columns | the crossing index is piecewise constant; gradients flow through the interpolation |
| `ControlVolume.from_region` | no | no | no | host-side construction from a boolean region, once |
| Budget terms and residual | yes | yes | over time | the control volume is a pytree of static-shaped masks |
| Climatology, anomalies | yes | yes | over pixels | segment count static (366 or 12) |
| Isotropic spectrum, PSD score | yes | yes | over batch | `jnp.fft`; bin count static; the resolved wavelength is piecewise in k |
| Area-weighted EOFs | yes | through `jnp.linalg.svd`, unstable at repeated singular values | over batch | valid-cell selection fixed on the host first |
| RMSE, bias, correlation, CRPS, spread–skill | yes | yes (CRPS through the sort) | yes | weights and masks are arrays, not Python conditions |
| Wavelet transforms, ICA, NMF, k-means | no | no | no | stay host-only in xrops (pywt, sklearn) |
| xrft spectra in xrops | no | no | no | the oracle for the JAX spectra, not a code path |

## Testing

- **xrops as the oracle.** Every diagnostic that xrops already has runs on the same arrays through both, inside the tolerance of float64: geostrophic velocities, vorticity, strain, Okubo–Weiss, KE, enstrophy, PBL height, climatology and anomaly, spectra (with the normalisation fixed), RMSE, CRPS, spread–skill.
- **Closed forms.** A solid-body rotation has ζ = 2ω and zero strain; a pure strain field has ζ = 0 and W > 0; a Gaussian eddy has W < 0 at its core.
- **Conservation.** A tracer advected by a divergence-free field in a closed box has a budget residual at round-off; a box cut through the middle has nonzero boundary flux equal to the tendency.
- **Parseval.** The binned spectrum integrates to the field's variance (taper-corrected).
- **EOFs.** Weighted orthonormality of the modes; for a field made of two known patterns, the modes recover them up to sign; unweighted EOFs equal xrops' `eof`.
- **Scores.** CRPS of a one-member ensemble equals MAE; the sorted and pairwise CRPS agree; SSR ≈ 1 within its sampling error for draws from the same distribution as the truth (bound from the number of samples, as the stack's random-draw rule asks).
- **Transforms.** `jit`, `vmap` and `grad` on each row of the JAX table marked yes, and `finite` gradients on fields with masked cells.

## External dependencies

| Dependency | Why | Required |
| --- | --- | --- |
| jax, equinox, jaxtyping, einx | arrays, modules, shapes, contractions | yes |
| gridax | geometry, metrics, cell areas, face values | yes |
| geophysix | constants, virtual potential temperature, f and β | yes |
| gaussx | randomized SVD for large EOF problems | optional |
| xrops (xrtoolz) | the oracle in tests; the xarray layer users call | test only |
| finitevolX | nothing: C-grid diagnostics stay in finitevolX (decision 4) | never imported; finitevolX sits at GeoNumerics tier 3, above diagnostix |

pywt, xrft and sklearn are not dependencies: the routines that need them stay in xrops.

## Where it stands

Read on 5 Nov 2026 at xrtoolz b3855d7, somax aea3a03 and finitevolX 82090a3 (0.1.0).

| Code | Where | What it means for diagnostix |
| --- | --- | --- |
| Kinematics: `coriolis_parameter`, `streamfunction`, `geostrophic_velocities`, `kinetic_energy`, `relative_vorticity`, `absolute_vorticity`, `divergence`, `enstrophy`, shear and tensor strain, `strain_magnitude`, `okubo_weiss`, `ageostrophic_velocities`, `advection`, `frontogenesis`, `eddy_kinetic_energy`, `brunt_vaisala_frequency`, `lapse_rate`, `mixed_layer_depth`, `potential_vorticity_barotropic`, `density_from_ts` | xrops `ocn/_src/kinematics.py`, on `xrops.grad` | the oracle; the JAX versions take a gridax grid instead of coordinate names |
| `column_integral`, `hypsometric_height`, `pbl_height_bulk_richardson` (Ri\_c = 0.25) | xrops `atm/_src/vertical.py` | ported; the PBL default is kept |
| `wind_speed`, `wind_direction`, `wind_components` | xrops `atm/_src/wind.py` | trivial; stay in xrops or move with the rest |
| `boundary_flux` and the heat, KE, salt and volume budget residuals, `control_volume_integral`, `budget_residual` | xrops budgets | `boundary_flux` sums over every face of the field, not a region's boundary; the true boundary flux is new |
| `calculate_climatology` (with smoothed and seasonal variants), `remove_climatology`, `add_climatology`, `calculate_anomaly`, `remove_mean` | xrops `anomaly/detrend.py` | ported with a base period and day-of-year segments |
| `power_spectrum`, `isotropic_power_spectrum`, KE and enstrophy spectral flux, `integral_scale`, `fit_spectral_slope`, `compensated_spectrum`, `cross_spectrum`, `coherence`, `rotary_spectrum`, `stft` | xrops `transforms/fourier.py`, on xrft | the spectra are ported to `jnp.fft`; xrft stays the oracle |
| `cwt`, `dwt` | xrops `transforms/wavelet.py`, on pywt | stay host-only |
| `pca`, `eof`, `ica`, `nmf`, `kmeans` | xrops `transforms/decompose.py`, sklearn estimators | `eof` is unweighted PCA; area-weighted EOFs are new; the rest stay |
| Pixel, probabilistic, distributional, spectral, physical, structural and instance metrics; `dm_test`, `rank_methods`, `segmented_psd` | xrops metrics | the pointwise and ensemble scores and `psd_score` / `resolved_scale` (level 0.5) are ported; the rest stay |
| `rms_divergence`, `total_enstrophy`, `kinetic_energy`, `geostrophic_imbalance`, `qg_balance_residual`, `compute_eval_metrics` | somax `eval/metrics.py` | a third copy of KE and geostrophic imbalance; somax imports diagnostix instead |
| `kinetic_energy`, `relative_vorticity_cgrid`, PV, strains, `okubo_weiss`, `enstrophy`, APE, `qg_potential_vorticity`, `coriolis_param`, `beta_param`; `Energetics2D`, `Strain2D`, `QGPotentialVorticity2D`; pipekit Operators | finitevolX `operators/diagnostics.py`, `diagnostic_operators.py`, `pipekit/diagnostics.py` | already JAX on the C-grid; they stay in finitevolX, a tier above, so diagnostix never calls them. A C-grid field reaches diagnostix interpolated to cell centres (gridax), or is diagnosed in finitevolX |
