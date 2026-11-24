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

Most diagnostics take arrays and a gridax grid and return an array of the same grid. The types below are for the results that carry more than one array, and for the two inputs that need structure, a control volume and a grouping. Results are frozen dataclasses, registered as JAX pytrees once JAX loads; the two inputs hold NumPy tables, built once on the host, and are static under JAX.

```python
# Shapes: T times; Ny × Nx cells; Fx, Fy faces along x and y; K wavenumber bins;
#         M modes; N samples; G groups of at most Mg members; E ensemble members

import dataclasses

import numpy as np
from jaxtyping import Array, Bool, Float, Int


# today (diagnostix), private: diagnostix._src.groups
@dataclasses.dataclass(frozen=True, eq=False)
class Groups:
    """Which samples belong to which group; built once, on the host."""

    # positions of each group's samples, padded with 0
    index: Int[np.ndarray, "group member"]
    # which entries of `index` are real
    member: Bool[np.ndarray, "group member"]
    # the group of each sample, −1 for none
    labels: Int[np.ndarray, "sample"]


# proposed (diagnostix#18)
@dataclasses.dataclass(frozen=True)
class VelocityGradient:
    """The covariant derivatives of a horizontal velocity, s⁻¹."""

    dudx: Float[Array, "*batch y x"]
    dudy: Float[Array, "*batch y x"]
    dvdx: Float[Array, "*batch y x"]
    dvdy: Float[Array, "*batch y x"]


# proposed (diagnostix#23)
@dataclasses.dataclass(frozen=True)
class ControlVolume:
    """A region of a collocated grid and the signed faces that bound it."""

    inside: Bool[np.ndarray, "y x"]
    # +1 where the face leaves the region going east, −1 enters, 0 not on it
    east_sign: Int[np.ndarray, "y fx"]
    # the same for the faces along y
    north_sign: Int[np.ndarray, "fy x"]


# proposed (diagnostix#24)
@dataclasses.dataclass(frozen=True)
class Budget:
    """d/dt ∫ q dA + ∮ q u·n dl = ∫ S dA, term by term, and what is left."""

    tendency: Float[Array, "time"]
    boundary_flux: Float[Array, "time"]
    source: Float[Array, "time"]
    residual: Float[Array, "time"]


# proposed (diagnostix#25)
@dataclasses.dataclass(frozen=True)
class Climatology:
    """The mean of each slot, on the axis the samples were on."""

    mean: Float[Array, "*batch"]
    count: Int[Array, "*batch"]
    # 366 or 12; static
    n_slots: int


# proposed (diagnostix#11)
@dataclasses.dataclass(frozen=True)
class Spectrum:
    """Power against wavenumber, isotropic or along one axis."""

    wavenumber: Float[Array, "k"]
    power: Float[Array, "*batch k"]


# proposed (diagnostix#11)
@dataclasses.dataclass(frozen=True)
class EOFs:
    """Area-weighted EOFs of an anomaly field."""

    modes: Float[Array, "mode *grid"]
    principal_components: Float[Array, "time mode"]
    explained_variance: Float[Array, "mode"]
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `Groups` | the samples of each group, padded | a label is −1 or in [0, G); anything else raises; compared by identity |
| `VelocityGradient` | four covariant derivatives | on a sphere they carry the ± tan φ terms |
| `ControlVolume` | the region and its signed boundary faces | built on the host from a boolean region; every face sign is −1, 0 or +1; a face is on the boundary exactly when one side is inside |
| `Budget` | the four terms over time | residual = tendency + boundary\_flux − source, in the same units |
| `Climatology` | the mean and the number of samples of each slot | the mean of an empty slot is NaN |
| `Spectrum` | wavenumbers and power | wavenumbers ascending and positive |
| `EOFs` | modes, principal components, variance fractions | modes orthonormal under the area weights; fractions sum to at most 1 |

The records of the scores (`PSDScore`, `SpreadSkill`, `Reliability`) are specified in [diagnostix#12](https://github.com/jejjohnson/diagnostix/issues/12).

Single-valued scores (RMSE, bias, correlation, CRPS) return plain arrays.

## Types to adhere to

| Input | Comes from | Form |
| --- | --- | --- |
| The grid | gridax | a collocated grid: `SphericalAGrid2D` or `CartesianAGrid2D` |
| Fields | a product, a model's output, xrops' arrays | arrays of any array-API library with trailing `(y, x)` axes and any leading axes; a function along one axis takes `axis=-1`, and assumes no position for time or the vertical |
| Weights | gridax cell areas, or the caller | broadcast against the values, non-negative, not necessarily normalised |
| Validity | the caller | boolean, True = use: gridax's `mask=` on a grid, `valid=` in statistics; xrops converts geoops' "True means drop" masks at its boundary |
| Time | the caller | float seconds or days since an epoch, plus an integer calendar day for climatologies, on a leap-year calendar (1–366, so 29 Feb is 60 and 1 Mar is 61 in every year); never `datetime` |
| Constants | geophysix | records passed as keywords with a default (g, ρ₀); f is an argument |

```python
# Shapes: T times; Ny × Nx cells

import gridax
from diagnostix import budgets

# today (gridax)
grid = gridax.SphericalAGrid2D.from_coords(lat=lat, lon=lon)
weights = gridax.area_weights(grid)  # → (Ny, Nx)
# proposed (diagnostix#23)
# a boolean region, made into signed boundary faces once, on the host
box = budgets.ControlVolume.from_region(region, grid)  # (Ny, Nx) → volume
```

NaN is not a mask: a diagnostic that sees NaN in a valid cell returns NaN there, so missing data must be marked in `mask` or `valid`. An undefined result is NaN through a double `where`, so that the gradients of the other cells stay finite.

## Algorithms

### Geostrophy, vorticity, strain and Okubo–Weiss

These are gridax's functions on every grid ([geostack#287](https://github.com/jejjohnson/geostack/issues/287)), with the collocated registrations of [gridax#565](https://github.com/jejjohnson/gridax/issues/565); the same derivatives are diagnostix's `VelocityGradient`, for what gridax does not name. From sea-surface height η (or geopotential) to geostrophic velocity, then the velocity-gradient invariants. On a lon–lat grid, λ is longitude, φ latitude, R the Earth's radius, and u, v the eastward and northward components.

```{math}
f = 2 \Omega \sin \varphi
```

```{math}
u_g = -\frac{g}{f R} \frac{\partial \eta}{\partial \varphi}
```

```{math}
v_g = \frac{g}{f R \cos\varphi} \frac{\partial \eta}{\partial \lambda}
```

```{math}
\zeta = \frac{1}{R \cos\varphi} \left( \frac{\partial v}{\partial \lambda} - \frac{\partial (u \cos\varphi)}{\partial \varphi} \right)
```

```{math}
s_n = \frac{1}{R \cos\varphi} \frac{\partial u}{\partial \lambda} - \frac{1}{R} \frac{\partial v}{\partial \varphi} - \frac{v \tan\varphi}{R}
```

```{math}
s_s = \frac{1}{R \cos\varphi} \frac{\partial v}{\partial \lambda} + \frac{1}{R} \frac{\partial u}{\partial \varphi} + \frac{u \tan\varphi}{R}
```

```{math}
W = s_n^2 + s_s^2 - \zeta^2
```

1. Take the derivatives with gridax: ζ is gridax's spherical curl, and s\_n, s\_s are the covariant derivatives of the velocity, so they carry the tan φ connection terms, which scalar metric factors alone miss. On a Cartesian (projected) grid those terms vanish and the formulas reduce to ζ = ∂v/∂x − ∂u/∂y, s\_n = ∂u/∂x − ∂v/∂y, s\_s = ∂v/∂x + ∂u/∂y; on a curvilinear grid, use the grid's own metric terms.
2. Centred differences on the A-grid inside the domain, one-sided at the edges.
3. Mask geostrophy where |φ| is below a latitude cut (default 5°), where f → 0.
4. Eddy cores are W < −W₀ with W₀ = 0.2 σ\_W, the standard deviation of W over valid cells.

Cost: O(N) per field, one pass of stencils. Pitfalls: a stencil that reaches across a coast mixes land fill into the gradient (the mask is the fix); the Cartesian formulas with only the scalar metric factors give solid-body rotation u = ωR cos φ a vorticity of ω sin φ, half the true 2ω sin φ, and a spurious shear strain, which then corrupts W; W₀ depends on the domain, so it is a parameter, not a constant.

### Boundary-layer height by the bulk Richardson number

```{math}
\mathrm{Ri}_b(z) = \frac{g}{\theta_{v,s}} \frac{(\theta_v(z) - \theta_{v,s}) (z - z_s)}{(u(z) - u_s)^2 + (v(z) - v_s)^2 + b u_*^2}
```

1. Virtual potential temperature on each level from T, p and specific humidity (geophysix).
2. Ri\_b on every level above the surface reference.
3. The PBL height is the lowest z where Ri\_b crosses the critical value Ri\_c (default 0.25), linearly interpolated between the two levels that bracket the crossing.
4. No crossing in the column returns NaN, never the model top.

Cost: O(N\_z) per column, vectorised over columns. The first crossing is `argmax` of the boolean along z, then a check that it is a real crossing; no Python loop over levels. It equals a loop exactly on 1,000 random columns, on NumPy and JAX, with finite gradients. Pitfalls: a zero-shear column divides by zero without the u∗ term (b ≈ 100) or a floor; levels must be ordered surface first, which the caller does.

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
2. Face values are the mean of the product (q u) of the two neighbour cells, times the length of the face.
3. Q(t) is the integral of q with gridax cell areas; the tendency is a centred difference in time.
4. The residual is reported per time, with each term, so a budget is checked term by term.
5. The per-cell divergence is built from the same face fluxes, so its area sum over a region is the boundary flux to round-off.

Cost: O(N) per time. Pitfalls: xrops' `boundary_flux` sums φ u A over every face of the field (its docstring says so), which is the net flux through a boundary only when the field covers exactly the region; diagnostix sums over the region's own boundary. A centred divergence times the cell area differs from the box flux by 9 × 10⁻⁵ of the summed divergence on a 1° sphere (on a Cartesian grid both agree to 10⁻¹⁶), so gridax's `divergence` is not the per-cell term of a budget. Snapshot fields for the tendency and time-mean fluxes for F is correct; the other way round is not. The face interpolation must follow the grid convention of the model that made the fields, or the residual is mostly interpolation error.

### Climatology and anomalies

```{math}
\bar{x}(d) = \frac{1}{|Y_d|} \sum_{y \in Y_d} x(y, d)
```

```{math}
\tilde{x}(d) = a_0 + \sum_{k=1}^{K} a_k \cos\left(\frac{2 \pi k d}{366}\right) + b_k \sin\left(\frac{2 \pi k d}{366}\right)
```

Here d is the calendar day of a sample's (month, day) on a leap-year calendar, not its raw day of year: in a non-leap year 1 March is day 60 of the year but d = 61, the same slot as 1 March of a leap year, and d = 60 holds only 29 February.

1. Map each sample's (month, day) to its calendar day d, on the host or with a 12-entry table of month offsets, then group with `Groups` (below), counting only valid samples. The label is zero-based, d − 1 ∈ [0, 366) (or month − 1 ∈ [0, 12)); a label outside that range raises, where `segment_sum` would drop it silently. Grouping by the raw day of year would put 1 March of non-leap years with 29 February, and shift every date after it by one between leap and non-leap years.
2. The smoothed climatology fits the first K harmonics (default 3) to the daily means by weighted least squares, the weights being the counts.
3. The anomaly subtracts the climatology at each sample's day, looked up at the same label d − 1; adding it back is the inverse.

Cost: O(N) to group, O(K²) per pixel to fit. Pitfalls: 29 February is its own segment with a quarter of the samples, so smoothing matters most there; a climatology fitted on the evaluation period leaks into the anomaly scores, so the base period is an argument.

### Isotropic spectra and the PSD score

```{math}
\hat{\phi}(\mathbf{k}) = \sum_{m,n} w_{mn} \phi_{mn} e^{-2 \pi i (k_x x_m + k_y y_n)}
```

```{math}
P_2(\mathbf{k}) = \frac{\Delta x \Delta y}{N_x N_y U} |\hat{\phi}(\mathbf{k})|^2
```

```{math}
P(k) = \frac{\Delta k_x \Delta k_y}{\Delta k} \sum_{k - \Delta k / 2 \le |\mathbf{k}| < k + \Delta k / 2} P_2(\mathbf{k})
```

Here Δk\_x = 1 / (N\_x Δx) and Δk\_y = 1 / (N\_y Δy) are the spacings of the discrete wavenumbers, so P₂ is a 2-D density (Σ P₂ Δk\_x Δk\_y is the variance) and P is a 1-D density in |k| (Σ P(k) Δk is the same variance).

```{math}
S(k) = 1 - \frac{P_{\hat{y} - y}(k)}{P_{y}(k)}
```

1. Remove the mean (or a plane), multiply by a 2-D Hann taper w, and divide by its energy U = mean(w²) so the taper does not remove variance.
2. The 2-D FFT (`xp.fft`), then |φ̂|² with the Δx Δy / (N\_x N\_y U) normalisation, giving the 2-D density P₂.
3. Bin |k| into K annuli of width Δk = min(Δk\_x, Δk\_y), centred on the multiples of Δk, with `Groups` (K static), and multiply each annulus sum by Δk\_x Δk\_y / Δk, the area of one wavenumber cell over the bin width. Without that factor P(k) is bin-integrated power, not a density, and its amplitude changes with the grid size and spacing.
4. The PSD score S(k) uses the error's spectrum over the reference's; the resolved wavelength is 1 / k∗ at the first k where S falls below the level (default 0.5), interpolated in log k.

Cost: O(N log N) per field. Pitfalls: the FFT needs a uniform Cartesian grid, so a lon–lat field is first put on a local tangent plane (gridax) or the domain kept small enough that cos φ barely changes; gaps cannot go through an FFT, so a field with invalid cells is filled (and the fraction reported) or refused; xrft and the numpy convention differ by the normalisation in step 2, so the oracle test fixes one; Parseval (Σ P(k) Δk = variance) is the check: on an 8 × 8 grid with unit spacing and Δk = 1/8, the factor is 1/8, and leaving it out integrates to eight times the taper-corrected mean square. Rings that start at the multiples of Δk (a `floor`) misfiled 56 modes of exactly integer radius over seven grid sizes, by round-off; centred rings misfile none of them.

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

Cost: O(T N min(T, N)) for the thin SVD; there is no randomized SVD, because gaussx needs JAX. Pitfalls: xrops' `eof` is sklearn PCA without weights, so polar cells count as much as tropical ones; the valid-cell selection must be static (made once on the host) to run under `jit`; mode signs are arbitrary until step 4, so comparisons across runs fail without it.

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

### Grouped sums

A climatology groups by day, an isotropic spectrum by ring, a rank histogram by rank, and the array API standard has no `bincount`, `segment_sum` or `nansum`. `Groups` gathers each group's samples into one padded row and sums the rows, with static shapes.

| Method, 512 × 512 values into 363 rings | Largest relative error, float64 | float32 | Time, CPU |
| --- | --- | --- | --- |
| sort, cumulative sum, difference | 8 × 10⁻¹² | 7 × 10⁻³ | not timed |
| padded gather, masked sum | 3 × 10⁻¹⁵ | 1 × 10⁻⁷ | 1.1 ms (JAX) |
| `jax.ops.segment_sum` | 0 | 2 × 10⁻⁶ | 0.6 ms |

The gather reads 589,512 values for 262,144 samples; a very uneven grouping costs memory. Measured on 24 Nov 2026 ([diagnostix#15](https://github.com/jejjohnson/diagnostix/issues/15)), built in [diagnostix#17](https://github.com/jejjohnson/diagnostix/issues/17).

## JAX compatibility

JAX is optional. These are the transforms of the one array-API implementation when its inputs are JAX arrays; nothing inside the library calls `jax.jit`.

| Routine | `jit` | `grad` | `vmap` | Notes |
| --- | --- | --- | --- | --- |
| Geostrophy, vorticity, strain, Okubo–Weiss | yes | yes | yes | stencils on fixed shapes; NaN where invalid, so `grad` is taken over valid cells |
| Bulk-Richardson PBL height | yes | yes, away from crossings | over columns | the crossing index is piecewise constant; gradients flow through the interpolation |
| `ControlVolume.from_region` | no | no | no | host-side construction from a boolean region, once |
| Budget terms and residual | yes | yes | over time | the control volume is static |
| Climatology, anomalies | yes | yes | over pixels | the grouping is static (366 or 12 slots) |
| Isotropic spectrum, PSD score | yes | yes | over batch | `xp.fft`; bin count static; the resolved wavelength is piecewise in k |
| Area-weighted EOFs | yes | through `xp.linalg.svd`, unstable at repeated singular values | over batch | valid-cell selection fixed on the host first |
| RMSE, bias, correlation, CRPS, spread–skill | yes | yes (CRPS through the sort) | yes | weights and masks are arrays, not Python conditions |
| Wavelet transforms (Morlet, Paul, derivative of Gaussian) | yes | yes | over batch | on `xp.fft`; the significance test is host-only (SciPy) |
| PyWavelets transforms, ICA, NMF, k-means | no | no | no | stay host-only in xrops (pywt, sklearn) |
| xrft spectra in xrops | no | no | no | an oracle for the spectra, not a code path |

## Testing

- **Closed forms first.** Eight xrops behaviours were wrong when its values were stored, so a closed form comes before them ([diagnostix#16](https://github.com/jejjohnson/diagnostix/issues/16)).
- **Stored xrops values**, written once from geotoolz `a8b2f17` with the commit in each file name: advection, Cartesian frontogenesis, the geostrophic residual, shear and curvature vorticity on a plane, N² on uniform levels, column integrals, boundary-layer height, the monthly climatology, power and rotary spectra, pointwise scores without gaps, the PSD score and the resolved scale. Each is compared with its formula. xrops is not a test dependency, because it will import diagnostix.
- **Library oracles**, in one CI job: MetPy, xrft, xskillscore, scikit-learn and functax's `eof_basis`.
- **Backends.** NumPy, JAX and `array_api_strict` give the same values and dtype; a function that calls gridax runs on NumPy and JAX only.
- **Closed forms.** A solid-body rotation has ζ = 2ω and zero strain on a plane, and on the sphere ζ = 2ω sin φ with zero strain for rotation about the polar axis (u = ωR cos φ), the case that catches missing tan φ terms; rotation about any other axis gives ζ = 2ω · r̂ and zero strain too; a pure strain field has ζ = 0 and W > 0; a Gaussian eddy has W < 0 at its core.
- **Conservation.** A tracer advected by a divergence-free field in a closed box has a budget residual at round-off; a box cut through the middle has nonzero boundary flux equal to the tendency.
- **Parseval.** The binned spectrum integrates to the field's variance (taper-corrected).
- **EOFs.** Weighted orthonormality of the modes; for a field made of two known patterns, the modes recover them up to sign; unweighted EOFs equal scikit-learn's PCA and functax's `eof_basis`.
- **Scores.** CRPS of a one-member ensemble equals MAE; the sorted and pairwise CRPS agree; SSR ≈ 1 within its sampling error for draws from the same distribution as the truth (bound from the number of samples, as the stack's random-draw rule asks).
- **Transforms.** `jit`, `vmap` and `grad` on each row of the JAX table marked yes, and `finite` gradients on fields with masked cells.

## External dependencies

| Dependency | Why | Required |
| --- | --- | --- |
| NumPy, jaxtyping | the default namespace, host tables, shapes | yes |
| gridax | grids, derivatives, cell areas, the diagnostics it names | yes |
| geophysix | constants, facts at a point | yes |
| pipekit-array (`numpy`, `compat`), with pipekit-ops | the array backend and the import hook of the records | yes |
| JAX, xarray | the same functions on their arrays | optional extras |
| functax, MetPy, xrft, xskillscore, scikit-learn | reference results | test only |
| xrops | stored values; the xarray layer users call | never imported |
| integrix, gaussx | nothing: both need JAX | never imported |

pywt is not a dependency: the routines that need it stay in xrops.

## Where it stands

Read on 5 Nov 2026 at xrtoolz b3855d7, somax aea3a03 and gridax 82090a3 (0.1.0, under its former name). On 24 Nov 2026 the repository had its scaffold, test harness, grouped sums and masked reductions.

| Code | Where | What it means for diagnostix |
| --- | --- | --- |
| Kinematics: `coriolis_parameter`, `streamfunction`, `geostrophic_velocities`, `kinetic_energy`, `relative_vorticity`, `absolute_vorticity`, `divergence`, `enstrophy`, shear and tensor strain, `strain_magnitude`, `okubo_weiss`, `ageostrophic_velocities`, `advection`, `frontogenesis`, `eddy_kinetic_energy`, `brunt_vaisala_frequency`, `lapse_rate`, `mixed_layer_depth`, `potential_vorticity_barotropic`, `density_from_ts` | xrops `ocn/_src/kinematics.py`, on `xrops.grad` | ported, except what gridax names; the versions here take a gridax grid instead of coordinate names |
| `column_integral`, `hypsometric_height`, `pbl_height_bulk_richardson` (Ri\_c = 0.25) | xrops `atm/_src/vertical.py` | ported; the PBL default is kept |
| `wind_speed`, `wind_direction`, `wind_components` | xrops `atm/_src/wind.py` | facts at a point; geophysix's |
| `boundary_flux` and the heat, KE, salt and volume budget residuals, `control_volume_integral`, `budget_residual` | xrops budgets | `boundary_flux` sums over every face of the field, not a region's boundary; the true boundary flux is new |
| `calculate_climatology` (with smoothed and seasonal variants), `remove_climatology`, `add_climatology`, `calculate_anomaly`, `remove_mean` | xrops `anomaly/detrend.py` | ported with a base period and calendar-day segments (leap-year calendar) |
| `power_spectrum`, `isotropic_power_spectrum`, KE and enstrophy spectral flux, `integral_scale`, `fit_spectral_slope`, `compensated_spectrum`, `cross_spectrum`, `coherence`, `rotary_spectrum`, `stft` | xrops `transforms/fourier.py`, on xrft | the spectra are ported to `xp.fft`; xrft is an oracle |
| `cwt`, `dwt` | xrops `transforms/wavelet.py`, on pywt | stay in xrops |
| the Morlet, Paul and derivative-of-Gaussian transforms | xrops `wavelets/_src/`, on `scipy.fft` | ported to `xp.fft` |
| `pca`, `eof`, `ica`, `nmf`, `kmeans` | xrops `transforms/decompose.py`, sklearn estimators | `eof` is unweighted PCA; area-weighted EOFs are new; the rest stay |
| Pixel, probabilistic, distributional, spectral, physical, structural and instance metrics; `dm_test`, `rank_methods`, `segmented_psd` | xrops metrics | the pointwise and ensemble scores and `psd_score` / `resolved_scale` (level 0.5) are ported; the rest stay |
| `rms_divergence`, `total_enstrophy`, `kinetic_energy`, `geostrophic_imbalance`, `qg_balance_residual`, `compute_eval_metrics` | somax `eval/metrics.py` | a third copy of KE and geostrophic imbalance; somax imports diagnostix instead |
| `kinetic_energy`, vorticity, PV, strains, `okubo_weiss`, `enstrophy`, APE, `qg_potential_vorticity`; pipekit Operators | gridax `operators/diagnostics.py` | stay in gridax, which registers its collocated grids for them ([gridax#565](https://github.com/jejjohnson/gridax/issues/565)); diagnostix calls them and defines no second name |
