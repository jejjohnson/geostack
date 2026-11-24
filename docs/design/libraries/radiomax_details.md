---
title: "radiomax: design details"
short_title: "radiomax details"
description: "The reference behind radiomax: core types, the arrays callers pass in, the algorithms with their pitfalls, JAX notes, tests, dependencies, and a survey of the code radiomax takes over, read on 5 Nov 2026."
date: 2026-11-05
keywords:
  - radiomax
  - design
  - details
  - GeoStack
---

The reference behind radiomax: core types, the arrays callers pass in, the algorithms with their pitfalls, JAX notes, tests, dependencies, and a survey of the code radiomax takes over, read on 5 Nov 2026.

## Core types

Functions take the arrays they need; only coefficients that travel together are a record. A record is a frozen dataclass, registered as a JAX pytree only when JAX is importable, as geophysix's are ([radiomax#15](https://github.com/jejjohnson/radiomax/issues/15)). Angles are degrees unless `degrees=False`, and lengths are metres.

```python
# Shapes: B bands; K coefficients of an empirical formula

import dataclasses

from radiomax._typing import ArrayLike


# proposed (radiomax)
@dataclasses.dataclass(frozen=True, slots=True)
class PlanckLinear:
    """GOES-style band inverse: T = (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂."""

    fk1: ArrayLike
    fk2: ArrayLike
    bc1: ArrayLike
    bc2: ArrayLike


# proposed (radiomax)
@dataclasses.dataclass(frozen=True, slots=True)
class PlanckQuadratic:
    """Himawari-style: Planck at the central wavelength, then T = c₀ + c₁ Tₑ + c₂ Tₑ²."""

    wavelength: ArrayLike
    c0: ArrayLike
    c1: ArrayLike
    c2: ArrayLike


# proposed (radiomax)
@dataclasses.dataclass(frozen=True, slots=True)
class EmpiricalCoefficients:
    """Fitted coefficients of one formula for one sensor (split-window, BRDF, …)."""

    coefficients: ArrayLike
    formula: str
    sensor: str
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `PlanckLinear` | the four Planck coefficients GOES publishes per band; Landsat's K₁ and K₂ with b₁ = 0 and b₂ = 1 | fk₁, fk₂, b₂ > 0 |
| `PlanckQuadratic` | central wavelength and the quadratic correction Himawari publishes | wavelength > 0, in metres |
| `EmpiricalCoefficients` | a coefficient vector, the formula and sensor it belongs to | the length matches the formula; `formula` and `sensor` are static under JAX |

Results are arrays of the input's kind: radiance, reflectance and brightness temperature come back with the shape they went in, and a DataArray keeps its coordinates. A zenith past 90° (night side, off the limb) is kept, not clipped, and the functions that divide by its cosine mask it.

## Types to adhere to

radiomax reads no files. Readers and Operators load sensor metadata and pass arrays:

| Input | Comes from | Form |
| --- | --- | --- |
| Spectral responses | georeader's SRF tables, loaded by a geoproducts reader or a geoops Operator | `wavelength` (W,) and `responses` (B, W), not necessarily normalised |
| Solar spectrum | the caller, such as radtrax's TSIS-1 HSRS reader | (W,), on the same wavelengths as the responses, per nm or per cm⁻¹ |
| Calibration and Planck coefficients | the file's metadata, read by geoproducts | gain and offset arrays, `PlanckLinear`, `PlanckQuadratic` |
| Positions | the scene's lon/lat leaves, or gridax geometry | degrees, plus the satellite's position in Earth-centred metres from geophysix |
| Time | the acquisition time | `day` (UTC midnights since 2000-01-01) and `seconds` (of the UTC day), so that it traces and survives float32 |

```python
# Shapes: B bands

# user: a reader turns file metadata into coefficient records
coefficients = rmx.thermal.PlanckLinear(fk1, fk2, bc1, bc2)  # (B,) each
# proposed (radiomax)
# UTC midnights since 2000-01-01, and the second of the day, on the host
day, seconds = rmx.geometry.utc_day_seconds("2026-11-05T10:30:00")  # → (), ()
```

Units are SI at the boundary, with two named exceptions: angles in degrees, and wavenumbers in cm⁻¹ in arguments named `wavenumber_cm`. Himawari's radiances are per micrometre, so its reader converts (the factor of 10⁶ its brightness-temperature code applies today) before calling radiomax.

## Algorithms

### Solar position and Earth–Sun distance

The low-precision formulas of the Astronomical Almanac. Over 2000–2050 the zenith is within 0.0121° of SPA (20,000 random times and places, pvlib 0.16.1). With n the days since J2000.0, n = day + seconds / 86400 − 1/2:

```{math}
L = 280.460^\circ + 0.9856474^\circ n
```

```{math}
g = 357.528^\circ + 0.9856003^\circ n
```

1. Mean longitude L and mean anomaly g from n (above).
2. Ecliptic longitude λ☉ = L + 1.915° sin g + 0.020° sin 2g, obliquity ε = 23.439° − 4 × 10⁻⁷° n.
3. Declination δ = asin(sin ε sin λ☉) and right ascension α = atan2(cos ε sin λ☉, cos λ☉).
4. Hour angle ω = θ\_GMST + λ − α, with λ the pixel's longitude, and θ\_GMST reduced modulo 360° from `day` and `seconds` separately.
5. The zenith from atan2 of the horizontal and vertical components, where the vertical one is cos θₛ = sin φ sin δ + cos φ cos δ cos ω; the azimuth from atan2 of the east and north components.
6. Earth–Sun distance d = 1.00014 − 0.01671 cos g − 0.00014 cos 2g, in AU.

Cost: O(1) per pixel, about twenty transcendental calls. NREL's SPA reaches about 0.0003° at many times the cost and is not offered (design decision 6).

Pitfalls. A Julian date near 2.46 × 10⁶ has a float32 resolution of a quarter of a day, and days since J2000 one of 84 s, or 0.35° of Earth rotation, so time is two arrays (design decision 9); the split stays within 0.0124° of SPA in float32. Refraction is ignored, so angles within a degree of the horizon are off by up to half a degree. The distance is within 8.7 × 10⁻⁵ AU of SPA; geoops' and radtrax's, d = 1 − 0.01673 cos(0.0172 (t − 4)) on the day of year t, is off by up to 8.0 × 10⁻⁴ AU.

### Viewing angles from the satellite's position

With p the pixel and s the satellite in Earth-centred coordinates from geophysix, the line of sight is ℓ = s − p. At geodetic latitude φ and longitude λ the local frame is

```{math}
\hat u = (\cos\varphi \cos\lambda, \cos\varphi \sin\lambda, \sin\varphi)
```

```{math}
\hat e = (-\sin\lambda, \cos\lambda, 0), \hat n = \hat u \times \hat e
```

1. ℓ = s − p and its norm.
2. cos θᵥ = (ℓ · û) / |ℓ|.
3. view azimuth = atan2(ℓ · ê, ℓ · n̂).

Cost: O(1) per pixel. Pitfalls: the normal must be the geodetic one, not the geocentric one, or θᵥ is off by up to 0.19° at mid-latitudes; ℓ · û < 0 means the satellite is below the horizon (off the limb) and the pixel is masked.

### Band solar irradiance

```{math}
E_0 = \frac{\int E_\odot(\lambda) R(\lambda) d\lambda}{\int R(\lambda) d\lambda}
```

1. On the response's own grid, Σ w E R / Σ w R with the composite trapezoid weights w, or any rule's weights passed as `weights=` (a functax rule's fit).
2. On another grid, the response matrix A_bg = ∫ R_b φ_g dλ / ∫ R_b dλ, with φ_g the grid's hat functions and R_b linear between table wavelengths: both are linear on each interval of the union of the two grids, so the integral is exact ([radiomax#23](https://github.com/jejjohnson/radiomax/issues/23)).

Cost: O(B W). Pitfalls: sampling a narrow response on a coarse grid loses it, and summing over the table's wavelengths only leaves most points of a fine grid without weight (radtrax's projection did, [radtrax#322](https://github.com/jejjohnson/radtrax/issues/322)); a response with negative tails (some published SRFs) is clipped at zero first. Masks and NaN samples follow integrix's `weighted_mean`: both sums run over the kept samples.

### Reflectance

```{math}
\rho_{\mathrm{TOA}} = \frac{\pi L d^2}{E_0 \cos\theta_s}
```

Cost: O(B N). Pitfalls: cos θₛ → 0 at the terminator makes ρ blow up, so pixels past a threshold (85° by default) are masked rather than divided; geoops today evaluates θₛ once at the scene centre, which is wrong by more than a degree across a wide swath, and radiomax evaluates it per pixel.

### Planck's law and its band inverses

```{math}
B(\lambda, T) = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp(h c / (\lambda k_B T)) - 1}
```

```{math}
T_e(L) = \frac{h c / (\lambda k_B)}{\ln(2 h c^2 / (\lambda^5 L) + 1)}
```

The two published band forms:

radtrax's wavenumber form moves (design decision 10), B(ν̃, T) = c₁ ν̃³ e⁻ˣ / (1 − e⁻ˣ) with x = c₂ ν̃ / T, and its inverse T = c₂ ν̃ / ln(1 + c₁ ν̃³ / L); autodiff of this form equals radtrax's hand-written ∂B/∂T to 7.3 × 10⁻¹⁶ relative.

1. GOES (`PlanckLinear`): T = (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂.
2. Himawari (`PlanckQuadratic`): Tₑ from the central wavelength, then T = c₀ + c₁ Tₑ + c₂ Tₑ².
3. From a full response: radtrax's `band_brightness_temperature`, a Newton solve through an instrument, which stays in radtrax.

Cost: O(B N). Pitfalls: L ≤ 0 (noise in a cold, dark band) makes the logarithm undefined, so those pixels become NaN, as both geoproducts readers do today; radiance per micrometre against per metre changes the argument by 10⁶; dT/dL is finite for L > 0, so the inverse differentiates.

### Geometric air-mass factor

```{math}
m = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_v}
```

Cost: O(N). Pitfalls: the plane-parallel form overestimates the path past about 75°, as radtrax's docstring already warns; the spherical-shell form per layer is geometry too, and moves from radtrax, which has had it since radtrax#104.

### Parallax from geostationary orbit

A target at height H above a pixel at p is seen along the ray from the satellite s through the elevated point q = p + H û. Its apparent position is where that ray meets the Earth's surface:

```{math}
|s + t (q - s)|_{\mathcal{E}} = 1, t > 0
```

with |·|ₑ the ellipsoid's norm (x²/a² + y²/a² + z²/b²), a quadratic in t.

1. q from p, H and the geodetic normal.
2. Solve the quadratic for the near root t₁; no real root means off the limb.
3. The apparent point s + t₁ (q − s), back to lat/lon.
4. Resampling the field from the apparent to the true position stays in geoops' Operator.

Cost: O(N). Pitfalls: geoops solves this on a sphere whose radius defaults to the equatorial 6,378,137 m, which shifts high-latitude pixels; radiomax takes the ellipsoid from geophysix.

### Ross–Li BRDF kernels

With θₛ, θᵥ and the relative azimuth ϕ, the phase angle ξ satisfies cos ξ = cos θₛ cos θᵥ + sin θₛ sin θᵥ cos ϕ, and the volumetric (RossThick) kernel is

```{math}
K_{\mathrm{vol}} = \frac{(\pi/2 - \xi) \cos\xi + \sin\xi}{\cos\theta_s + \cos\theta_v} - \frac{\pi}{4}
```

The geometric kernel (LiSparse-Reciprocal) uses the crown shape ratios h/b = 2 and b/r = 1, as MODIS does. The reflectance is R = f\_iso + f\_vol K\_vol + f\_geo K\_geo, with f fitted per band.

Cost: O(N) per band. Pitfalls: the LiSparse overlap term takes an arccos of a quantity that leaves \[−1, 1\] by round-off near the hot spot, so it is clipped, and its angle comes from atan2, which keeps the gradient finite without radtrax's custom derivatives; a clip on cos ξ instead breaks the gradient at exact nadir; NBAR evaluates the same kernels at nadir view and a reference sun angle.

### Lee despeckling

The filters stay in geoops (decision 7).

With μ and σ² the local mean and variance over a w × w window, C\_u the speckle's coefficient of variation (about 1 / √L for L looks) and Cᵢ = σ / μ the local one:

```{math}
\hat x = \mu + w (x - \mu)
```

```{math}
w = \max\left( 0, 1 - \frac{C_u^2}{C_i^2} \right)
```

1. Box sums of x and x² by cumulative sums, so the cost does not grow with w.
2. μ, σ² and Cᵢ per pixel.
3. The weight and the update.

Cost: O(N) per image. Pitfalls: Cᵢ → 0 in flat regions, so the weight is computed as max(0, 1 − C\_u² μ² / σ²) with σ² floored; edges use reflected padding. geoops' default C\_u = 0.523 corresponds to about 3.7 looks. Refined Lee and Frost follow the same pattern with directional sub-windows and an exponential kernel.

## JAX compatibility

| Function | jit | vmap | grad | Notes |
| --- | --- | --- | --- | --- |
| `solar_zenith`, `earth_sun_distance` | yes | yes | yes, in lon, lat and seconds | time as `day` and `seconds` |
| `view_angles` | yes | yes | yes | masked off the limb with `jnp.where` |
| `band_average`, `band_irradiance` | yes | yes | yes, in the spectrum and the response | NaN samples and masks by a double `where` |
| `response_matrix` | no | no | no | host-side: it merges two grids |
| `counts_to_radiance`, `reflectance` | yes | yes | yes | masks rather than branches at the terminator |
| `brightness_temperature` | yes | yes | yes, for L > 0 | NaN for L ≤ 0 |
| `air_mass_factor` | yes | yes | yes |  |
| `apparent_position` | yes | yes | yes, in H | |
| `ross_thick`, `li_sparse_reciprocal` | yes | yes | yes, in the angles | finite at nadir and the hotspot |
| `utc_day_seconds` | no | no | no | host-side, from timestamps |

Static: formula and sensor names. Traced: every angle, radiance and coefficient. Every function keeps the input's dtype, and the host-side ones return float64 NumPy. The same code runs on every array-API library; `array_api_strict` in the tests refuses anything outside the standard.

## Testing

Every public forward function gets the four tests of the house style.

| Kind | radiomax's version |
| --- | --- |
| Oracles | `georeader.reflectance` for radiance to reflectance and back; pvlib's SPA for solar position (within 0.015° for the almanac formulas); integrix and functax for band averages, in the job with JAX; radtrax `fadf2a6`'s values, stored as a file; the GOES and Himawari readers' current brightness temperatures on the same radiances; closed forms such as m = 2 at nadir and a noon sun at the subsolar point |
| Properties | reflectance is linear in L; m ≥ 2; brightness temperature increases with L; Planck then its inverse returns T|
| Gradients | `jacrev` against central differences for brightness temperature, reflectance, the viewing angles and the Ross–Li kernels |
| Transforms and backends | eager, `jit` and `vmap` agree with a Python loop; NumPy, JAX, `array_api_strict` and DataArray agree |

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| NumPy, array-api-compat | arrays, and the namespace of an input | required |
| geophysix | h, c, k\_B, the ellipsoid, Earth-centred positions | required |
| JAX | `jit`, `vmap`, `grad` | optional extra |
| xarray | DataArray in, DataArray out | optional extra |
| georeader, pvlib, pyproj | reference values in tests | test only |
| integrix, functax | reference values in tests; both need JAX for what radiomax compares with | test only, in the job with every extra |

radiomax imports no SRF tables, no file readers and no pysolar: geoops' solar functions reach pysolar through georeader today, and that path goes.

## Where it stands

Read on 5 Nov 2026 at geotoolz `d2f53f4`, radtrax `ed04971` and plumax `28f257e`, and again on 23 Nov 2026 at geotoolz `3d05a9c`, radtrax `fadf2a6` and plumax `96449d2`.

**geoops `radiometry/_src/array.py`.** `dn_to_radiance`, `radiance_to_dn` and `dn_to_reflectance` (a linear decode with scale and offset per band); `bt_from_radiance`, the two-coefficient form T = K₂ / ln(K₁ / L + 1) used for Landsat, without GOES's b₁ and b₂; `dos1`, a dark-object subtraction by `np.nanpercentile`. All NumPy. They move, except `dos1`, which is a percentile over a scene; the display helpers in the same file (`min_max_normalize`, `percentile_clip`, `gamma_correct`) are not radiometry and stay in geoops.

**geoops `radiometry/_src/solar.py`.** `earth_sun_distance_correction_factor`, `compute_sza` and `observation_date_correction_factor` are thin wrappers over `georeader.reflectance`, which uses pysolar. They take a `datetime` and return one float per scene, evaluated at the scene centre. radiomax replaces them with per-pixel, traceable versions.

**geoops `radiometry/_src/operators.py`.** The Operators, including `RadianceToReflectance` over `georeader.reflectance.radiance_to_reflectance`, stay in geoops and call radiomax.

**geoops `geom/_src/array.py` and `operators.py`.** `bowtie_detector_positions` and `scan_pixel_growth` give the MODIS and VIIRS scan geometry; they move. `GeostationaryParallaxCorrect` does its ray intersection inside `_apply`, on NumPy only, on a sphere of 6,378,137 m; its maths moves and the Operator stays.

**geoops `restore/_src/array.py`.** `despeckle_lee` (window 7, C\_u = 0.523), `despeckle_refined_lee` (which calls the Lee filter) and `despeckle_frost` (damping 2); they stay.

**geoproducts.** `himawari/reader.py` `_brightness_temperature` applies Planck at the central wavelength and then the quadratic c₀ + c₁ T + c₂ T² from the file header; `goes/l1b.py` applies (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂. Both move as `PlanckQuadratic` and `PlanckLinear`; the readers keep reading the coefficients. In `_src/geostationary.py`, `on_earth` is navigation maths and moves; `scan_angle_transform`, `geos_crs` and `FixedGrid` build raster metadata and stay.

**radtrax.** `radiometry/planck.py` and `brightness.py` hold Planck's law and its inverses; `atmosphere/geometry.py` both air masses; `surface/ross_li.py` the kernels; `spectroscopy/solar.py` the Earth–Sun distance; `instrument/srf.py`, `instrument/spectral.py` and `geo/srf.py` the response matrices; `geo/` the relative azimuth in four readers and the radiance arithmetic of Landsat and Sentinel-2. What moves is listed in [radiomax#15](https://github.com/jejjohnson/radiomax/issues/15).

**plumax.** Nothing: its two air-mass copies are gone.
