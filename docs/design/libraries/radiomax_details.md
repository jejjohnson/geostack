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

Every type is an Equinox module of arrays, so it traces, batches and differentiates. Sensor-specific coefficients are leaves; the sensor's name is static. Angles are in radians and lengths in metres, as everywhere below geophysix.

```python
# Shapes: B bands; Ny × Nx pixels; K coefficients of an empirical formula

import equinox as eqx
from jaxtyping import Array, Float


# proposed (radiomax)
class ViewGeometry(eqx.Module):
    """Solar and viewing angles per pixel, in radians."""

    solar_zenith: Float[Array, "y x"]
    solar_azimuth: Float[Array, "y x"]
    view_zenith: Float[Array, "y x"]
    view_azimuth: Float[Array, "y x"]


# proposed (radiomax)
class BandCalibration(eqx.Module):
    """Counts to radiance per band: L = g · DN + o."""

    gain: Float[Array, "band"]
    offset: Float[Array, "band"]


# proposed (radiomax)
class PlanckLinear(eqx.Module):
    """GOES-style band inverse: T = (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂."""

    fk1: Float[Array, "band"]
    fk2: Float[Array, "band"]
    bc1: Float[Array, "band"]
    bc2: Float[Array, "band"]


# proposed (radiomax)
class PlanckQuadratic(eqx.Module):
    """Himawari-style: Planck at the central wavelength, then T = c₀ + c₁ Tₑ + c₂ Tₑ²."""

    wavelength: Float[Array, "band"]
    c0: Float[Array, "band"]
    c1: Float[Array, "band"]
    c2: Float[Array, "band"]


# proposed (radiomax)
class EmpiricalCoefficients(eqx.Module):
    """Fitted coefficients of one formula for one sensor (split-window, BRDF, …)."""

    coefficients: Float[Array, "coef"]
    formula: str = eqx.field(static=True)
    sensor: str = eqx.field(static=True)
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `ViewGeometry` | four angle fields per pixel | angles in radians; a zenith past π/2 (night side, off the limb) is kept, not clipped, and the functions that divide by its cosine mask it |
| `BandCalibration` | gain and offset per band | gain > 0; one entry per band of the scene |
| `PlanckLinear` | the four Planck coefficients GOES publishes per band | fk₁, fk₂, b₂ > 0 |
| `PlanckQuadratic` | central wavelength and the quadratic correction Himawari publishes | wavelength > 0, in metres |
| `EmpiricalCoefficients` | a coefficient vector, the formula and sensor it belongs to | the length matches the formula; `formula` and `sensor` are static, so a change retraces |

Results are plain arrays: radiance, reflectance and brightness temperature come back with the shape they went in.

## Types to adhere to

radiomax reads no files. Readers and Operators load sensor metadata and pass arrays:

| Input | Comes from | Form |
| --- | --- | --- |
| Spectral responses | georeader's SRF tables, loaded by a geoproducts reader or a geoops Operator | `wavelength: Float[Array, "W"]` in metres, `responses: Float[Array, "band W"]`, not necessarily normalised |
| Solar spectrum | a reference spectrum shipped as data | `Float[Array, "W"]`, on the same wavelengths as the responses, W m⁻² m⁻¹ |
| Calibration and Planck coefficients | the file's metadata, read by geoproducts | `BandCalibration`, `PlanckLinear`, `PlanckQuadratic` |
| Positions | the scene's lat/lon leaves, or gridax geometry | radians, plus the satellite's position in Earth-centred metres from geophysix |
| Time | the acquisition time | a Julian date as a float64 array, never a `datetime`, so that it traces |

```python
# Shapes: B bands; W wavelengths; Ny × Nx pixels

# user: a reader turns file metadata into coefficient records
calibration = rmx.radiometry.BandCalibration(gain=gain, offset=offset)  # (B,), (B,)
# proposed (radiomax)
# JD from a UTC timestamp, on the host, once per scene
julian_date = rmx.geometry.julian_date("2026-11-05T10:30:00Z")  # → () float64
```

Units are SI at the boundary. Himawari's radiances are per micrometre, so its reader converts (the factor of 10⁶ its brightness-temperature code applies today) before calling radiomax.

## Algorithms

### Solar position and Earth–Sun distance

The low-precision formulas of the Astronomical Almanac, about 0.01° between 1950 and 2050. With JD the Julian date and n = JD − 2451545.0 the days since J2000:

```{math}
L = 280.460^\circ + 0.9856474^\circ n
```

```{math}
g = 357.528^\circ + 0.9856003^\circ n
```

1. Mean longitude L and mean anomaly g from n (above).
2. Ecliptic longitude λ☉ = L + 1.915° sin g + 0.020° sin 2g, obliquity ε = 23.439° − 4 × 10⁻⁷° n.
3. Declination δ = asin(sin ε sin λ☉) and right ascension α = atan2(cos ε sin λ☉, cos λ☉).
4. Hour angle ω = θ\_GMST(n) + λ − α, with λ the pixel's longitude.
5. cos θₛ = sin φ sin δ + cos φ cos δ cos ω; the azimuth from atan2 of the east and north components.
6. Earth–Sun distance d = 1.00014 − 0.01671 cos g − 0.00014 cos 2g, in AU.

Cost: O(1) per pixel, about twenty transcendental calls. NREL's SPA, behind an argument, reaches about 0.0003° at many times the cost (design decision 6).

Pitfalls. A Julian date near 2.46 × 10⁶ has a float32 resolution of about a quarter of a day, so n is formed in float64 before any cast. Refraction is ignored, so angles within a degree of the horizon are off by up to half a degree. geoops' distance today is d = 1 − 0.01673 cos(0.0172 (t − 4)) on the day of year t, which differs from the formula above by up to about 10⁻⁴ AU; tests state which one they compare to.

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

1. Bring the solar spectrum onto the response's wavelengths (a functax interpolation plan), or both onto a common grid.
2. Build an integrix rule on that grid (trapezoid by default).
3. Two rule applications per band and a division.

Cost: O(B W). Pitfalls: a response sampled more coarsely than the solar spectrum's lines can bias E₀ where the solar spectrum has strong lines, so the common grid is the finer of the two; a response with negative tails (some published SRFs) is clipped at zero first.

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

1. GOES (`PlanckLinear`): T = (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂.
2. Himawari (`PlanckQuadratic`): Tₑ from the central wavelength, then T = c₀ + c₁ Tₑ + c₂ Tₑ².
3. From a full response: invert the band-integrated Planck function by a few Newton steps on T, starting from the central-wavelength Tₑ.

Cost: O(B N), plus a fixed number of Newton steps in form 3. Pitfalls: L ≤ 0 (noise in a cold, dark band) makes the logarithm undefined, so those pixels become NaN, as both geoproducts readers do today; radiance per micrometre against per metre changes the argument by 10⁶; dT/dL is finite for L > 0, so the inverse differentiates.

### Geometric air-mass factor

```{math}
m = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_v}
```

Cost: O(N). Pitfalls: the plane-parallel form overestimates the path past about 75°, as radtrax's docstring already warns; a spherical (Chapman) correction would still be geometry, but radtrax's roadmap puts it with its v1 forward model, so the two must agree on where it lives before either writes it.

### Parallax from geostationary orbit

A target at height H above a pixel at p is seen along the ray from the satellite s through the elevated point q = p + H û. Its apparent position is where that ray meets the Earth's surface:

```{math}
|s + t (q - s)|_{\mathcal{E}} = 1, t > 0
```

with |·|ₑ the ellipsoid's norm (x²/a² + y²/a² + z²/b²), a quadratic in t.

1. q from p, H and the geodetic normal.
2. Solve the quadratic for the near root t₁; no real root means off the limb.
3. The apparent point s + t₁ (q − s), back to lat/lon.
4. Resample the field from the apparent to the true position (a functax plan).

Cost: O(N) plus the resampling. Pitfalls: geoops solves this on a sphere whose radius defaults to the equatorial 6,378,137 m, which shifts high-latitude pixels; radiomax takes the ellipsoid from geophysix.

### Ross–Li BRDF kernels

With θₛ, θᵥ and the relative azimuth ϕ, the phase angle ξ satisfies cos ξ = cos θₛ cos θᵥ + sin θₛ sin θᵥ cos ϕ, and the volumetric (RossThick) kernel is

```{math}
K_{\mathrm{vol}} = \frac{(\pi/2 - \xi) \cos\xi + \sin\xi}{\cos\theta_s + \cos\theta_v} - \frac{\pi}{4}
```

The geometric kernel (LiSparse-Reciprocal) uses the crown shape ratios h/b = 2 and b/r = 1, as MODIS does. The reflectance is R = f\_iso + f\_vol K\_vol + f\_geo K\_geo, with f fitted per band.

Cost: O(N) per band. Pitfalls: the LiSparse overlap term takes an arccos of a quantity that leaves \[−1, 1\] by round-off near the hot spot, so it is clipped; NBAR evaluates the same kernels at nadir view and a reference sun angle.

### Lee despeckling

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
| `solar_zenith`, `earth_sun_distance` | yes | yes | yes, in lat, lon and time | time as a float64 Julian date |
| `view_zenith`, `view_azimuth` | yes | yes | yes | masked off the limb with `jnp.where` |
| `band_irradiance` | yes | over bands | yes, in the response | the wavelength grid is static |
| `counts_to_radiance`, `reflectance` | yes | yes | yes | masks rather than branches at the terminator |
| `brightness_temperature` | yes | yes | yes, for L > 0 | NaN for L ≤ 0 |
| `air_mass_factor` | yes | yes | yes |  |
| `parallax` | yes | yes | yes, in H | the resampling plan is built outside `jit` |
| `ross_li` | yes | yes | yes, in the weights | arccos argument clipped |
| `despeckle_lee` | yes | over images | yes | window size static |
| `julian_date` | no | no | no | host-side, from a timestamp string |

Static: window sizes, formula and sensor names, quadrature grids. Traced: every angle, radiance and coefficient. Every function keeps the input's dtype, except that the Julian date and n are formed in float64.

## Testing

Every public forward function gets the four tests of the house style.

| Kind | radiomax's version |
| --- | --- |
| Oracles | `georeader.reflectance` for radiance to reflectance and back; pvlib's SPA for solar position (within 0.01° for the almanac formulas); the GOES and Himawari readers' current brightness temperatures on the same radiances; closed forms such as m = 2 at nadir and a noon sun at the subsolar point |
| Properties | reflectance is linear in L; m ≥ 2; brightness temperature increases with L; Planck then its inverse returns T; Lee with C\_u = 0 is the identity |
| Gradients | `jacrev` against central differences for brightness temperature, reflectance, the viewing angles and the Ross–Li kernels |
| Transforms | eager, `jit` and `vmap` agree with a Python loop over bands and scenes |

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| JAX, Equinox, jaxtyping | arrays, modules, types | required |
| geophysix | h, c, k\_B, the ellipsoid, Earth-centred positions | required |
| integrix | band integrals over a spectral response | required |
| functax | resampling for parallax; the solar spectrum onto a response's grid | optional; the geoops Operator can resample instead (not an edge on the layer map yet) |
| georeader, pvlib | reference values in tests | test only |

radiomax imports no SRF tables, no file readers and no pysolar: geoops' solar functions reach pysolar through georeader today, and that path goes.

## Where it stands

Read on 5 Nov 2026 at geotoolz `d2f53f4`, radtrax `ed04971` and plumax `28f257e`.

**geoops `radiometry/_src/array.py`.** `dn_to_radiance`, `radiance_to_dn` and `dn_to_reflectance` (a linear decode with scale and offset per band); `bt_from_radiance`, the two-coefficient form T = K₂ / ln(K₁ / L + 1) used for Landsat, without GOES's b₁ and b₂; `dos1`, a dark-object subtraction by `np.nanpercentile`. All NumPy. They move; the display helpers in the same file (`min_max_normalize`, `percentile_clip`, `gamma_correct`) are not radiometry and stay in geoops.

**geoops `radiometry/_src/solar.py`.** `earth_sun_distance_correction_factor`, `compute_sza` and `observation_date_correction_factor` are thin wrappers over `georeader.reflectance`, which uses pysolar. They take a `datetime` and return one float per scene, evaluated at the scene centre. radiomax replaces them with per-pixel, traceable versions.

**geoops `radiometry/_src/operators.py`.** The Operators, including `RadianceToReflectance` over `georeader.reflectance.radiance_to_reflectance`, stay in geoops and call radiomax.

**geoops `geom/_src/array.py` and `operators.py`.** `bowtie_detector_positions` and `scan_pixel_growth` give the MODIS and VIIRS scan geometry; they move. `GeostationaryParallaxCorrect` does its ray intersection inside `_apply`, on NumPy only, on a sphere of 6,378,137 m; its maths moves and the Operator stays.

**geoops `restore/_src/array.py`.** `despeckle_lee` (window 7, C\_u = 0.523), `despeckle_refined_lee` and `despeckle_frost` (damping 2); they move.

**geoproducts.** `himawari/reader.py` `_brightness_temperature` applies Planck at the central wavelength and then the quadratic c₀ + c₁ T + c₂ T² from the file header; `goes/l1b.py` applies (fk₂ / ln(fk₁ / L + 1) − b₁) / b₂. Both move as `PlanckQuadratic` and `PlanckLinear`; the readers keep reading the coefficients. In `_src/geostationary.py`, `scan_angle_transform` and `on_earth` are navigation maths and move; `geos_crs` and `FixedGrid` build CRSs and stay.

**radtrax and plumax.** radtrax's `atmosphere/geometry.py` already consolidated the air-mass factor (radtrax#27), and `slab/config.py`'s property calls it. plumax still has two copies, `hapi_lut/beers.py` and `radtran/config.py`. So there are three implementations, not the four the Design tab counts.
