---
title: "radiomax: observation geometry and radiometry on any array library"
short_title: "radiomax"
description: "radiomax converts and locates what a sensor saw: solar and viewing angles per pixel, the Earth–Sun distance, the conversions from counts to radiance to reflectance, Planck and brightness temperature, the geometric air-mass factor, parallax and geostationary navigation, swath and bowtie geometry, and the empirical retrievals applied pixel by pixel. It sits at GeoNumerics tier 1 and has no state to retrieve; radiative transfer, which does, is radtrax's. Status: Wave 0 is built."
date: 2026-11-05
keywords:
  - radiomax
  - design
  - GeoStack
---

radiomax converts and locates what a sensor saw: solar and viewing angles per pixel, the Earth–Sun distance, the conversions from counts to radiance to reflectance, Planck and brightness temperature, the geometric air-mass factor, parallax and geostationary navigation, swath and bowtie geometry, and the empirical retrievals applied pixel by pixel. It sits at GeoNumerics tier 1 and has no state to retrieve; radiative transfer, which does, is radtrax's. Status: Wave 0 is built.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoNumerics, tier 1 |
| Imports | geophysix, array-api-compat, NumPy; JAX and xarray optional |
| Imported by | radtrax; geoops, geoproducts, xrops (planned) |
| Takes over | geoops `radiometry` (counts, radiance, reflectance, a scene-centre solar zenith, Earth–Sun distance, brightness temperature); the bowtie and parallax maths in geoops `geom`; the brightness-temperature copies in geoproducts; geostationary navigation from geoproducts; from radtrax, Planck's law and the monochromatic brightness temperature, the air masses, the Earth–Sun distance, the Ross–Li kernels and the spectral response matrices |
| Does not own | radiative transfer and full air-mass factors (radtrax); spectral indices (geoops); SRF tables (georeader); cell corners and areas of a swath (gridax) |
| Status | Wave 0 built in [radiomax](https://github.com/jejjohnson/radiomax); the modules are planned |
| Tracked in | [geostack#300](https://github.com/jejjohnson/geostack/issues/300); the [layer map](../stack_layer_map.md) and its section "radiomax and radtrax: where the line is"; the companion [geophysix](geophysix.md) and [integrix](integrix.md) designs |

```{figure} ../../images/libraries/radiomax.svg
:alt: where radiomax sits · GeoNumerics tier 1
:width: 100%

Where radiomax sits · geonumerics tier 1.
```

The test that places a name, from the layer map: does it need a model of radiation moving through the atmosphere? Then it is radtrax's. Is it geometry, a unit conversion, or an empirical formula with fitted coefficients applied pixel by pixel? Then it is radiomax's. The import goes one way, radtrax to radiomax. This page is the design; core types, algorithms and the code survey are in [radiomax: design details](radiomax_details.md).

## The problem

Before a matched filter or a retrieval sees a methane pixel, the scene is calibrated, located and converted, and each of those steps is coded more than once:

| Step | What it computes | Where it is coded today |
| --- | --- | --- |
| Counts to radiance | L = g · DN + o, with a gain and offset per band | geoops `dn_to_radiance` and `DNToRadiance`; geoproducts' Himawari `_calibrate` and GOES `calibration_coefficients` |
| Radiance to top-of-atmosphere reflectance | ρ = π L d² / (E₀ cos θₛ) | geoops `RadianceToReflectance`, `earth_sun_distance_correction_factor`, `compute_sza`; and lazy calls into `georeader.reflectance` |
| The geometric air-mass factor of the target signature | AMF = 1 / cos θₛ + 1 / cos θᵥ | radtrax `atmosphere/geometry.py` |
| Cloud screening in the thermal bands of GOES and Himawari | brightness temperature from radiance | geoops `bt_from_radiance`; geoproducts' Himawari `_brightness_temperature`; geoproducts' GOES L1B formula; radtrax `radiometry/brightness.py` |
| Where a plume at height H really is, seen from geostationary orbit | parallax and fixed-grid navigation | geoops `GeostationaryParallaxCorrect`; geoproducts `_src/geostationary.py` |
| Band averages over a spectral response | f̄ = ∫ f R dλ / ∫ R dλ | geoops, through georeader; radtrax `instrument/srf.py`, `instrument/spectral.py`, `geo/srf.py` |
| VIIRS and MODIS scan geometry | bowtie detector positions | geoops `bowtie_detector_positions` and `BowtieCorrection` |

Brightness temperature is written three times in geotoolz and once in radtrax. What geoops holds lives inside its Operators, so radtrax, which sits on plain arrays, cannot reuse it without importing a GeoStack toolkit; what radtrax holds needs JAX, so geoops and geoproducts cannot reuse that either.

## Angles, units and fitted formulas

Everything radiomax computes is a function of a pixel's position, the time, the sensor and a few coefficients. None of it follows light through the air.

**Solar geometry.** With δ the solar declination and ω the hour angle at time t, the solar zenith angle θₛ at latitude φ is

```{math}
\cos\theta_s = \sin\varphi \sin\delta + \cos\varphi \cos\delta \cos\omega
```

and the Earth–Sun distance d(t), in astronomical units, scales the solar irradiance by 1 / d².

**Viewing geometry.** From the satellite's position s and the pixel's position p, both in Earth-centred coordinates from geophysix, the line of sight is s − p; its angle from the local vertical is the view zenith θᵥ, and its direction in the local east–north plane is the view azimuth. The relative azimuth and the scattering angle follow from the solar and viewing directions.

**Radiometry.** Counts become radiance through a gain g and offset o, and radiance becomes top-of-atmosphere reflectance through the band's solar irradiance E₀:

```{math}
L = g \cdot DN + o
```

```{math}
\rho_{\mathrm{TOA}} = \frac{\pi L d^2}{E_0 \cos\theta_s}
```

```{math}
E_0 = \frac{\int E_\odot(\lambda) R(\lambda) d\lambda}{\int R(\lambda) d\lambda}
```

with R(λ) the band's spectral response; the two integrals are weighted sums, or, for a response tabulated on another grid, the exact integrals of [radiomax#23](https://github.com/jejjohnson/radiomax/issues/23).

**Thermal bands.** Planck's law gives the radiance of a black body at temperature T, with h, c and k\_B from geophysix:

```{math}
B(\lambda, T) = \frac{2 h c^2}{\lambda^5} \frac{1}{\exp(h c / (\lambda k_B T)) - 1}
```

A sensor's brightness temperature inverts it with band coefficients; GOES, for one, publishes T = (fₖ₂ / ln(fₖ₁ / L + 1) − b₁) / b₂.

**Geometric air-mass factor.** The path length through a plane-parallel atmosphere, down to the surface and back up to the sensor, relative to the vertical:

```{math}
\mathrm{AMF} = \frac{1}{\cos\theta_s} + \frac{1}{\cos\theta_v}
```

The full air-mass factor, with scattering weights per layer, is radtrax's.

**Parallax.** A cloud or plume at height H, seen at view zenith θᵥ, appears displaced along the view azimuth by H tan θᵥ; geostationary navigation maps a sensor's scan angles to latitude and longitude on geophysix's ellipsoid.

**Empirical retrievals.** Formulas with fitted coefficients, applied pixel by pixel: split-window sea- and land-surface temperature from two thermal bands, SST = a₀ + a₁ T₁₁ + a₂ (T₁₁ − T₁₂) + a₃ (sec θᵥ − 1); ocean-colour band ratios; and kernel BRDF, R = f\_iso + f\_vol K\_vol + f\_geo K\_geo, with fixed Ross–Li kernels and fitted weights.

## Modules and boundaries

Six modules. Each takes arrays of any array-API library, or DataArrays (angles, radiances, coefficients), and returns the same kind; sensor metadata such as calibration coefficients and spectral responses arrive as arguments from the readers.

| Module | Owns | Comes from |
| --- | --- | --- |
| `radiomax.geometry` | solar position and zenith, Earth–Sun distance, viewing angles, relative azimuth and scattering angle, air masses, parallax, geostationary navigation, swath and bowtie scan geometry | geoops `compute_sza`, `earth_sun_distance_correction_factor`, `bowtie_detector_positions`, the maths of `GeostationaryParallaxCorrect`; geoproducts `on_earth`; radtrax's air masses and `earth_sun_distance` |
| `radiomax.radiometry` | counts to radiance and back, radiance to reflectance and back, band averages, response matrices and band solar irradiance, spectral densities | geoops `dn_to_radiance`, `radiance_to_dn`, `dn_to_reflectance`; the lazy `georeader.reflectance` calls; radtrax's response matrices |
| `radiomax.thermal` | Planck's law, brightness temperature and its inverse, from band coefficients or a spectral response | radtrax `radiometry/planck.py` and the monochromatic half of `brightness.py`; geoops `bt_from_radiance`; geoproducts' Himawari `_brightness_temperature` and GOES L1B formula |
| `radiomax.retrievals` | split-window SST and LST, ocean-colour ratios, Ross–Li BRDF and NBAR | the Ross–Li kernels from radtrax `surface/ross_li.py`; the rest is new |
| `radiomax.sar` | SAR calibration to σ⁰; despeckling only if its design issue says so | new |
| `radiomax.altimetry` | altimetry range corrections (tropospheric, ionospheric, inverse barometer) | new |

### What radiomax owns, and what it hands off

| Neighbour | radiomax owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| radtrax | angles, unit conversions, the geometric air-mass factor, Planck with band coefficients | radiative transfer, spectroscopy, full air-mass factors, the instrument model that applies a response matrix to a modelled spectrum, band brightness temperature (a solve through an instrument), methane retrievals | does it need radiation moving through an atmospheric state? Then radtrax |
| geoops | the maths | Operators on GeoArray, spectral indices, masks, dark-object subtraction (a percentile over a scene) | geoops' radiometry Operators stay and call radiomax |
| geoproducts | navigation and brightness-temperature maths | readers, file formats, the calibration coefficients as metadata | a reader passes its coefficients to radiomax |
| georeader | nothing of its own | the SRF tables, read-only (spaceml-org's) | SRFs arrive as arrays; radiomax does not import georeader |
| geophysix | sensor geometry | the Earth, its constants, h, c, k\_B | radiomax reads geophysix for every constant |
| gridax | the scan geometry of a swath | the cell corners and areas of a swath grid | how a sensor scanned is radiomax's; the grid it produced is gridax's |
| integrix, functax | the band average, with its own trapezoid weights | quadrature rules and integration on JAX | both need JAX, so they are test references; a functax rule's weights plug in through `weights=` |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| geoops `radiometry` | the array functions in `_src/array.py` and `_src/solar.py` | radiomax; the Operators stay in geoops and call it |
| geoops `geom` | bowtie and parallax maths | radiomax; the Operators stay |
| geoproducts | Himawari and GOES brightness temperature; `on_earth` | radiomax; the readers, `scan_angle_transform` and `geos_crs` stay |
| radtrax | Planck and its views, the monochromatic brightness temperature, both air masses, the Earth–Sun distance, the Ross–Li kernels, the response matrices, the relative azimuths | radiomax; `ViewingGeometry`, `RossLi`, the band brightness temperature, the instruments and the readers stay |
| geoops (via georeader) | the lazy `georeader.reflectance` calls | radiomax, with georeader as a test reference |

## In use

A methane scene from counts to the inputs a matched filter needs: angles, reflectance, the air-mass factor of the target signature, and a brightness temperature for cloud screening. Every radiomax name is proposed.

```python
# Shapes: B bands; Ny × Nx pixels; W wavelengths of the spectral responses

import geophysix as gpx
import radiomax as rmx

# --- angles ---
# proposed (radiomax), on the host
# () datetime64 → (), ()
day, seconds = rmx.geometry.utc_day_seconds(time)
# proposed (radiomax)
# cos θₛ = sin φ sin δ + cos φ cos δ cos ω
solar_zenith = rmx.geometry.solar_zenith(lon, lat, day, seconds)  # → (Ny, Nx)
# proposed (radiomax)
# θᵥ = angle between the line of sight s − p and the local vertical
view_zenith, view_azimuth = rmx.geometry.view_angles(  # → (Ny, Nx), (Ny, Nx)
    lon, lat, *satellite_position, earth=gpx.earth.WGS84
)
# proposed (radiomax)
# d(t), in astronomical units
distance = rmx.geometry.earth_sun_distance(day, seconds)  # → ()

# --- counts to reflectance ---
# proposed (radiomax)
# L = g · DN + o, per band
radiance = rmx.radiometry.counts_to_radiance(counts, gain, offset)  # (B, Ny, Nx)
# proposed (radiomax)
# E₀ = ∫ E⊙ R dλ / ∫ R dλ
irradiance = rmx.radiometry.band_irradiance(  # (W,), (B, W), (W,) → (B,)
    solar_spectrum, responses, wavelength
)
# proposed (radiomax)
# ρ = π L d² / (E₀ cos θₛ)
reflectance = rmx.radiometry.reflectance(  # (B, Ny, Nx) → (B, Ny, Nx)
    radiance, irradiance[:, None, None], solar_zenith, distance
)

# --- what the matched filter and the cloud screen need ---
# proposed (radiomax)
# AMF = 1 / cos θₛ + 1 / cos θᵥ
air_mass = rmx.geometry.air_mass_factor(solar_zenith, view_zenith)  # → (Ny, Nx)
# proposed (radiomax)
# T = (fₖ₂ / ln(fₖ₁ / L + 1) − b₁) / b₂
brightness = rmx.thermal.brightness_temperature_band(  # (Ny, Nx) → (Ny, Nx)
    thermal_radiance, coefficients
)
```

radtrax takes `solar_zenith`, `view_zenith` and `air_mass` as inputs to its forward model and builds the target signature from them; nothing in this block models the atmosphere.

## Integration across the stack

Four libraries would import radiomax. geoops, geoproducts and xrops keep their Operators and readers and call radiomax inside them; radtrax drops its copies. plumax computes no angles or radiometry of its own and takes them from radtrax.

| Consumer | Uses today | With radiomax |
| --- | --- | --- |
| radtrax | its own Planck, air masses, Ross–Li kernels, response matrices and Earth–Sun distance | radiomax; keeps radiative transfer and the instrument model |
| geoops | radiometry array functions, bowtie and parallax maths, lazy `georeader.reflectance` calls | the same Operators, calling radiomax; the georeader calls go |
| geoproducts | Himawari and GOES brightness temperature; geostationary navigation | radiomax inside the readers, still without JAX; the calibration coefficients stay reader metadata |
| xrops | nothing yet ([geotoolz#719](https://github.com/jejjohnson/geotoolz/issues/719) asks for it) | thin Operators: radiomax takes and returns DataArrays |

For a geoops user nothing changes; for a library below GeoStack the import moves down:

```python
# Today:
# today (geoops)
from geoops.radiometry import bt_from_radiance

# After:
# proposed (radiomax)
from radiomax.thermal import brightness_temperature
```

No deprecated alias is kept; geoops' `BTFromRadiance` Operator keeps its name.

## Decisions and open questions

Nine decided by the owner on 23 Nov 2026 in [radiomax#15](https://github.com/jejjohnson/radiomax/issues/15), which holds the measurements, and [geostack#300](https://github.com/jejjohnson/geostack/issues/300); one leaning, one open. Each entry gives the position first, then the trade-off.

### 1. Layer and dependencies: GeoNumerics tier 1, on NumPy, geophysix and array-api-compat

**Decided (5 and 23 Nov 2026).** radiomax imports geophysix and nothing else from the stack, so radtrax can use it on plain arrays and geoops and geoproducts can use it without JAX. integrix requires JAX, and functax's `quad` and `interp` need its `jax` extra, so both are reference libraries in the tests. `array-api-compat` is required, because without it an array that has no `__array_namespace__` falls back to NumPy silently.

### 2. The line with radtrax

**Decided (5 and 23 Nov 2026).** Geometry, unit conversions and empirical formulas with given coefficients are radiomax's; anything that needs radiation moving through an atmospheric state is radtrax's. As much as possible moves out of radtrax: what solves through an instrument, reads a file or is an Equinox module stays. Kernel BRDF falls on radiomax's side (fixed kernels, fitted weights, no path). For spectral responses, radiomax builds the matrix and radtrax's instruments apply it. Planck's law and the Ross–Li kernels keep exact gradients without radtrax's hand-written derivatives.

### 3. Spectral responses and coefficients arrive as arguments

**Decided (23 Nov 2026).** The SRF tables stay in georeader, which is spaceml-org's and read-only to us. radiomax reads no files and never imports georeader: a reader or a geoops Operator loads the SRF and the calibration coefficients and passes arrays in.

### 4. Replace the lazy calls into georeader.reflectance

**Decided (23 Nov 2026).** geoops reaches into `georeader.reflectance` for counts, radiance and reflectance conversions. Those become radiomax calls, with georeader's results kept as the test reference.

### 5. Coefficient tables as data

**Leaning.** Split-window, ocean-colour and BRDF coefficients differ by sensor and by version. They ship as plain data specs, one per sensor, passed to the formula, as geoproducts' presets become plain data; the formula itself has no sensor in it.

### 6. Which solar-position algorithm

**Decided (23 Nov 2026).** The Astronomical Almanac formulas, for the solar position and the Earth–Sun distance; no SPA option until a user needs one. Over 2000–2050 the zenith is within 0.0121° of SPA, and the distance within 8.7 × 10⁻⁵ AU. The formula geoops and radtrax use for the distance is off by up to 8.0 × 10⁻⁴ AU, so switching moves a reflectance by up to 1.5 × 10⁻³ relative.

### 7. SAR and altimetry: now or when needed?

**Open.** The layer map gives radiomax SAR calibration, despeckling and altimetry corrections, but only geoops' despeckling filters exist today and nothing below GeoStack calls them. Despeckling is a window filter and not a per-pixel formula, so the leaning is that it stays in geoops; Wave 4 opens with that question. The altimetry corrections are written when a consumer asks.

### 8. Arrays, labels and units

**Decided (23 Nov 2026).** One implementation in the Python array API standard, as in geophysix, for NumPy, JAX, CuPy and others. A DataArray in gives a DataArray out, through `xr.apply_ufunc`; an array type the backend does not know raises. Degrees by default with `degrees=False`, SI otherwise, records as keyword arguments, frozen dataclasses registered as JAX pytrees only when JAX is importable.

### 9. Time is two arrays

**Decided (23 Nov 2026).** `day`, UTC midnights since 2000-01-01, and `seconds` of the UTC day, fraction included. JAX runs in float32 by default, where a Julian date resolves a quarter of a day; the split keeps the solar zenith within 0.0124° of SPA in float32.

### 10. Planck's axis

**Decided (23 Nov 2026).** Wavenumber in cm⁻¹ and radiance in W m⁻² sr⁻¹ (cm⁻¹)⁻¹, as radtrax has it, with views per µm and per GHz. The argument is named `wavenumber_cm`.

### 11. Where the repository lives

**Decided (23 Nov 2026).** Standalone, outside the workspaces that [geostack#254](https://github.com/jejjohnson/geostack/issues/254) discusses.

## Roadmap

Six waves, each an epic in the radiomax repository. Geometry and radiometry come first: they end the most copies and every methane retrieval needs them.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| [0. Scaffold and the array backend](https://github.com/jejjohnson/radiomax/issues/8) | the repository from the library template; CI, capability index; the array backend; the test harness with pvlib, georeader and stored radtrax values | — | geophysix wave 1 |
| [1. Geometry and radiometry](https://github.com/jejjohnson/radiomax/issues/9) | solar and viewing angles, Earth–Sun distance, air masses, counts, radiance and reflectance, band averages and response matrices | geoops' radiometry arrays; the lazy georeader calls (decision 4) | 0 |
| [2. Thermal and navigation](https://github.com/jejjohnson/radiomax/issues/10) | Planck and brightness temperature; parallax, geostationary navigation, bowtie geometry | the brightness-temperature copies; the navigation and parallax maths in geoproducts and geoops | 1 |
| [3. Empirical retrievals](https://github.com/jejjohnson/radiomax/issues/11) | Ross–Li BRDF and NBAR, split-window SST and LST, ocean-colour ratios, with coefficient tables as data (decision 5) | radtrax's kernels | 1 |
| [4. SAR and altimetry](https://github.com/jejjohnson/radiomax/issues/12) | SAR calibration; despeckling and altimetry corrections when asked for (decision 7) | — | 1 |
| [5. Consumers switch](https://github.com/jejjohnson/radiomax/issues/13) | radtrax, geoops, geoproducts and xrops onto radiomax | the remaining copies | 1–3 |

Waves 2, 3 and 4 can run side by side. Wave 5 is one issue per consumer, filed in that consumer's repository and linked to the radiomax epic.

## Sources

- [The GeoStack layer map](../stack_layer_map.md): radiomax's layer, tier, imports and moves, and the section "radiomax and radtrax: where the line is".
- @reda2004spa: NREL's solar position algorithm.
- @goesr_pug_l1b: the Planck coefficients behind brightness temperature.
- @lucht2000brdf: the Ross–Li kernels.
- Code read on 5 Nov 2026 at geotoolz `d2f53f4` and geostack's submodule pins: geoops `radiometry/_src/` (`array.py`, `solar.py`, `operators.py`), `geom/_src/array.py` and `operators.py`, `restore/_src/array.py`; geoproducts `himawari/reader.py`, `goes/l1b.py`, `_src/geostationary.py`; radtrax `atmosphere/geometry.py`, `slab/config.py`. Read again on 23 Nov 2026 at radtrax `fadf2a6`, geotoolz `3d05a9c` and plumax `96449d2`.
