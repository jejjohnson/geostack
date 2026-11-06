---
title: "radiomax: observation geometry and radiometry for JAX"
short_title: "radiomax"
description: "radiomax converts and locates what a sensor saw: solar and viewing angles per pixel, the Earth–Sun distance, the conversions from counts to radiance to reflectance, Planck and brightness temperature, the geometric air-mass factor, parallax and geostationary navigation, swath and bowtie geometry, and the empirical retrievals applied pixel by pixel. It sits at GeoNumerics tier 1 and has no state to retrieve; radiative transfer, which does, is radtrax's. Status: proposed, at design-sketch depth."
date: 2026-11-05
keywords:
  - radiomax
  - design
  - GeoStack
---

radiomax converts and locates what a sensor saw: solar and viewing angles per pixel, the Earth–Sun distance, the conversions from counts to radiance to reflectance, Planck and brightness temperature, the geometric air-mass factor, parallax and geostationary navigation, swath and bowtie geometry, and the empirical retrievals applied pixel by pixel. It sits at GeoNumerics tier 1 and has no state to retrieve; radiative transfer, which does, is radtrax's. Status: proposed, at design-sketch depth.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoNumerics, tier 1 |
| Imports | geophysix, integrix; JAX, NumPy |
| Imported by | radtrax, plumax; geoops, geoproducts |
| Takes over | geoops `radiometry` (counts, radiance, reflectance, a scene-centre solar zenith, Earth–Sun distance, brightness temperature, dark-object subtraction); the bowtie and parallax maths in geoops `geom`; the despeckling filters in geoops `restore`; the brightness-temperature copies in geoproducts; the geometric air-mass factors in radtrax and plumax; geostationary navigation from geoproducts |
| Does not own | radiative transfer and full air-mass factors (radtrax); spectral indices (geoops); SRF tables (georeader); cell corners and areas of a swath (gridax) |
| Status | proposed; no repository yet |
| Tracked in | the [layer map](../stack_layer_map.md) and its section "radiomax and radtrax: where the line is"; the companion [geophysix](geophysix.md) and [integrix](integrix.md) designs |

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
| The geometric air-mass factor of the target signature | AMF = 1 / cos θₛ + 1 / cos θᵥ | radtrax `atmosphere/geometry.py`; plumax `hapi_lut/beers.py` and its radtran config |
| Cloud screening in the thermal bands of GOES and Himawari | brightness temperature from radiance | geoops `bt_from_radiance`; geoproducts' Himawari `_brightness_temperature`; geoproducts' GOES L1B formula |
| Where a plume at height H really is, seen from geostationary orbit | parallax and fixed-grid navigation | geoops `GeostationaryParallaxCorrect`; geoproducts `_src/geostationary.py` |
| VIIRS and MODIS scan geometry | bowtie detector positions | geoops `bowtie_detector_positions` and `BowtieCorrection` |

Brightness temperature is written three times in geotoolz and the geometric air-mass factor three times across radtrax and plumax. The rest lives inside geoops Operators, so radtrax and plumax, which sit on plain arrays, cannot reuse it without importing a GeoStack toolkit; they copy it instead.

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

with R(λ) the band's spectral response; the two integrals are an integrix rule.

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

Six modules. Each takes plain arrays (angles, radiances, coefficients) and returns plain arrays; sensor metadata such as calibration coefficients and spectral responses arrive as arguments from the readers.

| Module | Owns | Comes from |
| --- | --- | --- |
| `radiomax.geometry` | solar position and zenith, Earth–Sun distance, viewing angles, relative azimuth and scattering angle, parallax, geostationary navigation, swath and bowtie scan geometry | geoops `compute_sza`, `earth_sun_distance_correction_factor`, `bowtie_detector_positions`, the maths of `GeostationaryParallaxCorrect`; geoproducts `_src/geostationary.py` |
| `radiomax.radiometry` | counts to radiance and back, radiance to reflectance and back, band solar irradiance, the geometric air-mass factor, dark-object subtraction and empirical-line correction | geoops `dn_to_radiance`, `radiance_to_dn`, `dn_to_reflectance`, `dos1`, `SimpleAtmosphericCorrection`; the lazy `georeader.reflectance` calls; radtrax's and plumax's `air_mass_factor` |
| `radiomax.thermal` | Planck's law, brightness temperature and its inverse, from band coefficients or a spectral response | geoops `bt_from_radiance`; geoproducts' Himawari `_brightness_temperature` and GOES L1B formula |
| `radiomax.retrievals` | split-window SST and LST, ocean-colour ratios, Ross–Li BRDF and NBAR | new |
| `radiomax.sar` | SAR calibration to σ⁰, Lee, refined Lee and Frost despeckling | geoops `despeckle_lee`, `despeckle_refined_lee`, `despeckle_frost` |
| `radiomax.altimetry` | altimetry range corrections (tropospheric, ionospheric, inverse barometer) | new |

### What radiomax owns, and what it hands off

| Neighbour | radiomax owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| radtrax | angles, unit conversions, the geometric air-mass factor, Planck with band coefficients | radiative transfer, spectroscopy, full air-mass factors, the instrument model that convolves a modelled spectrum with an SRF, methane retrievals | does it need radiation moving through an atmospheric state? Then radtrax |
| geoops | the maths | Operators on GeoArray, spectral indices, masks | geoops' radiometry Operators stay and call radiomax |
| geoproducts | navigation and brightness-temperature maths | readers, file formats, the calibration coefficients as metadata | a reader passes its coefficients to radiomax |
| georeader | nothing of its own | the SRF tables, read-only (spaceml-org's) | SRFs arrive as arrays; radiomax does not import georeader |
| geophysix | sensor geometry | the Earth, its constants, h, c, k\_B | radiomax reads geophysix for every constant |
| gridax | the scan geometry of a swath | the cell corners and areas of a swath grid | how a sensor scanned is radiomax's; the grid it produced is gridax's |
| integrix | which integral a band needs | the rule that computes it | band irradiance is an `ix.rule` over the SRF |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| geoops `radiometry` | the array functions in `_src/array.py` and `_src/solar.py` | radiomax; the Operators stay in geoops and call it |
| geoops `geom`, `restore` | bowtie and parallax maths; the despeckling filters | radiomax; the Operators stay |
| geoproducts | Himawari and GOES brightness temperature; geostationary navigation maths | radiomax; the readers and `geos_crs` stay |
| radtrax, plumax | three `air_mass_factor` copies | `radiomax.radiometry` |
| geoops (via georeader) | the lazy `georeader.reflectance` calls | radiomax, with georeader as a test reference |

## In use

A methane scene from counts to the inputs a matched filter needs: angles, reflectance, the air-mass factor of the target signature, and a brightness temperature for cloud screening. Every radiomax name is proposed.

```python
# Shapes: B bands; Ny × Nx pixels; W wavelengths of the spectral responses

import geophysix as gpx
import radiomax as rmx

# --- angles ---
# proposed (radiomax)
# cos θₛ = sin φ sin δ + cos φ cos δ cos ω
solar_zenith = rmx.geometry.solar_zenith(lat, lon, time)  # (Ny, Nx) → (Ny, Nx)
# proposed (radiomax)
# θᵥ = angle between the line of sight s − p and the local vertical
view_zenith = rmx.geometry.view_zenith(  # (Ny, Nx) → (Ny, Nx)
    lat, lon, satellite_position, earth=gpx.earth.WGS84
)
# proposed (radiomax)
# d(t), in astronomical units
distance = rmx.geometry.earth_sun_distance(time)  # → ()

# --- counts to reflectance ---
# proposed (radiomax)
# L = g · DN + o, per band
radiance = rmx.radiometry.counts_to_radiance(counts, gain, offset)  # (B, Ny, Nx)
# proposed (radiomax)
# E₀ = ∫ E⊙ R dλ / ∫ R dλ, an integrix rule over each response
irradiance = rmx.radiometry.band_irradiance(  # (W,), (B, W) → (B,)
    solar_spectrum, responses, wavelength
)
# proposed (radiomax)
# ρ = π L d² / (E₀ cos θₛ)
reflectance = rmx.radiometry.reflectance(  # (B, Ny, Nx) → (B, Ny, Nx)
    radiance, irradiance, solar_zenith, distance
)

# --- what the matched filter and the cloud screen need ---
# proposed (radiomax)
# AMF = 1 / cos θₛ + 1 / cos θᵥ
air_mass = rmx.radiometry.air_mass_factor(solar_zenith, view_zenith)  # → (Ny, Nx)
# proposed (radiomax)
# T = (fₖ₂ / ln(fₖ₁ / L + 1) − b₁) / b₂
brightness = rmx.thermal.brightness_temperature(  # (Ny, Nx) → (Ny, Nx)
    thermal_radiance, coefficients
)
```

radtrax takes `solar_zenith`, `view_zenith` and `air_mass` as inputs to its forward model and builds the target signature from them; nothing in this block models the atmosphere.

## Integration across the stack

Four libraries would import radiomax. geoops and geoproducts keep their Operators and readers and call radiomax inside them; radtrax and plumax drop their copies.

| Consumer | Uses today | With radiomax |
| --- | --- | --- |
| radtrax | its own geometric `air_mass_factor`; angles passed in by the caller | `radiomax.radiometry.air_mass_factor`; angles from `radiomax.geometry`; keeps radiative transfer and the instrument model |
| plumax | `air_mass_factor` in `hapi_lut/beers.py` and its radtran config | radiomax |
| geoops | radiometry array functions, bowtie and parallax maths, despeckling filters, lazy `georeader.reflectance` calls | the same Operators, calling radiomax; the georeader calls go |
| geoproducts | Himawari and GOES brightness temperature; geostationary navigation | radiomax inside the readers; the calibration coefficients stay reader metadata |

For a geoops user nothing changes; for a library below GeoStack the import moves down:

```python
# Today:
# today (geoops)
from geoops.radiometry import bt_from_radiance

# After:
# proposed (radiomax)
from radiomax.thermal import brightness_temperature
```

geoops keeps `bt_from_radiance` as a deprecated alias for one release, and its `BTFromRadiance` Operator keeps its name.

## Decisions and open questions

Two decided on the layer map, three leaning, two open. Each entry gives the position first, then the trade-off.

### 1. Layer: GeoNumerics tier 1

**Decided (5 Nov 2026, on the layer map).** radiomax imports geophysix and integrix and nothing from GeoStack, so radtrax and plumax can use it on plain arrays. geoops and geoproducts sit above it.

### 2. The line with radtrax

**Decided (5 Nov 2026, on the layer map).** Geometry, unit conversions and empirical formulas with fitted coefficients are radiomax's; anything that needs radiation moving through an atmospheric state is radtrax's. Kernel BRDF falls on radiomax's side (fixed kernels, fitted weights, no path); convolving a modelled spectrum with an SRF falls on radtrax's.

### 3. Spectral responses and coefficients arrive as arguments

**Leaning.** The SRF tables stay in georeader, which is spaceml-org's and read-only to us. radiomax never imports it: a reader or a geoops Operator loads the SRF and the calibration coefficients and passes arrays in. That keeps radiomax free of file formats and of any GeoStack dependency.

### 4. Replace the lazy calls into georeader.reflectance

**Leaning.** geoops reaches into `georeader.reflectance` for counts, radiance and reflectance conversions. Those become radiomax calls, traceable under JAX, with georeader's results kept as the test reference.

### 5. Coefficient tables as data

**Leaning.** Split-window, ocean-colour and BRDF coefficients differ by sensor and by version. They ship as plain data specs, one per sensor, passed to the formula, as geoproducts' presets become plain data; the formula itself has no sensor in it.

### 6. Which solar-position algorithm

**Open.** The simple almanac formulas are accurate to about 0.01° and trace cheaply; NREL's SPA reaches about 0.0003° at many times the cost. Leaning: the almanac formulas by default, SPA behind an argument, both tested against pvlib.

### 7. SAR and altimetry: now or when needed?

**Open.** The layer map gives radiomax SAR calibration, despeckling and altimetry corrections, but only geoops' despeckling filters exist today and nothing below GeoStack calls them. Leaning: move the despeckling filters in wave 4, and write the altimetry corrections when a consumer asks.

## Roadmap

Six waves, each a GitHub issue epic in the radiomax repository once it exists. Geometry and radiometry come first: they end the most copies and every methane retrieval needs them.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| 0. Scaffold | the repository from the library template; CI, capability index; reference tests against `georeader.reflectance` and pvlib | — | geophysix wave 1 |
| 1. Geometry and radiometry | solar and viewing angles, Earth–Sun distance, counts, radiance and reflectance, band irradiance, the geometric air-mass factor | geoops' radiometry arrays; the three air-mass factors; the lazy georeader calls (decision 4) | 0; integrix wave 0 |
| 2. Thermal and navigation | Planck and brightness temperature; parallax, geostationary navigation, bowtie geometry | the three brightness-temperature copies; the navigation and parallax maths in geoproducts and geoops | 1 |
| 3. Empirical retrievals | split-window SST and LST, ocean-colour ratios, Ross–Li BRDF and NBAR, with coefficient tables as data (decision 5) | — | 1 |
| 4. SAR and altimetry | SAR calibration and despeckling; altimetry corrections when asked for (decision 7) | geoops' despeckling arrays | 1 |
| 5. Consumers switch | radtrax, plumax, geoops and geoproducts onto radiomax | the remaining copies | 1–4 |

Waves 2, 3 and 4 can run side by side. Wave 5 is one issue per consumer, filed in that consumer's repository and linked to the radiomax epic.

## Sources

- [The GeoStack layer map](../stack_layer_map.md): radiomax's layer, tier, imports and moves, and the section "radiomax and radtrax: where the line is".
- Reda, I. and Andreas, A. (2004). Solar position algorithm for solar radiation applications. *Solar Energy* 76, 577–589: NREL's SPA.
- NOAA and NASA, *GOES-R Series Product Definition and Users' Guide*, Volume 3 (Level 1b products): the Planck coefficients behind brightness temperature.
- Lucht, W., Schaaf, C. B. and Strahler, A. H. (2000). An algorithm for the retrieval of albedo from space using semiempirical BRDF models. *IEEE Transactions on Geoscience and Remote Sensing* 38, 977–998: the Ross–Li kernels.
- Code read on 5 Nov 2026 at geotoolz `d2f53f4` and geostack's submodule pins: geoops `radiometry/_src/` (`array.py`, `solar.py`, `operators.py`), `geom/_src/array.py` and `operators.py`, `restore/_src/array.py`; geoproducts `himawari/reader.py`, `goes/l1b.py`, `_src/geostationary.py`; radtrax `atmosphere/geometry.py`, `slab/config.py`; plumax `hapi_lut/beers.py`, `radtran/config.py`.
