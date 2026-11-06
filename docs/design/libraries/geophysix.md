---
title: "geophysix: physical constants, Earth geometry and atmospheric state for JAX"
short_title: "geophysix"
description: "geophysix is the stack's library of physical facts about the Earth: its constants, its size and shape, its rotation, the thermodynamics of its air and the equation of state of its sea water, and the units a gas column is reported in. It sits at GeoNumerics tier 0, imports nothing from the stack, and gives every library above it one value for each constant where today there are several. Status: proposed, at design-sketch depth."
date: 2026-11-05
keywords:
  - geophysix
  - design
  - GeoStack
---

geophysix is the stack's library of physical facts about the Earth: its constants, its size and shape, its rotation, the thermodynamics of its air and the equation of state of its sea water, and the units a gas column is reported in. It sits at GeoNumerics tier 0, imports nothing from the stack, and gives every library above it one value for each constant where today there are several. Status: proposed, at design-sketch depth.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoNumerics, tier 0 |
| Imports | NumPy and pipekit-array (the array namespace); JAX optional; nothing else from the stack |
| Imported by | gridax, diagnostix, radiomax, tracerax, finitevolX, spectraldiffx; geoarray, geoops, xrops; sesmax-core, plumax, radtrax |
| Takes over | the constants modules of finitevolX, sesmax-core and xrops; WGS84 and the ellipsoid from geonnax; Coriolis and β-planes from finitevolX and xrops; seawater density from finitevolX and xrops; ideal-gas and column-unit conversions from radtrax, plumax, geoops and xrops |
| Does not own | geometry on the unit sphere (functax); grids and their metrics (gridax); diagnostics such as N² and vorticity (diagnostix); model scales and non-dimensionalisation (the models) |
| Status | proposed; no repository yet |
| Tracked in | the [layer map](../stack_layer_map.md) |

```{figure} ../../images/libraries/geophysix.svg
:alt: where geophysix sits · GeoNumerics tier 0
:width: 100%

Where geophysix sits · geonumerics tier 0.
```

geophysix sits at the bottom of GeoNumerics because everything above it needs the same Earth. Core cannot import it, so Core functions that need the Earth's size or shape take it as an argument, and geophysix supplies the value. This page is the design; core types, algorithms and the code survey are in [geophysix: design details](geophysix_details.md).

## The problem

Turning a satellite methane plume into an emission rate needs a handful of physical facts about the Earth and its air, and each library that needs one writes its own copy:

| Step in the retrieval | The fact it needs | Where it is coded today |
| --- | --- | --- |
| Column enhancement from ppm·m to kg m⁻² | the molar volume of air (geoops fixes it at 25 °C and 1 atm), the molar mass of CH₄ | geoops `convert_column_units`; plumax and xrops, each a `column_mass_to_delta_vmr` |
| Air number density along the light path | n = p / (k\_B T) | radtrax `number_density`; plumax `number_density`, and again inline in `coupled/rtm.py` |
| Dry-air column, to turn a column into XCH₄ | N\_dry = N\_A (p\_s / g − W) / M\_dry | xrops `dry_air_column` |
| Pixel areas for the integrated mass, on a lon/lat grid | the area of a cell on the ellipsoid | not yet: geoops `pixel_area` is the determinant of the affine transform, so it is right only on a projected grid |
| Wind at plume height from pressure levels | the hypsometric equation, with R\_d and g | xrops `hypsometric_height`; plumax `met/wrf.py` with its own `KAPPA` and `GRAVITY = 9.81` |
| Distance from the source to each pixel | the great-circle distance | gaussx `haversine_distance`, kernellib `great_circle_distance`, and a private `_haversine_km` in geopatcher and in xrproducts |

The copies do not agree. The same constant appears with different values across the stack, so two libraries given the same inputs can return different answers:

| Constant | Values found | Where |
| --- | --- | --- |
| Earth's mean radius | 6,371,000 m; 6,371,008.8 m; 6,371,200 m, which is IGRF's geomagnetic reference radius | xrops `grad`, gaussx; geonnax, kernellib, geoops, xrproducts, xrops `metrics`; finitevolX, somax, sesmax-core |
| Standard gravity | 9.80665 m s⁻²; 9.81 m s⁻² | finitevolX, somax, sesmax-core, xrops; xrops `atm/wrf.py`, plumax `met/wrf.py` and the default of every somax model |
| Rotation rate Ω | 7.2921159 × 10⁻⁵ s⁻¹; 2π / 86,164 s | xrops `grad`; finitevolX, somax, sesmax-core |

The Boltzmann constant and the standard atmosphere are written out four times (radtrax, plumax twice, xrops), and three repositories hold the same `constants.py`, copied from one to the next (finitevolX, somax and sesmax-core). Each difference is small, but each makes a cross-library test fail at a tolerance it should pass, and none of them is a choice anyone made.

## One Earth, passed as data

Everything geophysix holds is a closed-form function of a few constants, and the design rests on that. The constants live in one record. Every function takes the record, or the one constant it needs, as an argument whose default is that record. Nothing reads a module-level global inside a traced function.

**The ellipsoid.** The Earth is an ellipsoid of revolution with semi-major axis a and flattening f; WGS84 has a = 6,378,137 m and f = 1 / 298.257223563. With e² = f (2 − f) the squared eccentricity, a point at geodetic longitude λ, latitude φ and height h sits at

```{math}
N(\varphi) = \frac{a}{\sqrt{1 - e^2 \sin^2 \varphi}}
```

```{math}
(X, Y, Z) = \left( (N + h) \cos\varphi \cos\lambda, (N + h) \cos\varphi \sin\lambda, (N (1 - e^2) + h) \sin\varphi \right)
```

The area of a cell between latitudes φ₁ and φ₂, spanning Δλ of longitude, has a closed form, with b = a (1 − f):

```{math}
A = \frac{b^2 \Delta\lambda}{2} \left[ \frac{\sin\varphi}{1 - e^2 \sin^2\varphi} + \frac{1}{2e} \ln \frac{1 + e \sin\varphi}{1 - e \sin\varphi} \right]_{\varphi_1}^{\varphi_2}
```

On a sphere (e → 0) it reduces to A = R² Δλ (sin φ₂ − sin φ₁). A sphere is the same record with f = 0 and a chosen mean radius, so every function works on both.

**Rotation.** With Ω the rotation rate and R the radius, the Coriolis parameter and its meridional gradient are f = 2Ω sin φ and β = 2Ω cos φ / R. A β-plane linearises them about a reference latitude: f(y) = f₀ + β₀ (y − y₀).

**The air.** Dry air is an ideal gas, p = ρ R\_d T\_v with T\_v the virtual temperature, and its number density is n = p / (k\_B T). Potential temperature is θ = T (p₀ / p)^κ with κ = R\_d / c\_p, and the thickness of a layer between two pressure levels is

```{math}
z_2 - z_1 = \frac{R_d \bar T_v}{g} \ln \frac{p_1}{p_2}
```

**Sea water.** Density from temperature T, salinity S and pressure: the linear equation of state ρ = ρ₀ (1 − α (T − T₀) + β\_S (S − S₀)) first, TEOS-10 later.

**Columns.** A gas column is reported in several units. With X the enhancement in ppm·m, V\_m = R T / p the molar volume of air and M the molar mass of the gas:

```{math}
\Omega_{\mathrm{mol}} = \frac{10^{-6} X}{V_m}
```

```{math}
\Omega_{\mathrm{kg}} = M \Omega_{\mathrm{mol}}
```

```{math}
N_{\mathrm{dry}} = \frac{N_A}{M_{\mathrm{dry}}} \left( \frac{p_s}{g} - W \right)
```

None of these needs a grid, a carrier or data. They are plain functions of arrays and constants, which is why they sit at the bottom of GeoNumerics.

## Modules and boundaries

Six modules, one per kind of fact. geophysix owns closed-form physics at a point; anything on a grid, a field or a carrier belongs to the libraries above it.

| Module | Owns | Comes from |
| --- | --- | --- |
| `geophysix.constants` | SI and CODATA constants (k\_B, N\_A, R, the standard atmosphere); the Earth's g₀ and Ω; dry air and common gases (R\_d, c\_p, M\_dry, the molar masses of CH₄, CO₂, H₂O); one frozen record per convention | finitevolX `utils/constants.py`; somax and sesmax-core `constants.py`; xrops `grad` and `atm` constants; radtrax `atmosphere/gas.py`; plumax `radtran/config.py` and `hapi_lut/beers.py` |
| `geophysix.earth` | reference ellipsoids (WGS84, GRS80) and spheres of a given radius; geodetic to ECEF and back; radii of curvature; geodesic distance and azimuth on the sphere and the ellipsoid; the area of a cell | geonnax `WGS84_A`, `WGS84_F`, `lonlat_to_ecef`, `ecef_to_lonlat`; the private haversines of geopatcher and xrproducts; the radius default of geoops' parallax correction |
| `geophysix.rotation` | the Coriolis parameter, β, f-planes and β-planes | finitevolX `coriolis_param`, `coriolis_fn`; xrops `coriolis_parameter` |
| `geophysix.thermo` | the ideal gas and number density, virtual and potential temperature, the Exner function, hypsometric thickness, saturation vapour pressure | radtrax and plumax `number_density`; the maths of xrops `hypsometric_height`; plumax `met/wrf.py` constants; saturation vapour pressure is new |
| `geophysix.seawater` | equations of state, linear first and TEOS-10 later, with α and β\_S | finitevolX `eos.py` (`linear_density`, `linear_density_anomaly`); the maths of xrops `density_from_ts`, which calls gsw today |
| `geophysix.columns` | column units (ppm·m, mol m⁻², kg m⁻², molecules m⁻²), mixing ratio to column and back, the dry-air column | geoops `convert_column_units`; plumax and xrops, each a `column_mass_to_delta_vmr`; xrops `dry_air_column` |

### What geophysix owns, and what it hands off

The test for every name: is it a closed-form fact about the Earth, its air or its sea water, evaluated at a point? Then geophysix owns it.

| Neighbour | geophysix owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| functax | the Earth's size and shape: ellipsoids, radii, geodesics on the Earth | geometry on the unit sphere: frames, rotations, sphere grids, spherical harmonics | a formula with an Earth constant in it is geophysix's; one that holds on any sphere is functax's |
| gridax | the area of one cell from its corner latitudes | grids, staggering, metric arrays, the cell areas of a whole grid | gridax calls geophysix per cell; the grid is gridax's |
| diagnostix | the equation of state and the gas law | N², vorticity, geostrophy, budgets | a fact at a point is geophysix's; anything that differentiates a field is diagnostix's |
| radiomax | the gas law and column units under a radiance | solar and viewing geometry, radiometric conversions, Planck | radiometry is radiomax's; the air it passes through is geophysix's |
| Core: gaussx, kernellib, geonnax | the values of the constants | functions that take a radius, or a and f, as an argument | Core cannot import geophysix, so it takes the constants in |
| sesmax-core, plumax, radtrax | the constants and closed-form physics | model scales, non-dimensionalisation, parametrisations, spectroscopy | the models keep their scales and closures |
| xrops | the maths | the DataArray API: names, attributes, units | xrops wraps geophysix functions over xarray |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| finitevolX | `utils/constants.py` (`R_EARTH`, `GRAVITY`, `OMEGA`, `DEG2M`, `RHO`); `coriolis_param`, `coriolis_fn`; `eos.py` | geophysix; finitevolX keeps its operators |
| somax, sesmax-core | `constants.py`, the same file in both | geophysix; sesmax-core keeps `scales.py` |
| xrops | the `grad` and `atm` constants; the maths of `coriolis_parameter`, `hypsometric_height`, `dry_air_column`, `column_mass_to_delta_vmr` and `density_from_ts` | geophysix underneath; xrops keeps the DataArray wrappers |
| geonnax | `WGS84_A`, `WGS84_F`, `EARTH_MEAN_RADIUS_M`, and the ellipsoidal half of `_frames.py` | geophysix; the unit-sphere frames go to functax |
| radtrax, plumax | `BOLTZMANN_J_PER_K`, `ATM_TO_PA`, `number_density`, `column_mass_to_delta_vmr`, the molar masses | geophysix |
| geoops, geoproducts, geopatcher, xrproducts | Earth radii, `convert_column_units`, the private haversines | geophysix |

## In use

The methane example once more: pixel areas on the ellipsoid, the enhancement converted to a mass, the dry-air column for XCH₄, the heights of the wind levels, and the Coriolis field a transport model needs. Every geophysix name is proposed; each is a function of arrays with the Earth passed in.

```python
# Shapes: Ny × Nx plume pixels on a lon/lat grid; Z pressure levels

import einx
import geophysix as gpx

# proposed (geophysix)
# WGS84: a = 6,378,137 m, f = 1 / 298.257223563
earth = gpx.earth.WGS84

# --- pixel areas, on the ellipsoid ---
# proposed (geophysix)
# Aᵢⱼ = (b² Δλⱼ / 2) [q(φ)] from φᵢ to φᵢ₊₁
area = gpx.earth.cell_area(
    lat_edges, lon_edges, earth=earth
)  # (Ny+1,), (Nx+1,) → (Ny, Nx)

# --- the enhancement as a mass ---
# proposed (geophysix)
# Ω_kg = M_CH₄ · 10⁻⁶ X / V_m, with V_m = R T / p
column = gpx.columns.convert(  # (Ny, Nx) → (Ny, Nx)
    enhancement,
    units_in="ppm_m",
    units_out="kg_m2",
    gas="CH4",
    pressure=surface_pressure,
    temperature=surface_temperature,
)
# IME = Σᵢⱼ Ω_kg Aᵢⱼ
mass = einx.dot("y x, y x ->", column, area)  # → ()

# --- dry-air column, for XCH₄ ---
# proposed (geophysix)
# N_dry = N_A (p_s / g − W) / M_dry
dry_air = gpx.columns.dry_air_column(  # (Ny, Nx) → (Ny, Nx)
    surface_pressure, water_vapour=water_vapour_column
)

# --- heights of the wind levels ---
# proposed (geophysix)
# zₖ₊₁ − zₖ = (R_d T̄_v / g) ln(pₖ / pₖ₊₁)
thickness = gpx.thermo.hypsometric_thickness(  # (Z,), (Z,) → (Z−1,)
    pressure_levels, virtual_temperature
)

# --- the Coriolis field of a transport model ---
# proposed (geophysix)
# f = 2 Ω sin φ
coriolis = gpx.rotation.coriolis(lat_centres)  # (Ny,) → (Ny,)
```

Each call takes its constants from the defaults, so a caller that passes nothing gets WGS84, CODATA 2018 and standard gravity, and a caller that wants a sphere of radius 6,371,000 m passes `earth=gpx.earth.sphere(6_371_000.0)`. The sum for the mass is shown with einx; inside a pipeline it is `integrix.rule` with these areas as weights.

## Integration across the stack

Twelve libraries would import geophysix, across three layers. For most the change is an import and the end of a local `constants.py`; for xrops it is a thin DataArray layer over geophysix functions.

| Consumer | Uses today | With geophysix |
| --- | --- | --- |
| finitevolX | its own `utils/constants.py`, `coriolis_param`, `coriolis_fn` and `eos.py` | `gpx.constants`, `gpx.rotation`, `gpx.seawater`; keeps the C-grid operators |
| spectraldiffx | no constants: spherical operators infer R from the grid (`grid.Ly / π`) | grids sized from `gpx.earth` radii; no change to the operators |
| gridax, diagnostix, radiomax, tracerax | planned | cell areas and radii (gridax); the equation of state and the gas law (diagnostix); the gas law and column units (radiomax); g and Ω for particles (tracerax) |
| geoarray, geoops | radii in four places; `pixel_area` as an affine determinant; `convert_column_units` | ellipsoidal areas through gridax and geophysix; `gpx.columns` |
| xrops | `grad` and `atm` constants; Coriolis, hypsometry, dry-air columns and seawater density written for xarray | the same DataArray functions, calling geophysix underneath |
| sesmax-core | `constants.py`, copied from somax | `gpx.constants`; keeps `scales.py` |
| plumax | its own Boltzmann, standard atmosphere and molar masses; `number_density`; `column_mass_to_delta_vmr`; `met/wrf.py` constants | `gpx.thermo`, `gpx.columns`; keeps the plume physics |
| radtrax | `atmosphere/gas.py`: `BOLTZMANN_J_PER_K`, `ATM_TO_PA`, `number_density` | `gpx.thermo`; keeps the spectroscopy |

The change at each site is an import. The constants module of finitevolX, somax and sesmax-core is the clearest case:

```python
# Today:
# today (finitevolX)
from finitevolx._src.utils.constants import GRAVITY, OMEGA, R_EARTH

# After:
# proposed (geophysix)
from geophysix.constants import STANDARD

# proposed (geophysix)
from geophysix.earth import SPHERE
```

The values change with the import: R\_EARTH = 6,371,200 m becomes the record's mean radius. That is the one place the switch moves a number, so each consumer's switch lands with its own tolerance check (decision 4).

## Decisions and open questions

Two decided on the layer map, four leaning, three open. Each entry gives the position first, then the trade-off.

### 1. Layer: GeoNumerics tier 0

**Decided (5 Nov 2026, on the layer map).** geophysix imports nothing from the stack, and every GeoNumerics library, three GeoStack packages and three models import it. Core sits below and cannot import it: a Core function that needs the Earth takes the radius, or a and f, as an argument.

### 2. Scope: the Earth, its air and its sea water

**Decided (5 Nov 2026, on the layer map).** Constants, the ellipsoid and sphere metrics, Coriolis and β-planes, geodesics, atmospheric thermodynamics, seawater equations of state and column units. Model scales and non-dimensionalisation stay in the models; diagnostics of fields go to diagnostix.

### 3. Constants as records passed in

**Leaning: frozen records, passed as arguments with defaults.** `gpx.constants.STANDARD` and `gpx.earth.WGS84` are immutable records, and every function takes the one it needs as a keyword argument. A model that wants another planet, a sphere or a different convention passes its own record; nothing patches a global. Plain module-level floats would be shorter, but they are the reason three libraries hold three radii today.

### 4. Which value for each constant

**Leaning: the published conventions, one each.** The IUGG mean radius R₁ = 6,371,008.8 m for the default sphere; standard gravity g₀ = 9.80665 m s⁻²; Ω = 7.292115 × 10⁻⁵ s⁻¹ from WGS84; CODATA 2018 for k\_B, N\_A and R. finitevolX, somax and sesmax-core use 6,371,200 m today, IGRF's geomagnetic reference radius rather than a mean Earth radius, so their results move by about 3 × 10⁻⁵ relative when they switch; each switch carries a test of that size, and a model that must reproduce an old run passes the old radius explicitly.

### 5. Geodesics: the sphere in Core, the ellipsoid here

**Leaning.** Great-circle distance on a sphere of given radius stays where Core needs it (gaussx's localisation, kernellib's kernels), with its default radius equal to geophysix's and a test that says so. Geodesics on the ellipsoid (distance and azimuth, after Karney) are geophysix's. The private haversines in geopatcher and xrproducts go.

### 6. Units: SI floats, no unit library

**Leaning.** Every argument and return is in SI, with the unit in the docstring and, where two are common, in the name (`pressure_pa`). The column module is the one place units are converted, and it does so explicitly by name. A unit library such as pint would not trace under JAX.

### 7. TEOS-10: port or wrap?

**Open.** xrops calls gsw, which is NumPy and C, so it neither traces nor differentiates. Leaning: port the subset the stack uses (density, α, β\_S, conservative temperature, absolute salinity) to JAX, with gsw as the test reference, as integrix does with quadax.

### 8. Saturation vapour pressure and moist thermodynamics

**Open.** No library in the stack computes it today, and the formulas disagree with one another (Magnus, Goff–Gratch, Murphy–Koop). Leaning: one formula, Murphy–Koop, chosen by an argument, and added only when sesmax-atmos or plumax needs it.

### 9. What to take from pyproj and Shapely

**Open, to be investigated.** pyproj (PROJ and GeographicLib) and Shapely (GEOS) are C libraries, so nothing they compute traces under `jit` or differentiates. The candidates to port as maths, not code, are:

- from pyproj: the ellipsoid table; the auxiliary latitudes (geocentric, parametric, authalic, conformal, rectifying); Karney's direct and inverse geodesic and the geodesic polygon area.
- from Shapely: point-in-polygon masks, planar polygon area and distance to a polygon. These are generic geometry, so they would go to gridax or functax, not here.

The working position: map projections stay with pyproj on the host, through geoarray's `pyproj.CRS`, until something has to differentiate through one; pyproj's `Geod` is the test reference for whatever is ported. To settle: which consumers need ellipsoidal geodesics and polygon areas under `jit` (swath footprints and basin-scale budgets are the likely ones; plume-scale work is served by a local tangent plane); how Karney's iterative inverse behaves under `jit` and `grad`, including at coincident and antipodal points; and the licences, since PROJ, GeographicLib and pyproj are MIT and Shapely is BSD-3, but GEOS is LGPL, so nothing is copied from its source.

## Roadmap

Five waves, each a GitHub issue epic in the geophysix repository once it exists. The constants and the Earth come first: every other module reads them, and they end the most copies.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| 0. Scaffold | the repository from the library template; the constants and Earth records; CI, capability index; reference tests against pyproj's `Geod` and CODATA | — | — |
| 1. Constants and the Earth | `gpx.constants`, `gpx.earth`: ellipsoids and spheres, geodetic to ECEF, cell areas, geodesics on the ellipsoid | the `constants.py` of finitevolX and sesmax-core; geonnax's WGS84; the private haversines | 0 |
| 2. Rotation and columns | Coriolis, β and β-planes; column units, mixing ratio to column, the dry-air column, number density | finitevolX's and xrops' Coriolis; the column and gas-law copies in geoops, plumax, radtrax and xrops | 1 |
| 3. Thermodynamics and sea water | the ideal gas, virtual and potential temperature, hypsometric thickness; the linear equation of state, then the TEOS-10 port (decision 7) | finitevolX `eos.py`; xrops' gsw calls | 1 |
| 4. Consumers switch | the twelve importers onto geophysix, each with a tolerance test for any value that moves (decision 4) | the remaining local constants | 1–3 |

Waves 2 and 3 can run side by side. Wave 4 is one issue per consumer, filed in that consumer's repository and linked to the geophysix epic; somax needs none, since sesmax-core replaces it.

## Sources

- [The GeoStack layer map](../stack_layer_map.md): geophysix's layer, tier, imports and moves.
- WGS84: National Geospatial-Intelligence Agency, *Department of Defense World Geodetic System 1984*, NGA.STND.0036: a, f and Ω.
- [CODATA recommended values](https://physics.nist.gov/cuu/Constants/): k\_B, N\_A, R.
- Karney, C. F. F. (2013). [Algorithms for geodesics](https://doi.org/10.1007/s00190-012-0578-z). *Journal of Geodesy* 87, 43–55: geodesics on the ellipsoid.
- [TEOS-10](https://www.teos-10.org/): the thermodynamic equation of seawater, and the gsw reference code.
- Code read on 5 Nov 2026 at geostack's submodule pins and sesmax `main`: finitevolX `_src/utils/constants.py`, `operators/diagnostics.py`, `operators/eos.py`; somax `_src/constants/`; sesmax-core `_src/constants.py`, `_src/scales.py`; xrops `grad/_src/constants.py`, `atm/_src/`, `atm/gas/ch4/_src/column.py`, `ocn/_src/kinematics.py`; geonnax `_frames.py`; kernellib `functional/_geo.py`; gaussx `_inference/_localization.py`; geoops `plume/_src/array.py`, `geom/_src/operators.py`; geoproducts `himawari/constants.py`; radtrax `atmosphere/gas.py`; plumax `coupled/rtm.py`, `hapi_lut/beers.py`, `met/wrf.py`.
