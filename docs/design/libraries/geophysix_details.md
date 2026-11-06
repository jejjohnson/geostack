---
title: "geophysix: design details"
short_title: "geophysix details"
description: "The implementation reference behind the geophysix design: the records and their invariants, the formulas step by step, JAX behaviour, tests, dependencies, and the survey of what each library holds today."
date: 2026-11-05
keywords:
  - geophysix
  - design
  - details
  - GeoStack
---

The implementation reference behind the [geophysix design](geophysix.md): the records and their invariants, the formulas step by step, JAX behaviour, tests, dependencies, and the survey of what each library holds today.

## Core types

Two kinds of record: a set of physical constants, and a body (an ellipsoid, of which a sphere is the case f = 0). Both are frozen dataclasses of floats, not `eqx.Module`s, because geoarray must install and run without JAX (gridax decision 6) and geoarray reaches geophysix through gridax. Under JAX a float field is a compile-time constant; a caller who needs a gradient with respect to a constant (fitting an Earth radius, say) passes a JAX scalar in that field, and the record is registered as a pytree when JAX is importable.

```python
# Shapes: every field is a scalar (float or 0-d array); no array dims

import dataclasses
import math


# proposed (geophysix)
@dataclasses.dataclass(frozen=True)
class Constants:
    """One convention's physical constants, SI throughout.

    Attributes:
        boltzmann: k_B, J K⁻¹.
        avogadro: N_A, mol⁻¹.
        planck: h, J s.
        speed_of_light: c, m s⁻¹.
        standard_gravity: g₀, m s⁻².
        standard_pressure: p₀, Pa.
        dry_air_molar_mass: M_dry, kg mol⁻¹.
        dry_air_heat_capacity: c_p, J kg⁻¹ K⁻¹.
    """

    boltzmann: float = 1.380649e-23
    avogadro: float = 6.02214076e23
    planck: float = 6.62607015e-34
    speed_of_light: float = 299_792_458.0
    standard_gravity: float = 9.80665
    standard_pressure: float = 101_325.0
    dry_air_molar_mass: float = 0.0289644
    dry_air_heat_capacity: float = 1004.5

    @property
    def gas_constant(self) -> float:
        # R = k_B N_A
        return self.boltzmann * self.avogadro

    @property
    def dry_air_gas_constant(self) -> float:
        # R_d = R / M_dry
        return self.gas_constant / self.dry_air_molar_mass


# proposed (geophysix)
@dataclasses.dataclass(frozen=True)
class Gas:
    """A trace gas by name and molar mass M, kg mol⁻¹."""

    name: str
    molar_mass: float


# proposed (geophysix)
@dataclasses.dataclass(frozen=True)
class Ellipsoid:
    """An ellipsoid of revolution: semi-major axis a, flattening f, rotation Ω."""

    semi_major: float
    flattening: float
    rotation_rate: float = 7.292115e-5

    def __post_init__(self) -> None:
        if not self.semi_major > 0.0:
            raise EarthError(f"semi_major must be > 0, got {self.semi_major}")
        if not 0.0 <= self.flattening < 1.0:
            raise EarthError(f"flattening must be in [0, 1), got {self.flattening}")

    @property
    def eccentricity_squared(self) -> float:
        # e² = f (2 − f)
        return self.flattening * (2.0 - self.flattening)

    @property
    def semi_minor(self) -> float:
        # b = a (1 − f)
        return self.semi_major * (1.0 - self.flattening)

    @property
    def mean_radius(self) -> float:
        # R₁ = (2a + b) / 3, the IUGG mean radius
        return (2.0 * self.semi_major + self.semi_minor) / 3.0


# proposed (geophysix)
def sphere(radius: float, rotation_rate: float = 7.292115e-5) -> Ellipsoid:
    # a sphere is the ellipsoid with f = 0
    return Ellipsoid(semi_major=radius, flattening=0.0, rotation_rate=rotation_rate)


# proposed (geophysix)
STANDARD = Constants()
# proposed (geophysix)
WGS84 = Ellipsoid(semi_major=6_378_137.0, flattening=1.0 / 298.257223563)
# proposed (geophysix)
GRS80 = Ellipsoid(semi_major=6_378_137.0, flattening=1.0 / 298.257222101)
# proposed (geophysix)
# R₁ of WGS84 = 6,371,008.77 m
SPHERE = sphere(WGS84.mean_radius)
# proposed (geophysix)
CH4 = Gas("CH4", 0.0160425)
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `Constants` | CODATA 2018 exact constants (k\_B, N\_A, h, c), g₀, p₀, dry-air M and c\_p | frozen; SI; derived values (R, R\_d, κ = R\_d / c\_p) are properties, never stored, so they cannot drift from their parts |
| `Gas` | a name and a molar mass | M > 0; one instance per gas in the module (`CH4`, `CO2`, `H2O`, `N2O`, `CO`) |
| `Ellipsoid` | a, f, Ω | a > 0, 0 ≤ f < 1 (checked at construction, raising `EarthError`); a sphere is f = 0 |
| `SeawaterEOS` (wave 3) | ρ₀, α, β\_S, T₀, S₀ of a linear equation of state | ρ₀ > 0; α, β\_S ≥ 0 |

Errors subclass one base, `GeophysixError(ValueError)`: `EarthError` for a bad body, `UnitError` for an unknown column unit, `RangeError` for a latitude outside \[−90°, 90°\] when a function checks it.

## Types to adhere to

Functions read only the attributes they need, so any object with them works; the protocols say which:

| Protocol | Attributes a caller's object must have | Read by |
| --- | --- | --- |
| `EarthLike` | `semi_major`, `flattening`, and for rotation `rotation_rate` | `earth.*`, `rotation.*` |
| `ConstantsLike` | the `Constants` fields a function uses | `thermo.*`, `columns.*` |
| `GasLike` | `molar_mass` | `columns.*` |

A model on another planet, or a test that wants round numbers, passes its own record:

```python
# user
mars = geophysix.earth.Ellipsoid(semi_major=3_396_200.0, flattening=1.0 / 169.8)
# proposed (geophysix)
# f = 2 Ω sin φ on Mars
coriolis = geophysix.rotation.coriolis(lat, earth=mars)  # (N,) → (N,)
```

Arrays follow the array-namespace contract: a function takes NumPy or JAX arrays, computes with the namespace of its first array argument (`pipekit_array.array_namespace`, as geoops' stencils already do), and returns an array of the same kind and dtype. Core libraries, which cannot import geophysix, take the number they need (`radius=SPHERE.semi_major`) rather than the record.

## Algorithms

Every algorithm here is closed form or a fixed number of steps, so each is a pure function that traces once.

### Geodetic to Earth-centred coordinates, and back

With N(φ) the prime-vertical radius of curvature, the forward map from longitude λ, latitude φ and height h is

```{math}
N(\varphi) = \frac{a}{\sqrt{1 - e^2 \sin^2\varphi}}
```

```{math}
(X, Y, Z) = \left( (N + h) \cos\varphi \cos\lambda, (N + h) \cos\varphi \sin\lambda, (N (1 - e^2) + h) \sin\varphi \right)
```

The inverse has no closed form in φ. geophysix uses Bowring's method, as geonnax does today:

1. λ = atan2(Y, X) and p = √(X² + Y²).
2. Parametric latitude β₀ = atan2(a Z, b p).
3. φ = atan2(Z + e'² b sin³β₀, p − e² a cos³β₀), with e'² = e² / (1 − e²).
4. h = p / cos φ − N(φ), or (Z / sin φ) − N (1 − e²) near the poles.

One Bowring step is accurate to well under a millimetre for heights within ±10 km; a second fixed step covers orbital heights. Cost: a handful of trigonometric calls per point.

Pitfalls: an Earth-centred coordinate of 6.4 × 10⁶ m has a float32 spacing of 0.5 m, so sub-metre work needs float64; at the poles p → 0 and step 4 switches branch, which needs a double `where` so the gradient of the unused branch is not NaN.

### Cell area on the ellipsoid

With s = sin φ and q the authalic function, the area between latitudes φ₁ and φ₂ across Δλ of longitude is

```{math}
q(\varphi) = \frac{s}{1 - e^2 s^2} + \frac{\operatorname{artanh}(e s)}{e}
```

```{math}
A = \frac{b^2 \Delta\lambda}{2} \left( q(\varphi_2) - q(\varphi_1) \right)
```

1. Evaluate q at each edge latitude (Ny + 1 values for Ny rows).
2. Difference along latitude, multiply by b² Δλ / 2 per column.

Cost O(Ny + Nx), then an outer product for the grid. Pitfalls: written as (1 / 2e) ln((1 + es) / (1 − es)), the second term cancels badly for small e; artanh(es) / e does not, and as e → 0 it is replaced by its limit s behind a double `where`, which recovers the sphere's A = R² Δλ (sin φ₂ − sin φ₁) exactly and keeps the gradient finite. The sum over the globe is 4π Rₚ² with Rₚ the authalic radius, which is a test.

### Coriolis and β-planes

```{math}
f = 2 \Omega \sin\varphi
```

```{math}
\beta = \frac{2 \Omega \cos\varphi}{R}
```

A β-plane at reference latitude φ₀ is f(y) = f₀ + β₀ (y − y₀). finitevolX's `coriolis_fn` defaults, f₀ = 9.375 × 10⁻⁵ s⁻¹ and β = 1.754 × 10⁻¹¹ m⁻¹ s⁻¹, are this pair at 40° N. Pitfalls: degrees or radians, decided by an argument and never guessed; on the ellipsoid R becomes the meridional radius R\_M(φ).

### Hypsometric thickness

With T\_v = T (1 + 0.61 q\_v) the virtual temperature (q\_v the specific humidity) and T̄\_v its mean in ln p over the layer,

```{math}
z_2 - z_1 = \frac{R_d \bar T_v}{g} \ln \frac{p_1}{p_2}
```

1. Virtual temperature at each level.
2. Layer mean in ln p (the average of the two levels to second order).
3. Thickness per layer; heights are its cumulative sum from the surface, an integrix scan.

Pitfalls: pressure in Pa against hPa, and level order (top-down in most reanalyses); the function takes the order as an argument and checks monotonicity outside `jit`.

### Column units

With X a column enhancement in ppm·m, V\_m = R T / p the molar volume and M the gas's molar mass:

```{math}
\Omega_{\mathrm{mol}} = \frac{10^{-6} X}{V_m}
```

```{math}
\Omega_{\mathrm{kg}} = M \Omega_{\mathrm{mol}}
```

```{math}
N_{\mathrm{dry}} = \frac{N_A}{M_{\mathrm{dry}}} \left( \frac{p_s}{g} - W \right)
```

with W the water-vapour column in kg m⁻²; molecules per m² are N\_A Ω\_mol, and XCH₄ = N\_CH₄ / N\_dry. Pitfall: geoops fixes V\_m = 0.024465 m³ mol⁻¹ (25 °C, 1 atm), which is about 9 % off at 0 °C (22.414 litres per mole); geophysix takes p and T, with that value only as an explicit default.

### Equation of state of sea water

```{math}
\rho = \rho_0 \left( 1 - \alpha (T - T_0) + \beta_S (S - S_0) \right)
```

The linear form first, with finitevolX's defaults (ρ₀ = 1025 kg m⁻³, α = 2 × 10⁻⁴ K⁻¹, β\_S = 7 × 10⁻⁴, T₀ = 10 °C, S₀ = 35). TEOS-10 follows as the 75-term polynomial for specific volume of Roquet et al. (2015), which is differentiable and cheap, tested against gsw.

## JAX compatibility

| Function | jit | vmap | grad | Notes |
| --- | --- | --- | --- | --- |
| `earth.geodetic_to_ecef` | ✓ | ✓ | ✓ | closed form |
| `earth.ecef_to_geodetic` | ✓ | ✓ | ✓ | fixed Bowring steps; double `where` at the poles |
| `earth.cell_area` | ✓ | ✓ | ✓ | double `where` at e = 0 |
| `earth.geodesic` (decision 9) | ✓ | ✓ | to check | Karney's inverse iterates; a fixed count under `jit`, coincident and antipodal points to test |
| `rotation.coriolis`, `beta`, `beta_plane` | ✓ | ✓ | ✓ |  |
| `thermo.*` | ✓ | ✓ | ✓ |  |
| `seawater.linear_density` | ✓ | ✓ | ✓ |  |
| `seawater.teos10_density` (wave 3) | ✓ | ✓ | ✓ | polynomial |
| `columns.convert` | ✓ | ✓ | ✓ | the units are static strings |

- **Static or traced.** Record fields are Python floats by default and so compile in as constants; a JAX scalar in a field is traced, which is how a radius or a molar mass is fitted. Unit names, the level order and `degrees=` flags are static arguments.
- **Dtypes.** Every function preserves its input's dtype; constants enter as Python scalars or as `xp.asarray(c, dtype=x.dtype)`, never as float64 arrays that would promote a float32 caller. Geodetic conversions document that float32 limits Earth-centred precision to about half a metre.
- **Array namespace.** Each function computes with the namespace of its first array argument, so NumPy in gives NumPy out without importing JAX, which is what geoarray needs through gridax.

## Testing

Every public function gets the four tests of the house style.

1. **Oracles.** pyproj: `Transformer` between EPSG:4979 and EPSG:4978 for geodetic and Earth-centred coordinates, `Geod` for geodesics and polygon areas, at `rtol=1e-12` in float64. CODATA 2018 for the constants, compared exactly. The closed forms against their limits: the cell area of the whole sphere is 4πR², of the ellipsoid 4πRₚ². gsw for TEOS-10 densities.
2. **Properties** (hypothesis). Geodetic to ECEF and back returns the input to 10⁻⁹ ° and 10⁻⁴ m for |h| < 10 km; cell areas are positive and additive under splitting a cell; f(−φ) = −f(φ); column conversions are inverses of each other.
3. **Gradients.** `jax.jacrev` against central differences, including at the poles for `ecef_to_geodetic` and at f = 0 for `cell_area`.
4. **Transforms.** Eager, `jit` and `vmap` agree with a Python loop; NumPy and JAX inputs give the same numbers.

A cross-library test pins the consumers' values once they switch: finitevolX's and sesmax-core's results with R = 6,371,200 m against R₁, at the 3 × 10⁻⁵ relative tolerance of decision 4.

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| NumPy | the default array namespace | required |
| pipekit-array | `array_namespace`, the NumPy and JAX dispatch | required |
| JAX | traced use, gradients | optional (`[jax]`) |
| pyproj | the oracle in tests only | test |
| gsw | the TEOS-10 oracle in tests only | test |
| hypothesis | property tests | test |

pipekit-array is Core, so geophysix still imports nothing from GeoNumerics or GeoStack. Whether a dependency on pipekit-array is acceptable at GeoNumerics tier 0, or geophysix carries its own five-line dispatch, is a small decision for wave 0.

## Where it stands

Read on 5 Nov 2026 at: finitevolX `82090a3` (0.1.0), somax `aea3a03`, sesmax `314de57`, xrtoolz `b3855d7`, geotoolz `d2f53f4`, geonnax `9579986`, kernellib `98fbbc7`, gaussx `7fa1a38`, radtrax `ed04971`, plumax `28f257e`.

### The values side by side

| Quantity | Value | Where |
| --- | --- | --- |
| Earth radius | 6,371,200 m | finitevolX `R_EARTH`; somax and sesmax-core, the same file |
|  | 6,371,000 m | xrops `grad` `EARTH_RADIUS`; gaussx `haversine_distance(radius=6.371e6)` |
|  | 6,371,008.8 m | geonnax `EARTH_MEAN_RADIUS_M`; kernellib `EARTH_RADIUS_KM = 6371.0088`; geoops `_EARTH_MEAN_RADIUS_KM` |
|  | 6,378,137 m (semi-major, used as a radius) | geoops parallax `earth_radius_m`; geoproducts Himawari `EARTH_SEMI_MAJOR_M` |
| Gravity | 9.80665 m s⁻² | finitevolX, somax, sesmax-core, xrops `grad` and `atm` |
|  | 9.81 m s⁻² | plumax `met/wrf.py` |
| Rotation rate | 2π / 86,164 s = 7.29212 × 10⁻⁵ s⁻¹ | finitevolX, somax, sesmax-core |
|  | 7.2921159 × 10⁻⁵ s⁻¹ | xrops `grad` |
| Dry air | R\_d = 287.04, c\_p = 1004.5, M\_dry = 0.0289644 | xrops `atm`; plumax `met/wrf.py` (κ as 287.04 / 1004.5) |
| CH₄ molar mass | 0.0160425 kg mol⁻¹ | plumax `coupled/rtm.py` |
|  | 0.01604 kg mol⁻¹ | geoops `MOLAR_MASS_KG_PER_MOL` |
| k\_B, 1 atm | 1.380649 × 10⁻²³ J K⁻¹, 101,325 Pa | radtrax `atmosphere/gas.py`; plumax `hapi_lut/beers.py` and `radtran/config.py`; xrops `atm` (k\_B only) |

The radii differ by up to 3 × 10⁻⁵ relative between mean radii and by 1.1 × 10⁻³ when the semi-major axis stands in for a mean radius; the two CH₄ masses by 1.6 × 10⁻⁴. The 6,371,200 m radius is the one somax inherited and finitevolX and sesmax-core copied; it is the geomagnetic reference radius of the IGRF, not a geodetic mean radius of the Earth.

### Per library

- **finitevolX** `_src/utils/constants.py`: `R_EARTH`, `GRAVITY`, `OMEGA`, `DEG2M`, `RHO = 1000`. `_src/operators/diagnostics.py`: `coriolis_param(lat, omega=OMEGA)` and the β-plane `coriolis_fn(Y, f0=9.375e-5, beta=1.754e-11)`. `_src/operators/eos.py`: `linear_density`, `linear_density_anomaly` (ρ₀ = 1025, α = 2 × 10⁻⁴, β = 7 × 10⁻⁴, T₀ = 10, S₀ = 35). The constants, both Coriolis functions and the linear EOS move; the C-grid Coriolis operator in `_src/pipekit/stencils.py` stays and calls geophysix.
- **somax** `_src/constants/__init__.py` and **sesmax-core** `_src/constants.py`: the same five constants as finitevolX. sesmax-core switches; somax retires.
- **xrops** `grad/_src/constants.py` (written to replace metpy's), `atm/_src/constants.py` (`R_D`, `C_P`, `KAPPA`, `G`, `M_DRY`, `N_A`, `K_B`, `EPSILON_TV = 0.61`), `atm/gas/ch4/_src/column.py` (`dry_air_column`, `mixing_ratio_to_column`, `column_mass_to_delta_vmr`), `ocn/_src/kinematics.py` (`density_from_ts`, TEOS-10 through a lazy gsw import). The maths moves; the DataArray functions stay and call it.
- **geonnax** `_frames.py`: `WGS84_A`, `WGS84_F`, `EARTH_MEAN_RADIUS_M`; `lonlat_to_ecef` and `ecef_to_lonlat` with `ellipsoid="wgs84" | "sphere"`, the inverse by a Bowring-type step. The ellipsoid half moves; the unit-sphere frames go to functax.
- **kernellib** `functional/_geo.py` and **gaussx** `_inference/_localization.py`: `great_circle_distance`, `chordal_distance`, `haversine_distance`. They stay in Core with a radius argument; gaussx's default (6,371,000 m) moves to R₁ so all three agree, and a test says so (Design decision 5).
- **geoops** `plume/_src/array.py`: `convert_column_units`, NumPy only, casting to float64, with V\_m fixed at 25 °C and 1 atm; `geom/_src/operators.py`: the two radii above. **geoproducts** `himawari/constants.py` and `_src/geostationary.py`: the semi-major and semi-minor axes of the geostationary projection, which stay with the readers as projection parameters but should read WGS84 from geophysix.
- **radtrax** `atmosphere/gas.py`: `BOLTZMANN_J_PER_K`, `ATM_TO_PA`, `number_density(p_atm, T_K)`. **plumax** `coupled/rtm.py`, `hapi_lut/beers.py`, `radtran/config.py`: the same constants twice more and `number_density` again; `met/wrf.py`: its own `GRAVITY = 9.81` and κ. All move to `geophysix.thermo` and `geophysix.constants`.

## Sources

- Bowring, B. R. (1976). Transformation from spatial to geographical coordinates. *Survey Review* 23, 323–327.
- Moritz, H. (2000). Geodetic Reference System 1980. *Journal of Geodesy* 74, 128–133: GRS80 and the IUGG mean radius R₁.
- Roquet, F., Madec, G., McDougall, T. J. and Barker, P. M. (2015). Accurate polynomial expressions for the density and specific volume of seawater using the TEOS-10 standard. *Ocean Modelling* 90, 29–43.
- Snyder, J. P. (1987). *Map Projections: A Working Manual*. USGS Professional Paper 1395: the authalic latitude and the ellipsoidal area formula.
