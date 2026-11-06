---
title: "xrcf: design details"
short_title: "xrcf details"
description: "The implementation reference behind the xrcf design: the host-side types, what readers and operators must pass in, the algorithms with their pitfalls, the host-side contract, tests, dependencies, and a survey of the code that moves in."
date: 2026-11-05
keywords:
  - xrcf
  - design
  - details
  - GeoStack
---

The implementation reference behind the [xrcf design](xrcf.md): the host-side types, what readers and operators must pass in, the algorithms with their pitfalls, the host-side contract, tests, dependencies, and a survey of the code that moves in.

## Core types

Every type is a host-side frozen dataclass: hashable, comparable, and never traced. The fields below are the ones in xrproducts.types at b3855d7; the move keeps them.

```python
@dataclass(frozen=True)
class Variable:
    name: str  # canonical short name: "sst", "u", "u10"
    standard_name: str | None  # CF: "sea_surface_temperature", "eastward_wind"
    long_name: str
    units: str  # UDUNITS string: "K", "m s-1"
    aliases: Mapping[str, str] = field(hash=False)  # source -> request id, read-only
    valid_range: tuple[float, float] | None
    dtype: str | None
    cmap: str | None

    def for_source(self, source: str) -> str: ...  # the id a provider expects
    def cf_attrs(self) -> dict[str, str]: ...  # attrs to stamp on a DataArray


REGISTRY: Mapping[str, Variable]  # 61 variables today


def resolve(name: str | Variable) -> Variable: ...  # KeyError on an unknown name
def register(variable: Variable) -> None: ...


@dataclass(frozen=True)
class BBox:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float


@dataclass(frozen=True)
class TimeRange:
    start: pd.Timestamp
    end: pd.Timestamp
    freq: str | None


@dataclass(frozen=True)
class DepthRange:
    min: float
    max: float  # metres, positive down


@dataclass(frozen=True)
class PressureLevels:
    levels: tuple[int, ...]  # hPa, positive


@dataclass(frozen=True)
class Request:
    variables: tuple[str, ...]
    bbox: BBox
    time: TimeRange
    depth: DepthRange | None
    levels: PressureLevels | None
    extras: Mapping[str, Any]


class Severity(StrEnum):
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Issue:
    variable: str
    severity: Severity
    code: str
    message: str


@dataclass
class ValidationReport:
    issues: list[Issue]

    def add(self, issue: Issue) -> None: ...

    ok: bool
    errors: list[Issue]
    warnings: list[Issue]

    def raise_if_errors(self) -> None: ...
```

| Type | Holds | Invariants |
| --- | --- | --- |
| `Variable` | one physical quantity: canonical name, CF standard name, units, per-source request ids, valid range | `name` unique in the registry; `aliases` maps a source ("cmems", "cds", "wrf") to that provider's request id and is read-only (`MappingProxyType`); `units` parses as UDUNITS |
| `REGISTRY` | the canonical variables | keys equal each value's `name`; `register` refuses a clash |
| `BBox` | a lon/lat box in degrees | −90 ≤ lat\_min ≤ lat\_max ≤ 90; `lon_min > lon_max` means the box crosses the antimeridian; `to_180` / `to_360` change only the convention |
| `TimeRange` | a closed time interval and an optional frequency | start and end are UTC-aware and start ≤ end |
| `DepthRange` | an ocean depth interval | 0 ≤ min ≤ max, metres positive down |
| `PressureLevels` | a set of pressure levels | positive hPa, unique |
| `Request` | what a reader is asked for | every name in `variables` resolves |
| `Station` / `StationCollection` | point sites: id, name, lon, lat, altitude, WMO id, source, attrs | ids unique within a collection |
| `Issue` / `ValidationReport` | the result of a validation | a missing attribute is a warning, a mismatch an error; `ok` means no errors |

`ValidationReport` is the one mutable type: it is built up by `add` during a single validation and returned, never shared.

## Types to adhere to

Two kinds of caller use xrcf, and each has one rule.

**A reader** (an xrproducts source, a geocatalog loader) takes a `Request` and returns a Dataset in canonical form:

1. For each requested name, ask the provider for `resolve(name).for_source("<source>")`, its id for that variable.
2. Rename what comes back to the canonical `name`, so `thetao` from CMEMS and `sea_surface_temperature` from CDS both arrive as `sst`.
3. Normalise coordinates (`validate_longitude`, `validate_latitude`, `decode_cf_time`) and stamp attrs with `apply_cf_attrs`.

**An operator** that creates a quantity (wind speed from u and v, a column average) names its output after a registry variable and stamps that variable's attrs:

```python
speed = np.hypot(ds["u"], ds["v"])  # (time, lat, lon) float32
speed = apply_cf_attrs(speed.rename("wind_speed"), resolve("wind_speed"))
```

A quantity not yet in the registry is registered first (`register(Variable(...))`), not stamped with hand-written attrs. A downstream project that needs a source the registry does not know builds its own copy of the variable with `dataclasses.replace` and a wider `aliases`, never by mutating the registry's.

## Algorithms

### Resolving a variable

1. If the argument is a `Variable`, return it.
2. Look the string up in `REGISTRY` by canonical name.
3. Otherwise raise `KeyError` naming the closest canonical names.

`for_source(source)` returns `aliases[source]` when present and the canonical name otherwise. Pitfall: aliases are request ids for one provider, not synonyms; looking a provider's id up across all sources ("which variable is `U`?") is ambiguous, because two sources can use one id for different quantities. Renaming a file's variables by meaning goes through their `standard_name` attribute instead (`rename_to_cf_standard_names`), which checks for collisions.

### Longitude convention

Wrapping to \[−180, 180):

```{math}
\lambda' = ((\lambda + 180) \bmod 360) - 180
```

1. Rename the longitude coordinate to `lon` (from `longitude`, `x`, `nav_lon`, …).
2. Apply the wrap.
3. If the coordinate is 1-D and the wrap changed its order, `sortby("lon")`.

Pitfall: today's `validate_longitude` wraps but does not sort, so a 0–360 global grid comes back with its western half after its eastern half, and a `sel` slice across the seam returns nothing. Sorting copies the data along that axis; for dask arrays it is a lazy reindex, but it changes the chunking.

### Latitude order

1. Rename the latitude coordinate to `lat`.
2. Check the range \[−90, 90\]; a value outside it is an error, not something to wrap (today's code wraps it, which hides a swapped lon/lat).
3. Leave the order as it is, and record it: `BBox.as_xarray_sel(lat_descending=True)` already builds the reversed slice ERA5's north-to-south grids need. Reordering to ascending is an opt-in step (`sortby("lat")`), because flipping a north-up raster changes the sign of its transform for every geospatial consumer downstream.

### Time decoding

1. Decode with `xr.decode_cf` (CF `units: "<unit> since <epoch>"` and `calendar`).
2. Rename to `time`.
3. Convert to UTC-naive `datetime64[ns]` when the calendar is standard; keep cftime objects for 360-day and no-leap calendars rather than coercing them.

Pitfall: model output on a non-standard calendar cannot be converted to `datetime64` without dropping or inventing days; `TimeRange.to_index` must refuse to compare it with a standard calendar, not silently shift.

### Grid mapping

A CF grid mapping is a scalar variable whose attrs describe the CRS, referenced by each data variable's `grid_mapping` attribute. rioxarray's `write_crs` writes exactly this, as a variable called `spatial_ref` that carries `crs_wkt` and the CF projection parameters.

1. From a `pyproj.CRS`, build the attrs with `CRS.to_cf()` (CF parameters plus `crs_wkt`).
2. Add a scalar variable holding them and set `grid_mapping` on every variable with the spatial dimensions.
3. Reading back, `CRS.from_cf(attrs)` recovers the CRS.

With pyproj alone this needs no rioxarray; the choice is only whether to call the variable `spatial_ref` (so rioxarray readers find it) or `crs`.

### Validating a variable against the registry

For a DataArray and its `Variable`:

1. `standard_name` attr missing → warning; present and different → error.
2. `units` missing → warning; different → error. Compare parsed units, so `m/s` and `m s-1` agree.
3. When `valid_range` is set: the finite minimum and maximum, computed lazily on dask, must lie in it; otherwise error.
4. When `check_dtype` is on: dtype differs → warning.

`validate_dataset` runs this over every variable whose name resolves and collects one `ValidationReport`. Pitfall: the range check is a full pass over the data; on a large dask array it is the one expensive step, which is why it can be turned off.

### Subsetting by a box

1. If `bbox.lon_min ≤ bbox.lon_max`, select one longitude interval.
2. If `bbox.lon_min > bbox.lon_max` (crossing the antimeridian), split it into \[lon\_min, 180) and \[−180, lon\_max\], select each, and concatenate along `lon`, in that order so longitude stays increasing across the seam in the 0–360 sense.
3. On 1-D coordinates use `sel` slices (with the reversed latitude slice for descending grids); on 2-D curvilinear coordinates use a `where(mask, drop=True)` mask.

Pitfall: today's `subset_bbox` does step 3 only and treats a crossing box as empty, and the `as_cmems` / `as_cds_area` serialisers raise on it. The split must happen before a provider request too, as two requests merged afterwards.

## Host-side contract

- **No JAX.** xrcf imports neither JAX nor Equinox, and none of its types is a pytree. It runs before and after the numerics, on the host, and its types are dictionary keys and cache keys (frozen and hashable).
- **Lazy where the data are.** Renames, attribute stamping, coordinate wraps and grid mappings touch metadata and coordinates only, so a dask-backed Dataset stays lazy. The steps that read data are named: the range check in validation (one pass, can be turned off), and `sortby` after a longitude wrap (a lazy reindex that changes chunks).
- **Eager by design.** Registry lookups, `BBox` / `TimeRange` validation and the provider serialisers (`as_cmems`, `as_cds_area`, `as_cds_form`) are pure Python on small values.
- **Never mutates inputs.** Every function returns a new Dataset or DataArray; `apply_cf_attrs(..., overwrite=False)` keeps attrs a file already has.
- **Errors are specific.** An unknown variable is a `KeyError`, an invalid box or interval a `ValueError` at construction, a failed validation a `ValidationReport` (and `raise_if_errors` raises one `ValueError` listing every error).

## Testing

- **Parity with the code it replaces.** Until the old modules are deleted, every moved function gives identical output to `xrproducts.types` and `xrops.cf` at b3855d7 on the existing test fixtures.
- **Registry consistency.** Every `Variable` has a valid UDUNITS `units`; every `standard_name` is in the CF standard-name table (a pinned copy in the tests); names are unique; `cf_attrs()` round-trips through `validate_variable` with no issues.
- **CF compliance.** A Dataset produced by a reader and then normalised passes `cfchecker` (or `cf_xarray`'s checks) with no errors, in the integration tier.
- **Box properties.** With Hypothesis: `to_180` then `to_360` returns the same box; a crossing box's two halves cover the same longitudes as the original and do not overlap; `subset_bbox` on a global grid returns exactly the cells whose centres lie in the box, for crossing and non-crossing boxes.
- **Conventions.** A 0–360 grid wrapped and sorted equals the same grid built on −180–180; descending latitudes survive `validate_latitude` unchanged and `as_xarray_sel(lat_descending=True)` selects the same cells as on the ascending copy.
- **Laziness.** Normalising a dask-backed Dataset triggers no compute (a dask callback that counts tasks executed, asserted zero) except in the named steps.
- **Grid mapping round-trip.** `CRS.from_cf` of the written attrs equals the original CRS for geographic, UTM, Lambert conformal and geostationary projections.

## External dependencies

| Library | Used for | Required or optional |
| --- | --- | --- |
| xarray | the Datasets and DataArrays every function takes | required |
| pandas | `TimeRange` timestamps and indices | required |
| NumPy | coordinate arithmetic | required |
| pyproj | `CRS.to_cf` / `from_cf` for grid mappings | required |
| cftime | non-standard calendars | optional (pulled in by xarray when needed) |
| dask | lazy range checks | optional |
| rioxarray | not needed; xrops' reprojection keeps it | none |
| cf\_xarray | CF compliance checks | tests only |
| regionmask, shapely | region masks | stays with xrops' `subset_to_region`, not xrcf |

## Where it stands

Nothing named xrcf exists yet. The code that moves in sits in two places in **xrtoolz b3855d7**:

| Source | Lines | What it holds | Moves to |
| --- | --- | --- | --- |
| `xrproducts/types/_src/variable.py` | 1414 | `Variable`, `REGISTRY` (61 variables), `resolve`, `register` | xrcf registry |
| `xrproducts/types/_src/geometry.py` | — | `BBox` with antimeridian flag, `to_180` / `to_360`, `width`, `as_cmems`, `as_cds_area`, `as_xarray_sel` | xrcf types |
| `xrproducts/types/_src/levels.py`, `time.py`, `request.py` | — | `DepthRange`, `PressureLevels`, `TimeRange` (UTC-aware; `parse`, `to_index`, `as_cmems`, `as_cds_form`, `cds_times`), `Request` | xrcf types |
| `xrproducts/types/_src/subset.py` | — | `subset_bbox` (mask-based, 2-D-aware, no antimeridian handling), `subset_where`, `subset_time` | xrcf subset, gaining the antimeridian split |
| `xrproducts/types/_src/validation.py` | — | `Severity`, `Issue`, `ValidationReport`, `validate_variable`, `validate_dataset`, `apply_cf_attrs` | xrcf validation |
| `xrproducts/types/_src/station.py` | — | `Station`, `StationCollection` | xrcf types |
| `xrops/cf/_src/validation.py` | 396 | `validate_longitude` (wraps, does not sort), `validate_latitude` (renames, wraps, does not check order), `rename_coords`, `rename_variables`, `decode_cf_time`, `validate_time`, `check_dataset_coords`, `rename_to_cf_standard_names` / `rename_from_cf_standard_names` (by attribute, with a lazy import of the registry) | xrcf conventions |

The whole of `xrproducts/types/_src` is 2350 lines. Eight modules in xrops import xrproducts today: `cf` twice (lazily, for the registry), `atm/wrf`, `atm/wind`, `atm/vertical`, `atm/gas/ch4/column`, `viz/cmaps` and `geom/subset`. Each of those edges becomes an import of xrcf, and xrops' dependency on xrproducts goes away once none of them needs anything beyond the types (to be checked edge by edge).

Left where it is: `xrops/geom/_src/crs.py` (568 lines: `assign_crs` through rioxarray, `reproject`, `reproject_match`, `lonlat_to_xy`, `utm_crs_for`, `LocalFrame`) is geometry, not conventions, and keeps rioxarray; xrcf takes only the grid-mapping attributes. geocatalog's `GeoSlice` (geotoolz d2f53f4) holds bounds in CRS units with a `pd.Interval`; it is a different query type and stays in geocatalog, with a conversion from `BBox` and `TimeRange` as the one meeting point.
