---
title: "xrcf: conventions for the xarray side of GeoStack"
short_title: "xrcf"
description: "xrcf is the floor of the xarray side of GeoStack, as geoarray is of the GeoArray side: the CF variable registry, the conventions for names, attributes, coordinates and time, the CRS as a CF grid mapping, validation, the typed vocabulary of a request (BBox, TimeRange, levels), and subsetting by it. It computes nothing. It is a proposed package of the xrtoolz workspace (distribution xrtoolz-cf), at its tier 0, below both the readers and the operators. Status: proposed, at design-sketch depth."
date: 2026-11-05
keywords:
  - xrcf
  - design
  - GeoStack
---

xrcf is the floor of the xarray side of GeoStack, as geoarray is of the GeoArray side: the CF variable registry, the conventions for names, attributes, coordinates and time, the CRS as a CF grid mapping, validation, the typed vocabulary of a request (BBox, TimeRange, levels), and subsetting by it. It computes nothing. It is a proposed package of the xrtoolz workspace (distribution `xrtoolz-cf`), at its tier 0, below both the readers and the operators. Status: proposed, at design-sketch depth.

## At a glance

|  |  |
| --- | --- |
| Layer, tier | GeoStack, xrtoolz workspace, tier 0 |
| Imports | xarray, pandas, NumPy, pyproj; nothing from GeoNumerics or Core |
| Imported by | xrops, xrproducts |
| Takes over | `xrproducts.types` (the `Variable` registry, the request vocabulary, subsetting, validation); `xrops.cf`; the CRS half of `xrops.geom` |
| Does not own | computing anything (xrops); reading archives (xrproducts); the carrier itself (xarray); reprojection (xrops, through rioxarray) |
| Status | proposed; no package yet |
| Tracked in | the [layer map](../stack_layer_map.md), "The gap: a floor for the xarray side" |

```{figure} ../../images/libraries/xrcf.svg
:alt: where xrcf sits · the floor of xrtoolz
:width: 100%

Where xrcf sits · the floor of xrtoolz.
```

With xrcf at the bottom, xrops and xrproducts become siblings over it, as geoops and geoproducts are over geoarray, and xrops no longer imports a reader to name a variable. This page is the design; core types, the registry and the code survey are in [xrcf: design details](xrcf_details.md).

## The problem

The methane example reads ERA5 winds from the Climate Data Store and a TROPOMI L2 file, and the two disagree on everything but the physics: ERA5 calls the eastward wind `u`, its global longitudes run 0 to 360 and its latitudes descend; TROPOMI has its own names and no CRS. Before xrops can compute anything, both are renamed, normalised, given a CRS, validated and cut to the plume's box:

| Step | The convention it needs | Where it is coded today |
| --- | --- | --- |
| Ask for the region, time and levels | a typed request | `xrproducts.types`: `Request`, `BBox`, `TimeRange`, `DepthRange`, `PressureLevels` |
| Rename to CF standard names | a registry of variables with per-source aliases | `xrproducts.types`: `Variable`, `REGISTRY` (61 variables), `resolve`, `register`, `apply_cf_attrs`; `xrops.cf`: `rename_to_cf_standard_names`, `rename_variables` |
| Normalise coordinates and time | longitude range, latitude order, CF time | `xrops.cf`: `validate_longitude`, `validate_latitude`, `check_dataset_coords`, `decode_cf_time`, `validate_time` |
| Attach a CRS | a grid mapping | `xrops.geom`: `assign_crs`, `get_crs`, `utm_crs_for`, `local_frame`, through rioxarray |
| Check the result | a validation report | `xrproducts.types`: `validate_variable`, `validate_dataset`, `ValidationReport` |
| Cut to the plume's box | subsetting by bbox, time and condition | `xrproducts.types`: `subset_bbox`, `subset_time`, `subset_where`; `xrops.geom.subset_to_region` |
| Name what an operator outputs | the same registry | `xrops` imports `Variable`, `apply_cf_attrs` or `REGISTRY` from `xrproducts.types` eight times: `atm` (`wrf`, `wind`, `vertical`), `atm/gas/ch4`, `viz/cmaps`, `cf`, `geom/subset` |

That last row is the problem. On the GeoArray side the carrier's vocabulary lives at the bottom, in geoarray, and geotoolz has a rule that geoops never depends on geoproducts. On the xarray side the same vocabulary is split between a reader and the operators, so xrops imports a reader package to name a variable, which is exactly the edge geotoolz forbids.

## A convention, written down

xrcf holds four kinds of object, each a frozen record or a pure function on a Dataset.

**A variable.** A `Variable` names a physical quantity once: its name in the stack, its CF standard name, units, long name, valid range, dtype and colour map, and its request identifier at each provider (`{"cmems": "thetao", "cds": "sea_surface_temperature"}`). The registry holds the curated ones, 61 today; users register their own. Every reader and every operator names a quantity through it, so the eastward wind (canonical name `u`, standard name `eastward_wind`) is one entry whether ERA5 delivered it or an operator computed it.

**A convention.** A Dataset is in convention when its variables carry standard names and units, longitudes lie in \[−180, 180), latitudes ascend, time is decoded to `datetime64`, and its spatial dims are `lon`/`lat`, or `x`/`y` with a CRS. Each rule is a check and a fix, so a reader can normalise what it opens and an operator can refuse what it cannot use.

Today's code checks more than it fixes: after wrapping longitudes it does not sort them, it does not reorder descending latitudes, and `subset_bbox` returns nothing for a box that crosses the antimeridian. xrcf adds those three fixes.

**A CRS.** Stored the CF way: a grid-mapping variable holding `crs_wkt`, referenced from each data variable's `grid_mapping` attribute, EPSG:4326 for lon/lat data. pyproj parses it; rioxarray reads the same attributes when xrops reprojects.

**A request.** Variables, a `BBox`, a `TimeRange`, and pressure levels or a depth range: the same object a reader fetches with and a subset cuts with, so what was asked for and what was kept cannot drift apart.

Validation returns a report of issues with a severity each, not an exception, so a pipeline decides what to stop on.

## Modules and boundaries

Six modules, all on xarray, pandas, NumPy and pyproj; none needs dask, rioxarray or JAX.

| Module | Owns | Comes from |
| --- | --- | --- |
| `xrcf.variables` | `Variable`, the registry, `register`, `resolve`, per-source aliases, `apply_cf_attrs` | `xrproducts.types` (`variable.py`) |
| `xrcf.conventions` | renaming to and from CF standard names; longitude range and latitude order; CF time decoding; coordinate checks | `xrops.cf`: `rename_to_cf_standard_names`, `rename_from_cf_standard_names`, `rename_coords`, `rename_variables`, `validate_longitude`, `validate_latitude`, `decode_cf_time`, `check_dataset_coords` |
| `xrcf.crs` | attaching and reading a CRS as a CF grid mapping; the UTM zone for a point; local tangent frames; lon/lat to x/y and back; the resolution of a dataset | `xrops.geom`: `assign_crs`, `get_crs`, `utm_crs_for`, `LocalFrame`, `local_frame`, `assign_local_xy`, `lonlat_to_xy`, `xy_to_lonlat`, `calc_latlon`, `get_dataset_resolution` |
| `xrcf.validation` | `Severity`, `Issue`, `ValidationReport`, `validate_variable`, `validate_dataset`, `validate_time` | `xrproducts.types` (`validation.py`); `xrops.cf.validate_time` |
| `xrcf.vocabulary` | `BBox`, `TimeRange`, `DepthRange`, `PressureLevels`, `Request`, `Station`, `StationCollection` | `xrproducts.types` |
| `xrcf.subset` | subsetting by a bbox, a time range or a condition; selecting variables | `xrproducts.types`: `subset_bbox`, `subset_time`, `subset_where`; `xrops.geom.select_variables` |

### What xrcf owns, and what it hands off

The test for every name: does it say what a Dataset means, or check or fix that it says it in the stack's convention, without computing a new field? Then xrcf owns it.

| Neighbour | xrcf owns | The neighbour owns | The line |
| --- | --- | --- | --- |
| xrops | names, conventions, the CRS, requests | everything that computes: reprojection, rasterising, regions, mosaics, footprints, every diagnostic | computing anything is xrops' |
| xrproducts | the vocabulary a reader uses | reading archives (CMEMS, CDS, AEMET), local L2 openers, inventories | a reader names what it reads through xrcf |
| xarray, rioxarray | the stack's conventions | the carrier; raster reprojection | xrcf writes CF attributes that rioxarray reads |
| geoarray | the xarray side's vocabulary | the GeoArray side's carrier and its coordinates | the two floors mirror each other; whether they share a request vocabulary is decision 6 |
| geocatalog | — | `GeoSlice` queries over a file index | the same open question |

### What moves in

| From | What | Its users then import |
| --- | --- | --- |
| `xrproducts.types` | the whole subpackage: the registry, the vocabulary, subsetting, validation | xrcf; xrproducts' readers and xrops both |
| `xrops.cf` | renaming, coordinate and time conventions, validation helpers | xrcf |
| `xrops.geom` | `crs.py` apart from `reproject` and `reproject_match`; `select_variables` from `subset.py` | xrcf; reprojection, rasterising, regions, mosaics, footprints and `subset_to_region` (which needs regionmask) stay in xrops |

## In use

ERA5 winds for the plume, from request to a Dataset xrops can compute on. Every xrcf name is proposed; the reader call is xrproducts'.

```python
# Shapes: T times; K pressure levels; Ny × Nx cells (the Dataset's dims)

import xrcf

# proposed (xrcf)
request = xrcf.vocabulary.Request(
    variables=("u", "v"),
    bbox=xrcf.vocabulary.BBox(lon_min=-3.5, lon_max=-2.5, lat_min=39.0, lat_max=40.0),
    time=xrcf.vocabulary.TimeRange(start, end),
    levels=xrcf.vocabulary.PressureLevels((850, 900, 950)),
)
# not pinned (xrproducts)
era5 = cds_source.fetch(request)  # → (T, K, Ny, Nx)

# proposed (xrcf)
# u → eastward_wind, v → northward_wind, through the registry's aliases
era5 = xrcf.conventions.to_standard_names(era5, source="cds")
# proposed (xrcf)
# longitudes into [−180, 180), latitudes ascending, time decoded
era5 = xrcf.conventions.normalise(era5)
# proposed (xrcf)
# a CF grid mapping holding EPSG:4326's WKT
era5 = xrcf.crs.assign(era5, "EPSG:4326")
# proposed (xrcf)
report = xrcf.validation.validate_dataset(era5)
# proposed (xrcf)
era5 = xrcf.subset.by_request(era5, request)  # → (T, K, Ny', Nx')
```

From here xrops computes: the wind at plume height, a reprojection to the instrument's grid, a diagnostic. Each output names its variable through the same registry.

## Integration across the stack

| Consumer | Uses today | With xrcf |
| --- | --- | --- |
| xrops | eight imports of `xrproducts.types`; its own `cf` and `geom` CRS helpers | xrcf for names, conventions, the CRS and requests; no import of xrproducts |
| xrproducts | `types` as a subpackage of its own | xrcf; its readers keep their aliases in the registry (decision 5) |
| geocatalog | `GeoSlice` | unchanged unless decision 6 lands |

The change is an import, and it removes xrops' only reason to import a reader:

```python
# Today:
# today (xrproducts)
from xrproducts.types import Variable, apply_cf_attrs

# After:
# proposed (xrcf)
from xrcf.variables import Variable, apply_cf_attrs
```

`xrproducts.types` keeps deprecated aliases for one release.

## Decisions and open questions

One decided, three leaning, four open. Each entry gives the position first, then the trade-off.

### 1. A floor for the xarray side

**Leaning: yes, one small package.** Proposed on the layer map: xrops and xrproducts both import it and neither imports the other, mirroring geoops and geoproducts over geoarray. The alternative, moving `types` into xrops, would make every reader import the operators.

### 2. Name and home: xrcf, in the xrtoolz workspace

**Leaning.** Distribution `xrtoolz-cf`, import `xrcf`, as a workspace member next to `xrtoolz-ops` and `xrtoolz-products`, so it versions and releases with them.

### 3. It computes nothing

**Decided (5 Nov 2026, on the layer map).** Computing anything stays in xrops, reading in xrproducts. That keeps xrcf's dependencies to xarray, pandas, NumPy and pyproj.

### 4. How the CRS is stored

**Open.** xrops attaches a CRS through rioxarray, whose `spatial_ref` coordinate already is a CF grid-mapping variable holding `crs_wkt`. So the convention is settled; the choice is the dependency and the name. Leaning: write the same CF attributes with pyproj alone, under rioxarray's name `spatial_ref`, so rioxarray reads them unchanged and xrcf does not import it.

### 5. Where a source's aliases live

**Open.** Today each `Variable` carries the identifiers of every source (`cmems`, `cds`, …), so the registry knows every reader. The alternative is for each reader to register its aliases when it is imported. Leaning: keep them in the registry, where one look shows how a variable is spelled everywhere; revisit when a third-party reader appears.

### 6. One request vocabulary with geocatalog?

**Open, on the layer map.** geocatalog's `GeoSlice` and xrcf's `Request` both say "this region, this time". Sharing one would need a home below both workspaces, which neither has. Leaning: keep them apart, with converters both ways, until a second user needs both.

### 7. Build on cf\_xarray?

**Open.** cf\_xarray already reads standard names, axes and grid mappings. Nothing in the stack uses it today. Leaning: use it inside xrcf where it saves code, but keep xrcf's API, so the registry and the stack's stricter conventions stay ours.

### 8. Validation reports, not exceptions

**Leaning.** Keep `xrproducts.types`' design: validation returns a `ValidationReport` of issues with severities, and a caller decides what to stop on.

## Roadmap

Four waves, each a GitHub issue epic in the xrtoolz repository.

| Wave | Delivers | Ends | Needs |
| --- | --- | --- | --- |
| 0. The package | `xrtoolz-cf` as a workspace member; CI; the capability index | — | — |
| 1. Registry and vocabulary | `xrproducts.types` moved whole into `xrcf.variables`, `xrcf.vocabulary`, `xrcf.validation`, `xrcf.subset`, with aliases left behind for one release | xrops' eight imports of xrproducts | 0 |
| 2. Conventions and CRS | `xrops.cf` and the CRS half of `xrops.geom` into `xrcf.conventions` and `xrcf.crs` (decision 4) | the convention code in xrops | 1 |
| 3. Consumers switch | xrops and xrproducts import xrcf; the aliases go | the old paths | 1, 2 |

Wave 1 is mostly a move: `xrproducts.types` already depends on nothing but xarray, pandas and NumPy.

## Sources

- [The GeoStack layer map](../stack_layer_map.md), "The gap: a floor for the xarray side": xrcf's place, imports and moves.
- [The CF Conventions](https://cfconventions.org/): standard names, units, coordinates and grid mappings.
- Code read on 5 Nov 2026 at xrtoolz `b3855d7`: `xrproducts/types/_src/` (`variable.py`, `geometry.py`, `levels.py`, `request.py`, `station.py`, `subset.py`, `time.py`, `validation.py`); xrops `cf/_src/validation.py`, `geom/_src/crs.py`, `geom/_src/subset.py`, and the eight call sites that import `xrproducts.types`.
