---
title: "geotoolz's skill shelf: steps by operator family, for two bands of users"
subtitle: "Design · the skill tree for a workspace at the step rung"
short_title: Skill tree · geotoolz
description: "geotoolz's shelf of agent skills: 30 use skills over thirteen units, across six packages, with two listed skills for two bands of users. geotoolz is the first shelf at the step rung, where the role no longer tells entries apart and the operator family does, and the first with almost no dependent package, so its cards go to people and to capstones. Measured at the submodule pins by scripts/skill_shelf.py."
date: 2026-11-12
keywords:
  - agents
  - skills
  - geotoolz
  - geoops
  - role ladder
  - skill tree
numbering:
  heading_1: true
  heading_2: true
  table: true
---

(st-gz-decision)=
## Decision

Recommended: geotoolz's shelf is 30 use skills and four extend skills over thirteen units; step skills are tagged by operator family as well as by role; and two skills are listed, one per band: today's `build-geo-pipeline` for geoscientists and a new `find-read-and-store-geodata` for data engineers.
Alternative: one listed skill per package, six in all, which matches the install units and costs about three times as much in every session (estimate).
Status: open, 12 Nov 2026.

What does not change: the six packages, their public API, the four contributor skills, and the name of today's skill.
This page follows the pattern of [gaussx's shelf](./skill_tree_gaussx.md) and says only where geotoolz differs.

(st-gz-motivation)=
## Motivation

geotoolz is the widest surface in the stack, and one skill of 245 lines covers it.

:::{table} geotoolz's public names by unit, at the submodule pin, over its six packages (`geoarray`, `geoops`, `geopatcher`, `geocatalog`, `geocloud`, `geoproducts`).
:label: tbl-gz-units

<!-- shelf:units:begin -->
| Unit | Rungs | Home band | Subpackages | Names |
|---|---|---|---|---|
| Carriers and grids | ⓪ ① | geoscience | `geoarray.types` 18, `geoarray.coords` 12, `geoarray` 10, `geoarray.geometry` 7, `geoarray.errors` 5, `geoarray.adapter` 4, `geoarray.grid` 4, `geoarray.array` 3, `geoarray.carriers` 3, `geoarray.options` 3, `geoarray.attrs` 2, `geoarray.sampling` 2, `geoarray.source` 2 | 75 |
| Operator framework | ⑤ ⑥ | geoscience | `geoops.carrier` 45, `geoops` 35, `geoops.einx` 7, `geoops.patch_ops` 6, `geoops.testing` 1 | 94 |
| Geometry and I/O | ⑤ | geoscience | `geoops.geom` 67, `geoops.io` 17 | 84 |
| Radiometry and spectra | ⑤ | geoscience | `geoops.indices` 42, `geoops.normalize` 29, `geoops.radiometry` 24, `geoops.spectral` 16 | 111 |
| Masks and quality | ⑤ | geoscience | `geoops.mask` 35, `geoops.qa` 25 | 60 |
| Image processing | ⑤ | geoscience | `geoops.restore` 32, `geoops.segment` 18, `geoops.feature` 14, `geoops.measure` 13 | 77 |
| Detection and plumes | ⑤ | geoscience | `geoops.matched_filter` 33, `geoops.plume` 14 | 47 |
| Display and composites | ⑤ | geoscience | `geoops.viz` 21, `geoops.compositing` 18 | 39 |
| Learning | ⑤ | geoscience, ML | `geoops.augment` 19, `geoops.learn` 14 | 33 |
| Patching | ⑥, ② | geoscience, data | `geopatcher.run` 14, `geopatcher.fields` 13, `geopatcher.observe` 8, `geopatcher` 7, `geopatcher.spatial` 6, `geopatcher.config` 4, `geopatcher.patch` 3, `geopatcher.protocols` 3, `geopatcher.temporal` 2, `geopatcher.spatial_time` 1 | 61 |
| Catalogs | ⓪ ① | data | `geocatalog` 11, `geocatalog.states` 11, `geocatalog.matchup` 10, `geocatalog.sources` 10, `geocatalog.storage` 9, `geocatalog.utils` 9, `geocatalog.backends` 7, `geocatalog.grid` 7, `geocatalog.build` 5, `geocatalog.load` 5, `geocatalog.ops` 3, `geocatalog.patch` 2, `geocatalog.base` 1, `geocatalog.factory` 1, `geocatalog.geoslice` 1, `geocatalog.staging` 1 | 93 |
| Cloud storage | ① | data | `geocloud.hdf` 16, `geocloud.files` 13, `geocloud.cog` 12, `geocloud.store` 9, `geocloud.credentials` 8, `geocloud` 7, `geocloud.cache` 3, `geocloud.fs` 2 | 70 |
| Product readers | ⓪ | data, geoscience | `geoproducts.carbonmapper` 65, `geoproducts.goes` 17, `geoproducts` 4, `geoproducts.himawari` 3, `geoproducts.toy_sensor` 2, `geoproducts.base` 1, `geoproducts.stack` 1 | 93 |
| **All** | | | | **937** |
<!-- shelf:units:end -->
:::

:::{table} Which part of geotoolz each dependent package uses.
:label: tbl-gz-consumers

<!-- shelf:uses:begin -->
| Dependent package | Names used | Carriers and grids | Operator framework | Geometry and I/O | Radiometry and spectra | Masks and quality | Image processing | Detection and plumes | Display and composites | Learning | Patching | Catalogs | Cloud storage | Product readers |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| xrtoolz-ops | 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
<!-- shelf:uses:end -->
:::

Two things differ from a library lower in the stack.
More than half of the names are steps in `geoops`, in 18 operator families.
And one package in the stack takes any of them: geotoolz is used by people, in scripts, notebooks and recipes, so the split cannot come from its dependents.

(st-gz-shelf)=
## The shelf

At the step rung the role stops telling entries apart, so each step skill carries a second tag, its operator family.

:::{table} geotoolz's shelf. Each use skill is one file; the extend skills exist today in `.claude/skills/`.
:label: tbl-gz-shelf

| Role | Skill | Covers | Package |
| --- | --- | --- | --- |
| ⓪ structure | The carriers | `GeoArray`, `GeoSwath`, `GeoPoints`, their dims, attributes and fill | `geoarray` |
| ⓪ structure | Grids and windows | `Grid`, `PixelWindow`, `as_crs`, `as_transform`, coordinates | `geoarray` |
| ⓪ structure | Product readers | GOES, Himawari and CarbonMapper readers, the `RasterSource` protocol | `geoproducts` |
| ⓪ structure | Catalogs | `GeoCatalog`, `GeoSlice`, `open_catalog`, sources and backends | `geocatalog` |
| ⓪ structure | `add-product-reader`, `add-catalog-source` | extend | |
| ① primitive | Carrier verbs | `crop`, `pad`, `read_window`, `concatenate`, `sel`, `reduce`, `interp_at`, `to_pixel` | `geoarray` |
| ① primitive | Catalog queries | `query`, `intersect`, `union`, matchups | `geocatalog` |
| ① primitive | Object storage | file verbs, credentials, the store pool, COG and HDF reads and writes | `geocloud` |
| ② strategy | Patcher axes | the geometry, sampler, window and aggregation a patcher is built from | `geopatcher` |
| ② strategy | `add-field-adapter` | extend | |
| ⑤ step | Writing an operator on a carrier | the unwrap, call and rewrap helpers of `geoops.carrier`, `Einx`, the patch bridge | `geoops` |
| ⑤ step · geom | Geometry | reprojection, cropping, georeferencing, differential operators on a grid | `geoops` |
| ⑤ step · io | Reading and writing | windows, tiles, STAC and Earth Engine loads, COG, GeoTIFF and Zarr writes | `geoops` |
| ⑤ step · radiometry | Radiometry | DN, radiance and reflectance conversions, solar geometry, stretches | `geoops` |
| ⑤ step · spectral | Bands | band maths, ratios, selection, binning, continuum removal | `geoops` |
| ⑤ step · indices | Spectral indices | NDVI and the other indices, `AppendIndex` | `geoops` |
| ⑤ step · normalize | Normalisation | scalers, histogram stretches and matching, CLAHE | `geoops` |
| ⑤ step · qa | Quality flags | cloud, shadow, snow, water and saturation masks from product QA bands | `geoops` |
| ⑤ step · mask | Masks | building, combining, buffering and applying masks | `geoops` |
| ⑤ step · restore | Restoration | denoising, despeckling, destriping, gap filling | `geoops` |
| ⑤ step · feature | Features | blobs, edges, corners, texture | `geoops` |
| ⑤ step · segment | Segmentation | thresholds, superpixels, watershed, instance clean-up | `geoops` |
| ⑤ step · measure | Measurement | contours, connected components, region properties | `geoops` |
| ⑤ step · matched_filter | Matched filters | background estimators, the filter, its SNR and threshold | `geoops` |
| ⑤ step · plume | Plumes | masks, footprints, column to mass, IME and flux estimates | `geoops` |
| ⑤ step · viz | Display | true and false colour, colormaps, overlays, hillshade | `geoops` |
| ⑤ step · compositing | Composites | cloud-free, median, best-pixel and matched stacks | `geoops` |
| ⑤ step · learn | Learning on pixels | scikit-learn estimators as operators, `ModelOp` | `geoops` |
| ⑤ step · augment | Augmentation | noise, jitter, crops, flips, simulated clouds | `geoops` |
| ⑤ step | `add-operator` | extend | |
| ⑥ recipe | Composing a pipeline | `Sequential`, `Graph`, `Branch`, `Fanout`, as `geoops` re-exports them from pipekit | `geoops` |
| ⑥ recipe | Patch, process, stitch | spatial, temporal and spatio-temporal patchers, and running them | `geopatcher` |
| ⑥ recipe | Find, load and stage | building a catalog, loading a query, staging files | `geocatalog` |
:::

That is four structures, three primitives, one strategy, nineteen steps and three recipes.
The family tag is the namespace, and the [operator coupling](./operator_coupling.md) page adds what the family name cannot: whether an operator mixes bands, pixels or time, which decides how it may be patched.
A step entry states its family's coupling class in its first line.

(st-gz-units)=
## Units and cards

:::{table} The cards each dependent package would receive from geotoolz.
:label: tbl-gz-cards

<!-- shelf:cards:begin -->
| Dependent package | Cards | Also uses |
|---|---|---|
| xrtoolz-ops | none | Geometry and I/O (2) |
<!-- shelf:cards:end -->
:::

No package reaches the threshold, so no library's block carries a geotoolz card.
The thirteen cards are written all the same, for two other readers.
The band front doors for geoscientists and for data engineers are built from them, and so are the capstones: a capstone for a methane plume lists the detection and plumes card, the masks card and the patching card as its floor.

From below, geotoolz's packages receive pipekit's cards: geotoolz-ops and geotoolz-array are pipekit's two largest dependents.
They also receive a card for georeader that geostack owns, because georeader is third party.

(st-gz-listed)=
## Listed skills

:::{table} The listed use skills. Line budgets are proposals.
:label: tbl-gz-skills

| Skill | Band | Units | Asked for as |
| --- | --- | --- | --- |
| `build-geo-pipeline`, the front door, at most 150 lines | geoscience | carriers and grids; the operator framework; the seven operator units; patching | "compute NDVI on this scene", "mask clouds", "run a matched filter tile by tile" |
| `find-read-and-store-geodata`, at most 150 lines | data | catalogs; cloud storage; product readers | "find scenes over this box", "read this GOES file", "write a COG to a bucket" |
:::

The front door holds a table of the 18 families, each one line with its coupling class, and loads a family's entry on demand.

(st-gz-differs)=
## Where geotoolz bends the pattern

1. **No dependents, so units follow namespaces and bands.** gaussx's units came from which library uses what. Here they come from the operator families and from who asks: a geoscientist or a data engineer.
2. **The role needs a second tag.** Nineteen of thirty entries are ⑤. The family and its coupling class do the work the rung does lower down.
3. **One repository, two bands.** The same workspace serves people composing steps and people moving files, so it has two front doors. A package is not the unit of listing; the band is.
4. **The shelf is long by design.** One entry per family keeps each under the budget, and adding an operator family adds an entry without touching the others.

(st-gz-gates)=
## Gates

Gates GX-0 to GX-6 of [gaussx's shelf](./skill_tree_gaussx.md#sg-gates) apply with geotoolz's names, and one is added.

| Gate | Passes when |
| --- | --- |
| GZ-1 Families covered | every public submodule of `geoops` belongs to one shelf entry, and each ⑤ entry names its coupling class |

(st-gz-risks)=
## Risks

- **The name count is inflated by submodules.** A family's submodule exports functions as well as Operators, so `geoops.indices` counts 42 names for 22 Operators. Fallback: the counts size the units and are not a budget.
- **The front door grows past its budget.** Eighteen family lines, the carrier rules and the composition rules compete for 150 lines. Fallback: the carrier rules move to the carriers entry.
- **Families move.** The pin is `9e92120`; the operator rename to geotoolz-ops is recent. Fallback: units follow namespaces, so a move changes one row and the test fails until the page is regenerated.

## Relationships

- Parent: jejjohnson/geostack#150
- Related: jejjohnson/geostack#151 (the design), jejjohnson/geostack#156 (capstones, which take geotoolz's cards)
- Related: [skill tree](./skill_tree.md), [gaussx's shelf](./skill_tree_gaussx.md), [operator coupling](./operator_coupling.md)
