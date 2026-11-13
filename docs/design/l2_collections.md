---
title: "Level 2 collections: a catalog that lives with its data"
subtitle: "Design · working page · keeping geocatalog's table inside an Icechunk repository, so scenes, products and their index are versioned and committed together"
short_title: Level 2 collections
description: "A verified current state of how GeoStack finds, loads and writes scenes, and the smallest change that puts the catalog inside an Icechunk repository: an Icechunk catalog backend, a write half for the catalog protocol, a scene-per-node loader, one Zarr georeferencing convention and virtual ingest."
date: 2026-11-12
keywords:
  - Icechunk
  - VirtualiZarr
  - geocatalog
  - Level 2
  - DataTree
---

+++ {"part": "abstract"}

Prompted by the CNG Forum 2026 talk "Level 2 Data Collections in Zarr" (Tom Nicholas, Earthmover; Sean Harkins, Development Seed). The page reads geostack, geotoolz, xrtoolz and pipekit at their 12 November 2026 commits, draws the current state in two layers (work and persist), and shows a target where every current box stays and five workstreams add to it. Read [Storage formats](./storage_formats.md) first for COG, Zarr and Icechunk.

+++

{button}`Open the full page <./pages/l2-collections.html>`

**Status.** A draft. Workstreams A to E are tracked as GitHub issues; the product catalog (F) and the recipe registries (G to J) are their own pages.

## TL;DR: change where the catalog lives, not the carriers

- **Verified today.** geocatalog already holds a table catalog (`GeoCatalog` with in-memory and DuckDB backends, saved as GeoParquet); geoarray already converts to and from `DataTree`; xrops already dispatches over `DataTree`. Nothing in the four repositories mentions Icechunk, VirtualiZarr or kerchunk.
- **Five gaps.** The table and the data are separate artifacts; rows point at whole files; no raster loader keeps scenes apart; two Zarr georeferencing conventions; the catalog protocol has no write half.
- **A.** An `IcechunkGeoCatalog` whose table is a `meta/` group at a pinned snapshot.
- **B.** A write half: a transaction writes a product group and its row, and commits both or neither.
- **C.** `load_scenes`: one `DataTree` node per scene, each on its own grid, through `geoarray.to_datatree`.
- **D.** One Zarr georeferencing convention: `WriteZarr` writes the CF `spatial_ref` that `geoarray.to_xarray` writes.
- **E.** Virtual ingest of archives with VirtualiZarr, no copy.

## On the page

1. Current state, verified in code: a two-layer diagram, the components and where each lives, five gaps
2. Revised target: the same boxes, with what each gains
3. API sketch: repository layout, `meta/` to `CatalogRow`, the backend, the write half, the scene loader
4. In action: the matched filter over a month of scenes, today and in the target
5. Testing gates, sequencing and risks
