---
title: "Storage formats: COG, GeoParquet, Zarr, and pointers to them"
subtitle: "Background · explainer · how each cloud-native format turns a window or a query into byte ranges, and how kerchunk, VirtualiZarr and Icechunk serve existing files as versioned Zarr"
short_title: Storage formats
description: "A visual explainer of COG, GeoParquet and Zarr, how they map onto each other, and how kerchunk, VirtualiZarr and Icechunk read rasters in chunks without copying them; background for the Level 2 collections, product catalog and recipe registry designs."
date: 2026-11-12
keywords:
  - COG
  - GeoParquet
  - Zarr
  - Icechunk
  - VirtualiZarr
  - kerchunk
---

+++ {"part": "abstract"}

Background for the three storage design pages that follow it. One running example, band B12 of a Sentinel-2 L2A tile read for a 2 km site, goes through every format: the same 100 × 100 window read from a COG, from Zarr, and through an Icechunk repository that only references the COG. Diagrams carry the explanation; the maths sits in folded panels.

+++

{button}`Open the full page <./pages/storage-formats.html>`

**Status.** Background, kept in step with the design pages; the reader table reflects the projects' docs as of November 2026.

## TL;DR: four indexes over the same bytes

- **COG.** A GeoTIFF whose header lists every tile's offset and length, so a window read is a few range requests: 4 tiles, about 2 MB, out of a 60 MB band.
- **GeoParquet.** A table stored by column and by row group, with per-group bounding boxes and times, so a query skips most of the file; geocatalog's catalogs are GeoParquet.
- **Zarr.** An N-dimensional array cut into chunks, one object per key; the key is computed from the position, so nothing has to be looked up.
- **COG and Zarr map one-to-one for pixels and bytes**, not for georeferencing, overviews or extra dimensions, which need a convention.
- **Pointers instead of copies.** kerchunk writes chunk references to a file, VirtualiZarr builds them as xarray objects, and Icechunk stores them with real chunks in versioned commits. Reading a raster through Icechunk is a Zarr key lookup, a manifest lookup, then the same range requests a COG read makes.

## On the page

1. The running example: one Sentinel-2 SWIR band and a 2 km site
2. Part 1: COG, GeoParquet and Zarr, each with a diagram, a code sketch and when to use it
3. COG and Zarr: what maps to what, whether it is one-to-one, can Zarr read a COG, and converting with the stack
4. Part 2: chunk references, kerchunk, VirtualiZarr, Icechunk, and how a raster is read in chunks through Icechunk
5. Compatibility: a decision guide, and which readers open which format
6. The bytes fetched, path by path, and where each idea goes in the design pages
