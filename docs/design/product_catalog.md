---
title: "Product catalog: recording what was made, from what, with which code"
subtitle: "Design · working page · a catalog protocol rather than a database, one record in three encodings, and pgstac as a synced index"
short_title: Product catalog
description: "Design doc for GeoStack's product catalog: publish tagged Icechunk snapshots through a ProductCatalog protocol to PostGIS, pgstac, GeoParquet or another store; define the record once for SQL, Arrow and STAC; keep pgstac as an index synced from your own table."
date: 2026-11-12
keywords:
  - product catalog
  - STAC
  - pgstac
  - PostGIS
  - GeoParquet
---

+++ {"part": "abstract"}

The persist layer of [Level 2 collections](./l2_collections.md) has a repository for data and a catalog for what was made from it. This page designs the catalog: who owns what, a running example (a month of matched-filter products over one site, published, found again and traced back), a current-state audit, three workstreams and their gates.

+++

{button}`Open the full page <./pages/product-catalog.html>`

**Status.** A draft. Workstreams F, K and L are tracked as GitHub issues.

## TL;DR: a protocol, not a database

- **The repo stays the truth.** The catalog is derived from tagged snapshots and can be rebuilt from them; publishing happens after the commit and is idempotent.
- **F.** `publish` writes one STAC-shaped record per product through a `ProductCatalog` adapter: PostGIS, pgstac, a STAC API with transactions, or GeoParquet.
- **K.** The record's core fields are defined once and the GeoParquet schema, the SQL table and the STAC mapping are generated from them; an ORM maps only the core, so apps add their own columns.
- **L.** pgstac stores items dehydrated in partitioned tables, so it is written through its functions, never an ORM: your own PostGIS table is the catalog of record and pgstac an index synced from it.

## On the page

1. Summary and ownership rules
2. The running example in three stages: publish, find, trace back
3. Current-state audit
4. Workstreams F (publish), K (one record, three encodings), L (pgstac as a synced index)
5. Testing gates, sequencing and risks
