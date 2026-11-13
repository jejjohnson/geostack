---
title: "Databases and catalogs: tables, indexes, transactions, STAC"
subtitle: "Background · explainer · how a catalog answers which scenes cover a site in a month, and what sits under it"
short_title: Databases and catalogs
description: "A visual explainer of tables, B-tree and spatial indexes, row and column stores, transactions and upserts, where a database lives, SQL toolkits, STAC and pgstac, with measured query times; background for the product catalog design."
date: 2026-11-13
keywords:
  - databases
  - indexes
  - PostGIS
  - Parquet
  - DuckDB
  - STAC
  - pgstac
---

+++ {"part": "abstract"}

Background for the product catalog page. One question, which of 100 000 scenes cover a site in June 2026, is answered by a scan, a B-tree, a spatial index and a column store, each drawn and timed. Diagrams carry the explanation.

+++

{button}`Open the full page <./pages/databases-catalogs.html>`

**Status.** Background. Timings measured in November 2026 on 100 000 synthetic rows with SQLite 3.53 and DuckDB 2.0; PostgreSQL and PostGIS were not measured.

## TL;DR: a table, an index, and a promise about writes

- **A catalog is a table:** one row per scene, a unique id, a column pointing at the data.
- **An index** lets a query skip rows: the June query drops from 9.1 ms to 0.28 ms with a B-tree on time.
- **Row stores** suit single-row reads and writes; **column stores** suit scans (29 ms against 4.5 ms for a monthly mean, 24 MB against 5.2 MB on disk).
- **A transaction** makes several changes succeed or fail together; an **upsert** makes a write safe to repeat.
- **STAC** is a vocabulary for catalog rows; **pgstac** stores it in PostgreSQL and is written through its functions.

## On the page

1. The running example: 100 000 scenes over Iberia
2. A table, its schema and primary key
3. Indexes: B-tree and spatial, with measured times
4. Row stores and column stores
5. Transactions and upserts
6. Server, in-process or file
7. SQL, drivers and ORMs
8. STAC and pgstac
9. In the stack: geocatalog and the product catalog design
