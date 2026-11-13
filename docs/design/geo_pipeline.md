---
title: "The geo pipeline: ten stages from search to publish"
subtitle: "Background · explainer · what every geoscience pipeline does in order, what each stage hands to the next, and where versioned storage and a product catalog fit"
short_title: The geo pipeline
description: "A visual explainer of the geoscience pipeline as ten stages in three acts (search, catalog, save; stream, split, apply, combine; write, commit, publish), extending the eight-stage geotoolz diagram with commit and publish, run end to end on a small archive."
date: 2026-11-13
keywords:
  - pipeline
  - split-apply-combine
  - geotoolz
  - Icechunk
  - product catalog
---

+++ {"part": "abstract"}

A broader view than [the role ladder](./role_ladder.md): the ladder says what is inside a library, this page says what a job does from start to finish. It starts from the eight-stage diagram in geotoolz, adds commit and publish, and runs all ten stages on one example, NDVI over a site for June.

+++

{button}`Open the full page <./pages/geo-pipeline.html>`

**Status.** Background. Stages 1 to 8 run geotoolz at `640670c` as it ships; commit and publish use Icechunk and a Parquet row directly and stand in for APIs that are proposed (geotoolz#665, geotoolz#671). Run in November 2026 on a synthetic local archive.

## TL;DR: find, compute, keep, and around again

- **Find:** search an archive, catalog what it holds, save only the hits; no pixel is decoded.
- **Compute:** stream the hits as one lazy field, then split → apply → combine.
- **Keep:** write the result, commit it as a version, publish a record of it; the last two are new.
- **Typed seams:** neighbouring stages hand over an object (a catalog, a field, a patch, a snapshot id), never a bare path.
- **It is a loop:** a published product is an archive the next pipeline searches.
- **The test:** 6 of 24 scenes match the request; 20 patches give the same NDVI as the whole site to 2 × 10⁻¹⁶; the product reopened through its record's snapshot id is identical.

## On the page

1. The running example
2. Ten stages in three acts, and the shape of the counts
3. Act 1, find: search, catalog, save
4. Act 2, compute: stream, split, apply, combine, with the overlap-add maths folded away
5. Act 3, keep: write, commit, publish, and the record
6. The loop
7. What crosses each seam
8. The pipeline and the role ladder
9. What Icechunk and the product catalog change, stage by stage
