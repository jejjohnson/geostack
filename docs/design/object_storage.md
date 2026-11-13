---
title: "Object storage: buckets, keys and five verbs"
subtitle: "Background · explainer · what S3, Azure Blob and GCS can and cannot do, and why formats and catalogs are shaped by it"
short_title: Object storage
description: "A visual explainer of object storage: keys instead of folders, the five operations, range requests, why listing is slow, and how to write several objects safely; background for the storage formats page and the storage design pages."
date: 2026-11-13
keywords:
  - object storage
  - S3
  - Azure Blob Storage
  - range requests
  - obstore
---

+++ {"part": "abstract"}

Background for the storage pages. The running example's B12 band is one object in a bucket; the page follows a window read as HTTP requests, counts what a listing costs, and shows why several writes are not one write. Diagrams carry the explanation.

+++

{button}`Open the full page <./pages/object-storage.html>`

**Status.** Background. Provider facts are from the S3, Azure and GCS documentation as known in November 2026, not re-checked online for this page; the obstore calls are marked `not pinned`.

## TL;DR: a dictionary over HTTP

- **A bucket maps keys to objects.** There are no folders; a `/` in a key is a character.
- **Five verbs:** PUT, GET (whole or a byte range), HEAD, LIST, DELETE. No append, no edit in place, no rename.
- **Range requests** are what make COG, Parquet and Zarr sharding work.
- **Listing is slow** (1 000 keys per page, in sequence), so layouts keep their own index.
- **One PUT is atomic; several are not.** Write new objects, then swap one pointer with a conditional PUT.

## On the page

1. The running example: the B12 band as one object
2. Keys, not folders
3. Five verbs, and the four that are missing
4. Range requests: a window read as HTTP
5. Listing, and when not to
6. Writing safely: one PUT, or a pointer swap
7. S3, Azure and GCS side by side
8. In the stack: geocloud's client pool, geocatalog, pipekit's S3 registry
